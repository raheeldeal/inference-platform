"""
autosim: replay a bursty traffic trace against autoscaling policies and chart cost vs SLO.

The model (deliberately simple, every knob maps to a real platform setting):

  replicas     each has `capacity` concurrent slots (= vLLM max_num_seqs you're
               willing to run at, i.e. the point on the frontier you chose on Day 1).
               Service time grows mildly with how full the replica is (memory-bound
               decode), exactly like the mock backend.
  queue        requests that find no free slot wait FIFO. Latency = wait + service.
  autoscaler   every `decision_interval_s`, desired = ceil(avg_concurrency_over_window
               / target_concurrency), clamped to [min_replicas, max_replicas].
               Scale UP happens immediately but new replicas take `cold_start_s`
               to become ready (weights loading -> what BDN attacks).
               Scale DOWN waits `scale_down_delay_s` of continuously-lower demand,
               then drains (finishes in-flight, takes no new work) before dying.
               min_replicas=0 is scale-to-zero: the first request after idle eats
               a full cold start.
  cost         replica-seconds (booting + ready + draining) * $/hr.
  SLO          fraction of requests with latency <= slo_s. Also p50/p99.

What you get: results/autoscaler.png with the cost-vs-SLO scatter (each dot a
policy; the Pareto frontier is the set you'd actually offer customers) and a
time-series panel for two contrasting policies so you can SEE why one burns money
and the other burns p99.

  uv run autosim --trace bursty --duration 3600 --cold-start 30,180
"""

from __future__ import annotations

import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import typer

# ----------------------------------------------------------------------------
# Traffic traces: requests/second at 1s resolution
# ----------------------------------------------------------------------------
def make_trace(kind: str, duration: int, base_rps: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(duration)
    if kind == "steady":
        rps = np.full(duration, base_rps)
    elif kind == "diurnal":
        rps = base_rps * (1 + 0.8 * np.sin(2 * math.pi * t / duration - math.pi / 2))
    elif kind == "bursty":
        # diurnal baseline + random 30-120s bursts at 4-10x + a couple of flash crowds
        rps = base_rps * (0.6 + 0.4 * np.sin(2 * math.pi * t / duration))
        n_bursts = max(3, duration // 400)
        for _ in range(n_bursts):
            s = rng.integers(0, duration); d = rng.integers(30, 120); mult = rng.uniform(4, 10)
            rps[s:s + d] *= mult
        for _ in range(2):
            s = rng.integers(duration // 4, duration); d = rng.integers(120, 300)
            rps[s:s + d] = np.maximum(rps[s:s + d], base_rps * 15)
    elif kind == "idle-spiky":
        # mostly idle with short spikes: the scale-to-zero stress test
        rps = np.full(duration, base_rps * 0.05)
        for _ in range(max(4, duration // 300)):
            s = rng.integers(0, duration); d = rng.integers(20, 90)
            rps[s:s + d] = base_rps * rng.uniform(2, 6)
    else:
        raise ValueError(kind)
    return np.maximum(rps, 0)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
@dataclass
class Policy:
    target_concurrency: int = 16
    min_replicas: int = 0
    max_replicas: int = 20
    scale_down_delay_s: float = 300
    window_s: float = 30
    decision_interval_s: float = 5
    cold_start_s: float = 60
    # concurrency: desired = avg(inflight + queued) / target     (what most platforms do)
    # rate:        desired = arrival_rate * est_service_time / target   (Little's law;
    #              ignores the backlog, so a cold-start-induced queue can't make it overshoot)
    signal: str = "concurrency"

    def label(self) -> str:
        return (f"{self.signal[:4]} tc={self.target_concurrency} min={self.min_replicas} "
                f"sdd={int(self.scale_down_delay_s)}s cs={int(self.cold_start_s)}s")


@dataclass
class Replica:
    born: float
    ready_at: float
    inflight: int = 0
    draining: bool = False
    died: float | None = None

    def ready(self, now: float) -> bool:
        return now >= self.ready_at and not self.draining and self.died is None


@dataclass
class Result:
    policy: Policy
    cost_usd: float
    slo_attainment: float
    p50: float
    p99: float
    p999: float
    requests: int
    cold_starts: int
    max_queue: int
    replica_seconds: float
    series: dict[str, list] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        d = dict(label=self.policy.label(), **self.policy.__dict__)
        d.update(cost_usd=round(self.cost_usd, 2), slo_attainment=round(self.slo_attainment, 4),
                 p50=round(self.p50, 2), p99=round(self.p99, 2), p999=round(self.p999, 2), requests=self.requests,
                 cold_starts=self.cold_starts, max_queue=self.max_queue, replica_hours=round(self.replica_seconds / 3600, 2))
        return d


def simulate(trace: np.ndarray, pol: Policy, capacity: int, service_s: float, slo_s: float,
             price_per_hour: float, seed: int = 0, dt: float = 0.25, keep_series: bool = False) -> Result:
    rng = random.Random(seed)
    duration = len(trace)
    now = 0.0
    replicas: list[Replica] = []
    queue: list[tuple[float, float]] = []            # (arrival_time, service_time)
    completions: list[tuple[float, int]] = []        # heap of (finish_time, replica_idx)
    latencies: list[float] = []
    cold_starts = 0
    max_queue = 0
    conc_samples: list[tuple[float, int]] = []       # (time, inflight+queued) for the window avg
    below_since: float | None = None
    next_decision = 0.0
    arrival_times: list[float] = []                  # for the rate signal
    est_service = service_s                          # EMA of observed service time
    series = dict(t=[], rps=[], ready=[], booting=[], queue=[], inflight=[]) if keep_series else None

    def spawn(n: int):
        nonlocal cold_starts
        for _ in range(n):
            replicas.append(Replica(born=now, ready_at=now + pol.cold_start_s))
            cold_starts += 1

    spawn(pol.min_replicas)
    # pre-warm min replicas so the sim doesn't start with a cold start artifact
    for r in replicas:
        r.ready_at = 0.0

    arrivals_carry = 0.0
    while now < duration:
        # ---- arrivals (Poisson per tick) ----
        lam = trace[min(int(now), duration - 1)] * dt
        k = np.random.default_rng(int(now * 1000) + seed).poisson(lam) if lam > 0 else 0
        for _ in range(k):
            st = service_s * rng.lognormvariate(0, 0.25)
            queue.append((now, st))
            arrival_times.append(now)

        # ---- completions ----
        while completions and completions[0][0] <= now:
            _, idx = heapq.heappop(completions)
            replicas[idx].inflight -= 1
            r = replicas[idx]
            if r.draining and r.inflight == 0:
                r.died = now

        # ---- admission: fill free slots, FIFO ----
        if queue:
            ready = [(i, r) for i, r in enumerate(replicas) if r.ready(now)]
            for i, r in sorted(ready, key=lambda x: x[1].inflight):
                while queue and r.inflight < capacity:
                    arr, st = queue.pop(0)
                    load = r.inflight / capacity
                    st_eff = st * (1 + 0.5 * load)       # decode slows as the batch fills
                    r.inflight += 1
                    est_service = 0.98 * est_service + 0.02 * st_eff
                    heapq.heappush(completions, (now + st_eff, i))
                    latencies.append(now - arr + st_eff)
                if not queue:
                    break
        max_queue = max(max_queue, len(queue))

        # ---- autoscaler ----
        inflight = sum(r.inflight for r in replicas if r.died is None)
        conc_samples.append((now, inflight + len(queue)))
        while conc_samples and conc_samples[0][0] < now - pol.window_s:
            conc_samples.pop(0)
        while arrival_times and arrival_times[0] < now - pol.window_s:
            arrival_times.pop(0)
        if now >= next_decision:
            next_decision = now + pol.decision_interval_s
            if pol.signal == "rate":
                rate = len(arrival_times) / pol.window_s
                desired = math.ceil(rate * est_service / pol.target_concurrency)
            else:
                avg_conc = sum(c for _, c in conc_samples) / max(len(conc_samples), 1)
                desired = math.ceil(avg_conc / pol.target_concurrency)
            desired = max(pol.min_replicas, min(pol.max_replicas, desired))
            # scale-to-zero wake-up: anything queued with nothing alive -> at least 1
            if queue and not any(r.died is None and not r.draining for r in replicas):
                desired = max(desired, 1)
            alive = [r for r in replicas if r.died is None and not r.draining]
            if desired > len(alive):
                spawn(desired - len(alive))
                below_since = None
            elif desired < len(alive):
                below_since = below_since if below_since is not None else now
                if now - below_since >= pol.scale_down_delay_s:
                    # drain the newest first (they're the least likely to be warm-cached)
                    for r in sorted(alive, key=lambda r: -r.born)[: len(alive) - desired]:
                        r.draining = True
                        if r.inflight == 0:
                            r.died = now
                    below_since = None
            else:
                below_since = None

        if series is not None and int(now * 4) % 4 == 0:
            series["t"].append(now)
            series["rps"].append(float(trace[min(int(now), duration - 1)]))
            series["ready"].append(sum(1 for r in replicas if r.ready(now)))
            series["booting"].append(sum(1 for r in replicas if r.died is None and now < r.ready_at))
            series["queue"].append(len(queue))
            series["inflight"].append(inflight)
        now += dt

    # drain whatever is left so latencies are complete (they're already recorded on admit)
    replica_seconds = sum((r.died if r.died is not None else duration) - r.born for r in replicas)
    lat = np.array(latencies) if latencies else np.array([0.0])
    # requests still queued at the end never got served: count them as SLO misses
    unserved = len(queue)
    return Result(policy=pol, cost_usd=replica_seconds / 3600 * price_per_hour,
                  slo_attainment=float((lat <= slo_s).sum()) / (len(lat) + unserved),
                  p50=float(np.percentile(lat, 50)), p99=float(np.percentile(lat, 99)), p999=float(np.percentile(lat, 99.9)),
                  requests=len(lat) + unserved, cold_starts=cold_starts, max_queue=max_queue,
                  replica_seconds=replica_seconds, series=series or {})


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot(results: list[Result], trace: np.ndarray, slo_s: float, out: Path, service_s: float):
    from inference_platform.viz import SERIES, STATUS, TEXT2, plt, style
    style()
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 0.7, 0.9])
    ax = fig.add_subplot(gs[0, :])

    cold_starts = sorted({r.policy.cold_start_s for r in results})
    color_for = {cs: SERIES[i % len(SERIES)] for i, cs in enumerate(cold_starts)}
    for r in results:
        c = color_for[r.policy.cold_start_s]
        marker = "o" if r.policy.min_replicas == 0 else "s"
        ax.scatter(r.cost_usd, 100 * r.slo_attainment, color=c, marker=marker, s=60, alpha=0.85,
                   edgecolors="white", linewidths=1)
    # Pareto frontier: cheapest for each attainment level
    pts = sorted(((r.cost_usd, r.slo_attainment, r) for r in results), key=lambda x: x[0])
    frontier, best = [], -1
    for cost, att, r in pts:
        if att > best:
            frontier.append((cost, att, r)); best = att
    ax.plot([p[0] for p in frontier], [100 * p[1] for p in frontier], "-", color=TEXT2, alpha=0.5, lw=1.5, zorder=0)
    for cost, att, r in frontier:
        ax.annotate(f"{r.policy.signal[:4]} tc={r.policy.target_concurrency} min={r.policy.min_replicas} sdd={int(r.policy.scale_down_delay_s)}",
                    (cost, 100 * att), fontsize=7, color=TEXT2, xytext=(5, -9), textcoords="offset points")
    for cs in cold_starts:
        ax.scatter([], [], color=color_for[cs], label=f"cold start {int(cs)}s")
    ax.scatter([], [], color=TEXT2, marker="o", label="min replicas = 0 (scale-to-zero)")
    ax.scatter([], [], color=TEXT2, marker="s", label="min replicas >= 1")
    ax.axhline(99, color=STATUS["warning"], lw=1, ls="--"); ax.text(ax.get_xlim()[0], 99.1, "99% target", fontsize=8, color=TEXT2)
    ax.set(title=f"Cost vs SLO attainment (latency <= {slo_s:.0f}s) over {len(trace)/3600:.1f}h of '{trace_kind_global}' traffic",
           xlabel="cost (USD)", ylabel="% requests within SLO")
    ax.legend(fontsize=8, loc="lower right")

    # time series: the frontier's best-attainment point vs its cheapest point.
    # Two stacked panels per policy (offered load; replicas + queue) — no dual axes.
    picks = [frontier[-1][2], frontier[0][2]] if len(frontier) > 1 else [results[0]]
    for k, r in enumerate(picks):
        s = r.series
        ax = fig.add_subplot(gs[1, k])
        ax.fill_between(s["t"], s["rps"], color=SERIES[0], alpha=0.2)
        ax.plot(s["t"], s["rps"], color=SERIES[0], lw=1.2)
        ax.set(title=f"{r.policy.label()}  ->  ${r.cost_usd:.2f}, SLO {100*r.slo_attainment:.1f}%, p99 {r.p99:.1f}s",
               ylabel="offered req/s")
        ax.tick_params(labelbottom=False)
        ax = fig.add_subplot(gs[2, k])
        ax.step(s["t"], s["ready"], color=SERIES[2], lw=2, label="ready replicas", where="post")
        ax.step(s["t"], s["booting"], color=SERIES[3], lw=1.5, label="booting (cold start)", where="post")
        ax.plot(s["t"], [q / 10 for q in s["queue"]], color=SERIES[7], lw=1, alpha=0.9, label="queued requests / 10")
        ax.set(xlabel="time (s)", ylabel="replicas  |  queue/10")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


trace_kind_global = ""
cli = typer.Typer(add_completion=False, help=__doc__)


@cli.command()
def main_cmd(
    trace: str = typer.Option("bursty", help="steady | diurnal | bursty | idle-spiky"),
    duration: int = typer.Option(3600, help="seconds of traffic"),
    base_rps: float = typer.Option(4.0),
    capacity: int = typer.Option(32, help="concurrent requests one replica can run (your chosen frontier point)"),
    service_s: float = typer.Option(2.0, help="mean request service time at low load (e.g. 128 tok @ 64 tok/s)"),
    slo_s: float = typer.Option(4.0, help="latency SLO in seconds"),
    price_per_hour: float = typer.Option(3.95, help="$/hr per replica (Modal H100 = 3.95, Baseten H100 = 6.50)"),
    target_concurrency: str = typer.Option("8,16,32"),
    signal: str = typer.Option("concurrency,rate", help="scaling signal(s) to compare"),
    scale_down_delay: str = typer.Option("60,300,900"),
    min_replicas: str = typer.Option("0,1"),
    cold_start: str = typer.Option("30,180", help="compare platforms: 30s (BDN-style) vs 180s (pull weights from HF)"),
    max_replicas: int = typer.Option(20),
    seed: int = typer.Option(1),
    out: Path = typer.Option(Path("results/autoscaler.json")),
    plot_out: Path = typer.Option(Path("results/autoscaler.png")),
):
    global trace_kind_global
    trace_kind_global = trace
    tr = make_trace(trace, duration, base_rps, seed)
    grid = [Policy(target_concurrency=tc, min_replicas=mr, scale_down_delay_s=sdd, cold_start_s=cs, max_replicas=max_replicas, signal=sig)
            for sig in signal.split(",")
            for tc in map(int, target_concurrency.split(","))
            for mr in map(int, min_replicas.split(","))
            for sdd in map(float, scale_down_delay.split(","))
            for cs in map(float, cold_start.split(","))]
    print(f"trace={trace} duration={duration}s mean_rps={tr.mean():.2f} peak_rps={tr.max():.1f} policies={len(grid)}")
    results = []
    for pol in grid:
        r = simulate(tr, pol, capacity, service_s, slo_s, price_per_hour, seed=seed, keep_series=True)
        results.append(r)
        print(f"  {pol.label():40s} cost=${r.cost_usd:7.2f} slo={100*r.slo_attainment:6.2f}% p50={r.p50:5.2f}s p99={r.p99:6.2f}s cold_starts={r.cold_starts:3d} max_q={r.max_queue}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(trace=trace, duration=duration, base_rps=base_rps, capacity=capacity, service_s=service_s,
                                   slo_s=slo_s, price_per_hour=price_per_hour, results=[r.row() for r in results]), indent=2))
    print(f"wrote {out}")
    plot(results, tr, slo_s, plot_out, service_s)


def main():
    cli()


if __name__ == "__main__":
    main()
