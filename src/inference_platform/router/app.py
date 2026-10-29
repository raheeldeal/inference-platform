"""
router: a small inference gateway in front of N OpenAI-compatible backends.

Implements the things a serving platform's traffic layer must get right:

  health checks   active (/health polling) + passive (a connect error marks the
                  backend down immediately, so failover is fast, and active checks
                  bring it back).
  retries         a request is retried on another backend if the upstream fails
                  BEFORE we've sent any bytes to the client. Once streaming has
                  started we can't retry (the client already saw tokens) — this is
                  why TTFT-timeout and connect-timeout matter more than total timeout.
  failover        the retry picks a different backend; if none is eligible -> 503.
  weighted split  canary at 5/95 via weights; promote by changing weights live.
  shadow traffic  backends marked shadow get a fire-and-forget copy; we record
                  their latency/status so you can compare a new build on real
                  traffic without exposing users to it.
  drain           /admin/backends/{name}/drain stops NEW traffic, waits for in-flight
                  to hit zero, then marks it removed. That's a rolling deploy that
                  never drops a request. Compare with /admin/kill on the backend.
  policies        weighted | least_inflight | prefix_affinity (see config.yaml).

Run:   uv run router --config src/inference_platform/router/config.yaml
Then:  curl localhost:8000/metrics | jq
       curl -X POST localhost:8000/admin/backends/a10g/drain
       curl -X POST 'localhost:8000/admin/backends/a10g/weight?weight=50'
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import typer
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
@dataclass
class Backend:
    name: str
    url: str
    weight: float = 1.0
    stage: str = "stable"
    shadow: bool = False
    healthy: bool = False            # unknown until first check passes
    draining: bool = False
    removed: bool = False
    inflight: int = 0
    consec_fail: int = 0
    consec_ok: int = 0
    last_check: float = 0.0
    last_error: str | None = None
    # counters
    requests: int = 0
    errors: int = 0
    retries_absorbed: int = 0        # requests that landed here after failing elsewhere
    lat_ms: deque = field(default_factory=lambda: deque(maxlen=2000))
    ttft_ms: deque = field(default_factory=lambda: deque(maxlen=2000))

    @property
    def eligible(self) -> bool:
        return self.healthy and not self.draining and not self.removed and not self.shadow

    def pct(self, series: deque, q: float) -> float | None:
        if not series:
            return None
        s = sorted(series)
        return s[min(len(s) - 1, int(q / 100 * len(s)))]

    def snapshot(self) -> dict[str, Any]:
        return dict(name=self.name, url=self.url, weight=self.weight, stage=self.stage, shadow=self.shadow,
                    healthy=self.healthy, draining=self.draining, removed=self.removed,
                    inflight=self.inflight, requests=self.requests, errors=self.errors,
                    retries_absorbed=self.retries_absorbed, last_error=self.last_error,
                    ttft_p50=self.pct(self.ttft_ms, 50), ttft_p99=self.pct(self.ttft_ms, 99),
                    lat_p50=self.pct(self.lat_ms, 50), lat_p99=self.pct(self.lat_ms, 99))


class Router:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.policy = cfg.get("policy", "weighted")
        self.backends: dict[str, Backend] = {}
        for b in cfg["backends"]:
            self.backends[b["name"]] = Backend(name=b["name"], url=b["url"].rstrip("/"), weight=float(b.get("weight", 1)),
                                               stage=b.get("stage", "stable"), shadow=bool(b.get("shadow", False)))
        h = cfg.get("health", {})
        self.h_interval = float(h.get("interval_s", 1.0))
        self.h_timeout = float(h.get("timeout_s", 1.0))
        self.unhealthy_after = int(h.get("unhealthy_after", 2))
        self.healthy_after = int(h.get("healthy_after", 2))
        r = cfg.get("retry", {})
        self.max_attempts = int(r.get("max_attempts", 3))
        self.retry_on = set(r.get("retry_on_status", [500, 502, 503, 504]))
        self.timeout = httpx.Timeout(connect=float(r.get("connect_timeout_s", 2)),
                                     read=float(r.get("stream_timeout_s", 120)),
                                     write=10.0, pool=5.0)
        self.ttft_timeout = float(r.get("ttft_timeout_s", 30))
        self.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=2000, max_keepalive_connections=500))
        self.totals = dict(requests=0, ok=0, errors=0, retries=0, failovers=0, no_backend=0,
                           shadow_sent=0, shadow_errors=0, mid_stream_errors=0)
        self.events: deque = deque(maxlen=200)   # human-readable log of state transitions

    def log(self, msg: str):
        self.events.append(f"{time.strftime('%H:%M:%S')} {msg}")
        print(msg, flush=True)

    # ---------------- health ----------------
    async def health_loop(self):
        while True:
            await asyncio.gather(*(self.check(b) for b in self.backends.values() if not b.removed))
            await asyncio.sleep(self.h_interval)

    async def check(self, b: Backend):
        b.last_check = time.time()
        try:
            r = await self.client.get(f"{b.url}/health", timeout=self.h_timeout)
            ok = r.status_code == 200
            err = None if ok else f"health {r.status_code}"
        except Exception as e:  # noqa: BLE001
            ok, err = False, f"health {type(e).__name__}"
        self._observe(b, ok, err)

    def _observe(self, b: Backend, ok: bool, err: str | None):
        if ok:
            b.consec_ok += 1
            b.consec_fail = 0
            if not b.healthy and b.consec_ok >= self.healthy_after:
                b.healthy = True
                self.log(f"[health] {b.name} -> HEALTHY")
        else:
            b.consec_fail += 1
            b.consec_ok = 0
            b.last_error = err
            if b.healthy and b.consec_fail >= self.unhealthy_after:
                b.healthy = False
                self.log(f"[health] {b.name} -> UNHEALTHY ({err})")

    def mark_down(self, b: Backend, err: str):
        """Passive check: a hard failure on a live request takes it out immediately."""
        if b.healthy:
            b.healthy = False
            b.consec_ok = 0
            b.last_error = err
            self.log(f"[passive] {b.name} -> UNHEALTHY ({err})")

    # ---------------- selection ----------------
    def pick(self, exclude: set[str], prompt_key: str | None) -> Backend | None:
        cands = [b for b in self.backends.values() if b.eligible and b.name not in exclude]
        if not cands:
            return None
        if self.policy == "least_inflight":
            return min(cands, key=lambda b: (b.inflight + 1) / max(b.weight, 1e-9))
        if self.policy == "prefix_affinity" and prompt_key:
            # Rendezvous (highest-random-weight) hashing: stable per key, respects
            # weights, and only keys on a removed backend move when it goes away.
            def score(b: Backend) -> float:
                h = int(hashlib.blake2b(f"{prompt_key}|{b.name}".encode(), digest_size=8).hexdigest(), 16)
                u = (h + 1) / 2**64                      # uniform in (0, 1]
                return -max(b.weight, 1e-9) / math.log(u)  # weighted rendezvous score
            return max(cands, key=score)
        weights = [b.weight for b in cands]
        return random.choices(cands, weights=weights, k=1)[0]

    # ---------------- drain ----------------
    async def drain(self, b: Backend, timeout_s: float = 60.0) -> dict[str, Any]:
        b.draining = True
        self.log(f"[drain] {b.name} draining, inflight={b.inflight}")
        t0 = time.perf_counter()
        while b.inflight > 0 and time.perf_counter() - t0 < timeout_s:
            await asyncio.sleep(0.05)
        b.removed = b.inflight == 0
        self.log(f"[drain] {b.name} {'REMOVED' if b.removed else 'TIMEOUT (still inflight)'} after {time.perf_counter() - t0:.1f}s")
        return dict(name=b.name, removed=b.removed, drained_in_s=round(time.perf_counter() - t0, 2), inflight=b.inflight)

    # ---------------- shadow ----------------
    async def shadow_copy(self, path: str, body: bytes, headers: dict):
        for b in self.backends.values():
            if not (b.shadow and b.healthy and not b.removed):
                continue
            self.totals["shadow_sent"] += 1
            asyncio.create_task(self._shadow_one(b, path, body, headers))

    async def _shadow_one(self, b: Backend, path: str, body: bytes, headers: dict):
        t0 = time.perf_counter()
        b.inflight += 1
        b.requests += 1
        try:
            async with self.client.stream("POST", f"{b.url}{path}", content=body, headers=headers, timeout=self.timeout) as r:
                async for _ in r.aiter_raw():
                    pass  # drain and discard
                if r.status_code >= 400:
                    b.errors += 1
                    self.totals["shadow_errors"] += 1
        except Exception as e:  # noqa: BLE001
            b.errors += 1
            b.last_error = f"shadow {type(e).__name__}"
            self.totals["shadow_errors"] += 1
        finally:
            b.inflight -= 1
            b.lat_ms.append((time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def build_app(router: Router) -> FastAPI:
    app = FastAPI(title="inference-router")
    HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(router.health_loop())

    @app.get("/health")
    async def health():
        n = sum(1 for b in router.backends.values() if b.eligible)
        return JSONResponse({"status": "ok" if n else "no_backends", "eligible": n}, status_code=200 if n else 503)

    @app.get("/metrics")
    async def metrics():
        return dict(policy=router.policy, totals=router.totals,
                    backends=[b.snapshot() for b in router.backends.values()],
                    events=list(router.events)[-20:])

    @app.get("/metrics/prom", response_class=PlainTextResponse)
    async def metrics_prom():
        lines = []
        for k, v in router.totals.items():
            lines.append(f'router_{k}_total {v}')
        for b in router.backends.values():
            lab = f'backend="{b.name}",stage="{b.stage}"'
            lines += [f'router_backend_healthy{{{lab}}} {int(b.healthy)}',
                      f'router_backend_inflight{{{lab}}} {b.inflight}',
                      f'router_backend_requests_total{{{lab}}} {b.requests}',
                      f'router_backend_errors_total{{{lab}}} {b.errors}',
                      f'router_backend_weight{{{lab}}} {b.weight}']
        return "\n".join(lines) + "\n"

    # ---- admin ----
    @app.get("/admin/backends")
    async def list_backends():
        return [b.snapshot() for b in router.backends.values()]

    @app.post("/admin/backends")
    async def add_backend(body: dict):
        b = Backend(name=body["name"], url=body["url"].rstrip("/"), weight=float(body.get("weight", 1)),
                    stage=body.get("stage", "canary"), shadow=bool(body.get("shadow", False)))
        router.backends[b.name] = b
        router.log(f"[admin] added {b.name} weight={b.weight} stage={b.stage} shadow={b.shadow}")
        return b.snapshot()

    def _get(name: str) -> Backend:
        if name not in router.backends:
            raise HTTPException(404, f"no backend {name}")
        return router.backends[name]

    @app.post("/admin/backends/{name}/weight")
    async def set_weight(name: str, weight: float):
        b = _get(name); b.weight = weight
        router.log(f"[admin] {name} weight -> {weight}")
        return b.snapshot()

    @app.post("/admin/backends/{name}/drain")
    async def drain(name: str, timeout_s: float = 60.0):
        return await router.drain(_get(name), timeout_s)

    @app.post("/admin/backends/{name}/undrain")
    async def undrain(name: str):
        b = _get(name); b.draining = False; b.removed = False
        router.log(f"[admin] {name} undrained")
        return b.snapshot()

    @app.delete("/admin/backends/{name}")
    async def remove_now(name: str):
        """The WRONG way: remove without draining. In-flight requests on it die."""
        b = _get(name); b.removed = True
        router.log(f"[admin] {name} REMOVED immediately with inflight={b.inflight}")
        return b.snapshot()

    @app.post("/admin/policy")
    async def set_policy(policy: str):
        router.policy = policy
        router.log(f"[admin] policy -> {policy}")
        return {"policy": policy}

    # ---- the proxy ----
    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def proxy(req: Request):
        path = req.url.path
        body = await req.body()
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
        router.totals["requests"] += 1
        prompt_key = None
        if router.policy == "prefix_affinity":
            try:
                msgs = json.loads(body).get("messages") or []
                prompt_key = (msgs[0].get("content") or "")[:64] if msgs else None
            except Exception:  # noqa: BLE001
                prompt_key = None

        await router.shadow_copy(path, body, headers)

        tried: set[str] = set()
        attempts = 0
        last_err = "no attempt"
        while attempts < router.max_attempts:
            b = router.pick(tried, prompt_key)
            if b is None:
                break
            attempts += 1
            tried.add(b.name)
            if attempts > 1:
                router.totals["retries"] += 1
                b.retries_absorbed += 1
            b.inflight += 1
            b.requests += 1
            t0 = time.perf_counter()
            try:
                upstream = router.client.build_request("POST", f"{b.url}{path}", content=body, headers=headers, timeout=router.timeout)
                r = await asyncio.wait_for(router.client.send(upstream, stream=True), timeout=router.ttft_timeout)
            except Exception as e:  # noqa: BLE001
                b.inflight -= 1
                b.errors += 1
                last_err = f"{b.name}: {type(e).__name__}"
                if isinstance(e, (httpx.ConnectError, httpx.RemoteProtocolError, asyncio.TimeoutError, httpx.ReadTimeout)):
                    router.mark_down(b, last_err)
                router.totals["failovers"] += 1
                continue

            if r.status_code in router.retry_on:
                await r.aclose()
                b.inflight -= 1
                b.errors += 1
                last_err = f"{b.name}: HTTP {r.status_code}"
                router._observe(b, False, last_err)  # counts toward unhealthy_after
                router.totals["failovers"] += 1
                continue

            # Committed: from here we stream and cannot retry.
            b.ttft_ms.append((time.perf_counter() - t0) * 1000)
            resp_headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP}
            resp_headers["x-router-backend"] = b.name
            resp_headers["x-router-attempts"] = str(attempts)

            async def body_iter(r=r, b=b, t0=t0):
                try:
                    async for chunk in r.aiter_raw():
                        yield chunk
                    router.totals["ok"] += 1
                except Exception as e:  # noqa: BLE001
                    b.errors += 1
                    router.totals["mid_stream_errors"] += 1
                    router.totals["errors"] += 1
                    router.mark_down(b, f"mid-stream {type(e).__name__}")
                    # Can't retry: client already has partial output. Surface it in-band.
                    yield b'data: {"error": "upstream died mid-stream"}\n\n'
                finally:
                    b.inflight -= 1
                    b.lat_ms.append((time.perf_counter() - t0) * 1000)
                    await r.aclose()

            return StreamingResponse(body_iter(), status_code=r.status_code,
                                     headers=resp_headers, media_type=r.headers.get("content-type"))

        router.totals["errors"] += 1
        if attempts == 0:
            router.totals["no_backend"] += 1
        return JSONResponse({"error": {"message": f"no healthy backend after {attempts} attempt(s): {last_err}"}},
                            status_code=503, headers={"x-router-attempts": str(attempts)})

    return app


cli = typer.Typer(add_completion=False, help=__doc__)


@cli.command()
def serve(config: Path = typer.Option(Path(__file__).with_name("config.yaml")),
          port: int | None = typer.Option(None), host: str = typer.Option("127.0.0.1")):
    cfg = yaml.safe_load(config.read_text())
    listen = cfg.get("listen", "127.0.0.1:8000")
    h, p = listen.split(":")
    router = Router(cfg)
    uvicorn.run(build_app(router), host=host or h, port=port or int(p), log_level="warning")


def main():
    cli()


if __name__ == "__main__":
    main()
