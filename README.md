# inference-platform

A mini LLM inference platform you can run on a laptop and then point at real GPUs:
serve a model, put your own router in front, break it, autoscale it, and measure cost vs latency.

```
                         ┌──────────────────────────────┐
   loadtest ──────────▶  │  router  :8000               │
   (sweep, chaos)        │  health · retry · failover   │
                         │  canary 5/95 · shadow · drain│
                         └───┬──────────┬──────────┬────┘
                             ▼          ▼          ▼
                       mockllm:8001  mockllm:8002  mockllm:8003     ← simulated vLLM (continuous
                       "h100"        "a10g"        "a10g-awq"          batching, KV cache, cold start)
                             or Baseten / Modal / RunPod / mlx-lm  ← real backends, same API

   autosim ───▶ replays bursty traffic against scaling policies ───▶ cost-vs-SLO chart
```

Everything speaks OpenAI's `/v1/chat/completions`, so the client code never changes when you swap
a simulated backend for a real one.

## Quickstart (10 minutes, no GPU, no accounts)

```bash
uv sync
scripts/up.sh                  # 3 simulated backends + the router, logs in results/
scripts/sweep.sh               # Day 1: concurrency sweep on each backend -> results/frontier.png
scripts/chaos.sh               # Day 2: inject 500s, then hard-kill a backend under load
scripts/rolling-deploy.sh      # Day 2: drain -> kill -> re-add; zero dropped requests
uv run autosim                 # Day 2 PM: 72 autoscaling policies -> results/autoscaler.png
scripts/down.sh
```

Open `results/frontier.png` and `results/autoscaler.png`. Then read `docs/LEARNING.md` with the
numbers in front of you, and work through `docs/EXERCISES.md`.

## Layout

| Path | What |
|---|---|
| `src/inference_platform/mockllm/` | Simulated vLLM-style server. Models continuous batching, prefill vs decode, KV-cache admission, chunked prefill, cold start, fault injection. |
| `src/inference_platform/loadtest/` | Closed-loop load generator: TTFT, tok/s, p50/p99, error rate, $/1M tokens; concurrency sweep; frontier plot. |
| `src/inference_platform/router/` | FastAPI gateway: active+passive health, retry before first byte, failover, weighted canary, shadow traffic, drain, three routing policies. |
| `src/inference_platform/autoscaler/` | Discrete-time autoscaler simulator: traces, policies (concurrency vs rate signal, cold start, scale-to-zero, scale-down delay), cost-vs-SLO chart. |
| `deploy/baseten/` | Truss `config.yaml` for Qwen2.5-7B on vLLM, verified Oct 2026, plus CLI and pricing notes. |
| `deploy/modal/` | Modal `@app.server` vLLM app for the same model. |
| `docs/LEARNING.md` | The concepts, each tied to a knob in this repo and an experiment that shows it. |
| `docs/EXERCISES.md` | Break-it drills with expected observations. |
| `docs/placement-policy-spec.md` | The one-page "unified placement policy" spec (Day 3). |
| `docs/interview-map.md` | Their posts (BDN, MCM, Nebius) mapped to what you built, with your numbers. |
| `scripts/` | up / down / sweep / chaos / rolling-deploy. |
| `results/` | Generated JSON, logs, PNGs. |

## The three days, as commands

**Day 1 — serve and measure.** `scripts/sweep.sh` locally first so you know what the curve should
look like. Then deploy `deploy/baseten` and `deploy/modal`, and run `loadtest` against each with
`--price-per-hour` set (prices in `deploy/README.md`). Plot all of them on one frontier:

```bash
uv run python -m inference_platform.loadtest.plot results/baseten-h100.json results/modal-h100.json results/sim-h100.json
```

Then tune the `mockllm` presets so the simulated curve matches the real one. Now your chaos and
autoscaling experiments run on a calibrated model instead of a guess.

**Day 2 AM — router.** Put the two real backends in `router/config.yaml` (or keep the sims), run
`scripts/chaos.sh` and `scripts/rolling-deploy.sh`. Then the exercises: shadow a new build,
promote a canary by weight, switch to `prefix_affinity` and watch what it does to TTFT.

**Day 2 PM — autoscaler.** `uv run autosim --trace bursty --cold-start 30,180`. Re-run with
Baseten's defaults (`--target-concurrency 1 --scale-down-delay 900`) and Modal's
(`--scale-down-delay 60`). Argue the tradeoff with the chart.

**Day 3 — ammo.** Fill in `docs/interview-map.md` with your real numbers, finish
`docs/placement-policy-spec.md`, ship.

## Honest scope

The simulator reproduces the *shape* of real serving behaviour (why throughput saturates, why p99
explodes at the knee, why cold starts make autoscalers overshoot). The absolute numbers are
presets until you calibrate them against a real deployment on Day 1. Say that out loud in the
interview; it is more credible than pretending the sim is the truth.
