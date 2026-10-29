# Deploy targets

Every backend here speaks the same OpenAI-compatible `/v1/chat/completions`, so the router and
load tester don't care which one they hit. Measure each with `loadtest`, then put two of them
behind the router and run the Day 2 drills against real GPUs.

| Target | Folder | GPU $/h (Oct 2026) | Notes |
|---|---|---|---|
| Baseten (vLLM in a custom docker server) | `baseten/` | H100 6.50, A10G 1.21, L4 0.85 | Autoscaling set in UI/API. BDN prefetches weights. Scale-to-zero default. |
| Modal (vLLM via `@app.server`) | `modal/` | H100 3.95, A10 1.10, L4 0.80 | `min_containers`, `scaledown_window`, `target_concurrency` in code. Per-second billing. $30/mo free credits. |
| RunPod serverless (vLLM worker) | below | A100 2.72, H100 5.39 | No code: pick the `runpod-workers/worker-vllm` Hub template, set the model. |
| Local real model (Apple Silicon) | below | your laptop | `mlx-lm` serves an OpenAI API; real TTFT/tok/s numbers with zero cloud cost. |
| Simulated (`mockllm`) | `src/.../mockllm` | configurable | Realistic batching/KV/cold-start physics; the default for drills. |

## RunPod serverless (optional third comparison point)

1. console.runpod.io → Serverless → New endpoint → Hub template `runpod-workers/worker-vllm`.
2. Model `Qwen/Qwen2.5-7B-Instruct`, Max Model Length 8192, GPU A100 or H100, workers min 0 / max 2.
3. Base URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai` (note: the load tester appends `/v1/...`).

```bash
uv run loadtest --url https://api.runpod.ai/v2/<ENDPOINT_ID>/openai --api-key "$RUNPOD_API_KEY" \
  --model Qwen/Qwen2.5-7B-Instruct --price-per-hour 2.72 --label runpod-a100 --out results/runpod-a100.json
```

RunPod "flex" workers scale from zero per request, so cold starts show up as p99 TTFT, not as a
`/health` transition. Compare that with Baseten's queue-while-waking and Modal's 429-when-over-cap
behaviours: three different answers to "what happens when a request arrives and there's no capacity".

## Local real model with mlx-lm (Apple Silicon)

```bash
uv add mlx-lm
uv run mlx_lm.server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8003
uv run loadtest --url http://127.0.0.1:8003 --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --concurrency 1,2,4 --requests 8 --max-tokens 64 --price-per-hour 0 --label mlx-1.5b
```

mlx-lm's server is single-sequence (no continuous batching), so the concurrency sweep flattens
immediately: aggregate tok/s stays flat while latency grows linearly. Put it next to the `mockllm`
sweep and you have the clearest picture possible of what continuous batching buys.
