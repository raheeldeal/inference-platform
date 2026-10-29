# Baseten (Truss) deployment

Facts below verified on 2026-10-29 against docs.baseten.co. Re-check before the interview; they move fast.

## Deploy

```bash
brew tap basetenlabs/baseten && brew install baseten     # Baseten CLI (current)
baseten auth login
cd deploy/baseten
baseten model push --tail --wait                          # prints the model id
```

Legacy path, still works: `uvx truss push --publish` from the same directory.

Get an API key at https://app.baseten.co/settings/api_keys and `export BASETEN_API_KEY=...`.

## Call it

Any route on the vLLM server is reachable under `/sync/`:

```bash
export BASETEN_URL="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync"
curl $BASETEN_URL/v1/chat/completions \
  -H "Authorization: Bearer $BASETEN_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

## Measure it (Day 1)

```bash
uv run loadtest --url $BASETEN_URL --api-key "$BASETEN_API_KEY" \
  --model Qwen/Qwen2.5-7B-Instruct --price-per-hour 6.50 \
  --concurrency 1,2,4,8,16,32,64 --label baseten-h100 --out results/baseten-h100.json

# cold start: scale to zero in the UI, wait, then
uv run loadtest --url $BASETEN_URL --api-key "$BASETEN_API_KEY" --cold-start
```

Baseten `/health` on a scaled-to-zero deployment may return a wake-up response rather than 503;
if `--cold-start` returns instantly, time the first real request instead
(`--concurrency 1 --requests 1`) and read its TTFT.

## Autoscaling knobs (set in UI/CLI/API, not config.yaml)

```bash
baseten model deployment update-autoscaling --model-id <id> --deployment-id <id> \
  --min-replica 0 --max-replica 4
# or PATCH https://api.baseten.co/v1/models/{model_id}/deployments/{deployment_id}/autoscaling_settings
# {"min_replica":0,"max_replica":4,"autoscaling_window":60,"scale_down_delay":900,
#  "concurrency_target":32,"target_utilization_percentage":70}
```

Defaults: min 0 (scale-to-zero), max 1, window 60 s, scale-down delay 900 s, concurrency target 1,
target utilisation 70 %. Desired replicas = ceil(avg concurrent / (concurrency_target × utilisation)).
These are exactly the knobs in `autosim` (`--target-concurrency`, `--scale-down-delay`, `--min-replicas`,
`window_s`), so you can replay their defaults against your own trace.

Cold-start phases they document: resource acquisition → image pull → weight pull (BDN) → model load.
Wake endpoint: `POST https://model-<id>.api.baseten.co/deployment/<dep_id>/wake`.

## Prices (dedicated, per hour, Oct 2026)

| GPU | $/h |
|---|---|
| T4 | 0.63 |
| L4 | 0.85 |
| A10G | 1.21 |
| H100 MIG 40 GB | 3.75 |
| A100 80 GB | 4.00 |
| H100 80 GB | 6.50 |
| B200 | 9.98 |

Billed per minute while deploying, scaling or serving; nothing while scaled to zero.

## TensorRT-LLM alternative (Engine Builder)

Same model via their compiled-engine path, for comparing against vLLM:

```yaml
model_name: qwen25-7b-trtllm
model_metadata:
  tags: [openai-compatible]
resources:
  accelerator: H100
  use_gpu: true
trt_llm:
  build:
    base_model: decoder
    checkpoint_repository:
      source: HF
      repo: "Qwen/Qwen2.5-7B-Instruct"
    max_seq_len: 8192
    quantization_type: fp8        # the quantisation lever, see docs/LEARNING.md
    tensor_parallel_count: 1
```

## Their posts to map to what you built

- Baseten Delivery Network (BDN), Mar 19 2026: https://www.baseten.co/blog/baseten-delivery-network-fast-cold-starts-big-models/
- How BDN makes cold starts fast, Apr 9 2026: https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/
- How we built Multi-cloud Capacity Management (MCM), Jun 23 2025: https://www.baseten.co/blog/how-we-built-multi-cloud-capacity-management/
- MCM unifies cloud / self-hosted / hybrid, Jun 9 2025: https://www.baseten.co/blog/how-baseten-multi-cloud-capacity-management-mcm-powers-cloud-self-hosted-and-hybr/
- AI video on Nebius, Oct 6 2025: https://www.baseten.co/blog/ai-video-nebius-baseten-inference-stack/
