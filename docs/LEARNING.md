# Learning notes: what each concept is, where it lives in this repo, how to see it

Read this with `results/frontier.png` and `results/autoscaler.png` open.

## 1. Serving physics

### Prefill vs decode
A request has two phases. **Prefill** runs the whole prompt through the model in one go to build
the KV cache; it is compute-bound and its cost scales with prompt length. **Decode** generates one
token per forward pass; each pass has to read all the model weights (and the KV cache) from GPU
memory to produce one token per sequence, so it is memory-bandwidth-bound and the GPU's compute
units sit mostly idle at batch size 1.

- In the repo: `mockllm/server.py`, `Engine._loop`. Prefill cost is `prompt_tokens * prefill_us`;
  decode is one `step_ms` per token for the whole batch.
- See it: run `loadtest --prompt-words 20` and `--prompt-words 800` at concurrency 1. TTFT moves,
  per-request tok/s barely does.

### Time to first token (TTFT) vs tokens per second
TTFT = queue wait + prefill. Tok/s = 1 / decode step time. They are governed by different resources,
so a platform can be great at one and bad at the other. Users feel TTFT; dashboards show tok/s.

- See it: in the sweep table, TTFT p99 climbs *before* aggregate tok/s stops climbing. That gap is
  queueing.

### Continuous batching
Static batching waits for a batch to finish before starting the next. Continuous (iteration-level)
batching lets a new sequence join the running batch at the next decode step and lets finished ones
leave. Because decode is memory-bound, adding a sequence to the batch costs almost nothing until
the batch is big enough to become compute-bound. That is why aggregate throughput rises ~10x from
concurrency 1 to 32 while per-request tok/s only falls ~20 %.

- In the repo: `batch_slope` in the presets is the "almost nothing"; `step_ms * (1 + slope * batch)`.
- See it: panel 1 of `frontier.png`. The curve is steep then flat. The knee is where you stop
  getting free throughput.
- Compare: `mlx_lm.server` (no continuous batching) gives a flat aggregate tok/s line.

### KV cache: the real capacity limit
Every token in flight (prompt + generated so far) holds a key/value vector per layer in GPU memory.
For an 8B model in bf16 that is roughly 128 KB per token; 80 GB of H100 minus 16 GB of weights
leaves room for ~400k tokens, but an A10G's 24 GB minus 16 GB leaves ~50k. That, not compute, is
what caps `max_num_seqs` in practice. vLLM's PagedAttention allocates this cache in fixed-size
blocks so it is not fragmented, and `--gpu-memory-utilization` sets how much to reserve.

- In the repo: `kv_tokens` in the preset; `Engine._try_admit` refuses to admit when
  `prompt + max_tokens` would not fit, and the request sits in `waiting` (counted as
  `preempt_waits` in `/metrics`).
- See it: run the a10g sim at concurrency 64 with `--max-tokens 512 --prompt-words 400`.
  Watch `curl :8002/metrics` show `waiting > 0` and `kv_util ~ 1.0`; p99 explodes.

### Chunked prefill
Without it, a long prompt's prefill occupies a whole engine step and every running sequence's next
token is delayed by it (visible as token-cadence jitter). With it, prefill is split into chunks
mixed into decode steps: smoother per-token latency for running requests, slightly higher TTFT for
the new one. vLLM flag: `--enable-chunked-prefill` (default on in V1).

- In the repo: `--chunked-prefill/--no-chunked-prefill` on `mockllm`, `chunk_size=512`.
- See it: run concurrency 8 with `--prompt-words 1500` both ways and compare per-request tok/s p50.

### Quantization
Weights in int4/int8/fp8 instead of bf16. Two effects: less memory traffic per decode step (faster
single-stream tok/s on memory-bound hardware) and a smaller weight footprint, which leaves more
room for KV cache (higher concurrency before the knee). Costs: some quality loss, and prefill may
not get faster. AWQ and GPTQ are weight-only int4; fp8 on H100 is nearly free in quality.

- In the repo: the `a10g-8b-awq` preset: `step_ms` 28→18, `kv_tokens` 40k→110k.
- See it: `frontier.png` series `a10g-awq-sim` vs `a10g-sim`. Same price, further knee.

### Cold starts
Time from "a replica is needed" to "it serves": acquire a GPU, pull the container image, pull the
weights (15 GB for 7B bf16), load them into GPU memory, warm up CUDA graphs / compile. Weight pull
dominates for big models, which is what Baseten's BDN (mirror weights near the GPUs, multi-tier
cache, prefetch before the container starts) and Modal's Volumes attack.

- In the repo: `--cold-start-s` on `mockllm`; `/health` returns 503 until ready and requests wait.
  `loadtest --cold-start` times it. `autosim --cold-start 30,180` shows what it does to scaling.
- See it: the two colours in `autoscaler.png`. Same policy, 6x cold start, SLO goes from 97 % to
  below 50 %.

## 2. The cost/performance frontier

Pick a concurrency; it buys you a throughput and costs you a latency. `$ per 1M tokens` =
`$/hr ÷ (tok/s × 3600) × 1e6`. At concurrency 1 on an H100 the sim gives ~$11 per 1M tokens; at 32
it is ~$1. The whole business of an inference platform is choosing points on this curve for
customers and packing them efficiently.

Questions to be able to answer:
- Where is the knee for this model on this GPU, and what sets it (KV cache? `max_num_seqs`? compute)?
- What is the p99 at the knee? Is it within the customer's SLO?
- Which is cheaper per token at the customer's SLO: one H100 at concurrency 32, or three L4s at 8?

## 3. Traffic management (the router)

### Health: active and passive
Active checks poll `/health` every second and need N consecutive failures to flip state (hysteresis
so one blip does not thrash routing). Passive checks mark a backend down the moment a live request
hits a connection error, because waiting two check intervals is two seconds of errors.

- In the repo: `Router.health_loop` and `Router.mark_down`. Events show in `/metrics`.

### Retry only before the first byte
A streaming response commits you. Once a token reached the client you cannot replay the request
elsewhere without duplicating output. So the router retries on connect errors, on 5xx, and on a
TTFT timeout, and never after. This is why TTFT timeout and connect timeout are the important
timeouts; total timeout barely matters.

- See it: `scripts/chaos.sh`. Injected 500s → failovers, zero client errors. Hard kill →
  `mid_stream_errors` for whatever was streaming, zero errors for new requests.

### Canary by weight, promote by weight
5/95 split. Compare the canary's `ttft_p99`, `errors`, `lat_p99` in `/metrics` against stable
*on the same traffic mix*. Promote by raising the weight. Roll back by setting it to 0. The
abstraction customers want is "stage", the mechanism is a number.

### Shadow traffic
A shadow backend receives a copy of every request and its response is discarded. You get real
traffic on a new build with zero user exposure. Cost: it doubles load on the shadowed slice, and
you must not shadow non-idempotent requests (fine for inference).

### Drain
Stop sending new requests, wait for in-flight to hit zero, then remove. A rolling deploy is: add
new → health → shift weight → drain old → remove old. `DELETE /admin/backends/x` without drain is
the anti-pattern; `scripts/rolling-deploy.sh` does both so you can see the difference.

### Routing policies
- `weighted`: simple, stateless, the default for canaries.
- `least_inflight`: better p99 when backends differ in speed (an H100 and an A10G behind one
  router); weighted random would send the A10G too much.
- `prefix_affinity`: hash the prompt prefix so the same system prompt lands on the same replica,
  so vLLM's prefix cache actually hits. Uses rendezvous hashing so only the keys of a removed
  backend move. This is what cache-aware routers (SGLang's, for instance) do with more
  sophistication.

## 4. Autoscaling

### The control loop
Signal → desired replicas → actuator with delay. Desired = ceil(signal / target). The actuator's
delay is the cold start; the signal's definition is the whole game.

### Why it overshoots
If the signal is observed concurrency *including queued requests*, then during a cold start the
queue grows, the signal grows, and the scaler asks for more replicas that will arrive after the
queue would have drained anyway. The sim shows a one-replica deployment overshoot to 20 on a burst
it could have handled with 4. Fixes: a rate-based signal (Little's law: replicas = arrival rate ×
service time ÷ target), a panic window with a cap on step size, or counting booting replicas as
pending capacity with a predicted drain time. Knative's KPA and Baseten's `autoscaling_window` are
both versions of this.

- In the repo: `--signal concurrency,rate`. Compare their dots on the chart.

### Scale-down delay vs cost
Short delay (60 s): cheap, but every burst after a lull pays a cold start. Long delay (900 s,
Baseten's default): expensive, but p99 is stable. The chart shows the price of each extra nine.

### Scale-to-zero
`min_replicas = 0` is free when idle and pays a full cold start on the first request. Only
sensible when cold start is short relative to the user's patience, which is why cold-start
engineering (BDN) and scale-to-zero are the same product decision.

### Little's law
`concurrency = arrival rate × latency`. If you know two you know the third. Use it to sanity-check
any autoscaling target: 20 req/s × 2 s = 40 in flight, so at target concurrency 16 you need 3
replicas, before any burst headroom.

## 5. Vocabulary you should be able to use without hesitating
TTFT, ITL (inter-token latency), TPOT (time per output token), tok/s, goodput, p50/p99/p999,
closed-loop vs open-loop load, offered load, saturation, knee, head-of-line blocking, admission
control, backpressure, 429 vs queue, hysteresis, draining, canary, shadow, blue/green, rolling,
PagedAttention, KV cache, prefix caching, chunked prefill, speculative decoding, tensor parallel,
quantization (AWQ/GPTQ/fp8), CUDA graphs, cold start, scale-to-zero, Little's law.
