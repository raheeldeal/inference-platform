# Exercises: break it, then explain what you saw

Each one: a command, what you should see, and the question to answer in one sentence afterwards.
Write your answers in `results/notes.md`; they are your interview stories.

Backends and router up first: `scripts/up.sh`.

## Day 1: measuring

**E1. Find the knee.** `scripts/sweep.sh`, open `results/frontier.png`.
See: throughput climbs steeply to about concurrency 16-32 on the h100 preset, then flattens, while
p99 keeps climbing.
Q: What resource saturated at the knee? (Check `/metrics` on the backend during the c=128 level:
is `waiting > 0`? is `kv_util` near 1? is `running` pinned at `max_num_seqs`?)

**E2. Long prompts.** `uv run loadtest --url :8002 --concurrency 1,8,32 --prompt-words 1200 --max-tokens 64`.
See: TTFT dominates latency; the knee moves left.
Q: Why does prompt length hurt the a10g preset more than the h100 one?

**E3. Chunked prefill.** Restart backend 8002 with `--no-chunked-prefill`; rerun E2 at c=8.
See: per-request tok/s p50 drops and gets noisier; TTFT p50 improves slightly.
Q: Who pays for a long prompt in each mode?

**E4. Quantization.** Compare `a10g-sim` and `a10g-awq-sim` on the frontier.
Q: At the customer's SLO of p99 < 2 s, how many tokens/s does each give, and what is $/1M at that point?

**E5. Cold start.** `uv run mockllm --name cold --port 8004 --preset h100-8b --cold-start-s 45 &`
then immediately `uv run loadtest --url http://127.0.0.1:8004 --cold-start`.
Q: Which of the four cold-start phases (GPU acquire, image pull, weight pull, model load) can a
platform shorten without the customer changing anything?

**E6. Real numbers.** Deploy `deploy/baseten` and `deploy/modal`. Sweep both. Overlay with the
sim. Then edit `PRESETS["h100-8b"]` until the sim matches within ~20 %.
Q: Which preset number was furthest off, and what does that tell you about the model?

## Day 2 AM: the router

**E7. Failover.** `scripts/chaos.sh`.
See: 500s → `failovers > 0, errors = 0`. Hard kill → `mid_stream_errors = N`, where N is the
number of streams on the victim at that instant.
Q: Why can those N not be retried?

**E8. Rolling deploy done right and wrong.** `scripts/rolling-deploy.sh`.
See: drain path → 0 errors. The "remove immediately" path → errors equal to in-flight on the victim.
Q: What would you need from the backend (not the router) to make drain take less time?

**E9. Promote a canary.** With load running (`uv run loadtest --url :8000 --concurrency 16 --requests 2000 &`):
```bash
curl -s localhost:8000/metrics | jq '.backends[] | {name, weight, requests, ttft_p99, lat_p99, errors}'
curl -X POST 'localhost:8000/admin/backends/a10g/weight?weight=50'
curl -X POST 'localhost:8000/admin/backends/a10g/weight?weight=0'
```
Q: The a10g canary has worse p99 than stable. Is that a regression or a hardware difference? How
would you tell with only the router's metrics?

**E10. Shadow.** Add a shadow backend:
```bash
uv run mockllm --name shadow --port 8005 --preset h100-8b --step-ms 14 &
curl -X POST localhost:8000/admin/backends -H 'content-type: application/json' \
  -d '{"name":"shadow","url":"http://127.0.0.1:8005","shadow":true}'
```
See: `shadow_sent` climbs; shadow's `lat_p50` is ~40 % worse than h100's on identical traffic.
Q: What is the one thing shadow traffic cannot tell you that a canary can?

**E11. Routing policy.** Put the h100 and a10g sims at equal weight. Run c=32 under `weighted`,
then `curl -X POST 'localhost:8000/admin/policy?policy=least_inflight'` and rerun.
See: p99 improves under `least_inflight` because the slow backend stops getting half the traffic.
Q: Why does weighted random put the same *number* of requests on each but not the same *load*?

**E12. Prefix affinity.** `policy=prefix_affinity`, send 200 requests with 3 distinct system prompts.
See: each prompt lands on exactly one backend (check `x-router-backend`).
Q: What happens to the keys when you drain one backend? Why rendezvous hashing and not `hash % n`?

**E13. Timeouts.** `curl -X POST 'localhost:8002/admin/fail?mode=timeout'` then send requests through the router.
See: TTFT timeout (30 s in config) fires, request fails over. Lower `ttft_timeout_s` to 2 and retry.
Q: Which timeout protects users and which protects the platform?

## Day 2 PM: autoscaling

**E14. Defaults.** Replay Baseten's defaults and Modal's:
```bash
uv run autosim --target-concurrency 1,8,32 --scale-down-delay 900 --min-replicas 0 --cold-start 30,180 --plot-out results/baseten-like.png
uv run autosim --target-concurrency 32 --scale-down-delay 60 --min-replicas 0,1 --cold-start 30 --plot-out results/modal-like.png
```
Q: What is the price of each extra nine of SLO attainment on the bursty trace?

**E15. The overshoot.** `uv run autosim --signal concurrency --min-replicas 1 --cold-start 30 --target-concurrency 16 --scale-down-delay 900`.
Look at the bottom-left panel: one burst takes replicas from 1 to 20.
Q: Describe the feedback loop in one sentence. Then rerun with `--signal rate`.

**E16. Scale-to-zero.** `uv run autosim --trace idle-spiky --min-replicas 0,1 --cold-start 30,180`.
Q: At what cold-start time does scale-to-zero stop being worth it for this trace?

**E17. Your own trace.** Replace `make_trace` with a CSV of real request timestamps (a day of your
product's logs, or a public trace). Everything else works unchanged.

## Day 3: write it down

**E18.** Fill in `docs/interview-map.md` with your real numbers.
**E19.** Finish `docs/placement-policy-spec.md`.
**E20.** Post the frontier chart and the autoscaler chart with three sentences each.
