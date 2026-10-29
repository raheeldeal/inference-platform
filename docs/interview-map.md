# Interview map: their posts → what I built → the number I can quote

Fill the right-hand column from your own runs. Quoting a number you measured beats any amount of
"I understand the tradeoff".

## Baseten Delivery Network (BDN): fast cold starts for big models
Posts: Mar 19 2026 and Apr 9 2026 (links in `deploy/baseten/README.md`).
Claim: mirror weights into Baseten storage near the GPUs, multi-tier cache (node-local disk →
in-cluster peer cache → mirrored origin), single-flight downloads, weights transferred before the
container starts; 2–3x faster cold starts.

| What I built | Number |
|---|---|
| `mockllm --cold-start-s`: /health 503 until weights "load"; requests wait. `loadtest --cold-start` measures it. | Measured cold start on Baseten: __ s; on Modal (Volume-cached): __ s |
| `autosim --cold-start 30,180`: same policy, 6x cold start. | SLO attainment __ % → __ %; the cold-start colour is the biggest separator on the chart |
| `deploy/baseten/config.yaml` uses the `weights:` block so BDN prefetches before the container starts. | |

Talking point: cold start is not one number; it is four phases, and BDN attacks the one that scales
with model size. Scale-to-zero is only a product you can sell if that phase is short, so BDN and
the autoscaler defaults are one decision.

## Multi-cloud Capacity Management (MCM)
Posts: Jun 9 2025 (unifies cloud / self-hosted / hybrid), Jun 23 2025 (how we built it), Sep 3 2025.
Claim: one scheduler over GPU pools across 20+ clouds and dozens of regions; "Kubernetes at global
scale"; same stack whether capacity is Baseten's, the customer's own, or hybrid.

| What I built | Number |
|---|---|
| The router treats heterogeneous backends (H100 sim, A10G sim, real Baseten, real Modal) as one pool with weights and health. | Under `least_inflight` vs `weighted` with mixed hardware, p99 __ ms → __ ms |
| `docs/placement-policy-spec.md`: require / prefer / fallback over pools with attributes. | |
| Drain-before-remove = moving a replica between pools without dropping traffic. | Rolling deploy: __ requests in flight on the old backend, __ errors |

Talking point: multi-cloud is a placement problem (where does the next replica go) plus a traffic
problem (how do requests find it, and how do you move them). I built the second and wrote the spec
for the first.

## Baseten on Nebius (AI video)
Post: Oct 6 2025. Claim: text-to-video on Nebius capacity (US / Finland / France) via the Baseten
Inference Stack + MCM; bursty, long-running, GPU-heavy workloads.

| What I built | Number |
|---|---|
| `autosim --trace bursty` and `idle-spiky`: the scaling pathologies of bursty workloads with long service times. | With `service_s` set to 60 s (video-like), the concurrency signal overshoots to max replicas __ times in an hour; the rate signal __ times |
| Prefix-affinity routing: the video analogue is session/asset affinity. | |

Talking point: video jobs are minutes, not seconds, so a 30 s cold start is cheap relative to the
job but the overshoot problem gets *worse* (the queue signal grows for the whole cold start). Rate-
based or predictive signals matter more, not less.

## The JD line by line

| JD phrase | Where I did it |
|---|---|
| scaling | `autosim`: 72 policies, cost-vs-SLO frontier, concurrency vs rate signal |
| routing | `router`: weighted / least-inflight / prefix-affinity, health, retries |
| failover | `scripts/chaos.sh`: injected 500s → 0 user errors; hard kill → only in-flight streams lost |
| safe rollouts | canary by weight, shadow traffic, drain; `scripts/rolling-deploy.sh` |
| cost/perf | `loadtest` + `frontier.png`: $/1M tokens vs p99 per backend |
| unified placement policy | `docs/placement-policy-spec.md` |
| customer config UX | the `placement:` YAML and its validation rules |
| observability surface | `/metrics` per backend; the per-replica placement fields in the spec |

## Questions I want to ask them

- How does the autoscaler's signal treat requests queued at the ingress during a cold start?
- When `prefer` cannot be met, does a replica ever migrate back automatically, or only on scale-down?
- What does a customer see today when a replica is placed somewhere they did not expect?
- Is prefix-cache-aware routing on the roadmap for dedicated deployments, and who owns it?

## Honest framing for the "8+ years" gap

"I have not run this at your scale. I have run a version of it end to end, broken it in the ways
your posts describe, and I have numbers for each tradeoff. Here they are." Then show the two charts.
