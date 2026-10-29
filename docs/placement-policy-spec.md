# Spec: Unified Placement Policy

*Draft 0.1. One page. Fill the `[ ]` sections with numbers from your experiments.*

## Problem

A customer deploys a model and today answers several unrelated questions in several places: which
region, which clouds are acceptable, which GPU, how many replicas, whether to allow spillover when
their preferred capacity is full. The answers interact (the cheapest capacity is often in the wrong
region; the compliant region may have no H100s at 2 pm), and when they conflict the platform picks
silently. Customers find out from a latency graph or an invoice.

## Goal

One declarative policy per deployment that says what the customer *requires*, what they *prefer*,
and what they will *accept* when preferences cannot be met, evaluated by one scheduler across all
capacity pools (MCM), with the decision and its reasons visible to the customer.

Non-goals: per-request routing (that is the router's job), pricing changes, new regions.

## Policy model

```yaml
placement:
  # Hard constraints: never violated. A deployment that cannot satisfy them stays Pending.
  require:
    regions: [us-east, us-west]          # data residency
    compliance: [soc2, hipaa]            # pool attributes
    clouds_excluded: [cloud-x]           # contractual
    accelerators: [H100, H200]
  # Soft preferences, in order. The scheduler satisfies as many as it can, highest first.
  prefer:
    - { region: us-east }                # latency to the customer's app
    - { cloud: cloud-a }                 # committed-use discount
    - { accelerator: H100 }
  # What happens when preferences cannot all be met.
  fallback:
    mode: spill                          # spill | wait | fail
    max_cost_multiplier: 1.3             # spill only to pools <= 1.3x the preferred price
    max_wait_s: 120                      # wait mode: how long before escalating to spill/fail
  # Minimum spread for resilience; interacts with min_replicas.
  spread:
    min_pools: 2
```

Semantics worth stating:
- `require` is a filter, `prefer` is a sort, `fallback` is what to do when the sorted list's head is
  full. Writing it this way makes the policy explainable: every placement decision is "filtered by
  X, sorted by Y, chose Z because W was full".
- Preferences are ordered, not weighted. Weights invite "0.7 region vs 0.3 cost" conversations
  that nobody can reason about at 3 am.
- Fallback applies per replica, not per deployment: scaling from 4 to 8 replicas during a burst may
  place the last two in a spill pool. The customer sees that in the observability surface.
- A policy change never moves running replicas by itself. It changes where the *next* replica goes.
  Migration is an explicit action (drain old, place new), reusing the rolling-deploy path.

## Customer UX

- Config: the YAML above in the deployment config and in the UI as three sections (Require /
  Prefer / When full). Defaults: require nothing, prefer the customer's home region, spill up to 1.3x.
- Validation at save time: "No pool currently satisfies `require`" is an error with the nearest
  satisfiable alternative suggested. "`prefer` can be satisfied in 0 pools right now" is a warning.
- Dry run: `baseten placement plan` prints, for the current replica count, where each replica would
  go and why.

## Observability surface

Per deployment, per replica: `pool`, `region`, `cloud`, `accelerator`, `placement_reason`
(`preferred`, `spilled:<which preference was violated>`, `pending:<which require failed>`),
`price_multiplier`. Aggregated: % replica-hours in preferred placement, spill cost delta, pending
time. Alerts: replicas spilled for more than N minutes; deployment pending for more than M.

This is the same observability shape the router here exposes per backend (`/metrics`: stage,
healthy, inflight, errors, latency), lifted one level up from "which replica" to "which pool".

## Scheduler changes

- Pools carry attributes (region, cloud, compliance tags, accelerator, price). Already true for MCM.
- Placement = filter(require) → sort(prefer) → first pool with free capacity, else fallback.
- Scale-down drains spilled replicas first (the inverse of placement order), so the deployment
  converges back to preferred capacity when the burst ends. This matches the simulator's
  "drain the newest first" heuristic in `autosim`, generalised.

## Rollout

Shadow mode first: compute the policy's decision for every placement for two weeks, log it
alongside the actual decision, diff. Then opt-in per deployment. Then default.

## Open questions

- [ ] Should `max_cost_multiplier` be a customer knob or a plan attribute?
- [ ] Multi-region active/active for one deployment: same policy with `spread.min_regions`, or a
      separate feature?
- [ ] How does a `require.compliance` change interact with model weights already cached in a
      non-compliant pool's BDN tier?

## Numbers from my experiments to cite

- [ ] Cold-start cost of a wrong placement: cs=30 s vs cs=180 s moved SLO attainment from __ % to __ %
      on the bursty trace (`results/autoscaler.json`).
- [ ] Price of preference: H100 at $6.50/h vs L4 at $0.85/h gave $/1M tokens of __ vs __ at the
      customer's p99 SLO (`results/frontier.png`).
- [ ] Spill latency: __ ms added p50 when a replica lands one region over (measure with the router's
      `/admin/slow` on one backend).
