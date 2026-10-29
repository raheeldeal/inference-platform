#!/usr/bin/env bash
# Day 1 frontier: sweep concurrency against each backend and plot. Backends must be up (scripts/up.sh).
set -eu
C=${CONCURRENCY:-1,2,4,8,16,32,64,128}
uv run loadtest --url http://127.0.0.1:8001 --concurrency $C --max-tokens 128 --label h100-sim     --out results/sim-h100.json
uv run loadtest --url http://127.0.0.1:8002 --concurrency $C --max-tokens 128 --label a10g-sim     --out results/sim-a10g.json
uv run loadtest --url http://127.0.0.1:8003 --concurrency $C --max-tokens 128 --label a10g-awq-sim --out results/sim-a10g-awq.json
uv run python -m inference_platform.loadtest.plot results/sim-h100.json results/sim-a10g.json results/sim-a10g-awq.json --out results/frontier.png
