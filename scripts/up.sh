#!/usr/bin/env bash
# Start three simulated backends and the router. Logs in results/.
set -eu
cd "$(dirname "$0")/.."
mkdir -p results
uv sync -q
nohup uv run mockllm --name h100     --port 8001 --preset h100-8b     > results/mock-h100.log 2>&1 &
nohup uv run mockllm --name a10g     --port 8002 --preset a10g-8b     > results/mock-a10g.log 2>&1 &
nohup uv run mockllm --name a10g-awq --port 8003 --preset a10g-8b-awq > results/mock-a10g-awq.log 2>&1 &
nohup uv run router > results/router.log 2>&1 &
sleep 3
for p in 8001 8002 8003 8000; do printf "  :%s  " $p; curl -s localhost:$p/health; echo; done
echo "router admin: curl localhost:8000/metrics | jq"
