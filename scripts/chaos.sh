#!/usr/bin/env bash
# Chaos drill: run load through the router, then break backends in increasingly
# rude ways and watch error rate + failovers. Expect:
#   1. /admin/fail 500   -> router retries on the other backend: 0 client errors, failovers > 0
#   2. /admin/kill       -> connection errors; in-flight streams on that backend die
#                            (mid_stream_errors > 0) but new requests fail over.
#   3. drain + kill      -> 0 errors: that's a rolling deploy done right.
set -u
ROUTER=${ROUTER:-http://127.0.0.1:8000}
VICTIM=${VICTIM:-http://127.0.0.1:8002}
VICTIM_NAME=${VICTIM_NAME:-a10g}
j() { python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d["totals"]))'; }

echo "== baseline"; curl -s $ROUTER/metrics | j
echo "== starting 20s of load at concurrency 16 through the router"
uv run loadtest --url $ROUTER --concurrency 16 --requests 400 --max-tokens 48 --out results/chaos.json > results/chaos-load.log 2>&1 &
LT=$!
sleep 4
echo "== (1) injecting HTTP 500s on $VICTIM_NAME"; curl -s -X POST "$VICTIM/admin/fail?mode=500" >/dev/null
sleep 4; curl -s $ROUTER/metrics | j
echo "== recovering $VICTIM_NAME"; curl -s -X POST $VICTIM/admin/recover >/dev/null
sleep 4; curl -s $ROUTER/metrics | j
echo "== (2) HARD KILL of $VICTIM_NAME mid-stream"; curl -s -X POST $VICTIM/admin/kill >/dev/null
sleep 4; curl -s $ROUTER/metrics | j
wait $LT
echo "== load finished; final:"; curl -s $ROUTER/metrics | j
grep -E "^\s+16 " results/chaos-load.log || tail -5 results/chaos-load.log
echo "== router events:"; curl -s $ROUTER/metrics | python3 -c 'import sys,json; [print("  ",e) for e in json.load(sys.stdin)["events"]]'
