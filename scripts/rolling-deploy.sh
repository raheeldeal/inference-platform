#!/usr/bin/env bash
# Rolling deploy drill. Two paths, same load:
#   RIGHT: drain a10g (no new traffic, wait for in-flight=0) -> kill it -> start new build -> add to router
#   WRONG: DELETE a10g from the router with requests in flight -> those requests error
# Expect: RIGHT path 0 errors. WRONG path errors == in-flight on a10g at that moment.
set -u
cd "$(dirname "$0")/.."
ROUTER=http://127.0.0.1:8000
j() { python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d["totals"]))'; }
bk() { curl -s $ROUTER/admin/backends | python3 -c 'import sys,json; [print("   ", b["name"], "healthy=%s draining=%s removed=%s inflight=%s errors=%s" % (b["healthy"], b["draining"], b["removed"], b["inflight"], b["errors"])) for b in json.load(sys.stdin)]'; }

# make the canary carry real load so draining is non-trivial
curl -s -X POST "$ROUTER/admin/backends/a10g/weight?weight=50" > /dev/null
echo "== load: c=24 through the router"
uv run loadtest --url $ROUTER --concurrency 24 --requests 600 --max-tokens 64 --out results/rolling.json > results/rolling-load.log 2>&1 &
LT=$!; sleep 4
echo "== RIGHT: drain a10g"; bk
curl -s -X POST "$ROUTER/admin/backends/a10g/drain?timeout_s=30"; echo
bk
echo "== a10g has no in-flight: safe to kill it"; curl -s -X POST http://127.0.0.1:8002/admin/kill > /dev/null
sleep 1
echo "== start the 'new build' on the same port and re-add"
nohup uv run mockllm --name a10g-v2 --port 8002 --preset a10g-8b-awq --cold-start-s 3 > results/mock-a10g-v2.log 2>&1 &
sleep 2; curl -s -X POST $ROUTER/admin/backends -H 'content-type: application/json' -d '{"name":"a10g-v2","url":"http://127.0.0.1:8002","weight":50,"stage":"canary"}' > /dev/null
sleep 6; echo "== after re-add (health gates traffic until ready):"; bk; curl -s $ROUTER/metrics | j
echo
echo "== WRONG: remove a10g-v2 immediately while it has in-flight requests"
curl -s -X DELETE "$ROUTER/admin/backends/a10g-v2" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("    removed with inflight =", d["inflight"])'
curl -s -X POST http://127.0.0.1:8002/admin/kill > /dev/null
sleep 3; curl -s $ROUTER/metrics | j
wait $LT
echo "== final:"; curl -s $ROUTER/metrics | j
grep -E "^\s+24 " results/rolling-load.log
echo "== events:"; curl -s $ROUTER/metrics | python3 -c 'import sys,json; [print("  ",e) for e in json.load(sys.stdin)["events"][-12:]]'
