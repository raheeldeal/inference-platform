"""
loadtest: measure an OpenAI-compatible LLM endpoint the way an inference platform does.

Per request we record:
  ttft_ms     time to first token (streaming: first content chunk)
  latency_ms  end-to-end
  out_tokens  completion tokens (from usage if present, else chunk count)
  tps         per-request decode rate = (out_tokens-1) / (latency - ttft)
Per concurrency level we report p50/p95/p99 of those, aggregate throughput
(tokens/sec across all in-flight requests), error rate, and
$ per 1M output tokens = price_per_hour / (throughput_tps * 3600) * 1e6.

Sweep concurrency and plot throughput-vs-latency: that curve is the
cost/performance frontier. Where it bends is where the batch is saturating
(KV cache or max_num_seqs); past the bend you pay in p99 for little throughput.

Examples:
  uv run loadtest --url http://127.0.0.1:8001 --concurrency 1,2,4,8,16,32,64 --requests 32
  uv run loadtest --url https://model-xxx.api.baseten.co/environments/production/sync \\
      --api-key $BASETEN_API_KEY --model qwen --price-per-hour 3.5 --out results/baseten.json
  uv run loadtest --url http://127.0.0.1:8001 --cold-start      # time /health 503 -> 200
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

console = Console(width=160)

PROMPT_WORDS = ("Explain how continuous batching in an LLM inference server changes the "
                "throughput latency tradeoff compared to static batching and why the KV cache "
                "is the resource that ultimately limits concurrency on a single GPU").split()


def make_prompt(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(PROMPT_WORDS) for _ in range(n_words))


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str,
                      max_tokens: int, headers: dict, stream: bool, timeout: float) -> dict[str, Any]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": stream, "temperature": 0}
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = None
    backend = None
    try:
        if stream:
            async with client.stream("POST", f"{url}/v1/chat/completions", json=body,
                                     headers=headers, timeout=timeout) as r:
                backend = r.headers.get("x-backend")
                if r.status_code >= 400:
                    await r.aread()
                    return dict(ok=False, status=r.status_code, latency_ms=(time.perf_counter() - t0) * 1000, backend=backend)
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        # In-band error after headers were sent (e.g. upstream died
                        # mid-stream behind the router). Status was 200; it is NOT ok.
                        return dict(ok=False, status="mid-stream", latency_ms=(time.perf_counter() - t0) * 1000,
                                    backend=backend, ttft_ms=ttft)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    ch = obj.get("choices") or []
                    if ch and ch[0].get("delta", {}).get("content"):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        n_chunks += 1
        else:
            r = await client.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=timeout)
            backend = r.headers.get("x-backend")
            if r.status_code >= 400:
                return dict(ok=False, status=r.status_code, latency_ms=(time.perf_counter() - t0) * 1000, backend=backend)
            obj = r.json()
            usage = obj.get("usage")
            ttft = (time.perf_counter() - t0) * 1000  # non-streaming: TTFT == latency
            n_chunks = usage.get("completion_tokens", 0) if usage else 0
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, status=type(e).__name__, latency_ms=(time.perf_counter() - t0) * 1000, backend=backend)

    latency = (time.perf_counter() - t0) * 1000
    out_tokens = (usage or {}).get("completion_tokens") or n_chunks
    decode_ms = max(latency - (ttft or latency), 1e-6)
    tps = (out_tokens - 1) / (decode_ms / 1000) if out_tokens > 1 else 0.0
    return dict(ok=True, status=200, ttft_ms=ttft, latency_ms=latency, out_tokens=out_tokens,
                tps=tps, backend=backend)


async def run_level(url: str, model: str, concurrency: int, n_requests: int, prompt_words: int,
                    max_tokens: int, headers: dict, stream: bool, timeout: float) -> dict[str, Any]:
    """Closed-loop load: `concurrency` workers, each fires the next request when its
    previous one finishes. This is what a sweep means: offered concurrency, not RPS."""
    results: list[dict] = []
    counter = {"i": 0}
    lock = asyncio.Lock()
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(wid: int):
            while True:
                async with lock:
                    if counter["i"] >= n_requests:
                        return
                    i = counter["i"]
                    counter["i"] += 1
                prompt = make_prompt(prompt_words, seed=i)
                results.append(await one_request(client, url, model, prompt, max_tokens, headers, stream, timeout))

        # warmup so connection setup doesn't pollute TTFT
        await one_request(client, url, model, make_prompt(8, 0), 4, headers, stream, timeout)
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(w) for w in range(concurrency)))
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    total_tokens = sum(r["out_tokens"] for r in ok)

    def pct(key: str, q: float) -> float | None:
        vals = [r[key] for r in ok if r.get(key) is not None]
        return float(np.percentile(vals, q)) if vals else None

    return dict(
        concurrency=concurrency, requests=len(results), errors=len(errs),
        error_rate=len(errs) / max(len(results), 1), wall_s=wall,
        throughput_tps=total_tokens / wall if wall else 0.0,
        rps=len(ok) / wall if wall else 0.0,
        ttft_p50=pct("ttft_ms", 50), ttft_p99=pct("ttft_ms", 99),
        lat_p50=pct("latency_ms", 50), lat_p95=pct("latency_ms", 95), lat_p99=pct("latency_ms", 99),
        per_req_tps_p50=pct("tps", 50),
        error_kinds=sorted({str(r["status"]) for r in errs}),
        backends={b: sum(1 for r in results if r.get("backend") == b) for b in {r.get("backend") for r in results}},
    )


async def measure_cold_start(url: str, headers: dict, timeout_s: float = 600) -> dict[str, Any]:
    """Poll /health until 200. Run this right after `truss push` / `modal deploy` /
    starting mockllm with --cold-start-s, to measure load-weights time."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        while time.perf_counter() - t0 < timeout_s:
            try:
                r = await client.get(f"{url}/health", headers=headers, timeout=5)
                if r.status_code == 200:
                    return dict(cold_start_s=time.perf_counter() - t0, status=200)
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.25)
    return dict(cold_start_s=None, status="timeout")


cli = typer.Typer(add_completion=False, help=__doc__)


@cli.command()
def main_cmd(
    url: str = typer.Option("http://127.0.0.1:8001", help="Base URL (no /v1)"),
    model: str = typer.Option("mock", help="Model name sent in the request body"),
    api_key: str = typer.Option("", envvar="LLM_API_KEY", help="Bearer token (Baseten: 'Api-Key xxx' is also accepted)"),
    concurrency: str = typer.Option("1,2,4,8,16,32,64", help="Comma-separated sweep"),
    requests: int = typer.Option(0, help="Requests per level (default: 4x concurrency, min 16)"),
    prompt_words: int = typer.Option(120),
    max_tokens: int = typer.Option(128),
    no_stream: bool = typer.Option(False),
    timeout: float = typer.Option(120.0),
    price_per_hour: float | None = typer.Option(None, help="$/hr for the backend; if omitted, read from /info"),
    out: Path = typer.Option(Path("results/loadtest.json")),
    label: str = typer.Option("", help="Series label for plots (defaults to url host)"),
    cold_start: bool = typer.Option(False, help="Only measure /health 503->200 time, then exit"),
):
    headers = {}
    if api_key:
        headers["Authorization"] = api_key if api_key.startswith("Api-Key") else f"Bearer {api_key}"

    if cold_start:
        res = asyncio.run(measure_cold_start(url, headers))
        console.print(res)
        return

    if price_per_hour is None:
        try:
            info = httpx.get(f"{url}/info", headers=headers, timeout=5).json()
            price_per_hour = float(info["cfg"]["price_per_hour"])
        except Exception:  # noqa: BLE001
            price_per_hour = 0.0

    levels = [int(x) for x in concurrency.split(",") if x.strip()]
    rows = []
    table = Table(title=f"loadtest {url}  (prompt~{prompt_words}w, max_tokens={max_tokens}, stream={not no_stream})")
    for col in ["conc", "reqs", "err%", "tok/s (agg)", "req/s", "TTFT p50", "TTFT p99", "lat p50", "lat p99", "tok/s/req", "$/1M tok"]:
        table.add_column(col, justify="right")
    for c in levels:
        n = requests or max(16, 4 * c)
        r = asyncio.run(run_level(url, model, c, n, prompt_words, max_tokens, headers, not no_stream, timeout))
        r["price_per_hour"] = price_per_hour
        r["usd_per_1m_tokens"] = (price_per_hour / 3600 / r["throughput_tps"] * 1e6) if r["throughput_tps"] else None
        r["label"] = label or httpx.URL(url).host
        rows.append(r)
        f = lambda v, d=0: ("-" if v is None else f"{v:.{d}f}")  # noqa: E731
        table.add_row(str(c), str(n), f(100 * r["error_rate"], 1), f(r["throughput_tps"]), f(r["rps"], 2),
                      f(r["ttft_p50"]), f(r["ttft_p99"]), f(r["lat_p50"]), f(r["lat_p99"]),
                      f(r["per_req_tps_p50"], 1), f(r["usd_per_1m_tokens"], 2))
        console.print(f"[dim]c={c}: tok/s={r['throughput_tps']:.0f} p99={r['lat_p99'] and round(r['lat_p99'])}ms "
                      f"err={r['errors']} backends={r['backends']}[/dim]")
    console.print(table)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(url=url, prompt_words=prompt_words, max_tokens=max_tokens,
                                   stream=not no_stream, levels=rows), indent=2))
    console.print(f"wrote {out}")


def main():
    cli()


if __name__ == "__main__":
    main()
