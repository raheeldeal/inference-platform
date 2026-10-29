"""
mockllm: a simulated vLLM-style LLM server with an OpenAI-compatible API.

Why simulate? So you can run the whole platform (router, load tests, chaos,
autoscaling) on a laptop with *realistic* latency physics, then swap in real
backends (Baseten / Modal / a local mlx-lm server) with zero client changes.

What it models (each maps to a real vLLM concept):

  * Continuous batching ........ one engine loop; every "step" emits one token for
                                  every running sequence. New requests join the
                                  batch between steps instead of waiting for the
                                  batch to finish (that was static batching).
  * Decode is memory-bound ..... a step gets only slightly slower as the batch grows
                                  (step_ms * (1 + slope * batch)), so aggregate
                                  tokens/sec rises with concurrency while per-request
                                  tokens/sec falls. That curve IS the cost/perf frontier.
  * Prefill is compute-bound ... admitting a request costs prompt_tokens * prefill_us.
                                  Without chunked prefill, that cost stalls every other
                                  sequence's decode for one step (visible as a
                                  jitter in their token cadence). With chunked prefill
                                  it's spread over several steps.
  * KV cache capacity .......... --kv-tokens caps total (prompt + generated) tokens
                                  in flight; --max-num-seqs caps batch size. Hitting
                                  either puts requests in the WAITING queue, which is
                                  where p99 latency goes to die.
  * Cold start ................. --cold-start-s seconds of "loading weights" during
                                  which /health returns 503 and requests wait.
  * Fault injection ............ /admin/fail, /admin/slow, /admin/recover, /admin/kill
                                  so the router can be tested under failure.
  * Pricing .................... --price-per-hour is reported on /info so the load
                                  tester can compute $ per 1M tokens.

Run:  uv run mockllm --name a --port 8001 --preset h100-8b
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import typer
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ----------------------------------------------------------------------------
# Presets: rough, order-of-magnitude numbers for an 8B model in bf16.
# Tune them against the real numbers you measure on Baseten/Modal on Day 1.
# ----------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    # H100: ~80-100 tok/s single-stream for 8B bf16, prefill ~10k tok/s, big KV cache.
    "h100-8b": dict(step_ms=10.0, batch_slope=0.015, prefill_us=90, max_num_seqs=256,
                    kv_tokens=200_000, price_per_hour=3.5),
    # A10G / L4-class: ~35 tok/s single-stream, far less KV headroom.
    "a10g-8b": dict(step_ms=28.0, batch_slope=0.03, prefill_us=350, max_num_seqs=64,
                    kv_tokens=40_000, price_per_hour=1.0),
    # A "quantized" variant: int4/AWQ weights -> faster decode (less memory
    # traffic) and roughly 3x more KV room, at some quality cost.
    "a10g-8b-awq": dict(step_ms=18.0, batch_slope=0.03, prefill_us=380, max_num_seqs=128,
                        kv_tokens=110_000, price_per_hour=1.0),
    # Laptop-ish numbers so sweeps finish fast when you just want to test the router.
    "fast": dict(step_ms=3.0, batch_slope=0.01, prefill_us=10, max_num_seqs=256,
                 kv_tokens=1_000_000, price_per_hour=1.0),
}

WORDS = ("the model router batches tokens into steps while the scheduler admits "
         "sequences from the waiting queue kv cache pages are allocated per block "
         "latency throughput tradeoff canary shadow drain failover cold start").split()


@dataclass
class Seq:
    id: str
    prompt_tokens: int
    max_tokens: int
    created: float = field(default_factory=time.perf_counter)
    admitted: float | None = None
    prefill_remaining: int = 0          # prompt tokens still to prefill (chunked)
    generated: int = 0
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    done: bool = False

    @property
    def tokens_in_flight(self) -> int:
        return self.prompt_tokens + self.generated


class Engine:
    """One continuous-batching engine loop. All timing physics live here."""

    def __init__(self, name: str, cfg: dict[str, Any], cold_start_s: float,
                 chunked_prefill: bool, chunk_size: int = 512):
        self.name = name
        self.cfg = cfg
        self.cold_start_s = cold_start_s
        self.chunked_prefill = chunked_prefill
        self.chunk_size = chunk_size
        self.waiting: list[Seq] = []
        self.running: list[Seq] = []
        self.ready = asyncio.Event()           # set when weights are "loaded"
        self.wake = asyncio.Event()            # set when a new request arrives
        self.started_at = time.time()
        # fault injection state
        self.fail_mode: str | None = None      # None | "500" | "timeout" | "reset"
        self.extra_latency_ms: float = 0.0
        # counters
        self.stats = dict(requests=0, completed=0, errors=0, tokens_out=0,
                          tokens_in=0, steps=0, preempt_waits=0,
                          max_batch_seen=0, max_waiting_seen=0)
        self.step_hist: list[float] = []

    # ---------------- lifecycle ----------------
    async def start(self):
        asyncio.create_task(self._boot())
        asyncio.create_task(self._loop())

    async def _boot(self):
        # "Loading weights": this is what Baseten's BDN / Modal's volume caching /
        # vLLM's safetensors mmap work is all trying to shrink.
        await asyncio.sleep(self.cold_start_s)
        self.ready.set()

    # ---------------- admission ----------------
    def _kv_used(self) -> int:
        return sum(s.tokens_in_flight for s in self.running)

    def _try_admit(self):
        max_seqs = self.cfg["max_num_seqs"]
        kv_cap = self.cfg["kv_tokens"]
        while self.waiting and len(self.running) < max_seqs:
            s = self.waiting[0]
            # Reserve KV for prompt + worst-case output, like vLLM's block manager
            # must (it can preempt if it guessed wrong; we just don't admit).
            if self._kv_used() + s.prompt_tokens + s.max_tokens > kv_cap:
                self.stats["preempt_waits"] += 1
                break
            self.waiting.pop(0)
            s.admitted = time.perf_counter()
            s.prefill_remaining = s.prompt_tokens
            self.running.append(s)
        self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(self.running))
        self.stats["max_waiting_seen"] = max(self.stats["max_waiting_seen"], len(self.waiting))

    # ---------------- the engine loop ----------------
    async def _loop(self):
        await self.ready.wait()
        c = self.cfg
        while True:
            self._try_admit()
            if not self.running:
                self.wake.clear()
                await self.wake.wait()
                continue

            # 1) prefill work this step
            prefill_tokens = 0
            for s in self.running:
                if s.prefill_remaining > 0:
                    chunk = min(s.prefill_remaining, self.chunk_size) if self.chunked_prefill else s.prefill_remaining
                    s.prefill_remaining -= chunk
                    prefill_tokens += chunk

            # 2) decode work: one token for every sequence past prefill
            batch = len(self.running)
            step_ms = c["step_ms"] * (1 + c["batch_slope"] * batch)
            step_ms += prefill_tokens * c["prefill_us"] / 1000.0
            step_ms += self.extra_latency_ms
            step_ms *= random.uniform(0.95, 1.05)  # jitter
            await asyncio.sleep(step_ms / 1000.0)
            self.stats["steps"] += 1
            if len(self.step_hist) < 5000:
                self.step_hist.append(step_ms)

            finished = []
            for s in self.running:
                if s.prefill_remaining > 0:
                    continue  # still prefilling (chunked) — no token yet
                s.generated += 1
                self.stats["tokens_out"] += 1
                s.out.put_nowait(random.choice(WORDS) + " ")
                if s.generated >= s.max_tokens:
                    s.done = True
                    s.out.put_nowait(None)
                    finished.append(s)
            for s in finished:
                self.running.remove(s)
                self.stats["completed"] += 1

    # ---------------- request entry ----------------
    def submit(self, prompt_tokens: int, max_tokens: int) -> Seq:
        s = Seq(id=uuid.uuid4().hex[:12], prompt_tokens=prompt_tokens, max_tokens=max_tokens)
        self.stats["requests"] += 1
        self.stats["tokens_in"] += prompt_tokens
        self.waiting.append(s)
        self.wake.set()
        return s

    def snapshot(self) -> dict[str, Any]:
        hist = self.step_hist
        return dict(
            name=self.name, ready=self.ready.is_set(), uptime_s=round(time.time() - self.started_at, 1),
            running=len(self.running), waiting=len(self.waiting),
            kv_used=self._kv_used(), kv_cap=self.cfg["kv_tokens"],
            kv_util=round(self._kv_used() / self.cfg["kv_tokens"], 3),
            fail_mode=self.fail_mode, extra_latency_ms=self.extra_latency_ms,
            avg_step_ms=round(sum(hist) / len(hist), 2) if hist else None,
            **self.stats,
        )


# ----------------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------------
def build_app(engine: Engine, model_name: str) -> FastAPI:
    app = FastAPI(title=f"mockllm:{engine.name}")

    @app.on_event("startup")
    async def _startup():
        await engine.start()

    @app.get("/health")
    async def health():
        if not engine.ready.is_set():
            return JSONResponse({"status": "loading"}, status_code=503)
        if engine.fail_mode:
            return JSONResponse({"status": "failing", "mode": engine.fail_mode}, status_code=503)
        return {"status": "ok", "name": engine.name}

    @app.get("/info")
    async def info():
        return {"name": engine.name, "model": model_name, "cfg": engine.cfg,
                "cold_start_s": engine.cold_start_s, "chunked_prefill": engine.chunked_prefill}

    @app.get("/metrics")
    async def metrics():
        return engine.snapshot()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    # ---- fault injection ----
    @app.post("/admin/fail")
    async def admin_fail(mode: str = "500"):
        engine.fail_mode = mode
        return {"fail_mode": mode}

    @app.post("/admin/slow")
    async def admin_slow(ms: float = 200):
        engine.extra_latency_ms = ms
        return {"extra_latency_ms": ms}

    @app.post("/admin/recover")
    async def admin_recover():
        engine.fail_mode = None
        engine.extra_latency_ms = 0
        return {"ok": True}

    @app.post("/admin/kill")
    async def admin_kill():
        # Hard-kill the process: simulates a node dying / OOM / spot preemption.
        asyncio.get_event_loop().call_later(0.05, lambda: os._exit(1))
        return {"bye": True}

    # ---- OpenAI-compatible chat completions ----
    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        if engine.fail_mode == "500":
            engine.stats["errors"] += 1
            return JSONResponse({"error": {"message": "injected failure"}}, status_code=500)
        if engine.fail_mode == "timeout":
            await asyncio.sleep(3600)
        if engine.fail_mode == "reset":
            # drop the connection abruptly
            engine.stats["errors"] += 1
            raise RuntimeError("injected connection reset")
        if not engine.ready.is_set():
            # Real servers either 503 here or (worse) accept and hang. We wait,
            # so cold start shows up as a huge TTFT, which is what users feel.
            await engine.ready.wait()

        msgs = body.get("messages", [])
        prompt_text = " ".join(m.get("content", "") for m in msgs if isinstance(m.get("content"), str))
        prompt_tokens = max(1, int(len(prompt_text.split()) * 1.3))  # crude tokenizer
        max_tokens = int(body.get("max_tokens") or 128)
        stream = bool(body.get("stream", False))
        seq = engine.submit(prompt_tokens, max_tokens)
        rid = f"chatcmpl-{seq.id}"
        created = int(time.time())

        async def gen_chunks():
            first = True
            while True:
                tok = await seq.out.get()
                if tok is None:
                    break
                chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": ({"role": "assistant", "content": tok} if first else {"content": tok}),
                                      "finish_reason": None}]}
                first = False
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": seq.prompt_tokens, "completion_tokens": seq.generated,
                               "total_tokens": seq.prompt_tokens + seq.generated}}
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(gen_chunks(), media_type="text/event-stream",
                                     headers={"x-backend": engine.name})
        text = []
        while True:
            tok = await seq.out.get()
            if tok is None:
                break
            text.append(tok)
        return JSONResponse({
            "id": rid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": seq.prompt_tokens, "completion_tokens": seq.generated,
                      "total_tokens": seq.prompt_tokens + seq.generated},
        }, headers={"x-backend": engine.name})

    return app


cli = typer.Typer(add_completion=False, help=__doc__)


@cli.command()
def serve(
    name: str = typer.Option("mock", help="Backend name, returned in x-backend header"),
    port: int = typer.Option(8001),
    host: str = typer.Option("127.0.0.1"),
    preset: str = typer.Option("h100-8b", help=f"One of {list(PRESETS)}"),
    model: str = typer.Option("mock/Qwen2.5-7B-Instruct"),
    cold_start_s: float = typer.Option(0.0, help="Seconds spent 'loading weights' before ready"),
    chunked_prefill: bool = typer.Option(True, help="Spread prefill across steps (vLLM --enable-chunked-prefill)"),
    max_num_seqs: int | None = typer.Option(None, help="Override preset batch cap"),
    kv_tokens: int | None = typer.Option(None, help="Override preset KV-cache token capacity"),
    step_ms: float | None = typer.Option(None, help="Override base decode step time"),
    price_per_hour: float | None = typer.Option(None, help="Override $/hour for cost math"),
):
    cfg = dict(PRESETS[preset])
    for k, v in dict(max_num_seqs=max_num_seqs, kv_tokens=kv_tokens, step_ms=step_ms,
                     price_per_hour=price_per_hour).items():
        if v is not None:
            cfg[k] = v
    engine = Engine(name, cfg, cold_start_s, chunked_prefill)
    app = build_app(engine, model)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    cli()


if __name__ == "__main__":
    main()
