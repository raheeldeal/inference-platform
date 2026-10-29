"""
Modal deployment of Qwen2.5-7B-Instruct served by vLLM, using Modal's `@app.server` API.
Adapted from modal.com/docs/examples/vllm_inference (verified 2026-10-29).

    uv add modal && uv run modal setup          # one-time browser auth
    uv run modal serve deploy/modal/vllm_qwen.py   # dev URL, hot reload
    uv run modal deploy deploy/modal/vllm_qwen.py  # stable URL

Then:
    export MODAL_URL=https://<workspace>--qwen25-vllm-server.modal.run
    uv run loadtest --url $MODAL_URL --model Qwen/Qwen2.5-7B-Instruct --price-per-hour 3.95 \
        --concurrency 1,2,4,8,16,32,64 --label modal-h100 --out results/modal-h100.json

Cold start: the first request after `scaledown_window` of idle pays container boot +
weight load from the Volume (~ tens of seconds for 7B). Measure it:
    uv run loadtest --url $MODAL_URL --cold-start

Knobs that map 1:1 onto the autoscaler simulator:
    min_containers      -> --min-replicas
    max_containers      -> max_replicas
    target_concurrency  -> --target-concurrency
    scaledown_window    -> --scale-down-delay
"""

import modal

MINUTES = 60
VLLM_PORT = 8000
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "main"          # pin a commit hash in prod
N_GPU = 1
GPU = f"H100!:{N_GPU}"           # "H100" may auto-upgrade to H200; "!" pins it. Cheap point: "L4" or "A10"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")   # version pinned by Modal's example; 0.32 is current, bump and test
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "1"})
)

# Volumes persist weights + vLLM's compile cache across containers: that's the
# difference between a 3-minute cold start (pull 15 GB from HF) and a ~30s one.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen25-vllm")


@app.server(
    image=vllm_image,
    gpu=GPU,
    # secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN; needed for gated models (Llama)
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    port=VLLM_PORT,
    startup_timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,   # idle time before a container stops (scale-to-zero)
    min_containers=0,                # 1 = always warm, no cold start, pay 24/7
    max_containers=2,
    target_concurrency=32,           # soft autoscaling target per replica
    max_concurrency=64,              # hard cap; excess gets 429 (compare: Baseten queues)
    unauthenticated=True,            # otherwise send Modal-Key / Modal-Secret headers
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", MODEL_NAME, "llm",
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--tensor-parallel-size", str(N_GPU),
            "--max-model-len", "8192",
            "--max-num-seqs", "128",
            "--gpu-memory-utilization", "0.90",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
        ]
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def main():
    import json
    import urllib.request

    url = Server.get_url()
    body = json.dumps({"model": MODEL_NAME, "messages": [{"role": "user", "content": "Say hi in five words."}],
                       "max_tokens": 32}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    print(url)
    print(urllib.request.urlopen(req).read().decode())
