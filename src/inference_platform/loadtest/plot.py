"""
Plot the cost/performance frontier from one or more loadtest JSON files.

  uv run python -m inference_platform.loadtest.plot results/a.json results/b.json --out results/frontier.png

Panels:
  1. throughput (tok/s) vs p50 and p99 latency  — THE frontier. Read it right-to-left:
     how much latency do you pay for each extra unit of throughput?
  2. TTFT p50/p99 vs concurrency — when does queueing start?
  3. $ per 1M output tokens vs concurrency — the business view of panel 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inference_platform.viz import SERIES, TEXT2, plt, style


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    out = Path("results/frontier.png")
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out = Path(argv[i + 1]); i += 2
        else:
            files.append(Path(argv[i])); i += 1
    if not files:
        print(__doc__); return

    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, f in enumerate(files):
        d = json.loads(f.read_text())
        rows = [r for r in d["levels"] if r["throughput_tps"]]
        label = rows[0].get("label") or f.stem
        color = SERIES[k % len(SERIES)]
        conc = [r["concurrency"] for r in rows]
        tps = [r["throughput_tps"] for r in rows]

        ax = axes[0]
        ax.plot([r["lat_p50"] for r in rows], tps, "-o", color=color, label=f"{label} p50")
        ax.plot([r["lat_p99"] for r in rows], tps, "--s", color=color, alpha=0.6, label=f"{label} p99")
        for r in rows:
            ax.annotate(f"c={r['concurrency']}", (r["lat_p50"], r["throughput_tps"]), fontsize=7,
                        color=TEXT2, xytext=(4, 4), textcoords="offset points")

        ax = axes[1]
        ax.plot(conc, [r["ttft_p50"] for r in rows], "-o", color=color, label=f"{label} p50")
        ax.plot(conc, [r["ttft_p99"] for r in rows], "--s", color=color, alpha=0.6, label=f"{label} p99")

        ax = axes[2]
        ax.plot(conc, [r["usd_per_1m_tokens"] for r in rows], "-o", color=color, label=label)

    axes[0].set(title="Cost/performance frontier", xlabel="end-to-end latency (ms)", ylabel="aggregate tokens / s")
    axes[1].set(title="Time to first token", xlabel="concurrency", ylabel="TTFT (ms)", xscale="log")
    axes[2].set(title="$ per 1M output tokens", xlabel="concurrency", ylabel="USD", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
