#!/usr/bin/env python3
"""Cross-context scaling figure (§17 P2): does the combined method's low-budget
gain over SnapKV-only keep climbing with context, or saturate?

Reads synthetic_json Llama EM for snapkv-only vs snapkv-no-key (α=0) at
ctx 8k / 16k (from ac_combined.csv) and ctx 32k (from ac_combined_ctx32k.csv,
2026-06-02). Emits results/plan2/fig_ctx_scaling.png:
  Panel A — B=5% EM vs context length (combined vs SnapKV-only); the combined
            curve is flat (method ceiling) while SnapKV-only climbs, so the gap
            (annotated) saturates.
  Panel B — Δ(combined − SnapKV) vs budget, one line per context length.
Analyzer-only, no GPU.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

CTX_SRC = {
    8000: "results/plan2/ac_combined.csv",
    16000: "results/plan2/ac_combined.csv",
    32000: "results/plan2/ac_combined_ctx32k.csv",
}
CONFIGS = ["snapkv-only", "snapkv-no-key"]
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]


def em_table(ctx: int) -> pl.DataFrame:
    df = pl.read_csv(CTX_SRC[ctx]).filter(
        (pl.col("ctx_tokens") == ctx)
        & (pl.col("source") == "synthetic_json")
        & (pl.col("config").is_in(CONFIGS))
    )
    return (
        df.group_by(["config", "budget_frac"])
        .agg(pl.col("correct").mean().alias("EM"))
        .sort(["config", "budget_frac"])
    )


def em(tbl: pl.DataFrame, cfg: str, b: float) -> float:
    r = tbl.filter((pl.col("config") == cfg) & (pl.col("budget_frac") == b))
    return float(r["EM"][0])


def main() -> None:
    ctxs = [8000, 16000, 32000]
    tables = {c: em_table(c) for c in ctxs}
    xs = [c / 1000 for c in ctxs]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A — B=5% EM vs context
    only5 = [em(tables[c], "snapkv-only", 0.05) for c in ctxs]
    comb5 = [em(tables[c], "snapkv-no-key", 0.05) for c in ctxs]
    axA.plot(xs, comb5, "o-", color="#1b7837", lw=2.2, ms=8, label="combined (snapkv-no-key, α=0)")
    axA.plot(xs, only5, "s--", color="#762a83", lw=2.2, ms=8, label="SnapKV-only")
    for x, c, o in zip(xs, comb5, only5):
        axA.annotate(f"+{c - o:.2f}", (x, (c + o) / 2), ha="center", va="center",
                     fontsize=10, fontweight="bold", color="#444")
    axA.set_xticks(xs); axA.set_xticklabels([f"{c}k" for c in [8, 16, 32]])
    axA.set_xlabel("context length (tokens)")
    axA.set_ylabel("EM @ B=5%")
    axA.set_title("Low-budget gain saturates, doesn't climb\n(combined flat ~0.56; baseline climbs at fixed fraction)")
    axA.set_ylim(0, 1.0); axA.grid(alpha=0.3); axA.legend(loc="lower right", fontsize=9)

    # Panel B — Δ vs budget, one line per context
    colors = {8000: "#a6611a", 16000: "#1b7837", 32000: "#2166ac"}
    for c in ctxs:
        deltas = [em(tables[c], "snapkv-no-key", b) - em(tables[c], "snapkv-only", b) for b in BUDGETS]
        axB.plot([b * 100 for b in BUDGETS], deltas, "o-", color=colors[c], lw=2, ms=6,
                 label=f"ctx {c // 1000}k")
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xlabel("budget (% of context)")
    axB.set_ylabel("Δ EM (combined − SnapKV-only)")
    axB.set_title("Combined-method edge over SnapKV vs budget\n(largest at mid-context, binding budgets)")
    axB.grid(alpha=0.3); axB.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = Path("results/plan2/fig_ctx_scaling.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    print("B=5% combined:", dict(zip([8, 16, 32], [round(v, 3) for v in comb5])))
    print("B=5% snapkv:  ", dict(zip([8, 16, 32], [round(v, 3) for v in only5])))


if __name__ == "__main__":
    main()
