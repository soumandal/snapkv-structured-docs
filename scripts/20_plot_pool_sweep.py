#!/usr/bin/env python3
"""Render fig_snapkv_ablation.png — the C7 pool-kernel sweep.

Two panels (snapkv-only = SnapKV baseline; snapkv-no-key = combined method).
Each: EM vs budget, one solid line per swept maxpool kernel k in {3,5,11,15},
the locked k=7 headline overlaid as a dotted black reference (SnapKV's default
pool), and the maxpool-OFF (`-nomax`) control as a flat dashed grey line.

The C7 story the figure tells: (1) turning the pool off (dashed) collapses
low-budget accuracy — maxpool is the load-bearing op; (2) among pooled runs,
EM rises monotonically with kernel size at the budget-binding regime, so
SnapKV's default k=7 is conservative on dense-structured retrieval.

Source: results/plan2/pool_sweep_k{N}_ctx16k.csv (Llama, synthetic_json,
ctx 16k, n=50). Driver written 2026-06-02.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results/plan2")
OUT = RESULTS / "fig_snapkv_ablation.png"
BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.5]
KERNELS = [3, 5, 11, 15]
KCOLORS = {3: "#9ecae1", 5: "#4292c6", 11: "#08519c", 15: "#08306b"}

# Locked k=7 headline (Llama SDPA, synthetic_json ctx16k) — not in the sweep.
HEADLINE_K7 = {
    "snapkv-only":   [0.20, 0.34, 0.70, 0.80, 0.88],
    "snapkv-no-key": [0.56, 0.76, 0.86, 0.98, 0.96],
}
PANELS = [
    ("snapkv-only", "SnapKV baseline  (windowed mass + maxpool=k)"),
    ("snapkv-no-key", "Combined no-key  (role allocator + maxpool=k)"),
]


def load() -> pd.DataFrame:
    frames = []
    for k in KERNELS:
        p = RESULTS / f"pool_sweep_k{k}_ctx16k.csv"
        df = pd.read_csv(p)
        df["pool_kernel"] = k  # trust filename; column also carries it
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def curve(df, cfg, k):
    sub = df[(df["config"] == cfg) & (df["pool_kernel"] == k)]
    g = sub.groupby("budget_frac")["correct"].mean()
    return [g.get(round(b, 3), float("nan")) for b in BUDGETS]


def main():
    df = load()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (cfg, title) in zip(axes, PANELS):
        for k in KERNELS:
            ax.plot(BUDGETS, curve(df, cfg, k), marker="o", ms=4,
                    color=KCOLORS[k], label=f"k={k}")
        ax.plot(BUDGETS, HEADLINE_K7[cfg], ":", color="black", lw=1.6,
                marker="s", ms=4, label="k=7 (SnapKV default)")
        nomax = curve(df, "snapkv-no-key-nomax", KERNELS[0])
        ax.plot(BUDGETS, nomax, "--", color="grey", lw=1.4,
                label="maxpool OFF (-nomax)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("KV budget fraction")
        ax.set_xticks(BUDGETS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("C7 pool-kernel ablation — Llama-3.1-8B, synthetic_json ctx 16k (n=50)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
