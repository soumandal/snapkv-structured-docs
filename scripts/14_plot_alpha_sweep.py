"""Plot α-sweep curves from results/plan2/ac_combined.csv.

One figure with 2x3 panels: rows = ctx (8k, 16k), cols = (indented JSON, compact JSON, XML).
Each panel: x = budget, y = accuracy, one line per α + SnapKV baseline.
Wikitable_long is plotted as a separate single-panel figure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = Path("results/plan2/ac_combined.csv")
OUT_DIR = Path("results/plan2")

ALPHA_MAP = {
    "snapkv-only":            ("snapkv", None),
    "snapkv-no-key":          ("rc-max", 0.00),
    "snapkv-no-key-soft02":   ("rc-max", 0.02),
    "snapkv-no-key-soft":     ("rc-max", 0.05),
    "snapkv-no-key-soft10":   ("rc-max", 0.10),
    "snapkv-no-key-soft20":   ("rc-max", 0.20),
    "snapkv-no-key-nomax":         ("rc-mean", 0.00),
    "snapkv-no-key-nomax-soft02":  ("rc-mean", 0.02),
    "snapkv-no-key-nomax-soft":    ("rc-mean", 0.05),
    "snapkv-no-key-nomax-soft10":  ("rc-mean", 0.10),
    "snapkv-no-key-nomax-soft20":  ("rc-mean", 0.20),
}

ALPHA_VALUES = [0.00, 0.02, 0.05, 0.10, 0.20]
ALPHA_COLORS = {
    0.00: "#1f77b4",
    0.02: "#2ca02c",
    0.05: "#ff7f0e",
    0.10: "#d62728",
    0.20: "#9467bd",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    unknown = sorted(set(df["config"]) - set(ALPHA_MAP))
    if unknown:
        print(f"skipping {len(unknown)} non-α-sweep configs: {unknown}")
        df = df[df["config"].isin(ALPHA_MAP)].copy()
    df["family"] = df["config"].map(lambda c: ALPHA_MAP[c][0])
    df["alpha"]  = df["config"].map(lambda c: ALPHA_MAP[c][1])
    return df


def acc_table(df: pd.DataFrame, source: str, ctx: int) -> pd.DataFrame:
    """Return columns: budget, snapkv, α=0.00 ... α=0.20."""
    sub = df[(df["source"] == source) & (df["ctx_tokens"] == ctx)]
    rcmax = sub[sub["family"] == "rc-max"]
    snap  = sub[sub["family"] == "snapkv"]
    if rcmax.empty:
        return pd.DataFrame()
    pivot = (rcmax.groupby(["budget_frac", "alpha"])["correct"].mean()
                  .unstack("alpha"))
    snap_curve = snap.groupby("budget_frac")["correct"].mean()
    pivot.insert(0, "snapkv", snap_curve)
    return pivot


def plot_panel(ax, tab: pd.DataFrame, title: str):
    if tab.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.set_xticks([]); ax.set_yticks([])
        return
    budgets = tab.index.values
    if "snapkv" in tab.columns and tab["snapkv"].notna().any():
        ax.plot(budgets, tab["snapkv"].values, "k--", marker="s",
                label="SnapKV", linewidth=1.2, markersize=4)
    for a in ALPHA_VALUES:
        if a not in tab.columns:
            continue
        y = tab[a].values
        if np.all(np.isnan(y)):
            continue
        ax.plot(budgets, y, marker="o", color=ALPHA_COLORS[a],
                label=f"α={a:.2f}", linewidth=1.4, markersize=4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("KV budget (frac)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def main():
    df = load()

    # Main figure: synthetic corpora × ctx
    panels = [
        ("synthetic_json",          8000,  "indented JSON  ctx=8k"),
        ("synthetic_json_compact",  8000,  "compact JSON   ctx=8k"),
        ("synthetic_xml",           8000,  "XML            ctx=8k"),
        ("synthetic_json",          16000, "indented JSON  ctx=16k"),
        ("synthetic_json_compact",  16000, "compact JSON   ctx=16k"),
        ("synthetic_xml",           16000, "XML            ctx=16k"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for ax, (src, ctx, title) in zip(axes.flat, panels):
        plot_panel(ax, acc_table(df, src, ctx), title)
    # single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.suptitle("α-sweep: role-conditional SnapKV (rc-max family) vs SnapKV baseline",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_main = OUT_DIR / "fig_alpha_sweep_synthetic.png"
    fig.savefig(out_main, dpi=130, bbox_inches="tight")
    print(f"wrote {out_main}")

    # Side figure: wikitable_long (only ctx 8k)
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    plot_panel(ax2, acc_table(df, "wikitable_long", 8000),
               "wikitable_long  ctx=8k")
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    fig2.tight_layout()
    out_wt = OUT_DIR / "fig_alpha_sweep_wikitable.png"
    fig2.savefig(out_wt, dpi=130, bbox_inches="tight")
    print(f"wrote {out_wt}")


if __name__ == "__main__":
    main()
