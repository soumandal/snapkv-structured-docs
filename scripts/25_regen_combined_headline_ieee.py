"""Regenerate fig_combined_headline.png for IEEE double-column inclusion.

Same data and two-panel layout as scripts/18_plot_combined_headline.py, but
restyled for legibility when printed at full (double-column) width: larger
fonts throughout, thicker lines / bigger markers, and 300 dpi. Writes to
white_paper_figures/ rather than results/plan2/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("white_paper_figures/fig_combined_headline.png")
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
ORACLE = 0.88  # H2O @ budget=1.0 on synthetic_json (matches both ctxs)

COMBINED_CFG = {8000: "snapkv-no-key-soft02", 16000: "snapkv-no-key"}
COMBINED_LABEL = {8000: "Combined (α=0.02)", 16000: "Combined (α=0)"}
AC_CSV = {8000: "results/plan2/ac_combined.csv",
          16000: "results/plan2/ac_combined_llama_sdpa.csv"}


def curve_em(df: pd.DataFrame, mask) -> list:
    g = df[mask].groupby("budget_frac")["correct"].mean()
    return [float(g.get(b, float("nan"))) for b in BUDGETS]


def plot_panel(ax, ctx: int):
    e  = pd.read_csv("results/plan2/e_counterfactual.csv")
    a  = pd.read_csv("results/plan2/a_role_conditional.csv")
    ac = pd.read_csv(AC_CSV[ctx])

    h2o = curve_em(e,  (e["source"] == "synthetic_json") & (e["condition"] == "All") & (e["ctx_tokens"] == ctx))
    snap = curve_em(ac, (ac["source"] == "synthetic_json") & (ac["config"] == "snapkv-only") & (ac["ctx_tokens"] == ctx))
    comb = curve_em(ac, (ac["source"] == "synthetic_json") & (ac["config"] == COMBINED_CFG[ctx]) & (ac["ctx_tokens"] == ctx))
    a_nokey = curve_em(a, (a["source"] == "synthetic_json") & (a["policy"] == "no-key") & (a["ctx_tokens"] == ctx))

    ax.axhline(ORACLE, color="gray", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(0.5, ORACLE + 0.012, "full-budget oracle (0.88)",
            ha="right", va="bottom", fontsize=11, color="gray")

    ax.plot(BUDGETS, h2o,  color="black",   linestyle="--", marker="s",
            label="H2O",          linewidth=1.8, markersize=7, zorder=2)
    ax.plot(BUDGETS, snap, color="#1f77b4", linestyle="-",  marker="o",
            label="SnapKV-only",  linewidth=2.2, markersize=7, zorder=3)
    ax.plot(BUDGETS, a_nokey, color="#2ca02c", linestyle="-", marker="^",
            label="A no-key (role allocator)",
            linewidth=2.2, markersize=7, zorder=3)
    ax.plot(BUDGETS, comb, color="#d62728", linestyle="-",  marker="*",
            label=COMBINED_LABEL[ctx],
            linewidth=3.0, markersize=14, zorder=4)

    if ctx == 16000:
        ax.annotate(
            "over-oracle:\n0.98 > 0.88",
            xy=(0.30, 0.98), xytext=(0.33, 0.62),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
            fontsize=12, color="#d62728", fontweight="bold",
        )

    ax.set_title(f"indented JSON, ctx {ctx//1000}k", fontsize=15)
    ax.set_xlabel("KV budget (fraction)", fontsize=14)
    ax.set_xticks(BUDGETS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.03, 0.52)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    plot_panel(axes[0], 8000)
    plot_panel(axes[1], 16000)
    axes[0].set_ylabel("EM accuracy", fontsize=14)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04), fontsize=12.5, frameon=False)

    fig.suptitle(
        "Combined SnapKV × role-conditional eviction beats SnapKV across budgets;\n"
        "exceeds the full-budget oracle at B ≥ 0.30 on ctx 16k",
        fontsize=14, y=1.04,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
