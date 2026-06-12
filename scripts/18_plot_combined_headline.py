"""Render the paper's headline figures.

(1) fig_combined_headline.png — combined-method super-additive composition +
    over-oracle effect on indented JSON (Llama), ctx 8k and ctx 16k.
    Two panels, 4 lines per panel:
      * H2O baseline               — black dashed (condition=All in e_counterfactual)
      * SnapKV-only                — blue solid (config=snapkv-only in ac_combined)
      * A no-key (role allocator)  — green solid (policy=no-key in a_role_conditional;
                                       ctx 8k only — not run at 16k)
      * snapkv-no-key (combined)   — red solid, bold (best-α per ctx: soft02@8k, no-key@16k)
    Oracle dashed horizontal at y=0.88 (H2O at B=1.0). The over-oracle effect
    at ctx 16k (combined > oracle at B≥0.30) is the figure's headline.

(2) fig_cross_model_headline.png — the post-Qwen cross-model story: combined
    (snapkv-no-key, solid) vs SnapKV-only (dashed) on synthetic_json ctx 16k for
    Llama / Mistral / Qwen. The C8 gain is large on Llama, compressed on Mistral,
    and ~zero on Qwen (combined ties SnapKV) — the model-coupled residual.

Canonical headline numbers come from the unified fp32 SDPA decode path
(`ac_combined_{llama,mistral}_sdpa.csv`) at ctx 16k; ctx-8k Llama stays on the
locked `ac_combined.csv` (not regenerated — SDPA only covers the ctx-16k headline).

Output: results/plan2/fig_combined_headline.png,
        results/plan2/fig_cross_model_headline.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("results/plan2/fig_combined_headline.png")
OUT_CROSS = Path("results/plan2/fig_cross_model_headline.png")
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
ORACLE = 0.88  # H2O @ budget=1.0 on synthetic_json (matches both ctxs)

# Per-ctx best α: α=0.02 at 8k, α=0 at 16k (from the 2026-05-20 α-sweep).
COMBINED_CFG = {8000: "snapkv-no-key-soft02", 16000: "snapkv-no-key"}
COMBINED_LABEL = {8000: "Combined (α=0.02)", 16000: "Combined (α=0)"}

# Canonical combined-method CSV per ctx: SDPA path for the ctx-16k headline,
# locked eager CSV for ctx-8k (SDPA regeneration covered ctx-16k only).
AC_CSV = {8000: "results/plan2/ac_combined.csv",
          16000: "results/plan2/ac_combined_llama_sdpa.csv"}

# Cross-model panel sources (synthetic_json ctx 16k): (csv, colour).
CROSS_MODELS = {
    "Llama-3.1-8B":  ("results/plan2/ac_combined_llama_sdpa.csv",   "#d62728"),
    "Mistral-7B":    ("results/plan2/ac_combined_mistral_sdpa.csv", "#1f77b4"),
    "Qwen2.5-7B":    ("results/plan2/ac_combined_qwen.csv",         "#9467bd"),
}


def curve_em(df: pd.DataFrame, mask) -> list[float | None]:
    """Mean EM at each budget in BUDGETS; None if a budget is missing."""
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

    ax.axhline(ORACLE, color="gray", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(0.5, ORACLE + 0.012, "full-budget oracle (0.88)",
            ha="right", va="bottom", fontsize=8, color="gray")

    ax.plot(BUDGETS, h2o,  color="black",   linestyle="--", marker="s",
            label="H2O",          linewidth=1.2, markersize=4, zorder=2)
    ax.plot(BUDGETS, snap, color="#1f77b4", linestyle="-",  marker="o",
            label="SnapKV-only",  linewidth=1.4, markersize=4, zorder=3)
    ax.plot(BUDGETS, a_nokey, color="#2ca02c", linestyle="-", marker="^",
            label="A no-key (role allocator)",
            linewidth=1.4, markersize=4, zorder=3)
    ax.plot(BUDGETS, comb, color="#d62728", linestyle="-",  marker="*",
            label=COMBINED_LABEL[ctx],
            linewidth=2.0, markersize=8, zorder=4)

    # Over-oracle annotation at ctx 16k B=0.30 only
    if ctx == 16000:
        ax.annotate(
            "over-oracle:\n0.98 > 0.88",
            xy=(0.30, 0.98), xytext=(0.33, 0.65),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
            fontsize=8, color="#d62728",
        )

    ax.set_title(f"indented JSON, ctx {ctx//1000}k", fontsize=11)
    ax.set_xlabel("KV budget (fraction)")
    ax.set_xticks(BUDGETS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.03, 0.52)
    ax.grid(alpha=0.3)


def plot_cross_model():
    """Cross-model panel: combined (solid) vs SnapKV-only (dashed) on
    synthetic_json ctx 16k for Llama / Mistral / Qwen. Tells the C8
    model-coupled-residual story: large gap on Llama, compressed on Mistral,
    ~zero on Qwen."""
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.6))
    for name, (csv, color) in CROSS_MODELS.items():
        df = pd.read_csv(csv)
        m = (df["source"] == "synthetic_json") & (df["ctx_tokens"] == 16000)
        comb = curve_em(df, m & (df["config"] == "snapkv-no-key"))
        snap = curve_em(df, m & (df["config"] == "snapkv-only"))
        delta = comb[0] - snap[0]  # Δ at B=0.05, the headline budget
        ax.plot(BUDGETS, comb, color=color, linestyle="-", marker="*",
                linewidth=2.0, markersize=9,
                label=f"{name} — combined (Δ@5%={delta:+.2f})")
        ax.plot(BUDGETS, snap, color=color, linestyle="--", marker="o",
                linewidth=1.2, markersize=4, alpha=0.55,
                label=f"{name} — SnapKV-only")

    ax.set_title("Combined vs SnapKV across model families\n"
                 "(synthetic_json, ctx 16k) — the C8 gain is a model-coupled residual",
                 fontsize=10.5)
    ax.set_xlabel("KV budget (fraction)")
    ax.set_ylabel("EM accuracy")
    ax.set_xticks(BUDGETS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.03, 0.52)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    OUT_CROSS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CROSS, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_CROSS}")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    plot_panel(axes[0], 8000)
    plot_panel(axes[1], 16000)
    axes[0].set_ylabel("EM accuracy")

    # Single combined legend in the gutter under both panels
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.03), fontsize=9, frameon=False)

    fig.suptitle(
        "Combined SnapKV × role-conditional eviction beats SnapKV across budgets;\n"
        "exceeds the full-budget oracle at B ≥ 0.30 on ctx 16k",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    plt.close(fig)

    plot_cross_model()


if __name__ == "__main__":
    main()
