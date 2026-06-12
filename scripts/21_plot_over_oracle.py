#!/usr/bin/env python3
"""Render fig_over_oracle_scatter.png — the C9 over-oracle / denoising figure.

Per-prompt contingency between the combined method (`snapkv-no-key`) and the
full-budget oracle (`All` at B=1.0) on synthetic_json ctx 16k, Llama, n=50.
Each prompt at each budget falls into one of four outcomes:

  parity      both correct        (combined keeps what the oracle gets)
  recovery    oracle wrong, combined correct   (denoising win — above the diagonal)
  regression  oracle correct, combined wrong   (the cost of eviction)
  both-wrong  neither correct

The C9 story: at B>=0.30 the regression count is **zero** — the combined method
is a strict superset of the full-budget oracle, plus denoising recoveries. A
random sampling fluctuation would produce regressions too; zero is incompatible
with the noise null.

Originally produced by inline polars (findings §10.4); promoted to a committed
driver so the figure can be regenerated under a shared style. Driver written
2026-06-02.

Source: results/plan2/over_oracle_scatter_contingency.csv (one row per budget;
matches the paper's Table tab:scatter) + over_oracle_scatter_per_prompt.csv.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results/plan2")
OUT = RESULTS / "fig_over_oracle_scatter.png"

# outcome column -> (legend label, color)
OUTCOMES = [
    ("both_correct", "parity (both correct)", "#bdbdbd"),
    ("nokey_only", "recovery (denoising win)", "#2ca02c"),
    ("oracle_only", "regression (eviction cost)", "#d62728"),
    ("both_wrong", "both wrong", "#f0f0f0"),
]


def plot(cont: pd.DataFrame, out: Path) -> None:
    cont = cont.sort_values("budget_frac").reset_index(drop=True)
    x = range(len(cont))
    labels = [f"{b:.2f}" for b in cont["budget_frac"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = [0.0] * len(cont)
    for col, label, color in OUTCOMES:
        vals = cont[col].tolist()
        ax.bar(x, vals, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]

    n = int(cont["n_prompts"].iloc[0])
    # Annotate net margin (combined EM - oracle EM) above each bar.
    for xi, (_, row) in zip(x, cont.iterrows()):
        net = row["nokey_em"] - row["oracle_em"]
        ax.text(xi, n + 1.5, f"{net:+.2f}",
                ha="center", va="bottom", fontsize=9,
                color=("#2ca02c" if net >= 0 else "#d62728"))

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("budget fraction $B$")
    ax.set_ylabel(f"prompts (n={n})")
    ax.set_ylim(0, n + 14)
    ax.set_title("Combined vs full-budget oracle, per-prompt outcomes "
                 "(synthetic_json ctx 16k, Llama)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    # Mark the zero-regression regime (placed in the headroom above the bars).
    zero_reg = cont[cont["oracle_only"] == 0]["budget_frac"].tolist()
    if zero_reg:
        ax.text(0.5, 0.995,
                "zero regressions at B = " + ", ".join(f"{b:.2f}" for b in zero_reg),
                transform=ax.transAxes, fontsize=8, ha="center", va="top",
                color="#d62728")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS,
                    help="dir holding over_oracle_scatter_contingency.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--style", type=Path, default=None,
                    help="optional matplotlib style file (.mplstyle) for restyling")
    args = ap.parse_args()

    if args.style is not None:
        plt.style.use(str(args.style))

    cont = pd.read_csv(args.results_dir / "over_oracle_scatter_contingency.csv")
    plot(cont, args.out)


if __name__ == "__main__":
    main()
