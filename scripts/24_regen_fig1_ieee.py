"""Regenerate fig1_q1_mass_by_role.png for IEEE double-column inclusion.

Same data as scripts/04_pilot_figures.py::plot_q1, but retitled (no "Q1:")
and restyled for legibility at small (single-column) print size: larger fonts,
tighter figure, value labels removed in favour of a clean log-scale grid.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

_MAIN_ROLES = ["KEY", "HEADER", "VALUE", "PROSE", "DELIM", "WS"]


def plot_q1_ieee(df: pl.DataFrame, out: Path) -> None:
    # Bump base font sizes: IEEE columns are ~3.5in wide, so the figure is
    # printed small and default matplotlib text becomes unreadable.
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    p = df.filter(pl.col("role").is_in(_MAIN_ROLES)).to_pandas()
    sns.barplot(p, x="source", y="mean_mass", hue="role",
                hue_order=_MAIN_ROLES, ax=ax)
    ax.set_title("Mean accumulated attention mass per (source, role)")
    ax.set_ylabel("Mean accumulated mass")
    ax.set_xlabel("")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    # Legend inside the axes on the upper left (as in the original), just
    # with larger text for IEEE print legibility.
    ax.legend(title="Role", loc="upper left", fontsize=12,
              title_fontsize=12, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("white_paper_figures/data/q1_attention_mass.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("white_paper_figures/fig1_q1_mass_by_role.png"))
    args = ap.parse_args()
    plot_q1_ieee(pl.read_csv(args.csv), args.out)


if __name__ == "__main__":
    main()
