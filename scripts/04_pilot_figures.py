"""Produce pilot figures from the Q1/Q2/Q4 CSV outputs.

Figures:
  fig1_q1_mass_by_role.png — bar plot of mean attention mass per role, per source
  fig2_q2_retention.png    — H2O retention rate by role × budget × source
  fig3_q4_layer_gap.png    — per-layer VALUE − SCHEMA mass gap (schema = max(KEY, HEADER))
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


# Roles to highlight in plots. Sink-like roles (DELIM, WS) are bundled but
# excluded from the main schema/value contrast to keep the y-axes legible.
_MAIN_ROLES = ["KEY", "HEADER", "VALUE", "PROSE", "DELIM", "WS"]


def plot_q1(df: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    p = df.filter(pl.col("role").is_in(_MAIN_ROLES)).to_pandas()
    sns.barplot(p, x="source", y="mean_mass", hue="role",
                hue_order=_MAIN_ROLES, ax=ax)
    ax.set_title("Q1: Mean accumulated attention mass per (source, role)")
    ax.set_ylabel("Mean accumulated mass")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_q2(df: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    p = df.filter(pl.col("role").is_in(_MAIN_ROLES)).to_pandas()
    sns.lineplot(
        p, x="budget_frac", y="retention_rate",
        hue="role", hue_order=_MAIN_ROLES,
        style="source", markers=True, ax=ax,
    )
    # Reference: random retention (= budget_frac).
    budgets = sorted(p["budget_frac"].unique())
    ax.plot(budgets, budgets, color="black", linestyle=":", linewidth=0.8,
            label="random (= budget)")
    ax.set_title("Q2: H2O retention rate per role")
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Retention rate")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_q4(df: pl.DataFrame, out: Path) -> None:
    # Compute "schema mass" = max(KEY, HEADER) per (source, layer), then
    # gap = VALUE − schema. Captures both JSON-shaped and table-shaped corpora.
    work = df.with_columns(
        pl.max_horizontal("mean_key_mass", "HEADER").alias("schema_mass"),
    ).with_columns(
        (pl.col("mean_value_mass") - pl.col("schema_mass")).alias("gap_value_minus_schema"),
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    p = work.to_pandas()
    sns.lineplot(p, x="layer", y="gap_value_minus_schema",
                 hue="source", markers=True, ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set_title("Q4: Per-layer attention gap (VALUE − max(KEY, HEADER))")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean mass: VALUE − schema")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("/mnt/kvr/results/pilot"))
    args = ap.parse_args()

    q1 = pl.read_csv(args.results_dir / "q1_attention_mass.csv")
    q2 = pl.read_csv(args.results_dir / "q2_h2o_retention.csv")
    q4 = pl.read_csv(args.results_dir / "q4_layer_hetero.csv")

    plot_q1(q1, args.results_dir / "fig1_q1_mass_by_role.png")
    plot_q2(q2, args.results_dir / "fig2_q2_retention.png")
    plot_q4(q4, args.results_dir / "fig3_q4_layer_gap.png")

    print(f"Figures written to {args.results_dir}")


if __name__ == "__main__":
    main()
