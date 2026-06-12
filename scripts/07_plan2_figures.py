"""Produce Plan-2 figures from the Experiment E and C CSVs.

Figures:
  fig_e_counterfactual.png — EM accuracy vs budget, one line per role-eviction
                             condition (All / No-DELIM / No-KEY / No-VALUE).
  fig_c_recent_window.png  — EM accuracy vs budget, one line per recent-window
                             spec (abs64 / frac0.10 / frac0.25 / frac0.50),
                             with the H2O "All" baseline overlaid for reference.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


_E_CONDITIONS = ["All", "No-DELIM", "No-KEY", "No-VALUE"]
_C_WINDOWS = ["abs64", "frac0.10", "frac0.25", "frac0.50"]


def _accuracy_by(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    return (df.group_by(group_cols)
              .agg(pl.col("correct").mean().alias("acc"),
                   pl.len().alias("n"))
              .sort(group_cols))


def plot_e(e: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    agg = _accuracy_by(e, ["source", "condition", "budget_frac"])
    p = agg.to_pandas()
    sns.lineplot(p, x="budget_frac", y="acc",
                 hue="condition", hue_order=_E_CONDITIONS,
                 style="source", markers=True, dashes=False, ax=ax)
    ceiling = e.filter(pl.col("budget_frac") == 1.0)["correct"].mean()
    ax.axhline(ceiling, color="black", linestyle=":", linewidth=0.8,
               label=f"full-budget ceiling ({ceiling:.2f})")
    ax.set_title("Experiment E: counterfactual role eviction (synthetic_json, ctx=8000)")
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_c(c: pl.DataFrame, e: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    agg_c = _accuracy_by(c, ["source", "window_spec", "budget_frac"]).to_pandas()
    sns.lineplot(agg_c, x="budget_frac", y="acc",
                 hue="window_spec", hue_order=_C_WINDOWS,
                 style="source", markers=True, dashes=False, ax=ax)

    h2o = _accuracy_by(e.filter(pl.col("condition") == "All"),
                       ["budget_frac"]).to_pandas()
    ax.plot(h2o["budget_frac"], h2o["acc"],
            color="black", linestyle="--", marker="x", linewidth=1.0,
            label="H2O baseline (E: All)")

    ceiling = c.filter(pl.col("budget_frac") == 1.0)["correct"].mean()
    ax.axhline(ceiling, color="black", linestyle=":", linewidth=0.8,
               label=f"full-budget ceiling ({ceiling:.2f})")
    ax.set_title("Experiment C: recent-window eviction vs H2O baseline (synthetic_json, ctx=8000)")
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results/plan2"))
    ap.add_argument("--source-filter", default="synthetic_json",
                    help="Restrict CSVs to one source before plotting "
                         "(empty string = all sources).")
    ap.add_argument("--ctx-filter", type=int, default=None,
                    help="Restrict CSVs to one ctx_tokens bucket "
                         "(default: all in CSV).")
    args = ap.parse_args()

    e = pl.read_csv(args.results_dir / "e_counterfactual.csv")
    c = pl.read_csv(args.results_dir / "c_recent_window.csv")
    if args.source_filter:
        e = e.filter(pl.col("source") == args.source_filter)
        c = c.filter(pl.col("source") == args.source_filter)
    if args.ctx_filter is not None:
        e = e.filter(pl.col("ctx_tokens") == args.ctx_filter)
        c = c.filter(pl.col("ctx_tokens") == args.ctx_filter)

    suffix = f"_ctx{args.ctx_filter}" if args.ctx_filter is not None else ""
    plot_e(e, args.results_dir / f"fig_e_counterfactual{suffix}.png")
    plot_c(c, e, args.results_dir / f"fig_c_recent_window{suffix}.png")

    print(f"Figures written to {args.results_dir}")


if __name__ == "__main__":
    main()
