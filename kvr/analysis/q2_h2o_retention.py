"""Pilot Q2 — H2O retention gap.

Question: at budgets {20%, 10%, 5%}, what fraction of KEY tokens does H2O
retain vs VALUE tokens?

We simulate H2O retention per (prompt, layer, head) by computing top-k
of accumulated_mass at the given budget, then count how many of each role
survive. Aggregate across (layer, head, prompt) for the per-(source, role,
budget) retention rate.
"""
import polars as pl


def compute_h2o_retention_by_role(
    df: pl.DataFrame,
    budget_fracs: list[float],
) -> pl.DataFrame:
    """Compute retention rate per role per budget per source.

    Returns: columns (source, role, budget_frac, retention_rate, n_tokens).
    """
    head_keys = ["prompt_id", "source", "layer", "head"]
    ranked = df.with_columns(
        pl.col("accumulated_mass")
          .rank(method="ordinal", descending=True)
          .over(head_keys).alias("_rank"),
        pl.len().over(head_keys).alias("_n_kv"),
    )
    budgets = pl.DataFrame({"budget_frac": budget_fracs})
    expanded = ranked.join(budgets, how="cross").with_columns(
        (pl.col("_n_kv") * pl.col("budget_frac")).ceil().cast(pl.Int64).alias("_k"),
    ).with_columns(
        (pl.col("_rank") <= pl.col("_k")).alias("_retained"),
    )
    return (
        expanded.group_by(["source", "role", "budget_frac"])
                .agg(
                    pl.col("_retained").mean().alias("retention_rate"),
                    pl.len().alias("n_tokens"),
                )
                .sort(["source", "role", "budget_frac"])
    )
