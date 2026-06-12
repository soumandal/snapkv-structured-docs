"""Pilot Q3 — does low KEY retention cause accuracy drops?

We compute Pearson correlation between per-prompt KEY-retention rate (under
H2O) and per-prompt correctness (0/1), grouped by source and budget.

Upstream: the caller is responsible for producing the input frame with
columns (prompt_id, source, budget_frac, key_retention_rate, correct).
The pilot-analysis script generates this by:
  1. For each prompt + budget, simulate H2O eviction.
  2. Compute KEY-retention rate (Q2 logic, prompt-level).
  3. Decode an answer under H2O eviction and grade against the gold answer.
"""
import polars as pl


def compute_accuracy_vs_key_retention(df: pl.DataFrame) -> pl.DataFrame:
    """Pearson correlation between KEY retention and correctness.

    Args:
        df: columns (prompt_id, source, budget_frac, key_retention_rate, correct).

    Returns:
        DataFrame with (source, budget_frac, pearson_r, n_prompts).
    """
    return (
        df.with_columns(pl.col("correct").cast(pl.Float64))
          .group_by(["source", "budget_frac"])
          .agg(
              pl.corr("key_retention_rate", "correct").fill_null(0.0).alias("pearson_r"),
              pl.len().alias("n_prompts"),
          )
          .filter(pl.col("n_prompts") >= 2)
          .sort(["source", "budget_frac"])
    )
