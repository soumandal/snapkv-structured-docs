"""Pilot Q1 — Attention-mass gap.

Question: do KEY tokens have *lower* mean accumulated attention than VALUE
tokens, conditional on being in the context?
"""
import polars as pl


def compute_attention_mass_by_role(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-token accumulated mass into mean per (source, role).

    Args:
        df: Output of stacking all per-prompt dumps. Columns include
            "source", "role", "accumulated_mass". Each (prompt_id, layer,
            head, kv_pos) is one row.

    Returns:
        DataFrame with columns: source, role, mean_mass, n_tokens.
    """
    per_token = (
        df.group_by(["prompt_id", "source", "kv_pos", "role"])
          .agg(pl.col("accumulated_mass").mean().alias("token_mean_mass"))
    )
    out = (
        per_token.group_by(["source", "role"])
                 .agg([
                     pl.col("token_mean_mass").mean().alias("mean_mass"),
                     pl.len().alias("n_tokens"),
                 ])
                 .sort(["source", "role"])
    )
    return out
