"""Pilot Q4 — Are KEY tokens disproportionately important in some layers?

We compute the per-layer attention-mass gap (mean VALUE mass − mean KEY mass).
A consistent positive gap across layers → uniform schema-budget treatment is
fine. A gap that varies dramatically by layer → C+'s per-layer policy
has signal.
"""
import polars as pl

from kvr.structure.roles import Role


def compute_per_layer_attention_gap(df: pl.DataFrame) -> pl.DataFrame:
    """Per (source, layer) gap = mean(VALUE mass) − mean(KEY mass).

    Args:
        df: per-row stacked dumps with (source, layer, head, kv_pos,
            accumulated_mass, role).

    Returns:
        columns (source, layer, mean_key_mass, mean_value_mass,
                 gap_value_minus_key).
    """
    per_layer_role = (
        df.group_by(["source", "layer", "role"])
          .agg([
              pl.col("accumulated_mass").mean().alias("mean_mass"),
          ])
    )
    pivot = per_layer_role.pivot(
        values="mean_mass",
        index=["source", "layer"],
        on="role",
    ).fill_null(0.0)
    out = pivot.with_columns(
        (pl.col(Role.VALUE.value) - pl.col(Role.KEY.value)).alias("gap_value_minus_key"),
    )
    if Role.KEY.value in out.columns:
        out = out.rename({Role.KEY.value: "mean_key_mass", Role.VALUE.value: "mean_value_mass"})
    return out.sort(["source", "layer"])
