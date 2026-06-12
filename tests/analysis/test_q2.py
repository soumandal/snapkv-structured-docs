import polars as pl

from kvr.analysis.q2_h2o_retention import compute_h2o_retention_by_role


def test_retention_at_full_budget_is_100pct():
    # 4 tokens, 1 layer, 1 head; budget = 4 → keep all → retention 100%.
    df = pl.DataFrame({
        "prompt_id": ["p1"] * 4,
        "source": ["spider"] * 4,
        "layer": [0] * 4,
        "head": [0] * 4,
        "kv_pos": [0, 1, 2, 3],
        "accumulated_mass": [1.0, 2.0, 3.0, 4.0],
        "role": ["KEY", "VALUE", "KEY", "VALUE"],
        "depth": [0] * 4,
    })
    out = compute_h2o_retention_by_role(df, budget_fracs=[1.0])
    assert all(r == 1.0 for r in out["retention_rate"])


def test_at_50pct_budget_top_mass_tokens_retained():
    df = pl.DataFrame({
        "prompt_id": ["p1"] * 4,
        "source": ["spider"] * 4,
        "layer": [0] * 4,
        "head": [0] * 4,
        "kv_pos": [0, 1, 2, 3],
        "accumulated_mass": [1.0, 10.0, 2.0, 20.0],  # VALUE tokens have higher mass
        "role": ["KEY", "VALUE", "KEY", "VALUE"],
        "depth": [0] * 4,
    })
    out = compute_h2o_retention_by_role(df, budget_fracs=[0.5])
    # At budget 50% (keep 2 of 4): top-2 by mass = positions 1 and 3 (both VALUE).
    # KEY retention: 0/2 = 0; VALUE retention: 2/2 = 1.0.
    key_row = out.filter((pl.col("role") == "KEY") & (pl.col("budget_frac") == 0.5))
    val_row = out.filter((pl.col("role") == "VALUE") & (pl.col("budget_frac") == 0.5))
    assert key_row["retention_rate"][0] == 0.0
    assert val_row["retention_rate"][0] == 1.0
