import polars as pl

from kvr.analysis.q4_layer_hetero import compute_per_layer_attention_gap


def test_returns_per_layer_per_source_gap():
    df = pl.DataFrame({
        "prompt_id": ["p1"] * 8,
        "source": ["spider"] * 8,
        "layer": [0, 0, 0, 0, 1, 1, 1, 1],
        "head": [0] * 8,
        "kv_pos": [0, 1, 2, 3, 0, 1, 2, 3],
        "accumulated_mass": [1.0, 5.0, 2.0, 6.0, 10.0, 1.0, 11.0, 2.0],
        "role": ["KEY", "VALUE", "KEY", "VALUE"] * 2,
        "depth": [0] * 8,
    })
    out = compute_per_layer_attention_gap(df)
    assert "layer" in out.columns
    assert "gap_value_minus_key" in out.columns
    # Layer 0: KEY mean = 1.5, VALUE mean = 5.5 → gap = +4.0
    # Layer 1: KEY mean = 10.5, VALUE mean = 1.5 → gap = -9.0
    l0 = out.filter(pl.col("layer") == 0)
    l1 = out.filter(pl.col("layer") == 1)
    assert l0["gap_value_minus_key"][0] == 4.0
    assert l1["gap_value_minus_key"][0] == -9.0
