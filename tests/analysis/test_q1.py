import polars as pl

from kvr.analysis.q1_attention_mass import compute_attention_mass_by_role


def test_returns_per_source_per_role_means():
    df = pl.DataFrame({
        "prompt_id": ["p1"] * 6 + ["p2"] * 6,
        "source": ["spider"] * 6 + ["wikitable"] * 6,
        "layer": [0] * 12,
        "head": [0] * 12,
        "kv_pos": list(range(6)) + list(range(6)),
        "accumulated_mass": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "role": ["KEY", "VALUE", "KEY", "VALUE", "PROSE", "DELIM"] * 2,
        "depth": [0] * 12,
    })
    out = compute_attention_mass_by_role(df)
    # Expect columns: source, role, mean_mass, n_tokens
    assert {"source", "role", "mean_mass", "n_tokens"}.issubset(out.columns)
    # KEY mean for spider: (1+3)/2 = 2.0
    key_spider = out.filter((pl.col("source") == "spider") & (pl.col("role") == "KEY"))
    assert key_spider["mean_mass"][0] == 2.0


def test_includes_gap_metric():
    df = pl.DataFrame({
        "prompt_id": ["p1"] * 4,
        "source": ["spider"] * 4,
        "layer": [0] * 4,
        "head": [0] * 4,
        "kv_pos": [0, 1, 2, 3],
        "accumulated_mass": [1.0, 5.0, 2.0, 6.0],
        "role": ["KEY", "VALUE", "KEY", "VALUE"],
        "depth": [0] * 4,
    })
    out = compute_attention_mass_by_role(df)
    spider_rows = out.filter(pl.col("source") == "spider")
    key_mean = spider_rows.filter(pl.col("role") == "KEY")["mean_mass"][0]
    value_mean = spider_rows.filter(pl.col("role") == "VALUE")["mean_mass"][0]
    # Q1 premise (KEY < VALUE) should be directly inspectable.
    assert key_mean == 1.5
    assert value_mean == 5.5
