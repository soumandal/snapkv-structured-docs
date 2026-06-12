import polars as pl

from kvr.analysis.q3_accuracy_corr import compute_accuracy_vs_key_retention


def test_correlation_when_low_key_retention_hurts_accuracy():
    # 4 prompts: 2 with low key retention, 2 with high.
    df = pl.DataFrame({
        "prompt_id": ["p1", "p2", "p3", "p4"],
        "source": ["synthetic_json"] * 4,
        "budget_frac": [0.1] * 4,
        "key_retention_rate": [0.2, 0.3, 0.9, 0.95],
        "correct": [0, 0, 1, 1],
    })
    out = compute_accuracy_vs_key_retention(df)
    # We expect a positive correlation between key retention and correctness.
    spider_row = out.filter(pl.col("source") == "synthetic_json")
    assert spider_row["pearson_r"][0] > 0.5


def test_returns_per_source_per_budget_rows():
    df = pl.DataFrame({
        "prompt_id": ["p1", "p2", "p3", "p4"],
        "source": ["spider", "spider", "wikitable", "wikitable"],
        "budget_frac": [0.1, 0.1, 0.1, 0.1],
        "key_retention_rate": [0.2, 0.9, 0.3, 0.8],
        "correct": [0, 1, 0, 1],
    })
    out = compute_accuracy_vs_key_retention(df)
    sources = set(out["source"].to_list())
    assert sources == {"spider", "wikitable"}
