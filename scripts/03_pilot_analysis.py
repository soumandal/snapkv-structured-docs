"""Pilot analysis driver: reads dumps, produces Q1–Q4 outputs.

Memory strategy: each dump file is loaded inside a short-lived worker
process so Polars' Rust arena is freed between files. Partials are
streamed to a temp dir on disk and then combined.

For Q3 see docs/pilot_report.md — deferred to Plan 2.
"""
import argparse
import shutil
import tempfile
from multiprocessing import Pool, get_context
from pathlib import Path

import polars as pl
from tqdm import tqdm

from kvr.structure.roles import Role


BUDGET_FRACS = [0.20, 0.10, 0.05]


def _q1_partial(df: pl.DataFrame) -> pl.DataFrame:
    # Drop NaN mass rows before the mean. Polars mean() propagates a single NaN
    # to NaN for the whole group, and the Qwen full-mass dumps carry ~0.25% NaN
    # positions from the recorder's discarded-output fp16 drift (§15.5/§16.7).
    # No-op on the Llama/Mistral/Phi dumps (0 NaN); makes Q1 sink-identity
    # computable on Qwen without a re-record (the drift is inherent, not a
    # schema gap — re-recording reproduces it).
    per_token = (
        df.filter(pl.col("accumulated_mass").is_not_nan())
          .group_by(["prompt_id", "source", "kv_pos", "role"])
          .agg(pl.col("accumulated_mass").mean().alias("token_mean_mass"))
    )
    return (
        per_token.group_by(["source", "role"])
                 .agg(
                     pl.col("token_mean_mass").sum().alias("_sum"),
                     pl.len().alias("_count"),
                 )
    )


def _q2_partial(df: pl.DataFrame, budget_fracs: list[float]) -> pl.DataFrame:
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
        (pl.col("_rank") <= pl.col("_k")).cast(pl.Int64).alias("_retained"),
    )
    return (
        expanded.group_by(["source", "role", "budget_frac"])
                .agg(
                    pl.col("_retained").sum().alias("_retained_sum"),
                    pl.len().alias("_count"),
                )
    )


def _q4_partial(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["source", "layer", "role"])
          .agg(
              pl.col("accumulated_mass").sum().alias("_sum"),
              pl.len().alias("_count"),
          )
    )


def _process_one(args: tuple[str, str]) -> str:
    """Worker: load one parquet, write three partial parquets to tmpdir."""
    file_path, tmpdir = args
    f = Path(file_path)
    tdir = Path(tmpdir)
    df = pl.read_parquet(file_path)
    _q1_partial(df).write_parquet(tdir / f"q1_{f.stem}.parquet")
    _q2_partial(df, BUDGET_FRACS).write_parquet(tdir / f"q2_{f.stem}.parquet")
    _q4_partial(df).write_parquet(tdir / f"q4_{f.stem}.parquet")
    return f.stem


def _q1_combine(tmpdir: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(str(tmpdir / "q1_*.parquet"))
          .group_by(["source", "role"])
          .agg(
              pl.col("_sum").sum().alias("_sum"),
              pl.col("_count").sum().alias("n_tokens"),
          )
          .with_columns((pl.col("_sum") / pl.col("n_tokens")).alias("mean_mass"))
          .select(["source", "role", "mean_mass", "n_tokens"])
          .sort(["source", "role"])
    )


def _q2_combine(tmpdir: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(str(tmpdir / "q2_*.parquet"))
          .group_by(["source", "role", "budget_frac"])
          .agg(
              pl.col("_retained_sum").sum().alias("_retained_sum"),
              pl.col("_count").sum().alias("n_tokens"),
          )
          .with_columns(
              (pl.col("_retained_sum") / pl.col("n_tokens")).alias("retention_rate"),
          )
          .select(["source", "role", "budget_frac", "retention_rate", "n_tokens"])
          .sort(["source", "role", "budget_frac"])
    )


def _q4_combine(tmpdir: Path) -> pl.DataFrame:
    per_layer_role = (
        pl.read_parquet(str(tmpdir / "q4_*.parquet"))
          .group_by(["source", "layer", "role"])
          .agg(
              pl.col("_sum").sum().alias("_sum"),
              pl.col("_count").sum().alias("_count"),
          )
          .with_columns((pl.col("_sum") / pl.col("_count")).alias("mean_mass"))
          .select(["source", "layer", "role", "mean_mass"])
    )
    pivot = per_layer_role.pivot(
        values="mean_mass", index=["source", "layer"], on="role"
    ).fill_null(0.0)
    key_col, val_col = Role.KEY.value, Role.VALUE.value
    if key_col not in pivot.columns:
        pivot = pivot.with_columns(pl.lit(0.0).alias(key_col))
    if val_col not in pivot.columns:
        pivot = pivot.with_columns(pl.lit(0.0).alias(val_col))
    return (
        pivot.with_columns(
            (pl.col(val_col) - pl.col(key_col)).alias("gap_value_minus_key"),
        )
        .rename({key_col: "mean_key_mass", val_col: "mean_value_mass"})
        .sort(["source", "layer"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/kvr/results/pilot"))
    ap.add_argument("--tmpdir", type=Path, default=None,
                    help="Where to stage per-file partials (default: a fresh /tmp subdir)")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--skip-q3", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = args.tmpdir or Path(tempfile.mkdtemp(prefix="pilot_partials_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.run_root.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No dumps found under {args.run_root}")
    print(f"Found {len(files)} parquet files; staging partials in {tmpdir}")

    # Skip files we've already processed (in case of restart).
    todo = [
        f for f in files
        if not (tmpdir / f"q1_{f.stem}.parquet").exists()
    ]
    print(f"  {len(files) - len(todo)} already staged; processing {len(todo)} fresh")

    ctx = get_context("spawn")
    with ctx.Pool(processes=2, maxtasksperchild=10) as pool:
        tasks = [(str(f), str(tmpdir)) for f in todo]
        for _ in tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks), desc="files"):
            pass

    print("Combining Q1...")
    q1 = _q1_combine(tmpdir)
    q1.write_csv(args.out_dir / "q1_attention_mass.csv")
    print(q1)

    print("Combining Q2...")
    q2 = _q2_combine(tmpdir)
    q2.write_csv(args.out_dir / "q2_h2o_retention.csv")
    print(q2)

    print("Combining Q4...")
    q4 = _q4_combine(tmpdir)
    q4.write_csv(args.out_dir / "q4_layer_hetero.csv")
    print(q4)

    if args.skip_q3:
        print("Q3 → SKIPPED")
    else:
        print("Q3 → deferred to Plan 2 (see docs/pilot_report.md)")

    if not args.keep_tmp:
        shutil.rmtree(tmpdir)
        print(f"Cleaned {tmpdir}")
    else:
        print(f"Kept {tmpdir}")

    print(f"\nOutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
