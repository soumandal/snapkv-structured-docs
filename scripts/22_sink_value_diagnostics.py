#!/usr/bin/env python3
"""Two cheap mechanism diagnostics over the Llama full-mass pilot dumps (§17 P2).

Both run analyzer-only over the durable per-prompt parquet dumps (no GPU). Each
dump is per-(layer, head, kv_pos) accumulated H2O mass with a structural role
label, so we process one prompt-file at a time and accumulate small partials.

DIAGNOSTIC A — BOS-sink vs DELIM-sink retention.
  Sharpens the StreamingLLM distinction. StreamingLLM keeps a single, fixed BOS
  sink (kv_pos 0). We show the *operational* H2O sink in structured inputs is the
  DELIM role: many positions (not one), at text-inferable locations (not fixed).
  We split the sink into BOS (kv_pos==0) vs DELIM (role==DELIM, kv_pos>0) and
  report H2O retention rate + mean mass + distinct-positions-per-prompt for each.
  If DELIM is retained at high rate across *many* positions while BOS is one
  position, the sink phenomenon is not reducible to the BOS attention sink.

DIAGNOSTIC B — VALUE tokens in the bottom mass-quartile (direct H2 support).
  Within each (layer, head), bucket kv positions into mass quartiles. Report the
  fraction of each role landing in the bottom quartile (lowest 25% mass) vs the
  top quartile. If VALUE (the answer-carrying role) is over-represented in the
  bottom quartile (>25% baseline) and KEY in the top, that is direct evidence
  that H2O's score under-attends the answer tokens — the mechanism behind the
  88-point collapse. (Caveat: the dump labels role=VALUE but carries no gold
  answer-span flag, so this is all-VALUE as a proxy for answer-VALUE.)

Source: /mnt/kvr/dumps/2026-WW20-pilot/ctx_16000/*.parquet (Llama-3.1-8B,
synthetic_json, ctx 16k, n=200, full-mass eager schema). Driver written 2026-06-02.
Outputs: results/plan2/diag_sink_bos_vs_delim.csv, diag_value_quartile.csv.
"""
import argparse
import glob
from pathlib import Path

import polars as pl

BUDGETS = [0.05, 0.10, 0.20]
HEAD_KEYS = ["layer", "head"]


def _sink_class() -> pl.Expr:
    return (
        pl.when(pl.col("kv_pos") == 0).then(pl.lit("BOS (pos 0)"))
        .when(pl.col("role") == "DELIM").then(pl.lit("DELIM (non-BOS)"))
        .otherwise(pl.col("role"))
        .alias("sink_class")
    )


def process_file(path: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = pl.read_parquet(path).with_columns(_sink_class())
    ranked = df.with_columns(
        pl.col("accumulated_mass").rank("ordinal", descending=True)
          .over(HEAD_KEYS).alias("rank"),
        pl.len().over(HEAD_KEYS).alias("n_kv"),
    )

    # --- Diagnostic A: retention by sink_class x budget ---
    ret_parts = []
    for b in BUDGETS:
        r = ranked.with_columns(
            (pl.col("rank") <= (pl.col("n_kv") * b).ceil()).cast(pl.Int64).alias("ret")
        )
        ret_parts.append(
            r.group_by("sink_class").agg(
                pl.col("ret").sum().alias("ret_sum"), pl.len().alias("cnt")
            ).with_columns(pl.lit(b).alias("budget"))
        )
    diag_a_ret = pl.concat(ret_parts)

    # mean mass + distinct kv positions per prompt, by sink_class
    diag_a_mass = df.group_by("sink_class").agg(
        pl.col("accumulated_mass").sum().alias("mass_sum"),
        pl.len().alias("mass_cnt"),
        pl.col("kv_pos").n_unique().alias("distinct_pos"),  # same across heads
    )

    # --- Diagnostic B: quartile membership by role ---
    q = ranked.with_columns((pl.col("rank") / pl.col("n_kv")).alias("pct"))
    diag_b = q.with_columns(
        (pl.col("pct") > 0.75).cast(pl.Int64).alias("bottomQ"),   # lowest 25% mass
        (pl.col("pct") <= 0.25).cast(pl.Int64).alias("topQ"),     # highest 25% mass
    ).group_by("role").agg(
        pl.col("bottomQ").sum().alias("bottom_sum"),
        pl.col("topQ").sum().alias("top_sum"),
        pl.len().alias("cnt"),
    )
    return diag_a_ret, diag_a_mass, diag_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path,
                    default=Path("/mnt/kvr/dumps/2026-WW20-pilot/ctx_16000"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/plan2"))
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.run_root / "*.parquet")))
    if args.max_files:
        files = files[: args.max_files]
    print(f"processing {len(files)} files from {args.run_root}")

    a_ret, a_mass, b = [], [], []
    for i, f in enumerate(files, 1):
        ra, ma, db = process_file(f)
        a_ret.append(ra); a_mass.append(ma); b.append(db)
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    n = len(files)

    # Diagnostic A: retention rate + mass + distinct positions per prompt
    ret = (
        pl.concat(a_ret).group_by(["sink_class", "budget"])
        .agg(pl.col("ret_sum").sum(), pl.col("cnt").sum())
        .with_columns((pl.col("ret_sum") / pl.col("cnt")).alias("retention_rate"))
    )
    mass = (
        pl.concat(a_mass).group_by("sink_class")
        .agg(pl.col("mass_sum").sum(), pl.col("mass_cnt").sum(),
             pl.col("distinct_pos").sum())
        .with_columns(
            (pl.col("mass_sum") / pl.col("mass_cnt")).alias("mean_mass"),
            (pl.col("distinct_pos") / n).alias("mean_positions_per_prompt"),
        )
    )
    diag_a = (
        ret.join(mass.select(["sink_class", "mean_mass", "mean_positions_per_prompt"]),
                 on="sink_class", how="left")
        .sort(["budget", "retention_rate"], descending=[False, True])
        .select(["sink_class", "budget", "retention_rate", "mean_mass",
                 "mean_positions_per_prompt", "cnt"])
    )
    out_a = args.out_dir / "diag_sink_bos_vs_delim.csv"
    diag_a.write_csv(out_a)

    # Diagnostic B: quartile fractions by role
    diag_b = (
        pl.concat(b).group_by("role")
        .agg(pl.col("bottom_sum").sum(), pl.col("top_sum").sum(), pl.col("cnt").sum())
        .with_columns(
            (pl.col("bottom_sum") / pl.col("cnt")).alias("frac_bottom_quartile"),
            (pl.col("top_sum") / pl.col("cnt")).alias("frac_top_quartile"),
        )
        .sort("frac_bottom_quartile", descending=True)
        .select(["role", "frac_bottom_quartile", "frac_top_quartile", "cnt"])
    )
    out_b = args.out_dir / "diag_value_quartile.csv"
    diag_b.write_csv(out_b)

    print(f"\n=== Diagnostic A — BOS vs DELIM sink (retention by budget) ===\n{diag_a}")
    print(f"\nwrote {out_a}")
    print(f"\n=== Diagnostic B — role x mass-quartile (baseline 0.25) ===\n{diag_b}")
    print(f"\nwrote {out_b}")


if __name__ == "__main__":
    main()
