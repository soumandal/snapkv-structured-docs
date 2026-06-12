"""Re-label token roles in existing pilot dumps using the current dispatcher.

The pilot's first run baked in the role column at recording time. We later
fixed the dispatcher to handle `{...json...}\n\nQuestion:` prompts (the
common shape — see kvr/structure/dispatcher.py::_json_prefix_end). This
script regenerates only the `role` / `depth` columns of each parquet by
re-deriving the prompt text from the seeded corpus, re-tokenizing, and
re-aligning. Accumulated mass values are untouched.

Workers run in spawned subprocesses (Polars retains its Rust arena between
read_parquet calls in-process; respawning lets memory return to the OS).
"""
import argparse
from datetime import date
from multiprocessing import get_context
from pathlib import Path

import polars as pl
from tqdm import tqdm

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_pilot_corpus
from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.dispatcher import label_structure


CONTEXT_BUCKETS = [8000, 16000, 32000]


def _truncate_or_pad_prompt(tokenizer, prompt, target_tokens: int) -> str | None:
    """Mirror of scripts/02_run_pilot.py::truncate_or_pad_prompt. Must stay
    in sync — both functions need to reproduce the same text per prompt."""
    enc = tokenizer(prompt.context + "\n\nQuestion: " + prompt.question, add_special_tokens=False)
    cur_tokens = len(enc["input_ids"])
    if cur_tokens < target_tokens * 0.7:
        return None
    if cur_tokens > target_tokens * 1.3:
        ratio = (target_tokens * 1.1) / cur_tokens
        new_ctx_len = int(len(prompt.context) * ratio)
        return prompt.context[:new_ctx_len] + "\n\nQuestion: " + prompt.question
    return prompt.context + "\n\nQuestion: " + prompt.question


def _relabel_one(args: tuple[str, int, int]) -> tuple[str, str]:
    """Worker: relabel one parquet. Returns (prompt_id, status)."""
    parquet_path, ctx_tokens, seed = args
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    p = Path(parquet_path)
    prompt_id = p.stem

    # Rebuild corpus and find the matching prompt.
    corpus = build_pilot_corpus(
        n_wikitable=200,
        n_synthetic=200,
        seed=seed,
        synthetic_target_tokens=ctx_tokens,
    )
    matches = [pr for pr in corpus if pr.id == prompt_id]
    if not matches:
        return (prompt_id, "no_corpus_match")
    prompt = matches[0]

    text = _truncate_or_pad_prompt(tok, prompt, ctx_tokens)
    if text is None:
        return (prompt_id, "skip_truncation")

    char_spans = label_structure(text)
    token_roles = align_char_spans_to_tokens(text, char_spans, tok)

    df = pl.read_parquet(parquet_path)
    n_kv_recorded = int(df["kv_pos"].max()) + 1
    if n_kv_recorded != len(token_roles):
        return (
            prompt_id,
            f"tokenizer_mismatch(recorded={n_kv_recorded}, regenerated={len(token_roles)})",
        )

    role_df = pl.DataFrame({
        "kv_pos": list(range(len(token_roles))),
        "_new_role": [tr.role.value for tr in token_roles],
        "_new_depth": [tr.depth for tr in token_roles],
    })
    df = (
        df.drop(["role", "depth"])
          .join(role_df, on="kv_pos", how="left")
          .rename({"_new_role": "role", "_new_depth": "depth"})
    )
    df.write_parquet(parquet_path)
    return (prompt_id, "ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None,
                    help="Path to pilot run root (default: current WW pilot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--maxtasks", type=int, default=15)
    args = ap.parse_args()

    if args.run_root is None:
        run_id = f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot"
        args.run_root = Path("/mnt/kvr/dumps") / run_id
    print(f"Relabeling under {args.run_root}")

    all_tasks: list[tuple[str, int, int]] = []
    for ctx_tokens in CONTEXT_BUCKETS:
        bucket_dir = args.run_root / f"ctx_{ctx_tokens}"
        if not bucket_dir.exists():
            continue
        for parquet in sorted(bucket_dir.glob("*.parquet")):
            all_tasks.append((str(parquet), ctx_tokens, args.seed))
    print(f"Tasks: {len(all_tasks)}")

    ctx = get_context("spawn")
    counters: dict[str, int] = {}
    failures: list[tuple[str, str]] = []
    with ctx.Pool(processes=args.processes, maxtasksperchild=args.maxtasks) as pool:
        for prompt_id, status in tqdm(
            pool.imap_unordered(_relabel_one, all_tasks),
            total=len(all_tasks),
            desc="relabel",
        ):
            counters[status] = counters.get(status, 0) + 1
            if status != "ok":
                failures.append((prompt_id, status))

    print("\nSummary:")
    for status, n in sorted(counters.items()):
        print(f"  {status}: {n}")
    if failures:
        print("\nFirst 20 failures:")
        for pid, st in failures[:20]:
            print(f"  {pid}: {st}")


if __name__ == "__main__":
    main()
