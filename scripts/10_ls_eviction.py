"""LS-projection eviction sweep — wraps a base selector (H2O for v0) with
a low-rank summary of evicted KV slots.

For each (prompt, budget):
  1. Load per-(layer, head, kv_pos) accumulated mass from the pilot parquet.
  2. Pick top-k positions by mass (H2O baseline) → keep_mask.
  3. Run prefill + decode under `LSMaskedAttentionPatch(keep_mask, ls_rank)`.
     During prefill the patch captures the mean K/V of the evicted block
     per (layer, head). During decode the model can attend to those
     summary slots in addition to the retained positions.
  4. Grade against the gold leaf value.

v0 fixes base = H2O (mass) and ls_rank = 1. If this shifts the budget=0.5
H2O number meaningfully above 0.14 on synthetic_json ctx=8k, scale to
SnapKV-scored bases and higher ls_rank.

Output: results/plan2/ls_eviction.csv.
"""
import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_corpus_for_source, build_pilot_corpus
from kvr.eviction.ls_masked_attention import LSMaskedAttentionPatch
from kvr.instrumentation.windowed_recorder import WindowedAttentionRecorder


BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
MAX_NEW_TOKENS = 32


def _truncate_or_pad_prompt(tokenizer, prompt, target_tokens: int) -> str | None:
    enc = tokenizer(prompt.context + "\n\nQuestion: " + prompt.question, add_special_tokens=False)
    cur_tokens = len(enc["input_ids"])
    if cur_tokens < target_tokens * 0.7:
        return None
    if cur_tokens > target_tokens * 1.3:
        ratio = (target_tokens * 1.1) / cur_tokens
        new_ctx_len = int(len(prompt.context) * ratio)
        return prompt.context[:new_ctx_len] + "\n\nQuestion: " + prompt.question
    return prompt.context + "\n\nQuestion: " + prompt.question


def _grade(answer: str, gold: str) -> int:
    return int(gold.lower().strip() in answer.lower())


def _load_mass(parquet_path: Path) -> torch.Tensor:
    """Load (n_layers, n_heads, n_kv) accumulated-mass tensor from a pilot dump."""
    df = pl.read_parquet(parquet_path).sort(["layer", "head", "kv_pos"])
    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1
    n_kv = int(df["kv_pos"].max()) + 1
    if len(df) != n_layers * n_heads * n_kv:
        raise RuntimeError(
            f"parquet has {len(df)} rows but expected "
            f"{n_layers}×{n_heads}×{n_kv}={n_layers*n_heads*n_kv}"
        )
    mass_flat = df["accumulated_mass"].to_numpy()
    return torch.from_numpy(mass_flat.reshape(n_layers, n_heads, n_kv).copy()).float()


def _keep_mask_from_mass(mass: torch.Tensor, budget_frac: float,
                         pool_kernel: int = 1) -> torch.Tensor:
    """Top-k by accumulated mass per (layer, head), with optional 1D maxpool
    over KV (SnapKV-exact recipe uses kernel=7).
    """
    n_layers, n_heads, n_kv = mass.shape
    if pool_kernel > 1:
        pad = (pool_kernel - 1) // 2
        flat = mass.reshape(n_layers * n_heads, 1, n_kv)
        pooled = F.max_pool1d(flat, kernel_size=pool_kernel, stride=1, padding=pad)
        scored = pooled.reshape(n_layers, n_heads, -1)[..., :n_kv]
    else:
        scored = mass
    k = max(1, min(int(round(budget_frac * n_kv)), n_kv))
    _, top_idx = torch.topk(scored, k=k, dim=-1, largest=True, sorted=False)
    keep = torch.zeros(n_layers, n_heads, n_kv, dtype=torch.bool, device=mass.device)
    keep.scatter_(dim=-1, index=top_idx, value=True)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None,
                    help="Pilot run root with bucketed parquet dumps")
    ap.add_argument("--ctx-tokens", type=int, default=8000,
                    choices=[8000, 16000, 32000])
    ap.add_argument("--max-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-filter", default="synthetic_json")
    ap.add_argument("--ls-rank", type=int, default=1,
                    help="Rank of the LS summary. >1 requires anchor_strategy=kmeans.")
    ap.add_argument("--anchor-strategy", default="mean_evicted_pos",
                    choices=["post_rope_mean", "mean_evicted_pos",
                             "content_weighted", "kmeans"])
    ap.add_argument("--base-selector", default="h2o", choices=["h2o", "snapkv"],
                    help="h2o = full-context accumulated mass from pilot parquet; "
                         "snapkv = windowed mass from a live recorder pass, then maxpool=7.")
    ap.add_argument("--snapkv-window", type=int, default=64)
    ap.add_argument("--snapkv-pool-kernel", type=int, default=7)
    ap.add_argument("--out-path", type=Path,
                    default=Path("results/plan2/ls_eviction.csv"))
    args = ap.parse_args()

    if args.run_root is None:
        run_id = f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot"
        args.run_root = Path("/mnt/kvr/dumps") / run_id
        if not args.run_root.exists():
            args.run_root = Path("/mnt/kvr/dumps/2026-WW20-pilot")
    bucket_dir = args.run_root / f"ctx_{args.ctx_tokens}"
    if args.base_selector == "h2o" and not bucket_dir.exists():
        sys.exit(f"No bucket dir: {bucket_dir}")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_LLAMA}...")
    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LLAMA, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    if args.source_filter:
        corpus = build_corpus_for_source(
            args.source_filter,
            n_prompts=args.max_prompts,
            seed=args.seed,
            target_tokens=args.ctx_tokens,
            tokenizer=tok if args.source_filter == "wikitable_long" else None,
        )
    else:
        corpus = build_pilot_corpus(
            n_wikitable=args.max_prompts, n_synthetic=args.max_prompts,
            seed=args.seed, synthetic_target_tokens=args.ctx_tokens,
        )
    corpus = corpus[: args.max_prompts]

    csv_path = args.out_path
    write_header = not csv_path.exists()
    fieldnames = [
        "prompt_id", "source", "ctx_tokens", "base_selector", "ls_rank",
        "anchor_strategy",
        "budget_frac", "prompt_len_tokens", "gold", "model_answer", "correct",
        "decode_sec",
    ]
    csv_fp = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    skipped = 0
    pool_kernel = args.snapkv_pool_kernel if args.base_selector == "snapkv" else 1
    for prompt in tqdm(corpus, desc=f"ctx={args.ctx_tokens}"):
        text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
        if text is None:
            skipped += 1
            continue
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_len = int(enc["input_ids"].shape[-1])

        if args.base_selector == "h2o":
            parquet_path = bucket_dir / f"{prompt.id}.parquet"
            if not parquet_path.exists():
                skipped += 1
                continue
            try:
                mass = _load_mass(parquet_path)
            except Exception as e:
                print(f"  skip {prompt.id}: {e}")
                skipped += 1
                continue
            if prompt_len != mass.shape[-1]:
                print(f"  skip {prompt.id}: token len {prompt_len} != recorded n_kv {mass.shape[-1]}")
                skipped += 1
                continue
            mass_gpu = mass.to("cuda")
        else:  # snapkv
            recorder = WindowedAttentionRecorder(model, window_size=args.snapkv_window)
            recorder.start()
            try:
                with torch.inference_mode():
                    model(**enc)
                mass = recorder.stop()
            finally:
                recorder.reset()
            mass_gpu = mass.to("cuda")

        for budget_frac in BUDGETS:
            keep = _keep_mask_from_mass(mass_gpu, budget_frac, pool_kernel=pool_kernel)
            t0 = time.perf_counter()
            with torch.inference_mode():
                with LSMaskedAttentionPatch(model, keep, ls_rank=args.ls_rank,
                                            anchor_strategy=args.anchor_strategy):
                    gen_ids = model.generate(
                        **enc, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, pad_token_id=tok.pad_token_id,
                    )
            decode_sec = round(time.perf_counter() - t0, 3)
            new_ids = gen_ids[0, prompt_len:]
            model_answer = tok.decode(new_ids, skip_special_tokens=True)
            correct = _grade(model_answer, prompt.answer)
            writer.writerow({
                "prompt_id": prompt.id,
                "source": prompt.source,
                "ctx_tokens": args.ctx_tokens,
                "base_selector": args.base_selector,
                "ls_rank": args.ls_rank,
                "anchor_strategy": args.anchor_strategy,
                "budget_frac": budget_frac,
                "prompt_len_tokens": prompt_len,
                "gold": prompt.answer,
                "model_answer": model_answer.replace("\n", " ").strip()[:200],
                "correct": correct,
                "decode_sec": decode_sec,
            })
            csv_fp.flush()
            del keep
        del mass, mass_gpu, enc
        if "recorder" in locals():
            del recorder
        torch.cuda.empty_cache()

    csv_fp.close()
    print(f"Done. {skipped} prompts skipped.")


if __name__ == "__main__":
    main()
