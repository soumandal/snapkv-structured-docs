"""TODO #22 control — KEY-only-evicted at full budget B=1.0.

Drop every KEY token, keep every non-KEY token, no top-k by attention mass.
If EM ≈ 0.98 on indented `synthetic_json` ctx 16k, the eviction-as-denoising
story holds independently of budget: it is *which* tokens are removed, not
*how many*. Companion to the per-prompt scatter in TODO #22.

Output: results/plan2/key_only_evicted_ctx16k.csv (one row per prompt).
"""
import argparse
import csv
import time
from datetime import date
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_corpus_for_source
from kvr.eviction.masked_attention import MaskedAttentionPatch
from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.dispatcher import label_structure


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


def _load_roles(parquet_path: Path) -> list[str]:
    df = pl.read_parquet(parquet_path).sort(["layer", "head", "kv_pos"])
    roles_df = (
        df.filter((pl.col("layer") == 0) & (pl.col("head") == 0))
          .sort("kv_pos")
          .select(["kv_pos", "role"])
    )
    return roles_df["role"].to_list()


def _compute_roles(text: str, tokenizer) -> list[str]:
    spans = label_structure(text)
    token_roles = align_char_spans_to_tokens(text, spans, tokenizer)
    return [tr.role.value for tr in token_roles]


def _grade(model_answer: str, gold: str) -> int:
    return int(gold.lower().strip() in model_answer.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--ctx-tokens", type=int, default=16000, choices=[8000, 16000, 32000])
    ap.add_argument("--max-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-path", type=Path,
                    default=Path("results/plan2/key_only_evicted_ctx16k.csv"))
    ap.add_argument("--source-filter", default="synthetic_json")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    if args.run_root is None:
        run_id = f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot"
        args.run_root = Path("/mnt/kvr/dumps") / run_id
        if not args.run_root.exists():
            args.run_root = Path("/mnt/kvr/dumps/2026-WW20-pilot")
    bucket_dir = args.run_root / f"ctx_{args.ctx_tokens}"
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
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    corpus = build_corpus_for_source(
        args.source_filter,
        n_prompts=args.max_prompts,
        seed=args.seed,
        target_tokens=args.ctx_tokens,
        tokenizer=tok if args.source_filter == "wikitable_long" else None,
    )
    corpus = corpus[: args.max_prompts]

    fieldnames = [
        "prompt_id", "source", "ctx_tokens", "config", "budget_frac",
        "prompt_len_tokens", "n_key_tokens", "key_frac",
        "gold", "model_answer", "correct", "decode_sec",
    ]
    write_header = not args.out_path.exists()
    csv_fp = open(args.out_path, "a", newline="")
    writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    n_rows = 0
    skipped = 0
    for prompt in tqdm(corpus, desc=f"KEY-only-evicted ctx={args.ctx_tokens}"):
        parquet_path = bucket_dir / f"{prompt.id}.parquet"
        text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
        if text is None:
            skipped += 1
            continue

        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_len = int(enc["input_ids"].shape[-1])

        if parquet_path.exists():
            try:
                roles = _load_roles(parquet_path)
            except Exception as e:
                print(f"  skip {prompt.id}: roles load error {e}")
                skipped += 1
                del enc
                torch.cuda.empty_cache()
                continue
        else:
            roles = _compute_roles(text, tok)

        if prompt_len != len(roles):
            print(f"  skip {prompt.id}: tok len {prompt_len} != roles len {len(roles)}")
            skipped += 1
            del enc
            torch.cuda.empty_cache()
            continue

        # Drop every KEY token; keep everything else. Uniform across layers/heads.
        keep_1d = torch.tensor(
            [r != "KEY" for r in roles], dtype=torch.bool, device="cuda"
        )
        n_key = int((~keep_1d).sum().item())
        key_frac = n_key / prompt_len
        keep = keep_1d.view(1, 1, -1).expand(n_layers, n_heads, prompt_len).contiguous()

        t0 = time.perf_counter()
        with torch.inference_mode():
            with MaskedAttentionPatch(model, keep):
                gen_ids = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
        t1 = time.perf_counter()
        new_ids = gen_ids[0, prompt_len:]
        model_answer = tok.decode(new_ids, skip_special_tokens=True)
        correct = _grade(model_answer, prompt.answer)
        writer.writerow({
            "prompt_id": prompt.id,
            "source": prompt.source,
            "ctx_tokens": args.ctx_tokens,
            "config": "key-only-evicted",
            "budget_frac": 1.0,
            "prompt_len_tokens": prompt_len,
            "n_key_tokens": n_key,
            "key_frac": round(key_frac, 4),
            "gold": prompt.answer,
            "model_answer": model_answer.replace("\n", " ").strip()[:200],
            "correct": correct,
            "decode_sec": round(t1 - t0, 3),
        })
        csv_fp.flush()
        n_rows += 1
        del enc, keep, keep_1d
        torch.cuda.empty_cache()

    csv_fp.close()
    print(f"\nDone. {n_rows} rows written to {args.out_path}; {skipped} prompts skipped.")


if __name__ == "__main__":
    main()
