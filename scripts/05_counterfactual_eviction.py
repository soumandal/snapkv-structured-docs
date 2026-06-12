"""Experiment E — counterfactual eviction profiling.

For each (prompt, condition, budget) cell, run Llama-3.1-8B prefill on the
prompt, install a per-(layer, head) keep mask for that condition+budget,
decode up to 32 answer tokens, and grade against the gold leaf value.

Per-prompt mass + role labels are loaded from the pilot parquet dumps to
avoid re-running the recorder pass. Prefill cost is paid once per
(prompt, condition, budget) cell — there is no shared-cache optimization
in this v1 driver; rerunning prefill per condition is simpler and the
~10-40s prefill cost is the bulk of wall-clock time.

Output: results/plan2/e_counterfactual.csv with one row per
(prompt_id, source, ctx_tokens, condition, budget_frac, answer, gold, correct).
"""
import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA, MODEL_QWEN, MODEL_QWEN3, MODEL_MISTRAL, MODEL_PHI, MODEL_LLAMA_BASE
from kvr.data.pilot_corpus import build_corpus_for_source, build_pilot_corpus
from kvr.eviction.keep_mask import keep_mask_for_condition
from kvr.eviction.masked_attention import MaskedAttentionPatch

MODEL_IDS = {"llama": MODEL_LLAMA, "qwen": MODEL_QWEN, "qwen3": MODEL_QWEN3, "mistral": MODEL_MISTRAL, "phi": MODEL_PHI, "llama-base": MODEL_LLAMA_BASE}


CONDITIONS = ["All", "No-DELIM", "No-KEY", "No-VALUE"]
# Budget grid spans the regime where vLLM/TRT-LLM cares about eviction (5–20%),
# the elbow region (20–75%), and a full-budget anchor (1.0 ≈ no eviction).
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
MAX_NEW_TOKENS = 32


def _truncate_or_pad_prompt(tokenizer, prompt, target_tokens: int) -> str | None:
    """Mirror of scripts/02_run_pilot.py — must produce the same text as the
    pilot recording for the recorded mass to apply."""
    enc = tokenizer(prompt.context + "\n\nQuestion: " + prompt.question, add_special_tokens=False)
    cur_tokens = len(enc["input_ids"])
    if cur_tokens < target_tokens * 0.7:
        return None
    if cur_tokens > target_tokens * 1.3:
        ratio = (target_tokens * 1.1) / cur_tokens
        new_ctx_len = int(len(prompt.context) * ratio)
        return prompt.context[:new_ctx_len] + "\n\nQuestion: " + prompt.question
    return prompt.context + "\n\nQuestion: " + prompt.question


def _wrap_in_chat_template(tokenizer, raw_text: str) -> str:
    """Wrap raw user text in the tokenizer's chat template + assistant generation prompt.

    Required for ungraceful instruct models (Phi-3-mini, Qwen2.5). Must match
    the wrapping used by scripts/02 when the dumps were recorded, else the
    token-count assertion (prompt_len == mass.shape[-1]) will fail.
    """
    messages = [{"role": "user", "content": raw_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _load_mass_and_roles(parquet_path: Path) -> tuple[torch.Tensor, list[str]]:
    """Load (n_layers, n_heads, n_kv) mass tensor and per-kv role list."""
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
    mass = torch.from_numpy(mass_flat.reshape(n_layers, n_heads, n_kv).copy()).float()

    # Roles: one row per kv_pos (same across all layer × head — verify by taking layer=0,head=0).
    roles_df = (
        df.filter((pl.col("layer") == 0) & (pl.col("head") == 0))
          .sort("kv_pos")
          .select(["kv_pos", "role"])
    )
    roles = roles_df["role"].to_list()
    if len(roles) != n_kv:
        raise RuntimeError(f"role list length {len(roles)} != n_kv {n_kv}")
    return mass, roles


def _grade(model_answer: str, gold: str) -> int:
    """Robust grader: gold leaf value appears anywhere in the model output."""
    return int(gold.lower().strip() in model_answer.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None,
                    help="Pilot run root with bucketed parquet dumps")
    ap.add_argument("--ctx-tokens", type=int, default=8000,
                    choices=[8000, 16000, 32000])
    ap.add_argument("--max-prompts", type=int, default=200)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Resume from this prompt index (default 0). Set to "
                         "extend a prior run without re-doing earlier prompts; "
                         "rows are appended to --out-path.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-path", type=Path,
                    default=Path("results/plan2/e_counterfactual.csv"))
    ap.add_argument("--source-filter", default="synthetic_json",
                    help="Restrict to one corpus source (default: synthetic_json)")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--model", choices=list(MODEL_IDS), default="llama",
                    help="Which HF model to load (llama=Llama-3.1-8B-Instruct, "
                         "qwen=Qwen2.5-7B-Instruct, "
                         "mistral=Mistral-7B-Instruct-v0.3, "
                         "phi=Phi-3-mini-128k-instruct). Non-default "
                         "models need their own pilot dump root (passed via "
                         "--run-root, or auto-discovered as `<year>-WW<week>-"
                         "pilot-<model>`) since tokenizers differ.")
    ap.add_argument("--use-chat-template", action="store_true",
                    help="Wrap each prompt in the tokenizer's chat template + "
                         "assistant generation prefix before tokenizing. "
                         "Required for Phi / Qwen / other instruct models that "
                         "aren't graceful on raw text. Auto-discovered dump "
                         "root gets a `-ct` suffix when this is on, so it "
                         "picks up the matching scripts/02 dumps; the dumps "
                         "must have been recorded with --use-chat-template too "
                         "or the prompt_len == mass.shape[-1] check will fail.")
    args = ap.parse_args()
    model_id = MODEL_IDS[args.model]
    model_suffix = "" if args.model == "llama" else f"-{args.model}"
    if args.use_chat_template:
        model_suffix += "-ct"

    if args.run_root is None:
        run_id = f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot{model_suffix}"
        args.run_root = Path("/mnt/kvr/dumps") / run_id
        if not args.run_root.exists() and args.model == "llama":
            # Historic Llama pilot fallback. Non-llama models have no such
            # fallback — the caller must record dumps first.
            args.run_root = Path("/mnt/kvr/dumps/2026-WW20-pilot")

    bucket_dir = args.run_root / f"ctx_{args.ctx_tokens}"
    if not bucket_dir.exists():
        sys.exit(f"No bucket dir: {bucket_dir}")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    # Rebuild corpus deterministically. Single-source path (matches recorder
    # id scheme exactly, including ctx_tokens suffix for wikitable_long).
    n_corpus = args.start_idx + args.max_prompts
    if args.source_filter:
        corpus = build_corpus_for_source(
            args.source_filter,
            n_prompts=n_corpus,
            seed=args.seed,
            target_tokens=args.ctx_tokens,
            tokenizer=tok if args.source_filter == "wikitable_long" else None,
        )
    else:
        corpus = build_pilot_corpus(
            n_wikitable=n_corpus,
            n_synthetic=n_corpus,
            seed=args.seed,
            synthetic_target_tokens=args.ctx_tokens,
        )
    corpus = corpus[args.start_idx : args.start_idx + args.max_prompts]

    out_rows: list[dict] = []
    csv_path = args.out_path
    write_header = not csv_path.exists()
    fieldnames = [
        "prompt_id", "source", "ctx_tokens", "condition", "budget_frac",
        "prompt_len_tokens", "gold", "model_answer", "correct",
        "prefill_sec", "decode_sec",
    ]
    csv_fp = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    skipped = 0
    for prompt in tqdm(corpus, desc=f"ctx={args.ctx_tokens}"):
        parquet_path = bucket_dir / f"{prompt.id}.parquet"
        if not parquet_path.exists():
            skipped += 1
            continue
        text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
        if text is None:
            skipped += 1
            continue
        if args.use_chat_template:
            text = _wrap_in_chat_template(tok, text)

        try:
            mass, roles = _load_mass_and_roles(parquet_path)
        except Exception as e:
            print(f"  skip {prompt.id}: load error {e}")
            skipped += 1
            continue

        # Move mass to GPU once per prompt (used to compute keep_masks).
        mass_gpu = mass.to("cuda")

        # Tokenize once.
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_len = int(enc["input_ids"].shape[-1])
        if prompt_len != mass.shape[-1]:
            print(f"  skip {prompt.id}: token len {prompt_len} != recorded n_kv {mass.shape[-1]}")
            skipped += 1
            del mass_gpu
            torch.cuda.empty_cache()
            continue

        for condition in CONDITIONS:
            for budget_frac in BUDGETS:
                # Build keep mask on GPU.
                keep = keep_mask_for_condition(mass_gpu, roles, condition, budget_frac)

                # Fresh prefill + decode per cell.
                # (No cache reuse — simpler and the ~10s prefill dominates the
                # decode cost regardless.)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    with MaskedAttentionPatch(model, keep):
                        gen_ids = model.generate(
                            **enc,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tok.pad_token_id,
                        )
                t1 = time.perf_counter()
                new_ids = gen_ids[0, prompt_len:]
                model_answer = tok.decode(new_ids, skip_special_tokens=True)
                correct = _grade(model_answer, prompt.answer)
                row = {
                    "prompt_id": prompt.id,
                    "source": prompt.source,
                    "ctx_tokens": args.ctx_tokens,
                    "condition": condition,
                    "budget_frac": budget_frac,
                    "prompt_len_tokens": prompt_len,
                    "gold": prompt.answer,
                    "model_answer": model_answer.replace("\n", " ").strip()[:200],
                    "correct": correct,
                    "prefill_sec": round(t1 - t0, 3),  # generate covers prefill + decode together
                    "decode_sec": 0.0,
                }
                writer.writerow(row)
                csv_fp.flush()
                out_rows.append(row)

                del keep
        del mass_gpu, enc
        torch.cuda.empty_cache()

    csv_fp.close()
    print(f"\nDone. {len(out_rows)} rows written to {csv_path}; {skipped} prompts skipped.")


if __name__ == "__main__":
    main()
