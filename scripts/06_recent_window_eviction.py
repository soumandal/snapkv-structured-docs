"""Experiment C — SnapKV-style recent-window eviction.

For each prompt and observation-window size W, run a windowed-mass recorder
prefill pass to score every kv position by attention from the last W
queries only, then top-k retain by budget. Decode answer under
MaskedAttentionPatch with that keep mask. Grade.

Output: results/plan2/c_recent_window.csv with one row per
(prompt_id, window_size, budget_frac, ...).
"""
import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA
import torch.nn.functional as F  # noqa: F401 — used by _keep_mask_from_mass

from kvr.data.pilot_corpus import build_corpus_for_source, build_pilot_corpus
from kvr.eviction.masked_attention import MaskedAttentionPatch
from kvr.instrumentation.windowed_recorder import WindowedAttentionRecorder


# Observation-window sizes:
#   - 64: SnapKV's typical default (Li et al. 2024)
#   - 0.10/0.25/0.50 fractions of context: the plan's sweep
WINDOW_ABS = [64]
WINDOW_FRAC = [0.10, 0.25, 0.50]
# SnapKV-exact = windowed mass + 1D maxpool over KV (kernel=7) before top-k.
# Disabled by default to keep the existing C sweep clean; enabled via --snapkv.
SNAPKV_POOL_KERNEL = 7
SNAPKV_WINDOWS = [("snapkv_abs64", 64)]
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


def _keep_mask_from_mass(mass: torch.Tensor, budget_frac: float,
                         pool_kernel: int = 1) -> torch.Tensor:
    n_layers, n_heads, n_kv = mass.shape
    if pool_kernel > 1:
        # SnapKV: 1D max-pool over KV with stride=1, padding=(k-1)//2 so the
        # pooled tensor has the same length as the original. Captures neighbor
        # contributions to the importance score.
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
    ap.add_argument("--ctx-tokens", type=int, default=8000, choices=[8000, 16000, 32000])
    ap.add_argument("--max-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-path", type=Path,
                    default=Path("results/plan2/c_recent_window.csv"))
    ap.add_argument("--source-filter", default="synthetic_json")
    ap.add_argument("--snapkv", action="store_true",
                    help="Add SnapKV-exact window specs (windowed mass + maxpool=7).")
    args = ap.parse_args()

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
        "prompt_id", "source", "ctx_tokens", "window_spec", "window_tokens",
        "budget_frac", "prompt_len_tokens", "gold", "model_answer", "correct",
        "elapsed_sec",
    ]
    csv_fp = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    skipped = 0
    for prompt in tqdm(corpus, desc=f"ctx={args.ctx_tokens}"):
        text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
        if text is None:
            skipped += 1
            continue
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_len = int(enc["input_ids"].shape[-1])

        # Build (name, window_tokens, pool_kernel) list. pool_kernel=1 = no pool
        # (existing abs/frac specs); 7 = SnapKV-exact maxpool over KV.
        window_specs: list[tuple[str, int, int]] = [
            (f"abs{w}", w, 1) for w in WINDOW_ABS
        ] + [
            (f"frac{f:.2f}", max(1, int(round(f * prompt_len))), 1) for f in WINDOW_FRAC
        ]
        if args.snapkv:
            # Exclusive: --snapkv means *only* the SnapKV-exact specs. Existing
            # abs/frac data should already be on disk from the default run; the
            # head-to-head join happens at analysis time.
            window_specs = [(name, w, SNAPKV_POOL_KERNEL) for name, w in SNAPKV_WINDOWS]

        for window_spec, window_tokens, pool_kernel in window_specs:
            # 1) Record windowed mass for this W.
            recorder = WindowedAttentionRecorder(model, window_size=window_tokens)
            recorder.start()
            try:
                with torch.inference_mode():
                    model(**enc)
                mass = recorder.stop()  # CPU fp32
            finally:
                recorder.reset()
            mass_gpu = mass.to("cuda")

            for budget_frac in BUDGETS:
                keep = _keep_mask_from_mass(mass_gpu, budget_frac, pool_kernel=pool_kernel)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    with MaskedAttentionPatch(model, keep):
                        gen_ids = model.generate(
                            **enc, max_new_tokens=MAX_NEW_TOKENS,
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
                    "window_spec": window_spec,
                    "window_tokens": window_tokens,
                    "budget_frac": budget_frac,
                    "prompt_len_tokens": prompt_len,
                    "gold": prompt.answer,
                    "model_answer": model_answer.replace("\n", " ").strip()[:200],
                    "correct": correct,
                    "elapsed_sec": round(t1 - t0, 3),
                })
                csv_fp.flush()
                del keep
            del mass, mass_gpu

        del enc
        torch.cuda.empty_cache()

    csv_fp.close()
    print(f"Done. {skipped} prompts skipped.")


if __name__ == "__main__":
    main()
