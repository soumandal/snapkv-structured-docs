"""Principled-eviction sweep — Method B (entropy-discount) + VPE + H2O baseline.

For each prompt, runs ONE forward pass via `ScoringSignalsRecorder` to
capture (mass, plogp, v_norm) per (layer, head, kv_pos). Then for each
(scorer, α, smooth_kernel) configuration, computes the score, builds a
top-k keep mask for each budget, installs it via MaskedAttentionPatch,
decodes, grades.

Cells per prompt with default grid:
  h2o:     1 (α=0)
  entropy: |alphas|
  vpe:     |alphas| × |smooth_kernels|

Output: results/plan2/b_principled_eviction.csv, one row per
(prompt_id, source, ctx_tokens, scorer, alpha, smooth_kernel, budget_frac).
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
from kvr.data.pilot_corpus import build_corpus_for_source, build_pilot_corpus
from kvr.eviction.masked_attention import MaskedAttentionPatch
from kvr.instrumentation.scoring_recorder import (
    ScoringSignalsRecorder,
    entropy_score,
    value_prior_score,
)


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


def _topk_keep_mask(score: torch.Tensor, budget_frac: float) -> torch.Tensor:
    n_layers, n_heads, n_kv = score.shape
    k = max(1, min(int(round(budget_frac * n_kv)), n_kv))
    _, top_idx = torch.topk(score, k=k, dim=-1, largest=True, sorted=False)
    keep = torch.zeros(n_layers, n_heads, n_kv, dtype=torch.bool, device=score.device)
    keep.scatter_(dim=-1, index=top_idx, value=True)
    return keep


def _expand_configs(scorers: list[str],
                    alphas: list[float],
                    smooth_kernels: list[int]) -> list[tuple[str, float, int]]:
    """Yield (scorer, alpha, smooth_kernel) cells. h2o ignores alpha & kernel."""
    out: list[tuple[str, float, int]] = []
    if "h2o" in scorers:
        out.append(("h2o", 0.0, 1))
    if "entropy" in scorers:
        for a in alphas:
            out.append(("entropy", a, 1))
    if "vpe" in scorers:
        for a in alphas:
            for kk in smooth_kernels:
                out.append(("vpe", a, kk))
    return out


def _compute_score(scorer: str, alpha: float, smooth_kernel: int,
                   mass: torch.Tensor, plogp: torch.Tensor,
                   v_norm: torch.Tensor) -> torch.Tensor:
    if scorer == "h2o":
        return mass
    if scorer == "entropy":
        return entropy_score(mass, plogp, alpha=alpha)
    if scorer == "vpe":
        return value_prior_score(mass, v_norm, alpha=alpha, smooth_kernel=smooth_kernel)
    raise ValueError(f"unknown scorer: {scorer!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-tokens", type=int, default=8000, choices=[8000, 16000, 32000])
    ap.add_argument("--max-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-filter", default="synthetic_json")
    ap.add_argument("--out-path", type=Path,
                    default=Path("results/plan2/b_principled_eviction.csv"))
    ap.add_argument("--scorers", default="h2o,entropy,vpe",
                    help="Comma-separated subset of {h2o,entropy,vpe}.")
    ap.add_argument("--alphas", default="0.5,1.0,2.0,4.0",
                    help="Comma-separated α values for entropy and vpe scorers.")
    ap.add_argument("--vpe-smooth-kernels", default="1,7",
                    help="Comma-separated odd ints; only used by the vpe scorer.")
    args = ap.parse_args()

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    smooth_kernels = [int(k) for k in args.vpe_smooth_kernels.split(",") if k.strip()]
    configs = _expand_configs(scorers, alphas, smooth_kernels)
    print(f"Running {len(configs)} configs × {len(BUDGETS)} budgets = "
          f"{len(configs) * len(BUDGETS)} cells per prompt")

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
        "prompt_id", "source", "ctx_tokens", "scorer", "alpha", "smooth_kernel",
        "budget_frac", "prompt_len_tokens", "gold", "model_answer", "correct",
        "recorder_sec", "decode_sec",
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

        # One prefill → all three signals.
        recorder = ScoringSignalsRecorder(model)
        recorder.start()
        t_rec = time.perf_counter()
        try:
            with torch.inference_mode():
                model(**enc)
            mass_cpu, plogp_cpu, v_norm_cpu = recorder.stop()
        finally:
            recorder.reset()
        recorder_sec = round(time.perf_counter() - t_rec, 3)

        mass = mass_cpu.to("cuda")
        plogp = plogp_cpu.to("cuda")
        v_norm = v_norm_cpu.to("cuda")

        for scorer, alpha, smooth_kernel in configs:
            score = _compute_score(scorer, alpha, smooth_kernel, mass, plogp, v_norm)
            for budget_frac in BUDGETS:
                keep = _topk_keep_mask(score, budget_frac)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    with MaskedAttentionPatch(model, keep):
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
                    "scorer": scorer,
                    "alpha": alpha,
                    "smooth_kernel": smooth_kernel,
                    "budget_frac": budget_frac,
                    "prompt_len_tokens": prompt_len,
                    "gold": prompt.answer,
                    "model_answer": model_answer.replace("\n", " ").strip()[:200],
                    "correct": correct,
                    "recorder_sec": recorder_sec,
                    "decode_sec": decode_sec,
                })
                csv_fp.flush()
                del keep
            del score

        del mass, plogp, v_norm, mass_cpu, plogp_cpu, v_norm_cpu, enc
        torch.cuda.empty_cache()

    csv_fp.close()
    print(f"Done. {skipped} prompts skipped.")


if __name__ == "__main__":
    main()
