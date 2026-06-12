"""Diagnostic: how much softmax weight does the LS summary slot actually get?

Runs a small handful of (prompt, budget) cells with
`LSMaskedAttentionPatch(collect_summary_weights=True)` and reports the
per-layer aggregate weight on the summary slot across decode steps. If
those weights are ~0, the decoder is effectively ignoring the summary —
explaining why H2O+LS gives identical EM to plain H2O.
"""
import argparse
import sys
from pathlib import Path

import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_corpus_for_source
from kvr.eviction.ls_masked_attention import LSMaskedAttentionPatch


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


def _load_mass(parquet_path: Path) -> torch.Tensor:
    df = pl.read_parquet(parquet_path).sort(["layer", "head", "kv_pos"])
    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1
    n_kv = int(df["kv_pos"].max()) + 1
    return torch.from_numpy(
        df["accumulated_mass"].to_numpy().reshape(n_layers, n_heads, n_kv).copy()
    ).float()


def _h2o_keep_mask(mass: torch.Tensor, budget_frac: float) -> torch.Tensor:
    n_layers, n_heads, n_kv = mass.shape
    k = max(1, min(int(round(budget_frac * n_kv)), n_kv))
    _, top_idx = torch.topk(mass, k=k, dim=-1, largest=True, sorted=False)
    keep = torch.zeros(n_layers, n_heads, n_kv, dtype=torch.bool, device=mass.device)
    keep.scatter_(dim=-1, index=top_idx, value=True)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path,
                    default=Path("/mnt/kvr/dumps/2026-WW20-pilot"))
    ap.add_argument("--ctx-tokens", type=int, default=8000)
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="Which synthetic_json prompt to diagnose.")
    ap.add_argument("--budgets", default="0.05,0.20,0.50,0.75",
                    help="Comma-separated budgets to test.")
    ap.add_argument("--anchor-strategy",
                    default="mean_evicted_pos",
                    choices=["post_rope_mean", "mean_evicted_pos",
                             "content_weighted", "kmeans"])
    ap.add_argument("--ls-rank", type=int, default=1)
    args = ap.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]

    bucket = args.run_root / f"ctx_{args.ctx_tokens}"

    print(f"Loading {MODEL_LLAMA}...")
    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LLAMA, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    corpus = build_corpus_for_source(
        "synthetic_json", n_prompts=args.prompt_index + 1, seed=0,
        target_tokens=args.ctx_tokens,
    )
    prompt = corpus[args.prompt_index]
    text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
    if text is None:
        sys.exit("prompt too short for ctx bucket")
    parquet_path = bucket / f"{prompt.id}.parquet"
    mass = _load_mass(parquet_path).to("cuda")
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

    for budget in budgets:
        keep = _h2o_keep_mask(mass, budget)
        with torch.inference_mode():
            patch = LSMaskedAttentionPatch(model, keep, ls_rank=args.ls_rank,
                                            anchor_strategy=args.anchor_strategy,
                                            collect_summary_weights=True)
            with patch:
                model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            stats = patch.summary_weight_stats()

        # Aggregate across layers.
        means = [stats[l]["mean"] for l in sorted(stats)]
        maxes = [stats[l]["max"] for l in sorted(stats)]
        head_max_means = [stats[l]["head_max_mean"] for l in sorted(stats)]
        n_steps = stats[sorted(stats)[0]]["n_steps"]
        n_heads = stats[sorted(stats)[0]]["n_heads"]

        print(f"\n=== prompt {prompt.id}, budget={budget}, anchor={args.anchor_strategy}, "
              f"ls_rank={args.ls_rank}, decode_steps={n_steps}, n_heads={n_heads} ===")
        print(f"global mean summary weight: {sum(means)/len(means):.6e}")
        print(f"global max  summary weight: {max(maxes):.6e}")
        print(f"global mean of per-head-max: {sum(head_max_means)/len(head_max_means):.6e}")
        # Highlight any layer where the summary slot got noticeable mass.
        loud = [l for l in sorted(stats) if stats[l]["head_max_mean"] > 0.01]
        if loud:
            print(f"layers with head_max_mean > 0.01: {loud}")
            for l in loud[:5]:
                s = stats[l]
                print(f"  layer {l}: mean={s['mean']:.4f} "
                      f"max={s['max']:.4f} head_max_mean={s['head_max_mean']:.4f}")
        else:
            print("no layer has any head with appreciable (>1%) summary attention.")

        # Magnitude diagnostic: ‖k_summary‖ vs mean ‖k_kept‖.
        norm = patch.norm_stats()
        if norm:
            ratios = [norm[l]["ratio_mean"] for l in sorted(norm)]
            sum_norms = [norm[l]["summary_norm_mean"] for l in sorted(norm)]
            kept_norms = [norm[l]["kept_norm_mean"] for l in sorted(norm)]
            print(f"norm ratio (k_summary/k_kept) — mean across layers: "
                  f"{sum(ratios)/len(ratios):.4f}  "
                  f"min layer: {min(ratios):.4f}  max layer: {max(ratios):.4f}")
            print(f"  mean ‖k_summary‖ across layers: "
                  f"{sum(sum_norms)/len(sum_norms):.4f}")
            print(f"  mean ‖k_kept‖    across layers: "
                  f"{sum(kept_norms)/len(kept_norms):.4f}")


if __name__ == "__main__":
    main()
