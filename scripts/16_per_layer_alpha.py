"""Per-layer α_KEY policy via greedy coordinate descent (TODO #24, Plan 2 §D).

Background: the 2026-05-20 α-sweep showed the optimum is bimodal in α — α=0
wins on cleanly-separable indented JSON at ctx 16k, α=0.02 wins everywhere
else. Per-layer policy asks whether a heterogeneous α vector (different α per
transformer layer) can beat the best uniform-α policy on the headline cell
(indented JSON ctx 16k, where the over-oracle margin lives).

Search procedure:
  1. Cache the SnapKV windowed-mass + role labels per prompt to disk once
     (the recorder pass is the dominant per-prompt cost; reusing the cache
     across coordinate-descent passes amortizes it).
  2. Start from a uniform initial α vector (default: α=0 across all 32 layers
     — the global optimum on this cell).
  3. For each layer L in `--layers` order, evaluate the candidate α values
     `--alpha-grid` (default {0, 0.02, 0.05, 0.10, 0.20}); keep the one with
     the highest mean EM at the chosen budget. Re-baseline (re-evaluate the
     current α for L) at each visit to absorb any earlier-layer drift.
  4. Repeat for `--max-passes` (default 1). Stop early if a full pass changes
     no layer.

Outputs:
  results/plan2/per_layer_alpha_eval.csv — one row per (prompt, layer, α_cand)
  results/plan2/per_layer_alpha_state.json — final α vector + EM trajectory

Smoke-test invocation (15 min on one A100):
  .venv/bin/python scripts/16_per_layer_alpha.py \
    --max-prompts 10 --layers 0 8 16 24 --max-passes 1

Full run (one pass, ~4.5h):
  .venv/bin/python scripts/16_per_layer_alpha.py --max-prompts 50
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_corpus_for_source
from kvr.eviction.masked_attention import MaskedAttentionPatch
from kvr.eviction.role_conditional import keep_mask_for_role_allocation
from kvr.instrumentation.windowed_recorder import WindowedAttentionRecorder
from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.dispatcher import label_structure

WINDOW_TOKENS = 64
POOL_KERNEL = 7
MAX_NEW_TOKENS = 32
# Matches POLICY_PRESETS no-key family in scripts/13: as α_KEY grows, only
# VALUE shrinks (DELIM/PROSE/WS stay fixed). So per-layer α=0.02 is identical
# to the published `no-key-soft02` global policy when applied uniformly.
NON_KEY_FIXED = {"DELIM": 0.20, "PROSE": 0.05, "WS": 0.05}
VALUE_BASE = 1.0 - sum(NON_KEY_FIXED.values())  # = 0.70


def _truncate_or_pad_prompt(tokenizer, prompt, target_tokens):
    enc = tokenizer(prompt.context + "\n\nQuestion: " + prompt.question, add_special_tokens=False)
    cur = len(enc["input_ids"])
    if cur < target_tokens * 0.7:
        return None
    if cur > target_tokens * 1.3:
        ratio = (target_tokens * 1.1) / cur
        new_ctx_len = int(len(prompt.context) * ratio)
        return prompt.context[:new_ctx_len] + "\n\nQuestion: " + prompt.question
    return prompt.context + "\n\nQuestion: " + prompt.question


def _compute_roles(text, tokenizer):
    spans = label_structure(text)
    token_roles = align_char_spans_to_tokens(text, spans, tokenizer)
    return [tr.role.value for tr in token_roles]


def _maxpool_kv(mass, kernel):
    if kernel <= 1:
        return mass
    n_layers, n_heads, n_kv = mass.shape
    pad = (kernel - 1) // 2
    flat = mass.reshape(n_layers * n_heads, 1, n_kv)
    pooled = F.max_pool1d(flat, kernel_size=kernel, stride=1, padding=pad)
    return pooled.reshape(n_layers, n_heads, -1)[..., :n_kv]


def _grade(model_answer, gold):
    return int(gold.lower().strip() in model_answer.lower())


def _alpha_to_alloc(alpha_key: float) -> dict[str, float]:
    """Convert a scalar α_KEY to the role allocation dict matching the
    POLICY_PRESETS no-key family: α_VALUE = 0.70 − α_KEY, with DELIM/PROSE/WS
    fixed at their presets."""
    alloc: dict[str, float] = {"VALUE": VALUE_BASE - alpha_key, **NON_KEY_FIXED}
    if alpha_key > 0:
        alloc["KEY"] = alpha_key
    return alloc


def build_cache(args, tokenizer, model):
    """One pass over the corpus: cache (mass_pooled, roles, input_ids, gold) per prompt."""
    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus_for_source(
        args.source, n_prompts=args.max_prompts, seed=args.seed,
        target_tokens=args.ctx_tokens,
        tokenizer=tokenizer if args.source == "wikitable_long" else None,
    )[: args.max_prompts]

    cached_ids = []
    skipped = 0
    for prompt in tqdm(corpus, desc="cache"):
        cache_path = cache_dir / f"{prompt.id}.pt"
        if cache_path.exists():
            cached_ids.append(prompt.id)
            continue

        text = _truncate_or_pad_prompt(tokenizer, prompt, args.ctx_tokens)
        if text is None:
            skipped += 1
            continue

        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_len = int(enc["input_ids"].shape[-1])
        roles = _compute_roles(text, tokenizer)
        if prompt_len != len(roles):
            print(f"  skip {prompt.id}: tok len {prompt_len} != roles len {len(roles)}")
            skipped += 1
            del enc
            torch.cuda.empty_cache()
            continue

        recorder = WindowedAttentionRecorder(model, window_size=WINDOW_TOKENS)
        recorder.start()
        try:
            with torch.inference_mode():
                model(**enc)
            mass = recorder.stop().to("cuda")
        finally:
            recorder.reset()
        mass_pooled = _maxpool_kv(mass, POOL_KERNEL).to("cpu")

        torch.save({
            "mass_pooled": mass_pooled,
            "roles": roles,
            "input_ids": enc["input_ids"].to("cpu"),
            "gold": prompt.answer,
            "prompt_len": prompt_len,
        }, cache_path)
        cached_ids.append(prompt.id)
        del enc, mass
        torch.cuda.empty_cache()

    print(f"cache: {len(cached_ids)} prompts ready, {skipped} skipped")
    return cached_ids


def evaluate(args, tokenizer, model, cached_ids, alpha_vec, writer, pass_idx, layer, alpha_cand):
    """Evaluate a candidate α vector — mean EM over the cached prompts at the
    chosen budget. Writes one row per prompt to the eval CSV."""
    cache_dir: Path = args.cache_dir
    n_correct, n_total = 0, 0
    n_layers = len(alpha_vec)
    per_layer_alloc = [_alpha_to_alloc(a) for a in alpha_vec]
    for pid in cached_ids:
        rec = torch.load(cache_dir / f"{pid}.pt", weights_only=False)
        mass = rec["mass_pooled"].to("cuda")
        if mass.shape[0] != n_layers:
            raise RuntimeError(f"cached mass has {mass.shape[0]} layers; alpha_vec has {n_layers}")
        roles = rec["roles"]
        input_ids = rec["input_ids"].to("cuda")
        prompt_len = rec["prompt_len"]
        gold = rec["gold"]
        keep = keep_mask_for_role_allocation(mass, roles, args.budget, per_layer_alloc)
        with torch.inference_mode():
            with MaskedAttentionPatch(model, keep):
                gen_ids = model.generate(
                    input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
        new_ids = gen_ids[0, prompt_len:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True)
        correct = _grade(answer, gold)
        n_correct += correct
        n_total += 1
        writer.writerow({
            "pass": pass_idx, "layer": layer, "alpha_cand": alpha_cand,
            "prompt_id": pid, "correct": correct,
            "model_answer": answer.replace("\n", " ").strip()[:200],
            "gold": gold,
        })
        del mass, input_ids, keep
        torch.cuda.empty_cache()
    return n_correct / max(1, n_total)


def coordinate_descent(args, tokenizer, model, cached_ids):
    """Greedy descent: for each layer, try each α candidate, keep the best."""
    # Determine n_layers from any cached mass.
    sample = torch.load(args.cache_dir / f"{cached_ids[0]}.pt", weights_only=False)
    n_layers = sample["mass_pooled"].shape[0]
    layers_to_search = args.layers if args.layers else list(range(n_layers))
    alpha_vec = [args.init_alpha] * n_layers
    print(f"init alpha vec: uniform {args.init_alpha} across {n_layers} layers")

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    fp = open(out_csv, "a", newline="")
    writer = csv.DictWriter(fp, fieldnames=[
        "pass", "layer", "alpha_cand", "prompt_id", "correct", "model_answer", "gold",
    ])
    if write_header:
        writer.writeheader()

    trajectory = []
    for pass_idx in range(args.max_passes):
        print(f"\n=== pass {pass_idx} ===")
        changes = 0
        for layer in layers_to_search:
            cand_ems = {}
            for alpha_cand in args.alpha_grid:
                trial = list(alpha_vec)
                trial[layer] = alpha_cand
                t0 = time.perf_counter()
                em = evaluate(args, tokenizer, model, cached_ids, trial, writer, pass_idx, layer, alpha_cand)
                fp.flush()
                t1 = time.perf_counter()
                cand_ems[alpha_cand] = em
                print(f"  pass={pass_idx} layer={layer:2d} α={alpha_cand:>4} EM={em:.3f}  ({t1-t0:.1f}s)")
            # Pick the best; tie-break to the closer-to-current α to reduce churn.
            best_em = max(cand_ems.values())
            best_alphas = [a for a, em in cand_ems.items() if em == best_em]
            best = min(best_alphas, key=lambda a: abs(a - alpha_vec[layer]))
            old = alpha_vec[layer]
            if best != old:
                changes += 1
            if not args.no_greedy:
                alpha_vec[layer] = best
            trajectory.append({
                "pass": pass_idx, "layer": layer,
                "candidates": cand_ems, "chosen": best, "prev": old,
                "applied": not args.no_greedy,
            })
            mode = "greedy" if not args.no_greedy else "independent"
            print(f"  ⇒ layer {layer}: {old} → {best}  (EM {best_em:.3f}) [{mode}]")

        # Save state after each pass.
        state_path = args.out_state
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "alpha_vec": alpha_vec,
            "pass_idx": pass_idx,
            "trajectory": trajectory,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }, indent=2))
        if changes == 0:
            print(f"converged after pass {pass_idx} (0 layer changes)")
            break

    fp.close()
    return alpha_vec, trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic_json")
    ap.add_argument("--ctx-tokens", type=int, default=16000)
    ap.add_argument("--max-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--alpha-grid", nargs="+", type=float,
                    default=[0.0, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--init-alpha", type=float, default=0.0,
                    help="Initial uniform α_KEY across all layers.")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="Subset of layer indices to search (default: all).")
    ap.add_argument("--max-passes", type=int, default=1)
    ap.add_argument("--no-greedy", action="store_true",
                    help="Independent-perturbation mode: evaluate each layer's "
                         "candidates against the initial uniform-α baseline; "
                         "do not propagate flips. Produces a per-layer signal "
                         "map rather than a coordinate-descent solution.")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/mnt/kvr/per_layer_cache/synth_json_ctx16k"))
    ap.add_argument("--out-csv", type=Path,
                    default=Path("results/plan2/per_layer_alpha_eval.csv"))
    ap.add_argument("--out-state", type=Path,
                    default=Path("results/plan2/per_layer_alpha_state.json"))
    args = ap.parse_args()

    print(f"Loading {MODEL_LLAMA}...")
    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LLAMA, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    cached_ids = build_cache(args, tok, model)
    if not cached_ids:
        raise RuntimeError("no prompts cached; aborting")

    alpha_vec, trajectory = coordinate_descent(args, tok, model, cached_ids)
    print("\nfinal α vec:", alpha_vec)


if __name__ == "__main__":
    main()
