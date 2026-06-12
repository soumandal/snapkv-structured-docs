"""Experiment A — role-conditional budget allocation.

For each (prompt, policy, budget) cell, build a per-(layer, head) keep mask
that allocates the total budget across role buckets per a fixed policy
α: role → fraction. Run prefill + decode under that mask, grade against gold.

Reuses the pilot parquet dumps for per-(layer, head, kv_pos) mass and roles
(same as scripts/05). Reuses MaskedAttentionPatch for the decode-time mask.

Output: results/plan2/a_role_conditional.csv, one row per cell.

The policy grid is hand-picked to test the hypotheses surfaced by E:
  * `mass-proportional`: each role gets the share of total mass it accumulated
    on this prompt (rough control — should approximate H2O's "All").
  * `no-key`: KEY share goes to zero; redistribute proportionally to VALUE.
  * `value-only`: VALUE takes everything (extreme of `no-key`).
  * `value-heavy-with-delim-anchor`: VALUE dominant + a small DELIM anchor
    (tests whether some DELIM retention is needed for routing).
  * `value-heavy-with-bos-anchor`: VALUE-only above token 0, plus position 0
    retained unconditionally (StreamingLLM-style).
  * `uniform-no-key`: equal share to {VALUE, DELIM, PROSE, WS}; KEY=0.
  * `equal-content-vs-delim`: half VALUE, half DELIM (the two roles E says
    *can* be load-bearing).
  * `h2o-via-uniform`: equal share to every role present (a coarse control —
    if this matches H2O, the role labels don't add information; if it loses,
    they do).
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_corpus_for_source, build_pilot_corpus
from kvr.eviction.masked_attention import MaskedAttentionPatch
from kvr.eviction.role_conditional import keep_mask_for_role_allocation


# Policies are static role→weight dicts. `mass-proportional` is computed
# per-prompt from the recorded mass and inserted at runtime; the placeholder
# below is consumed by the driver, not by the allocation function.
POLICIES: dict[str, dict[str, float] | str] = {
    "mass-proportional": "__mass_proportional__",
    "value-only": {"VALUE": 1.0},
    "no-key": {"VALUE": 0.7, "DELIM": 0.2, "PROSE": 0.05, "WS": 0.05},
    "value-heavy-delim-anchor": {"VALUE": 0.8, "DELIM": 0.2},
    "uniform-no-key": {"VALUE": 0.4, "DELIM": 0.3, "PROSE": 0.2, "WS": 0.1},
    "equal-value-delim": {"VALUE": 0.5, "DELIM": 0.5},
    "uniform-all-roles": {
        "VALUE": 0.2, "KEY": 0.2, "DELIM": 0.2, "PROSE": 0.2, "WS": 0.2,
    },
}
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
MAX_NEW_TOKENS = 32


def _truncate_or_pad_prompt(tokenizer, prompt, target_tokens: int) -> str | None:
    """Mirror of scripts/05's helper (which mirrors 02_run_pilot)."""
    enc = tokenizer(prompt.context + "\n\nQuestion: " + prompt.question, add_special_tokens=False)
    cur_tokens = len(enc["input_ids"])
    if cur_tokens < target_tokens * 0.7:
        return None
    if cur_tokens > target_tokens * 1.3:
        ratio = (target_tokens * 1.1) / cur_tokens
        new_ctx_len = int(len(prompt.context) * ratio)
        return prompt.context[:new_ctx_len] + "\n\nQuestion: " + prompt.question
    return prompt.context + "\n\nQuestion: " + prompt.question


def _load_mass_and_roles(parquet_path: Path) -> tuple[torch.Tensor, list[str]]:
    df = pl.read_parquet(parquet_path).sort(["layer", "head", "kv_pos"])
    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1
    n_kv = int(df["kv_pos"].max()) + 1
    if len(df) != n_layers * n_heads * n_kv:
        raise RuntimeError(
            f"parquet has {len(df)} rows but expected {n_layers}×{n_heads}×{n_kv}"
        )
    mass_flat = df["accumulated_mass"].to_numpy()
    mass = torch.from_numpy(mass_flat.reshape(n_layers, n_heads, n_kv).copy()).float()
    roles_df = (
        df.filter((pl.col("layer") == 0) & (pl.col("head") == 0))
          .sort("kv_pos")
          .select(["kv_pos", "role"])
    )
    roles = roles_df["role"].to_list()
    return mass, roles


def _mass_proportional_allocation(
    mass: torch.Tensor, roles: list[str]
) -> dict[str, float]:
    """Per-role share of total accumulated mass on this prompt.

    Sum over (layer, head, kv_pos) within each role. Roles with zero
    occurrence in `roles` are absent from the dict.
    """
    role_to_total: dict[str, float] = defaultdict(float)
    total = mass.sum().item()
    if total <= 0:
        return {}
    flat = mass.sum(dim=(0, 1))  # (n_kv,)
    for i, r in enumerate(roles):
        role_to_total[r] += float(flat[i].item())
    return {r: v / total for r, v in role_to_total.items() if v > 0}


def _grade(model_answer: str, gold: str) -> int:
    return int(gold.lower().strip() in model_answer.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--ctx-tokens", type=int, default=8000, choices=[8000, 16000, 32000])
    ap.add_argument("--max-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-path", type=Path,
                    default=Path("results/plan2/a_role_conditional.csv"))
    ap.add_argument("--source-filter", default="synthetic_json")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES.keys()),
                    help="Subset of policy names to run")
    ap.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    if args.run_root is None:
        run_id = f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot"
        args.run_root = Path("/mnt/kvr/dumps") / run_id
        if not args.run_root.exists():
            args.run_root = Path("/mnt/kvr/dumps/2026-WW20-pilot")

    bucket_dir = args.run_root / f"ctx_{args.ctx_tokens}"
    if not bucket_dir.exists():
        sys.exit(f"No bucket dir: {bucket_dir}")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    unknown = [p for p in args.policies if p not in POLICIES]
    if unknown:
        sys.exit(f"Unknown policies: {unknown}. Known: {list(POLICIES)}")

    print(f"Loading {MODEL_LLAMA}...")
    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LLAMA,
        torch_dtype=torch.float16,
        device_map="cuda",
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
            n_wikitable=args.max_prompts,
            n_synthetic=args.max_prompts,
            seed=args.seed,
            synthetic_target_tokens=args.ctx_tokens,
        )
    corpus = corpus[: args.max_prompts]

    fieldnames = [
        "prompt_id", "source", "ctx_tokens", "policy", "budget_frac",
        "prompt_len_tokens", "gold", "model_answer", "correct", "elapsed_sec",
        "allocation_summary",
    ]
    write_header = not args.out_path.exists()
    csv_fp = open(args.out_path, "a", newline="")
    writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    n_rows = 0
    skipped = 0
    for prompt in tqdm(corpus, desc=f"A ctx={args.ctx_tokens}"):
        parquet_path = bucket_dir / f"{prompt.id}.parquet"
        if not parquet_path.exists():
            skipped += 1
            continue
        text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
        if text is None:
            skipped += 1
            continue

        try:
            mass, roles = _load_mass_and_roles(parquet_path)
        except Exception as e:
            print(f"  skip {prompt.id}: load error {e}")
            skipped += 1
            continue

        mass_gpu = mass.to("cuda")
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_len = int(enc["input_ids"].shape[-1])
        if prompt_len != mass.shape[-1]:
            print(f"  skip {prompt.id}: token len {prompt_len} != recorded {mass.shape[-1]}")
            skipped += 1
            del mass_gpu
            torch.cuda.empty_cache()
            continue

        # Resolve mass-proportional allocation once per prompt.
        resolved_policies = {}
        for name in args.policies:
            policy = POLICIES[name]
            if policy == "__mass_proportional__":
                resolved_policies[name] = _mass_proportional_allocation(mass, roles)
            else:
                resolved_policies[name] = dict(policy)

        for policy_name in args.policies:
            allocation = resolved_policies[policy_name]
            alloc_summary = ",".join(
                f"{r}={v:.2f}" for r, v in sorted(allocation.items(), key=lambda x: -x[1])
            )
            for budget_frac in args.budgets:
                keep = keep_mask_for_role_allocation(
                    mass_gpu, roles, budget_frac, allocation
                )
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
                    "policy": policy_name,
                    "budget_frac": budget_frac,
                    "prompt_len_tokens": prompt_len,
                    "gold": prompt.answer,
                    "model_answer": model_answer.replace("\n", " ").strip()[:200],
                    "correct": correct,
                    "elapsed_sec": round(t1 - t0, 3),
                    "allocation_summary": alloc_summary,
                }
                writer.writerow(row)
                csv_fp.flush()
                n_rows += 1
                del keep
        del mass_gpu, enc
        torch.cuda.empty_cache()

    csv_fp.close()
    print(f"\nDone. {n_rows} rows written to {args.out_path}; {skipped} prompts skipped.")


if __name__ == "__main__":
    main()
