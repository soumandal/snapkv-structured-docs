"""Experiment A × C — SnapKV-style scoring with role-conditional allocation.

The two methods that worked so far address orthogonal failure modes:
  * SnapKV (C, `snapkv_abs64`): windowed mass over last 64 queries + 1-D
    max-pool kernel 7 over kv axis before top-k. Addresses *position* bias
    (recency captures answer-relevant tokens).
  * A `no-key` allocation: per-role budget α with KEY share zeroed and the
    saved budget redistributed to VALUE/DELIM/PROSE/WS. Addresses *content*
    bias (H2O over-retains JSON KEY tokens at the expense of leaf VALUE).

This driver runs three configs per prompt to test composition:
  1. snapkv-only           = windowed + maxpool=7 + plain top-k (replicates C)
  2. snapkv-no-key         = windowed + maxpool=7 + role-cond(no-key)
  3. snapkv-no-key-nomax   = windowed (no pool)  + role-cond(no-key)

The recorder pass happens once per prompt; the three configs reuse its
output. Per-prompt wall clock ≈ recorder(~5s) + 3 configs × 5 budgets × 2s.

Output: results/plan2/ac_combined.csv.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA, MODEL_QWEN, MODEL_QWEN3, MODEL_MISTRAL, MODEL_PHI, MODEL_LLAMA_BASE

MODEL_IDS = {"llama": MODEL_LLAMA, "qwen": MODEL_QWEN, "qwen3": MODEL_QWEN3, "mistral": MODEL_MISTRAL, "phi": MODEL_PHI, "llama-base": MODEL_LLAMA_BASE}
from kvr.data.pilot_corpus import build_corpus_for_source, build_pilot_corpus
from kvr.eviction.masked_attention import MaskedAttentionPatch
from kvr.eviction.role_conditional import keep_mask_for_role_allocation
from kvr.instrumentation.windowed_recorder import WindowedAttentionRecorder
from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.dispatcher import label_structure
from kvr.structure.roles import Role, set_role_precedence


WINDOW_TOKENS = 64
POOL_KERNEL = 7
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
MAX_NEW_TOKENS = 32

# Allocation presets. `no-key` is the A-sweep winner. `no-key-soft` adds a
# small α_KEY share to probe whether the compact/XML collapse is a binary-zero
# artifact rather than a true loss of the role signal (TODO #20).
# `fair-rate` is a sentinel — allocation is built per-prompt from the actual
# role distribution (α_r = n_r / n), the Chen-style fair-eviction baseline
# (TODO #23). It uniformly retains fraction B of each role's tokens.
POLICY_PRESETS = {
    "no-key":        {"VALUE": 0.70,                 "DELIM": 0.20, "PROSE": 0.05, "WS": 0.05},
    "no-key-soft02": {"VALUE": 0.68, "KEY": 0.02,    "DELIM": 0.20, "PROSE": 0.05, "WS": 0.05},
    "no-key-soft":   {"VALUE": 0.65, "KEY": 0.05,    "DELIM": 0.20, "PROSE": 0.05, "WS": 0.05},
    "no-key-soft10": {"VALUE": 0.60, "KEY": 0.10,    "DELIM": 0.20, "PROSE": 0.05, "WS": 0.05},
    "no-key-soft20": {"VALUE": 0.50, "KEY": 0.20,    "DELIM": 0.20, "PROSE": 0.05, "WS": 0.05},
    "fair-rate":     None,
}


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


def _wrap_in_chat_template(tokenizer, raw_text: str) -> str:
    """Wrap raw user text in the tokenizer's chat template + assistant generation prompt.

    Required for instruct models that aren't graceful on raw-text prompts
    (Qwen2.5, Phi-3-mini, etc.). Llama-3.1-Instruct and Mistral-Instruct-v0.3
    work without this but their EM may shift slightly when wrapped — quantify
    via a smoke test before relying on cross-template comparisons.
    """
    messages = [{"role": "user", "content": raw_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _load_roles(parquet_path: Path) -> list[str]:
    df = pl.read_parquet(parquet_path).sort(["layer", "head", "kv_pos"])
    roles_df = (
        df.filter((pl.col("layer") == 0) & (pl.col("head") == 0))
          .sort("kv_pos")
          .select(["kv_pos", "role"])
    )
    return roles_df["role"].to_list()


def _compute_roles(text: str, tokenizer) -> list[str]:
    """Compute per-token role labels at runtime via the structure dispatcher.

    Used for corpora without pre-recorded pilot parquet dumps (e.g.
    `synthetic_json_compact`). Mirrors the labeling step in scripts/03a.
    """
    spans = label_structure(text)
    token_roles = align_char_spans_to_tokens(text, spans, tokenizer)
    return [tr.role.value for tr in token_roles]


def _maxpool_kv(mass: torch.Tensor, kernel: int) -> torch.Tensor:
    """1-D max-pool over the kv axis with same-length output (SnapKV style)."""
    if kernel <= 1:
        return mass
    n_layers, n_heads, n_kv = mass.shape
    pad = (kernel - 1) // 2
    flat = mass.reshape(n_layers * n_heads, 1, n_kv)
    pooled = F.max_pool1d(flat, kernel_size=kernel, stride=1, padding=pad)
    return pooled.reshape(n_layers, n_heads, -1)[..., :n_kv]


def _keep_mask_topk(mass: torch.Tensor, budget_frac: float) -> torch.Tensor:
    n_layers, n_heads, n_kv = mass.shape
    k = max(1, min(int(round(budget_frac * n_kv)), n_kv))
    _, top_idx = torch.topk(mass, k=k, dim=-1, largest=True, sorted=False)
    keep = torch.zeros_like(mass, dtype=torch.bool)
    keep.scatter_(dim=-1, index=top_idx, value=True)
    return keep


def _grade(model_answer: str, gold: str) -> int:
    return int(gold.lower().strip() in model_answer.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None,
                    help="Pilot dumps root (for role labels).")
    ap.add_argument("--ctx-tokens", type=int, default=8000, choices=[8000, 16000, 32000])
    ap.add_argument("--max-prompts", type=int, default=50)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Resume from this prompt index (default 0). Set to "
                         "extend a prior run without re-doing earlier prompts; "
                         "rows are appended to --out-path.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-path", type=Path,
                    default=Path("results/plan2/ac_combined.csv"))
    ap.add_argument("--source-filter", default="synthetic_json")
    ap.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--pool-kernel", type=int, default=POOL_KERNEL,
                    help="1-D max-pool kernel size over the kv axis (SnapKV's "
                         "load-bearing op, C7). Default 7. Sweep {3,5,7,11,15} "
                         "for the C7 kernel-size ablation; kernel<=1 disables "
                         "pooling (equivalent to the -nomax config).")
    ap.add_argument("--policy", choices=list(POLICY_PRESETS), default="no-key",
                    help="Allocation preset for role-conditional configs.")
    ap.add_argument("--delim-priority", action="store_true",
                    help="Flip role_precedence so DELIM > KEY for merged tokens; "
                         "forces runtime labeling (ignores cached parquet roles).")
    ap.add_argument("--only-snapkv-only", action="store_true",
                    help="Emit ONLY the snapkv-only config (skip the two role "
                         "configs). For the C7 pool-kernel ablation, where the "
                         "snapkv-only EM-vs-kernel curve is the artifact.")
    ap.add_argument("--skip-snapkv-only", action="store_true",
                    help="Skip the snapkv-only config (use when the unchanged "
                         "baseline is already in the output CSV).")
    ap.add_argument("--use-chat-template", action="store_true",
                    help="Wrap the prompt in the tokenizer's chat template + "
                         "assistant generation prefix. Required for Qwen / Phi "
                         "/ other instruct models that aren't graceful on raw "
                         "text. Default off to preserve reproducibility of the "
                         "existing Llama / Mistral runs; turn on for new "
                         "cross-model work and quantify the shift via smoke "
                         "test. Forces runtime role labeling because the "
                         "template tokens don't align with cached parquet "
                         "roles.")
    ap.add_argument("--model", choices=list(MODEL_IDS), default="llama",
                    help="Which HF model to load (llama=Llama-3.1-8B-Instruct, "
                         "qwen=Qwen2.5-7B-Instruct, "
                         "qwen3=Qwen3-4B-Instruct-2507, "
                         "mistral=Mistral-7B-Instruct-v0.3, "
                         "phi=Phi-3-mini-128k-instruct). When passing "
                         "non-default, also pass --out-path to avoid colliding "
                         "with llama results. Phi, Qwen, and Qwen3 also require "
                         "--use-chat-template — they are not graceful on raw "
                         "text prompts.")
    args = ap.parse_args()
    model_id = MODEL_IDS[args.model]

    if args.delim_priority:
        # KEY=5, DELIM=2 originally; flip to DELIM=6 so it wins on merged
        # tokens that span both (compact JSON `,"` or XML `</tag>`).
        set_role_precedence(Role.DELIM, 6)
        print("delim-priority enabled: Role.DELIM precedence raised to 6 "
              "(originally 2; KEY remains 5).")

    # Config name suffix encodes the ablation knobs so all variants coexist
    # in ac_combined.csv without overwriting prior rows. fair-rate is its own
    # config family (not a no-key variant) so it uses a distinct base name.
    if args.policy == "fair-rate":
        cfg_base = "snapkv-fair-rate"
        cfg_base_nomax = "snapkv-fair-rate-nomax"
        suffix = ""
        if args.delim_priority:
            suffix += "-delimprio"
    else:
        cfg_base = "snapkv-no-key"
        cfg_base_nomax = "snapkv-no-key-nomax"
        suffix = ""
        if args.policy != "no-key":
            suffix += f"-{args.policy.removeprefix('no-key-')}"  # e.g. "-soft"
        if args.delim_priority:
            suffix += "-delimprio"

    static_allocation = POLICY_PRESETS[args.policy]

    if args.run_root is None:
        run_id = f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot"
        args.run_root = Path("/mnt/kvr/dumps") / run_id
        if not args.run_root.exists():
            args.run_root = Path("/mnt/kvr/dumps/2026-WW20-pilot")
    bucket_dir = args.run_root / f"ctx_{args.ctx_tokens}"
    if not bucket_dir.exists():
        # No pre-recorded dumps for this bucket — roles will be computed at runtime.
        print(f"Note: no bucket dir {bucket_dir}; using runtime role labeling.")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

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
            n_wikitable=n_corpus, n_synthetic=n_corpus,
            seed=args.seed, synthetic_target_tokens=args.ctx_tokens,
        )
    corpus = corpus[args.start_idx : args.start_idx + args.max_prompts]

    fieldnames = [
        "prompt_id", "source", "ctx_tokens", "config", "budget_frac",
        "prompt_len_tokens", "gold", "model_answer", "correct",
        "recorder_sec", "decode_sec", "pool_kernel",
    ]
    write_header = not args.out_path.exists()
    csv_fp = open(args.out_path, "a", newline="")
    writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    n_rows = 0
    skipped = 0
    for prompt in tqdm(corpus, desc=f"AC ctx={args.ctx_tokens}"):
        parquet_path = bucket_dir / f"{prompt.id}.parquet"
        # Runtime role labeling is the fallback when no pre-recorded dump exists.
        raw_text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
        if raw_text is None:
            skipped += 1
            continue
        text = _wrap_in_chat_template(tok, raw_text) if args.use_chat_template else raw_text

        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_len = int(enc["input_ids"].shape[-1])

        # Prefer pre-recorded role labels (from pilot dumps) for parity with
        # earlier runs; fall back to runtime labeling when no dump exists.
        # Under --delim-priority the cached labels are stale (they were dumped
        # with the original precedence) so we must re-label at runtime. Same
        # if the active model isn't the one used to record the dumps — the
        # tokenizers differ and the cached per-token roles wouldn't align.
        # --use-chat-template prepends/appends template tokens that don't
        # exist in the cached parquet roles, so it also forces re-labeling.
        use_cached = (
            parquet_path.exists()
            and not args.delim_priority
            and not args.use_chat_template
            and args.model == "llama"
        )
        if use_cached:
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

        # One recorder pass per prompt produces the windowed mass that all
        # three configs consume.
        t_rec0 = time.perf_counter()
        recorder = WindowedAttentionRecorder(model, window_size=WINDOW_TOKENS)
        recorder.start()
        try:
            with torch.inference_mode():
                model(**enc)
            mass_windowed = recorder.stop().to("cuda")
        finally:
            recorder.reset()
        t_rec1 = time.perf_counter()
        recorder_sec = round(t_rec1 - t_rec0, 3)

        mass_pooled = _maxpool_kv(mass_windowed, args.pool_kernel)

        # fair-rate builds its allocation per-prompt from the actual role
        # counts (α_r = n_r / n) — uniform fraction-B retention per role.
        if static_allocation is None:  # fair-rate sentinel
            allocation = dict(Counter(roles))
        else:
            allocation = static_allocation

        configs = [
            # (name, score_tensor, use_role_conditional)
            ("snapkv-only", mass_pooled, False),
            (f"{cfg_base}{suffix}", mass_pooled, True),
            (f"{cfg_base_nomax}{suffix}", mass_windowed, True),
        ]
        if args.skip_snapkv_only:
            configs = [c for c in configs if c[0] != "snapkv-only"]
        if args.only_snapkv_only:
            configs = [c for c in configs if c[0] == "snapkv-only"]

        for cfg_name, score, use_role in configs:
            for budget_frac in args.budgets:
                if use_role:
                    keep = keep_mask_for_role_allocation(
                        score, roles, budget_frac, allocation
                    )
                else:
                    keep = _keep_mask_topk(score, budget_frac)
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
                    "config": cfg_name,
                    "budget_frac": budget_frac,
                    "prompt_len_tokens": prompt_len,
                    "gold": prompt.answer,
                    "model_answer": model_answer.replace("\n", " ").strip()[:200],
                    "correct": correct,
                    "recorder_sec": recorder_sec,
                    "decode_sec": round(t1 - t0, 3),
                    "pool_kernel": args.pool_kernel,
                })
                csv_fp.flush()
                n_rows += 1
                del keep
        del enc, mass_windowed, mass_pooled
        torch.cuda.empty_cache()

    csv_fp.close()
    print(f"\nDone. {n_rows} rows written to {args.out_path}; {skipped} prompts skipped.")


if __name__ == "__main__":
    main()
