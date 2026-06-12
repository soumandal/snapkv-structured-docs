"""Pilot orchestrator: record attention for the full pilot corpus.

Outputs per-prompt parquet dumps under
  /mnt/kvr/dumps/2026-WW##-pilot/<context-bucket>/<prompt_id>.parquet

A second pass (script 03) reads these dumps and answers Q1–Q4.
"""
import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA, MODEL_QWEN, MODEL_QWEN3, MODEL_MISTRAL, MODEL_PHI, MODEL_LLAMA_BASE
from kvr.data.pilot_corpus import PilotPrompt, build_pilot_corpus
from kvr.instrumentation.attention_hook import AttentionRecorder
from kvr.instrumentation.dump import save_prompt_dump
from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.dispatcher import label_structure
from kvr.structure.roles import CharRoleSpan, Role


MODEL_IDS = {"llama": MODEL_LLAMA, "qwen": MODEL_QWEN, "qwen3": MODEL_QWEN3, "mistral": MODEL_MISTRAL, "phi": MODEL_PHI, "llama-base": MODEL_LLAMA_BASE}
CONTEXT_BUCKETS = [8000, 16000, 32000]


def _wrap_in_chat_template(tokenizer, raw_text: str) -> str:
    """Wrap raw user text in the tokenizer's chat template + assistant generation prompt.

    Required for instruct models that aren't graceful on raw-text prompts
    (Phi-3-mini, Qwen2.5, etc.). When this is on, the resulting dump
    bucket carries a `-ct` suffix so chat-template runs don't collide with
    raw-text runs for the same model.
    """
    messages = [{"role": "user", "content": raw_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def truncate_or_pad_prompt(tokenizer, prompt: PilotPrompt, target_tokens: int) -> str | None:
    """Adjust the prompt's context length to fall within target ± 10%.

    Strategy: if the prompt's context is too short, skip. If too long, truncate
    by character count proportionally. The pilot's premise check is about
    *long* structured contexts, so we don't pad short ones with junk.
    """
    enc = tokenizer(prompt.context + "\n\nQuestion: " + prompt.question, add_special_tokens=False)
    cur_tokens = len(enc["input_ids"])
    if cur_tokens < target_tokens * 0.7:
        return None  # too short for this bucket
    if cur_tokens > target_tokens * 1.3:
        # truncate context proportionally
        ratio = (target_tokens * 1.1) / cur_tokens
        new_ctx_len = int(len(prompt.context) * ratio)
        return prompt.context[:new_ctx_len] + "\n\nQuestion: " + prompt.question
    return prompt.context + "\n\nQuestion: " + prompt.question


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", type=Path, default=Path("/mnt/kvr/dumps"))
    ap.add_argument("--max-prompts-per-source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=list(MODEL_IDS), default="llama",
                    help="Which HF model to load (llama=Llama-3.1-8B-Instruct, "
                         "qwen=Qwen2.5-7B-Instruct, "
                         "qwen3=Qwen3-4B-Instruct-2507, "
                         "mistral=Mistral-7B-Instruct-v0.3, "
                         "phi=Phi-3-mini-128k-instruct). Non-llama runs "
                         "write to `<year>-WW<week>-pilot-<model>` so dumps "
                         "don't collide with Llama dumps (tokenizers differ). "
                         "Qwen and Qwen3 also require --use-chat-template.")
    ap.add_argument("--ctx-buckets", nargs="+", type=int, default=CONTEXT_BUCKETS,
                    help="Subset of context-token buckets to record. Default "
                         "records all three; pass e.g. `--ctx-buckets 16000` "
                         "to record just the Tier 1 anchor.")
    ap.add_argument("--use-chat-template", action="store_true",
                    help="Wrap each prompt in the tokenizer's chat template + "
                         "assistant generation prefix before tokenizing and "
                         "recording. Required for Phi / Qwen / other instruct "
                         "models that aren't graceful on raw text. Dump path "
                         "gets a `-ct` suffix so wrapped and raw-text dumps "
                         "for the same model never collide. Consumers "
                         "(scripts/05) must pass --use-chat-template too so "
                         "the role array stays aligned with the model input.")
    args = ap.parse_args()
    model_id = MODEL_IDS[args.model]
    model_suffix = "" if args.model == "llama" else f"-{args.model}"
    if args.use_chat_template:
        model_suffix += "-ct"

    run_id = f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot{model_suffix}"
    run_root = args.dump_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Run root: {run_root}")

    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    for ctx_tokens in args.ctx_buckets:
        bucket_dir = run_root / f"ctx_{ctx_tokens}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Bucket {ctx_tokens} tokens → {bucket_dir} ===")

        # Synthetic prompts are sized per-bucket; wikitable is whatever it is
        # and most of it will be skipped at the larger buckets.
        corpus = build_pilot_corpus(
            n_wikitable=args.max_prompts_per_source,
            n_synthetic=args.max_prompts_per_source,
            seed=args.seed,
            synthetic_target_tokens=ctx_tokens,
        )
        print(f"  corpus: {len(corpus)} prompts")
        recorded = 0
        skipped = 0

        for prompt in tqdm(corpus, desc=f"ctx={ctx_tokens}"):
            out_path = bucket_dir / f"{prompt.id}.parquet"
            if out_path.exists():
                continue  # resume support

            text = truncate_or_pad_prompt(tok, prompt, ctx_tokens)
            if text is None:
                skipped += 1
                continue

            # label_structure must run on the raw content (before chat-template
            # wrapping) because the special tokens <|im_start|>/<|im_end|> etc.
            # confuse the JSON/XML dispatcher.  If a chat template is applied
            # we shift every span by the prefix length and treat the
            # prefix/suffix as PROSE.
            raw_text = text
            char_spans = label_structure(raw_text)
            if args.use_chat_template:
                text = _wrap_in_chat_template(tok, raw_text)
                offset = text.find(raw_text)
                if offset > 0:
                    char_spans = (
                        [CharRoleSpan(role=Role.PROSE, start=0, end=offset, depth=0)]
                        + [CharRoleSpan(role=sp.role, start=sp.start + offset, end=sp.end + offset, depth=sp.depth) for sp in char_spans]
                        + [CharRoleSpan(role=Role.PROSE, start=offset + len(raw_text), end=len(text), depth=0)]
                    )
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            token_roles = align_char_spans_to_tokens(text, char_spans, tok)

            recorder = AttentionRecorder(model)
            recorder.start()
            try:
                with torch.inference_mode():
                    model(**enc, return_dict=True)
                mass = recorder.stop()
            except torch.cuda.OutOfMemoryError:
                recorder.reset()
                torch.cuda.empty_cache()
                skipped += 1
                continue

            save_prompt_dump(
                out_path,
                prompt_id=prompt.id,
                source=prompt.source,
                mass=mass,
                token_roles=token_roles,
            )
            recorded += 1

            del mass, enc
            torch.cuda.empty_cache()

        print(f"  recorded: {recorded}, skipped: {skipped}")

    print("\nPilot recording done.")


if __name__ == "__main__":
    main()
