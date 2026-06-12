"""Record attention-mass parquets for wikitable_long prompts.

Mirrors scripts/02_run_pilot.py but uses the wikitable_long corpus
(distractor-concatenated wikitables sized to the target context bucket).
Writes per-prompt parquets to the same dump tree the pilot uses so
scripts/05 (E) can consume them via --source-filter wikitable_long.

Usage:
    python scripts/08_record_wikitable_long.py --ctx-tokens 8000 --n-prompts 100
    python scripts/08_record_wikitable_long.py --ctx-tokens 16000 --n-prompts 50
"""
import argparse
from datetime import date
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_LLAMA
from kvr.data.pilot_corpus import build_corpus_for_source
from kvr.instrumentation.attention_hook import AttentionRecorder
from kvr.instrumentation.dump import save_prompt_dump
from kvr.structure.align import align_char_spans_to_tokens
from kvr.structure.dispatcher import label_structure


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", type=Path, default=Path("/mnt/kvr/dumps"))
    ap.add_argument("--run-id", type=str, default=None,
                    help="Run id under dump-root; defaults to the current WW-pilot tree")
    ap.add_argument("--ctx-tokens", type=int, required=True,
                    choices=[8000, 16000, 32000])
    ap.add_argument("--n-prompts", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_id = args.run_id or f"{date.today().isocalendar().year}-WW{date.today().isocalendar().week:02d}-pilot"
    bucket_dir = args.dump_root / run_id / f"ctx_{args.ctx_tokens}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {bucket_dir}")

    print(f"Loading {MODEL_LLAMA}...")
    tok = AutoTokenizer.from_pretrained(MODEL_LLAMA, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LLAMA, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    corpus = build_corpus_for_source(
        "wikitable_long",
        n_prompts=args.n_prompts,
        seed=args.seed,
        target_tokens=args.ctx_tokens,
        tokenizer=tok,
    )
    print(f"Built {len(corpus)} wikitable_long prompts at ctx={args.ctx_tokens}")

    recorded = 0
    skipped = 0
    for prompt in tqdm(corpus, desc=f"ctx={args.ctx_tokens}"):
        out_path = bucket_dir / f"{prompt.id}.parquet"
        if out_path.exists():
            continue
        text = _truncate_or_pad_prompt(tok, prompt, args.ctx_tokens)
        if text is None:
            skipped += 1
            continue

        char_spans = label_structure(text)
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

    print(f"recorded: {recorded}, skipped: {skipped}")


if __name__ == "__main__":
    main()
