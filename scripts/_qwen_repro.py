"""Throwaway repro: isolate which patched forward corrupts Qwen2 output.

Compares, on ONE small synthetic_json prompt wrapped in the chat template:
  (a) vanilla model.generate          — ground truth for "does Qwen work at all"
  (b) MaskedAttentionPatch all-keep   — scripts/13 decode path at budget=1.0
  (c) WindowedAttentionRecorder fwd   — scripts/13 prefill scoring path

If (a) is sane and (b)/(c) garbage, the bug is in our patched forward.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvr.config import MODEL_QWEN
from kvr.data.pilot_corpus import build_corpus_for_source
from kvr.eviction.masked_attention import MaskedAttentionPatch
from kvr.instrumentation.windowed_recorder import WindowedAttentionRecorder

CTX = 2000  # small for speed

tok = AutoTokenizer.from_pretrained(MODEL_QWEN, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL_QWEN, torch_dtype=torch.float16, device_map="cuda",
)
model.eval()

corpus = build_corpus_for_source("synthetic_json", n_prompts=1, seed=0, target_tokens=CTX)
prompt = corpus[0]
raw = prompt.context + "\n\nQuestion: " + prompt.question
text = tok.apply_chat_template(
    [{"role": "user", "content": raw}], tokenize=False, add_generation_prompt=True
)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
prompt_len = enc["input_ids"].shape[-1]
print(f"gold = {prompt.answer!r}   prompt_len = {prompt_len}")


def show(tag, gen_ids):
    ans = tok.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    print(f"[{tag}] {ans.strip()[:120]!r}")


with torch.inference_mode():
    # (a) vanilla
    show("a-vanilla", model.generate(**enc, max_new_tokens=24, do_sample=False,
                                     pad_token_id=tok.pad_token_id))

    # (b) MaskedAttentionPatch, keep everything (n_layers, n_heads, prompt_len) all True
    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    keep = torch.ones(n_layers, n_heads, prompt_len, dtype=torch.bool, device="cuda")
    with MaskedAttentionPatch(model, keep):
        show("b-masked-allkeep", model.generate(**enc, max_new_tokens=24, do_sample=False,
                                                pad_token_id=tok.pad_token_id))

    # (c) recorder prefill then vanilla generate (recorder only patches the
    # single forward call, then is reset — but verify its forward is non-garbage
    # by checking the prefill logits match vanilla)
    out_vanilla = model(**enc).logits[0, -1]
    rec = WindowedAttentionRecorder(model, window_size=64)
    rec.start()
    try:
        out_rec = model(**enc).logits[0, -1]
        rec.stop()
    finally:
        rec.reset()
    max_logit_diff = (out_vanilla.float() - out_rec.float()).abs().max().item()
    argmax_match = int(out_vanilla.argmax() == out_rec.argmax())
    print(f"[c-recorder] prefill last-token logit max|Δ| vs vanilla = {max_logit_diff:.4f}  "
          f"argmax_match={argmax_match}")

    # (b') also check masked-allkeep prefill logits vs vanilla
    with MaskedAttentionPatch(model, keep):
        out_masked = model(**enc).logits[0, -1]
    max_logit_diff_b = (out_vanilla.float() - out_masked.float()).abs().max().item()
    argmax_match_b = int(out_vanilla.argmax() == out_masked.argmax())
    print(f"[b-masked]   prefill last-token logit max|Δ| vs vanilla = {max_logit_diff_b:.4f}  "
          f"argmax_match={argmax_match_b}")
