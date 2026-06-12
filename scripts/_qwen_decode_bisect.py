"""Bisect the Qwen manual-decode corruption.

Hypothesis: the manual fp16 matmul decode branch in masked_attention diverges
for Qwen. Test three decode variants on the SAME all-keep mask:
  (man)   manual matmul + fp32 softmax  (current code)
  (sdpa)  F.scaled_dot_product_attention with additive evict bias
  (vanilla) unpatched model.generate
Print decoded text for each so we can see which is coherent.
"""
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from kvr.config import MODEL_QWEN
from kvr.data.pilot_corpus import build_corpus_for_source

CTX = 2000
tok = AutoTokenizer.from_pretrained(MODEL_QWEN, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL_QWEN, torch_dtype=torch.float16, device_map="cuda")
model.eval()

corpus = build_corpus_for_source("synthetic_json", n_prompts=1, seed=0, target_tokens=CTX)
p = corpus[0]
raw = p.context + "\n\nQuestion: " + p.question
text = tok.apply_chat_template([{"role": "user", "content": raw}], tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
prompt_len = enc["input_ids"].shape[-1]
n_layers = len(model.model.layers)
n_heads = model.config.num_attention_heads
keep = torch.ones(n_layers, n_heads, prompt_len, dtype=torch.bool, device="cuda")


def make_forward(attn, layer_keep, mode):
    def fwd(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
            output_attentions=False, use_cache=False, cache_position=None,
            position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        q = attn.q_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        if position_embeddings is None:
            cos, sin = attn.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        kv_len = k.shape[-2]
        scale = 1.0 / math.sqrt(attn.head_dim)
        if q_len > 1:
            am = attention_mask[:, :, :q_len, :kv_len] if attention_mask is not None else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, is_causal=am is None, scale=scale)
        elif mode == "sdpa":
            am = attention_mask[:, :, :q_len, :kv_len] if attention_mask is not None else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, is_causal=False, scale=scale)
        else:  # manual
            scores = torch.matmul(q, k.transpose(2, 3)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :q_len, :kv_len]
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return attn.o_proj(out), None, past_key_value
    return fwd


def run(mode):
    patched = []
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.forward = make_forward(layer.self_attn, keep[li], mode)
        patched.append(layer.self_attn)
    try:
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
    finally:
        for a in patched:
            del a.forward
    return tok.decode(g[0, prompt_len:], skip_special_tokens=True).strip()[:110]


with torch.inference_mode():
    van = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
print(f"gold      = {p.answer!r}")
print(f"[vanilla] {tok.decode(van[0, prompt_len:], skip_special_tokens=True).strip()[:110]!r}")
print(f"[man   ]  {run('manual')!r}")
print(f"[sdpa  ]  {run('sdpa')!r}")
