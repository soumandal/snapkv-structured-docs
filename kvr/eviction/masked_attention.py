"""Monkey-patch LlamaAttention to apply per-(layer, head) keep masks.

Used by Experiment E (counterfactual eviction): the prefill runs normally;
between prefill and decode we install per-(layer, head) keep masks so that
during decode the attention treats evicted prefill positions as -inf.

The patch is engaged via `MaskedAttentionPatch` as a context manager:

    patch = MaskedAttentionPatch(model, keep_masks)
    with patch:
        out = model.generate(...)

`keep_masks` is shape (n_layers, n_heads, prompt_len) bool: True = keep.
Positions beyond `prompt_len` in the KV cache (decode tokens) are always
kept — we extend the mask on the fly to match `kv_len`.

This mirrors the eager-attention path in transformers 4.45's LlamaAttention.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class MaskedAttentionPatch:
    """Context manager that installs per-(layer, head) keep masks on a model.

    Args:
        model: HF Llama model.
        keep_masks: (n_layers, n_heads, prompt_len) bool on the model's device.
    """

    def __init__(self, model, keep_masks: torch.Tensor) -> None:
        self.model = model
        if keep_masks.dtype != torch.bool:
            raise ValueError(f"keep_masks must be bool, got {keep_masks.dtype}")
        if keep_masks.ndim != 3:
            raise ValueError(f"keep_masks must be (n_layers, n_heads, prompt_len)")
        self.keep_masks = keep_masks
        self._patched: list = []

    def __enter__(self):
        layers = self._iter_decoder_layers()
        if self.keep_masks.shape[0] != len(layers):
            raise ValueError(
                f"keep_masks has {self.keep_masks.shape[0]} layers but model has {len(layers)}"
            )
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            if "forward" in vars(attn):
                raise RuntimeError(
                    f"Layer {layer_idx} self_attn already has an instance-level "
                    "forward; refusing to patch over it."
                )
            attn.forward = _make_masked_forward(attn, self.keep_masks[layer_idx])
            self._patched.append(attn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for attn in self._patched:
            del attn.forward
        self._patched = []
        return False

    def _iter_decoder_layers(self):
        base = getattr(self.model, "model", self.model)
        return base.layers


def _project_qkv(attn_module, hidden_states):
    """Return (q, k, v) projections, handling split-projection modules
    (Llama / Mistral / Qwen2) and fused-projection modules (Phi-3, which
    packs q/k/v into a single ``qkv_proj``). The split path is unchanged so
    existing-model numerics are bit-for-bit identical."""
    if hasattr(attn_module, "qkv_proj"):
        qkv = attn_module.qkv_proj(hidden_states)
        q_size = attn_module.num_heads * attn_module.head_dim
        kv_size = attn_module.num_key_value_heads * attn_module.head_dim
        return (
            qkv[..., :q_size],
            qkv[..., q_size : q_size + kv_size],
            qkv[..., q_size + kv_size :],
        )
    return (
        attn_module.q_proj(hidden_states),
        attn_module.k_proj(hidden_states),
        attn_module.v_proj(hidden_states),
    )


def _make_masked_forward(attn_module, layer_keep_mask: torch.Tensor):
    """Build a forward closure that applies the keep mask before softmax.

    layer_keep_mask: (n_heads, prompt_len) bool — True = keep.
    """

    def masked_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,  # noqa: ARG001 — eviction mode never returns probs
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        q, k, v = _project_qkv(attn_module, hidden_states)

        q = q.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = attn_module.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, attn_module.layer_idx, cache_kwargs)

        k = repeat_kv(k, attn_module.num_key_value_groups)
        v = repeat_kv(v, attn_module.num_key_value_groups)

        kv_len = k.shape[-2]
        prompt_len = layer_keep_mask.shape[-1]
        scale = 1.0 / math.sqrt(attn_module.head_dim)

        # Both prefill (plain causal) and decode (eviction-masked) route
        # through SDPA. Prefill must use SDPA so the (q_len, kv_len) matrix
        # never materializes (the manual softmax-in-fp32 path OOMs at ctx ≥
        # 16k on an 80 GB A100). Decode must use SDPA too: the manual fp16
        # matmul + softmax path silently corrupts Qwen2 output during decode
        # (coherent words interleaved with garbage `!` tokens) because Qwen's
        # larger activation magnitudes overflow the fp16 score accumulation;
        # SDPA accumulates in fp32 and is bit-exact with the unpatched model.
        # The per-(layer, head) eviction penalty is passed as an additive
        # float mask, which is mathematically identical to the old score bias.
        if q_len > 1:
            attn_mask_arg = (
                attention_mask[:, :, :q_len, :kv_len]
                if attention_mask is not None
                else None
            )
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_arg,
                is_causal=attn_mask_arg is None,
                scale=scale,
            )
        else:
            device, dtype = q.device, q.dtype
            additive_mask = None
            if kv_len > prompt_len:
                evict_bias = torch.zeros(attn_module.num_heads, kv_len, device=device, dtype=dtype)
                evict_bias[:, :prompt_len] = (~layer_keep_mask).to(device=device, dtype=dtype) * -1e4
                additive_mask = evict_bias[None, :, None, :]  # (1, H, 1, kv_len)
            if attention_mask is not None:
                am = attention_mask[:, :, :q_len, :kv_len]
                additive_mask = am if additive_mask is None else additive_mask + am
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=additive_mask,
                is_causal=False,
                scale=scale,
            )

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_module.o_proj(out)

        return out, None, past_key_value

    return masked_forward
