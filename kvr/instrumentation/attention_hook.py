"""Streaming forward-pass attention recorder.

Replaces each decoder layer's `self_attn.forward` with a query-chunked
variant that computes attention one Q-chunk at a time and accumulates the
per-(head, kv_pos) attention mass on the fly. The full `(n_heads, q, kv)`
probability tensor is never materialized — at 32k context that tensor
would be 64 GB in fp16 alone, which OOMs the A100 80GB before any hook
can fire.

Peak in-layer GPU memory is O(n_heads · Q_CHUNK · kv_len) instead of
O(n_heads · q_len · kv_len). With Q_CHUNK=512 and kv_len=32k that's
~4 GB transient per layer (vs ~64 GB unchunked), comfortably fitting the
Llama-3.1-8B prefill on a single A100 at all three pilot context buckets.

Supports Llama-family and Qwen2-family attention modules. The patched
forward mirrors the eager-path math: Q/K/V projections → rotary →
repeat_kv → softmax(QK^T/√d + mask) → out = probs @ V. Mass is read
from the same fp16 probabilities the original code consumes, summed in
an fp32 accumulator.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import transformers
from packaging.version import parse as _parse_version


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(bsz, num_kv_heads, n_rep, slen, head_dim)
        .reshape(bsz, num_kv_heads * n_rep, slen, head_dim)
    )


# Models whose decoder layer unpacks self_attn output as a 2-tuple
# (attn_output, attn_weights) rather than the Llama 3-tuple
# (attn_output, attn_weights, present_key_value).
_TWO_TUPLE_ATTN_CLASSES = {"Qwen2Attention", "Qwen3Attention"}

# The standardized-attention refactor (transformers >= 4.48) made attention
# modules return the 2-tuple above with cache handling moved into the decoder
# layer. On older builds — including this repo's pinned 4.45.x env — *every*
# decoder layer (Llama / Mistral / Qwen2 alike) unpacks the 3-tuple, so the
# 2-tuple return must be gated on the version or Qwen2 crashes on prefill with
# "not enough values to unpack (expected 3, got 2)". Llama/Mistral/Phi are
# unaffected either way (they always take the 3-tuple branch on 4.45.x).
_ATTN_TWO_TUPLE_API = _parse_version(transformers.__version__).release >= (4, 48)


# Q-chunk size. 512 keeps per-chunk score+probs tensors small enough to fit
# on an 80 GB A100 even at 32k kv_len; smaller chunks reduce peak memory
# but add Python-loop overhead and underutilize the GEMM units.
Q_CHUNK_DEFAULT = 512


class AttentionRecorder:
    """Record per-layer attention mass during a forward pass.

    Usage:
        recorder = AttentionRecorder(model)
        recorder.start()
        try:
            with torch.inference_mode():
                model(**inputs)
            mass = recorder.stop()  # (n_layers, n_heads, kv_len), CPU, fp32
        finally:
            recorder.reset()  # idempotent — safe to call after stop()
    """

    def __init__(self, model, q_chunk: int = Q_CHUNK_DEFAULT) -> None:
        self.model = model
        self.q_chunk = q_chunk
        self._patched: list = []  # attn modules we added an instance-level forward to
        self._per_layer: list[tuple[int, torch.Tensor]] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Recorder already started; call stop() or reset() first.")
        self._per_layer = []
        for layer_idx, layer in enumerate(self._iter_decoder_layers()):
            attn = layer.self_attn
            if "forward" in vars(attn):
                raise RuntimeError(
                    f"Layer {layer_idx} self_attn already has an instance-level "
                    "forward attribute; refusing to patch over it."
                )
            attn.forward = _make_chunked_forward(attn, layer_idx, self)
            self._patched.append(attn)
        self._started = True

    def stop(self) -> torch.Tensor:
        """Restore originals; return stacked mass (n_layers, n_heads, kv_len)."""
        self.reset()
        if not self._per_layer:
            raise RuntimeError(
                "Recorder captured no layers. Did the forward pass run "
                "between start() and stop()?"
            )
        sorted_by_layer = sorted(self._per_layer, key=lambda x: x[0])
        stacked = torch.stack([m for _, m in sorted_by_layer], dim=0)
        self._per_layer = []
        return stacked

    def reset(self) -> None:
        """Remove patched forwards. Idempotent — safe to call multiple times."""
        if not self._started:
            return
        for attn in self._patched:
            del attn.forward
        self._patched = []
        self._started = False

    def _iter_decoder_layers(self):
        base = getattr(self.model, "model", self.model)
        return base.layers


def _make_chunked_forward(attn_module, layer_idx: int, recorder: AttentionRecorder):
    """Build a chunked forward closure for one attention module."""
    head_dim = attn_module.head_dim
    fused_qkv = hasattr(attn_module, "qkv_proj")
    # num_heads is not stored directly on Qwen2Attention; derive from projection
    # shape. Phi-3 packs q/k/v into one qkv_proj, so use the module attribute there.
    if fused_qkv:
        num_heads = attn_module.num_heads
    else:
        num_heads = attn_module.q_proj.out_features // head_dim
    two_tuple = _ATTN_TWO_TUPLE_API and type(attn_module).__name__ in _TWO_TUPLE_ATTN_CLASSES

    def chunked_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,  # noqa: ARG001 — accepted for API parity, mass returns via recorder
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Phi-3 fuses q/k/v into one projection; Llama/Mistral/Qwen keep them split.
        if fused_qkv:
            qkv = attn_module.qkv_proj(hidden_states)
            q_size = num_heads * head_dim
            kv_size = attn_module.num_key_value_heads * head_dim
            q_proj_out = qkv[..., :q_size]
            k_proj_out = qkv[..., q_size : q_size + kv_size]
            v_proj_out = qkv[..., q_size + kv_size :]
        else:
            q_proj_out = attn_module.q_proj(hidden_states)
            k_proj_out = attn_module.k_proj(hidden_states)
            v_proj_out = attn_module.v_proj(hidden_states)

        # Qwen3 applies per-head RMS norms on Q and K before RoPE; other models skip this.
        query_states = q_proj_out.view(bsz, q_len, -1, head_dim)
        if hasattr(attn_module, "q_norm"):
            query_states = attn_module.q_norm(query_states)
        query_states = query_states.transpose(1, 2)

        key_states = k_proj_out.view(bsz, q_len, -1, head_dim)
        if hasattr(attn_module, "k_norm"):
            key_states = attn_module.k_norm(key_states)
        key_states = key_states.transpose(1, 2)

        value_states = v_proj_out.view(bsz, q_len, -1, head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = attn_module.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, attn_module.layer_idx, cache_kwargs
            )

        key_states = _repeat_kv(key_states, attn_module.num_key_value_groups)
        value_states = _repeat_kv(value_states, attn_module.num_key_value_groups)

        kv_len = key_states.shape[-2]
        scale = 1.0 / math.sqrt(head_dim)
        attn_output = torch.empty_like(query_states)
        mass_accum = torch.zeros(num_heads, kv_len, dtype=torch.float32, device=query_states.device)

        for chunk_start in range(0, q_len, recorder.q_chunk):
            chunk_end = min(chunk_start + recorder.q_chunk, q_len)
            q_chunk = query_states[:, :, chunk_start:chunk_end, :]
            scores = torch.matmul(q_chunk, key_states.transpose(2, 3)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, chunk_start:chunk_end, :kv_len]
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # bsz=1 in the pilot; sum over batch then chunk-Q dim into the accumulator.
            mass_accum += probs.sum(dim=(0, -2), dtype=torch.float32)
            attn_output[:, :, chunk_start:chunk_end, :] = torch.matmul(probs, value_states)
            del scores, probs

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = attn_module.o_proj(attn_output)

        recorder._per_layer.append((layer_idx, mass_accum.detach().cpu()))
        del mass_accum

        if two_tuple:
            return attn_output, None
        return attn_output, None, past_key_value

    return chunked_forward
