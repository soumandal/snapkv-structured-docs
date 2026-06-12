"""Windowed attention-mass recorder (for SnapKV-style scoring).

Like `AttentionRecorder` but accumulates per-(layer, head, kv_pos) attention
mass only from queries inside a configurable observation window: the last
`window_size` query positions of the prefill. The chunked path mirrors
`attention_hook.py` so memory stays bounded at long context.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


_Q_CHUNK_DEFAULT = 512


class WindowedAttentionRecorder:
    """Record attention mass from only the last `window_size` queries.

    Usage:
        rec = WindowedAttentionRecorder(model, window_size=64)
        rec.start()
        with torch.inference_mode():
            model(**inputs)
        mass = rec.stop()  # (n_layers, n_heads, kv_len), fp32, cpu
    """

    def __init__(self, model, window_size: int, q_chunk: int = _Q_CHUNK_DEFAULT) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.model = model
        self.window_size = window_size
        self.q_chunk = q_chunk
        self._patched: list = []
        self._per_layer: list[tuple[int, torch.Tensor]] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Recorder already started.")
        self._per_layer = []
        for layer_idx, layer in enumerate(self._iter_layers()):
            attn = layer.self_attn
            if "forward" in vars(attn):
                raise RuntimeError(f"Layer {layer_idx} self_attn already patched.")
            attn.forward = _make_windowed_forward(attn, layer_idx, self)
            self._patched.append(attn)
        self._started = True

    def stop(self) -> torch.Tensor:
        self.reset()
        if not self._per_layer:
            raise RuntimeError("Recorder captured no layers.")
        sorted_by_layer = sorted(self._per_layer, key=lambda x: x[0])
        stacked = torch.stack([m for _, m in sorted_by_layer], dim=0)
        self._per_layer = []
        return stacked

    def reset(self) -> None:
        if not self._started:
            return
        for attn in self._patched:
            del attn.forward
        self._patched = []
        self._started = False

    def _iter_layers(self):
        base = getattr(self.model, "model", self.model)
        return base.layers


def _project_qkv(attn_module, hidden_states):
    """Return (q, k, v) projections for one attention module.

    Handles both split-projection modules (Llama / Mistral / Qwen2, which
    expose separate ``q_proj`` / ``k_proj`` / ``v_proj``) and fused-projection
    modules (Phi-3, which packs all three into a single ``qkv_proj``). The
    split path is unchanged from the original recorder, so Llama/Mistral/Qwen2
    numerics are bit-for-bit identical.
    """
    if hasattr(attn_module, "qkv_proj"):
        qkv = attn_module.qkv_proj(hidden_states)
        q_size = attn_module.num_heads * attn_module.head_dim
        kv_size = attn_module.num_key_value_heads * attn_module.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size : q_size + kv_size]
        v = qkv[..., q_size + kv_size :]
        return q, k, v
    return (
        attn_module.q_proj(hidden_states),
        attn_module.k_proj(hidden_states),
        attn_module.v_proj(hidden_states),
    )


def _make_windowed_forward(attn_module, layer_idx: int, recorder: WindowedAttentionRecorder):

    def windowed_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,  # noqa: ARG001
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
        scale = 1.0 / math.sqrt(attn_module.head_dim)
        out = torch.empty_like(q)
        mass = torch.zeros(
            attn_module.num_heads, kv_len, dtype=torch.float32, device=q.device
        )
        window_lo = max(0, q_len - recorder.window_size)  # first query in window (inclusive)

        for chunk_start in range(0, q_len, recorder.q_chunk):
            chunk_end = min(chunk_start + recorder.q_chunk, q_len)
            qc = q[:, :, chunk_start:chunk_end, :]
            scores = torch.matmul(qc, k.transpose(2, 3)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, chunk_start:chunk_end, :kv_len]
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

            # Accumulate mass only for queries inside the window.
            lo = max(window_lo - chunk_start, 0)
            hi = chunk_end - chunk_start  # always equal to probs.shape[-2]
            if lo < hi:
                mass += probs[0, :, lo:hi, :].sum(dim=-2, dtype=torch.float32)

            out[:, :, chunk_start:chunk_end, :] = torch.matmul(probs, v)
            del scores, probs

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_module.o_proj(out)

        recorder._per_layer.append((layer_idx, mass.detach().cpu()))
        del mass
        return out, None, past_key_value

    return windowed_forward
