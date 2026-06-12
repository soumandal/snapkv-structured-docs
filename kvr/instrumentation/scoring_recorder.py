"""Three-signal attention recorder for Methods B and B' (VPE).

Captures per-(layer, head, kv_pos) statistics during a single forward
pass that downstream scorers can combine in different ways:

- `mass = Σ_q p_t(q)` — same as `AttentionRecorder`.
- `plogp = Σ_q p_t(q) · log p_t(q)` — gives entropy via
  `entropy(t) = log(mass) − plogp/mass`. Used by **Method B**
  (entropy-discounted scoring).
- `v_norm = ‖V_t‖₂` — L2 norm of the value vector at position t (after
  GQA broadcast). Used by **Method B' / VPE** (value-prior eviction): the
  hypothesis is that DELIM tokens have large `mass` but small `v_norm`
  (they route, they don't contribute) and so should be down-weighted.

Both scorers below have an α-knob; α=0 recovers H2O's mass-only baseline,
which makes A/B sweeps trivial.

Q-chunked along the query axis (same pattern as `attention_hook.py`) so
peak per-layer memory stays bounded at ctx ≥ 16k.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


_Q_CHUNK_DEFAULT = 512


class ScoringSignalsRecorder:
    """Record per-(layer, head, kv_pos) (mass, plogp, v_norm) during forward.

    Usage:
        rec = ScoringSignalsRecorder(model)
        rec.start()
        with torch.inference_mode():
            model(**inputs)
        mass, plogp, v_norm = rec.stop()
        # All three: (n_layers, n_heads, kv_len) fp32 on cpu.
    """

    def __init__(self, model, q_chunk: int = _Q_CHUNK_DEFAULT) -> None:
        self.model = model
        self.q_chunk = q_chunk
        self._patched: list = []
        self._per_layer: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Recorder already started.")
        self._per_layer = []
        for layer_idx, layer in enumerate(self._iter_layers()):
            attn = layer.self_attn
            if "forward" in vars(attn):
                raise RuntimeError(f"Layer {layer_idx} self_attn already patched.")
            attn.forward = _make_scoring_forward(attn, layer_idx, self)
            self._patched.append(attn)
        self._started = True

    def stop(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.reset()
        if not self._per_layer:
            raise RuntimeError("Recorder captured no layers.")
        sorted_by_layer = sorted(self._per_layer, key=lambda x: x[0])
        mass = torch.stack([m for _, m, _, _ in sorted_by_layer], dim=0)
        plogp = torch.stack([p for _, _, p, _ in sorted_by_layer], dim=0)
        v_norm = torch.stack([n for _, _, _, n in sorted_by_layer], dim=0)
        self._per_layer = []
        return mass, plogp, v_norm

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


def _make_scoring_forward(attn_module, layer_idx: int, recorder):

    def scoring_forward(
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

        q = attn_module.q_proj(hidden_states)
        k = attn_module.k_proj(hidden_states)
        v = attn_module.v_proj(hidden_states)
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
        plogp = torch.zeros(
            attn_module.num_heads, kv_len, dtype=torch.float32, device=q.device
        )
        # v_norm is per-position (no q reduction). One fp32 compute per layer.
        v_norm = v[0].to(torch.float32).norm(dim=-1)  # (n_heads, kv_len)

        for chunk_start in range(0, q_len, recorder.q_chunk):
            chunk_end = min(chunk_start + recorder.q_chunk, q_len)
            qc = q[:, :, chunk_start:chunk_end, :]
            scores = torch.matmul(qc, k.transpose(2, 3)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, chunk_start:chunk_end, :kv_len]
            probs = F.softmax(scores, dim=-1, dtype=torch.float32)

            mass += probs[0].sum(dim=1)
            # xlogy(p, p) = p · log(p) returning 0 at p == 0 (no NaN).
            plogp += torch.special.xlogy(probs[0], probs[0]).sum(dim=1)

            out[:, :, chunk_start:chunk_end, :] = torch.matmul(probs.to(q.dtype), v)
            del scores, probs

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_module.o_proj(out)

        recorder._per_layer.append((
            layer_idx,
            mass.detach().cpu(),
            plogp.detach().cpu(),
            v_norm.detach().cpu(),
        ))
        del mass, plogp, v_norm
        return out, None, past_key_value

    return scoring_forward


def entropy_score(
    mass: torch.Tensor,
    plogp: torch.Tensor,
    alpha: float,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Method B — entropy-discounted scoring.

        entropy = log(mass) − plogp / mass
        score   = mass / (1 + alpha · entropy)

    Targets the *distributional* axis: discounts positions whose attention
    column is spread across many queries (sinks). α=0 recovers H2O mass.
    """
    safe_mass = mass.clamp(min=eps)
    entropy = safe_mass.log() - plogp / safe_mass
    entropy = entropy.clamp(min=0.0)
    return safe_mass / (1.0 + alpha * entropy)


def value_prior_score(
    mass: torch.Tensor,
    v_norm: torch.Tensor,
    alpha: float = 1.0,
    smooth_kernel: int = 1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Method B' (VPE) — value-prior scoring.

        score(t) = mass(t) · smooth_avg(v_norm(t))^alpha

    Targets the *content* axis: boosts positions whose value vectors
    carry substantive content (high ‖V‖) and down-weights routing-only
    sinks (low ‖V‖). α=0 recovers H2O mass. 1D avg-pool over the KV axis
    (stride 1, "same" padding) optionally suppresses per-token v_norm
    noise — set `smooth_kernel=1` to disable.

    Args:
        mass:          (..., n_kv) per-position attention mass.
        v_norm:        (..., n_kv) per-position ‖V‖₂.
        alpha:         scaling strength (≥ 0).
        smooth_kernel: 1D avg-pool window over KV. Odd integer ≥ 1.
    """
    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            raise ValueError(f"smooth_kernel must be odd, got {smooth_kernel}")
        pad = (smooth_kernel - 1) // 2
        flat = v_norm.reshape(-1, 1, v_norm.shape[-1])
        smoothed = F.avg_pool1d(flat, kernel_size=smooth_kernel, stride=1, padding=pad)
        v_eff = smoothed.reshape(v_norm.shape)
    else:
        v_eff = v_norm
    v_eff = v_eff.clamp(min=eps)
    return mass * v_eff.pow(alpha)
