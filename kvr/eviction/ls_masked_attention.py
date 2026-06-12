"""LS-projection masked attention — preserves a low-rank summary of evicted KV.

Like `MaskedAttentionPatch` (selects top-k positions to retain at decode),
but during the prefill forward it ALSO captures a rank-`ls_rank` summary
of the would-be-evicted K and V vectors per (layer, head). During decode
it appends these summary slots to the K/V tensors so the decoder can
attend to a coarse approximation of the lost content. This is orthogonal
to the *selection* axis (any selector can be wrapped) — it lives on the
*information-loss* axis.

v0: `ls_rank=1` only. The summary is the simple mean of evicted K (post-
RoPE) and mean of evicted V per (layer, head). Higher-rank variants (top-r
SVD or k-means centroids) are a natural follow-up if rank-1 already moves
the needle.

Caveat on RoPE: the summary K is an average of position-rotated K vectors,
so it is *position-agnostic*. The decode query's RoPE-rotation interacts
with it as if attending to a positionless "evicted-bucket" representative.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class LSMaskedAttentionPatch:
    """Like MaskedAttentionPatch but adds rank-r LS-projection summary slots.

    Args:
        model: HF Llama model.
        keep_masks: (n_layers, n_heads, prompt_len) bool on the model device.
        ls_rank: number of summary slots per (layer, head). v0 supports 1.
    """

    def __init__(self, model, keep_masks: torch.Tensor, ls_rank: int = 1,
                 anchor_strategy: str = "mean_evicted_pos",
                 collect_summary_weights: bool = False) -> None:
        if keep_masks.dtype != torch.bool:
            raise ValueError(f"keep_masks must be bool, got {keep_masks.dtype}")
        if keep_masks.ndim != 3:
            raise ValueError("keep_masks must be (n_layers, n_heads, prompt_len)")
        if anchor_strategy not in {"post_rope_mean", "mean_evicted_pos",
                                    "content_weighted", "kmeans"}:
            raise ValueError(f"unknown anchor_strategy: {anchor_strategy!r}")
        if ls_rank > 1 and anchor_strategy != "kmeans":
            raise ValueError(f"ls_rank > 1 only valid with anchor_strategy='kmeans'")
        if ls_rank < 1:
            raise ValueError(f"ls_rank must be >= 1, got {ls_rank}")
        self.model = model
        self.keep_masks = keep_masks
        self.ls_rank = ls_rank
        # anchor_strategy:
        #   "post_rope_mean"   — v0 baseline. Mean of post-RoPE K (no anchor).
        #                        Position-agnostic; decoder ignores it.
        #   "mean_evicted_pos" — Option A. Mean of pre-RoPE K, then apply RoPE
        #                        at the per-head mean of evicted positions.
        #                        Restores q·k geometry; but rank-1 mean is
        #                        DELIM-cluster-dominated → still ignored.
        #   "content_weighted" — Option C2. Weight each evicted token's K and V
        #                        by ‖V_t‖ (content signal) before averaging.
        #                        Tilts the summary away from DELIM-like sinks
        #                        whose value vectors are near-zero. Anchor at
        #                        the V-weighted mean evicted position.
        #   "kmeans"           — Option B. K-means cluster the evicted pre-RoPE
        #                        K vectors into `ls_rank` clusters per (layer,
        #                        head); each cluster centroid becomes a summary
        #                        K (re-rotated at the cluster's mean evicted
        #                        position) and its V mean becomes summary V.
        #                        ls_rank ≥ 2 captures multiple modes of the
        #                        evicted content, addressing the failure mode
        #                        of rank-1 mean (too-smooth single vector).
        self.anchor_strategy = anchor_strategy
        self.collect_summary_weights = collect_summary_weights
        # Per-layer (k_summary, v_summary): each (n_heads, ls_rank, head_dim).
        # Filled in during the prefill forward; consumed during decode.
        self._summaries: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # When `collect_summary_weights` is on: per-layer list of (n_heads,)
        # tensors, one per decode step — the softmax weight assigned to the
        # summary slot. Used to verify the decoder is actually attending to it.
        self._summary_weights: dict[int, list[torch.Tensor]] = {}
        # Diagnostic only: per-layer (k_summary_norm, mean_k_kept_norm), each
        # (n_heads,). Populated at prefill when `collect_summary_weights`.
        self._norm_stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
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
                    f"Layer {layer_idx} self_attn already has an instance-level forward"
                )
            attn.forward = _make_ls_masked_forward(attn, self.keep_masks[layer_idx], self)
            self._patched.append(attn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for attn in self._patched:
            del attn.forward
        self._patched = []
        self._summaries = {}
        return False

    def norm_stats(self) -> dict[int, dict[str, float]]:
        """Per-layer summary-vs-kept K-norm comparison. Tells us whether the
        summary slot is being ignored because its K has near-zero magnitude.
        """
        out: dict[int, dict[str, float]] = {}
        for layer_idx, (k_sum_n, k_kept_n) in self._norm_stats.items():
            ratio = (k_sum_n / k_kept_n.clamp(min=1e-9))
            out[layer_idx] = {
                "summary_norm_mean": float(k_sum_n.mean()),
                "kept_norm_mean": float(k_kept_n.mean()),
                "ratio_mean": float(ratio.mean()),
                "ratio_min": float(ratio.min()),
                "ratio_max": float(ratio.max()),
            }
        return out

    def summary_weight_stats(self) -> dict[int, dict[str, float]]:
        """Per-layer aggregate stats for the softmax weight on the summary slot
        across all decode steps. Only populated when
        `collect_summary_weights=True`. Returns {} otherwise.

        For each layer: `mean` and `max` over (n_steps × n_heads), plus
        `head_max_mean` = mean across heads of (max weight across decode steps).
        """
        out: dict[int, dict[str, float]] = {}
        for layer_idx, weights in self._summary_weights.items():
            stacked = torch.stack(weights)  # (n_steps, n_heads)
            head_max = stacked.max(dim=0).values  # (n_heads,)
            out[layer_idx] = {
                "mean": float(stacked.mean()),
                "max": float(stacked.max()),
                "head_max_mean": float(head_max.mean()),
                "n_steps": int(stacked.shape[0]),
                "n_heads": int(stacked.shape[1]),
            }
        return out

    def _iter_decoder_layers(self):
        base = getattr(self.model, "model", self.model)
        return base.layers


def _kmeans_evicted(
    k_pre: torch.Tensor,      # (n_heads, prompt_len, head_dim) — pre-RoPE K
    v: torch.Tensor,          # (n_heads, prompt_len, head_dim) — V
    evict_mask_f: torch.Tensor,  # (n_heads, prompt_len), float, 1=evicted
    positions: torch.Tensor,  # (prompt_len,), float
    r: int,
    n_iter: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched-per-head k-means on evicted pre-RoPE K vectors.

    Returns:
        centroids:  (n_heads, r, head_dim) pre-RoPE.
        v_means:    (n_heads, r, head_dim) cluster V means.
        anchor_pos: (n_heads, r) cluster mean evicted position (float).
    """
    n_heads, prompt_len, hd = k_pre.shape
    device = k_pre.device
    dtype = k_pre.dtype
    large = torch.tensor(1e9, device=device, dtype=torch.float32)

    # Init: r evenly-spaced positions across the prompt (shared across heads).
    # Kept positions among these will be re-assigned in the first Lloyd step.
    init_idx = torch.linspace(0, prompt_len - 1, r, device=device).long()
    centroids = k_pre[:, init_idx].clone()  # (n_heads, r, hd)

    # Promote to fp32 for stable cluster math.
    k_f32 = k_pre.to(torch.float32)
    v_f32 = v.to(torch.float32)
    em = evict_mask_f.to(torch.float32)
    k_norm_sq = (k_f32 ** 2).sum(dim=-1, keepdim=True)  # (n_heads, prompt_len, 1)

    for _ in range(n_iter):
        c = centroids.to(torch.float32)
        c_norm_sq = (c ** 2).sum(dim=-1)  # (n_heads, r)
        kc = torch.bmm(k_f32, c.transpose(-1, -2))  # (n_heads, prompt_len, r)
        dist = k_norm_sq - 2 * kc + c_norm_sq.unsqueeze(1)
        # Penalize kept positions so they never get assigned.
        dist = dist + (1.0 - em).unsqueeze(-1) * large
        assign = dist.argmin(dim=-1)  # (n_heads, prompt_len)
        one_hot = F.one_hot(assign, num_classes=r).to(torch.float32)
        one_hot = one_hot * em.unsqueeze(-1)  # zero out kept positions
        counts = one_hot.sum(dim=-2)  # (n_heads, r)
        sums = torch.bmm(one_hot.transpose(-1, -2), k_f32)  # (n_heads, r, hd)
        centroids = (sums / counts.unsqueeze(-1).clamp(min=1.0)).to(dtype)

    # Final pass for v_means and anchor_pos using the converged assignments.
    c = centroids.to(torch.float32)
    c_norm_sq = (c ** 2).sum(dim=-1)
    kc = torch.bmm(k_f32, c.transpose(-1, -2))
    dist = k_norm_sq - 2 * kc + c_norm_sq.unsqueeze(1)
    dist = dist + (1.0 - em).unsqueeze(-1) * large
    assign = dist.argmin(dim=-1)
    one_hot = F.one_hot(assign, num_classes=r).to(torch.float32)
    one_hot = one_hot * em.unsqueeze(-1)
    counts = one_hot.sum(dim=-2).clamp(min=1.0)

    v_means = (torch.bmm(one_hot.transpose(-1, -2), v_f32)
               / counts.unsqueeze(-1)).to(dtype)
    pos_sums = torch.einsum("hpr,p->hr", one_hot, positions.to(torch.float32))
    anchor_pos = pos_sums / counts

    return centroids, v_means, anchor_pos


def _make_ls_masked_forward(
    attn_module, layer_keep_mask: torch.Tensor, patch: LSMaskedAttentionPatch,
):
    """Forward closure with an LS-summary slot appended during decode."""
    layer_idx = attn_module.layer_idx

    def ls_masked_forward(
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
        # Snapshot pre-RoPE k for summary aggregation (only matters at prefill;
        # any strategy that needs to re-rotate the summary at an anchor needs
        # the un-rotated K).
        k_pre_rope = k if (q_len > 1 and patch.anchor_strategy in {"mean_evicted_pos", "content_weighted", "kmeans"}) else None
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, attn_module.layer_idx, cache_kwargs)

        k = repeat_kv(k, attn_module.num_key_value_groups)
        v = repeat_kv(v, attn_module.num_key_value_groups)

        kv_len = k.shape[-2]
        prompt_len = layer_keep_mask.shape[-1]
        scale = 1.0 / math.sqrt(attn_module.head_dim)

        if q_len > 1:
            # Prefill: capture LS summary from evicted positions, then run
            # standard causal attention via SDPA (matches Approach A).
            # k, v shape: (1, n_heads, prompt_len, head_dim) at prefill.
            evict_mask = (~layer_keep_mask).to(k.dtype)  # (n_heads, prompt_len)
            positions = torch.arange(prompt_len, device=k.device, dtype=torch.float32)

            if patch.anchor_strategy == "post_rope_mean":
                # v0 — average over post-RoPE K. Decoder gives near-zero weight.
                # fp32: attention-sink K activations make the fp16 sum overflow.
                k_f32 = k[0].to(torch.float32)
                v_f32 = v[0].to(torch.float32)
                em_f32 = evict_mask.to(torch.float32)
                denom = em_f32.sum(dim=-1, keepdim=True).clamp(min=1.0)
                k_summary = ((k_f32 * em_f32.unsqueeze(-1)).sum(dim=-2) / denom).to(k.dtype)
                v_summary = ((v_f32 * em_f32.unsqueeze(-1)).sum(dim=-2) / denom).to(v.dtype)
            elif patch.anchor_strategy == "kmeans":
                k_pre_full = repeat_kv(k_pre_rope, attn_module.num_key_value_groups)
                centroids, v_means, anchor_pos_f = _kmeans_evicted(
                    k_pre=k_pre_full[0], v=v[0], evict_mask_f=evict_mask,
                    positions=positions, r=patch.ls_rank, n_iter=5,
                )
                # Per-cluster RoPE anchor. anchor_pos: (n_heads, r) → flatten
                # to gather per-(head, cluster) rows of cos/sin.
                anchor_pos = anchor_pos_f.round().long().clamp(0, prompt_len - 1)
                flat = anchor_pos.reshape(-1)
                cos_anchor = cos[0][flat].reshape(*anchor_pos.shape, -1).to(centroids.dtype)
                sin_anchor = sin[0][flat].reshape(*anchor_pos.shape, -1).to(centroids.dtype)
                hd = centroids.shape[-1]
                rot_half = torch.cat(
                    [-centroids[..., hd // 2:], centroids[..., : hd // 2]], dim=-1
                )
                # ks shape (n_heads, r, head_dim) — already has the rank dim.
                k_summary = centroids * cos_anchor + rot_half * sin_anchor
                v_summary = v_means
            else:
                # Both "mean_evicted_pos" (uniform) and "content_weighted" share
                # the pre-RoPE-then-rotate-at-anchor structure, differing only
                # in the per-position weight used for averaging.
                # fp32 throughout — fp16 sums across thousands of evicted
                # positions overflow at attention-sink layers (large K/V mags),
                # then RoPE rotation mixes +inf/-inf into NaN.
                k_pre_full = repeat_kv(k_pre_rope, attn_module.num_key_value_groups)
                k_pre_f32 = k_pre_full[0].to(torch.float32)
                v_f32 = v[0].to(torch.float32)
                em_f32 = evict_mask.to(torch.float32)
                if patch.anchor_strategy == "mean_evicted_pos":
                    weight = em_f32
                else:  # "content_weighted"
                    # Weight by ‖V_t‖ so the summary tilts toward semantic
                    # tokens and away from low-‖V‖ routing sinks.
                    v_norm_per_pos = v_f32.norm(dim=-1)
                    weight = em_f32 * v_norm_per_pos

                w_denom_kd = weight.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                k_summary_pre = (
                    (k_pre_f32 * weight.unsqueeze(-1)).sum(dim=-2) / w_denom_kd
                )  # (n_heads, head_dim) fp32
                v_summary = (
                    ((v_f32 * weight.unsqueeze(-1)).sum(dim=-2) / w_denom_kd).to(v.dtype)
                )

                # Anchor: weighted mean of evicted positions per head.
                anchor_pos_f = (weight * positions).sum(dim=-1) / weight.sum(dim=-1).clamp(min=1e-6)
                anchor_pos = anchor_pos_f.round().long().clamp(0, prompt_len - 1)

                # cos, sin during prefill: (bsz=1, q_len, head_dim). Gather
                # per-head anchor row → (n_heads, head_dim).
                cos_anchor = cos[0][anchor_pos].to(torch.float32)
                sin_anchor = sin[0][anchor_pos].to(torch.float32)
                hd = k_summary_pre.shape[-1]
                rot_half = torch.cat(
                    [-k_summary_pre[..., hd // 2:], k_summary_pre[..., : hd // 2]],
                    dim=-1,
                )
                k_summary = (k_summary_pre * cos_anchor + rot_half * sin_anchor).to(k.dtype)

            # Normalize to (n_heads, ls_rank, head_dim). Non-kmeans branches
            # produce (n_heads, head_dim) — add the rank dim of size 1.
            if k_summary.dim() == 2:
                k_summary = k_summary.unsqueeze(1)
                v_summary = v_summary.unsqueeze(1)
            patch._summaries[layer_idx] = (k_summary, v_summary)

            if patch.collect_summary_weights:
                # Diagnostic: compare ‖k_summary‖ to mean ‖k_kept‖ per head.
                # Compute in fp32 to avoid overflow at long context.
                keep_f = (1.0 - evict_mask).to(torch.float32)
                k_norms_per_pos = k[0].to(torch.float32).norm(dim=-1)
                keep_denom = keep_f.sum(dim=-1).clamp(min=1.0)
                k_kept_norms = (k_norms_per_pos * keep_f).sum(dim=-1) / keep_denom
                # For kmeans (rank>1), report the per-head MAX over clusters
                # — the strongest summary slot is what matters for softmax.
                # k_summary at this point is (n_heads, r, head_dim).
                k_summary_norms = k_summary.to(torch.float32).norm(dim=-1).max(dim=-1).values
                patch._norm_stats[layer_idx] = (
                    k_summary_norms.detach().cpu(),
                    k_kept_norms.detach().cpu(),
                )

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
            # Decode: append the summary slot(s) and apply the eviction bias
            # only to the original prompt positions.
            k_summary, v_summary = patch._summaries[layer_idx]
            # k_summary: (n_heads, ls_rank, head_dim) — add bsz dim
            k_ext = torch.cat([k, k_summary.unsqueeze(0)], dim=-2)  # (1, n_heads, kv_len+r, hd)
            v_ext = torch.cat([v, v_summary.unsqueeze(0)], dim=-2)

            scores = torch.matmul(q, k_ext.transpose(2, 3)) * scale
            device, dtype = scores.device, scores.dtype

            # Eviction bias over the original prompt positions; summary slot(s)
            # and decode-generated positions (between prompt_len and kv_len)
            # are always retained.
            ext_len = k_ext.shape[-2]
            evict_bias = torch.zeros(
                attn_module.num_heads, ext_len, device=device, dtype=dtype
            )
            evict_bias[:, :prompt_len] = (~layer_keep_mask).to(device=device, dtype=dtype) * -1e4
            scores = scores + evict_bias[None, :, None, :]

            if attention_mask is not None:
                # Pad the attention mask to match the extended kv_len.
                mask = attention_mask[:, :, :q_len, :kv_len]
                pad_w = ext_len - mask.shape[-1]
                if pad_w > 0:
                    mask = F.pad(mask, (0, pad_w), value=0.0)
                scores = scores + mask

            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            if patch.collect_summary_weights:
                # `ls_rank` summary slot(s) are the last `ls_rank` positions
                # along the kv axis. Capture per-head TOTAL weight across all
                # summary slots — the headline diagnostic is "does the decoder
                # attend to *any* summary slot."
                r = patch.ls_rank
                w = probs[0, :, 0, -r:].detach().to(torch.float32).sum(dim=-1).cpu()
                patch._summary_weights.setdefault(layer_idx, []).append(w)
            out = torch.matmul(probs, v_ext)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_module.o_proj(out)

        return out, None, past_key_value

    return ls_masked_forward
