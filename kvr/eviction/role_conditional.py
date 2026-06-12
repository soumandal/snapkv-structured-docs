"""Role-conditional KV-eviction (Plan 2 Experiment A).

Given per-(layer, head, kv_pos) accumulated attention mass plus per-kv role
labels, allocate the total retention budget across role buckets according to
a fixed policy `α: role → fraction`. Within each role bucket, select the
top-mass tokens per (layer, head).

This is the **label-conditional** baseline — at inference time we assume
oracle access to role labels for each kv position. The point is to measure
the upper bound on what label-aware eviction can achieve; downstream
experiments (Plan 2 §F) can ask whether a probe could replicate the labels.

Allocation contract:
  * `allocation` maps each role string (KEY, VALUE, DELIM, HEADER, PROSE, WS,
    or any string label present in `roles_per_kv`) to a non-negative fraction.
  * The fractions are normalized to sum to 1 before slot allocation; a role
    absent from the dict gets fraction 0 (its tokens are never retained).
  * Per (layer, head), slot count `k_total = round(budget_frac · n_kv)`.
  * Per role r, target slots = round(α_r · k_total). If the role has fewer
    tokens than its target, the slack is redistributed proportionally to the
    other roles with non-zero α. If integer rounding leaves leftover slots,
    they go to the highest-α role.
  * Within each role's slot quota, retain top-mass tokens (per layer, head).

Returns: bool tensor (n_layers, n_heads, n_kv); True = kept.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch


def keep_mask_for_role_allocation(
    mass: torch.Tensor,             # (n_layers, n_heads, n_kv), fp32 or fp16
    roles_per_kv: list[str],        # length n_kv
    budget_frac: float,             # 0 < budget_frac ≤ 1
    allocation: dict[str, float] | Sequence[dict[str, float]],
) -> torch.Tensor:
    """Return boolean keep mask (n_layers, n_heads, n_kv).

    `allocation` may be a single dict (one policy applied to every layer; fast
    path, vectorized topk across layers) or a sequence of n_layers dicts (one
    policy per layer; per-layer topk loop).
    """
    if mass.ndim != 3:
        raise ValueError(f"mass must be 3-D (n_layers, n_heads, n_kv); got {mass.shape}")
    n_layers, n_heads, n_kv = mass.shape
    if len(roles_per_kv) != n_kv:
        raise ValueError(
            f"roles_per_kv has length {len(roles_per_kv)} but mass has n_kv={n_kv}"
        )
    if not (0.0 < budget_frac <= 1.0):
        raise ValueError(f"budget_frac must be in (0, 1]; got {budget_frac}")

    k_total = max(1, int(round(budget_frac * n_kv)))
    k_total = min(k_total, n_kv)

    # Group kv indices by role.
    role_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(roles_per_kv):
        role_to_idx[r].append(i)
    role_sizes_full = {r: len(role_to_idx[r]) for r in role_to_idx}

    if isinstance(allocation, dict):
        return _keep_mask_single_policy(
            mass, role_to_idx, role_sizes_full, k_total, allocation
        )

    # Per-layer path: a list/tuple of n_layers dicts.
    if len(allocation) != n_layers:
        raise ValueError(
            f"per-layer allocation has length {len(allocation)} but mass has "
            f"n_layers={n_layers}"
        )
    keep = torch.zeros_like(mass, dtype=torch.bool)
    for layer_idx, layer_alloc in enumerate(allocation):
        layer_mass = mass[layer_idx : layer_idx + 1]  # (1, n_heads, n_kv)
        layer_keep = _keep_mask_single_policy(
            layer_mass, role_to_idx, role_sizes_full, k_total, layer_alloc
        )
        keep[layer_idx] = layer_keep[0]
    return keep


def _keep_mask_single_policy(
    mass: torch.Tensor,
    role_to_idx: dict[str, list[int]],
    role_sizes_full: dict[str, int],
    k_total: int,
    allocation: dict[str, float],
) -> torch.Tensor:
    """Apply a single (role → weight) policy across all layers in `mass`.

    Shared between the dict fast path and the per-layer loop (where `mass` is
    sliced to a single layer)."""
    eligible_roles = [r for r in role_to_idx if allocation.get(r, 0.0) > 0.0]
    if not eligible_roles:
        # Pathological: every weight is zero. Fall back to H2O over all tokens
        # so we still produce a valid keep mask of size k_total.
        keep = torch.zeros_like(mass, dtype=torch.bool)
        _, top_idx = torch.topk(mass, k=k_total, dim=-1, largest=True, sorted=False)
        keep.scatter_(dim=-1, index=top_idx, value=True)
        return keep

    total_weight = sum(allocation.get(r, 0.0) for r in eligible_roles)
    norm_alpha = {r: allocation[r] / total_weight for r in eligible_roles}

    quotas = _allocate_slots(
        k_total=k_total,
        alpha=norm_alpha,
        role_sizes={r: role_sizes_full[r] for r in eligible_roles},
    )

    keep = torch.zeros_like(mass, dtype=torch.bool)
    for role, q in quotas.items():
        if q <= 0:
            continue
        idx = torch.tensor(role_to_idx[role], dtype=torch.long, device=mass.device)
        role_mass = mass.index_select(dim=-1, index=idx)
        _, top_local = torch.topk(role_mass, k=q, dim=-1, largest=True, sorted=False)
        top_global = idx[top_local]
        keep.scatter_(dim=-1, index=top_global, value=True)
    return keep


def _allocate_slots(
    k_total: int,
    alpha: dict[str, float],          # already normalized: sums to 1
    role_sizes: dict[str, int],       # tokens available per role
) -> dict[str, int]:
    """Distribute k_total slots across roles per `alpha`, redistributing slack
    when a role has fewer tokens than its target.
    """
    remaining = k_total
    quotas: dict[str, int] = {r: 0 for r in alpha}
    # Loop until either all slots are placed or no role can take more.
    # Each pass: compute target per role, clip to capacity, accumulate, then
    # recompute alpha over roles that still have capacity.
    active_alpha = dict(alpha)
    while remaining > 0 and active_alpha:
        total_active = sum(active_alpha.values())
        if total_active <= 0:
            break
        # Float targets for this pass.
        targets = {
            r: active_alpha[r] / total_active * remaining for r in active_alpha
        }
        # Integer floor, then largest-remainder rounding for the leftover.
        floors = {r: int(targets[r]) for r in targets}
        leftover = remaining - sum(floors.values())
        # Sort by fractional remainder desc, then by alpha desc for stability.
        order = sorted(
            targets,
            key=lambda r: (-(targets[r] - floors[r]), -active_alpha[r]),
        )
        for r in order[:leftover]:
            floors[r] += 1

        # Clip to capacity and tally absorbed/saturated.
        new_active: dict[str, float] = {}
        progressed = False
        for r in list(active_alpha):
            want = floors[r]
            already = quotas[r]
            cap = role_sizes[r]
            can_take = min(cap - already, want)
            if can_take > 0:
                quotas[r] += can_take
                remaining -= can_take
                progressed = True
            if quotas[r] < cap and want > can_take:
                # We wanted more but capped — should not happen because want ≤
                # remaining ≤ cap-already by construction, but keep defensively.
                new_active[r] = active_alpha[r]
            elif quotas[r] < cap and want == can_take:
                # Still has capacity for the next round if slack came in.
                new_active[r] = active_alpha[r]
            # else: saturated, drop from active set.
        if not progressed:
            break
        active_alpha = new_active
    return quotas
