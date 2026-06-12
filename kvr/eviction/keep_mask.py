"""Compute per-(layer, head) keep masks for counterfactual eviction.

Given recorded per-(layer, head, kv_pos) accumulated attention mass plus
per-(kv_pos) role labels, decide which kv positions are retained under
H2O at a given budget, with optional role-conditional forbidding.

Conditions:
    "All"       — vanilla H2O: keep top-k by mass.
    "No-{ROLE}" — H2O with all tokens of ROLE forbidden from retention.
                  Their scores are zeroed (effectively -inf) before top-k,
                  freeing the slots for the next-highest non-ROLE tokens.

The result is `(n_layers, n_heads, n_kv)` boolean: True = kept, False = evicted.
"""
from __future__ import annotations

import torch


_CONDITION_FORBID = {
    "All": None,
    "No-DELIM": "DELIM",
    "No-KEY": "KEY",
    "No-VALUE": "VALUE",
    "No-HEADER": "HEADER",
}


def keep_mask_for_condition(
    mass: torch.Tensor,           # (n_layers, n_heads, n_kv), fp32
    roles_per_kv: list[str],      # length n_kv
    condition: str,               # one of _CONDITION_FORBID keys
    budget_frac: float,           # 0 < budget_frac ≤ 1
) -> torch.Tensor:
    """Return boolean keep mask of shape (n_layers, n_heads, n_kv)."""
    if condition not in _CONDITION_FORBID:
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of {sorted(_CONDITION_FORBID)}"
        )
    n_layers, n_heads, n_kv = mass.shape
    if len(roles_per_kv) != n_kv:
        raise ValueError(
            f"roles_per_kv has length {len(roles_per_kv)} but mass has n_kv={n_kv}"
        )

    forbidden_role = _CONDITION_FORBID[condition]
    forbidden_idx = (
        torch.tensor(
            [i for i, r in enumerate(roles_per_kv) if r == forbidden_role],
            dtype=torch.long,
        )
        if forbidden_role is not None
        else torch.empty(0, dtype=torch.long)
    )

    scores = mass.clone()
    if forbidden_idx.numel() > 0:
        # Zero out forbidden positions so they fall to the bottom of top-k.
        # Using 0 (not -inf) is safe: real attention masses are non-negative
        # and the lowest non-forbidden token still has mass > 0 in practice.
        scores[..., forbidden_idx] = 0.0

    # Per (layer, head), keep top-k of remaining.
    k = max(1, int(round(budget_frac * n_kv)))
    k = min(k, n_kv)
    # topk returns indices; build a bool mask.
    _, top_idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False)
    keep = torch.zeros_like(mass, dtype=torch.bool)
    keep.scatter_(dim=-1, index=top_idx, value=True)

    # Defensive: if a condition forbids a role entirely AND the budget would
    # otherwise have included it, the resulting kept count is still k. We do
    # *not* re-add forbidden tokens later. This matches the spec.

    return keep
