"""Persist + load per-prompt attention summaries as parquet.

We don't save full (q, k) attention matrices — too large at long context.
Instead we save the already-accumulated mass per (layer, head, kv_pos)
that the `AttentionRecorder` produces, plus per-token role metadata.
This is what every pilot analysis question needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from kvr.structure.roles import TokenRole


def save_prompt_dump(
    path: Path,
    *,
    prompt_id: str,
    source: str,
    mass: torch.Tensor | np.ndarray,  # (n_layers, n_heads, n_kv)
    token_roles: list[TokenRole],
) -> None:
    """Persist accumulated attention mass + role labels for one prompt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mass, torch.Tensor):
        arr = mass.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(mass, dtype=np.float32)
    n_layers, n_heads, n_kv = arr.shape
    assert n_kv == len(token_roles), (
        f"token_roles length {len(token_roles)} != attention n_kv {n_kv}"
    )

    rows: list[dict] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            for pos in range(n_kv):
                rows.append({
                    "prompt_id": prompt_id,
                    "source": source,
                    "layer": layer,
                    "head": head,
                    "kv_pos": pos,
                    "accumulated_mass": float(arr[layer, head, pos]),
                    "role": token_roles[pos].role.value,
                    "depth": token_roles[pos].depth,
                })
    df = pl.DataFrame(rows)
    df.write_parquet(path)


def load_prompt_dump(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
