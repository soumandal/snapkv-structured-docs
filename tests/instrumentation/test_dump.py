import tempfile
from pathlib import Path

import polars as pl
import torch

from kvr.instrumentation.dump import load_prompt_dump, save_prompt_dump
from kvr.structure.roles import Role, TokenRole


def test_save_and_load_round_trip():
    mass = torch.rand(2, 3, 10)  # 2 layers, 3 heads, 10 kv positions
    token_roles = [TokenRole(token_id=i, role=Role.PROSE, depth=0) for i in range(10)]
    token_roles[1] = TokenRole(token_id=1, role=Role.KEY, depth=0)
    token_roles[5] = TokenRole(token_id=5, role=Role.VALUE, depth=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p001.parquet"
        save_prompt_dump(
            path,
            prompt_id="p001",
            source="synthetic_json",
            mass=mass,
            token_roles=token_roles,
        )
        df = load_prompt_dump(path)

    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"prompt_id", "source", "layer", "head", "kv_pos", "accumulated_mass", "role", "depth"}
    # 2 layers × 3 heads × 10 positions = 60 rows
    assert len(df) == 60
    # Role at kv_pos=1 should be KEY for all (layer, head).
    key_rows = df.filter(pl.col("kv_pos") == 1)
    assert all(r == "KEY" for r in key_rows["role"].to_list())
