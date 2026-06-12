import torch

from kvr.eviction.keep_mask import keep_mask_for_condition


def test_all_keeps_topk_by_mass():
    mass = torch.tensor([
        [[0.5, 0.1, 0.9, 0.2]],  # one layer, one head, 4 kv positions
    ])
    roles = ["VALUE", "DELIM", "KEY", "VALUE"]
    keep = keep_mask_for_condition(mass, roles, "All", budget_frac=0.5)
    # top 2 by mass = positions 2 (0.9) and 0 (0.5)
    assert keep.tolist() == [[[True, False, True, False]]]


def test_no_key_evicts_keys_and_promotes_next():
    mass = torch.tensor([
        [[0.5, 0.1, 0.9, 0.2]],
    ])
    roles = ["VALUE", "DELIM", "KEY", "VALUE"]
    keep = keep_mask_for_condition(mass, roles, "No-KEY", budget_frac=0.5)
    # KEY position (2) is zeroed; top 2 of remaining = positions 0 (0.5) and 3 (0.2)
    assert keep.tolist() == [[[True, False, False, True]]]


def test_no_delim_promotes_next():
    mass = torch.tensor([
        [[0.5, 1.0, 0.9, 0.2]],
    ])
    roles = ["VALUE", "DELIM", "KEY", "VALUE"]
    keep = keep_mask_for_condition(mass, roles, "No-DELIM", budget_frac=0.5)
    # DELIM (1) zeroed; top 2 of {0.5, 0, 0.9, 0.2} = positions 2, 0
    assert keep.tolist() == [[[True, False, True, False]]]


def test_per_layer_per_head_independent():
    # 2 layers, 2 heads, 4 kv positions.
    mass = torch.tensor([
        [[0.5, 0.1, 0.9, 0.2],
         [0.1, 0.9, 0.2, 0.5]],
        [[1.0, 0.0, 0.5, 0.0],
         [0.0, 1.0, 0.0, 0.5]],
    ])
    roles = ["VALUE", "DELIM", "KEY", "VALUE"]
    keep = keep_mask_for_condition(mass, roles, "All", budget_frac=0.5)
    assert keep[0, 0].tolist() == [True, False, True, False]
    assert keep[0, 1].tolist() == [False, True, False, True]
    assert keep[1, 0].tolist() == [True, False, True, False]
    assert keep[1, 1].tolist() == [False, True, False, True]


def test_budget_rounding_and_floor():
    mass = torch.rand(1, 1, 100)
    roles = ["VALUE"] * 100
    keep5 = keep_mask_for_condition(mass, roles, "All", budget_frac=0.05)
    keep10 = keep_mask_for_condition(mass, roles, "All", budget_frac=0.10)
    assert keep5.sum().item() == 5
    assert keep10.sum().item() == 10
