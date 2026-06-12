import pytest
import torch

from kvr.eviction.role_conditional import keep_mask_for_role_allocation


def test_mass_proportional_equals_topk_when_alpha_matches_distribution():
    # 8 positions, 4 VALUE, 4 KEY. Mass: VALUE=1.0 each, KEY=2.0 each.
    # If we allocate proportional to mass-share (VALUE:0.33, KEY:0.67),
    # at budget 0.5 (k_total=4) we expect 1 VALUE + 3 KEY slots, but they
    # round to 1 and 3 by largest-remainder.
    mass = torch.tensor([[[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]]])
    roles = ["VALUE"] * 4 + ["KEY"] * 4
    keep = keep_mask_for_role_allocation(mass, roles, 0.5, {"VALUE": 1.0, "KEY": 2.0})
    # 1 VALUE (any one), 3 KEY (top-3 of KEY mass — all tied at 2.0, any 3).
    assert keep.sum().item() == 4
    assert keep[0, 0, :4].sum().item() == 1
    assert keep[0, 0, 4:].sum().item() == 3


def test_value_only_keeps_top_value():
    # 4 VALUE, 4 KEY. VALUE mass: 0.1, 0.9, 0.2, 0.8. KEY mass: 10.0 each.
    # allocation={VALUE:1.0} → all slots go to top-mass VALUE positions.
    mass = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 10.0, 10.0, 10.0, 10.0]]])
    roles = ["VALUE", "VALUE", "VALUE", "VALUE", "KEY", "KEY", "KEY", "KEY"]
    keep = keep_mask_for_role_allocation(mass, roles, 0.25, {"VALUE": 1.0})
    # k_total = round(0.25 * 8) = 2. Both go to VALUE → positions 1 and 3.
    assert keep[0, 0].tolist() == [False, True, False, True, False, False, False, False]


def test_zero_allocation_for_role_evicts_all_of_role():
    mass = torch.tensor([[[0.5, 0.9, 0.4, 0.8]]])
    roles = ["VALUE", "KEY", "VALUE", "KEY"]
    # KEY:0, VALUE:1 → all 2 slots go to VALUE positions 0, 2.
    keep = keep_mask_for_role_allocation(
        mass, roles, 0.5, {"VALUE": 1.0, "KEY": 0.0}
    )
    assert keep[0, 0].tolist() == [True, False, True, False]


def test_slack_redistribution_when_role_smaller_than_quota():
    # 1 VALUE, 7 KEY. allocation VALUE:0.5, KEY:0.5 at budget 1.0 (k=8).
    # Naive quotas: VALUE=4, KEY=4. But there's only 1 VALUE token.
    # Expected: VALUE=1 (all of it), KEY=7 (all of it). Total 8.
    mass = torch.ones(1, 1, 8)
    roles = ["VALUE"] + ["KEY"] * 7
    keep = keep_mask_for_role_allocation(
        mass, roles, 1.0, {"VALUE": 0.5, "KEY": 0.5}
    )
    assert keep.sum().item() == 8


def test_slack_redistribution_to_other_active_role():
    # 1 VALUE, 7 KEY, allocation VALUE:0.5, KEY:0.5 at budget 0.5 (k=4).
    # Initial: VALUE=2, KEY=2. VALUE has only 1, so the 1 leftover goes to KEY.
    # Final: VALUE=1, KEY=3, total 4.
    mass = torch.ones(1, 1, 8)
    roles = ["VALUE"] + ["KEY"] * 7
    keep = keep_mask_for_role_allocation(
        mass, roles, 0.5, {"VALUE": 0.5, "KEY": 0.5}
    )
    kept = keep[0, 0]
    assert kept[0].item() is True   # the only VALUE is kept
    assert kept[1:].sum().item() == 3  # plus 3 KEYs
    assert kept.sum().item() == 4


def test_per_layer_per_head_independent_selection():
    # 4 KEY tokens with different mass per (layer, head). VALUE:1.0, KEY:0.
    mass = torch.tensor([
        [[0.0, 0.0, 5.0, 1.0, 2.0, 3.0]],  # layer 0 head 0
        [[0.0, 0.0, 1.0, 5.0, 3.0, 2.0]],  # layer 1 head 0
    ])
    roles = ["VALUE", "VALUE", "KEY", "KEY", "KEY", "KEY"]
    keep = keep_mask_for_role_allocation(
        mass, roles, budget_frac=1 / 3, allocation={"KEY": 1.0}
    )
    # k_total = round(6/3)=2, all to KEY. Layer 0: positions 2, 5 (top 2 = 5,3).
    # Layer 1: positions 3, 4 (top 2 = 5,3).
    assert keep[0, 0].tolist() == [False, False, True, False, False, True]
    assert keep[1, 0].tolist() == [False, False, False, True, True, False]


def test_budget_count_matches_round_b_times_n():
    torch.manual_seed(0)
    mass = torch.rand(2, 4, 100)
    roles = (["VALUE"] * 50) + (["KEY"] * 30) + (["DELIM"] * 20)
    for b in [0.05, 0.10, 0.30, 0.50]:
        keep = keep_mask_for_role_allocation(
            mass, roles, b,
            {"VALUE": 0.5, "KEY": 0.0, "DELIM": 0.5},
        )
        expected = max(1, int(round(b * 100)))
        # Per (layer, head), the kept count equals expected.
        per_lh = keep.sum(dim=-1)
        assert (per_lh == expected).all(), (b, per_lh, expected)


def test_all_zero_allocation_falls_back_to_h2o():
    # Pathological: every role has alpha=0. We don't want a crash; we want
    # a valid mask of size k_total. Documented as the fallback.
    mass = torch.tensor([[[0.5, 0.1, 0.9, 0.2]]])
    roles = ["VALUE"] * 4
    keep = keep_mask_for_role_allocation(
        mass, roles, 0.5, {"KEY": 0.0}  # no role in allocation matches corpus roles
    )
    # Fallback selects top-2 by mass = positions 2 (0.9), 0 (0.5).
    assert keep[0, 0].tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# Per-layer allocation
# ---------------------------------------------------------------------------


def test_per_layer_list_all_same_equals_dict_path():
    torch.manual_seed(1)
    mass = torch.rand(3, 2, 20)
    roles = (["VALUE"] * 12) + (["KEY"] * 8)
    alloc = {"VALUE": 0.7, "KEY": 0.3}
    keep_dict = keep_mask_for_role_allocation(mass, roles, 0.4, alloc)
    keep_list = keep_mask_for_role_allocation(mass, roles, 0.4, [alloc] * 3)
    assert torch.equal(keep_dict, keep_list)


def test_per_layer_different_alpha_differs_from_uniform():
    # Layer 0 evicts all KEY (α_KEY=0); layer 1 keeps KEY-only (α_KEY=1).
    # The masks must differ on KEY positions.
    mass = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],  # layer 0 head 0
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],  # layer 1 head 0
    ])
    roles = ["VALUE", "VALUE", "VALUE", "KEY", "KEY", "KEY"]
    per_layer = [
        {"VALUE": 1.0, "KEY": 0.0},  # layer 0: VALUE only
        {"VALUE": 0.0, "KEY": 1.0},  # layer 1: KEY only
    ]
    keep = keep_mask_for_role_allocation(mass, roles, 0.5, per_layer)
    # k_total = round(0.5 * 6) = 3.
    # Layer 0: top-3 VALUE → positions 0, 1, 2 (all of VALUE).
    assert keep[0, 0].tolist() == [True, True, True, False, False, False]
    # Layer 1: top-3 KEY → positions 3, 4, 5.
    assert keep[1, 0].tolist() == [False, False, False, True, True, True]


def test_per_layer_length_mismatch_raises():
    mass = torch.rand(3, 1, 8)
    roles = ["VALUE"] * 8
    with pytest.raises(ValueError, match=r"per-layer allocation has length 2"):
        keep_mask_for_role_allocation(
            mass, roles, 0.5, [{"VALUE": 1.0}, {"VALUE": 1.0}]
        )


def test_per_layer_mixed_alpha_keeps_expected_counts():
    # 3 layers, each with a different α_KEY. Sanity check that per-layer
    # quotas land where they should under slack-free conditions.
    torch.manual_seed(2)
    mass = torch.rand(3, 2, 100)
    roles = (["VALUE"] * 50) + (["KEY"] * 50)
    per_layer = [
        {"VALUE": 1.0, "KEY": 0.0},   # layer 0: pure no-key
        {"VALUE": 0.98, "KEY": 0.02}, # layer 1: soft α=0.02
        {"VALUE": 0.5, "KEY": 0.5},   # layer 2: equal
    ]
    keep = keep_mask_for_role_allocation(mass, roles, 0.4, per_layer)
    # k_total = round(0.4 * 100) = 40.
    per_lh = keep.sum(dim=-1)
    assert (per_lh == 40).all(), per_lh
    # Layer 0: 40 VALUE, 0 KEY.
    assert keep[0, :, :50].sum().item() == 80   # 40 per head × 2 heads
    assert keep[0, :, 50:].sum().item() == 0
    # Layer 2: 20 VALUE, 20 KEY per head → 40+40 totals.
    assert keep[2, :, :50].sum().item() == 40
    assert keep[2, :, 50:].sum().item() == 40
