import torch

from kvr.eviction.h2o import simulate_h2o_eviction


def test_h2o_keeps_top_k_by_accumulated_attention():
    # 1 layer, 1 head, 5 source positions, all attending to the same single query.
    # Attention pattern: pos 2 gets the most mass, pos 0 the least.
    # attn_weights: shape (n_layers, n_heads, n_query_tokens, n_kv_tokens)
    attn = torch.tensor([[[[0.05, 0.10, 0.50, 0.20, 0.15]]]])  # 1×1×1×5
    keep = simulate_h2o_eviction(attn, budget=3)
    # Should keep positions 2 (0.50), 3 (0.20), 4 (0.15) — top 3 by mass.
    assert set(keep[0][0].tolist()) == {2, 3, 4}


def test_h2o_returns_one_keep_set_per_head():
    # 1 layer, 2 heads, different attention patterns.
    attn = torch.zeros(1, 2, 1, 4)
    attn[0, 0, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    attn[0, 1, 0] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    keep = simulate_h2o_eviction(attn, budget=2)
    assert set(keep[0][0].tolist()) == {2, 3}
    assert set(keep[0][1].tolist()) == {0, 1}


def test_h2o_budget_larger_than_seq_keeps_all():
    attn = torch.tensor([[[[0.1, 0.2, 0.3]]]])
    keep = simulate_h2o_eviction(attn, budget=10)
    assert set(keep[0][0].tolist()) == {0, 1, 2}


def test_h2o_accumulates_attention_over_query_tokens():
    # Two query tokens, one heavily attends to pos 0, the other to pos 4.
    # Accumulated mass: pos 0 gets 1.0, pos 4 gets 1.0, others get 0.
    attn = torch.zeros(1, 1, 2, 5)
    attn[0, 0, 0, 0] = 1.0
    attn[0, 0, 1, 4] = 1.0
    keep = simulate_h2o_eviction(attn, budget=2)
    assert set(keep[0][0].tolist()) == {0, 4}
