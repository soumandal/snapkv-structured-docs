import math

import pytest
import torch

from kvr.instrumentation.scoring_recorder import entropy_score, value_prior_score


# ----- entropy_score (Method B) -----

def test_entropy_score_delta_distribution_is_full_score():
    mass = torch.tensor([[[1.0]]])
    plogp = torch.tensor([[[0.0]]])
    score = entropy_score(mass, plogp, alpha=1.0)
    assert torch.allclose(score, torch.tensor([[[1.0]]]), atol=1e-6)


def test_entropy_score_uniform_renormalized_gives_log_n():
    n, p = 4, 0.5
    mass = torch.tensor([[[n * p]]])
    plogp = torch.tensor([[[n * p * math.log(p)]]])
    score = entropy_score(mass, plogp, alpha=1.0)
    expected_entropy = math.log(n)
    expected_score = (n * p) / (1.0 + 1.0 * expected_entropy)
    assert torch.allclose(score, torch.tensor([[[expected_score]]]), atol=1e-5)


def test_entropy_score_higher_alpha_widens_score_gap():
    n = 8
    mass = torch.tensor([[[1.0, 1.0]]])
    plogp = torch.tensor([[[0.0, -math.log(n)]]])
    s_lo = entropy_score(mass, plogp, alpha=0.5)
    s_hi = entropy_score(mass, plogp, alpha=4.0)
    spread_lo = (s_lo[..., 0] - s_lo[..., 1]).item()
    spread_hi = (s_hi[..., 0] - s_hi[..., 1]).item()
    assert spread_hi > spread_lo > 0


def test_entropy_score_alpha_zero_is_h2o_mass():
    mass = torch.tensor([[[3.14, 0.5, 7.0]]])
    plogp = torch.tensor([[[-1.0, -0.2, -3.0]]])
    score = entropy_score(mass, plogp, alpha=0.0)
    assert torch.allclose(score, mass, atol=1e-6)


def test_entropy_score_zero_mass_finite():
    score = entropy_score(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), alpha=1.0)
    assert torch.isfinite(score).all()


def test_entropy_score_shape_preserved():
    n_layers, n_heads, n_kv = 4, 8, 16
    mass = torch.rand(n_layers, n_heads, n_kv) + 0.1
    plogp = -torch.rand(n_layers, n_heads, n_kv) * 2.0
    score = entropy_score(mass, plogp, alpha=1.5)
    assert score.shape == (n_layers, n_heads, n_kv)
    assert torch.isfinite(score).all()


# ----- value_prior_score (Method B' / VPE) -----

def test_value_prior_score_alpha_zero_is_h2o_mass():
    mass = torch.tensor([[[3.14, 0.5, 7.0]]])
    v_norm = torch.tensor([[[1e-3, 100.0, 5.0]]])  # arbitrary
    score = value_prior_score(mass, v_norm, alpha=0.0)
    assert torch.allclose(score, mass, atol=1e-6)


def test_value_prior_score_high_v_norm_boosts_score():
    # Equal mass; high v_norm should outrank low v_norm under any α > 0.
    mass = torch.tensor([[[1.0, 1.0]]])
    v_norm = torch.tensor([[[0.1, 10.0]]])
    score = value_prior_score(mass, v_norm, alpha=1.0)
    assert score[..., 1] > score[..., 0]


def test_value_prior_score_alpha_increases_v_norm_sensitivity():
    mass = torch.tensor([[[1.0, 1.0]]])
    v_norm = torch.tensor([[[1.0, 4.0]]])  # ratio 4×
    ratio_low = (value_prior_score(mass, v_norm, alpha=0.5)[..., 1]
                 / value_prior_score(mass, v_norm, alpha=0.5)[..., 0]).item()
    ratio_high = (value_prior_score(mass, v_norm, alpha=2.0)[..., 1]
                  / value_prior_score(mass, v_norm, alpha=2.0)[..., 0]).item()
    # ratio = (v_hi/v_lo)^α = 4^α; α=0.5 → 2.0, α=2.0 → 16.0.
    assert math.isclose(ratio_low, 2.0, rel_tol=1e-4)
    assert math.isclose(ratio_high, 16.0, rel_tol=1e-4)


def test_value_prior_score_smoothing_with_kernel_1_is_identity():
    mass = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    v_norm = torch.tensor([[[0.1, 10.0, 0.1, 10.0]]])
    no_smooth = value_prior_score(mass, v_norm, alpha=1.0, smooth_kernel=1)
    assert torch.allclose(no_smooth, mass * v_norm, atol=1e-6)


def test_value_prior_score_smoothing_pulls_neighbors_together():
    # With a 3-wide avg pool, two isolated high-norm spikes get averaged with
    # their low-norm neighbors → score gap shrinks.
    mass = torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0]]])
    v_norm = torch.tensor([[[0.0, 10.0, 0.0, 10.0, 0.0]]])
    raw = value_prior_score(mass, v_norm, alpha=1.0, smooth_kernel=1)
    smoothed = value_prior_score(mass, v_norm, alpha=1.0, smooth_kernel=3)
    raw_spread = (raw.max() - raw.min()).item()
    sm_spread = (smoothed.max() - smoothed.min()).item()
    assert sm_spread < raw_spread


def test_value_prior_score_rejects_even_kernel():
    with pytest.raises(ValueError):
        value_prior_score(torch.ones(1, 1, 4), torch.ones(1, 1, 4),
                          alpha=1.0, smooth_kernel=4)


def test_value_prior_score_shape_preserved_with_smoothing():
    n_layers, n_heads, n_kv = 4, 8, 16
    mass = torch.rand(n_layers, n_heads, n_kv) + 0.1
    v_norm = torch.rand(n_layers, n_heads, n_kv) + 0.1
    score = value_prior_score(mass, v_norm, alpha=1.0, smooth_kernel=7)
    assert score.shape == (n_layers, n_heads, n_kv)
    assert torch.isfinite(score).all()
