"""M1 h1/h2 sampler — written before haidc.data.sampler (TESTING.md).

Two distinct votes drawn WITHOUT replacement from the per-image urn {n_smooth, n_features}
(0=smooth/early, 1=features/spiral). Marginal frequencies must match the urn composition;
counts < 2 must raise. Within-urn draws are slightly negatively correlated by construction
(HANDOFF §6) — the positive human-human correlation lives in between-image urn variance.
"""
import numpy as np
import pytest

from haidc.data.sampler import draw_h1_h2


def test_raises_when_total_votes_below_two():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        draw_h1_h2(1, 0, rng)
    with pytest.raises(ValueError):
        draw_h1_h2(0, 0, rng)


def test_one_each_urn_always_yields_distinct_labels():
    # urn {1 smooth, 1 features}: two votes without replacement must be one of each.
    rng = np.random.default_rng(0)
    for _ in range(200):
        h1, h2 = draw_h1_h2(1, 1, rng)
        assert {h1, h2} == {0, 1}


def test_homogeneous_urn_returns_that_label():
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert draw_h1_h2(2, 0, rng) == (0, 0)
        assert draw_h1_h2(0, 2, rng) == (1, 1)


def test_returns_binary_labels():
    rng = np.random.default_rng(0)
    h1, h2 = draw_h1_h2(3, 7, rng)
    assert h1 in (0, 1) and h2 in (0, 1)


def test_marginal_frequency_matches_urn():
    # urn {3 smooth, 7 features}: P(draw == features) = 7/10 for each position (exchangeable).
    rng = np.random.default_rng(0)
    n = 20000
    draws = np.array([draw_h1_h2(3, 7, rng) for _ in range(n)])
    p1 = draws[:, 0].mean()
    p2 = draws[:, 1].mean()
    assert abs(p1 - 0.7) < 0.02
    assert abs(p2 - 0.7) < 0.02


def test_within_urn_draws_negatively_correlated():
    # hypergeometric (without replacement) -> Corr(h1, h2) < 0 within a fixed urn.
    rng = np.random.default_rng(0)
    draws = np.array([draw_h1_h2(5, 5, rng) for _ in range(20000)])
    corr = np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]
    assert corr < 0
