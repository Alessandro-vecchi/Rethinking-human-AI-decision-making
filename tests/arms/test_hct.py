"""HCT arm — written before haidc.arms.hct (TESTING.md: red -> green).

HCT is a DECISION RULE, not a trainable model (GROUND_TRUTH §2, HANDOFF §8). Per instance, per
AI threshold theta: ai_label = 1[score >= theta]; draw (h1, h2) from the urn (M1 sampler).
  agree (ai_label == h1) -> decision = that label,  human_queries = 1
  disagree                -> decision = h2,          human_queries = 2
Expected human cost = 1 + P(ai != h1), bounded in [1, 2] for every theta.

These tests cover: the pure rule on hand-built cases (cheapest place to catch the worst bug),
the [1,2] cost bound + its identity with 1+P(disagree), the two limiting regimes, the label
encoding guard (1 = spiral everywhere), and seed determinism. Draws are made ONCE per
(instance, seed) and reused across theta (only ai_label moves with theta).
"""
import numpy as np
import pandas as pd
import pytest

from haidc.arms.hct import apply_hct, run_sweep


# --------------------------------------------------------------------------- pure rule
def test_apply_hct_agree_costs_one_and_keeps_label():
    # ai_label == h1 -> decision is that label, 1 query (h2 irrelevant)
    assert apply_hct(1, 1, 0) == (1, 1)
    assert apply_hct(0, 0, 1) == (0, 1)


def test_apply_hct_disagree_costs_two_and_takes_h2():
    # ai_label != h1 -> decision is h2, 2 queries
    assert apply_hct(1, 0, 0) == (0, 2)
    assert apply_hct(1, 0, 1) == (1, 2)
    assert apply_hct(0, 1, 0) == (0, 2)
    assert apply_hct(0, 1, 1) == (1, 2)


def test_apply_hct_vectorized_matches_scalar():
    ai = np.array([1, 1, 0, 0])
    h1 = np.array([1, 0, 0, 1])
    h2 = np.array([0, 1, 1, 0])
    dec, cost = apply_hct(ai, h1, h2)
    assert list(dec) == [1, 1, 0, 0]
    assert list(cost) == [1, 2, 1, 2]


# --------------------------------------------------------------------------- fixtures
def _label_df(rows):
    """rows: list of (GalaxyID, n_smooth, n_features, y_debiased)."""
    return pd.DataFrame(rows, columns=["GalaxyID", "n_smooth", "n_features", "y_debiased"])


def _scores_df(pairs):
    """pairs: list of (GalaxyID, score)."""
    return pd.DataFrame(pairs, columns=["GalaxyID", "score"])


def test_run_sweep_schema_and_row_count():
    ids = [10, 20, 30]
    labels = _label_df([(10, 5, 5, 0), (20, 8, 2, 0), (30, 1, 9, 1)])
    scores = _scores_df([(10, 0.3), (20, 0.6), (30, 0.95)])
    thetas = [0.2, 0.5, 0.8]
    seeds = [0, 1]
    out = run_sweep(scores, labels, ids, thetas, seeds)
    expected_cols = {
        "GalaxyID", "y_debiased", "ai_label", "h1", "h2",
        "decision", "human_queries", "theta", "seed",
    }
    assert set(out.columns) == expected_cols
    assert len(out) == len(ids) * len(thetas) * len(seeds)


def test_run_sweep_cost_in_unit_two_interval_and_identity():
    # cost mean per theta must be in [1,2] and equal 1 + P(ai != h1).
    rng = np.random.default_rng(0)
    n = 60
    ids = list(range(n))
    rows, scs = [], []
    for i in ids:
        ns, nf = int(rng.integers(2, 40)), int(rng.integers(0, 40))
        y = 1 if nf > ns else 0
        rows.append((i, ns, nf, y))
        scs.append((i, float(rng.random())))
    labels = _label_df(rows)
    scores = _scores_df(scs)
    thetas = [0.1, 0.5, 0.9]
    out = run_sweep(scores, labels, ids, thetas, [0, 1, 2])
    for th in thetas:
        sub = out[out.theta == th]
        cost = sub.human_queries.mean()
        assert 1.0 <= cost <= 2.0
        p_disagree = (sub.ai_label != sub.h1).mean()
        assert cost == pytest.approx(1.0 + p_disagree)


def test_run_sweep_all_agree_regime():
    # AI always matches h1 (homogeneous urns, score forces matching label) -> cost == 1,
    # decision == ai_label everywhere (AI-confirmed-by-human regime, P(disagree)->0).
    ids = [1, 2]
    # urn 1: all smooth (h1=h2=0); urn 2: all features (h1=h2=1)
    labels = _label_df([(1, 10, 0, 0), (2, 0, 10, 1)])
    # score below theta -> ai_label 0 for id1; above -> ai_label 1 for id2
    scores = _scores_df([(1, 0.0), (2, 1.0)])
    out = run_sweep(scores, labels, ids, [0.5], [0, 1, 2])
    assert (out.human_queries == 1).all()
    assert (out.decision == out.ai_label).all()


def test_run_sweep_all_disagree_regime():
    # AI always disagrees with h1 -> cost == 2, decision == h2 (single-human-like, P(disagree)->1).
    ids = [1, 2]
    # urn 1: all smooth -> h1=0; give it ai_label 1 (score above theta)
    # urn 2: all features -> h1=1; give it ai_label 0 (score below theta)
    labels = _label_df([(1, 10, 0, 0), (2, 0, 10, 1)])
    scores = _scores_df([(1, 1.0), (2, 0.0)])
    out = run_sweep(scores, labels, ids, [0.5], [0, 1, 2])
    assert (out.human_queries == 2).all()
    assert (out.decision == out.h2).all()


# --------------------------------------------------------------------------- guards
def test_run_sweep_labels_are_binary():
    ids = [1, 2, 3]
    labels = _label_df([(1, 5, 5, 0), (2, 3, 7, 1), (3, 9, 1, 0)])
    scores = _scores_df([(1, 0.4), (2, 0.6), (3, 0.2)])
    out = run_sweep(scores, labels, ids, [0.3, 0.7], [0, 1])
    for col in ["ai_label", "h1", "h2", "decision", "y_debiased"]:
        assert set(out[col].unique()).issubset({0, 1}), col


def test_run_sweep_ai_label_is_score_ge_theta():
    ids = [1, 2, 3]
    labels = _label_df([(1, 5, 5, 0), (2, 5, 5, 0), (3, 5, 5, 0)])
    scores = _scores_df([(1, 0.49), (2, 0.50), (3, 0.80)])
    out = run_sweep(scores, labels, ids, [0.5], [0])
    got = out.set_index("GalaxyID").ai_label.to_dict()
    assert got == {1: 0, 2: 1, 3: 1}  # >= theta is spiral(1)


# --------------------------------------------------------------------------- determinism
def test_run_sweep_deterministic_across_invocations():
    ids = list(range(40))
    rng = np.random.default_rng(7)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(0, 30)), 0) for i in ids]
    labels = _label_df(rows)
    scores = _scores_df([(i, float(rng.random())) for i in ids])
    a = run_sweep(scores, labels, ids, [0.3, 0.6], [0, 1])
    b = run_sweep(scores, labels, ids, [0.3, 0.6], [0, 1])
    pd.testing.assert_frame_equal(a, b)


def test_run_sweep_draws_constant_across_theta_within_seed():
    # decision #2: humans drawn once per (instance, seed); only ai_label moves with theta.
    ids = list(range(30))
    rng = np.random.default_rng(3)
    rows = [(i, int(rng.integers(1, 20)), int(rng.integers(1, 20)), 0) for i in ids]
    labels = _label_df(rows)
    scores = _scores_df([(i, float(rng.random())) for i in ids])
    out = run_sweep(scores, labels, ids, [0.2, 0.5, 0.8], [0])
    # within seed 0, (h1, h2) per GalaxyID must be identical across the three thetas
    piv = out.pivot_table(index="GalaxyID", columns="theta", values=["h1", "h2"], aggfunc="first")
    for col in ["h1", "h2"]:
        block = piv[col]
        assert (block.nunique(axis=1) == 1).all(), f"{col} varied across theta"
