"""L2D-Okati arm — written before haidc.arms.l2d_okati (TESTING.md: red -> green).

Option B (CLAUDE.md #2): the classifier is the FROZEN M2 backbone; we learn only Okati's triage.
The deferral mechanism is Okati Theorem 3 (GROUND_TRUTH §3): for a fixed model m, the optimal
triage is a threshold on the per-instance gap E[l(m,y)] - E[l(h,y)], with budget b capping the
deferral fraction. `find_machine_samples` is transcribed from Okati train.ipynb Cell-8.

Two policies, kept distinct (the orchestrator's hard constraint — M6 must never plot the oracle as
Okati's deployed method):
  - policy="oracle": find_machine_samples directly on the frozen TEST losses. Optimal UPPER BOUND;
    decides deferral using test labels => not deployable.
  - policy="learned": a rejector trained on backbone embeddings to approximate the gap ranking,
    applied label-free at test (the deployable head-to-head curve).

Cost semantics (different regime from HCT's [1,2]): human_queries = 1 iff defer else 0, so the
expected cost == the deferral fraction in [0,1].

Human term (invariant §3, same P(h|x) as HCT/single-human): the deferral DECISION uses the
deterministic expected human 0/1 loss = minority share of the urn; the realized decision when
deferred draws ONE human per (instance, seed) from the same urn.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from haidc.arms.l2d_okati import (
    ai_label_from_score,
    expected_human_loss,
    find_machine_samples,
    fit_rejector,
    machine_loss_01,
    operating_points,
    oracle_defer_mask,
    rejector_defer_proba,
    run_oracle_sweep,
    topk_defer_mask,
)
from haidc.eval.metrics import accuracy, expected_human_cost


# --------------------------------------------------------------------------- helpers/fixtures
def _label_df(rows):
    """rows: list of (GalaxyID, n_smooth, n_features, y_debiased)."""
    return pd.DataFrame(rows, columns=["GalaxyID", "n_smooth", "n_features", "y_debiased"])


def _scores_df(pairs):
    """pairs: list of (GalaxyID, score)."""
    return pd.DataFrame(pairs, columns=["GalaxyID", "score"])


# --------------------------------------------------------------------------- gap building blocks
def test_ai_label_is_score_ge_half():
    # AI prediction is FIXED at the backbone anchor threshold 0.5 (b, not theta, is Okati's knob).
    got = ai_label_from_score(np.array([0.49, 0.50, 0.80, 0.0, 1.0]))
    assert list(got) == [0, 1, 1, 0, 1]


def test_machine_loss_is_01_misclassification():
    score = np.array([0.9, 0.1, 0.9, 0.1])  # ai_label = 1,0,1,0
    y = np.array([1, 0, 0, 1])              # correct, correct, wrong, wrong
    assert list(machine_loss_01(score, y)) == [0, 0, 1, 1]


def test_expected_human_loss_is_minority_share_wrt_y():
    # y=1 (spiral) -> P(draw != y) = n_smooth / N ; y=0 -> n_features / N
    ns = np.array([8, 2, 5])
    nf = np.array([2, 8, 5])
    y = np.array([1, 0, 1])
    got = expected_human_loss(ns, nf, y)
    assert got == pytest.approx([8 / 10, 8 / 10, 5 / 10])


# --------------------------------------------------------------------------- find_machine_samples (Thm 3)
def test_find_machine_samples_defers_largest_positive_gap_first():
    # gaps: idx0=-0.5, idx1=+0.8, idx2=+0.2, idx3=-0.1  -> positive: {1,2}, 1 has the larger gap
    mloss = np.array([0.0, 1.0, 1.0, 0.0])
    hloss = np.array([0.5, 0.2, 0.8, 0.1])
    # budget large enough for 2 deferrals: defer {1,2}; machine = {0,3}
    machine = find_machine_samples(mloss, hloss, budget_b=0.5)  # int(0.5*4)=2
    assert set(machine.tolist()) == {0, 3}


def test_find_machine_samples_never_defers_nonpositive_gap():
    # only idx1 has a positive gap; a generous budget must still NOT defer the gap<=0 ones.
    mloss = np.array([0.0, 1.0, 0.0, 0.0])
    hloss = np.array([0.5, 0.2, 0.1, 0.9])
    machine = find_machine_samples(mloss, hloss, budget_b=1.0)  # would allow 4 deferrals
    deferred = sorted(set(range(4)) - set(machine.tolist()))
    assert deferred == [1]  # exactly the single positive-gap instance


def test_find_machine_samples_b0_defers_none():
    # the upstream `argsorted[:-0]` slice quirk would defer ALL at b=0; we fix it to defer NONE.
    mloss = np.array([1.0, 1.0, 1.0])
    hloss = np.array([0.0, 0.0, 0.0])
    machine = find_machine_samples(mloss, hloss, budget_b=0.0)
    assert sorted(machine.tolist()) == [0, 1, 2]  # all machine, zero deferrals


def test_oracle_defer_mask_complements_machine_samples():
    mloss = np.array([0.0, 1.0, 1.0, 0.0])
    hloss = np.array([0.5, 0.2, 0.8, 0.1])
    mask = oracle_defer_mask(mloss, hloss, budget_b=0.5)
    assert mask.dtype == bool
    assert list(mask) == [False, True, True, False]


# --------------------------------------------------------------------------- oracle sweep
def test_run_oracle_sweep_schema_and_policy_label():
    ids = [10, 20, 30]
    labels = _label_df([(10, 5, 5, 0), (20, 8, 2, 0), (30, 1, 9, 1)])
    scores = _scores_df([(10, 0.3), (20, 0.6), (30, 0.95)])
    out = run_oracle_sweep(scores, labels, ids, budgets=[0.0, 0.3], seeds=[0, 1])
    expected_cols = {
        "GalaxyID", "y_debiased", "ai_label", "defer", "human",
        "decision", "human_queries", "b", "seed", "policy",
    }
    assert set(out.columns) == expected_cols
    assert len(out) == len(ids) * 2 * 2
    assert (out["policy"] == "oracle").all()
    for col in ["ai_label", "defer", "human", "decision", "y_debiased", "human_queries"]:
        assert set(out[col].unique()).issubset({0, 1}), col


def test_run_oracle_sweep_b0_equals_ai_alone():
    # b=0 -> no deferral -> decision == ai_label everywhere -> accuracy == AI-alone accuracy.
    ids = list(range(20))
    rng = np.random.default_rng(0)
    rows, scs = [], []
    for i in ids:
        ns, nf = int(rng.integers(2, 30)), int(rng.integers(2, 30))
        y = 1 if nf > ns else 0
        rows.append((i, ns, nf, y))
        scs.append((i, float(rng.random())))
    labels, scores = _label_df(rows), _scores_df(scs)
    out = run_oracle_sweep(scores, labels, ids, budgets=[0.0], seeds=[0])
    assert (out["defer"] == 0).all()
    assert (out["human_queries"] == 0).all()
    assert (out["decision"] == out["ai_label"]).all()
    ai_alone = accuracy(out["ai_label"].to_numpy(), out["y_debiased"].to_numpy())
    got = accuracy(out["decision"].to_numpy(), out["y_debiased"].to_numpy())
    assert got == pytest.approx(ai_alone)


def test_run_oracle_sweep_cost_equals_deferral_fraction_in_unit_interval():
    ids = list(range(40))
    rng = np.random.default_rng(1)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(0, 2))) for i in ids]
    scs = [(i, float(rng.random())) for i in ids]
    out = run_oracle_sweep(_scores_df(scs), _label_df(rows), ids, budgets=[0.0, 0.2, 0.5], seeds=[0, 1])
    for b in [0.0, 0.2, 0.5]:
        sub = out[out.b == b]
        cost = expected_human_cost(sub.human_queries.to_numpy())
        defer_frac = float(sub.defer.mean())
        assert 0.0 <= cost <= 1.0
        assert cost == pytest.approx(defer_frac)


def test_run_oracle_sweep_deferral_monotone_in_b():
    ids = list(range(60))
    rng = np.random.default_rng(2)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(0, 2))) for i in ids]
    scs = [(i, float(rng.random())) for i in ids]
    budgets = [0.0, 0.1, 0.2, 0.4, 0.8]
    out = run_oracle_sweep(_scores_df(scs), _label_df(rows), ids, budgets=budgets, seeds=[0])
    fracs = [float(out[out.b == b].defer.mean()) for b in budgets]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(fracs, fracs[1:])), fracs


def test_run_oracle_sweep_never_defers_when_ai_correct():
    # gap = 1{ai!=y} - minority_share > 0 only when AI is WRONG; oracle must never defer a hit.
    ids = list(range(40))
    rng = np.random.default_rng(5)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(0, 2))) for i in ids]
    scs = [(i, float(rng.random())) for i in ids]
    out = run_oracle_sweep(_scores_df(scs), _label_df(rows), ids, budgets=[0.8], seeds=[0])
    deferred = out[out.defer == 1]
    assert (deferred.ai_label != deferred.y_debiased).all()  # every deferred instance is an AI error


def test_run_oracle_sweep_deferral_set_is_seed_independent():
    # the deferral DECISION uses the deterministic expected human loss -> same set across seeds.
    ids = list(range(30))
    rng = np.random.default_rng(8)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(0, 2))) for i in ids]
    scs = [(i, float(rng.random())) for i in ids]
    out = run_oracle_sweep(_scores_df(scs), _label_df(rows), ids, budgets=[0.3], seeds=[0, 1, 2])
    piv = out.pivot_table(index="GalaxyID", columns="seed", values="defer", aggfunc="first")
    assert (piv.nunique(axis=1) == 1).all()


def test_run_oracle_sweep_deterministic_across_invocations():
    ids = list(range(30))
    rng = np.random.default_rng(7)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(0, 2))) for i in ids]
    scs = [(i, float(rng.random())) for i in ids]
    a = run_oracle_sweep(_scores_df(scs), _label_df(rows), ids, budgets=[0.2, 0.5], seeds=[0, 1])
    b = run_oracle_sweep(_scores_df(scs), _label_df(rows), ids, budgets=[0.2, 0.5], seeds=[0, 1])
    pd.testing.assert_frame_equal(a, b)


def test_run_oracle_sweep_robust_to_row_reordering():
    # inputs are indexed by GalaxyID in the frozen test order; shuffling input rows must not change output.
    ids = list(range(25))
    rng = np.random.default_rng(11)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(0, 2))) for i in ids]
    scs = [(i, float(rng.random())) for i in ids]
    labels, scores = _label_df(rows), _scores_df(scs)
    base = run_oracle_sweep(scores, labels, ids, budgets=[0.3], seeds=[0])
    shuf = run_oracle_sweep(
        scores.sample(frac=1.0, random_state=3).reset_index(drop=True),
        labels.sample(frac=1.0, random_state=4).reset_index(drop=True),
        ids, budgets=[0.3], seeds=[0],
    )
    pd.testing.assert_frame_equal(base, shuf)


# --------------------------------------------------------------------------- operating points
def test_operating_points_schema_and_cost_bound():
    ids = list(range(30))
    rng = np.random.default_rng(4)
    rows = [(i, int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(0, 2))) for i in ids]
    scs = [(i, float(rng.random())) for i in ids]
    out = run_oracle_sweep(_scores_df(scs), _label_df(rows), ids, budgets=[0.0, 0.2], seeds=[0, 1])
    ops = operating_points(out)
    assert set(ops.columns) == {
        "b", "policy", "accuracy_mean", "accuracy_lo", "accuracy_hi",
        "cost_mean", "deferral_fraction", "n_seeds",
    }
    assert ((ops.cost_mean >= 0.0) & (ops.cost_mean <= 1.0)).all()
    assert np.allclose(ops.cost_mean.to_numpy(), ops.deferral_fraction.to_numpy())


# --------------------------------------------------------------------------- frozen-split guard
def test_run_rejects_test_ids_not_matching_committed_split(tmp_path):
    from haidc.arms import l2d_okati

    # a hand-built manifest whose test split-hash matches its test list (internally consistent)
    test_ids = [1, 2, 3]
    h = hashlib.sha256(json.dumps(list(map(int, test_ids)), sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    manifest = {"test": test_ids, "split_hashes": {"test": h}}
    mpath = tmp_path / "split_manifest.json"
    mpath.write_text(json.dumps(manifest))

    # scores/labels MISSING id 3 -> must raise (shared-input guard)
    labels = _label_df([(1, 5, 5, 0), (2, 8, 2, 0)])
    scores = _scores_df([(1, 0.3), (2, 0.6)])
    lpath, spath = tmp_path / "labels.parquet", tmp_path / "scores.parquet"
    labels.to_parquet(lpath, index=False)
    scores.to_parquet(spath, index=False)

    cfg = {
        "budget_b_sweep": [0.0, 0.3],
        "split_manifest_path": str(mpath),
        "label_table_path": str(lpath),
        "scores_path": str(spath),
        "export_predictions_path": str(tmp_path / "pred.parquet"),
        "export_operating_points_path": str(tmp_path / "ops.csv"),
        "seed": 0,
    }
    with pytest.raises(AssertionError):
        l2d_okati.run(cfg, {"rater_draw_seeds": [0]})


def test_run_rejects_tampered_split_hash(tmp_path):
    from haidc.arms import l2d_okati

    test_ids = [1, 2, 3]
    manifest = {"test": test_ids, "split_hashes": {"test": "deadbeef"}}  # wrong hash
    mpath = tmp_path / "split_manifest.json"
    mpath.write_text(json.dumps(manifest))
    labels = _label_df([(1, 5, 5, 0), (2, 8, 2, 0), (3, 1, 9, 1)])
    scores = _scores_df([(1, 0.3), (2, 0.6), (3, 0.9)])
    lpath, spath = tmp_path / "labels.parquet", tmp_path / "scores.parquet"
    labels.to_parquet(lpath, index=False)
    scores.to_parquet(spath, index=False)
    cfg = {
        "budget_b_sweep": [0.0],
        "split_manifest_path": str(mpath),
        "label_table_path": str(lpath),
        "scores_path": str(spath),
        "export_predictions_path": str(tmp_path / "pred.parquet"),
        "export_operating_points_path": str(tmp_path / "ops.csv"),
        "seed": 0,
    }
    with pytest.raises(AssertionError):
        l2d_okati.run(cfg, {"rater_draw_seeds": [0]})


# --------------------------------------------------------------------------- learned rejector (Stage B)
def test_topk_defer_mask_defers_floor_b_n_highest_scores():
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    mask = topk_defer_mask(s, budget_b=0.4)  # floor(0.4*5)=2 -> defer the 2 highest: idx1,idx3
    assert list(mask) == [False, True, False, True, False]
    assert topk_defer_mask(s, budget_b=0.0).sum() == 0


def test_fit_rejector_is_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((40, 16))
    target = (emb[:, 0] > 0).astype(int)  # linearly separable defer target
    m1 = fit_rejector(emb, target, seed=0, epochs=50)
    m2 = fit_rejector(emb, target, seed=0, epochs=50)
    p1 = rejector_defer_proba(m1, emb)
    p2 = rejector_defer_proba(m2, emb)
    assert np.allclose(p1, p2)


def test_learned_rejector_recovers_separable_defer_signal():
    # with perfectly informative features the rejector ranks the true-defer instances on top,
    # so deferring the top-k matches the oracle defer set (learned attains the oracle envelope).
    rng = np.random.default_rng(1)
    n = 80
    emb = rng.standard_normal((n, 16))
    true_defer = (emb[:, 0] > 0).astype(int)  # k true-defer instances
    model = fit_rejector(emb, true_defer, seed=0, epochs=300)
    proba = rejector_defer_proba(model, emb)
    k = int(true_defer.sum())
    topk = set(np.argsort(proba)[-k:].tolist())
    assert topk == set(np.where(true_defer == 1)[0].tolist())
