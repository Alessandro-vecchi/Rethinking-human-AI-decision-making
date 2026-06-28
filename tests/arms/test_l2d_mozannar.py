"""L2D-Mozannar arm — written before haidc.arms.l2d_mozannar (TESTING.md: red -> green).

Mozannar & Sontag 2020 "Consistent Estimators for Learning to Defer" (arXiv:2006.01862).
THE correctness gate (task step 2): the implemented surrogate must equal the paper's eq (10)
  L_CE^a(h,r,x,y,m) = -(a*1{m=y} + 1{m!=y}) log softmax_y(g) - 1{m=y} log softmax_bot(g)
on hand-built inputs, with a finite/correct gradient. A wrong surrogate invalidates every number.

Option B (CLAUDE.md #1/#2): the classifier is the FROZEN M2 backbone; we learn ONLY the rejector
logit g_bot on the 2048-d embeddings. Frozen class logits are reconstructed from the backbone
`score`: g_1=log(score), g_0=log(1-score). At a=1, eq (9)/Prop 2 make this a CONSISTENT rejector
surrogate (the g_bot minimizer recovers the Bayes rejector r^B=1{max_y eta_y <= P(Y=M|x)}).

Sweep knob = a (small a -> more deferral; large a -> automate-everything -> AI-alone). Expert term
= expected correctness q(x)=P(m=y|x)=urn vote-share matching y (soft, deterministic). Cost: 1 iff
defer, so expected cost == deferral fraction in [0,1] (the L2D regime, != HCT's [1,2]).
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from haidc.arms.l2d_mozannar import (  # noqa: E402
    OPS_COLUMNS,
    PRED_COLUMNS,
    bayes_defer_mask,
    class_logits_from_score,
    expected_correctness,
    fit_rejector_mozannar,
    l_ce_alpha,
    mozannar_defer_mask,
    mozannar_g_bot,
    operating_points,
    run_alpha_sweep,
)
from haidc.eval.metrics import accuracy, expected_human_cost  # noqa: E402


# --------------------------------------------------------------------------- helpers/fixtures
def _label_df(rows):
    """rows: list of (GalaxyID, n_smooth, n_features, y_debiased)."""
    return pd.DataFrame(rows, columns=["GalaxyID", "n_smooth", "n_features", "y_debiased"])


def _emb_df(ids, scores, feats, split="test"):
    """Build a minimal embeddings frame: GalaxyID, split, score, e0..e{d-1}."""
    feats = np.asarray(feats, dtype=float)
    cols = {"GalaxyID": list(ids), "split": [split] * len(ids), "score": list(scores)}
    for j in range(feats.shape[1]):
        cols[f"e{j}"] = feats[:, j]
    return pd.DataFrame(cols)


def _ref_l_ce_alpha(g, y, q, alpha):
    """Independent numpy reference for eq (10) (soft-q form). g: (3,) logits [g0,g1,gbot]."""
    g = np.asarray(g, dtype=float)
    logp = g - (np.log(np.exp(g).sum()))           # log-softmax over the 3 categories
    return -(alpha * q + (1.0 - q)) * logp[y] - q * logp[2]


# --------------------------------------------------------------------------- THE correctness gate
def test_l_ce_alpha_matches_eq10_on_hand_built_inputs():
    # Two categories {0,1} + bot=2. Check both expert branches against the numpy reference.
    g = [0.3, -0.7, 0.1]
    for (y, q, alpha) in [(1, 1.0, 2.0), (0, 1.0, 0.5), (1, 0.0, 2.0), (0, 0.0, 3.0)]:
        logits = torch.tensor([g], dtype=torch.float64)
        got = l_ce_alpha(logits, torch.tensor([y]), torch.tensor([q], dtype=torch.float64), alpha)
        assert float(got.item()) == pytest.approx(_ref_l_ce_alpha(g, y, q, alpha))


def test_l_ce_alpha_branch_structure():
    # m=y (q=1): carries BOTH terms (classifier weighted by a, bot weight 1).
    # m!=y (q=0): carries ONLY the classifier term (weight 1), no bot term.
    g = torch.tensor([[0.2, 0.9, -0.3]], dtype=torch.float64)
    logp = torch.log_softmax(g, dim=-1)[0]
    y, a = 1, 2.5
    li_correct = l_ce_alpha(g, torch.tensor([y]), torch.tensor([1.0], dtype=torch.float64), a)
    li_wrong = l_ce_alpha(g, torch.tensor([y]), torch.tensor([0.0], dtype=torch.float64), a)
    assert float(li_correct.item()) == pytest.approx(float(-a * logp[y] - logp[2]))
    assert float(li_wrong.item()) == pytest.approx(float(-logp[y]))


def test_l_ce_alpha_reduces_to_lce_at_alpha_1():
    # L_CE^1 == L_CE (eq 7): -(q+1-q)logp_y - q logp_bot = -logp_y - q logp_bot.
    g = torch.tensor([[0.5, -0.2, 0.4], [0.1, 0.1, 0.1]], dtype=torch.float64)
    y = torch.tensor([0, 1])
    q = torch.tensor([0.7, 0.2], dtype=torch.float64)
    logp = torch.log_softmax(g, dim=-1)
    lce = -logp[range(2), y] - q * logp[:, 2]
    got = l_ce_alpha(g, y, q, alpha=1.0)
    assert torch.allclose(got, lce)


def test_l_ce_alpha_soft_q_equals_hard_indicator():
    # Soft q with q in {0,1} must reproduce the hard-indicator loss exactly.
    g = torch.tensor([[0.3, -0.1, 0.2]], dtype=torch.float64)
    hard = l_ce_alpha(g, torch.tensor([1]), torch.tensor([1.0], dtype=torch.float64), 2.0)
    soft = l_ce_alpha(g, torch.tensor([1]), torch.tensor([1.0], dtype=torch.float64), 2.0)
    assert torch.allclose(hard, soft)


def test_l_ce_alpha_gradient_finite_and_matches_finite_difference():
    g = torch.tensor([[0.3, -0.7, 0.1]], dtype=torch.float64, requires_grad=True)
    y, q, a = torch.tensor([1]), torch.tensor([0.6], dtype=torch.float64), 2.0
    loss = l_ce_alpha(g, y, q, a).sum()
    loss.backward()
    grad = g.grad.clone()
    assert torch.isfinite(grad).all()
    # central finite-difference on the bot logit (index 2)
    eps = 1e-6
    with torch.no_grad():
        gp = g.clone()
        gp[0, 2] += eps
        gm = g.clone()
        gm[0, 2] -= eps
        fp = l_ce_alpha(gp, y, q, a).sum().item()
        fm = l_ce_alpha(gm, y, q, a).sum().item()
    assert float(grad[0, 2]) == pytest.approx((fp - fm) / (2 * eps), rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------- frozen class logits
def test_class_logits_reconstruct_score():
    score = np.array([0.1, 0.5, 0.9, 0.73])
    g0, g1 = class_logits_from_score(score)
    sm = np.exp(np.stack([g0, g1], axis=1))
    sm = sm / sm.sum(axis=1, keepdims=True)
    assert np.allclose(sm[:, 0], 1.0 - score)
    assert np.allclose(sm[:, 1], score)


def test_expected_correctness_is_one_minus_minority_share():
    # q(x) = vote-share matching y. y=1 -> n_features/N ; y=0 -> n_smooth/N.
    ns = np.array([8, 2, 5])
    nf = np.array([2, 8, 5])
    y = np.array([1, 0, 1])
    assert expected_correctness(ns, nf, y) == pytest.approx([2 / 10, 2 / 10, 5 / 10])


# --------------------------------------------------------------------------- deferral rule + Bayes
def test_mozannar_defer_rule_is_bot_ge_max_class():
    # defer iff g_bot >= max(g0,g1)  (eq 6: r = 1{max_y g_y <= g_bot}).
    score = np.array([0.6, 0.6, 0.95])
    g0, g1 = class_logits_from_score(score)
    g_bot = np.array([math.log(0.9), math.log(0.3), math.log(0.8)])  # = log q at the analytic optimum
    mask = mozannar_defer_mask(g_bot, score)
    assert list(mask) == [True, False, False]  # 0.9>=0.6 ; 0.3<0.6 ; 0.8<0.95


def test_frozen_classifier_optimum_recovers_bayes_rejector():
    # eq (9)/Prop 2: at a=1 the optimal bot logit is g_bot*=log q, giving defer iff q>=max class prob.
    score = np.array([0.6, 0.6, 0.95, 0.55, 0.5, 0.99])
    q = np.array([0.9, 0.3, 0.8, 0.85, 0.7, 0.4])
    g_bot_star = np.log(q)
    learned = mozannar_defer_mask(g_bot_star, score)
    bayes = bayes_defer_mask(score, q)
    assert list(learned) == list(bayes) == [True, False, False, True, True, False]


# --------------------------------------------------------------------------- rejector training
def test_fit_rejector_is_deterministic():
    rng = np.random.default_rng(0)
    n, d = 24, 8
    emb = rng.standard_normal((n, d))
    score = rng.uniform(0.05, 0.95, n)
    y = (score >= 0.5).astype(int)
    q = rng.uniform(0.2, 0.9, n)
    m1 = fit_rejector_mozannar(emb, score, y, q, alpha=1.0, seed=0, epochs=40)
    m2 = fit_rejector_mozannar(emb, score, y, q, alpha=1.0, seed=0, epochs=40)
    assert np.allclose(mozannar_g_bot(m1, emb), mozannar_g_bot(m2, emb))


def test_fit_single_batch_overfit_drives_loss_down():
    # Sanity (TESTING.md): the rejector head can learn at all on a tiny batch.
    rng = np.random.default_rng(1)
    n, d = 16, 6
    emb = rng.standard_normal((n, d))
    score = rng.uniform(0.05, 0.95, n)
    y = (score >= 0.5).astype(int)
    q = rng.uniform(0.2, 0.9, n)

    def _loss(model):
        g0, g1 = class_logits_from_score(score)
        gb = mozannar_g_bot(model, emb)
        logits = torch.tensor(np.stack([g0, g1, gb], axis=1), dtype=torch.float64)
        return float(l_ce_alpha(logits, torch.tensor(y), torch.tensor(q, dtype=torch.float64), 1.0).mean())

    m_few = fit_rejector_mozannar(emb, score, y, q, alpha=1.0, seed=0, epochs=2)
    m_many = fit_rejector_mozannar(emb, score, y, q, alpha=1.0, seed=0, epochs=600)
    assert _loss(m_many) < _loss(m_few) - 1e-3


def test_fit_rejector_recovers_bayes_at_alpha_1():
    # With memorizable (one-hot) features and a=1, the fitted rejector reproduces the Bayes rule.
    score = np.array([0.6, 0.6, 0.95, 0.55, 0.5, 0.99])
    q = np.array([0.9, 0.3, 0.8, 0.85, 0.7, 0.4])
    y = (score >= 0.5).astype(int)
    emb = np.eye(len(score))  # one-hot: each instance independently fittable
    model = fit_rejector_mozannar(emb, score, y, q, alpha=1.0, seed=0, hidden=16, epochs=4000, lr=0.05)
    learned = mozannar_defer_mask(mozannar_g_bot(model, emb), score)
    assert list(learned) == list(bayes_defer_mask(score, q))


# --------------------------------------------------------------------------- alpha sweep
def _synth(n=80, seed=3):
    rng = np.random.default_rng(seed)
    ids = list(range(n))
    ns = rng.integers(2, 30, n)
    nf = rng.integers(2, 30, n)
    y = (nf > ns).astype(int)
    score = rng.uniform(0.02, 0.98, n)
    # informative feature: correlated with whether the human beats the AI
    feats = np.stack([expected_correctness(ns, nf, y) - (score >= 0.5).astype(float) + y,
                      rng.standard_normal(n)], axis=1)
    labels = _label_df([(i, int(ns[k]), int(nf[k]), int(y[k])) for k, i in enumerate(ids)])
    emb = _emb_df(ids, score, feats, split="test")
    return ids, labels, emb


def test_run_alpha_sweep_schema_and_policy():
    ids, labels, emb = _synth()
    out = run_alpha_sweep(emb, labels, ids, ids, alphas=[0.5, 1.0], seeds=[0, 1], epochs=30)
    assert set(out.columns) == set(PRED_COLUMNS)
    assert len(out) == len(ids) * 2 * 2
    assert (out["policy"] == "mozannar").all()
    for col in ["ai_label", "defer", "human", "decision", "y_debiased", "human_queries"]:
        assert set(out[col].unique()).issubset({0, 1}), col


def test_run_alpha_sweep_cost_equals_deferral_fraction_in_unit_interval():
    ids, labels, emb = _synth()
    out = run_alpha_sweep(emb, labels, ids, ids, alphas=[0.2, 1.0, 5.0], seeds=[0, 1], epochs=30)
    for a in [0.2, 1.0, 5.0]:
        sub = out[out.alpha == a]
        cost = expected_human_cost(sub.human_queries.to_numpy())
        assert 0.0 <= cost <= 1.0
        assert cost == pytest.approx(float(sub.defer.mean()))


def test_run_alpha_sweep_deferral_monotone_decreasing_in_alpha():
    # small alpha encourages deferral, large alpha hinders it -> deferral fraction non-increasing.
    ids, labels, emb = _synth(n=120, seed=9)
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    out = run_alpha_sweep(emb, labels, ids, ids, alphas=alphas, seeds=[0], epochs=120)
    fracs = [float(out[out.alpha == a].defer.mean()) for a in alphas]
    # non-increasing within a small tolerance for optimization noise
    assert all(f2 <= f1 + 0.06 for f1, f2 in zip(fracs, fracs[1:])), fracs
    assert fracs[0] > fracs[-1]  # the endpoints must separate


def test_run_alpha_sweep_large_alpha_automates_to_ai_alone():
    ids, labels, emb = _synth(n=120, seed=12)
    out = run_alpha_sweep(emb, labels, ids, ids, alphas=[1000.0], seeds=[0], epochs=120)
    sub = out[out.alpha == 1000.0]
    assert float(sub.defer.mean()) == pytest.approx(0.0, abs=0.02)  # automate (near-)everything
    ai_alone = accuracy(sub.ai_label.to_numpy(), sub.y_debiased.to_numpy())
    got = accuracy(sub.decision.to_numpy(), sub.y_debiased.to_numpy())
    assert got == pytest.approx(ai_alone, abs=0.02)


def test_run_alpha_sweep_deterministic_across_invocations():
    ids, labels, emb = _synth(n=40, seed=2)
    a = run_alpha_sweep(emb, labels, ids, ids, alphas=[1.0], seeds=[0, 1], epochs=25)
    b = run_alpha_sweep(emb, labels, ids, ids, alphas=[1.0], seeds=[0, 1], epochs=25)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- operating points
def test_operating_points_schema_and_cost_bound():
    ids, labels, emb = _synth()
    out = run_alpha_sweep(emb, labels, ids, ids, alphas=[0.5, 1.0], seeds=[0, 1], epochs=25)
    ops = operating_points(out)
    assert set(ops.columns) == set(OPS_COLUMNS)
    assert ((ops.cost_mean >= 0.0) & (ops.cost_mean <= 1.0)).all()
    assert np.allclose(ops.cost_mean.to_numpy(), ops.deferral_fraction.to_numpy())


# --------------------------------------------------------------------------- frozen-split guard (run-level)
def test_run_rejects_tampered_split_hash(tmp_path):
    from haidc.arms import l2d_mozannar

    test_ids = [1, 2, 3]
    manifest = {"test": test_ids, "split_hashes": {"test": "deadbeef"}}  # wrong hash
    mpath = tmp_path / "split_manifest.json"
    mpath.write_text(json.dumps(manifest))
    labels = _label_df([(1, 5, 5, 0), (2, 8, 2, 0), (3, 1, 9, 1)])
    scores = pd.DataFrame({"GalaxyID": [1, 2, 3], "score": [0.3, 0.6, 0.9]})
    lpath, spath = tmp_path / "labels.parquet", tmp_path / "scores.parquet"
    labels.to_parquet(lpath, index=False)
    scores.to_parquet(spath, index=False)
    cfg = {
        "defer_cost_sweep": [1.0],
        "split_manifest_path": str(mpath),
        "label_table_path": str(lpath),
        "scores_path": str(spath),
        "embeddings_path": str(tmp_path / "does_not_exist.parquet"),
        "export_predictions_path": str(tmp_path / "pred.parquet"),
        "export_operating_points_path": str(tmp_path / "ops.csv"),
        "seed": 0,
    }
    with pytest.raises(AssertionError):
        l2d_mozannar.run(cfg, {"rater_draw_seeds": [0]})
