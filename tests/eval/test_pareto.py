"""Pareto frontier + adapter tests — written before haidc.eval.pareto (TESTING.md).

Covers the invariants the M6 ticket calls out: the adapter rejects malformed arm tables;
per-arm human-query cost accounting (AI-alone=0, single-human=1, HCT∈[1,2], L2D∈[0,1]); Pareto
dominance on a toy with known non-dominated points; the Okati ORACLE policy is excluded from the
deployable frontier; and the central-claim verdict (CI lower bound vs AI-alone) on a known toy.
"""
import pandas as pd
import pytest

from haidc.eval.pareto import (
    beats_ai_alone,
    deployable_frontier,
    score_points,
    to_common_frame,
)

COMMON_COLS = {
    "arm", "policy", "operating_param", "seed",
    "GalaxyID", "decision", "y_debiased", "human_queries",
}


def _hct_like(**over):
    base = {
        "GalaxyID": [1, 2], "y_debiased": [0, 1], "decision": [0, 1],
        "human_queries": [1, 2], "seed": [0, 0], "theta": [0.5, 0.5],
    }
    base.update(over)
    return pd.DataFrame(base)


# --------------------------------------------------------------------------- adapter
def test_adapter_rejects_table_missing_agreed_column():
    df = _hct_like().drop(columns=["decision"])
    with pytest.raises(ValueError):
        to_common_frame(df, arm="hct", sweep_col="theta")


def test_adapter_rejects_table_missing_sweep_column():
    df = _hct_like().drop(columns=["theta"])
    with pytest.raises(ValueError):
        to_common_frame(df, arm="hct", sweep_col="theta")


def test_adapter_normalizes_to_common_frame():
    out = to_common_frame(_hct_like(), arm="hct", sweep_col="theta")
    assert COMMON_COLS.issubset(out.columns)
    assert (out["arm"] == "hct").all()
    assert (out["operating_param"] == 0.5).all()


def test_adapter_reads_policy_column():
    df = pd.DataFrame({
        "GalaxyID": [1, 1], "y_debiased": [0, 0], "decision": [1, 0],
        "human_queries": [0, 0], "seed": [0, 0], "b": [0.1, 0.1],
        "policy": ["oracle", "learned"],
    })
    out = to_common_frame(df, arm="l2d_okati", sweep_col="b", policy_col="policy")
    assert set(out["policy"]) == {"oracle", "learned"}


# --------------------------------------------------------------------------- cost accounting
def test_cost_accounting_hct_in_unit_two_band():
    frame = to_common_frame(_hct_like(), arm="hct", sweep_col="theta")
    pts = score_points(frame, n_resamples=50, ci=0.95, seed=0)
    row = pts.iloc[0]
    assert row["cost_mean"] == pytest.approx(1.5)
    assert 1.0 <= row["cost_mean"] <= 2.0
    assert row["accuracy_mean"] == pytest.approx(1.0)  # decisions == y on the toy


def test_cost_accounting_l2d_in_zero_one_band():
    df = pd.DataFrame({
        "GalaxyID": [1, 2], "y_debiased": [0, 1], "decision": [0, 1],
        "human_queries": [0, 1], "seed": [0, 0], "b": [0.3, 0.3], "policy": ["learned", "learned"],
    })
    frame = to_common_frame(df, arm="l2d_okati", sweep_col="b", policy_col="policy")
    pts = score_points(frame, n_resamples=50, ci=0.95, seed=0)
    assert 0.0 <= pts.iloc[0]["cost_mean"] <= 1.0
    assert pts.iloc[0]["cost_mean"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- dominance
def test_deployable_frontier_known_nondominated():
    pts = pd.DataFrame({
        "arm": ["a", "a", "a"], "policy": ["p", "p", "p"], "operating_param": [1, 2, 3],
        "accuracy_mean": [0.80, 0.85, 0.82], "cost_mean": [0.0, 0.5, 0.5],
    })
    fr = deployable_frontier(pts)
    kept = set(zip(fr["cost_mean"].round(3), fr["accuracy_mean"].round(3)))
    assert (0.0, 0.80) in kept          # cheapest; nothing cheaper, so non-dominated
    assert (0.5, 0.85) in kept          # best accuracy
    assert (0.5, 0.82) not in kept      # dominated by (0.5, 0.85): same cost, lower accuracy


def test_deployable_frontier_excludes_okati_oracle():
    pts = pd.DataFrame({
        "arm": ["l2d_okati", "l2d_okati"], "policy": ["oracle", "learned"],
        "operating_param": [0.2, 0.1], "accuracy_mean": [0.95, 0.83], "cost_mean": [0.17, 0.10],
    })
    fr = deployable_frontier(pts)
    assert (fr["policy"] == "oracle").sum() == 0
    assert (fr["policy"] == "learned").any()


# --------------------------------------------------------------------------- central claim
def test_beats_ai_alone_uses_ci_lower_bound():
    pts = pd.DataFrame({
        "arm": ["x", "y"], "policy": ["p", "p"], "operating_param": [1, 1],
        "accuracy_mean": [0.86, 0.83], "ci_lo": [0.84, 0.80], "ci_hi": [0.88, 0.86],
        "cost_mean": [1.0, 0.5],
    })
    verdict = beats_ai_alone(pts, baseline=0.829)
    by_arm = {r["arm"]: bool(r["beats"]) for _, r in verdict.iterrows()}
    assert by_arm["x"] is True        # ci_lo 0.84 > 0.829
    assert by_arm["y"] is False       # ci_lo 0.80 <= 0.829 (overlaps -> no sig. difference)


def test_beats_ai_alone_excludes_oracle():
    pts = pd.DataFrame({
        "arm": ["l2d_okati"], "policy": ["oracle"], "operating_param": [0.2],
        "accuracy_mean": [0.95], "ci_lo": [0.94], "ci_hi": [0.96], "cost_mean": [0.17],
    })
    verdict = beats_ai_alone(pts, baseline=0.829)
    assert len(verdict) == 0          # oracle is not a deployable claimant
