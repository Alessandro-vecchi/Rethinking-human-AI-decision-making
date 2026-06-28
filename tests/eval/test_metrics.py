"""Shared eval metrics — written before haidc.eval.metrics (TESTING.md, invariant §4).

`accuracy` and `expected_human_cost` are the single source every arm uses. Tiny hand-computed
fixtures here; M6 extends this module with TPR/FPR + the shared bootstrap CI.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haidc.eval.metrics import (
    accuracy,
    bootstrap_accuracy_ci,
    expected_human_cost,
    fpr,
    tpr,
)

REPO = Path(__file__).resolve().parents[2]


def test_accuracy_all_correct():
    assert accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_accuracy_all_wrong():
    assert accuracy([0, 0, 0], [1, 1, 1]) == 0.0


def test_accuracy_half():
    # 2 of 4 match -> 0.5
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 1]) == 0.5


def test_accuracy_accepts_numpy():
    d = np.array([1, 0, 1])
    y = np.array([1, 0, 0])
    assert accuracy(d, y) == pytest.approx(2 / 3)


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([1, 0], [1, 0, 1])


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_expected_human_cost_all_ones():
    assert expected_human_cost([1, 1, 1, 1]) == 1.0


def test_expected_human_cost_all_twos():
    assert expected_human_cost([2, 2, 2]) == 2.0


def test_expected_human_cost_mixed():
    # three 1s and one 2 -> mean = 5/4 = 1.25
    assert expected_human_cost([1, 1, 1, 2]) == pytest.approx(1.25)


def test_expected_human_cost_empty_raises():
    with pytest.raises(ValueError):
        expected_human_cost([])


# --------------------------------------------------------------------------- TPR / FPR
# Label convention: 1 = spiral = positive. Hand fixture:
#   y    = [1, 1, 1, 0, 0]   (3 positives, 2 negatives)
#   pred = [1, 1, 0, 1, 0]   -> TP=2, FN=1 -> TPR=2/3 ; FP=1, TN=1 -> FPR=1/2
def test_tpr_hand_computed():
    assert tpr([1, 1, 0, 1, 0], [1, 1, 1, 0, 0]) == pytest.approx(2 / 3)


def test_fpr_hand_computed():
    assert fpr([1, 1, 0, 1, 0], [1, 1, 1, 0, 0]) == pytest.approx(1 / 2)


def test_tpr_all_positives_detected():
    assert tpr([1, 1, 1], [1, 1, 1]) == 1.0


def test_fpr_no_false_positives():
    assert fpr([0, 0, 1], [0, 0, 1]) == 0.0


def test_tpr_undefined_without_positives_raises():
    with pytest.raises(ValueError):
        tpr([0, 0], [0, 0])


def test_fpr_undefined_without_negatives_raises():
    with pytest.raises(ValueError):
        fpr([1, 1], [1, 1])


def test_tpr_length_mismatch_raises():
    with pytest.raises(ValueError):
        tpr([1, 0], [1, 0, 1])


# --------------------------------------------------------------------------- bootstrap CI
def test_bootstrap_ci_deterministic_given_seed():
    correct = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    a = bootstrap_accuracy_ci(correct, n_resamples=500, ci=0.95, seed=0)
    b = bootstrap_accuracy_ci(correct, n_resamples=500, ci=0.95, seed=0)
    assert a == b


def test_bootstrap_ci_brackets_point_estimate():
    correct = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    point = float(correct.mean())
    lo, hi = bootstrap_accuracy_ci(correct, n_resamples=1000, ci=0.95, seed=0)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_accuracy_ci(np.array([]), n_resamples=10, ci=0.95, seed=0)


def test_bootstrap_ci_matches_pinned_ai_alone(  # regression guard for the M2 absorption
):
    """The shared bootstrap must reproduce the frozen AI-alone CI in backbone_run.json byte-for-byte
    (REPRODUCIBILITY.md: absorbing backbone's inline bootstrap must not move the number)."""
    scores = pd.read_parquet(REPO / "results" / "backbone_scores.parquet")
    labels = pd.read_parquet(REPO / "data" / "label_table.parquet")[["GalaxyID", "y_debiased"]]
    df = scores.merge(labels, on="GalaxyID", how="left")
    correct = ((df["score"].to_numpy() >= 0.5).astype(int) == df["y_debiased"].to_numpy()).astype(int)
    lo, hi = bootstrap_accuracy_ci(correct, n_resamples=1000, ci=0.95, seed=0)
    assert (lo, hi) == pytest.approx((0.8011167146974063, 0.8573487031700289))
