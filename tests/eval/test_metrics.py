"""Shared eval metrics — written before haidc.eval.metrics (TESTING.md, invariant §4).

`accuracy` and `expected_human_cost` are the single source every arm uses. Tiny hand-computed
fixtures here; M6 extends this module with the Pareto frontier + bootstrap CIs.
"""
import numpy as np
import pytest

from haidc.eval.metrics import accuracy, expected_human_cost


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
