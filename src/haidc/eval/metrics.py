"""Shared evaluation metrics — the SINGLE source every arm uses (CLAUDE.md invariant #4).

Defined once here so HCT, both L2D lines, and the baselines cannot silently disagree on what
"accuracy" or "expected human cost" means. M6 extends this module with the Pareto frontier and
bootstrap confidence intervals; M3 needs only the two scalars below.

Label convention (project-wide): 0 = smooth/early-type, 1 = features/disk = spiral.
"""
from __future__ import annotations

import numpy as np


def accuracy(decisions, y) -> float:
    """Fraction of decisions equal to the (crowd-consensus) label `y`.

    Note (HANDOFF §9): `y` is the debiased crowd-majority label, so this is "agreement with crowd
    consensus", not agreement with an objective truth — the report must frame it that way.
    """
    d = np.asarray(decisions)
    t = np.asarray(y)
    if d.shape[0] != t.shape[0]:
        raise ValueError(f"length mismatch: decisions={d.shape[0]} vs y={t.shape[0]}")
    if d.shape[0] == 0:
        raise ValueError("accuracy is undefined on an empty input")
    return float((d == t).mean())


def expected_human_cost(queries) -> float:
    """Mean human-query cost — the project's x-axis (GROUND_TRUTH §2).

    For HCT each instance costs 1 (agreement) or 2 (disagreement), so the mean lies in [1, 2].
    """
    q = np.asarray(queries)
    if q.shape[0] == 0:
        raise ValueError("expected_human_cost is undefined on an empty input")
    return float(q.mean())


def _check_pair(decisions, y):
    d = np.asarray(decisions)
    t = np.asarray(y)
    if d.shape[0] != t.shape[0]:
        raise ValueError(f"length mismatch: decisions={d.shape[0]} vs y={t.shape[0]}")
    if d.shape[0] == 0:
        raise ValueError("metric is undefined on an empty input")
    return d, t


def tpr(decisions, y) -> float:
    """True-positive rate. Positive class = 1 (spiral, project label convention).

    TPR = #(decision==1 & y==1) / #(y==1). Raises if there are no positives (undefined).
    """
    d, t = _check_pair(decisions, y)
    pos = t == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        raise ValueError("TPR is undefined: no positive (y==1) instances")
    return float((d[pos] == 1).sum() / n_pos)


def fpr(decisions, y) -> float:
    """False-positive rate. FPR = #(decision==1 & y==0) / #(y==0).

    Raises if there are no negatives (undefined).
    """
    d, t = _check_pair(decisions, y)
    neg = t == 0
    n_neg = int(neg.sum())
    if n_neg == 0:
        raise ValueError("FPR is undefined: no negative (y==0) instances")
    return float((d[neg] == 1).sum() / n_neg)


def bootstrap_accuracy_ci(correct, *, n_resamples, ci, seed):
    """Percentile bootstrap CI for a mean over a per-instance correctness vector.

    The single bootstrap used project-wide (invariant §4). `correct` is a 0/1 (or boolean) vector of
    per-instance correctness; this resamples the instances with replacement and returns the
    (lower, upper) percentile interval of the resampled accuracy. Deterministic given `seed`.

    This is the estimator M2's backbone used inline (the AI-alone CI), now absorbed here so every
    arm — and the AI-alone point — share one definition. `backbone.bootstrap_ci` delegates to this;
    the RandomState seeding and quantile convention are byte-for-byte identical, guarded by a
    regression test pinning the frozen AI-alone CI.
    """
    c = np.asarray(correct, dtype=float)
    n = c.shape[0]
    if n == 0:
        raise ValueError("bootstrap_accuracy_ci is undefined on an empty input")
    rng = np.random.RandomState(seed)
    accs = np.empty(int(n_resamples))
    for i in range(int(n_resamples)):
        idx = rng.randint(0, n, size=n)
        accs[i] = c[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(accs, alpha)), float(np.quantile(accs, 1.0 - alpha))
