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
