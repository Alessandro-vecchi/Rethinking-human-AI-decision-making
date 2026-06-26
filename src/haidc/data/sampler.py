"""HCT h1/h2 draw from a per-image vote urn (M1).

Each image's urn is the raw Willett vote counts {n_smooth, n_features}. h1 and h2 are two
DISTINCT votes drawn WITHOUT replacement (a hypergeometric draw of 2 from N >= 2 votes).
Labels: 0 = smooth/early-type, 1 = features/disk-spiral.

Correlation note (HANDOFF §6): two without-replacement draws from a single fixed urn are
*slightly negatively* correlated within that urn. The positive human-human correlation Berger
cares about lives in the BETWEEN-image variance of urn composition (resolution-hard galaxies
-> near-50/50 urns -> frequent h1/h2 disagreement), not within a single urn.
"""
from __future__ import annotations

import numpy as np

SMOOTH, FEATURES = 0, 1


def draw_h1_h2(n_smooth: int, n_features: int, rng: np.random.Generator) -> tuple[int, int]:
    """Return (h1, h2): two distinct votes drawn without replacement from the urn.

    Raises ValueError if the urn holds fewer than 2 votes (no without-replacement draw).
    """
    n_smooth = int(n_smooth)
    n_features = int(n_features)
    total = n_smooth + n_features
    if total < 2:
        raise ValueError(
            f"urn needs >= 2 votes for an h1/h2 draw, got n_smooth={n_smooth}, "
            f"n_features={n_features} (total={total})"
        )
    urn = np.array([SMOOTH] * n_smooth + [FEATURES] * n_features, dtype=np.int64)
    i, j = rng.choice(total, size=2, replace=False)
    return int(urn[i]), int(urn[j])
