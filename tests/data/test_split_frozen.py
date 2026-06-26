"""M1 frozen split — written before haidc.data.split (TESTING.md, REPRODUCIBILITY.md).

The split must be deterministic from seed=0 and hash-stable; `make verify-split` recomputes it
and asserts the committed manifest hash. Guards CLAUDE.md invariant #2 (one frozen split).
"""
import json
import math
from pathlib import Path

import pytest

from haidc.data.split import build_manifest, compute_split

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
REPO = Path(__file__).resolve().parents[2]


def _ids(n):
    return list(range(1000, 1000 + n))


def test_split_is_deterministic():
    a = compute_split(_ids(20), seed=0, ratios=RATIOS)
    b = compute_split(_ids(20), seed=0, ratios=RATIOS)
    assert a == b


def test_split_sizes_floor_with_remainder_to_test():
    n = 4621
    s = compute_split(_ids(n), seed=0, ratios=RATIOS)
    n_train, n_val, n_test = len(s["train"]), len(s["val"]), len(s["test"])
    assert n_train == math.floor(0.70 * n)         # 3234
    assert n_val == math.floor(0.15 * n)           # 693
    assert n_train + n_val + n_test == n           # remainder -> test (694)


def test_splits_are_disjoint_and_cover_all():
    n = 50
    s = compute_split(_ids(n), seed=0, ratios=RATIOS)
    union = set(s["train"]) | set(s["val"]) | set(s["test"])
    assert union == set(_ids(n))
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == n  # no element twice


def test_different_seed_changes_assignment():
    a = compute_split(_ids(100), seed=0, ratios=RATIOS)
    b = compute_split(_ids(100), seed=1, ratios=RATIOS)
    assert a != b


def test_manifest_content_hash_is_stable_and_sensitive():
    ids = _ids(30)
    m1 = build_manifest(compute_split(ids, 0, RATIOS), seed=0, ratios=RATIOS, extra={})
    m2 = build_manifest(compute_split(ids, 0, RATIOS), seed=0, ratios=RATIOS, extra={})
    assert m1["content_hash"] == m2["content_hash"]
    # moving one id to a different split must change the hash
    perturbed = compute_split(ids, 0, RATIOS)
    perturbed["train"].append(perturbed["test"].pop())
    m3 = build_manifest(perturbed, seed=0, ratios=RATIOS, extra={})
    assert m3["content_hash"] != m1["content_hash"]


@pytest.mark.skipif(
    not (REPO / "data" / "split_manifest.json").exists()
    or not (REPO / "data" / "raw").exists(),
    reason="committed manifest and/or raw data not present (gate is `make verify-split`)",
)
def test_recomputed_split_matches_committed_manifest():
    # Authoritative reproduction guard when the data machine has the inputs.
    from haidc.data.verify_split import recompute_and_compare

    committed = json.loads((REPO / "data" / "split_manifest.json").read_text())
    recomputed = recompute_and_compare(REPO / "configs" / "data.yaml")
    assert recomputed["content_hash"] == committed["content_hash"]
