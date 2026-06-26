"""M2 backbone data join + reported-metric sanity — before haidc.arms.backbone (TESTING.md).

The probe (DECISIONS 2026-06-26) found Okati ships no aligned embeddings, so M2 trains fresh on
images keyed by GalaxyID. The join that must be loud (CODING.md "failure is loud") is therefore
label coverage of the frozen split + image presence. Also pins the two reported numbers
(AI-alone accuracy, bootstrap CI) on a hand-computed fixture, since the shared eval module (M6)
does not exist yet.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from haidc.arms.backbone import (
    ai_alone_accuracy,
    bootstrap_ci,
    load_images_for_ids,
    load_labels,
    load_split,
)

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "split_manifest.json"
LABELS = REPO / "data" / "label_table.parquet"


def test_every_split_id_has_a_debiased_label():
    split = load_split(MANIFEST)
    labels = load_labels(LABELS)
    all_ids = split["train"] + split["val"] + split["test"]
    missing = [g for g in all_ids if g not in labels]
    assert missing == [], f"{len(missing)} split GalaxyIDs lack a y_debiased label"
    assert set(labels[g] for g in all_ids) <= {0, 1}  # binary only


def test_missing_image_dir_fails_loud():
    with pytest.raises(FileNotFoundError):
        load_images_for_ids([100053], REPO / "data" / "raw" / "does_not_exist")


def test_ai_alone_accuracy_matches_hand_count():
    scores = np.array([0.9, 0.1, 0.6, 0.4])
    y = np.array([1, 0, 1, 1])
    # thr 0.5 -> preds [1,0,1,0] -> 3/4 correct
    assert ai_alone_accuracy(scores, y, threshold=0.5) == pytest.approx(0.75)


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.RandomState(0)
    scores = rng.rand(200)
    y = (scores > 0.5).astype(int)
    point = ai_alone_accuracy(scores, y, threshold=0.5)
    lo1, hi1 = bootstrap_ci(scores, y, threshold=0.5, n_boot=500, ci=0.95, seed=0)
    lo2, hi2 = bootstrap_ci(scores, y, threshold=0.5, n_boot=500, ci=0.95, seed=0)
    assert (lo1, hi1) == (lo2, hi2)           # deterministic given seed
    assert 0.0 <= lo1 <= point <= hi1 <= 1.0  # CI brackets the point estimate


def test_manifest_test_split_has_expected_size():
    split = json.loads(MANIFEST.read_text())
    assert len(split["test"]) == 694
