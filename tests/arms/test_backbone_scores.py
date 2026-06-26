"""M2 backbone score-export contract — written before haidc.arms.backbone (TESTING.md).

The exported score table is the single interface every later arm consumes (CODING.md). It must
have exactly one row per frozen TEST GalaxyID, in the manifest's order, with continuous
P(spiral|x) in [0,1]. These tests use synthetic features so they need no trained model / images.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from haidc.arms.backbone import (
    SPIRAL_CLASS,
    build_model,
    export_scores,
    predict_spiral_proba,
)

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "split_manifest.json"


def _test_ids():
    return json.loads(MANIFEST.read_text())["test"]


def test_spiral_is_class_index_one():
    # P(spiral|x) = P(class 1); class 0 = smooth/early-type (= y_debiased==1; DECISIONS 2026-06-26).
    assert SPIRAL_CLASS == 1


def test_export_one_row_per_test_id_in_manifest_order():
    ids = _test_ids()
    scores = np.linspace(0.0, 1.0, len(ids))
    df = export_scores(ids, scores)
    assert len(df) == len(ids) == 694
    assert df["GalaxyID"].tolist() == ids          # coverage + ordering match the manifest
    assert df["GalaxyID"].is_unique               # no dupes
    assert np.allclose(df["score"].to_numpy(), scores)


def test_export_rejects_length_mismatch():
    ids = _test_ids()
    with pytest.raises((ValueError, AssertionError)):
        export_scores(ids, np.zeros(len(ids) - 1))


def test_predict_spiral_proba_in_unit_interval():
    # An untrained model must still emit valid probabilities for every instance.
    model = build_model()
    x = np.random.RandomState(0).randn(6, 3, 224, 224).astype("float32")
    proba = predict_spiral_proba(model, x, batch_size=4)
    assert proba.shape == (6,)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
