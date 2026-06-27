"""Embeddings-export contract — written before backbone.export_embeddings (TESTING.md: red->green).

Stage-B infra: the L2D-Okati(learned) and M5 Mozannar arms consume a single
`results/backbone_embeddings.parquet` = [GalaxyID, split, score, e0..e(D-1)], one row per GalaxyID
for each of train/val/test in the frozen-manifest order. The GPU forward pass that produces it runs
on Colab (no CPU backbone tests — established convention); here we guard the PURE frame builder and
the validator the learned arm runs before trusting the file.
"""
import numpy as np
import pytest

from haidc.arms.backbone import build_embeddings_frame
from haidc.arms.l2d_okati import embedding_feature_cols, validate_embeddings_frame


def _toy():
    ids_by_split = {"train": [10, 11], "val": [20], "test": [30, 31, 32]}
    rng = np.random.default_rng(0)
    emb = {k: rng.standard_normal((len(v), 4)) for k, v in ids_by_split.items()}
    sc = {k: rng.random(len(v)) for k, v in ids_by_split.items()}
    return ids_by_split, emb, sc


def test_frame_one_row_per_id_in_split_order():
    ids_by_split, emb, sc = _toy()
    df = build_embeddings_frame(ids_by_split, emb, sc)
    assert list(df.GalaxyID) == [10, 11, 20, 30, 31, 32]            # train -> val -> test, in order
    assert list(df.split) == ["train", "train", "val", "test", "test", "test"]
    assert embedding_feature_cols(df) == ["e0", "e1", "e2", "e3"]   # numerically sorted feature cols
    assert "score" in df.columns
    assert not df.GalaxyID.duplicated().any()


def test_frame_rejects_length_mismatch():
    ids_by_split, emb, sc = _toy()
    sc["val"] = np.array([0.1, 0.2])  # 2 scores for 1 val id
    with pytest.raises(ValueError):
        build_embeddings_frame(ids_by_split, emb, sc)


def test_validate_accepts_complete_frame_and_rejects_missing_id():
    ids_by_split, emb, sc = _toy()
    df = build_embeddings_frame(ids_by_split, emb, sc)
    manifest = {"train": [10, 11], "val": [20], "test": [30, 31, 32]}
    validate_embeddings_frame(df, manifest)                         # complete -> no raise
    with pytest.raises(AssertionError):
        validate_embeddings_frame(df[df.GalaxyID != 32], manifest)  # missing a test id -> loud


def test_validate_rejects_frame_without_score():
    ids_by_split, emb, sc = _toy()
    df = build_embeddings_frame(ids_by_split, emb, sc).drop(columns=["score"])
    with pytest.raises(AssertionError):
        validate_embeddings_frame(df, {"train": [10, 11], "val": [20], "test": [30, 31, 32]})
