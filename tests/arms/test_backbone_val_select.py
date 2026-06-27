"""M2 backbone val tracking + best-VAL checkpoint selection — written before the implementation
(TESTING.md). This is the final dispositive run's new behavior (DECISIONS 2026-06-27):

- training records per-epoch train/val accuracy + loss curves (to read under-training vs a label
  ceiling off the train-acc curve),
- the exported backbone is the *best-VAL-accuracy* checkpoint, not the last epoch.

Tiny synthetic fixtures so everything runs on CPU in seconds; no trained model / real images.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from haidc.arms import backbone as bb
from haidc.arms.backbone import (
    build_model,
    select_best_epoch,
    train_model,
)
from haidc.seed import seed_everything


# --------------------------------------------------------------------------- select_best_epoch
def test_select_best_epoch_returns_argmax():
    # peak strictly before the last epoch -> index 1, not the final index.
    assert select_best_epoch([0.50, 0.90, 0.80]) == 1


def test_select_best_epoch_breaks_ties_to_earliest():
    # two equal maxima -> the earliest wins (cheaper checkpoint, deterministic).
    assert select_best_epoch([0.50, 0.70, 0.70, 0.60]) == 1


# --------------------------------------------------------------------------- train history
def _train(select_best_val):
    """Train a fresh model on a tiny train+val fixture; return (history, model)."""
    seed_everything(0)
    rs = np.random.RandomState(1)
    X = rs.randn(8, 3, 224, 224).astype("float32")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="int64")
    X_val = rs.randn(4, 3, 224, 224).astype("float32")
    y_val = np.array([0, 1, 1, 0], dtype="int64")
    model = build_model()
    hist = train_model(
        model, X, y, epochs=5, batch_size=4, lr=0.01,
        device="cpu", channels_last=False,
        X_val=X_val, y_val=y_val, select_best_val=select_best_val, return_history=True,
    )
    return hist, model


def test_history_has_per_epoch_curves():
    hist, _ = _train(select_best_val=False)
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert len(hist[key]) == 5, f"{key} curve length != epochs"
        assert all(math.isfinite(v) for v in hist[key]), f"{key} has non-finite values"
    for key in ("train_acc", "val_acc"):
        assert all(0.0 <= v <= 1.0 for v in hist[key]), f"{key} outside [0,1]"
    assert 1 <= hist["best_epoch"] <= 5


def test_best_epoch_is_argmax_of_val_curve():
    hist, _ = _train(select_best_val=True)
    assert hist["best_epoch"] == select_best_epoch(hist["val_acc"]) + 1  # 1-indexed in manifest
    assert hist["val_acc"][hist["best_epoch"] - 1] == max(hist["val_acc"])


def test_select_best_val_loads_best_not_last():
    """select_best_val swaps in the best-VAL weights. Training is identical either way (selection
    happens only at the end), so the two runs share a val curve; the exported *weights* differ iff
    the peak was earlier than the final epoch — and match exactly when the peak IS the last epoch.
    (Assert on weights, not scores: on this degenerate fixture the head saturates so both produce
    identical near-zero probabilities even though the underlying weights differ.)
    """
    import torch

    hist_best, model_best = _train(select_best_val=True)
    hist_last, model_last = _train(select_best_val=False)
    assert hist_best["val_acc"] == hist_last["val_acc"]  # identical training trajectory
    w_best = model_best.state_dict()["fc.0.weight"]
    w_last = model_last.state_dict()["fc.0.weight"]
    if hist_best["best_epoch"] != 5:
        assert not torch.equal(w_best, w_last)  # best-VAL checkpoint != last epoch
    else:
        assert torch.equal(w_best, w_last)


# --------------------------------------------------------------------------- back-compat
def test_return_losses_still_yields_train_loss_list():
    # existing callers (device/overfit/determinism tests) must keep working unchanged.
    seed_everything(0)
    x = np.random.RandomState(1).randn(4, 3, 224, 224).astype("float32")
    y = np.array([0, 1, 0, 1], dtype="int64")
    losses = train_model(build_model(), x, y, epochs=3, batch_size=4, lr=0.01, return_losses=True)
    assert len(losses) == 3 and all(math.isfinite(v) for v in losses)


# --------------------------------------------------------------------------- run() manifest
def test_run_manifest_carries_val_fields(tmp_path, monkeypatch):
    """run() trains on train, selects on val, scores test from the best-VAL checkpoint, and records
    the val fields + curves. Split/labels/images are stubbed tiny so it runs on CPU in seconds.
    """
    train_ids, val_ids, test_ids = [1, 2, 3, 4], [5, 6], [7, 8]
    monkeypatch.setattr(
        bb, "load_split", lambda p: {"train": train_ids, "val": val_ids, "test": test_ids}
    )
    monkeypatch.setattr(
        bb, "load_labels", lambda p: {g: g % 2 for g in train_ids + val_ids + test_ids}
    )

    def _fake_images(ids, images_dir):
        import torch

        g = torch.Generator().manual_seed(len(ids))
        return torch.randn(len(ids), 3, 224, 224, generator=g)

    monkeypatch.setattr(bb, "load_images_for_ids", _fake_images)

    cfg = {
        "seed": 0,
        "features": "fresh",
        "threshold_default": 0.5,
        "manifest_path": "unused",
        "label_table_path": "unused",
        "images_dir": "unused",
        "train": {"epochs": 2, "batch_size": 2, "lr": 0.01, "weight_decay": 0.0,
                  "device": "cpu", "channels_last": False},
        "bootstrap": {"n_boot": 10, "ci": 0.95},
        "artifact_path": str(tmp_path / "backbone.pt"),
        "run_manifest_path": str(tmp_path / "backbone_run.json"),
        "export_scores_path": str(tmp_path / "backbone_scores.parquet"),
    }
    m = bb.run(cfg)
    assert m["n_val"] == 2
    assert 0.0 <= m["val_accuracy"] <= 1.0
    assert 1 <= m["best_epoch"] <= 2
    for key in ("train_acc_curve", "val_acc_curve", "val_loss_curve", "train_loss_curve"):
        assert len(m[key]) == 2, f"{key} length != epochs"
    # persisted and reloadable
    on_disk = json.loads(Path(cfg["run_manifest_path"]).read_text())
    assert on_disk["val_accuracy"] == m["val_accuracy"]
