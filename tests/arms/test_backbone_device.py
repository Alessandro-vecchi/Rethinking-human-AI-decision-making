"""M2 backbone device plumbing — written before the GPU-native refactor (CODING.md/TESTING.md).

Wrapper-level guard that the backbone is no longer CPU-hardcoded: device resolution is centralized,
`train_model` trains through the device path, and `predict_spiral_proba` infers the model's device
(so a CUDA-resident model scores without a manual `.cpu()` dance). These run on CPU, so they stay
green on this machine (CUDA=False) while proving the same code auto-selects CUDA on Colab.
"""
import math

import numpy as np
import torch

from haidc.arms.backbone import (
    build_model,
    predict_spiral_proba,
    resolve_device,
    train_model,
)
from haidc.seed import seed_everything


def test_resolve_device_explicit_and_auto():
    assert resolve_device("cpu").type == "cpu"
    # 'auto' must return a usable device; cpu here (no CUDA), cuda on Colab.
    dev = resolve_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected


def test_train_model_runs_through_device_path():
    """Device plumbing: training runs on the requested device and yields finite per-epoch losses.

    Convergence itself is covered by test_backbone_overfit; here we only assert the device path
    executes end-to-end and places parameters on the requested device.
    """
    seed_everything(0)
    x = np.random.RandomState(1).randn(4, 3, 224, 224).astype("float32")
    y = np.array([0, 1, 0, 1], dtype="int64")
    model = build_model()
    losses = train_model(
        model, x, y, epochs=3, batch_size=4, lr=0.01,
        device="cpu", channels_last=False, return_losses=True,
    )
    assert len(losses) == 3
    assert all(math.isfinite(v) for v in losses)
    assert next(model.parameters()).device.type == "cpu"


def test_predict_infers_model_device():
    seed_everything(0)
    x = np.random.RandomState(2).randn(4, 3, 224, 224).astype("float32")
    model = build_model()  # parameters on CPU; predict must read the device off the model
    scores = predict_spiral_proba(model, x, batch_size=2)
    assert scores.shape == (4,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
