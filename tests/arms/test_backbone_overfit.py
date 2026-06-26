"""M2 backbone single-batch overfit — written before haidc.arms.backbone (TESTING.md).

Mandatory sanity check before any full run: the Okati-faithful model (scratch resnet50 +
Linear(2048,2) + LogSoftmax, NLLLoss) must be able to drive a single small batch to ~0 loss,
i.e. it can learn at all. Kept tiny (4 synthetic images) so it runs on CPU in seconds.
"""
import numpy as np

from haidc.arms.backbone import build_model, train_model
from haidc.seed import seed_everything


def test_single_batch_overfits_to_near_zero_loss():
    seed_everything(0)
    x = np.random.RandomState(0).randn(4, 3, 224, 224).astype("float32")
    y = np.array([0, 1, 0, 1], dtype="int64")
    model = build_model()
    losses = train_model(model, x, y, epochs=60, batch_size=4, lr=0.01, return_losses=True)
    assert losses[-1] < 0.05, f"failed to overfit one batch: final NLLLoss={losses[-1]:.4f}"
    assert losses[-1] < losses[0]  # loss actually decreased
