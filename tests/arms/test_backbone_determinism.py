"""M2 backbone determinism — written before haidc.arms.backbone (REPRODUCIBILITY.md).

Reproduction guard: a same-seed build+train+export must produce value-identical scores across
two invocations. Guards "if a number cannot be reproduced it is not a result." Tiny fixture so
it runs on CPU in seconds; catches nondeterminism in init, optimizer, and forward.
"""
import numpy as np

from haidc.arms.backbone import build_model, predict_spiral_proba, train_model
from haidc.seed import seed_everything


def _train_and_score():
    seed_everything(0)
    x = np.random.RandomState(1).randn(4, 3, 224, 224).astype("float32")
    y = np.array([0, 1, 0, 1], dtype="int64")
    model = build_model()
    train_model(model, x, y, epochs=3, batch_size=4, lr=0.01)
    return predict_spiral_proba(model, x, batch_size=4)


def test_same_seed_scores_are_identical():
    a = _train_and_score()
    b = _train_and_score()
    assert np.array_equal(a, b), "same-seed scores diverged (nondeterminism)"
