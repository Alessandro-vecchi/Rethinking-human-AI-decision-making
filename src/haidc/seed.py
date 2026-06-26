"""Single source of determinism (docs/conventions/REPRODUCIBILITY.md).

Call ``seed_everything(seed)`` at every entry point. ``torch`` is imported lazily so the
data milestone (M1) does not depend on it being installed.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed random / numpy / torch (CPU+CUDA) and force deterministic torch ops."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # M1 (data) needs no torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
