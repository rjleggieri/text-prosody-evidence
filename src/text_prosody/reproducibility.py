"""Deterministic seed settings for optional Torch benchmark workflows."""

from __future__ import annotations

import os
import random

import numpy as np


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch and request deterministic kernels where supported."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
