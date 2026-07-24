import random
import sys
from pathlib import Path

import numpy as np
import pytest

from text_prosody.reproducibility import set_deterministic_seed


def test_torch_seed_and_deterministic_settings_are_reproducible() -> None:
    torch = pytest.importorskip("torch")
    set_deterministic_seed(17)
    first = torch.rand(5)
    set_deterministic_seed(17)
    second = torch.rand(5)
    torch.testing.assert_close(first, second)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_core_seed_is_reproducible_when_optional_torch_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    set_deterministic_seed(17)
    first = (random.random(), float(np.random.random()))
    set_deterministic_seed(17)
    second = (random.random(), float(np.random.random()))
    assert first == second


def test_environment_and_code_license_are_explicitly_pinned() -> None:
    root = Path(__file__).parents[1]
    lock_lines = [
        line.strip()
        for line in (root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("--")
    ]
    assert lock_lines
    assert all("==" in line for line in lock_lines)
    assert (root / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
