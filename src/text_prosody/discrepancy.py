"""Pure discrepancy functions; no I/O."""

from __future__ import annotations

import numpy as np


class ScaleMismatchError(ValueError):
    """Raised when uncalibrated modality outputs would be compared."""


def _arrays(
    text_valence: object,
    text_arousal: object,
    audio_valence: object,
    audio_arousal: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tv, ta, av, aa = tuple(
        np.asarray(value, dtype=float)
        for value in (text_valence, text_arousal, audio_valence, audio_arousal)
    )
    arrays = (tv, ta, av, aa)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("All affect arrays must have identical shapes.")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Affect arrays must be finite.")
    return tv, ta, av, aa


def require_commensurate_scales(text_scale: str, audio_scale: str) -> None:
    if not text_scale or not audio_scale or text_scale != audio_scale:
        raise ScaleMismatchError(
            "Text and audio outputs require the same validated/calibrated output scale."
        )


def signed_components(
    text_valence: object,
    text_arousal: object,
    audio_valence: object,
    audio_arousal: object,
) -> tuple[np.ndarray, np.ndarray]:
    tv, ta, av, aa = _arrays(text_valence, text_arousal, audio_valence, audio_arousal)
    return tv - av, ta - aa


def l2_discrepancy(*values: object) -> np.ndarray:
    delta_v, delta_a = signed_components(*values)
    return np.hypot(delta_v, delta_a)


def l1_discrepancy(*values: object) -> np.ndarray:
    delta_v, delta_a = signed_components(*values)
    return np.abs(delta_v) + np.abs(delta_a)


def cosine_discrepancy(*values: object, epsilon: float = 1e-12) -> np.ndarray:
    tv, ta, av, aa = _arrays(*values)
    dot = tv * av + ta * aa
    denom = np.hypot(tv, ta) * np.hypot(av, aa)
    return 1.0 - np.divide(dot, denom, out=np.zeros_like(dot), where=denom > epsilon)
