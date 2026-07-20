import numpy as np
import pytest

from text_prosody.discrepancy import (
    ScaleMismatchError,
    l1_discrepancy,
    l2_discrepancy,
    require_commensurate_scales,
    signed_components,
)


def test_signed_direction_and_magnitudes() -> None:
    delta_v, delta_a = signed_components([0.8], [0.2], [0.3], [0.6])
    np.testing.assert_allclose(delta_v, [0.5])
    np.testing.assert_allclose(delta_a, [-0.4])
    np.testing.assert_allclose(l2_discrepancy([0.8], [0.2], [0.3], [0.6]), [np.hypot(0.5, 0.4)])
    np.testing.assert_allclose(l1_discrepancy([0.8], [0.2], [0.3], [0.6]), [0.9])


def test_scale_guard() -> None:
    require_commensurate_scales("human_va_1_7", "human_va_1_7")
    with pytest.raises(ScaleMismatchError):
        require_commensurate_scales("model_0_1", "human_va_1_7")
