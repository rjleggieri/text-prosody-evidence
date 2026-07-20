import numpy as np
import pytest

from text_prosody.calibration import (
    AffineCalibrator,
    CalibrationError,
    CrossDimensionalAffineCalibrator,
    DimensionwiseAffineCalibrator,
    MonotonicCalibrator,
    RobustLocationScaleCalibrator,
    RobustStandardizer,
)


def test_affine_calibration_recovers_linear_mapping() -> None:
    model = np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0], [0.2, 0.8]])
    human = np.column_stack([1 + 2 * model[:, 0], 3 - model[:, 1]])
    calibrator = AffineCalibrator().fit(model, human, ["A", "B", "C", "D"])
    transformed = calibrator.transform(np.array([[0.25, 0.5]]), ["HELD_OUT"])
    np.testing.assert_allclose(transformed, [[1.5, 2.5]], atol=1e-8)
    with pytest.raises(CalibrationError, match="training items"):
        calibrator.transform(model[:1], ["A"])


def test_primary_affine_mapping_cannot_cross_dimensions() -> None:
    model = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 4.0]])
    human = np.column_stack([1 + 2 * model[:, 0], 3 - model[:, 1]])
    calibrator = DimensionwiseAffineCalibrator().fit(model, human, ["A", "B", "C", "D"])
    first = calibrator.transform([[0.5, -100]], ["X"])
    second = calibrator.transform([[0.5, 100]], ["Y"])
    assert first[0, 0] == pytest.approx(second[0, 0])
    assert calibrator.coefficients_ is not None
    assert calibrator.coefficients_.shape == (2, 2)


def test_cross_dimensional_affine_is_explicit_sensitivity_only() -> None:
    model = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 4.0]])
    human = np.column_stack(
        [1 + 2 * model[:, 0] + 0.5 * model[:, 1], 3 - model[:, 1] + model[:, 0]]
    )
    calibrator = CrossDimensionalAffineCalibrator().fit(model, human, ["A", "B", "C", "D"])
    first = calibrator.transform([[0.5, -1]], ["X"])
    second = calibrator.transform([[0.5, 1]], ["Y"])
    assert first[0, 0] != pytest.approx(second[0, 0])
    assert calibrator.mapping_name == "cross_dimensional_affine_sensitivity"


def test_robust_standardizer() -> None:
    values = np.array([[0, 10], [1, 20], [2, 30], [3, 40]], dtype=float)
    transformed = RobustStandardizer().fit(values).transform(values)
    np.testing.assert_allclose(np.median(transformed, axis=0), [0, 0])


def test_robust_location_scale_calibration_is_fold_local() -> None:
    model = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
    human = 1 + 2 * model
    calibrator = RobustLocationScaleCalibrator().fit(model, human, ["A", "B", "C", "D", "E"])
    np.testing.assert_allclose(calibrator.transform([[2.5, 3.5]], ["HELD_OUT"]), [[6, 8]])
    with pytest.raises(CalibrationError, match="training items"):
        calibrator.transform(model[:1], ["A"])


def test_monotonic_calibrator_requires_adequate_data() -> None:
    values = np.column_stack([np.linspace(0, 1, 30), np.linspace(1, 0, 30)])
    human = np.column_stack([values[:, 0] ** 2, np.sqrt(values[:, 1])])
    output = MonotonicCalibrator().fit(values, human).transform(values)
    assert np.all(np.diff(output[:, 0]) >= 0)
    with pytest.raises(CalibrationError, match="Insufficient"):
        MonotonicCalibrator().fit(values[:10], human[:10])
