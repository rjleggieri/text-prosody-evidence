import numpy as np
import pytest

from text_prosody.metrics import regression_metrics


def test_regression_mean_baseline_requires_training_fold_predictions() -> None:
    observed = np.array([10.0, 12.0])
    prediction = np.array([10.5, 11.5])
    training_fold_means = np.array([2.0, 4.0])
    metrics = regression_metrics(
        observed,
        prediction,
        training_fold_mean_prediction=training_fold_means,
    )
    assert metrics["training_fold_mean_predictor_mae"] == pytest.approx(8.0)
    assert "mean_predictor_mae" not in metrics


def test_regression_metrics_reject_missing_training_baseline_shape() -> None:
    with pytest.raises(ValueError, match="aligned"):
        regression_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            training_fold_mean_prediction=[1.5],
        )
