"""Evaluation metrics with explicit baselines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y_true: object, probability: object) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    prevalence = float(y.mean())
    return {
        "n": int(len(y)),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "auprc_baseline": prevalence,
        "brier": float(brier_score_loss(y, p)),
        "brier_prevalence_baseline": prevalence * (1.0 - prevalence),
    }


def regression_metrics(
    y_true: object,
    prediction: object,
    *,
    training_fold_mean_prediction: object,
) -> dict[str, float]:
    """Score held-out predictions against baselines estimated in training folds only."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    baseline = np.asarray(training_fold_mean_prediction, dtype=float)
    if y.shape != p.shape or y.shape != baseline.shape or y.ndim != 1:
        raise ValueError("Regression metrics require aligned one-dimensional arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(baseline).all():
        raise ValueError("Regression metrics require finite values.")
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "training_fold_mean_predictor_mae": float(mean_absolute_error(y, baseline)),
        "training_fold_mean_predictor_rmse": float(np.sqrt(mean_squared_error(y, baseline))),
    }
