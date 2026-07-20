"""Fold-local calibration interfaces for modality outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


class CalibrationError(ValueError):
    """Calibration could not be fitted without violating its contract."""


def _validate_calibration_inputs(
    model_va: object,
    human_va: object,
    item_ids: object,
    *,
    minimum_items: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(model_va, dtype=float)
    y = np.asarray(human_va, dtype=float)
    ids = np.asarray(item_ids, dtype=object)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or len(ids) != len(x):
        raise CalibrationError("Expected aligned n-by-2 model/human arrays and n item IDs.")
    if len(set(ids.tolist())) != len(ids):
        raise CalibrationError("Calibration item IDs must be unique.")
    if len(x) < minimum_items or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CalibrationError(
            f"Calibration requires at least {minimum_items} finite observations."
        )
    return x, y, ids


def _assert_held_out(train_ids: frozenset[object], item_ids: object | None) -> None:
    if item_ids is not None and train_ids.intersection(np.asarray(item_ids).tolist()):
        raise CalibrationError(
            "Requested strict held-out transform includes calibration-training items."
        )


@dataclass
class DimensionwiseAffineCalibrator:
    """Primary mapping: human V from model V and human A from model A only."""

    coefficients_: np.ndarray | None = None
    train_ids_: frozenset[object] = frozenset()
    mapping_name: str = "dimensionwise_affine"

    def fit(
        self, model_va: object, human_va: object, item_ids: object
    ) -> DimensionwiseAffineCalibrator:
        x, y, ids = _validate_calibration_inputs(model_va, human_va, item_ids, minimum_items=3)
        self.coefficients_ = np.asarray(
            [
                np.linalg.lstsq(
                    np.column_stack([np.ones(len(x)), x[:, dimension]]),
                    y[:, dimension],
                    rcond=None,
                )[0]
                for dimension in range(2)
            ]
        )
        self.train_ids_ = frozenset(ids.tolist())
        return self

    def transform(self, model_va: object, item_ids: object | None = None) -> np.ndarray:
        if self.coefficients_ is None:
            raise CalibrationError("Calibrator is not fitted.")
        x = np.asarray(model_va, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise CalibrationError("Expected an n-by-2 model array.")
        _assert_held_out(self.train_ids_, item_ids)
        return np.column_stack(
            [
                self.coefficients_[dimension, 0]
                + self.coefficients_[dimension, 1] * x[:, dimension]
                for dimension in range(2)
            ]
        )


@dataclass
class CrossDimensionalAffineCalibrator:
    """Sensitivity mapping allowing both model dimensions to predict each human dimension."""

    coefficients_: np.ndarray | None = None
    train_ids_: frozenset[object] = frozenset()
    mapping_name: str = "cross_dimensional_affine_sensitivity"

    def fit(
        self, model_va: object, human_va: object, item_ids: object
    ) -> CrossDimensionalAffineCalibrator:
        x, y, ids = _validate_calibration_inputs(model_va, human_va, item_ids, minimum_items=3)
        design = np.column_stack([np.ones(len(x)), x])
        self.coefficients_ = np.column_stack(
            [np.linalg.lstsq(design, y[:, dimension], rcond=None)[0] for dimension in range(2)]
        )
        self.train_ids_ = frozenset(ids.tolist())
        return self

    def transform(self, model_va: object, item_ids: object | None = None) -> np.ndarray:
        if self.coefficients_ is None:
            raise CalibrationError("Calibrator is not fitted.")
        x = np.asarray(model_va, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise CalibrationError("Expected an n-by-2 model array.")
        _assert_held_out(self.train_ids_, item_ids)
        return np.column_stack([np.ones(len(x)), x]) @ self.coefficients_


class AffineCalibrator(DimensionwiseAffineCalibrator):
    """Backward-compatible name for the primary dimension-wise affine mapping."""


@dataclass
class RobustStandardizer:
    """Median/IQR transform fitted on a declared reference set."""

    median_: np.ndarray | None = None
    iqr_: np.ndarray | None = None

    def fit(self, values: object) -> RobustStandardizer:
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise CalibrationError("Expected finite n-by-2 reference values.")
        self.median_ = np.median(x, axis=0)
        self.iqr_ = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
        if (self.iqr_ <= 0).any():
            raise CalibrationError("Reference IQR must be positive in both dimensions.")
        return self

    def transform(self, values: object) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise CalibrationError("Standardizer is not fitted.")
        return (np.asarray(values, dtype=float) - self.median_) / self.iqr_


@dataclass
class RobustLocationScaleCalibrator:
    """Map model V/A to human V/A using train-fold medians and IQRs."""

    model_median_: np.ndarray | None = None
    model_iqr_: np.ndarray | None = None
    human_median_: np.ndarray | None = None
    human_iqr_: np.ndarray | None = None
    train_ids_: frozenset[object] = frozenset()

    def fit(
        self, model_va: object, human_va: object, item_ids: object
    ) -> RobustLocationScaleCalibrator:
        x = np.asarray(model_va, dtype=float)
        y = np.asarray(human_va, dtype=float)
        ids = np.asarray(item_ids, dtype=object)
        if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or len(ids) != len(x):
            raise CalibrationError("Expected aligned n-by-2 model/human arrays and n item IDs.")
        if len(x) < 4 or len(set(ids.tolist())) != len(ids):
            raise CalibrationError("Robust calibration requires four unique training items.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise CalibrationError("Robust calibration requires finite values.")
        self.model_median_ = np.median(x, axis=0)
        self.human_median_ = np.median(y, axis=0)
        self.model_iqr_ = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
        self.human_iqr_ = np.quantile(y, 0.75, axis=0) - np.quantile(y, 0.25, axis=0)
        if (self.model_iqr_ <= 0).any() or (self.human_iqr_ <= 0).any():
            raise CalibrationError("Robust calibration requires positive model and human IQRs.")
        self.train_ids_ = frozenset(ids.tolist())
        return self

    def transform(self, model_va: object, item_ids: object | None = None) -> np.ndarray:
        parameters = (
            self.model_median_,
            self.model_iqr_,
            self.human_median_,
            self.human_iqr_,
        )
        if any(value is None for value in parameters):
            raise CalibrationError("Robust calibrator is not fitted.")
        x = np.asarray(model_va, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise CalibrationError("Expected a finite n-by-2 model array.")
        if item_ids is not None and self.train_ids_.intersection(np.asarray(item_ids).tolist()):
            raise CalibrationError(
                "Requested strict held-out transform includes calibration-training items."
            )
        assert self.model_median_ is not None
        assert self.model_iqr_ is not None
        assert self.human_median_ is not None
        assert self.human_iqr_ is not None
        standardized = (x - self.model_median_) / self.model_iqr_
        return standardized * self.human_iqr_ + self.human_median_


@dataclass
class MonotonicCalibrator:
    """Independent isotonic V/A mappings for adequately sized calibration sets."""

    minimum_items: int = 30
    models_: tuple[IsotonicRegression, IsotonicRegression] | None = None

    def fit(self, model_va: object, human_va: object) -> MonotonicCalibrator:
        x = np.asarray(model_va, dtype=float)
        y = np.asarray(human_va, dtype=float)
        if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2:
            raise CalibrationError("Expected aligned n-by-2 model/human arrays.")
        if len(x) < self.minimum_items:
            raise CalibrationError("Insufficient items for prespecified monotonic calibration.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise CalibrationError("Monotonic calibration requires finite values.")
        self.models_ = (
            IsotonicRegression(out_of_bounds="clip").fit(x[:, 0], y[:, 0]),
            IsotonicRegression(out_of_bounds="clip").fit(x[:, 1], y[:, 1]),
        )
        return self

    def transform(self, model_va: object) -> np.ndarray:
        if self.models_ is None:
            raise CalibrationError("Monotonic calibrator is not fitted.")
        x = np.asarray(model_va, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise CalibrationError("Expected an n-by-2 model array.")
        return np.column_stack(
            [self.models_[dimension].predict(x[:, dimension]) for dimension in range(2)]
        )
