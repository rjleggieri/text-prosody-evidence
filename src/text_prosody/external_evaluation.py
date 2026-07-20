"""Fold-local calibration and evaluation against external human modality references."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from .calibration import (
    CrossDimensionalAffineCalibrator,
    DimensionwiseAffineCalibrator,
    RobustLocationScaleCalibrator,
)
from .external_benchmark import (
    LEAKAGE_RELATION_COLUMNS,
    assert_group_split_is_disjoint,
    attach_connected_leakage_components,
    composite_item_keys,
    paired_human_differences,
    validate_paired_human_reference,
)

MODEL_COLUMNS = [
    "model_text_valence",
    "model_text_arousal",
    "model_audio_valence",
    "model_audio_arousal",
]
OUTCOME_TOKENS = ("phq", "depress", "diagnos", "outcome")
MINIMUM_VALID_BOOTSTRAP_FRACTION = 0.90
PRIMARY_CALIBRATION_METHOD = "dimensionwise_affine"
CLIPPED_TO_SCALE_SENSITIVITY = "clipped_to_declared_scale"
PRIMARY_SELECTION_OBJECTIVE = "mean_text_valence_arousal_rmse"
PRIMARY_SELECTION_TIE_BREAK = "objective_then_candidate_name"
EXTERNAL_METRIC_PAIRS = {
    "text_valence": ("human_text_valence", "pred_text_valence"),
    "text_arousal": ("human_text_arousal", "pred_text_arousal"),
    "audio_valence": ("human_audio_valence", "pred_audio_valence"),
    "audio_arousal": ("human_audio_arousal", "pred_audio_arousal"),
    "delta_valence": ("human_delta_valence", "pred_delta_valence"),
    "delta_arousal": ("human_delta_arousal", "pred_delta_arousal"),
    "divergence_l2": ("human_divergence_l2", "pred_divergence_l2"),
}


Calibrator = (
    DimensionwiseAffineCalibrator | CrossDimensionalAffineCalibrator | RobustLocationScaleCalibrator
)


def _new_calibrator(method: str) -> Calibrator:
    if method in {"dimensionwise_affine", "affine"}:
        return DimensionwiseAffineCalibrator()
    if method == "cross_dimensional_affine_sensitivity":
        return CrossDimensionalAffineCalibrator()
    if method == "robust_location_scale":
        return RobustLocationScaleCalibrator()
    raise ValueError(f"Unsupported external calibration method: {method}")


def make_synthetic_external_benchmark(
    *,
    groups: int = 20,
    items_per_group: int = 3,
    seed: int = 20260719,
) -> pd.DataFrame:
    """Build a non-scientific paired-modality fixture for end-to-end verification."""
    if groups < 5 or items_per_group < 1:
        raise ValueError("Synthetic external benchmark requires at least five source groups.")
    rng = np.random.default_rng(seed)
    n = groups * items_per_group
    text_raw = rng.normal(size=(n, 2))
    audio_raw = rng.normal(size=(n, 2))
    text_human = 3 + text_raw * np.array([0.8, 0.6]) + rng.normal(0, 0.08, (n, 2))
    audio_human = 3 + audio_raw * np.array([0.7, 0.9]) + rng.normal(0, 0.08, (n, 2))
    frame = pd.DataFrame(
        {
            "dataset_id": "SYNTHETIC_EXTERNAL",
            "dataset_version": "synthetic-fixture-v1",
            "item_id": [f"SYNTHETIC_ITEM_{index:04d}" for index in range(n)],
            "provenance_group_id": [
                f"SYNTHETIC_SOURCE_{index // items_per_group:04d}" for index in range(n)
            ],
            "source_video_id": [
                f"SYNTHETIC_SOURCE_{index // items_per_group:04d}" for index in range(n)
            ],
            "text_annotation_condition": "text_only",
            "audio_annotation_condition": "audio_only",
            "scale_name": "synthetic_1_5",
            "scale_min": 1.0,
            "scale_max": 5.0,
            "human_text_valence": np.clip(text_human[:, 0], 1, 5),
            "human_text_arousal": np.clip(text_human[:, 1], 1, 5),
            "human_audio_valence": np.clip(audio_human[:, 0], 1, 5),
            "human_audio_arousal": np.clip(audio_human[:, 1], 1, 5),
            "model_text_valence": text_raw[:, 0],
            "model_text_arousal": text_raw[:, 1],
            "model_audio_valence": audio_raw[:, 0],
            "model_audio_arousal": audio_raw[:, 1],
        }
    )
    human_divergence = np.hypot(
        frame["human_text_valence"] - frame["human_audio_valence"],
        frame["human_text_arousal"] - frame["human_audio_arousal"],
    )
    frame["synthetic_reference_pattern"] = np.where(
        human_divergence <= np.median(human_divergence), "aligned", "divergent"
    )
    return frame


def _ccc(observed: np.ndarray, predicted: np.ndarray) -> float:
    covariance = float(np.mean((observed - observed.mean()) * (predicted - predicted.mean())))
    denominator = float(
        observed.var() + predicted.var() + (observed.mean() - predicted.mean()) ** 2
    )
    return float(2.0 * covariance / denominator) if denominator > 0 else float("nan")


def _invalid_metric_reason(observed: object, predicted: object, metric: str) -> str:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        return "nonfinite_input"
    if metric in {"pearson", "spearman"}:
        if np.unique(y).size < 2:
            return "constant_observed"
        if np.unique(p).size < 2:
            return "constant_prediction"
    if metric == "ccc":
        denominator = float(y.var() + p.var() + (y.mean() - p.mean()) ** 2)
        if denominator <= 0:
            return "zero_concordance_denominator"
    return "nonfinite_metric"


def _bootstrap_interval_record(
    values: list[float],
    invalid_reasons: Counter[str],
    *,
    replicates: int,
    lower_name: str,
    upper_name: str,
) -> dict[str, Any]:
    minimum_valid = int(np.ceil(MINIMUM_VALID_BOOTSTRAP_FRACTION * replicates))
    valid = len(values)
    available = valid >= minimum_valid
    return {
        lower_name: float(np.quantile(values, 0.025)) if available else None,
        upper_name: float(np.quantile(values, 0.975)) if available else None,
        "attempted_replicates": replicates,
        "valid_replicates": valid,
        "invalid_replicates": replicates - valid,
        "invalid_counts_by_reason": dict(sorted(invalid_reasons.items())),
        "minimum_valid_fraction": MINIMUM_VALID_BOOTSTRAP_FRACTION,
        "minimum_valid_replicates": minimum_valid,
        "interval_available": available,
    }


def continuous_metrics(observed: object, predicted: object) -> dict[str, float]:
    """Standard external-benchmark metrics for one continuous estimand."""
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or len(y) < 3:
        raise ValueError("Metrics require aligned one-dimensional arrays with at least 3 items.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite values.")
    return {
        "n": float(len(y)),
        "pearson": (
            float("nan")
            if np.unique(y).size < 2 or np.unique(p).size < 2
            else float(pearsonr(y, p).statistic)
        ),
        "spearman": (
            float("nan")
            if np.unique(y).size < 2 or np.unique(p).size < 2
            else float(spearmanr(y, p).statistic)
        ),
        "ccc": _ccc(y, p),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def _validate_model_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(MODEL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"External evaluation frame missing model columns: {missing}")
    values = frame[MODEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("External model outputs must be finite numeric values.")


def _reject_outcome_influence(frame: pd.DataFrame) -> None:
    if (
        "dataset_id" in frame
        and frame["dataset_id"].astype(str).str.contains("daic", case=False).any()
    ):
        raise ValueError("DAIC data are prohibited from external calibration and selection.")
    offending = sorted(
        column
        for column in frame.columns
        if any(token in column.lower() for token in OUTCOME_TOKENS)
    )
    if offending:
        raise ValueError(
            f"External calibration input contains prohibited clinical outcome columns: {offending}"
        )


def cross_validated_external_predictions(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    calibration_method: str = "dimensionwise_affine",
) -> pd.DataFrame:
    """Generate held-out calibrated text/audio and divergence predictions."""
    validate_paired_human_reference(frame)
    _validate_model_columns(frame)
    _reject_outcome_influence(frame)
    if folds < 2:
        raise ValueError("At least two external calibration folds are required.")
    working = attach_connected_leakage_components(frame).reset_index(drop=True)
    group_column = "leakage_component_id"
    groups = working[group_column].to_numpy()
    if len(np.unique(groups)) < folds:
        raise ValueError("External benchmark has fewer unique groups than requested folds.")

    splitter = GroupKFold(n_splits=folds)
    blocks: list[pd.DataFrame] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(working, groups=groups)):
        train = working.iloc[train_index]
        test = working.iloc[test_index]
        assert_group_split_is_disjoint(train, test, LEAKAGE_RELATION_COLUMNS)
        train_keys = composite_item_keys(train)
        test_keys = composite_item_keys(test)
        text_calibrator = _new_calibrator(calibration_method).fit(
            train[["model_text_valence", "model_text_arousal"]],
            train[["human_text_valence", "human_text_arousal"]],
            train_keys,
        )
        audio_calibrator = _new_calibrator(calibration_method).fit(
            train[["model_audio_valence", "model_audio_arousal"]],
            train[["human_audio_valence", "human_audio_arousal"]],
            train_keys,
        )
        text = text_calibrator.transform(
            test[["model_text_valence", "model_text_arousal"]], test_keys
        )
        audio = audio_calibrator.transform(
            test[["model_audio_valence", "model_audio_arousal"]], test_keys
        )
        block = test[
            [
                "dataset_id",
                "dataset_version",
                "item_id",
                group_column,
                "text_annotation_condition",
                "audio_annotation_condition",
                "scale_name",
                "scale_min",
                "scale_max",
                "human_text_valence",
                "human_text_arousal",
                "human_audio_valence",
                "human_audio_arousal",
            ]
        ].copy()
        block["fold"] = fold
        block["pred_text_valence"] = text[:, 0]
        block["pred_text_arousal"] = text[:, 1]
        block["pred_audio_valence"] = audio[:, 0]
        block["pred_audio_arousal"] = audio[:, 1]
        block["pred_delta_valence"] = text[:, 0] - audio[:, 0]
        block["pred_delta_arousal"] = text[:, 1] - audio[:, 1]
        block["pred_divergence_l2"] = np.hypot(
            block["pred_delta_valence"], block["pred_delta_arousal"]
        )
        blocks.append(block)
    output = pd.concat(blocks, ignore_index=True)
    if composite_item_keys(output).duplicated().any() or len(output) != len(working):
        raise RuntimeError("Held-out predictions must contain each composite item exactly once.")
    human = paired_human_differences(working)[
        [
            "dataset_id",
            "dataset_version",
            "item_id",
            "human_delta_valence",
            "human_delta_arousal",
            "human_divergence_l2",
        ]
    ]
    return output.merge(
        human,
        on=["dataset_id", "dataset_version", "item_id"],
        how="left",
        validate="one_to_one",
    )


def external_prediction_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    output = {
        name: continuous_metrics(predictions[observed], predictions[predicted])
        for name, (observed, predicted) in EXTERNAL_METRIC_PAIRS.items()
    }
    for estimand in ("delta_valence", "delta_arousal"):
        observed, predicted = EXTERNAL_METRIC_PAIRS[estimand]
        output[estimand]["signed_direction_accuracy"] = float(
            np.mean(np.sign(predictions[observed]) == np.sign(predictions[predicted]))
        )
    return output


def grouped_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    group_column: str = "leakage_component_id",
    replicates: int = 1000,
    seed: int = 20260719,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bootstrap whole source groups and return percentile intervals."""
    if replicates < 100:
        raise ValueError("Use at least 100 grouped bootstrap replicates.")
    groups = predictions[group_column].drop_duplicates().to_numpy()
    if len(groups) < 2:
        raise ValueError("Grouped bootstrap requires at least two groups.")
    rng = np.random.default_rng(seed)
    collected: dict[str, dict[str, list[float]]] = {}
    invalid: dict[str, dict[str, Counter[str]]] = {}
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat(
            [predictions.loc[predictions[group_column].eq(group)] for group in sampled],
            ignore_index=True,
        )
        for estimand, metric_values in external_prediction_metrics(sample).items():
            for metric, value in metric_values.items():
                if metric == "n":
                    continue
                if np.isfinite(value):
                    collected.setdefault(estimand, {}).setdefault(metric, []).append(value)
                else:
                    observed, predicted = EXTERNAL_METRIC_PAIRS[estimand]
                    reason = _invalid_metric_reason(sample[observed], sample[predicted], metric)
                    invalid.setdefault(estimand, {}).setdefault(metric, Counter())[reason] += 1
    intervals: dict[str, dict[str, dict[str, Any]]] = {}
    point_metrics = external_prediction_metrics(predictions)
    for estimand, metrics in point_metrics.items():
        intervals[estimand] = {}
        for metric in metrics:
            if metric == "n":
                continue
            intervals[estimand][metric] = _bootstrap_interval_record(
                collected.get(estimand, {}).get(metric, []),
                invalid.get(estimand, {}).get(metric, Counter()),
                replicates=replicates,
                lower_name="lower_95",
                upper_name="upper_95",
            )
    return intervals


def paired_group_bootstrap_model_comparison(
    predictions: pd.DataFrame,
    *,
    observed_column: str,
    first_prediction_column: str,
    second_prediction_column: str,
    group_column: str = "leakage_component_id",
    replicates: int = 1000,
    seed: int = 20260719,
) -> dict[str, dict[str, Any]]:
    """Compare models within each resampled source component, never via marginal CIs."""
    required = {
        observed_column,
        first_prediction_column,
        second_prediction_column,
        group_column,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Paired comparison is missing columns: {missing}")
    if replicates < 100:
        raise ValueError("Use at least 100 paired source-group bootstrap replicates.")
    groups = predictions[group_column].drop_duplicates().to_numpy()
    if len(groups) < 2:
        raise ValueError("Paired source-group bootstrap requires at least two groups.")

    metric_names = ("mae", "rmse", "ccc", "pearson", "spearman")

    def metric_values(sample: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
        first = continuous_metrics(sample[observed_column], sample[first_prediction_column])
        second = continuous_metrics(sample[observed_column], sample[second_prediction_column])
        return first, second

    point_first, point_second = metric_values(predictions)
    collected: dict[str, list[float]] = {name: [] for name in metric_names}
    invalid: dict[str, Counter[str]] = {name: Counter() for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat(
            [predictions.loc[predictions[group_column].eq(group)] for group in sampled],
            ignore_index=True,
        )
        first, second = metric_values(sample)
        for metric in metric_names:
            value = float(first[metric] - second[metric])
            if np.isfinite(value):
                collected[metric].append(value)
                continue
            if not np.isfinite(first[metric]):
                reason = "first_" + _invalid_metric_reason(
                    sample[observed_column], sample[first_prediction_column], metric
                )
            elif not np.isfinite(second[metric]):
                reason = "second_" + _invalid_metric_reason(
                    sample[observed_column], sample[second_prediction_column], metric
                )
            else:
                reason = "nonfinite_difference"
            invalid[metric][reason] += 1
    output: dict[str, dict[str, Any]] = {}
    for metric in metric_names:
        point = float(point_first[metric] - point_second[metric])
        output[metric] = {
            "difference_first_minus_second": point if np.isfinite(point) else None,
            **_bootstrap_interval_record(
                collected[metric],
                invalid[metric],
                replicates=replicates,
                lower_name="difference_lower_95",
                upper_name="difference_upper_95",
            ),
        }
    return output


@dataclass(frozen=True)
class NestedGroupedExternalResult:
    """One-row-per-item held-out predictions plus fold-local selection records."""

    predictions: pd.DataFrame
    sensitivity_predictions: pd.DataFrame
    primary_out_of_range: pd.DataFrame
    outer_selections: pd.DataFrame
    split_manifest: pd.DataFrame
    feasibility_report: dict[str, Any]


@dataclass(frozen=True)
class FinalExternalCandidateFreeze:
    """Final external candidate choice and calibration fitted after nested estimation."""

    selected_text_candidate: str
    candidate_objectives: pd.DataFrame
    calibration_artifact: dict[str, Any]
    selection_contract: dict[str, Any]
    feasibility_report: dict[str, Any]


def _model_columns_are_finite(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"External evaluation frame missing candidate columns: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("External candidate outputs must be finite numeric values.")


def _calibrated_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    text_columns: tuple[str, str],
    audio_columns: tuple[str, str],
    calibration_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    assert_group_split_is_disjoint(train, test, LEAKAGE_RELATION_COLUMNS)
    train_keys = composite_item_keys(train)
    test_keys = composite_item_keys(test)
    text = _new_calibrator(calibration_method).fit(
        train[list(text_columns)],
        train[["human_text_valence", "human_text_arousal"]],
        train_keys,
    )
    audio = _new_calibrator(calibration_method).fit(
        train[list(audio_columns)],
        train[["human_audio_valence", "human_audio_arousal"]],
        train_keys,
    )
    return (
        text.transform(test[list(text_columns)], test_keys),
        audio.transform(test[list(audio_columns)], test_keys),
    )


def _inner_text_objective(
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    text_columns: tuple[str, str],
    calibration_method: str,
) -> float:
    values: list[float] = []
    for inner_train, inner_test in splits:
        inner_fit = train.iloc[inner_train]
        inner_holdout = train.iloc[inner_test]
        assert_group_split_is_disjoint(inner_fit, inner_holdout, LEAKAGE_RELATION_COLUMNS)
        calibrator = _new_calibrator(calibration_method).fit(
            inner_fit[list(text_columns)],
            inner_fit[["human_text_valence", "human_text_arousal"]],
            composite_item_keys(inner_fit),
        )
        text = calibrator.transform(
            inner_holdout[list(text_columns)],
            composite_item_keys(inner_holdout),
        )
        values.extend(
            [
                float(np.sqrt(mean_squared_error(inner_holdout["human_text_valence"], text[:, 0]))),
                float(np.sqrt(mean_squared_error(inner_holdout["human_text_arousal"], text[:, 1]))),
            ]
        )
    return float(np.mean(values))


def _safe_inner_text_objective(
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    text_columns: tuple[str, str],
    calibration_method: str,
) -> float:
    try:
        return _inner_text_objective(
            train,
            splits,
            text_columns=text_columns,
            calibration_method=calibration_method,
        )
    except ValueError:
        return float("inf")


def connected_component_feasibility_report(
    frame: pd.DataFrame,
    *,
    proposed_outer_folds: int,
    proposed_inner_folds: int,
) -> dict[str, Any]:
    """Report mathematical group-fold feasibility without granting scientific approval."""
    validate_paired_human_reference(frame)
    if proposed_outer_folds < 2 or proposed_inner_folds < 2:
        raise ValueError("Proposed outer and inner folds must each be at least two.")
    working = attach_connected_leakage_components(frame).reset_index(drop=True)
    component_sizes = working.groupby("leakage_component_id", sort=True).size().to_numpy(dtype=int)
    item_count = int(len(working))
    component_count = int(len(component_sizes))
    largest_share = float(component_sizes.max() / item_count)
    kish_effective = float(item_count**2 / np.sum(component_sizes.astype(float) ** 2))
    outer_minimum_met = component_count >= proposed_outer_folds
    outer_partitions: list[dict[str, Any]] = []
    if outer_minimum_met:
        groups = working["leakage_component_id"].to_numpy()
        splitter = GroupKFold(n_splits=proposed_outer_folds)
        for fold, (train_index, test_index) in enumerate(splitter.split(working, groups=groups)):
            train_group_count = int(np.unique(groups[train_index]).size)
            test_group_count = int(np.unique(groups[test_index]).size)
            outer_partitions.append(
                {
                    "outer_fold": fold,
                    "training_item_count": int(len(train_index)),
                    "test_item_count": int(len(test_index)),
                    "training_component_count": train_group_count,
                    "test_component_count": test_group_count,
                    "requested_inner_folds": proposed_inner_folds,
                    "inner_mathematical_minimum_supported": (
                        train_group_count >= proposed_inner_folds
                    ),
                }
            )
    all_inner_supported = bool(outer_partitions) and all(
        row["inner_mathematical_minimum_supported"] for row in outer_partitions
    )
    quantiles: np.ndarray = np.quantile(component_sizes, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "item_count": item_count,
        "component_count": component_count,
        "component_size_distribution": {
            "minimum": int(quantiles[0]),
            "quartile_1": float(quantiles[1]),
            "median": float(quantiles[2]),
            "quartile_3": float(quantiles[3]),
            "maximum": int(quantiles[4]),
            "mean": float(component_sizes.mean()),
        },
        "largest_component_share": largest_share,
        "kish_effective_component_count": kish_effective,
        "proposed_outer_folds": proposed_outer_folds,
        "proposed_inner_folds": proposed_inner_folds,
        "outer_mathematical_minimum_supported": outer_minimum_met,
        "all_outer_training_partitions_support_inner_folds": all_inner_supported,
        "outer_partitions": outer_partitions,
        "scientific_approval_granted": False,
        "automatic_approval_prohibited": True,
        "interpretation": (
            "Meeting the mathematical group-count minimum does not establish stable or "
            "scientifically adequate grouped evaluation; component concentration and fold "
            "composition require explicit researcher review."
        ),
    }


def _seeded_grouped_splits(
    frame: pd.DataFrame,
    *,
    folds: int,
    seeds: tuple[int, ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["leakage_component_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError("Fewer connected components than requested grouped folds.")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in seeds:
        shuffled = unique_groups.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        group_codes = {group: index for index, group in enumerate(shuffled)}
        seeded_groups = np.asarray([group_codes[group] for group in groups])
        splits.extend(GroupKFold(n_splits=folds).split(frame, groups=seeded_groups))
    return splits


def primary_out_of_range_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count unclipped primary predictions outside the homogeneous declared scale."""
    rows: list[dict[str, Any]] = []
    dimensions = {
        "text_valence": "pred_text_valence",
        "text_arousal": "pred_text_arousal",
        "audio_valence": "pred_audio_valence",
        "audio_arousal": "pred_audio_arousal",
    }
    lower = predictions["scale_min"].to_numpy(float)
    upper = predictions["scale_max"].to_numpy(float)
    for dimension, column in dimensions.items():
        values = predictions[column].to_numpy(float)
        for direction, mask in (
            ("below_declared_minimum", values < lower),
            ("above_declared_maximum", values > upper),
        ):
            count = int(mask.sum())
            rows.append(
                {
                    "dimension": dimension,
                    "direction": direction,
                    "out_of_range_count": count,
                    "out_of_range_rate": float(count / len(values)),
                    "item_count": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def nested_grouped_external_evaluation(
    frame: pd.DataFrame,
    *,
    text_candidates: Mapping[str, tuple[str, str]],
    audio_columns: tuple[str, str] = ("model_audio_valence", "model_audio_arousal"),
    audio_estimator_id: str,
    outer_folds: int = 5,
    inner_folds: int = 4,
    primary_calibration_method: str = PRIMARY_CALIBRATION_METHOD,
    sensitivity_calibration_methods: tuple[str, ...] = (
        "cross_dimensional_affine_sensitivity",
        "robust_location_scale",
        CLIPPED_TO_SCALE_SENSITIVITY,
    ),
    development_seed_set: tuple[int, ...] = (20260719,),
    scientific_feasibility_approved: bool = False,
) -> NestedGroupedExternalResult:
    """Estimate the full primary selection procedure and report sensitivities in parallel."""
    validate_paired_human_reference(frame)
    _reject_outcome_influence(frame)
    if not text_candidates or not audio_estimator_id.strip():
        raise ValueError("Text candidates and a fixed audio estimator ID are required.")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("Nested external evaluation requires at least two folds per level.")
    if not development_seed_set or len(set(development_seed_set)) != len(development_seed_set):
        raise ValueError("Development seed set must be non-empty and unique.")
    if primary_calibration_method != PRIMARY_CALIBRATION_METHOD:
        raise ValueError("The locked primary calibration must remain dimensionwise_affine.")
    if primary_calibration_method in sensitivity_calibration_methods:
        raise ValueError("Primary and sensitivity calibration methods must be distinct.")
    if not sensitivity_calibration_methods or len(set(sensitivity_calibration_methods)) != len(
        sensitivity_calibration_methods
    ):
        raise ValueError("Prespecified sensitivity methods must be non-empty and unique.")
    if CLIPPED_TO_SCALE_SENSITIVITY not in sensitivity_calibration_methods:
        raise ValueError("Clipped-to-declared-scale sensitivity must be prespecified.")
    for method in sensitivity_calibration_methods:
        if method != CLIPPED_TO_SCALE_SENSITIVITY:
            _new_calibrator(method)
    candidate_columns = [column for pair in text_candidates.values() for column in pair]
    _model_columns_are_finite(frame, [*candidate_columns, *audio_columns])

    working = attach_connected_leakage_components(frame).reset_index(drop=True)
    feasibility = connected_component_feasibility_report(
        working,
        proposed_outer_folds=outer_folds,
        proposed_inner_folds=inner_folds,
    )
    if not feasibility["outer_mathematical_minimum_supported"]:
        raise ValueError("Fewer connected leakage components than requested outer folds.")
    if not feasibility["all_outer_training_partitions_support_inner_folds"]:
        raise ValueError("An outer-training partition cannot support the requested inner folds.")
    if not scientific_feasibility_approved:
        raise ValueError(
            "Nested evaluation requires explicit scientific feasibility approval; "
            "mathematical group-count minima are insufficient."
        )
    feasibility = {
        **feasibility,
        "scientific_approval_granted": True,
        "scientific_feasibility_approval_supplied": True,
    }
    groups = working["leakage_component_id"].to_numpy()
    outer = GroupKFold(n_splits=outer_folds)
    prediction_blocks: list[pd.DataFrame] = []
    sensitivity_blocks: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for outer_fold, (outer_train_index, outer_test_index) in enumerate(
        outer.split(working, groups=groups)
    ):
        outer_train = working.iloc[outer_train_index].reset_index(drop=True)
        outer_test = working.iloc[outer_test_index].reset_index(drop=True)
        assert_group_split_is_disjoint(outer_train, outer_test, LEAKAGE_RELATION_COLUMNS)
        inner_splits = _seeded_grouped_splits(
            outer_train,
            folds=inner_folds,
            seeds=development_seed_set,
        )

        primary_trials = [
            (
                _safe_inner_text_objective(
                    outer_train,
                    inner_splits,
                    text_columns=columns,
                    calibration_method=primary_calibration_method,
                ),
                name,
                columns,
            )
            for name, columns in text_candidates.items()
        ]
        primary_objective, selected_candidate, selected_columns = min(
            primary_trials, key=lambda value: (value[0], value[1])
        )
        if not np.isfinite(primary_objective):
            raise ValueError("Every primary text-candidate objective is unavailable.")

        primary_text, primary_audio = _calibrated_fold(
            outer_train,
            outer_test,
            text_columns=selected_columns,
            audio_columns=audio_columns,
            calibration_method=primary_calibration_method,
        )
        block = outer_test[
            [
                "dataset_id",
                "dataset_version",
                "item_id",
                "leakage_component_id",
                "text_annotation_condition",
                "audio_annotation_condition",
                "scale_name",
                "scale_min",
                "scale_max",
                "human_text_valence",
                "human_text_arousal",
                "human_audio_valence",
                "human_audio_arousal",
            ]
        ].copy()
        block["outer_fold"] = outer_fold
        block["selected_text_candidate"] = selected_candidate
        block["fixed_audio_estimator_id"] = audio_estimator_id
        block["primary_calibration_method"] = primary_calibration_method
        block["primary_predictions_clipped"] = False
        block["pred_text_valence"] = primary_text[:, 0]
        block["pred_text_arousal"] = primary_text[:, 1]
        block["pred_audio_valence"] = primary_audio[:, 0]
        block["pred_audio_arousal"] = primary_audio[:, 1]
        block["pred_delta_valence"] = primary_text[:, 0] - primary_audio[:, 0]
        block["pred_delta_arousal"] = primary_text[:, 1] - primary_audio[:, 1]
        block["pred_divergence_l2"] = np.hypot(
            block["pred_delta_valence"], block["pred_delta_arousal"]
        )

        for sensitivity_method in sensitivity_calibration_methods:
            if sensitivity_method == CLIPPED_TO_SCALE_SENSITIVITY:
                lower = outer_test["scale_min"].to_numpy(float)[:, None]
                upper = outer_test["scale_max"].to_numpy(float)[:, None]
                sensitivity_text = np.clip(primary_text, lower, upper)
                sensitivity_audio = np.clip(primary_audio, lower, upper)
            else:
                sensitivity_text, sensitivity_audio = _calibrated_fold(
                    outer_train,
                    outer_test,
                    text_columns=selected_columns,
                    audio_columns=audio_columns,
                    calibration_method=sensitivity_method,
                )
            sensitivity = block[
                [
                    "dataset_id",
                    "dataset_version",
                    "item_id",
                    "leakage_component_id",
                    "outer_fold",
                    "scale_name",
                    "scale_min",
                    "scale_max",
                    "text_annotation_condition",
                    "audio_annotation_condition",
                    "human_text_valence",
                    "human_text_arousal",
                    "human_audio_valence",
                    "human_audio_arousal",
                ]
            ].copy()
            sensitivity["selected_text_candidate"] = selected_candidate
            sensitivity["fixed_audio_estimator_id"] = audio_estimator_id
            sensitivity["sensitivity_method"] = sensitivity_method
            sensitivity["sensitivity_selected_as_best"] = False
            sensitivity["pred_text_valence"] = sensitivity_text[:, 0]
            sensitivity["pred_text_arousal"] = sensitivity_text[:, 1]
            sensitivity["pred_audio_valence"] = sensitivity_audio[:, 0]
            sensitivity["pred_audio_arousal"] = sensitivity_audio[:, 1]
            sensitivity["pred_delta_valence"] = sensitivity_text[:, 0] - sensitivity_audio[:, 0]
            sensitivity["pred_delta_arousal"] = sensitivity_text[:, 1] - sensitivity_audio[:, 1]
            sensitivity["pred_divergence_l2"] = np.hypot(
                sensitivity["pred_delta_valence"], sensitivity["pred_delta_arousal"]
            )
            sensitivity_blocks.append(sensitivity)
        prediction_blocks.append(block)
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "primary_text_candidate": selected_candidate,
                "primary_calibration_method": primary_calibration_method,
                "primary_inner_mean_text_va_rmse": primary_objective,
                "primary_selection_objective": PRIMARY_SELECTION_OBJECTIVE,
                "primary_selection_tie_break": PRIMARY_SELECTION_TIE_BREAK,
                "predefined_candidate_set": json.dumps(sorted(text_candidates)),
                "sensitivity_text_candidate_fixed": selected_candidate,
                "reported_sensitivity_methods": json.dumps(list(sensitivity_calibration_methods)),
                "sensitivity_selection_performed": False,
                "fixed_audio_estimator_id": audio_estimator_id,
                "development_seed_set": json.dumps(list(development_seed_set)),
                "inner_split_reused_across_all_candidates": True,
            }
        )
        for index in outer_test_index:
            split_rows.append(
                {
                    "composite_item_key": composite_item_keys(working.iloc[[index]]).iloc[0],
                    "outer_fold": outer_fold,
                    "leakage_component_id": working.iloc[index]["leakage_component_id"],
                }
            )

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    human = paired_human_differences(working)[
        [
            "dataset_id",
            "dataset_version",
            "item_id",
            "human_delta_valence",
            "human_delta_arousal",
            "human_divergence_l2",
        ]
    ]
    predictions = predictions.merge(
        human,
        on=["dataset_id", "dataset_version", "item_id"],
        how="left",
        validate="one_to_one",
    )
    sensitivity_predictions = pd.concat(sensitivity_blocks, ignore_index=True).merge(
        human,
        on=["dataset_id", "dataset_version", "item_id"],
        how="left",
        validate="many_to_one",
    )
    keys = composite_item_keys(predictions)
    if len(predictions) != len(working) or keys.duplicated().any():
        raise RuntimeError("Outer held-out predictions were not produced exactly once per item.")
    sensitivity_keys = (
        composite_item_keys(sensitivity_predictions)
        + "::"
        + sensitivity_predictions["sensitivity_method"].astype(str)
    )
    expected_sensitivity_rows = len(working) * len(sensitivity_calibration_methods)
    if (
        len(sensitivity_predictions) != expected_sensitivity_rows
        or sensitivity_keys.duplicated().any()
    ):
        raise RuntimeError(
            "Each prespecified sensitivity must contain exactly one prediction per item."
        )
    ordered_predictions = predictions.sort_values(
        ["dataset_id", "dataset_version", "item_id"]
    ).reset_index(drop=True)
    return NestedGroupedExternalResult(
        predictions=ordered_predictions,
        sensitivity_predictions=sensitivity_predictions.sort_values(
            ["sensitivity_method", "dataset_id", "dataset_version", "item_id"]
        ).reset_index(drop=True),
        primary_out_of_range=primary_out_of_range_report(ordered_predictions),
        outer_selections=pd.DataFrame(selection_rows),
        split_manifest=pd.DataFrame(split_rows)
        .sort_values("composite_item_key")
        .reset_index(drop=True),
        feasibility_report=feasibility,
    )


def freeze_final_external_candidate(
    frame: pd.DataFrame,
    *,
    nested_result: NestedGroupedExternalResult,
    text_candidates: Mapping[str, tuple[str, str]],
    text_candidate_metadata: Mapping[str, Mapping[str, str]],
    calibration_metadata: Mapping[str, str],
    audio_columns: tuple[str, str] = ("model_audio_valence", "model_audio_arousal"),
    audio_estimator_id: str,
    inner_folds: int = 4,
    primary_calibration_method: str = PRIMARY_CALIBRATION_METHOD,
    development_seed_set: tuple[int, ...] = (20260719,),
    scientific_feasibility_approved: bool = False,
) -> FinalExternalCandidateFreeze:
    """Select the final text candidate on the complete external benchmark, then calibrate it.

    Nested outer folds estimate the locked selection procedure. This separate finalization
    repeats only the predefined grouped inner-style selection rule on the complete external
    benchmark; it never uses outer-test metrics, DAIC data, or clinical outcomes.
    """
    validate_paired_human_reference(frame)
    _reject_outcome_influence(frame)
    if primary_calibration_method != PRIMARY_CALIBRATION_METHOD:
        raise ValueError("Final candidate freeze requires dimensionwise_affine calibration.")
    if not text_candidates or set(text_candidates) != set(text_candidate_metadata):
        raise ValueError("Candidate columns and candidate metadata must have identical names.")
    if not audio_estimator_id.strip():
        raise ValueError("The fixed audio estimator ID is required.")
    if inner_folds < 2:
        raise ValueError("Final candidate selection requires at least two grouped folds.")
    if not development_seed_set or len(set(development_seed_set)) != len(development_seed_set):
        raise ValueError("Development seed set must be non-empty and unique.")
    required_candidate_metadata = {"model_id", "model_revision"}
    for name, candidate_metadata in text_candidate_metadata.items():
        if any(
            not str(candidate_metadata.get(field, "")).strip()
            for field in required_candidate_metadata
        ):
            raise ValueError(f"Text candidate metadata is incomplete for {name}.")

    candidate_columns = [column for pair in text_candidates.values() for column in pair]
    _model_columns_are_finite(frame, [*candidate_columns, *audio_columns])
    working = attach_connected_leakage_components(frame).reset_index(drop=True)
    predicted_keys = composite_item_keys(nested_result.predictions)
    working_keys = composite_item_keys(working)
    if (
        predicted_keys.duplicated().any()
        or len(predicted_keys) != len(working_keys)
        or set(predicted_keys) != set(working_keys)
    ):
        raise ValueError(
            "Nested outer predictions must cover every finalization item exactly once."
        )

    selections = nested_result.outer_selections
    required_selection_columns = {
        "primary_calibration_method",
        "primary_selection_objective",
        "primary_selection_tie_break",
        "predefined_candidate_set",
        "fixed_audio_estimator_id",
        "development_seed_set",
        "sensitivity_selection_performed",
    }
    missing_selection_columns = sorted(required_selection_columns - set(selections))
    if missing_selection_columns or selections.empty:
        raise ValueError(
            "Nested selection contract is incomplete: "
            f"{missing_selection_columns or 'no outer selections'}"
        )
    expected_candidates = json.dumps(sorted(text_candidates))
    expected_seeds = json.dumps(list(development_seed_set))
    if not selections["predefined_candidate_set"].eq(expected_candidates).all():
        raise ValueError("Final candidate set differs from the nested-evaluation candidate set.")
    if not selections["development_seed_set"].eq(expected_seeds).all():
        raise ValueError("Final seed set differs from the nested-evaluation seed set.")
    if not selections["primary_selection_objective"].eq(PRIMARY_SELECTION_OBJECTIVE).all():
        raise ValueError("Final objective differs from the nested-evaluation objective.")
    if not selections["primary_selection_tie_break"].eq(PRIMARY_SELECTION_TIE_BREAK).all():
        raise ValueError("Final tie-breaking differs from the nested-evaluation tie-breaking.")
    if not selections["primary_calibration_method"].eq(primary_calibration_method).all():
        raise ValueError("Final calibration differs from the nested-evaluation primary method.")
    if not selections["fixed_audio_estimator_id"].eq(audio_estimator_id).all():
        raise ValueError("Final audio estimator differs from the fixed nested estimator.")
    if selections["sensitivity_selection_performed"].astype(bool).any():
        raise ValueError("Sensitivity results must never participate in candidate selection.")

    outer_folds = int(selections["outer_fold"].nunique())
    feasibility = connected_component_feasibility_report(
        working,
        proposed_outer_folds=outer_folds,
        proposed_inner_folds=inner_folds,
    )
    if not feasibility["all_outer_training_partitions_support_inner_folds"]:
        raise ValueError("The requested grouped evaluation is not mathematically feasible.")
    if not scientific_feasibility_approved:
        raise ValueError(
            "Final candidate freeze requires explicit scientific feasibility approval."
        )
    feasibility = {
        **feasibility,
        "scientific_approval_granted": True,
        "scientific_feasibility_approval_supplied": True,
    }

    splits = _seeded_grouped_splits(
        working,
        folds=inner_folds,
        seeds=development_seed_set,
    )
    trials = [
        (
            _safe_inner_text_objective(
                working,
                splits,
                text_columns=columns,
                calibration_method=primary_calibration_method,
            ),
            name,
            columns,
        )
        for name, columns in text_candidates.items()
    ]
    objective, selected_candidate, selected_columns = min(
        trials, key=lambda value: (value[0], value[1])
    )
    if not np.isfinite(objective):
        raise ValueError("Every final text-candidate objective is unavailable.")
    candidate_objectives = pd.DataFrame(
        [
            {
                "text_candidate": name,
                "primary_inner_mean_text_va_rmse": trial_objective,
                "selected": name == selected_candidate,
            }
            for trial_objective, name, _columns in sorted(trials, key=lambda value: value[1])
        ]
    )

    finalized = working.copy()
    finalized["model_text_valence"] = finalized[selected_columns[0]]
    finalized["model_text_arousal"] = finalized[selected_columns[1]]
    finalized["model_audio_valence"] = finalized[audio_columns[0]]
    finalized["model_audio_arousal"] = finalized[audio_columns[1]]
    candidate_metadata = text_candidate_metadata[selected_candidate]
    artifact_metadata = {
        **dict(calibration_metadata),
        "text_model_id": candidate_metadata["model_id"],
        "text_model_revision": candidate_metadata["model_revision"],
    }
    frame_dataset_id = str(working["dataset_id"].iloc[0])
    frame_dataset_version = str(working["dataset_version"].iloc[0])
    if (
        artifact_metadata.get("dataset_id") != frame_dataset_id
        or artifact_metadata.get("dataset_version") != frame_dataset_version
    ):
        raise ValueError("Calibration metadata dataset identity must match the external benchmark.")
    if artifact_metadata.get("audio_model_id") != audio_estimator_id:
        raise ValueError("Calibration metadata must identify the fixed audio estimator.")
    calibration_artifact = fit_frozen_external_calibration(
        finalized,
        metadata=artifact_metadata,
        calibration_method=primary_calibration_method,
        configuration={
            "final_candidate_selected_before_calibration_fit": True,
            "selection_objective": PRIMARY_SELECTION_OBJECTIVE,
            "selection_tie_break": PRIMARY_SELECTION_TIE_BREAK,
            "development_seed_set": list(development_seed_set),
            "predefined_candidate_set": sorted(text_candidates),
            "outer_test_metrics_used_for_final_selection": False,
            "sensitivity_results_used_for_final_selection": False,
            "predictions_clipped": False,
        },
    )
    selection_contract = {
        "nested_outer_folds_estimate_complete_selection_procedure": True,
        "final_selection_scope": "complete_external_benchmark_grouped_inner_style_evaluation",
        "predefined_candidate_set": sorted(text_candidates),
        "development_seed_set": list(development_seed_set),
        "objective": PRIMARY_SELECTION_OBJECTIVE,
        "tie_break": PRIMARY_SELECTION_TIE_BREAK,
        "primary_calibration_method": primary_calibration_method,
        "fixed_audio_estimator_id": audio_estimator_id,
        "outer_test_metrics_used": False,
        "sensitivity_results_used": False,
        "daic_or_phq_outcomes_used": False,
        "final_calibration_fit_after_candidate_selection": True,
    }
    return FinalExternalCandidateFreeze(
        selected_text_candidate=selected_candidate,
        candidate_objectives=candidate_objectives,
        calibration_artifact=calibration_artifact,
        selection_contract=selection_contract,
        feasibility_report=feasibility,
    )


def fit_frozen_external_calibration(
    frame: pd.DataFrame,
    *,
    metadata: Mapping[str, str],
    calibration_method: str = "dimensionwise_affine",
    configuration: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    benchmark_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a releaseable calibration record using external labels only."""
    validate_paired_human_reference(frame)
    _validate_model_columns(frame)
    _reject_outcome_influence(frame)
    required_metadata = {
        "dataset_id",
        "dataset_version",
        "split_hash",
        "text_model_id",
        "text_model_revision",
        "audio_model_id",
        "audio_model_revision",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing or any(not str(metadata.get(key, "")).strip() for key in required_metadata):
        raise ValueError(f"Frozen calibration metadata is incomplete: {missing or 'blank values'}")
    keys = composite_item_keys(frame)
    text = _new_calibrator(calibration_method).fit(
        frame[["model_text_valence", "model_text_arousal"]],
        frame[["human_text_valence", "human_text_arousal"]],
        keys,
    )
    audio = _new_calibrator(calibration_method).fit(
        frame[["model_audio_valence", "model_audio_arousal"]],
        frame[["human_audio_valence", "human_audio_arousal"]],
        keys,
    )
    affine_types = (DimensionwiseAffineCalibrator, CrossDimensionalAffineCalibrator)
    if isinstance(text, affine_types) and isinstance(audio, affine_types):
        assert text.coefficients_ is not None
        assert audio.coefficients_ is not None
        text_parameters: dict[str, Any] = {"coefficients": text.coefficients_.tolist()}
        audio_parameters: dict[str, Any] = {"coefficients": audio.coefficients_.tolist()}
    else:
        assert isinstance(text, RobustLocationScaleCalibrator)
        assert isinstance(audio, RobustLocationScaleCalibrator)
        assert text.model_median_ is not None and text.model_iqr_ is not None
        assert text.human_median_ is not None and text.human_iqr_ is not None
        assert audio.model_median_ is not None and audio.model_iqr_ is not None
        assert audio.human_median_ is not None and audio.human_iqr_ is not None
        text_parameters = {
            "model_median": text.model_median_.tolist(),
            "model_iqr": text.model_iqr_.tolist(),
            "human_median": text.human_median_.tolist(),
            "human_iqr": text.human_iqr_.tolist(),
        }
        audio_parameters = {
            "model_median": audio.model_median_.tolist(),
            "model_iqr": audio.model_iqr_.tolist(),
            "human_median": audio.human_median_.tolist(),
            "human_iqr": audio.human_iqr_.tolist(),
        }
    default_configuration: dict[str, Any] = {
        "calibration_method": calibration_method,
        "dimensions": ["valence", "arousal"],
        "primary_dimension_mapping": {
            "human_valence_from": ["model_valence"],
            "human_arousal_from": ["model_arousal"],
        }
        if calibration_method in {"dimensionwise_affine", "affine"}
        else None,
        "fit_scope": "external_training_data_only",
    }
    default_configuration.update(dict(configuration or {}))
    default_provenance: dict[str, Any] = {
        "dataset_id": metadata["dataset_id"],
        "dataset_version": metadata["dataset_version"],
        "annotation_conditions": ["text_only", "audio_only"],
        "available_group_columns": sorted(
            column
            for column in (
                "provenance_group_id",
                "source_video_id",
                "speaker_id",
                "creator_id",
            )
            if column in frame.columns
        ),
    }
    default_provenance.update(dict(provenance or {}))
    artifact: dict[str, Any] = {
        "schema_version": "1.0",
        "calibration_source": "external_human_modality_annotations_only",
        "phq_influence_prohibited": True,
        "metadata": dict(metadata),
        "configuration": default_configuration,
        "provenance": default_provenance,
        "benchmark_metrics": dict(benchmark_metrics or {}),
        "n_items": int(len(frame)),
        "text_calibration": text_parameters,
        "audio_calibration": audio_parameters,
    }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    artifact["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return artifact


def write_frozen_calibration_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Write an already-built non-participant calibration artifact."""
    Path(path).write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
