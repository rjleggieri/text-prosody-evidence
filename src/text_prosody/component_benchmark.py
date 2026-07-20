"""Public component-benchmark utilities with split and outcome guards."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .text_affect import EMOBANK_RELEASE, TextAffectError, normalize_emobank_frame

PUBLIC_COMPONENT_STATUS = "BOUNDED_PUBLIC_TEXT_COMPONENT_EVIDENCE"
PROHIBITED_TOKENS = ("phq", "depress", "diagnos", "outcome")
TARGETS = ("valence", "arousal")
METRIC_NAMES = (
    "mae",
    "rmse",
    "pearson",
    "spearman",
    "ccc",
    "calibration_intercept",
    "calibration_slope",
)
MINIMUM_VALID_BOOTSTRAP_FRACTION = 0.90


class ComponentBenchmarkError(RuntimeError):
    """Public component benchmark failed a scientific integrity guard."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_identifier_hash(values: pd.Series | list[object]) -> str:
    normalized = sorted(str(value) for value in values)
    return hashlib.sha256(("\n".join(normalized) + "\n").encode()).hexdigest()


def documented_source_prefix(item_id: object) -> str:
    """Extract the source-document prefix documented by EmoBank span IDs."""
    value = str(item_id)
    match = re.fullmatch(r"(.+)_\d+_\d+", value)
    if match is None or not match.group(1):
        raise ComponentBenchmarkError(
            "EmoBank item ID does not match the documented <source>_<start>_<end> form."
        )
    return match.group(1)


def _assert_no_outcome_columns(frame: pd.DataFrame) -> None:
    offending = sorted(
        column
        for column in frame.columns
        if any(token in column.casefold() for token in PROHIBITED_TOKENS)
    )
    if offending:
        raise ComponentBenchmarkError(
            f"Public component input contains prohibited outcome columns: {offending}"
        )


def load_emobank_reader_release(
    corpus_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join official v1.4 text, reader ratings, published split, and source metadata."""
    root = Path(corpus_dir)
    paths = {
        "raw": root / "raw.csv",
        "reader": root / "reader.csv",
        "published_split": root / "emobank.csv",
        "metadata": root / "meta.tsv",
    }
    missing_files = sorted(name for name, path in paths.items() if not path.is_file())
    if missing_files:
        raise ComponentBenchmarkError(f"EmoBank release files are missing: {missing_files}")

    raw = pd.read_csv(paths["raw"])
    reader = pd.read_csv(paths["reader"])
    split = pd.read_csv(paths["published_split"], usecols=["id", "split"])
    metadata = pd.read_csv(paths["metadata"], sep="\t")
    for name, frame, required in (
        ("raw", raw, {"id", "text"}),
        ("reader", reader, {"id", "V", "A"}),
        ("published_split", split, {"id", "split"}),
        ("metadata", metadata, {"id", "document", "category"}),
    ):
        absent = sorted(required - set(frame.columns))
        if absent:
            raise ComponentBenchmarkError(f"EmoBank {name} columns are missing: {absent}")
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ComponentBenchmarkError(f"EmoBank {name} IDs must be present and unique.")
        _assert_no_outcome_columns(frame)

    joined = (
        raw[["id", "text"]]
        .merge(reader[["id", "V", "A"]], on="id", how="inner", validate="one_to_one")
        .merge(split, on="id", how="inner", validate="one_to_one")
        .merge(
            metadata[["id", "document", "category", "subcategory"]],
            on="id",
            how="inner",
            validate="one_to_one",
        )
    )
    missing_text = joined["text"].isna() | joined["text"].astype(str).str.strip().eq("")
    numeric_targets = joined[["V", "A"]].apply(pd.to_numeric, errors="coerce")
    missing_target = numeric_targets.isna().any(axis=1)
    missing_split = joined["split"].isna() | joined["split"].astype(str).str.strip().eq("")
    invalid_scale = (
        numeric_targets.lt(1.0).any(axis=1) | numeric_targets.gt(5.0).any(axis=1)
    ) & ~missing_target
    eligible = ~(missing_text | missing_target | missing_split | invalid_scale)
    analysis_input = joined.loc[eligible].copy()
    normalized = normalize_emobank_frame(
        analysis_input,
        column_map={
            "item_id": "id",
            "text": "text",
            "valence": "V",
            "arousal": "A",
            "split": "split",
        },
        dataset_release=EMOBANK_RELEASE,
        perspective="reader",
        scale_min=1.0,
        scale_max=5.0,
    )
    normalized = normalized.merge(
        analysis_input[["id", "document", "category", "subcategory"]].rename(
            columns={"id": "item_id", "document": "source_document"}
        ),
        on="item_id",
        how="left",
        validate="one_to_one",
    )
    if normalized["source_document"].isna().any():
        raise ComponentBenchmarkError("EmoBank source-document grouping is incomplete.")

    expected_splits = {"train", "dev", "test"}
    observed_splits = set(normalized["split"].astype(str))
    if observed_splits != expected_splits:
        raise ComponentBenchmarkError(
            f"Expected all official train/dev/test splits; observed {sorted(observed_splits)}."
        )
    source_sets = {
        split_name: set(group["source_document"].astype(str))
        for split_name, group in normalized.groupby("split", sort=True)
    }
    overlaps = {
        f"{left}_{right}": len(source_sets[left] & source_sets[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    audit: dict[str, Any] = {
        "status": PUBLIC_COMPONENT_STATUS,
        "source_rows": {
            "raw": len(raw),
            "reader": len(reader),
            "published_split": len(split),
            "metadata": len(metadata),
            "joined_before_validity_exclusions": len(joined),
            "analyzed": len(normalized),
        },
        "join_exclusions": {
            "raw_without_joined_reader_and_split": len(raw) - len(joined),
            "reader_without_published_split_and_text": len(reader) - len(joined),
            "published_split_without_reader_and_text": len(split) - len(joined),
            "metadata_without_reader_split_and_text": len(metadata) - len(joined),
        },
        "analysis_exclusions": {
            "missing_or_empty_text": int(missing_text.sum()),
            "missing_or_non_numeric_reader_target": int(missing_target.sum()),
            "missing_split": int(missing_split.sum()),
            "target_outside_declared_scale": int(invalid_scale.sum()),
            "total_unique_rows_excluded": int((~eligible).sum()),
        },
        "split_counts": {
            key: int(value)
            for key, value in normalized["split"].value_counts().sort_index().items()
        },
        "split_identifier_sha256": {
            split_name: stable_identifier_hash(group["item_id"])
            for split_name, group in normalized.groupby("split", sort=True)
        },
        "source_document_counts": {
            split_name: int(group["source_document"].nunique())
            for split_name, group in normalized.groupby("split", sort=True)
        },
        "source_document_cross_split_overlap_counts": overlaps,
        "missingness": {
            column: int(normalized[column].isna().sum())
            for column in ("item_id", "text", "valence", "arousal", "split", "source_document")
        },
        "duplicate_item_ids": int(normalized["item_id"].duplicated().sum()),
        "scale": {"minimum": 1.0, "maximum": 5.0},
        "file_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "grouping_method": "official_meta_tsv_document",
    }
    return normalized, audit


def concordance_correlation(y_true: object, prediction: object) -> float | None:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if len(y) < 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        return None
    denominator = y.var(ddof=0) + p.var(ddof=0) + (y.mean() - p.mean()) ** 2
    if denominator <= 0:
        return None
    covariance = float(np.mean((y - y.mean()) * (p - p.mean())))
    return float(2.0 * covariance / denominator)


def calibration_line(y_true: object, prediction: object) -> tuple[float | None, float | None]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if len(y) < 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        return None, None
    if np.isclose(p.var(ddof=0), 0.0):
        return None, None
    design = np.column_stack([np.ones(len(p)), p])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def continuous_metrics(y_true: object, prediction: object) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or len(y) < 2:
        raise ComponentBenchmarkError("Continuous metrics require aligned one-dimensional arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ComponentBenchmarkError("Continuous metrics require finite values.")
    pearson = (
        None if np.unique(y).size < 2 or np.unique(p).size < 2 else float(pearsonr(y, p).statistic)
    )
    spearman = (
        None if np.unique(y).size < 2 or np.unique(p).size < 2 else float(spearmanr(y, p).statistic)
    )
    intercept, slope = calibration_line(y, p)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "pearson": pearson,
        "spearman": spearman,
        "ccc": concordance_correlation(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def evaluate_two_target_predictions(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    _assert_no_outcome_columns(frame)
    values = np.asarray(prediction, dtype=float)
    if values.shape != (len(frame), 2) or not np.isfinite(values).all():
        raise ComponentBenchmarkError("Predictions must be a finite n-by-2 V/A array.")
    return {
        target: continuous_metrics(frame[target].to_numpy(float), values[:, index])
        for index, target in enumerate(TARGETS)
    }


def development_objective(metrics: Mapping[str, Mapping[str, float | int | None]]) -> float:
    values = [metrics[target]["rmse"] for target in TARGETS]
    if any(value is None for value in values):
        raise ComponentBenchmarkError("Development RMSE is unavailable.")
    return float(np.mean(np.asarray(values, dtype=float)))


def grouped_bootstrap_intervals(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    group_column: str = "source_document",
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20260719,
) -> dict[str, dict[str, dict[str, Any]]]:
    if replicates < 100 or not 0.5 < confidence_level < 1.0:
        raise ComponentBenchmarkError("Bootstrap settings are invalid.")
    if group_column not in frame:
        raise ComponentBenchmarkError(f"Bootstrap group column is unavailable: {group_column}")
    values = np.asarray(prediction, dtype=float)
    if values.shape != (len(frame), 2):
        raise ComponentBenchmarkError("Bootstrap predictions must be n-by-2.")
    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ComponentBenchmarkError("Grouped bootstrap requires at least two source groups.")
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    collected: dict[str, dict[str, list[float]]] = {
        target: {metric: [] for metric in METRIC_NAMES} for target in TARGETS
    }
    invalid: dict[str, dict[str, Counter[str]]] = {
        target: {metric: Counter() for metric in METRIC_NAMES} for target in TARGETS
    }
    for _ in range(replicates):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        for target_index, target in enumerate(TARGETS):
            metrics = continuous_metrics(
                frame.iloc[indices][target].to_numpy(float), values[indices, target_index]
            )
            for metric in METRIC_NAMES:
                metric_value = metrics[metric]
                if metric_value is not None and np.isfinite(float(metric_value)):
                    collected[target][metric].append(float(metric_value))
                    continue
                observed = frame.iloc[indices][target].to_numpy(float)
                predicted = values[indices, target_index]
                if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
                    reason = "nonfinite_input"
                elif (
                    metric
                    in {
                        "pearson",
                        "spearman",
                        "calibration_intercept",
                        "calibration_slope",
                    }
                    and np.unique(predicted).size < 2
                ):
                    reason = "constant_prediction"
                elif metric in {"pearson", "spearman"} and np.unique(observed).size < 2:
                    reason = "constant_observed"
                elif metric == "ccc":
                    reason = "zero_concordance_denominator"
                else:
                    reason = "nonfinite_metric"
                invalid[target][metric][reason] += 1
    alpha = (1.0 - confidence_level) / 2.0
    minimum_valid = int(np.ceil(MINIMUM_VALID_BOOTSTRAP_FRACTION * replicates))
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for target in TARGETS:
        output[target] = {}
        for metric in METRIC_NAMES:
            metric_values = collected[target][metric]
            available = len(metric_values) >= minimum_valid
            output[target][metric] = {
                "low": float(np.quantile(metric_values, alpha)) if available else None,
                "high": (float(np.quantile(metric_values, 1.0 - alpha)) if available else None),
                "attempted_replicates": replicates,
                "valid_replicates": len(metric_values),
                "invalid_replicates": replicates - len(metric_values),
                "invalid_counts_by_reason": dict(sorted(invalid[target][metric].items())),
                "minimum_valid_fraction": MINIMUM_VALID_BOOTSTRAP_FRACTION,
                "minimum_valid_replicates": minimum_valid,
                "interval_available": available,
            }
    return output


def verify_test_predictions_once(
    predictions_by_model: Mapping[str, np.ndarray],
    test_frame: pd.DataFrame,
) -> None:
    if not predictions_by_model:
        raise ComponentBenchmarkError("At least one locked model prediction is required.")
    for name, prediction in predictions_by_model.items():
        values = np.asarray(prediction, dtype=float)
        if values.shape != (len(test_frame), 2) or not np.isfinite(values).all():
            raise ComponentBenchmarkError(f"Model {name} produced invalid test predictions.")


def content_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


PredictionFunction = Callable[[list[str]], np.ndarray]


def predict_texts_in_batches(
    texts: pd.Series,
    predictor: PredictionFunction,
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ComponentBenchmarkError("Prediction batch size must be positive.")
    blocks = []
    values = texts.astype(str).tolist()
    for start in range(0, len(values), batch_size):
        block = np.asarray(predictor(values[start : start + batch_size]), dtype=float)
        if block.shape != (min(batch_size, len(values) - start), 2):
            raise TextAffectError("Text prediction block has the wrong shape.")
        blocks.append(block)
    return np.concatenate(blocks, axis=0)
