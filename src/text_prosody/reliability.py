"""Human-reference reliability, rater bootstrap, and noise-ceiling interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .contracts import ContractError
from .external_benchmark import composite_item_keys, validate_paired_human_reference

RAW_RATING_COLUMNS = {
    "dataset_id",
    "dataset_version",
    "item_id",
    "annotation_condition",
    "dimension",
    "rater_id",
    "rating",
    "scale_name",
    "scale_min",
    "scale_max",
}

REFERENCE_COLUMN_BY_STRATUM = {
    ("text_only", "valence"): "human_text_valence",
    ("text_only", "arousal"): "human_text_arousal",
    ("audio_only", "valence"): "human_audio_valence",
    ("audio_only", "arousal"): "human_audio_arousal",
}


def validate_raw_human_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate long-form raw ratings without assuming an upstream dataset schema."""
    if frame.columns.duplicated().any():
        raise ContractError("Raw human ratings contain duplicate column metadata.")
    missing = sorted(RAW_RATING_COLUMNS - set(frame.columns))
    if missing:
        raise ContractError(f"Raw human ratings are missing columns: {missing}")
    string_columns = [
        "dataset_id",
        "dataset_version",
        "item_id",
        "annotation_condition",
        "dimension",
        "rater_id",
        "scale_name",
    ]
    if (
        frame[string_columns].isna().any().any()
        or frame[string_columns]
        .astype(str)
        .apply(lambda value: value.str.strip().eq(""))
        .any()
        .any()
    ):
        raise ContractError("Raw rating keys, conditions, raters, and scales cannot be blank.")
    if not set(frame["annotation_condition"]).issubset({"text_only", "audio_only"}):
        raise ContractError("Raw ratings require explicit text_only or audio_only conditions.")
    if not set(frame["dimension"]).issubset({"valence", "arousal"}):
        raise ContractError("Raw ratings support only valence and arousal dimensions.")
    if frame["scale_name"].astype(str).str.strip().nunique() != 1:
        raise ContractError("Raw ratings must use one homogeneous scale_name.")
    lower = pd.to_numeric(frame["scale_min"], errors="coerce")
    upper = pd.to_numeric(frame["scale_max"], errors="coerce")
    ratings = pd.to_numeric(frame["rating"], errors="coerce")
    if (
        lower.isna().any()
        or upper.isna().any()
        or ratings.isna().any()
        or not np.isfinite(ratings).all()
    ):
        raise ContractError("Raw rating values and scale bounds must be finite.")
    if lower.nunique() != 1 or upper.nunique() != 1 or (upper <= lower).any():
        raise ContractError("Raw ratings must use homogeneous valid scale bounds.")
    if ((ratings < lower) | (ratings > upper)).any():
        raise ContractError("Raw ratings fall outside the declared scale.")
    keyed = frame.assign(_composite_key=composite_item_keys(frame))
    duplicate_key = [
        "_composite_key",
        "annotation_condition",
        "dimension",
        "rater_id",
    ]
    if keyed.duplicated(duplicate_key).any():
        raise ContractError("Raw ratings contain duplicate item/condition/dimension/rater rows.")
    return frame


def _rating_matrix(frame: pd.DataFrame, condition: str, dimension: str) -> np.ndarray:
    selected = frame.loc[
        frame["annotation_condition"].eq(condition) & frame["dimension"].eq(dimension)
    ].copy()
    selected["_composite_key"] = composite_item_keys(selected)
    matrix = selected.pivot(
        index="_composite_key",
        columns="rater_id",
        values="rating",
    )
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ContractError("Reliability requires at least three items and two raters.")
    if matrix.isna().any().any():
        raise ContractError("ICC/split-half reliability requires a complete item-by-rater matrix.")
    return matrix.to_numpy(float)


def rating_design_assessment(frame: pd.DataFrame) -> dict[str, Any]:
    """Assess whether every requested stratum is a complete crossed item-by-rater design."""
    validate_raw_human_ratings(frame)
    strata: dict[str, Any] = {}
    supported = True
    for condition, dimension in REFERENCE_COLUMN_BY_STRATUM:
        selected = frame.loc[
            frame["annotation_condition"].eq(condition) & frame["dimension"].eq(dimension)
        ].copy()
        if selected.empty:
            n_items = 0
            n_raters = 0
            observed = 0
            expected = 0
            complete = False
        else:
            selected["_composite_key"] = composite_item_keys(selected)
            n_items = int(selected["_composite_key"].nunique())
            n_raters = int(selected["rater_id"].nunique())
            observed = int(len(selected))
            expected = n_items * n_raters
            complete = n_items >= 3 and n_raters >= 2 and observed == expected
        supported = supported and complete
        strata[f"{condition}:{dimension}"] = {
            "n_items": n_items,
            "n_raters": n_raters,
            "observed_ratings": observed,
            "expected_crossed_ratings": expected,
            "complete_crossed_matrix": complete,
        }
    return {
        "status": (
            "COMPLETE_CROSSED_ITEM_BY_RATER_DESIGN"
            if supported
            else "UNSUPPORTED_RATING_DESIGN_INCOMPLETE_OR_PARTIALLY_CROSSED"
        ),
        "supported": supported,
        "required_design": "complete_crossed_item_by_rater_matrix_within_each_condition_dimension",
        "strata": strata,
    }


def validate_raw_rating_linkage(
    aggregate_reference: pd.DataFrame,
    raw_ratings: pd.DataFrame,
    *,
    mean_tolerance: float = 1e-6,
) -> None:
    """Fail unless raw ratings exactly support the declared aggregate reference."""
    if not np.isfinite(mean_tolerance) or mean_tolerance < 0:
        raise ValueError("mean_tolerance must be finite and nonnegative.")
    aggregate = validate_paired_human_reference(aggregate_reference).copy()
    raw = validate_raw_human_ratings(raw_ratings).copy()

    aggregate["_composite_key"] = composite_item_keys(aggregate)
    raw["_composite_key"] = composite_item_keys(raw)
    aggregate_keys = set(aggregate["_composite_key"])
    if set(raw["_composite_key"]) != aggregate_keys:
        raise ContractError("Raw ratings and aggregate reference must have identical item keys.")
    if set(zip(raw["annotation_condition"], raw["dimension"], strict=False)) != set(
        REFERENCE_COLUMN_BY_STRATUM
    ):
        raise ContractError(
            "Raw ratings must contain exactly the required annotation conditions and dimensions."
        )

    aggregate_scale_name = aggregate["scale_name"].astype(str).str.strip().iloc[0]
    raw_scale_name = raw["scale_name"].astype(str).str.strip().iloc[0]
    aggregate_bounds = (
        float(aggregate["scale_min"].iloc[0]),
        float(aggregate["scale_max"].iloc[0]),
    )
    raw_bounds = (float(raw["scale_min"].iloc[0]), float(raw["scale_max"].iloc[0]))
    if raw_scale_name != aggregate_scale_name or raw_bounds != aggregate_bounds:
        raise ContractError(
            "Raw ratings and aggregate reference must use the same scale name and bounds."
        )

    aggregate_by_key = aggregate.set_index("_composite_key")
    for (condition, dimension), reference_column in REFERENCE_COLUMN_BY_STRATUM.items():
        selected = raw.loc[
            raw["annotation_condition"].eq(condition) & raw["dimension"].eq(dimension)
        ]
        means = selected.groupby("_composite_key", sort=True)["rating"].mean()
        if set(means.index) != aggregate_keys:
            raise ContractError(
                "Every aggregate item requires raw ratings for each condition and dimension."
            )
        aligned_reference = pd.to_numeric(
            aggregate_by_key.loc[means.index, reference_column], errors="raise"
        )
        absolute_difference = np.abs(means.to_numpy(float) - aligned_reference.to_numpy(float))
        if np.any(absolute_difference > mean_tolerance):
            raise ContractError(
                "Raw-rating means do not match aggregate references within the declared tolerance."
            )


def _icc_two_way_random_absolute(matrix: np.ndarray) -> tuple[float, float]:
    """Return ICC(2,1) and ICC(2,k) for a complete target-by-rater matrix."""
    n_items, n_raters = matrix.shape
    grand = float(matrix.mean())
    item_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    ms_items = n_raters * float(np.sum((item_means - grand) ** 2)) / (n_items - 1)
    ms_raters = n_items * float(np.sum((rater_means - grand) ** 2)) / (n_raters - 1)
    residual = matrix - item_means[:, None] - rater_means[None, :] + grand
    ms_error = float(np.sum(residual**2)) / ((n_items - 1) * (n_raters - 1))
    single_denominator = (
        ms_items + (n_raters - 1) * ms_error + n_raters * (ms_raters - ms_error) / n_items
    )
    average_denominator = ms_items + (ms_raters - ms_error) / n_items
    single = (
        float((ms_items - ms_error) / single_denominator)
        if single_denominator != 0
        else float("nan")
    )
    average = (
        float((ms_items - ms_error) / average_denominator)
        if average_denominator != 0
        else float("nan")
    )
    return single, average


def _split_half(matrix: np.ndarray) -> tuple[float, float]:
    first = matrix[:, ::2].mean(axis=1)
    second = matrix[:, 1::2].mean(axis=1)
    correlation = float(pearsonr(first, second).statistic)
    corrected = float(2 * correlation / (1 + correlation)) if correlation > -1 else float("nan")
    return correlation, corrected


def raw_rating_reliability_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute rater-level reliability and estimated correlation noise ceilings."""
    validate_raw_human_ratings(frame)
    design = rating_design_assessment(frame)
    if not design["supported"]:
        return {
            **design,
            "rater_level_reliability_available": False,
            "human_noise_ceiling_available": False,
            "reason": (
                "ICC(2,1), ICC(2,k), split-half reliability, and rater bootstrap are unsupported "
                "until the observed ratings form complete crossed item-by-rater matrices."
            ),
        }
    results: dict[str, Any] = {}
    for condition in ("text_only", "audio_only"):
        for dimension in ("valence", "arousal"):
            matrix = _rating_matrix(frame, condition, dimension)
            icc_single, icc_average = _icc_two_way_random_absolute(matrix)
            split_half, split_half_corrected = _split_half(matrix)
            results[f"{condition}:{dimension}"] = {
                "n_items": matrix.shape[0],
                "n_raters": matrix.shape[1],
                "icc_2_1_single_rater": icc_single,
                "icc_2_k_average_raters": icc_average,
                "split_half_pearson": split_half,
                "split_half_spearman_brown": split_half_corrected,
                "estimated_correlation_noise_ceiling": float(
                    np.sqrt(np.clip(icc_average, 0.0, 1.0))
                ),
                "noise_ceiling_interpretation": (
                    "ASSUMPTION_DEPENDENT_SQRT_ICC_2_K: interpretable as an estimated "
                    "correlation ceiling only under the classical attenuation assumptions that "
                    "the aggregate ICC applies to this reference and errors are independent."
                ),
            }
    return {
        "status": "RAW_RATER_RELIABILITY_AVAILABLE",
        "reliability_model": "two_way_random_absolute_agreement",
        "rating_design": design,
        "noise_ceiling_status": "ASSUMPTION_DEPENDENT_ESTIMATE_NOT_OBSERVED_UPPER_BOUND",
        "results": results,
    }


def rater_bootstrap_reliability(
    frame: pd.DataFrame,
    *,
    replicates: int = 1000,
    seed: int = 20260719,
) -> dict[str, Any]:
    """Bootstrap raters, explicitly separate from source-component resampling."""
    validate_raw_human_ratings(frame)
    if replicates < 100:
        raise ValueError("Use at least 100 rater bootstrap replicates.")
    design = rating_design_assessment(frame)
    if not design["supported"]:
        return {
            **design,
            "resampling_unit": "rater",
            "must_not_be_combined_with": "source_group_bootstrap",
            "rater_bootstrap_available": False,
        }
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    minimum_valid_replicates = max(100, int(np.ceil(0.9 * replicates)))
    for condition in ("text_only", "audio_only"):
        for dimension in ("valence", "arousal"):
            matrix = _rating_matrix(frame, condition, dimension)
            values: list[float] = []
            for _ in range(replicates):
                sampled = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
                _single, average = _icc_two_way_random_absolute(matrix[:, sampled])
                if np.isfinite(average):
                    values.append(average)
            interval_available = len(values) >= minimum_valid_replicates
            output[f"{condition}:{dimension}"] = {
                "icc_2_k_lower_95": (
                    float(np.quantile(values, 0.025)) if interval_available else None
                ),
                "icc_2_k_upper_95": (
                    float(np.quantile(values, 0.975)) if interval_available else None
                ),
                "attempted_replicates": replicates,
                "valid_replicates": len(values),
                "minimum_valid_replicates": minimum_valid_replicates,
                "interval_available": interval_available,
            }
    return {
        "status": "RATER_BOOTSTRAP",
        "resampling_unit": "rater",
        "must_not_be_combined_with": "source_group_bootstrap",
        "results": output,
    }


def human_reference_reliability_report(
    aggregate_reference: pd.DataFrame,
    *,
    raw_ratings: pd.DataFrame | None = None,
    aggregate_mean_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Report reliability when available, or an explicit aggregate-only fallback."""
    validate_paired_human_reference(aggregate_reference)
    if raw_ratings is None:
        return {
            "status": "AGGREGATE_RATINGS_ONLY",
            "rater_level_reliability_available": False,
            "rater_bootstrap_available": False,
            "human_noise_ceiling_available": False,
            "reason": (
                "Raw rater-level observations were not supplied; no reliability was inferred."
            ),
        }
    validate_raw_rating_linkage(
        aggregate_reference,
        raw_ratings,
        mean_tolerance=aggregate_mean_tolerance,
    )
    report = raw_rating_reliability_report(raw_ratings)
    available = report["status"] == "RAW_RATER_RELIABILITY_AVAILABLE"
    report["rater_level_reliability_available"] = available
    report["rater_bootstrap_available"] = available
    report["human_noise_ceiling_available"] = available
    report["aggregate_linkage_validated"] = True
    report["aggregate_mean_tolerance"] = aggregate_mean_tolerance
    return report
