"""Leakage-safe, aggregate-only public text-component demonstration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .component_benchmark import (
    ComponentBenchmarkError,
    development_objective,
    evaluate_two_target_predictions,
    grouped_bootstrap_intervals,
)
from .external_evaluation import paired_group_bootstrap_model_comparison
from .text_affect import (
    EMOBANK_RELEASE,
    MeanTextVaBaseline,
    TfidfRidgeTextRegressor,
    normalize_emobank_frame,
)

DEMONSTRATION_STATUS = "PUBLIC_TEXT_COMPONENT_DEMONSTRATION"
DEMONSTRATION_PARTITIONS = frozenset({"train", "dev"})
PROHIBITED_TOKENS = ("phq", "depress", "diagnos", "outcome")


@dataclass(frozen=True)
class GroupedTextDemoResult:
    """Internal predictions plus privacy-safe aggregate reporting artifacts."""

    predictions: pd.DataFrame
    component_results: dict[str, Any]
    development_histories: dict[str, Any]
    split_manifest: dict[str, Any]


def _stable_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _assert_no_outcome_columns(frame: pd.DataFrame) -> None:
    offending = sorted(
        str(column)
        for column in frame
        if any(token in str(column).casefold() for token in PROHIBITED_TOKENS)
    )
    if offending:
        raise ComponentBenchmarkError(
            f"Public demonstration contains prohibited outcome columns: {offending}"
        )


def _read_release_table(
    path: Path,
    *,
    required: set[str],
    sep: str = ",",
) -> pd.DataFrame:
    if not path.is_file():
        raise ComponentBenchmarkError(f"Required EmoBank release file is missing: {path.name}")
    frame = pd.read_csv(path, sep=sep, usecols=lambda column: column in required)
    missing = sorted(required - set(frame))
    if missing:
        raise ComponentBenchmarkError(f"{path.name} is missing required columns: {missing}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ComponentBenchmarkError(f"{path.name} IDs must be present and unique.")
    _assert_no_outcome_columns(frame)
    return frame


def load_emobank_reader_development_pool(
    corpus_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load reader V/A for published train+dev only; never return test-partition rows."""
    root = Path(corpus_dir)
    split = _read_release_table(
        root / "emobank.csv",
        required={"id", "split"},
    )
    observed_partitions = set(split["split"].astype(str))
    if (
        not DEMONSTRATION_PARTITIONS.issubset(observed_partitions)
        or "test" not in observed_partitions
    ):
        raise ComponentBenchmarkError("Expected the official train/dev/test partition labels.")
    pool_split = split.loc[split["split"].isin(DEMONSTRATION_PARTITIONS)].copy()
    pool_ids = frozenset(pool_split["id"].astype(str))

    raw = _read_release_table(root / "raw.csv", required={"id", "text"})
    reader = _read_release_table(root / "reader.csv", required={"id", "V", "A"})
    metadata = _read_release_table(
        root / "meta.tsv",
        required={"id", "document", "category", "subcategory"},
        sep="\t",
    )
    raw = raw.loc[raw["id"].astype(str).isin(pool_ids)]
    reader = reader.loc[reader["id"].astype(str).isin(pool_ids)]
    metadata = metadata.loc[metadata["id"].astype(str).isin(pool_ids)]

    joined = (
        pool_split.merge(raw, on="id", how="inner", validate="one_to_one")
        .merge(reader, on="id", how="inner", validate="one_to_one")
        .merge(metadata, on="id", how="inner", validate="one_to_one")
    )
    if len(joined) != len(pool_split):
        raise ComponentBenchmarkError("Train/development rows do not join one-to-one across files.")
    if not set(joined["split"].astype(str)).issubset(DEMONSTRATION_PARTITIONS):
        raise ComponentBenchmarkError("The public demonstration must exclude the test partition.")

    missing_text = joined["text"].isna() | joined["text"].astype(str).str.strip().eq("")
    numeric_targets = joined[["V", "A"]].apply(pd.to_numeric, errors="coerce")
    missing_target = numeric_targets.isna().any(axis=1)
    invalid_scale = (
        numeric_targets.lt(1.0).any(axis=1) | numeric_targets.gt(5.0).any(axis=1)
    ) & ~missing_target
    missing_group = joined["document"].isna() | joined["document"].astype(str).str.strip().eq("")
    if missing_group.any():
        raise ComponentBenchmarkError("EmoBank source-document grouping is incomplete or blank.")
    eligible = ~(missing_text | missing_target | invalid_scale)
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
    _assert_no_outcome_columns(normalized)
    if set(normalized["split"].astype(str)) - DEMONSTRATION_PARTITIONS:
        raise ComponentBenchmarkError("A prohibited partition entered the demonstration pool.")

    group_sizes = normalized.groupby("source_document", sort=True).size().to_numpy(float)
    group_shares = group_sizes / group_sizes.sum()
    audit: dict[str, Any] = {
        "status": DEMONSTRATION_STATUS,
        "release": EMOBANK_RELEASE,
        "included_partitions": sorted(DEMONSTRATION_PARTITIONS),
        "excluded_partitions": ["test"],
        "test_partition_used_for_modeling_or_evaluation": False,
        "monolithic_source_file_limitation": (
            "Official release tables contain all partitions; the loader filters to train/dev "
            "before joining text, references, and metadata."
        ),
        "eligible_item_count": int(len(normalized)),
        "source_document_count": int(normalized["source_document"].nunique()),
        "partition_counts": {
            str(key): int(value)
            for key, value in normalized["split"].value_counts().sort_index().items()
        },
        "partition_source_document_counts": {
            str(key): int(group["source_document"].nunique())
            for key, group in normalized.groupby("split", sort=True)
        },
        "analysis_exclusions": {
            "missing_or_empty_text": int(missing_text.sum()),
            "missing_or_non_numeric_reader_target": int(missing_target.sum()),
            "target_outside_declared_scale": int(invalid_scale.sum()),
            "missing_source_document": 0,
            "total_unique_rows_excluded": int((~eligible).sum()),
        },
        "scale": {"name": "modified_SAM", "minimum": 1.0, "maximum": 5.0},
        "human_reference_fields": {
            "valence": "reader.csv:V",
            "arousal": "reader.csv:A",
            "annotation_perspective": "reader",
            "release_values_are_aggregates": True,
        },
        "grouping": {
            "field": "meta.tsv:document",
            "blank_count": int(normalized["source_document"].isna().sum()),
            "largest_group_item_count": int(group_sizes.max()),
            "largest_group_share": float(group_shares.max()),
            "kish_effective_group_count": float(1.0 / np.square(group_shares).sum()),
        },
    }
    return normalized.sort_values("item_id").reset_index(drop=True), audit


def _fit_ridge(
    train: pd.DataFrame,
    *,
    alpha: float,
    max_features: int,
) -> TfidfRidgeTextRegressor:
    return TfidfRidgeTextRegressor(alpha=alpha, max_features=max_features).fit(
        train.assign(split="train")
    )


def _inner_objective(
    train: pd.DataFrame,
    *,
    alpha: float,
    folds: int,
    max_features: int,
) -> tuple[float, list[dict[str, Any]]]:
    groups = train["source_document"].astype(str)
    splitter = GroupKFold(n_splits=folds)
    predictions: np.ndarray = np.empty((len(train), 2), dtype=float)
    assigned: np.ndarray = np.zeros(len(train), dtype=int)
    fold_records: list[dict[str, Any]] = []
    for fold, (fit_index, holdout_index) in enumerate(splitter.split(train, groups=groups)):
        fit = train.iloc[fit_index]
        holdout = train.iloc[holdout_index]
        if set(fit["source_document"]) & set(holdout["source_document"]):
            raise ComponentBenchmarkError("A source document crossed an inner fold.")
        model = _fit_ridge(fit, alpha=alpha, max_features=max_features)
        fold_prediction = model.predict_batch(holdout["text"].astype(str).tolist())
        predictions[holdout_index] = fold_prediction
        assigned[holdout_index] += 1
        fold_records.append(
            {
                "inner_fold": fold,
                "training_item_count": int(len(fit)),
                "training_group_count": int(fit["source_document"].nunique()),
                "holdout_item_count": int(len(holdout)),
                "holdout_group_count": int(holdout["source_document"].nunique()),
                "holdout_metrics": evaluate_two_target_predictions(holdout, fold_prediction),
            }
        )
    if not np.all(assigned == 1):
        raise ComponentBenchmarkError(
            "Inner grouped predictions must cover each item exactly once."
        )
    metrics = evaluate_two_target_predictions(train, predictions)
    return development_objective(metrics), fold_records


def _out_of_range_report(
    predictions: pd.DataFrame,
    *,
    model_columns: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_name, columns in model_columns.items():
        for dimension, column in zip(("valence", "arousal"), columns, strict=True):
            values = predictions[column].to_numpy(float)
            for direction, mask in (
                ("below_declared_minimum", values < 1.0),
                ("above_declared_maximum", values > 5.0),
            ):
                output.append(
                    {
                        "model": model_name,
                        "dimension": dimension,
                        "direction": direction,
                        "count": int(mask.sum()),
                        "rate": float(mask.mean()),
                    }
                )
    return output


def run_grouped_text_component_demo(
    frame: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
    alpha_candidates: tuple[float, ...] = (1.0, 10.0, 100.0),
    max_features: int = 20_000,
    bootstrap_replicates: int = 2_000,
    seed: int = 20260724,
) -> GroupedTextDemoResult:
    """Estimate a fixed text selection procedure using source-grouped outer folds."""
    _assert_no_outcome_columns(frame)
    required = {
        "item_id",
        "text",
        "valence",
        "arousal",
        "split",
        "dataset_release",
        "source_document",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ComponentBenchmarkError(f"Demonstration input is missing columns: {missing}")
    if frame["item_id"].isna().any() or frame["item_id"].duplicated().any():
        raise ComponentBenchmarkError("Demonstration item IDs must be present and unique.")
    if (
        frame["source_document"].isna().any()
        or frame["source_document"].astype(str).str.strip().eq("").any()
    ):
        raise ComponentBenchmarkError("Source-document groups must be complete and nonblank.")
    if set(frame["split"].astype(str)) - DEMONSTRATION_PARTITIONS:
        raise ComponentBenchmarkError("The demonstration input contains a prohibited partition.")
    if len(alpha_candidates) == 0 or len(set(alpha_candidates)) != len(alpha_candidates):
        raise ComponentBenchmarkError("Alpha candidates must be non-empty and unique.")
    group_count = int(frame["source_document"].nunique())
    if group_count < outer_folds:
        raise ComponentBenchmarkError("Too few source groups for the requested outer folds.")

    working = frame.sort_values("item_id").reset_index(drop=True)
    groups = working["source_document"].astype(str)
    splitter = GroupKFold(n_splits=outer_folds)
    baseline_prediction: np.ndarray = np.empty((len(working), 2), dtype=float)
    ridge_prediction: np.ndarray = np.empty((len(working), 2), dtype=float)
    assigned: np.ndarray = np.zeros(len(working), dtype=int)
    outer_records: list[dict[str, Any]] = []
    development_records: list[dict[str, Any]] = []
    selected_alphas: list[float] = []
    model_revision_hashes: list[str] = []

    for outer_fold, (train_index, holdout_index) in enumerate(
        splitter.split(working, groups=groups)
    ):
        train = working.iloc[train_index]
        holdout = working.iloc[holdout_index]
        if set(train["source_document"]) & set(holdout["source_document"]):
            raise ComponentBenchmarkError("A source document crossed an outer fold.")
        if int(train["source_document"].nunique()) < inner_folds:
            raise ComponentBenchmarkError("An outer-training partition cannot support inner folds.")

        trials: list[dict[str, Any]] = []
        for alpha in alpha_candidates:
            objective, inner_records = _inner_objective(
                train.reset_index(drop=True),
                alpha=float(alpha),
                folds=inner_folds,
                max_features=max_features,
            )
            trials.append(
                {
                    "alpha": float(alpha),
                    "mean_valence_arousal_rmse": objective,
                    "inner_folds": inner_records,
                }
            )
        selected = min(
            trials,
            key=lambda trial: (
                trial["mean_valence_arousal_rmse"],
                alpha_candidates.index(trial["alpha"]),
            ),
        )
        selected_alpha = float(selected["alpha"])
        selected_alphas.append(selected_alpha)

        baseline = MeanTextVaBaseline().fit(train.assign(split="train"))
        ridge = _fit_ridge(train, alpha=selected_alpha, max_features=max_features)
        baseline_fold = baseline.predict_batch(holdout["text"].astype(str).tolist())
        ridge_fold = ridge.predict_batch(holdout["text"].astype(str).tolist())
        baseline_prediction[holdout_index] = baseline_fold
        ridge_prediction[holdout_index] = ridge_fold
        assigned[holdout_index] += 1
        if ridge.revision_ is None:
            raise ComponentBenchmarkError("The fitted ridge specification hash is unavailable.")
        model_revision_hashes.append(ridge.revision_)

        development_records.append(
            {
                "outer_fold": outer_fold,
                "selection_objective": "mean_valence_arousal_rmse",
                "tie_break": "listed_alpha_order",
                "candidate_alphas": [float(value) for value in alpha_candidates],
                "trials": trials,
                "selected_alpha": selected_alpha,
            }
        )
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "training_item_count": int(len(train)),
                "training_group_count": int(train["source_document"].nunique()),
                "holdout_item_count": int(len(holdout)),
                "holdout_group_count": int(holdout["source_document"].nunique()),
                "holdout_item_set_sha256": _stable_hash(holdout["item_id"].astype(str).tolist()),
                "selected_alpha": selected_alpha,
            }
        )
    if not np.all(assigned == 1):
        raise ComponentBenchmarkError(
            "Outer held-out predictions must cover each item exactly once."
        )

    predictions = working[["item_id", "source_document", "valence", "arousal"]].copy()
    predictions["mean_valence"] = baseline_prediction[:, 0]
    predictions["mean_arousal"] = baseline_prediction[:, 1]
    predictions["ridge_valence"] = ridge_prediction[:, 0]
    predictions["ridge_arousal"] = ridge_prediction[:, 1]

    models = {
        "training_fold_mean": ("mean_valence", "mean_arousal"),
        "nested_tfidf_ridge": ("ridge_valence", "ridge_arousal"),
    }
    metrics: dict[str, Any] = {}
    for model_name, columns in models.items():
        values = predictions[list(columns)].to_numpy(float)
        metrics[model_name] = {
            "point_estimates": evaluate_two_target_predictions(predictions, values),
            "source_group_bootstrap_95ci": grouped_bootstrap_intervals(
                predictions,
                values,
                replicates=bootstrap_replicates,
                seed=seed,
            ),
        }
    paired: dict[str, Any] = {}
    for dimension in ("valence", "arousal"):
        paired[dimension] = paired_group_bootstrap_model_comparison(
            predictions,
            observed_column=dimension,
            first_prediction_column=f"ridge_{dimension}",
            second_prediction_column=f"mean_{dimension}",
            group_column="source_document",
            replicates=bootstrap_replicates,
            seed=seed,
        )

    component_results = {
        "schema_version": 1,
        "status": DEMONSTRATION_STATUS,
        "scientific_scope": "EmoBank text-component evidence only",
        "analysis_pool": "published_train_and_dev_only",
        "test_partition_evaluated": False,
        "daic_or_phq_data_used": False,
        "audio_model_used": False,
        "cross_modal_discrepancy_computed": False,
        "item_count": int(len(working)),
        "source_document_count": group_count,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "held_out_prediction_count": int(len(predictions)),
        "held_out_predictions_per_item": 1,
        "metrics": metrics,
        "paired_ridge_minus_mean_comparison": paired,
        "out_of_range_diagnostics": _out_of_range_report(
            predictions,
            model_columns=models,
        ),
        "claim_boundary": (
            "Supports source-grouped performance statements about the demonstrated text "
            "estimators on the EmoBank v1.4 train/development pool only; does not validate "
            "audio, a common text/audio scale, a discrepancy construct, DAIC/PHQ association, "
            "internal state, or clinical use."
        ),
    }
    development_histories = {
        "schema_version": 1,
        "selection_used_outer_holdout_metrics": False,
        "identical_candidate_set_in_every_outer_fold": True,
        "candidate_alphas": [float(value) for value in alpha_candidates],
        "outer_fold_histories": development_records,
    }
    split_manifest = {
        "schema_version": 1,
        "grouping_field": "official_meta_tsv_document",
        "source_groups_cross_outer_folds": 0,
        "items_with_exactly_one_outer_holdout_prediction": int(np.sum(assigned == 1)),
        "item_identifiers_included": False,
        "outer_folds": outer_records,
    }
    model_revision_hashes.sort()
    development_histories["fitted_ridge_specification_hashes"] = model_revision_hashes
    return GroupedTextDemoResult(
        predictions=predictions,
        component_results=component_results,
        development_histories=development_histories,
        split_manifest=split_manifest,
    )
