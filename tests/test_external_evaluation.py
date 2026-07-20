import numpy as np
import pandas as pd
import pytest

from text_prosody.external_evaluation import (
    CLIPPED_TO_SCALE_SENSITIVITY,
    connected_component_feasibility_report,
    cross_validated_external_predictions,
    external_prediction_metrics,
    fit_frozen_external_calibration,
    freeze_final_external_candidate,
    grouped_bootstrap_intervals,
    make_synthetic_external_benchmark,
    nested_grouped_external_evaluation,
    paired_group_bootstrap_model_comparison,
    primary_out_of_range_report,
)


def benchmark_frame(groups: int = 20, items_per_group: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(20260719)
    n = groups * items_per_group
    text_raw = rng.normal(size=(n, 2))
    audio_raw = rng.normal(size=(n, 2))
    text_human = 3 + text_raw * np.array([0.8, 0.6]) + rng.normal(0, 0.08, (n, 2))
    audio_human = 3 + audio_raw * np.array([0.7, 0.9]) + rng.normal(0, 0.08, (n, 2))
    return pd.DataFrame(
        {
            "dataset_id": "SYNTHETIC_EXTERNAL",
            "dataset_version": "sha256:fixture",
            "item_id": [f"item-{index}" for index in range(n)],
            "provenance_group_id": [f"source-{index // items_per_group}" for index in range(n)],
            "source_video_id": [f"source-{index // items_per_group}" for index in range(n)],
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


def metadata() -> dict[str, str]:
    return {
        "dataset_id": "SYNTHETIC_EXTERNAL",
        "dataset_version": "sha256:fixture",
        "split_hash": "sha256:split",
        "text_model_id": "synthetic-text",
        "text_model_revision": "sha256:text",
        "audio_model_id": "synthetic-audio",
        "audio_model_revision": "sha256:audio",
    }


def test_package_synthetic_external_fixture_is_contract_valid() -> None:
    frame = make_synthetic_external_benchmark(groups=5, items_per_group=2)
    predictions = cross_validated_external_predictions(frame, folds=5)
    assert len(predictions) == 10
    assert predictions["item_id"].str.startswith("SYNTHETIC_").all()
    assert set(frame["synthetic_reference_pattern"]) == {"aligned", "divergent"}


def test_grouped_fold_calibration_is_held_out_and_scores_divergence() -> None:
    predictions = cross_validated_external_predictions(benchmark_frame(), folds=5)
    assert len(predictions) == 60
    assert predictions.groupby("leakage_component_id")["fold"].nunique().max() == 1
    metrics = external_prediction_metrics(predictions)
    assert metrics["text_valence"]["pearson"] > 0.9
    assert 0 <= metrics["delta_valence"]["signed_direction_accuracy"] <= 1
    assert set(metrics["divergence_l2"]) == {
        "n",
        "pearson",
        "spearman",
        "ccc",
        "mae",
        "rmse",
    }


def test_grouped_bootstrap_returns_paired_intervals() -> None:
    predictions = cross_validated_external_predictions(benchmark_frame(), folds=4)
    intervals = grouped_bootstrap_intervals(predictions, replicates=100, seed=3)
    assert intervals["delta_valence"]["pearson"]["valid_replicates"] == 100
    assert intervals["delta_valence"]["pearson"]["attempted_replicates"] == 100
    assert intervals["delta_valence"]["pearson"]["invalid_replicates"] == 0
    assert intervals["delta_valence"]["pearson"]["minimum_valid_fraction"] == 0.9
    assert intervals["delta_valence"]["pearson"]["interval_available"] is True
    assert (
        intervals["divergence_l2"]["mae"]["lower_95"]
        <= intervals["divergence_l2"]["mae"]["upper_95"]
    )


def test_grouped_bootstrap_with_too_few_valid_replicates_suppresses_interval() -> None:
    predictions = cross_validated_external_predictions(benchmark_frame(), folds=4)
    predictions["human_delta_valence"] = 0.0
    predictions["pred_delta_valence"] = 0.0
    intervals = grouped_bootstrap_intervals(predictions, replicates=100, seed=3)
    pearson = intervals["delta_valence"]["pearson"]
    assert pearson["attempted_replicates"] == 100
    assert pearson["valid_replicates"] == 0
    assert pearson["invalid_replicates"] == 100
    assert pearson["invalid_counts_by_reason"] == {"constant_observed": 100}
    assert pearson["interval_available"] is False
    assert pearson["lower_95"] is None
    assert pearson["upper_95"] is None


def test_frozen_artifact_has_provenance_and_rejects_phq_influence() -> None:
    frame = benchmark_frame()
    artifact = fit_frozen_external_calibration(frame, metadata=metadata())
    assert artifact["calibration_source"] == "external_human_modality_annotations_only"
    assert artifact["configuration"]["calibration_method"] == "dimensionwise_affine"
    assert artifact["provenance"]["available_group_columns"] == [
        "provenance_group_id",
        "source_video_id",
    ]
    assert "text_calibration" in artifact and "audio_calibration" in artifact
    assert len(artifact["content_sha256"]) == 64
    contaminated = frame.assign(phq8_score=4)
    with pytest.raises(ValueError, match="prohibited clinical outcome"):
        fit_frozen_external_calibration(contaminated, metadata=metadata())
    with pytest.raises(ValueError, match="DAIC data are prohibited"):
        fit_frozen_external_calibration(
            frame.assign(dataset_id="DAIC-WOZ"),
            metadata={**metadata(), "dataset_id": "DAIC-WOZ"},
        )


def test_robust_external_calibration_interface() -> None:
    frame = benchmark_frame()
    predictions = cross_validated_external_predictions(
        frame, folds=4, calibration_method="robust_location_scale"
    )
    assert np.isfinite(predictions["pred_divergence_l2"]).all()
    artifact = fit_frozen_external_calibration(
        frame,
        metadata=metadata(),
        calibration_method="robust_location_scale",
    )
    assert "model_median" in artifact["text_calibration"]


def test_nested_grouped_selection_is_outer_held_out_exactly_once() -> None:
    frame = benchmark_frame(groups=20, items_per_group=3)
    frame["weak_text_valence"] = 3.0
    frame["weak_text_arousal"] = 3.0
    candidates = {
        "strong": ("model_text_valence", "model_text_arousal"),
        "weak": ("weak_text_valence", "weak_text_arousal"),
    }
    result = nested_grouped_external_evaluation(
        frame,
        text_candidates=candidates,
        audio_estimator_id="fixed-synthetic-audio",
        outer_folds=5,
        inner_folds=3,
        scientific_feasibility_approved=True,
    )
    assert len(result.predictions) == len(frame)
    assert result.predictions[["dataset_id", "dataset_version", "item_id"]].duplicated().sum() == 0
    assert result.predictions.groupby("leakage_component_id")["outer_fold"].nunique().max() == 1
    assert result.outer_selections["primary_text_candidate"].eq("strong").all()
    assert result.outer_selections["fixed_audio_estimator_id"].eq("fixed-synthetic-audio").all()
    assert result.outer_selections["inner_split_reused_across_all_candidates"].all()
    assert not result.outer_selections["sensitivity_selection_performed"].any()
    assert result.predictions["primary_calibration_method"].eq("dimensionwise_affine").all()
    assert not result.predictions["primary_predictions_clipped"].any()
    assert set(result.sensitivity_predictions["sensitivity_method"]) == {
        "cross_dimensional_affine_sensitivity",
        "robust_location_scale",
        CLIPPED_TO_SCALE_SENSITIVITY,
    }
    assert result.sensitivity_predictions["selected_text_candidate"].eq("strong").all()
    assert not result.sensitivity_predictions["sensitivity_selected_as_best"].any()
    clipped = result.sensitivity_predictions.loc[
        result.sensitivity_predictions["sensitivity_method"].eq(CLIPPED_TO_SCALE_SENSITIVITY)
    ]
    prediction_columns = [
        "pred_text_valence",
        "pred_text_arousal",
        "pred_audio_valence",
        "pred_audio_arousal",
    ]
    assert clipped[prediction_columns].ge(clipped["scale_min"], axis=0).all().all()
    assert clipped[prediction_columns].le(clipped["scale_max"], axis=0).all().all()
    assert set(result.primary_out_of_range["direction"]) == {
        "below_declared_minimum",
        "above_declared_maximum",
    }
    assert result.feasibility_report["automatic_approval_prohibited"] is True

    fold_zero_keys = set(result.predictions.loc[result.predictions["outer_fold"].eq(0), "item_id"])
    adversarial = frame.copy()
    mask = adversarial["item_id"].isin(fold_zero_keys)
    adversarial.loc[mask, "human_text_valence"] = 5.0
    adversarial.loc[mask, "human_text_arousal"] = 1.0
    rerun = nested_grouped_external_evaluation(
        adversarial,
        text_candidates=candidates,
        audio_estimator_id="fixed-synthetic-audio",
        outer_folds=5,
        inner_folds=3,
        scientific_feasibility_approved=True,
    )
    original_selection = result.outer_selections.set_index("outer_fold").loc[0]
    rerun_selection = rerun.outer_selections.set_index("outer_fold").loc[0]
    assert original_selection["primary_text_candidate"] == rerun_selection["primary_text_candidate"]
    assert (
        original_selection["sensitivity_text_candidate_fixed"]
        == rerun_selection["sensitivity_text_candidate_fixed"]
    )


def test_nested_evaluation_never_self_approves_mathematical_feasibility() -> None:
    frame = benchmark_frame(groups=10, items_per_group=2)
    with pytest.raises(ValueError, match="explicit scientific feasibility approval"):
        nested_grouped_external_evaluation(
            frame,
            text_candidates={
                "only": ("model_text_valence", "model_text_arousal"),
            },
            audio_estimator_id="fixed-synthetic-audio",
            outer_folds=5,
            inner_folds=3,
        )


def test_connected_component_feasibility_reports_transitive_concentration() -> None:
    frame = benchmark_frame(groups=10, items_per_group=1)
    frame["speaker_id"] = [f"speaker-{index}" for index in range(10)]
    frame["creator_id"] = [f"creator-{index}" for index in range(10)]
    frame.loc[[0, 1], "speaker_id"] = "speaker-linked"
    frame.loc[[1, 2], "creator_id"] = "creator-linked"
    report = connected_component_feasibility_report(
        frame,
        proposed_outer_folds=3,
        proposed_inner_folds=2,
    )
    assert report["item_count"] == 10
    assert report["component_count"] == 8
    assert report["component_size_distribution"]["maximum"] == 3
    assert report["largest_component_share"] == pytest.approx(0.3)
    assert report["kish_effective_component_count"] == pytest.approx(6.25)
    assert report["all_outer_training_partitions_support_inner_folds"] is True
    assert report["scientific_approval_granted"] is False
    assert report["automatic_approval_prohibited"] is True


def test_primary_out_of_range_report_preserves_dimension_and_direction() -> None:
    predictions = pd.DataFrame(
        {
            "scale_min": [1.0, 1.0],
            "scale_max": [5.0, 5.0],
            "pred_text_valence": [0.0, 6.0],
            "pred_text_arousal": [2.0, 3.0],
            "pred_audio_valence": [0.5, 4.0],
            "pred_audio_arousal": [2.0, 5.5],
        }
    )
    report = primary_out_of_range_report(predictions).set_index(["dimension", "direction"])
    assert report.loc[("text_valence", "below_declared_minimum"), "out_of_range_count"] == 1
    assert report.loc[("text_valence", "above_declared_maximum"), "out_of_range_count"] == 1
    assert report.loc[("text_arousal", "above_declared_maximum"), "out_of_range_count"] == 0
    assert report.loc[("audio_arousal", "above_declared_maximum"), "out_of_range_rate"] == 0.5


def test_paired_bootstrap_compares_within_identical_source_resamples() -> None:
    rng = np.random.default_rng(7)
    observed = np.linspace(1, 5, 60)
    predictions = pd.DataFrame(
        {
            "observed": observed,
            "first": observed + rng.normal(0, 0.05, len(observed)),
            "second": observed + rng.normal(0, 0.7, len(observed)),
            "leakage_component_id": [f"group-{index // 3}" for index in range(60)],
        }
    )
    comparison = paired_group_bootstrap_model_comparison(
        predictions,
        observed_column="observed",
        first_prediction_column="first",
        second_prediction_column="second",
        replicates=100,
        seed=3,
    )
    assert comparison["mae"]["difference_first_minus_second"] < 0
    assert comparison["rmse"]["difference_first_minus_second"] < 0
    assert comparison["ccc"]["difference_first_minus_second"] > 0
    assert comparison["pearson"]["valid_replicates"] == 100
    assert comparison["pearson"]["attempted_replicates"] == 100
    assert comparison["pearson"]["invalid_replicates"] == 0
    assert comparison["pearson"]["interval_available"] is True


def test_paired_bootstrap_suppresses_invalid_correlation_difference() -> None:
    predictions = pd.DataFrame(
        {
            "observed": np.ones(20),
            "first": np.ones(20),
            "second": np.ones(20),
            "leakage_component_id": [f"group-{index // 2}" for index in range(20)],
        }
    )
    comparison = paired_group_bootstrap_model_comparison(
        predictions,
        observed_column="observed",
        first_prediction_column="first",
        second_prediction_column="second",
        replicates=100,
        seed=3,
    )
    pearson = comparison["pearson"]
    assert pearson["valid_replicates"] == 0
    assert pearson["invalid_replicates"] == 100
    assert pearson["invalid_counts_by_reason"] == {"first_constant_observed": 100}
    assert pearson["interval_available"] is False
    assert pearson["difference_lower_95"] is None
    assert pearson["difference_upper_95"] is None


def test_final_candidate_freeze_repeats_locked_rule_then_fits_calibration() -> None:
    frame = benchmark_frame(groups=20, items_per_group=2)
    frame["weak_text_valence"] = 3.0
    frame["weak_text_arousal"] = 3.0
    candidates = {
        "strong": ("model_text_valence", "model_text_arousal"),
        "weak": ("weak_text_valence", "weak_text_arousal"),
    }
    nested = nested_grouped_external_evaluation(
        frame,
        text_candidates=candidates,
        audio_estimator_id="fixed-synthetic-audio",
        outer_folds=5,
        inner_folds=3,
        development_seed_set=(11, 29),
        scientific_feasibility_approved=True,
    )
    frozen = freeze_final_external_candidate(
        frame,
        nested_result=nested,
        text_candidates=candidates,
        text_candidate_metadata={
            "strong": {"model_id": "text-strong", "model_revision": "sha256:strong"},
            "weak": {"model_id": "text-weak", "model_revision": "sha256:weak"},
        },
        calibration_metadata={
            "dataset_id": "SYNTHETIC_EXTERNAL",
            "dataset_version": "sha256:fixture",
            "split_hash": "sha256:split",
            "audio_model_id": "fixed-synthetic-audio",
            "audio_model_revision": "sha256:audio",
        },
        audio_estimator_id="fixed-synthetic-audio",
        inner_folds=3,
        development_seed_set=(11, 29),
        scientific_feasibility_approved=True,
    )
    assert frozen.selected_text_candidate == "strong"
    assert frozen.candidate_objectives["selected"].sum() == 1
    assert frozen.selection_contract["outer_test_metrics_used"] is False
    assert frozen.selection_contract["sensitivity_results_used"] is False
    assert frozen.selection_contract["daic_or_phq_outcomes_used"] is False
    assert frozen.selection_contract["final_calibration_fit_after_candidate_selection"] is True
    assert frozen.calibration_artifact["metadata"]["text_model_id"] == "text-strong"
    assert frozen.calibration_artifact["configuration"][
        "final_candidate_selected_before_calibration_fit"
    ]

    with pytest.raises(ValueError, match="prohibited clinical outcome"):
        freeze_final_external_candidate(
            frame.assign(phq8_score=0),
            nested_result=nested,
            text_candidates=candidates,
            text_candidate_metadata={
                "strong": {"model_id": "text-strong", "model_revision": "sha256:strong"},
                "weak": {"model_id": "text-weak", "model_revision": "sha256:weak"},
            },
            calibration_metadata={
                "dataset_id": "SYNTHETIC_EXTERNAL",
                "dataset_version": "sha256:fixture",
                "split_hash": "sha256:split",
                "audio_model_id": "fixed-synthetic-audio",
                "audio_model_revision": "sha256:audio",
            },
            audio_estimator_id="fixed-synthetic-audio",
            inner_folds=3,
            development_seed_set=(11, 29),
            scientific_feasibility_approved=True,
        )


def test_final_candidate_freeze_rejects_changed_seed_contract() -> None:
    frame = benchmark_frame(groups=10, items_per_group=2)
    candidates = {"only": ("model_text_valence", "model_text_arousal")}
    nested = nested_grouped_external_evaluation(
        frame,
        text_candidates=candidates,
        audio_estimator_id="fixed-synthetic-audio",
        outer_folds=5,
        inner_folds=2,
        development_seed_set=(11,),
        scientific_feasibility_approved=True,
    )
    with pytest.raises(ValueError, match="candidate set differs"):
        freeze_final_external_candidate(
            frame,
            nested_result=nested,
            text_candidates={
                **candidates,
                "added": ("model_text_valence", "model_text_arousal"),
            },
            text_candidate_metadata={
                "only": {"model_id": "text-only", "model_revision": "sha256:only"},
                "added": {"model_id": "text-added", "model_revision": "sha256:added"},
            },
            calibration_metadata={
                "dataset_id": "SYNTHETIC_EXTERNAL",
                "dataset_version": "sha256:fixture",
                "split_hash": "sha256:split",
                "audio_model_id": "fixed-synthetic-audio",
                "audio_model_revision": "sha256:audio",
            },
            audio_estimator_id="fixed-synthetic-audio",
            inner_folds=2,
            development_seed_set=(11,),
            scientific_feasibility_approved=True,
        )
    with pytest.raises(ValueError, match="seed set differs"):
        freeze_final_external_candidate(
            frame,
            nested_result=nested,
            text_candidates=candidates,
            text_candidate_metadata={
                "only": {"model_id": "text-only", "model_revision": "sha256:only"}
            },
            calibration_metadata={
                "dataset_id": "SYNTHETIC_EXTERNAL",
                "dataset_version": "sha256:fixture",
                "split_hash": "sha256:split",
                "audio_model_id": "fixed-synthetic-audio",
                "audio_model_revision": "sha256:audio",
            },
            audio_estimator_id="fixed-synthetic-audio",
            inner_folds=2,
            development_seed_set=(12,),
            scientific_feasibility_approved=True,
        )
