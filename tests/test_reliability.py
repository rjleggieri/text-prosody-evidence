import pandas as pd
import pytest

from text_prosody.contracts import ContractError
from text_prosody.reliability import (
    human_reference_reliability_report,
    rater_bootstrap_reliability,
    rating_design_assessment,
    raw_rating_reliability_report,
    validate_raw_human_ratings,
    validate_raw_rating_linkage,
)


def aggregate_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["SYNTHETIC_EXTERNAL"] * 5,
            "dataset_version": ["fixture-v1"] * 5,
            "item_id": [f"item-{index}" for index in range(5)],
            "provenance_group_id": [f"group-{index}" for index in range(5)],
            "text_annotation_condition": ["text_only"] * 5,
            "audio_annotation_condition": ["audio_only"] * 5,
            "scale_name": ["synthetic_1_5"] * 5,
            "scale_min": [1.0] * 5,
            "scale_max": [5.0] * 5,
            "human_text_valence": [1.5, 2.0, 3.0, 4.0, 4.5],
            "human_text_arousal": [4.5, 4.0, 3.0, 2.0, 1.5],
            "human_audio_valence": [1.7, 2.1, 3.1, 3.9, 4.4],
            "human_audio_arousal": [4.3, 3.8, 3.2, 2.2, 1.6],
        }
    )


def raw_ratings() -> pd.DataFrame:
    rows = []
    base = [1.5, 2.0, 3.0, 4.0, 4.5]
    for condition in ("text_only", "audio_only"):
        for dimension in ("valence", "arousal"):
            for item, center in enumerate(base):
                target = center if dimension == "valence" else 6 - center
                for rater, offset in enumerate((-0.15, -0.05, 0.05, 0.15)):
                    rows.append(
                        {
                            "dataset_id": "SYNTHETIC_EXTERNAL",
                            "dataset_version": "fixture-v1",
                            "item_id": f"item-{item}",
                            "annotation_condition": condition,
                            "dimension": dimension,
                            "rater_id": f"rater-{rater}",
                            "rating": target + offset,
                            "scale_name": "synthetic_1_5",
                            "scale_min": 1.0,
                            "scale_max": 5.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_raw_rater_reliability_and_noise_ceiling_interfaces() -> None:
    report = raw_rating_reliability_report(raw_ratings())
    assert report["status"] == "RAW_RATER_RELIABILITY_AVAILABLE"
    assert report["results"]["text_only:valence"]["icc_2_k_average_raters"] > 0.9
    assert 0 <= report["results"]["audio_only:arousal"]["estimated_correlation_noise_ceiling"] <= 1
    bootstrap = rater_bootstrap_reliability(raw_ratings(), replicates=100, seed=4)
    assert bootstrap["resampling_unit"] == "rater"
    assert bootstrap["must_not_be_combined_with"] == "source_group_bootstrap"


def test_aggregate_only_reliability_fallback_does_not_infer_values() -> None:
    report = human_reference_reliability_report(aggregate_reference())
    assert report["status"] == "AGGREGATE_RATINGS_ONLY"
    assert report["human_noise_ceiling_available"] is False


def test_raw_rating_duplicate_and_mixed_scale_guards() -> None:
    duplicate = pd.concat([raw_ratings(), raw_ratings().iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate item"):
        validate_raw_human_ratings(duplicate)
    mixed = raw_ratings()
    mixed.loc[0, "scale_name"] = "other"
    with pytest.raises(ContractError, match="homogeneous scale_name"):
        validate_raw_human_ratings(mixed)


def aggregate_from_raw() -> pd.DataFrame:
    aggregate = aggregate_reference()
    raw = raw_ratings()
    targets = {
        ("text_only", "valence"): "human_text_valence",
        ("text_only", "arousal"): "human_text_arousal",
        ("audio_only", "valence"): "human_audio_valence",
        ("audio_only", "arousal"): "human_audio_arousal",
    }
    for (condition, dimension), target in targets.items():
        means = (
            raw.loc[raw["annotation_condition"].eq(condition) & raw["dimension"].eq(dimension)]
            .groupby("item_id")["rating"]
            .mean()
        )
        aggregate[target] = aggregate["item_id"].map(means)
    return aggregate


def test_raw_ratings_must_link_exactly_to_the_aggregate_reference() -> None:
    aggregate = aggregate_from_raw()
    report = human_reference_reliability_report(
        aggregate,
        raw_ratings=raw_ratings(),
        aggregate_mean_tolerance=1e-12,
    )
    assert report["aggregate_linkage_validated"] is True
    assert report["noise_ceiling_status"].startswith("ASSUMPTION_DEPENDENT")

    wrong_version = raw_ratings()
    wrong_version.loc[0, "dataset_version"] = "unrelated-version"
    with pytest.raises(ContractError, match="identical item keys"):
        validate_raw_rating_linkage(aggregate, wrong_version)

    wrong_condition = raw_ratings()
    wrong_condition = wrong_condition.loc[
        ~(
            wrong_condition["annotation_condition"].eq("text_only")
            & wrong_condition["dimension"].eq("arousal")
        )
    ]
    with pytest.raises(ContractError, match="exactly the required"):
        validate_raw_rating_linkage(aggregate, wrong_condition)

    wrong_scale = raw_ratings()
    wrong_scale["scale_name"] = "different"
    with pytest.raises(ContractError, match="same scale name and bounds"):
        validate_raw_rating_linkage(aggregate, wrong_scale)

    wrong_mean = raw_ratings()
    wrong_mean.loc[0, "rating"] += 0.1
    with pytest.raises(ContractError, match="means do not match"):
        validate_raw_rating_linkage(aggregate, wrong_mean, mean_tolerance=1e-12)


def test_incomplete_or_partially_crossed_ratings_return_unsupported_outcome() -> None:
    incomplete = raw_ratings().drop(index=0).reset_index(drop=True)
    design = rating_design_assessment(incomplete)
    assert design["status"] == "UNSUPPORTED_RATING_DESIGN_INCOMPLETE_OR_PARTIALLY_CROSSED"
    report = raw_rating_reliability_report(incomplete)
    assert report["rater_level_reliability_available"] is False
    bootstrap = rater_bootstrap_reliability(incomplete, replicates=100)
    assert bootstrap["rater_bootstrap_available"] is False


def test_rater_bootstrap_reports_unavailable_interval_when_replicates_are_degenerate() -> None:
    constant = raw_ratings()
    constant["rating"] = 3.0
    report = rater_bootstrap_reliability(constant, replicates=100, seed=8)
    result = report["results"]["text_only:valence"]
    assert result["attempted_replicates"] == 100
    assert result["valid_replicates"] == 0
    assert result["interval_available"] is False
    assert result["icc_2_k_lower_95"] is None
    assert result["icc_2_k_upper_95"] is None
