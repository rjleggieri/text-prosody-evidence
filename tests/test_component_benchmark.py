from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from text_prosody.component_benchmark import (
    ComponentBenchmarkError,
    calibration_line,
    continuous_metrics,
    documented_source_prefix,
    evaluate_two_target_predictions,
    grouped_bootstrap_intervals,
    load_emobank_reader_release,
    stable_identifier_hash,
)


def test_documented_source_prefix_and_identifier_hash_are_deterministic() -> None:
    assert documented_source_prefix("fiction_source_10_20") == "fiction_source"
    assert stable_identifier_hash(["b", "a"]) == stable_identifier_hash(["a", "b"])
    with pytest.raises(ComponentBenchmarkError):
        documented_source_prefix("not-a-span")


def test_metrics_and_grouped_bootstrap_use_fictional_rows() -> None:
    frame = pd.DataFrame(
        {
            "item_id": [f"fiction_{index}" for index in range(8)],
            "source_document": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "valence": np.linspace(1.0, 5.0, 8),
            "arousal": np.linspace(5.0, 1.0, 8),
        }
    )
    prediction = frame[["valence", "arousal"]].to_numpy() + 0.1
    metrics = evaluate_two_target_predictions(frame, prediction)
    assert metrics["valence"]["mae"] == pytest.approx(0.1)
    intervals = grouped_bootstrap_intervals(frame, prediction, replicates=100, seed=4)
    assert intervals["arousal"]["rmse"]["valid_replicates"] == 100
    assert intervals["arousal"]["rmse"]["attempted_replicates"] == 100
    assert intervals["arousal"]["rmse"]["invalid_replicates"] == 0
    assert intervals["arousal"]["rmse"]["interval_available"] is True


def test_component_bootstrap_suppresses_interval_below_valid_fraction() -> None:
    frame = pd.DataFrame(
        {
            "source_document": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "valence": np.linspace(1.0, 5.0, 8),
            "arousal": np.linspace(5.0, 1.0, 8),
        }
    )
    prediction = np.full((len(frame), 2), 3.0)
    intervals = grouped_bootstrap_intervals(frame, prediction, replicates=100, seed=4)
    pearson = intervals["valence"]["pearson"]
    assert pearson["attempted_replicates"] == 100
    assert pearson["valid_replicates"] == 0
    assert pearson["invalid_replicates"] == 100
    assert pearson["invalid_counts_by_reason"] == {"constant_prediction": 100}
    assert pearson["minimum_valid_fraction"] == 0.9
    assert pearson["interval_available"] is False
    assert pearson["low"] is None
    assert pearson["high"] is None


def test_constant_predictions_have_no_calibration_or_correlations() -> None:
    metrics = continuous_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert metrics["pearson"] is None
    assert metrics["spearman"] is None
    assert calibration_line([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == (None, None)


def test_emobank_loader_joins_reader_perspective_and_rejects_outcomes(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "id": ["docA_0_1", "docA_2_3", "docB_0_1"],
            "text": ["fiction one", "fiction two", "fiction three"],
        }
    ).to_csv(tmp_path / "raw.csv", index=False)
    pd.DataFrame(
        {
            "id": ["docA_0_1", "docA_2_3", "docB_0_1"],
            "V": [2.0, 3.0, 4.0],
            "A": [4.0, 3.0, 2.0],
        }
    ).to_csv(tmp_path / "reader.csv", index=False)
    pd.DataFrame(
        {
            "id": ["docA_0_1", "docA_2_3", "docB_0_1"],
            "split": ["train", "dev", "test"],
        }
    ).to_csv(tmp_path / "emobank.csv", index=False)
    pd.DataFrame(
        {
            "id": ["docA_0_1", "docA_2_3", "docB_0_1"],
            "document": ["docA", "docA", "docB"],
            "category": ["x", "x", "y"],
            "subcategory": [None, None, None],
        }
    ).to_csv(tmp_path / "meta.tsv", sep="\t", index=False)
    frame, audit = load_emobank_reader_release(tmp_path)
    assert len(frame) == 3
    assert audit["missingness"]["text"] == 0
    assert audit["analysis_exclusions"]["missing_or_empty_text"] == 0

    reader = pd.read_csv(tmp_path / "reader.csv")
    reader["outcome"] = 0
    reader.to_csv(tmp_path / "reader.csv", index=False)
    with pytest.raises(ComponentBenchmarkError, match="prohibited outcome"):
        load_emobank_reader_release(tmp_path)
