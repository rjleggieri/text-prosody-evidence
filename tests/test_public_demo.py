from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from text_prosody.component_benchmark import ComponentBenchmarkError
from text_prosody.public_demo import (
    load_emobank_reader_development_pool,
    run_grouped_text_component_demo,
)


def _write_release(root: Path) -> None:
    identifiers = [
        "source_a_0_1",
        "source_a_2_3",
        "source_b_0_1",
        "source_c_0_1",
    ]
    pd.DataFrame(
        {
            "id": identifiers,
            "text": ["quiet day", "bright day", "busy day", "prohibited test text"],
        }
    ).to_csv(root / "raw.csv", index=False)
    pd.DataFrame(
        {
            "id": identifiers,
            "V": [2.0, 4.0, 3.0, 5.0],
            "A": [2.5, 3.5, 4.0, 5.0],
        }
    ).to_csv(root / "reader.csv", index=False)
    pd.DataFrame(
        {
            "id": identifiers,
            "split": ["train", "train", "dev", "test"],
        }
    ).to_csv(root / "emobank.csv", index=False)
    pd.DataFrame(
        {
            "id": identifiers,
            "document": ["source_a", "source_a", "source_b", "source_c"],
            "category": ["x", "x", "y", "z"],
            "subcategory": [None, None, None, None],
        }
    ).to_csv(root / "meta.tsv", sep="\t", index=False)


def _synthetic_component_frame(groups: int = 12, items_per_group: int = 3) -> pd.DataFrame:
    rows = []
    for group in range(groups):
        for item in range(items_per_group):
            value = 1.2 + 3.6 * (group * items_per_group + item) / (groups * items_per_group - 1)
            rows.append(
                {
                    "item_id": f"fiction_{group}_{item}",
                    "text": (
                        f"token{group % 4} "
                        + ("pleasant calm" if value >= 3 else "unpleasant active")
                    ),
                    "valence": value,
                    "arousal": 6.0 - value,
                    "split": "train" if group < groups - 2 else "dev",
                    "dataset_release": "v1.4",
                    "source_document": f"source-{group}",
                }
            )
    return pd.DataFrame(rows)


def test_development_pool_excludes_test_before_returning_rows(tmp_path: Path) -> None:
    _write_release(tmp_path)
    frame, audit = load_emobank_reader_development_pool(tmp_path)
    assert len(frame) == 3
    assert set(frame["split"]) == {"train", "dev"}
    assert "prohibited test text" not in set(frame["text"])
    assert audit["excluded_partitions"] == ["test"]
    assert audit["test_partition_used_for_modeling_or_evaluation"] is False
    assert "split_identifier_sha256" not in audit


def test_loader_fails_closed_on_blank_source_group(tmp_path: Path) -> None:
    _write_release(tmp_path)
    metadata = pd.read_csv(tmp_path / "meta.tsv", sep="\t")
    metadata.loc[metadata["id"].eq("source_b_0_1"), "document"] = ""
    metadata.to_csv(tmp_path / "meta.tsv", sep="\t", index=False)
    with pytest.raises(ComponentBenchmarkError, match="incomplete or blank"):
        load_emobank_reader_development_pool(tmp_path)


def test_grouped_demo_produces_one_prediction_per_item_and_aggregate_outputs() -> None:
    frame = _synthetic_component_frame()
    result = run_grouped_text_component_demo(
        frame,
        outer_folds=3,
        inner_folds=2,
        alpha_candidates=(1.0, 10.0),
        max_features=100,
        bootstrap_replicates=100,
        seed=8,
    )
    assert len(result.predictions) == len(frame)
    assert result.predictions["item_id"].nunique() == len(frame)
    assert result.component_results["held_out_predictions_per_item"] == 1
    assert result.component_results["test_partition_evaluated"] is False
    assert result.component_results["cross_modal_discrepancy_computed"] is False
    assert result.split_manifest["source_groups_cross_outer_folds"] == 0
    assert result.split_manifest["item_identifiers_included"] is False
    assert len(result.split_manifest["outer_folds"]) == 3
    assert all(
        record["training_group_count"] + record["holdout_group_count"]
        == frame["source_document"].nunique()
        for record in result.split_manifest["outer_folds"]
    )
    assert set(result.component_results["metrics"]) == {
        "training_fold_mean",
        "nested_tfidf_ridge",
    }
    paired = result.component_results["paired_ridge_minus_mean_comparison"]
    assert paired["valence"]["mae"]["attempted_replicates"] == 100


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("split", "test", "prohibited partition"),
        ("source_document", "", "complete and nonblank"),
    ],
)
def test_grouped_demo_rejects_scientifically_prohibited_inputs(
    column: str,
    value: str,
    message: str,
) -> None:
    frame = _synthetic_component_frame()
    frame.loc[0, column] = value
    with pytest.raises(ComponentBenchmarkError, match=message):
        run_grouped_text_component_demo(
            frame,
            outer_folds=3,
            inner_folds=2,
            bootstrap_replicates=100,
        )


def test_grouped_demo_rejects_outcome_columns_and_insufficient_groups() -> None:
    with_outcome = _synthetic_component_frame()
    with_outcome["phq_score"] = np.arange(len(with_outcome))
    with pytest.raises(ComponentBenchmarkError, match="prohibited outcome"):
        run_grouped_text_component_demo(
            with_outcome,
            outer_folds=3,
            inner_folds=2,
            bootstrap_replicates=100,
        )

    too_few = _synthetic_component_frame(groups=2)
    with pytest.raises(ComponentBenchmarkError, match="Too few source groups"):
        run_grouped_text_component_demo(
            too_few,
            outer_folds=3,
            inner_folds=2,
            bootstrap_replicates=100,
        )


def test_public_demo_environment_lock_is_core_only_and_fully_pinned() -> None:
    root = Path(__file__).parents[1]
    lines = [
        line.strip()
        for line in (root / "requirements-public-demo-lock.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines)
    assert all(not line.casefold().startswith(("torch", "transformers")) for line in lines)
