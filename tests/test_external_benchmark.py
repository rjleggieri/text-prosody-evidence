import numpy as np
import pandas as pd
import pytest

from text_prosody.contracts import ContractError
from text_prosody.external_benchmark import (
    assert_group_split_is_disjoint,
    composite_item_keys,
    connected_leakage_components,
    paired_human_differences,
    validate_paired_human_reference,
)


def reference_frame(n: int = 12) -> pd.DataFrame:
    values = np.linspace(1.2, 4.8, n)
    return pd.DataFrame(
        {
            "dataset_id": "SYNTHETIC_EXTERNAL",
            "dataset_version": "sha256:fixture",
            "item_id": [f"item-{index}" for index in range(n)],
            "provenance_group_id": [f"video-{index // 2}" for index in range(n)],
            "source_video_id": [f"video-{index // 2}" for index in range(n)],
            "text_annotation_condition": "text_only",
            "audio_annotation_condition": "audio_only",
            "scale_name": "sam_va_1_5",
            "scale_min": 1.0,
            "scale_max": 5.0,
            "human_text_valence": values,
            "human_text_arousal": values[::-1],
            "human_audio_valence": np.clip(values - 0.4, 1, 5),
            "human_audio_arousal": np.clip(values[::-1] + 0.3, 1, 5),
        }
    )


def test_human_reference_signed_and_magnitude_differences() -> None:
    output = paired_human_differences(reference_frame())
    assert np.allclose(
        output["human_divergence_l2"],
        np.hypot(output["human_delta_valence"], output["human_delta_arousal"]),
    )
    assert output["human_delta_valence"].mean() > 0
    assert output["human_delta_arousal"].mean() < 0


def test_condition_and_scale_contracts_fail_loudly() -> None:
    wrong_condition = reference_frame()
    wrong_condition.loc[0, "audio_annotation_condition"] = "full"
    with pytest.raises(ContractError, match="audio_only"):
        validate_paired_human_reference(wrong_condition)
    outside_scale = reference_frame()
    outside_scale.loc[0, "human_text_valence"] = 9
    with pytest.raises(ContractError, match="outside"):
        validate_paired_human_reference(outside_scale)


def test_mixed_scales_and_duplicate_composite_item_ids_are_rejected() -> None:
    mixed = reference_frame()
    mixed.loc[0, "scale_name"] = "different_scale"
    with pytest.raises(ContractError, match="homogeneous scale_name"):
        validate_paired_human_reference(mixed)
    mixed_bounds = reference_frame()
    mixed_bounds.loc[0, "scale_max"] = 7
    with pytest.raises(ContractError, match="homogeneous scale bounds"):
        validate_paired_human_reference(mixed_bounds)
    duplicated = reference_frame()
    duplicated.loc[1, "item_id"] = duplicated.loc[0, "item_id"]
    with pytest.raises(ContractError, match="item keys must be unique"):
        validate_paired_human_reference(duplicated)


def test_blank_grouping_metadata_and_duplicate_columns_are_rejected() -> None:
    blank = reference_frame()
    blank.loc[0, "source_video_id"] = " "
    with pytest.raises(ContractError, match="Grouping metadata cannot be blank"):
        validate_paired_human_reference(blank)
    duplicate_columns = reference_frame()
    duplicate_columns.columns = [
        *duplicate_columns.columns[:-1],
        duplicate_columns.columns[-2],
    ]
    with pytest.raises(ContractError, match="duplicate column metadata"):
        validate_paired_human_reference(duplicate_columns)


def test_connected_components_close_transitive_leakage_relations() -> None:
    frame = reference_frame(n=6)
    frame["speaker_id"] = [f"speaker-{index}" for index in range(6)]
    frame["channel_id"] = [f"channel-{index}" for index in range(6)]
    frame["duplicate_content_id"] = [f"content-{index}" for index in range(6)]
    frame["provenance_group_id"] = [f"provenance-{index}" for index in range(6)]
    frame.loc[1, "speaker_id"] = frame.loc[0, "speaker_id"]
    frame.loc[2, "channel_id"] = frame.loc[1, "channel_id"]
    frame.loc[3, "duplicate_content_id"] = frame.loc[2, "duplicate_content_id"]
    components = connected_leakage_components(frame)
    assert components.iloc[:4].nunique() == 1
    assert components.iloc[4:].nunique() == 1
    assert components.iloc[0] != components.iloc[4]
    assert composite_item_keys(frame).nunique() == len(frame)


def test_group_split_guard_detects_source_leakage() -> None:
    frame = reference_frame()
    with pytest.raises(ContractError, match="leakage detected"):
        assert_group_split_is_disjoint(frame.iloc[:3], frame.iloc[3:6])
