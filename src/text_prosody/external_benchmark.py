"""Contracts for external benchmarks with paired human modality annotations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .contracts import ContractError, assert_disjoint_identifiers

HUMAN_REFERENCE_COLUMNS = {
    "dataset_id",
    "dataset_version",
    "item_id",
    "text_annotation_condition",
    "audio_annotation_condition",
    "scale_name",
    "scale_min",
    "scale_max",
    "human_text_valence",
    "human_text_arousal",
    "human_audio_valence",
    "human_audio_arousal",
}

HUMAN_VALUE_COLUMNS = [
    "human_text_valence",
    "human_text_arousal",
    "human_audio_valence",
    "human_audio_arousal",
]
COMPOSITE_KEY_COLUMNS = ["dataset_id", "dataset_version", "item_id"]
LEAKAGE_RELATION_COLUMNS = (
    "source_video_id",
    "speaker_id",
    "creator_id",
    "channel_id",
    "duplicate_content_id",
    "provenance_group_id",
)


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def composite_item_keys(frame: pd.DataFrame) -> pd.Series:
    """Return collision-safe dataset/version/item keys for joins and held-out guards."""
    missing = sorted(set(COMPOSITE_KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ContractError(f"external benchmark missing composite-key columns: {missing}")
    if frame[COMPOSITE_KEY_COLUMNS].apply(_blank_mask).any().any():
        raise ContractError("External benchmark composite keys cannot be blank.")
    return (
        frame[COMPOSITE_KEY_COLUMNS]
        .astype(str)
        .apply(
            lambda row: json.dumps(row.tolist(), ensure_ascii=True, separators=(",", ":")),
            axis=1,
        )
    )


def validate_paired_human_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one paired text-only/audio-only human reference row per item."""
    if frame.columns.duplicated().any():
        raise ContractError("external benchmark contains duplicate column metadata.")
    missing = sorted(HUMAN_REFERENCE_COLUMNS - set(frame.columns))
    if missing:
        raise ContractError(f"external benchmark missing required columns: {missing}")
    if frame[list(HUMAN_REFERENCE_COLUMNS)].isna().any().any():
        raise ContractError("external benchmark contains missing required metadata or ratings.")
    keys = composite_item_keys(frame)
    if keys.duplicated().any():
        raise ContractError("external benchmark item keys must be unique.")
    required_text_metadata = [
        "dataset_id",
        "dataset_version",
        "item_id",
        "text_annotation_condition",
        "audio_annotation_condition",
        "scale_name",
    ]
    if frame[required_text_metadata].apply(_blank_mask).any().any():
        raise ContractError("External benchmark required metadata cannot be blank.")
    if not frame["text_annotation_condition"].eq("text_only").all():
        raise ContractError("Text references must come from an explicit text_only condition.")
    if not frame["audio_annotation_condition"].eq("audio_only").all():
        raise ContractError("Audio references must come from an explicit audio_only condition.")
    normalized_scale_names = frame["scale_name"].astype(str).str.strip()
    if normalized_scale_names.nunique(dropna=False) != 1:
        raise ContractError("An external analysis must use one homogeneous scale_name.")

    lower = pd.to_numeric(frame["scale_min"], errors="coerce")
    upper = pd.to_numeric(frame["scale_max"], errors="coerce")
    values = frame[HUMAN_VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if (
        lower.isna().any()
        or upper.isna().any()
        or values.isna().any().any()
        or not np.isfinite(values.to_numpy()).all()
    ):
        raise ContractError("Human-reference scales and ratings must be finite numeric values.")
    if (upper <= lower).any():
        raise ContractError("Each human-reference scale must have scale_max > scale_min.")
    if lower.nunique(dropna=False) != 1 or upper.nunique(dropna=False) != 1:
        raise ContractError("An external analysis must use homogeneous scale bounds.")
    outside = values.lt(lower, axis=0) | values.gt(upper, axis=0)
    if outside.any().any():
        raise ContractError("Human-reference ratings fall outside their declared scale.")
    available_grouping = [column for column in LEAKAGE_RELATION_COLUMNS if column in frame]
    for column in available_grouping:
        if _blank_mask(frame[column]).any():
            raise ContractError(f"Grouping metadata cannot be blank: {column}")
    return frame


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def connected_leakage_components(
    frame: pd.DataFrame,
    *,
    relation_columns: Sequence[str] = LEAKAGE_RELATION_COLUMNS,
) -> pd.Series:
    """Collapse every transitive source/speaker/creator/duplicate/provenance relation."""
    validate_paired_human_reference(frame)
    available = [column for column in relation_columns if column in frame.columns]
    if not available:
        raise ContractError("No leakage-relation metadata is available.")
    keys = composite_item_keys(frame).reset_index(drop=True)
    components = _DisjointSet(len(frame))
    seen: dict[tuple[str, str, str, str], int] = {}
    normalized = frame.reset_index(drop=True)
    for index, (_, row) in enumerate(normalized.iterrows()):
        namespace = (str(row["dataset_id"]), str(row["dataset_version"]))
        for column in available:
            relation = (*namespace, column, str(row[column]).strip())
            previous = seen.setdefault(relation, index)
            components.union(previous, index)
    members: dict[int, list[str]] = {}
    for index, key in enumerate(keys):
        members.setdefault(components.find(index), []).append(key)
    labels = {
        root: "component:" + hashlib.sha256("\n".join(sorted(component_keys)).encode()).hexdigest()
        for root, component_keys in members.items()
    }
    values = [labels[components.find(index)] for index in range(len(frame))]
    return pd.Series(values, index=frame.index, name="leakage_component_id")


def attach_connected_leakage_components(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["leakage_component_id"] = connected_leakage_components(frame)
    return output


def paired_human_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Return signed and magnitude human modality differences on the declared scale."""
    validated = validate_paired_human_reference(frame)
    output = validated.copy()
    output["human_delta_valence"] = output["human_text_valence"] - output["human_audio_valence"]
    output["human_delta_arousal"] = output["human_text_arousal"] - output["human_audio_arousal"]
    output["human_divergence_l2"] = np.hypot(
        output["human_delta_valence"], output["human_delta_arousal"]
    )
    return output


def assert_group_split_is_disjoint(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_columns: Sequence[str] = LEAKAGE_RELATION_COLUMNS,
) -> None:
    """Fail if any available direct leakage relation crosses a split."""
    if not group_columns:
        raise ContractError("At least one grouping column is required.")
    available = [column for column in group_columns if column in train and column in test]
    if not available:
        raise ContractError("No requested grouping column is available in both splits.")
    for column in available:
        if _blank_mask(train[column]).any() or _blank_mask(test[column]).any():
            raise ContractError(f"Grouping column contains blank values: {column}")
        assert_disjoint_identifiers(train[column], test[column], kind=column)
