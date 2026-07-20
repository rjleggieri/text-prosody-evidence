"""Continuous text V/A interfaces; no DAIC labels and no import-time model access."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

EMOBANK_RELEASE = "v1.4"
EMOBANK_LICENSE = "CC-BY-SA-4.0"
PROHIBITED_OUTCOME_TOKENS = ("phq", "depress", "diagnos", "outcome")


class TextAffectError(RuntimeError):
    """Text-affect adapter failed its contract."""


class ContinuousTextBackend(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class ContinuousTextAdapter:
    """Adapter for a separately benchmarked continuous V/A regression backend."""

    model_id: str
    model_revision: str
    output_scale: str
    backend: ContinuousTextBackend

    def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if (
            not self.model_id.strip()
            or not self.model_revision.strip()
            or not self.output_scale.strip()
        ):
            raise TextAffectError("Text adapter requires model ID, immutable revision, and scale.")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise TextAffectError("Text batches must contain non-empty strings.")
        values = np.asarray(self.backend(texts), dtype=float)
        if values.shape != (len(texts), 2) or not np.isfinite(values).all():
            raise TextAffectError("Backend must return a finite n-by-2 [valence, arousal] array.")
        return values[:, 0], values[:, 1]


def _reject_outcome_columns(frame: pd.DataFrame) -> None:
    offending = sorted(
        column
        for column in frame.columns
        if any(token in column.casefold() for token in PROHIBITED_OUTCOME_TOKENS)
    )
    if offending:
        raise TextAffectError(f"Text-affect input contains prohibited outcome columns: {offending}")


def normalize_emobank_frame(
    raw: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    dataset_release: str,
    perspective: str,
    scale_min: float,
    scale_max: float,
) -> pd.DataFrame:
    """Normalize a locally supplied EmoBank release through an explicit column map."""
    targets = {"item_id", "text", "valence", "arousal", "split"}
    missing_targets = sorted(targets - set(column_map))
    if missing_targets:
        raise TextAffectError(f"EmoBank column mapping is incomplete: {missing_targets}")
    missing_sources = sorted({column_map[target] for target in targets} - set(raw.columns))
    if missing_sources:
        raise TextAffectError(f"EmoBank source columns are unavailable: {missing_sources}")
    if dataset_release != EMOBANK_RELEASE:
        raise TextAffectError(f"Expected pinned EmoBank release {EMOBANK_RELEASE}.")
    if perspective not in {"reader", "writer"}:
        raise TextAffectError("EmoBank perspective must be explicitly reader or writer.")
    if not np.isfinite([scale_min, scale_max]).all() or scale_max <= scale_min:
        raise TextAffectError("EmoBank scale bounds must be finite and ordered.")
    _reject_outcome_columns(raw)
    output = raw.rename(columns={source: target for target, source in column_map.items()})
    output = output[sorted(targets)].copy()
    if output["item_id"].isna().any() or output["item_id"].duplicated().any():
        raise TextAffectError("EmoBank item IDs must be present and unique.")
    if output["text"].isna().any() or output["text"].astype(str).str.strip().eq("").any():
        raise TextAffectError("EmoBank text must be present and non-empty.")
    values = output[["valence", "arousal"]].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise TextAffectError("EmoBank V/A targets must be finite numeric values.")
    if values.lt(scale_min).any().any() or values.gt(scale_max).any().any():
        raise TextAffectError("EmoBank V/A targets fall outside the declared scale.")
    output[["valence", "arousal"]] = values
    allowed_splits = {"train", "dev", "test"}
    if not set(output["split"].astype(str)).issubset(allowed_splits):
        raise TextAffectError("EmoBank split must contain only train/dev/test.")
    output["dataset_release"] = dataset_release
    output["perspective"] = perspective
    output["scale_min"] = scale_min
    output["scale_max"] = scale_max
    return output


@dataclass
class MeanTextVaBaseline:
    """Transparent train-split mean baseline for continuous V/A."""

    mean_: np.ndarray | None = None
    train_ids_: frozenset[object] = frozenset()

    def fit(self, frame: pd.DataFrame) -> MeanTextVaBaseline:
        _reject_outcome_columns(frame)
        train = frame.loc[frame["split"].eq("train")]
        if len(train) < 2:
            raise TextAffectError("Mean baseline requires at least two train-split items.")
        self.mean_ = train[["valence", "arousal"]].to_numpy(float).mean(axis=0)
        self.train_ids_ = frozenset(train["item_id"])
        return self

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        if self.mean_ is None:
            raise TextAffectError("Mean baseline is not fitted.")
        if not texts or any(not str(text).strip() for text in texts):
            raise TextAffectError("Text batches must contain non-empty strings.")
        return np.tile(self.mean_, (len(texts), 1))


@dataclass
class TfidfRidgeTextRegressor:
    """Simple reproducible EmoBank V/A benchmark fitted on the published train split only."""

    alpha: float = 10.0
    max_features: int = 20_000
    model_: Pipeline | None = None
    train_ids_: frozenset[object] = frozenset()
    revision_: str | None = None

    def fit(self, frame: pd.DataFrame) -> TfidfRidgeTextRegressor:
        _reject_outcome_columns(frame)
        required = {"item_id", "text", "valence", "arousal", "split", "dataset_release"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise TextAffectError(f"Normalized EmoBank frame missing columns: {missing}")
        train = frame.loc[frame["split"].eq("train")]
        if len(train) < 4:
            raise TextAffectError("TF-IDF ridge baseline requires at least four train items.")
        if train["item_id"].duplicated().any():
            raise TextAffectError("Text training item IDs must be unique.")
        self.model_ = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=self.max_features,
                    ),
                ),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.model_.fit(train["text"].astype(str), train[["valence", "arousal"]].to_numpy(float))
        self.train_ids_ = frozenset(train["item_id"])
        record = {
            "kind": "tfidf_ridge_va",
            "alpha": self.alpha,
            "max_features": self.max_features,
            "dataset_release": sorted(set(train["dataset_release"].astype(str))),
            "train_item_count": len(train),
        }
        self.revision_ = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return self

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        if self.model_ is None:
            raise TextAffectError("TF-IDF ridge baseline is not fitted.")
        if not texts or any(not str(text).strip() for text in texts):
            raise TextAffectError("Text batches must contain non-empty strings.")
        output = np.asarray(self.model_.predict(texts), dtype=float)
        if output.shape != (len(texts), 2) or not np.isfinite(output).all():
            raise TextAffectError("Text regressor produced invalid V/A outputs.")
        return output


@dataclass
class PinnedTransformerTextRegressor:
    """Lazy two-target transformer adapter; no import-time or implicit download."""

    model_id: str
    revision: str
    device: str = "cpu"
    allow_download: bool = False
    tokenizer: Any = None
    model: Any = None

    def load(self) -> PinnedTransformerTextRegressor:
        if not self.model_id.strip() or not self.revision.strip():
            raise TextAffectError("Transformer text model ID and immutable revision are required.")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise TextAffectError("Install the optional transformer dependencies.") from exc
        local_only = not self.allow_download
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=local_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=local_only
        )
        if int(self.model.config.num_labels) != 2:
            raise TextAffectError(
                "Pinned text regression checkpoint must expose exactly two labels."
            )
        self.model.to(self.device).eval()
        return self

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise TextAffectError("Call load() explicitly before transformer inference.")
        if not texts or any(not str(text).strip() for text in texts):
            raise TextAffectError("Text batches must contain non-empty strings.")
        import torch

        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded).logits.detach().cpu().numpy()
        values = np.asarray(output, dtype=float)
        if values.shape != (len(texts), 2) or not np.isfinite(values).all():
            raise TextAffectError("Transformer text regressor produced invalid V/A outputs.")
        return values


@dataclass
class LexiconVadSensitivity:
    """Transparent sensitivity adapter; restricted lexicon contents are injected at runtime."""

    lexicon: dict[str, tuple[float, float]]
    license_record: str

    def predict_one(self, text: str) -> tuple[float, float, float]:
        tokens = [token.strip(".,!?;:\"'()[]{}").lower() for token in text.split()]
        matched = [self.lexicon[token] for token in tokens if token in self.lexicon]
        if not matched:
            return float("nan"), float("nan"), 0.0
        array: np.ndarray = np.asarray(matched, dtype=float)
        return float(array[:, 0].mean()), float(array[:, 1].mean()), len(matched) / len(tokens)


EMOBANK_TRAINING_PLAN = {
    "dataset": "JULIELab/EmoBank",
    "release": EMOBANK_RELEASE,
    "perspective": "reader",
    "targets": ["valence", "arousal"],
    "recommended_split": "published train/dev/test split stratified by text category",
    "license": EMOBANK_LICENSE,
    "prohibited_target": "DAIC-WOZ PHQ-8",
}
