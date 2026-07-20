"""Direct dimensional audio-affect adapter with lazy optional dependencies."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

AUDEERING_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
AUDEERING_REVISION = "6eba34a2485ea31cb03600241787c3a5edab8626"
AUDEERING_OUTPUT_ORDER = ("arousal", "dominance", "valence")
TARGET_SAMPLING_RATE = 16_000


class AudioAffectError(RuntimeError):
    """Audio-affect inference failed a scientific or runtime guard."""


def decode_audeering_output(raw_output: object) -> tuple[np.ndarray, np.ndarray]:
    """Return valence and arousal from the documented A-D-V regression order."""
    values = np.asarray(raw_output, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise AudioAffectError("Expected an n-by-3 regression output in A-D-V order.")
    if not np.isfinite(values).all():
        raise AudioAffectError("Audio regression output contains non-finite values.")
    return values[:, 2], values[:, 0]


def output_quality_flags(raw_output: object) -> list[str]:
    values = np.asarray(raw_output, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise AudioAffectError("Expected an n-by-3 regression output in A-D-V order.")
    return [
        "outside_approximate_0_1_range" if (row < 0).any() or (row > 1).any() else ""
        for row in values
    ]


def validate_audio_batch(signals: list[np.ndarray], sampling_rates: list[int]) -> None:
    if len(signals) != len(sampling_rates) or not signals:
        raise AudioAffectError("Signals and sampling rates must be aligned and non-empty.")
    if any(rate != TARGET_SAMPLING_RATE for rate in sampling_rates):
        raise AudioAffectError("Inputs must be resampled to exactly 16 kHz before inference.")
    if any(signal.ndim != 1 or signal.size == 0 for signal in signals):
        raise AudioAffectError("Each audio input must be a non-empty mono array.")


def batch_slices(total: int, batch_size: int, start: int = 0) -> list[slice]:
    if total < 0 or batch_size <= 0 or start < 0 or start > total:
        raise AudioAffectError("Invalid deterministic batching parameters.")
    return [
        slice(index, min(index + batch_size, total)) for index in range(start, total, batch_size)
    ]


def predict_in_batches(
    predictor: Callable[[list[np.ndarray], list[int]], tuple[np.ndarray, np.ndarray]],
    signals: list[np.ndarray],
    sampling_rates: list[int],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic batches and verify equivalence-compatible output contracts."""
    validate_audio_batch(signals, sampling_rates)
    valence_blocks: list[np.ndarray] = []
    arousal_blocks: list[np.ndarray] = []
    for batch in batch_slices(len(signals), batch_size):
        valence, arousal = predictor(signals[batch], sampling_rates[batch])
        valence = np.asarray(valence, dtype=float)
        arousal = np.asarray(arousal, dtype=float)
        expected = batch.stop - batch.start
        if (
            valence.shape != (expected,)
            or arousal.shape != (expected,)
            or not np.isfinite(valence).all()
            or not np.isfinite(arousal).all()
        ):
            raise AudioAffectError("Audio predictor returned invalid batched V/A outputs.")
        valence_blocks.append(valence)
        arousal_blocks.append(arousal)
    return np.concatenate(valence_blocks), np.concatenate(arousal_blocks)


@dataclass
class CheckpointLedger:
    """Atomic local resume ledger; real ledgers are restricted participant-level artifacts."""

    path: Path
    model_revision: str
    completed: set[str]

    @classmethod
    def load(cls, path: str | Path, model_revision: str) -> CheckpointLedger:
        target = Path(path)
        if not target.exists():
            return cls(target, model_revision, set())
        value = json.loads(target.read_text(encoding="utf-8"))
        if value.get("model_revision") != model_revision:
            raise AudioAffectError(
                "Checkpoint model revision does not match the requested revision."
            )
        return cls(target, model_revision, set(value.get("completed", [])))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "model_revision": self.model_revision,
                    "completed": sorted(self.completed),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


LEGACY_LABEL_ALIASES = {
    "neu": "neutral",
    "neutral": "neutral",
    "hap": "happy",
    "happy": "happy",
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "sad": "sad",
    "sadness": "sad",
}
LEGACY_VA = {
    "neutral": (0.0, 0.1),
    "happy": (0.8, 0.6),
    "angry": (-0.7, 0.8),
    "sad": (-0.8, 0.2),
}


def legacy_categorical_sensitivity(scores: list[dict[str, Any]]) -> tuple[float, float]:
    """Explicit sensitivity-only mapping that fails loudly on unknown labels."""
    if not scores:
        raise AudioAffectError("Categorical scores are empty.")
    total = valence = arousal = 0.0
    for item in scores:
        raw = str(item.get("label", "")).strip().lower()
        if raw not in LEGACY_LABEL_ALIASES:
            raise AudioAffectError(f"Unknown categorical emotion label: {raw!r}")
        probability = float(item["score"])
        canonical = LEGACY_LABEL_ALIASES[raw]
        v_value, a_value = LEGACY_VA[canonical]
        valence += probability * v_value
        arousal += probability * a_value
        total += probability
    if total <= 0:
        raise AudioAffectError("Categorical probabilities must sum to a positive value.")
    return valence / total, arousal / total


@dataclass
class AudeeringDimensionalAdapter:
    """Model-card-compatible Wav2Vec2 regression adapter.

    The revision is mandatory for real inference. Imports and downloads happen only in ``load``.
    """

    revision: str
    device: str = "cpu"
    model_id: str = AUDEERING_MODEL_ID
    processor: Any = None
    model: Any = None

    def load(self) -> AudeeringDimensionalAdapter:
        if not self.revision:
            raise AudioAffectError("Pin an immutable Hugging Face model revision before inference.")
        try:
            import torch
            import torch.nn as nn
            from transformers import Wav2Vec2Processor
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                Wav2Vec2Model,
                Wav2Vec2PreTrainedModel,
            )
        except ImportError as exc:
            raise AudioAffectError("Install the optional 'audio' dependencies.") from exc

        class RegressionHead(nn.Module):
            def __init__(self, config: Any) -> None:
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features: Any) -> Any:
                value = self.dropout(features)
                value = torch.tanh(self.dense(value))
                return self.out_proj(self.dropout(value))

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.post_init()

            def forward(self, input_values: Any) -> tuple[Any, Any]:
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return hidden, self.classifier(hidden)

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, revision=self.revision)
        self.model = EmotionModel.from_pretrained(self.model_id, revision=self.revision)
        self.model.to(self.device).eval()
        return self

    def predict_batch(
        self, signals: list[np.ndarray], sampling_rates: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.processor is None:
            raise AudioAffectError("Call load() explicitly before inference.")
        validate_audio_batch(signals, sampling_rates)
        import torch

        # The official reference implementation pools time steps without an attention mask.
        # Process each variable-length clip independently so zero padding cannot change predictions.
        rows: list[np.ndarray] = []
        for signal in signals:
            encoded = self.processor(
                signal,
                sampling_rate=TARGET_SAMPLING_RATE,
                return_tensors="pt",
            )
            with torch.no_grad():
                raw = self.model(encoded["input_values"].to(self.device))[1]
            rows.append(np.asarray(raw.detach().cpu().numpy(), dtype=float).reshape(1, 3))
        return decode_audeering_output(np.concatenate(rows, axis=0))


def model_metadata(revision: str, device: str) -> dict[str, object]:
    return {
        "model_id": AUDEERING_MODEL_ID,
        "model_revision": revision,
        "license": "CC-BY-NC-SA-4.0",
        "source_dataset": "MSP-Podcast v1.7",
        "device": device,
        "sampling_rate_hz": TARGET_SAMPLING_RATE,
        "raw_output_order": list(AUDEERING_OUTPUT_ORDER),
        "approximate_output_range": [0.0, 1.0],
    }
