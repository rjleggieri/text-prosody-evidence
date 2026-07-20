import numpy as np
import pytest

from text_prosody.audio_affect import (
    AUDEERING_REVISION,
    AudioAffectError,
    CheckpointLedger,
    batch_slices,
    decode_audeering_output,
    legacy_categorical_sensitivity,
    model_metadata,
    output_quality_flags,
    predict_in_batches,
    validate_audio_batch,
)


def test_documented_audio_output_order_is_locked() -> None:
    valence, arousal = decode_audeering_output([[0.2, 0.4, 0.8]])
    np.testing.assert_allclose(valence, [0.8])
    np.testing.assert_allclose(arousal, [0.2])
    assert output_quality_flags([[0.2, 0.4, 0.8], [-0.1, 0.4, 1.2]]) == [
        "",
        "outside_approximate_0_1_range",
    ]


def test_legacy_aliases_are_normalized_and_unknowns_fail() -> None:
    v_neu, a_neu = legacy_categorical_sensitivity([{"label": "neu", "score": 1.0}])
    v_hap, a_hap = legacy_categorical_sensitivity([{"label": "hap", "score": 1.0}])
    v_ang, a_ang = legacy_categorical_sensitivity([{"label": "ang", "score": 1.0}])
    assert (v_neu, a_neu) == (0.0, 0.1)
    assert (v_hap, a_hap) == (0.8, 0.6)
    assert (v_ang, a_ang) == (-0.7, 0.8)
    with pytest.raises(AudioAffectError, match="Unknown"):
        legacy_categorical_sensitivity([{"label": "mystery", "score": 1.0}])


def test_sampling_rate_batching_checkpoint_and_resume(tmp_path) -> None:
    validate_audio_batch([np.zeros(160)], [16_000])
    with pytest.raises(AudioAffectError, match="16 kHz"):
        validate_audio_batch([np.zeros(160)], [8_000])
    assert batch_slices(10, 4) == [slice(0, 4), slice(4, 8), slice(8, 10)]

    path = tmp_path / "checkpoint.json"
    ledger = CheckpointLedger.load(path, "revision-a")
    ledger.completed.update({"SYNTHETIC_002", "SYNTHETIC_001"})
    ledger.save()
    resumed = CheckpointLedger.load(path, "revision-a")
    assert resumed.completed == {"SYNTHETIC_001", "SYNTHETIC_002"}
    with pytest.raises(AudioAffectError, match="revision"):
        CheckpointLedger.load(path, "revision-b")


def test_pinned_revision_and_batching_equivalence_with_synthetic_audio() -> None:
    signals = [np.full(160 + index, index / 10, dtype=float) for index in range(1, 6)]
    rates = [16_000] * len(signals)

    def synthetic_predictor(
        batch_signals: list[np.ndarray], batch_rates: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        validate_audio_batch(batch_signals, batch_rates)
        return (
            np.asarray([signal.mean() for signal in batch_signals]),
            np.asarray([signal.std() for signal in batch_signals]),
        )

    one, one_arousal = predict_in_batches(synthetic_predictor, signals, rates, batch_size=1)
    many, many_arousal = predict_in_batches(synthetic_predictor, signals, rates, batch_size=3)
    np.testing.assert_allclose(one, many)
    np.testing.assert_allclose(one_arousal, many_arousal)
    assert model_metadata(AUDEERING_REVISION, "cpu")["model_revision"] == AUDEERING_REVISION
