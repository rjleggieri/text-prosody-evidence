import numpy as np
import pandas as pd
import pytest

from text_prosody.text_affect import (
    ContinuousTextAdapter,
    MeanTextVaBaseline,
    TextAffectError,
    TfidfRidgeTextRegressor,
    normalize_emobank_frame,
)


def test_continuous_text_backend_contract() -> None:
    adapter = ContinuousTextAdapter(
        "fictional", "rev", "va_1_7", lambda texts: np.ones((len(texts), 2))
    )
    valence, arousal = adapter.predict_batch(["fictional sentence"])
    np.testing.assert_allclose(valence, [1.0])
    np.testing.assert_allclose(arousal, [1.0])
    with pytest.raises(TextAffectError):
        adapter.predict_batch([""])


def synthetic_emobank() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "identifier": [f"SYN_TEXT_{index:02d}" for index in range(10)],
            "sentence": [
                "fictional calm scene",
                "fictional bright scene",
                "fictional tense scene",
                "fictional quiet scene",
                "fictional joyful moment",
                "fictional difficult moment",
                "fictional neutral moment",
                "fictional active moment",
                "fictional gentle moment",
                "fictional loud moment",
            ],
            "v": [3.0, 4.5, 2.0, 3.2, 4.8, 1.5, 3.0, 3.6, 4.0, 2.4],
            "a": [2.0, 3.5, 4.5, 1.5, 4.0, 3.8, 3.0, 4.2, 2.2, 4.7],
            "partition": ["train"] * 6 + ["dev"] * 2 + ["test"] * 2,
        }
    )
    return normalize_emobank_frame(
        raw,
        column_map={
            "item_id": "identifier",
            "text": "sentence",
            "valence": "v",
            "arousal": "a",
            "split": "partition",
        },
        dataset_release="v1.4",
        perspective="reader",
        scale_min=1,
        scale_max=5,
    )


def test_emobank_contract_and_simple_baselines_are_train_only() -> None:
    frame = synthetic_emobank()
    ridge = TfidfRidgeTextRegressor(alpha=1).fit(frame)
    mean = MeanTextVaBaseline().fit(frame)
    held_out = frame.loc[frame["split"].eq("test"), "text"].tolist()
    assert ridge.predict_batch(held_out).shape == (2, 2)
    assert mean.predict_batch(held_out).shape == (2, 2)
    assert ridge.revision_ is not None and len(ridge.revision_) == 64
    assert ridge.train_ids_.isdisjoint(frame.loc[frame["split"].eq("test"), "item_id"])


def test_emobank_contract_rejects_outcome_columns() -> None:
    frame = pd.DataFrame(
        {
            "identifier": ["SYN"],
            "sentence": ["fictional"],
            "v": [3],
            "a": [3],
            "partition": ["train"],
            "phq8_score": [0],
        }
    )
    with pytest.raises(TextAffectError, match="prohibited outcome"):
        normalize_emobank_frame(
            frame,
            column_map={
                "item_id": "identifier",
                "text": "sentence",
                "valence": "v",
                "arousal": "a",
                "split": "partition",
            },
            dataset_release="v1.4",
            perspective="reader",
            scale_min=1,
            scale_max=5,
        )
