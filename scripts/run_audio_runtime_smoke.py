"""Technical smoke test for the pinned audEERING dimensional audio model."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from text_prosody.audio_affect import (
    AUDEERING_MODEL_ID,
    AUDEERING_OUTPUT_ORDER,
    AUDEERING_REVISION,
    AudeeringDimensionalAdapter,
    AudioAffectError,
    CheckpointLedger,
    decode_audeering_output,
    output_quality_flags,
)
from text_prosody.component_benchmark import content_hash, sha256_file


def package_versions(names: list[str]) -> dict[str, str]:
    output = {}
    for name in names:
        try:
            output[name] = version(name)
        except PackageNotFoundError:
            output[name] = "not-installed"
    return output


def raw_output(adapter: AudeeringDimensionalAdapter, signal: np.ndarray) -> np.ndarray:
    import torch

    encoded = adapter.processor(signal, sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        result = adapter.model(encoded["input_values"].to(adapter.device))[1]
    return np.asarray(result.detach().float().cpu().numpy(), dtype=float)


def run_device(
    device: str,
    signals: dict[str, np.ndarray],
    *,
    process: Any,
) -> dict[str, Any]:
    import torch

    before_rss = int(process.memory_info().rss)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter = AudeeringDimensionalAdapter(revision=AUDEERING_REVISION, device=device).load()
    load_seconds = time.perf_counter() - load_started
    after_load_rss = int(process.memory_info().rss)

    raw = raw_output(adapter, signals["sine"])
    decoded_valence, decoded_arousal = decode_audeering_output(raw)
    inference_started = time.perf_counter()
    single_valence, single_arousal = adapter.predict_batch([signals["sine"]], [16_000])
    batch_valence, batch_arousal = adapter.predict_batch(
        [signals["sine"], signals["noise"]], [16_000, 16_000]
    )
    inference_seconds = time.perf_counter() - inference_started
    silence_valence, silence_arousal = adapter.predict_batch([signals["silence"]], [16_000])
    try:
        short_valence, short_arousal = adapter.predict_batch([signals["very_short"]], [16_000])
        short_observation: dict[str, Any] = {
            "accepted": True,
            "finite": bool(np.isfinite(short_valence).all() and np.isfinite(short_arousal).all()),
        }
    except (AudioAffectError, RuntimeError, ValueError) as exc:
        short_observation = {"accepted": False, "error_type": type(exc).__name__}

    sampling_rate_guard = False
    try:
        adapter.predict_batch([signals["sine"]], [8_000])
    except AudioAffectError:
        sampling_rate_guard = True
    result = {
        "device": device,
        "processor_loaded": adapter.processor is not None,
        "model_loaded": adapter.model is not None,
        "configured_num_labels": int(adapter.model.config.num_labels),
        "load_seconds": load_seconds,
        "inference_seconds_single_plus_requested_batch": inference_seconds,
        "rss_before_bytes": before_rss,
        "rss_after_load_bytes": after_load_rss,
        "rss_load_delta_bytes": max(0, after_load_rss - before_rss),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if device == "cuda"
        else None,
        "raw_output_shape": list(raw.shape),
        "raw_output_finite": bool(np.isfinite(raw).all()),
        "raw_output_order": list(AUDEERING_OUTPUT_ORDER),
        "decode_matches_adapter": bool(
            np.allclose(decoded_valence, single_valence, atol=1e-6, rtol=1e-6)
            and np.allclose(decoded_arousal, single_arousal, atol=1e-6, rtol=1e-6)
        ),
        "single_requested_batch_equivalent": bool(
            np.allclose(single_valence, batch_valence[:1], atol=1e-6, rtol=1e-6)
            and np.allclose(single_arousal, batch_arousal[:1], atol=1e-6, rtol=1e-6)
        ),
        "maximum_single_batch_absolute_difference": float(
            max(
                np.max(np.abs(single_valence - batch_valence[:1])),
                np.max(np.abs(single_arousal - batch_arousal[:1])),
            )
        ),
        "approximate_range_flag_count": sum(bool(flag) for flag in output_quality_flags(raw)),
        "sampling_rate_guard_passed": sampling_rate_guard,
        "silence_observation": {
            "accepted": True,
            "shape": [len(silence_valence), len(silence_arousal)],
            "finite": bool(
                np.isfinite(silence_valence).all() and np.isfinite(silence_arousal).all()
            ),
        },
        "very_short_observation": short_observation,
    }
    del adapter
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "LOCKED_BEFORE_MODEL_DOWNLOAD":
        raise RuntimeError("Audio smoke configuration is not locked.")
    if config["outcome_boundary"]["daic_data_allowed"]:
        raise RuntimeError("Audio smoke test must prohibit DAIC data.")
    if config["model"]["id"] != AUDEERING_MODEL_ID:
        raise RuntimeError("Audio smoke model ID differs from the locked adapter.")
    if config["model"]["revision"] != AUDEERING_REVISION:
        raise RuntimeError("Audio smoke revision differs from the locked adapter.")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir.resolve())

    import psutil
    import torch
    from huggingface_hub import model_info

    fixture = config["fixtures"]
    sampling_rate = int(config["model"]["input_sampling_rate_hz"])
    standard_samples = int(float(fixture["standard_duration_sec"]) * sampling_rate)
    times = np.arange(standard_samples, dtype=float) / sampling_rate
    rng = np.random.default_rng(int(fixture["seed"]))
    signals = {
        "sine": (
            float(fixture["sine_amplitude"])
            * np.sin(2 * np.pi * float(fixture["sine_frequency_hz"]) * times)
        ).astype(np.float32),
        "noise": rng.normal(
            0.0, float(fixture["noise_standard_deviation"]), standard_samples
        ).astype(np.float32),
        "silence": np.zeros(standard_samples, dtype=np.float32),
        "very_short": np.zeros(
            int(float(fixture["very_short_duration_sec"]) * sampling_rate),
            dtype=np.float32,
        ),
    }
    remote = model_info(AUDEERING_MODEL_ID, revision=AUDEERING_REVISION)
    process = psutil.Process()
    device_results = [run_device("cpu", signals, process=process)]
    if torch.cuda.is_available():
        device_results.append(run_device("cuda", signals, process=process))

    ledger_path = output_dir / "checkpoint_ledger.json"
    ledger = CheckpointLedger.load(ledger_path, AUDEERING_REVISION)
    ledger.completed.update({"unrestricted_sine", "unrestricted_noise"})
    ledger.save()
    reloaded = CheckpointLedger.load(ledger_path, AUDEERING_REVISION)
    mismatch_guard = False
    try:
        CheckpointLedger.load(ledger_path, "incorrect-revision")
    except AudioAffectError:
        mismatch_guard = True

    results: dict[str, Any] = {
        "schema_version": 1,
        "status": "TECHNICAL_SMOKE_TEST_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "daic_data_used": False,
        "daic_outcomes_used": False,
        "scientific_validation": False,
        "quality_threshold_selected": False,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "model": {
            "id": AUDEERING_MODEL_ID,
            "requested_revision": AUDEERING_REVISION,
            "resolved_revision": remote.sha,
            "immutable_revision_verified": remote.sha == AUDEERING_REVISION,
            "license": config["model"]["license"],
            "source_dataset": config["model"]["source_dataset"],
            "raw_output_order": list(AUDEERING_OUTPUT_ORDER),
        },
        "fixtures": {
            "provenance": config["fixtures"]["provenance"],
            "persisted_audio_files": False,
            "standard_samples": standard_samples,
            "very_short_samples": len(signals["very_short"]),
        },
        "devices": device_results,
        "checkpoint_resume": {
            "completed_count": len(reloaded.completed),
            "same_revision_reload": reloaded.model_revision == AUDEERING_REVISION,
            "revision_mismatch_guard": mismatch_guard,
            "ledger_sha256": sha256_file(ledger_path),
        },
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else None,
        },
        "packages": package_versions(
            ["numpy", "torch", "transformers", "huggingface-hub", "psutil"]
        ),
        "interpretation_limits": [
            "Generated signals test runtime behavior only.",
            "Outputs are not scientific validation or quality-threshold evidence.",
            "An independent audio valence/arousal benchmark remains required.",
            "No DAIC material or outcome was accessed.",
        ],
    }
    results["content_sha256"] = content_hash(results)
    result_path = output_dir / "audio_runtime_results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": results["status"],
                "resolved_revision_verified": results["model"]["immutable_revision_verified"],
                "devices_checked": len(device_results),
                "result": str(result_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
