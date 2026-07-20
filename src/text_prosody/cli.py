"""Side-effect-free command-line entry points for synthetic software verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .external_evaluation import (
    cross_validated_external_predictions,
    external_prediction_metrics,
    fit_frozen_external_calibration,
    grouped_bootstrap_intervals,
    make_synthetic_external_benchmark,
)
from .provenance import sha256_file, write_json


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_synthetic_reference(output: Path, seed: int, bootstrap_replicates: int) -> None:
    """Run a deterministic synthetic contract demonstration and write aggregates only."""
    output.mkdir(parents=True, exist_ok=False)
    frame = make_synthetic_external_benchmark(seed=seed)
    predictions = cross_validated_external_predictions(frame, folds=5)
    metrics = external_prediction_metrics(predictions)
    intervals = grouped_bootstrap_intervals(
        predictions,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    metric_payload: dict[str, Any] = {
        "status": "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY",
        "scientific_validation": False,
        "metrics": metrics,
        "grouped_bootstrap": intervals,
    }
    metadata = {
        "dataset_id": "SYNTHETIC_EXTERNAL",
        "dataset_version": "synthetic-fixture-v1",
        "split_hash": _canonical_hash({"seed": seed, "folds": 5}),
        "text_model_id": "synthetic-text-estimator",
        "text_model_revision": "synthetic-v1",
        "audio_model_id": "synthetic-audio-estimator",
        "audio_model_revision": "synthetic-v1",
    }
    calibration = fit_frozen_external_calibration(
        frame,
        metadata=metadata,
        configuration={
            "artifact_status": "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY",
            "scientific_validation": False,
            "real_data_index_frozen": False,
        },
    )
    calibration["status"] = "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY"
    write_json(output / "metrics.json", metric_payload)
    write_json(output / "reference_calibration.json", calibration)
    artifacts = {
        name: sha256_file(output / name) for name in ("metrics.json", "reference_calibration.json")
    }
    manifest = {
        "schema_version": "1.0",
        "status": "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY",
        "scientific_validation": False,
        "real_data_index_frozen": False,
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "item_level_outputs_persisted": False,
        "artifacts": artifacts,
    }
    write_json(output / "run_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text-prosody-evidence",
        description="Evidence-aware reference methods; synthetic verification only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser(
        "synthetic-reference",
        help="run the deterministic non-scientific reference demonstration",
    )
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--seed", type=int, default=20260720)
    synthetic.add_argument("--bootstrap-replicates", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "synthetic-reference":
        run_synthetic_reference(args.output, args.seed, args.bootstrap_replicates)
        print(
            json.dumps(
                {
                    "status": "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY",
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
