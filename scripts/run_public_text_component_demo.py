"""Run the issue-approved, aggregate-only public text-component demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from text_prosody.evidence_bundle import validate_public_component_demo_bundle
from text_prosody.provenance import sha256_file
from text_prosody.public_demo import (
    load_emobank_reader_development_pool,
    run_grouped_text_component_demo,
)
from text_prosody.reproducibility import set_deterministic_seed


def _json_write(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _package_versions(names: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = version(name)
        except PackageNotFoundError:
            output[name] = "not-installed"
    return output


def _git_revision(repository: Path) -> str:
    head_path = repository / ".git" / "HEAD"
    head = head_path.read_text(encoding="utf-8").strip()
    if len(head) == 40 and all(character in "0123456789abcdef" for character in head):
        return head
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _peak_process_memory_bytes() -> tuple[int | None, str]:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if success and counters.PeakWorkingSetSize > 0:
            return int(counters.PeakWorkingSetSize), "Windows PeakWorkingSetSize"
    return None, "unsupported_on_this_platform"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--emobank-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "LOCKED_ISSUE_APPROVED":
        raise RuntimeError("The public demonstration configuration is not issue-approved.")
    if config["dataset"]["prohibited_partition"] != "test":
        raise RuntimeError("The public demonstration must prohibit the EmoBank test partition.")
    if config["claim_boundary"]["daic_or_phq_use"] != "prohibited":
        raise RuntimeError("The public demonstration must prohibit DAIC and PHQ use.")
    if bool(config["evaluation"]["persist_item_predictions"]):
        raise RuntimeError("The public demonstration must not persist item predictions.")

    emobank_root = Path(args.emobank_root)
    observed_revision = _git_revision(emobank_root)
    if observed_revision != config["dataset"]["repository_revision"]:
        raise RuntimeError("The local EmoBank checkout does not match the locked revision.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    seed = int(config["evaluation"]["bootstrap_seed"])
    set_deterministic_seed(seed)
    started = time.perf_counter()
    frame, data_audit = load_emobank_reader_development_pool(emobank_root / "corpus")
    result = run_grouped_text_component_demo(
        frame,
        outer_folds=int(config["evaluation"]["outer_group_folds"]),
        inner_folds=int(config["evaluation"]["inner_group_folds"]),
        alpha_candidates=tuple(float(value) for value in config["evaluation"]["alpha_candidates"]),
        max_features=int(config["evaluation"]["max_tfidf_features"]),
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=seed,
    )
    runtime_seconds = time.perf_counter() - started
    peak_memory_bytes, peak_memory_method = _peak_process_memory_bytes()

    component_results = {
        **result.component_results,
        "dataset": {
            "id": config["dataset"]["id"],
            "release": config["dataset"]["release"],
            "repository_revision": observed_revision,
            "license": config["dataset"]["license"],
            "audit": data_audit,
        },
    }
    _json_write(output / "component_results.json", component_results)
    _json_write(output / "split_manifest.json", result.split_manifest)
    _json_write(output / "development_histories.json", result.development_histories)

    human_reference_contract = {
        "schema_version": 1,
        "dataset_id": config["dataset"]["id"],
        "dataset_release": config["dataset"]["release"],
        "repository_revision": observed_revision,
        "license": config["dataset"]["license"],
        "annotation_perspective": "reader",
        "human_reference_fields": config["dataset"]["human_reference_fields"],
        "scale": config["dataset"]["scale"],
        "ratings_in_release": "aggregate_only",
        "published_annotation_design": {
            "ratings_per_sentence_before_filtering": 5,
            "dimensions": ["valence", "arousal", "dominance"],
            "separate_perspectives": ["writer", "reader"],
            "aggregation": (
                "Mean of remaining valid ratings after the published removal rule; at least two "
                "remaining judgments required."
            ),
        },
        "published_reader_reliability": {
            "metric": "mean per-rater agreement with aggregate reference",
            "valence_pearson_r": 0.738,
            "arousal_pearson_r": 0.595,
            "valence_mae": 0.349,
            "arousal_mae": 0.441,
            "source": "Buechel and Hahn (2017), EACL, Table 2",
            "not_recomputed_from_release_aggregates": True,
        },
        "rater_level_bootstrap_available": False,
        "reliability_fallback": (
            "The public v1.4 aggregate tables used by this demonstration do not expose raw "
            "rater-by-item observations, so ICC, split-half reliability, rater bootstrap, and a "
            "run-specific noise ceiling are not estimated."
        ),
        "grouping": {
            "available_field": config["dataset"]["grouping_field"],
            "speaker_creator_duplicate_fields_available": False,
            "generalization_unit": "source_document",
        },
    }
    _json_write(output / "human_reference_contract.json", human_reference_contract)

    project_root = Path(__file__).resolve().parents[1]
    code_paths = [
        Path("scripts/run_public_text_component_demo.py"),
        Path("src/text_prosody/component_benchmark.py"),
        Path("src/text_prosody/evidence_bundle.py"),
        Path("src/text_prosody/external_evaluation.py"),
        Path("src/text_prosody/public_demo.py"),
        Path("src/text_prosody/reproducibility.py"),
        Path("src/text_prosody/text_affect.py"),
    ]
    configuration_hashes = {
        "schema_version": 1,
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "canonical_config_sha256": _canonical_hash(config),
        "source_code_normalized_lf_sha256": {
            path.as_posix(): _normalized_text_sha256(project_root / path) for path in code_paths
        },
    }
    _json_write(output / "configuration_hashes.json", configuration_hashes)

    model_specification_hashes = {
        "schema_version": 1,
        "training_fold_mean_specification_sha256": _canonical_hash(
            {"type": "training_fold_mean", "targets": ["valence", "arousal"]}
        ),
        "tfidf_ridge_candidate_specification_sha256": _canonical_hash(
            {
                "type": "tfidf_ridge",
                "lowercase": True,
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_features": config["evaluation"]["max_tfidf_features"],
                "alpha_candidates": config["evaluation"]["alpha_candidates"],
            }
        ),
        "trained_model_weights_persisted": False,
        "note": "These are hashes of locked estimator specifications, not model-weight hashes.",
    }
    _json_write(output / "model_specification_hashes.json", model_specification_hashes)

    source_lock = project_root / "requirements-public-demo-lock.txt"
    shutil.copyfile(source_lock, output / "environment_lock.txt")
    run_provenance = {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _package_versions(
            [
                "joblib",
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "PyYAML",
                "text-prosody-evidence",
            ]
        ),
        "runtime_seconds": runtime_seconds,
        "peak_process_memory_bytes": peak_memory_bytes,
        "peak_memory_measurement_method": peak_memory_method,
        "cpu_only": True,
        "network_required_during_analysis": False,
        "item_predictions_persisted": False,
        "dataset_rows_or_text_persisted": False,
    }
    _json_write(output / "run_provenance.json", run_provenance)

    role_paths = {
        "component_results": "component_results.json",
        "split_manifest": "split_manifest.json",
        "development_histories": "development_histories.json",
        "human_reference_contract": "human_reference_contract.json",
        "configuration_hashes": "configuration_hashes.json",
        "environment_lock": "environment_lock.txt",
        "model_specification_hashes": "model_specification_hashes.json",
        "run_provenance": "run_provenance.json",
    }
    artifacts = {
        role: {"path": path, "sha256": sha256_file(output / path)}
        for role, path in role_paths.items()
    }
    verified = validate_public_component_demo_bundle(output, artifacts)
    manifest = {
        "schema_version": 1,
        "profile": "aggregate_public_component_demonstration",
        "artifacts": artifacts,
        "validation": {
            "status": "VALID",
            "verified_role_count": len(verified),
            "validator": "text_prosody.evidence_bundle.validate_public_component_demo_bundle",
        },
    }
    _json_write(output / "evidence_manifest.json", manifest)


if __name__ == "__main__":
    main()
