"""Validation-only contract for public benchmark evidence bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .provenance import sha256_file

REQUIRED_PUBLIC_EVIDENCE = {
    "component_results",
    "test_predictions",
    "split_manifest",
    "development_histories",
    "audio_runtime_results",
    "configuration_hashes",
    "environment_lock",
    "model_hashes",
}

REQUIRED_PUBLIC_COMPONENT_DEMO_EVIDENCE = {
    "component_results",
    "split_manifest",
    "development_histories",
    "human_reference_contract",
    "configuration_hashes",
    "environment_lock",
    "model_specification_hashes",
    "run_provenance",
}


class EvidenceBundleError(RuntimeError):
    """A required evidence artifact is absent, unsafe, or hash-inconsistent."""


def _safe_relative_path(value: str) -> PurePosixPath:
    if "\\" in value or "\x00" in value:
        raise EvidenceBundleError("Evidence paths must be canonical POSIX relative paths.")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise EvidenceBundleError("Evidence paths must be canonical POSIX relative paths.")
    return path


def _validate_evidence_bundle(
    root: str | Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    required_roles: set[str],
) -> dict[str, str]:
    """Validate declared files only; never fabricate, regenerate, or repair evidence."""
    missing_roles = sorted(required_roles - set(artifacts))
    if missing_roles:
        raise EvidenceBundleError(f"Evidence bundle is missing required roles: {missing_roles}")
    unexpected_roles = sorted(set(artifacts) - required_roles)
    if unexpected_roles:
        raise EvidenceBundleError(f"Evidence bundle has unexpected roles: {unexpected_roles}")
    base = Path(root).resolve()
    verified: dict[str, str] = {}
    seen_paths: set[str] = set()
    for role in sorted(required_roles):
        record = artifacts[role]
        relative = _safe_relative_path(str(record.get("path", "")))
        expected = str(record.get("sha256", "")).lower()
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise EvidenceBundleError(f"Evidence role has an invalid SHA-256: {role}")
        if relative.as_posix().casefold() in seen_paths:
            raise EvidenceBundleError("Two evidence roles resolve to the same case-folded path.")
        seen_paths.add(relative.as_posix().casefold())
        target = (base / Path(*relative.parts)).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise EvidenceBundleError("Evidence artifact escapes the declared root.") from exc
        if not target.is_file():
            raise EvidenceBundleError(f"Evidence artifact is missing for role: {role}")
        observed = sha256_file(target)
        if observed != expected:
            raise EvidenceBundleError(f"Evidence artifact hash mismatch for role: {role}")
        verified[role] = observed
    return verified


def validate_public_evidence_bundle(
    root: str | Path,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Validate the complete external-benchmark evidence profile."""
    return _validate_evidence_bundle(root, artifacts, REQUIRED_PUBLIC_EVIDENCE)


def validate_public_component_demo_bundle(
    root: str | Path,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Validate the aggregate-only public component-demonstration profile."""
    return _validate_evidence_bundle(
        root,
        artifacts,
        REQUIRED_PUBLIC_COMPONENT_DEMO_EVIDENCE,
    )
