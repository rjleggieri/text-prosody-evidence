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


class EvidenceBundleError(RuntimeError):
    """A required evidence artifact is absent, unsafe, or hash-inconsistent."""


def _safe_relative_path(value: str) -> PurePosixPath:
    if "\\" in value or "\x00" in value:
        raise EvidenceBundleError("Evidence paths must be canonical POSIX relative paths.")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise EvidenceBundleError("Evidence paths must be canonical POSIX relative paths.")
    return path


def validate_public_evidence_bundle(
    root: str | Path,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Validate declared files only; never fabricate, regenerate, or repair evidence."""
    missing_roles = sorted(REQUIRED_PUBLIC_EVIDENCE - set(artifacts))
    if missing_roles:
        raise EvidenceBundleError(f"Evidence bundle is missing required roles: {missing_roles}")
    base = Path(root).resolve()
    verified: dict[str, str] = {}
    seen_paths: set[str] = set()
    for role in sorted(REQUIRED_PUBLIC_EVIDENCE):
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
