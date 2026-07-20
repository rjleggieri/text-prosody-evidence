from pathlib import Path

import pytest

from text_prosody.evidence_bundle import (
    REQUIRED_PUBLIC_EVIDENCE,
    EvidenceBundleError,
    validate_public_evidence_bundle,
)
from text_prosody.provenance import sha256_file


def test_evidence_bundle_verifies_every_required_artifact(tmp_path: Path) -> None:
    artifacts = {}
    for role in REQUIRED_PUBLIC_EVIDENCE:
        path = tmp_path / f"{role}.json"
        path.write_text(f'{{"role": "{role}", "status": "SYNTHETIC_ONLY"}}\n')
        artifacts[role] = {"path": path.name, "sha256": sha256_file(path)}
    verified = validate_public_evidence_bundle(tmp_path, artifacts)
    assert set(verified) == REQUIRED_PUBLIC_EVIDENCE


def test_evidence_bundle_never_fabricates_missing_or_unsafe_evidence(tmp_path: Path) -> None:
    with pytest.raises(EvidenceBundleError, match="missing required roles"):
        validate_public_evidence_bundle(tmp_path, {})
    assert list(tmp_path.iterdir()) == []

    artifacts = {}
    for role in REQUIRED_PUBLIC_EVIDENCE:
        path = tmp_path / f"{role}.json"
        path.write_text("{}\n")
        artifacts[role] = {"path": path.name, "sha256": sha256_file(path)}
    artifacts["component_results"]["path"] = "../component_results.json"
    with pytest.raises(EvidenceBundleError, match="canonical POSIX relative"):
        validate_public_evidence_bundle(tmp_path, artifacts)
