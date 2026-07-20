from pathlib import Path
from typing import Any

import pytest

from text_prosody.test_access import (
    AccessLedgerError,
    record_test_access,
    verify_test_access_record,
)


def test_test_access_ledger_is_non_overwriting_and_candid_about_proof(tmp_path: Path) -> None:
    arguments: dict[str, Any] = {
        "dataset_id": "SYNTHETIC_PUBLIC",
        "dataset_version": "v1",
        "split_hash": "sha256:synthetic",
        "run_id": "synthetic-run-1",
        "purpose": "synthetic contract test",
    }
    path = record_test_access(tmp_path, **arguments)
    original = path.read_bytes()
    record = verify_test_access_record(path)
    assert record["evidence_strength"] == ("non_overwriting_local_record_not_cryptographic_proof")
    with pytest.raises(AccessLedgerError, match="already exists"):
        record_test_access(tmp_path, **arguments)
    assert path.read_bytes() == original
