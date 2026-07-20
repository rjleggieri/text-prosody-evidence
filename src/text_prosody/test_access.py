"""Non-overwriting local test-access records for future real benchmark runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


class AccessLedgerError(RuntimeError):
    """A test-access record is invalid or would overwrite prior local evidence."""


def record_test_access(
    ledger_directory: str | Path,
    *,
    dataset_id: str,
    dataset_version: str,
    split_hash: str,
    run_id: str,
    purpose: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Create one exclusive ledger entry; this is a record, not cryptographic proof."""
    required = (dataset_id, dataset_version, split_hash, run_id, purpose)
    if any(not value.strip() for value in required):
        raise AccessLedgerError("Test-access ledger fields cannot be blank.")
    identity = json.dumps(
        [dataset_id, dataset_version, split_hash, run_id],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    entry_id = hashlib.sha256(identity.encode()).hexdigest()
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "entry_id": entry_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "split_hash": split_hash,
        "run_id": run_id,
        "purpose": purpose,
        "evidence_strength": "non_overwriting_local_record_not_cryptographic_proof",
        "limitations": [
            "A local file can be deleted or the clock can be changed.",
            "This ledger does not prove that undeclared test access never occurred.",
        ],
        "metadata": dict(metadata or {}),
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    root = Path(ledger_directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{entry_id}.json"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise AccessLedgerError(
            "A test-access record for this dataset/split/run already exists."
        ) from exc
    return path


def verify_test_access_record(path: str | Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(record.pop("content_sha256", ""))
    observed = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != observed:
        raise AccessLedgerError("Test-access record content hash mismatch.")
    record["content_sha256"] = expected
    return cast(dict[str, Any], record)
