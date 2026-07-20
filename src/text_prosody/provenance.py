"""Machine-readable provenance and content hashing."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = version(name)
        except PackageNotFoundError:
            output[name] = "not-installed"
    return output


def run_manifest(
    *,
    seed: int,
    synthetic_only: bool,
    config: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SYNTHETIC_ONLY" if synthetic_only else "RESTRICTED_REAL_DATA",
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML"]
        ),
        "config": config,
        "artifacts": artifacts,
    }


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
