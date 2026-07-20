"""Configuration loading without data access or side effects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a run configuration is incomplete or unsafe."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigurationError("Top-level YAML value must be a mapping.")
    return value


def resolve_data_root(
    cli_value: str | None = None, env_name: str = "TEXT_PROSODY_DATA_ROOT"
) -> Path:
    raw = cli_value or os.environ.get(env_name)
    if not raw:
        raise ConfigurationError(f"Set --data-root or the {env_name} environment variable.")
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise ConfigurationError("Configured data root does not exist or is not a directory.")
    return path.resolve()
