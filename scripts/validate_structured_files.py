"""Validate repository YAML and JSON without executing arbitrary content."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*")):
        if any(part.startswith(".venv") or part == ".git" for part in path.parts):
            continue
        if path.suffix.lower() in {".yaml", ".yml"} or path.name == "CITATION.cff":
            with path.open(encoding="utf-8") as handle:
                yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as handle:
                json.load(handle)


if __name__ == "__main__":
    main()
