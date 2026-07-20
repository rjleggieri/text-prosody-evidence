# Development setup

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy
```

Validate structured files:

```bash
python scripts/validate_structured_files.py
python scripts/check_markdown_links.py
```

Run the integration demonstration in a new directory:

```bash
text-prosody-evidence synthetic-reference --output outputs/dev-check
```

Tests must use synthetic fixtures or unrestricted public metadata. Do not add raw media, corpus
rows, real item identifiers or predictions, participant information, gated files, embeddings,
browser state, tokens, or private paths. Network and model loading must never occur at import.

For scientific guard changes, include an adversarial test that demonstrates the previous failure
mode. Use identical seed sets when comparing development candidates, keep outer-test folds isolated,
and report invalid bootstrap replicates.
