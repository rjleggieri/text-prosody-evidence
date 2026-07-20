# Contributing

Issues and pull requests are welcome. Before proposing a change, please read the evidence boundary
in the README and search existing issues.

## Development workflow

1. Fork the repository and create a focused branch.
2. Install `python -m pip install -e ".[dev]"`.
3. Add tests for behavior changes, including adversarial scientific guards where relevant.
4. Run `pytest`, `ruff check .`, `ruff format --check .`, and `mypy`.
5. Open a pull request describing the evidence level affected by the change.

Never attach restricted datasets, raw media, human-subject records, item-level real predictions,
credentials, or model files whose terms prohibit redistribution. Synthetic fixtures must be plainly
identified. A passing software test cannot be described as empirical validation.

Small documentation, test, and portability issues may be suitable for first-time contributors.
Scientific scope changes require maintainer review and a recorded rationale.
