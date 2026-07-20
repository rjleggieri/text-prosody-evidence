# Installation

## Supported Python

The core package is tested on CPython 3.10, 3.11, and 3.12 on GitHub Actions. A fresh virtual
environment is recommended:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The core and synthetic demonstration do not require model weights, audio files, gated data, or GPU
access. Optional benchmark dependencies can be installed with `.[benchmark]`; doing so neither
downloads a dataset nor grants rights to one. Review upstream licenses before explicit model access.

`requirements-lock.txt` records the verified Windows development environment. It is an audit aid,
not the cross-platform dependency declaration; portable bounds are in `pyproject.toml`.
