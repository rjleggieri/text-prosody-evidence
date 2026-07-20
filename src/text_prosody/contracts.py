"""Dataset-neutral schema and leakage guards."""

from __future__ import annotations

from collections.abc import Iterable


class ContractError(ValueError):
    """A schema or leakage invariant failed without exposing row contents."""


def assert_disjoint_identifiers(
    train: Iterable[object], test: Iterable[object], kind: str = "group"
) -> None:
    if set(train).intersection(test):
        raise ContractError(f"{kind} leakage detected between train and test.")
