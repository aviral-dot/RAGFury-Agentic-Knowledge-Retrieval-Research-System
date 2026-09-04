"""Shared evaluation dataset models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """Base representation of an evaluation case."""

    id: str
    input: str
    expected_output: str
