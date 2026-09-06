"""Deterministic ground truth for retrieval evaluation."""

from __future__ import annotations

import os


# The current retrieval golden set is based on the indexed employee handbook.
# Keep this configurable so the evaluation can be reused when the indexed
# source filename changes.
EXPECTED_SOURCE = os.getenv(
    "EVAL_EXPECTED_SOURCE",
    "Comp_Emp_Hand.pdf",
)


def expected_sources_for_query(_query: str) -> set[str]:
    """Return canonical source-level ground truth for a retrieval query."""

    return {EXPECTED_SOURCE}
