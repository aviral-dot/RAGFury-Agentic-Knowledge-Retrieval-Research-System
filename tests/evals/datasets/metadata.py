"""Metadata describing the RAGFury evaluation dataset."""

from __future__ import annotations

from tests.evals.config import DATASET_NAME, DATASET_VERSION

DATASET_DESCRIPTION = (
    "Golden evaluation dataset for RAGFury covering "
    "retrieval, grading, generation, and query rewriting."
)


def get_dataset_metadata() -> dict[str, str]:
    """Return metadata identifying the evaluation dataset."""

    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "description": DATASET_DESCRIPTION,
    }
