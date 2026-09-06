"""Central configuration for RAGFury evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationThresholds:
    """Minimum acceptable evaluation quality.

    These values act as regression gates. If an evaluation score falls
    below its configured threshold, the evaluation must fail.
    """

    # ------------------------------------------------------------------
    # Deterministic retrieval
    # ------------------------------------------------------------------
    retrieval_recall_at_k: float = 0.70
    retrieval_precision_at_k: float = 0.50
    retrieval_hit_rate_at_k: float = 0.90

    # ------------------------------------------------------------------
    # LLM-based retrieval
    # ------------------------------------------------------------------
    retrieval_context_relevancy: float = 0.70

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    grading_classification: float = 1.00
    grading_reason_correctness: float = 0.70

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    generation_faithfulness: float = 0.90
    generation_answer_relevancy: float = 0.85
    generation_answer_correctness: float = 0.85

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------
    rewrite_quality: float = 0.85


THRESHOLDS = EvaluationThresholds()


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASET_NAME = os.getenv(
    "EVAL_DATASET_NAME",
    "ragfury-golden",
)

DATASET_VERSION = os.getenv(
    "EVAL_DATASET_VERSION",
    "1.0.0",
)


# ---------------------------------------------------------------------------
# Evaluation profile
# ---------------------------------------------------------------------------

EVAL_PROFILE = os.getenv(
    "EVAL_PROFILE",
    "smoke",
)


def is_smoke_evaluation() -> bool:
    """Return True when running the lightweight evaluation suite."""

    return EVAL_PROFILE == "smoke"


def is_full_evaluation() -> bool:
    """Return True when running the complete evaluation suite."""

    return EVAL_PROFILE == "full"
