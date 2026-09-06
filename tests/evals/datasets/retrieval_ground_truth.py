"""Small, high-value deterministic ground truth for production retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass


EXPECTED_SOURCE = "Comp_Emp_Hand.pdf"


@dataclass(frozen=True)
class ProductionRetrievalGolden:
    """Ground truth for one production retrieval query."""

    input: str
    expected_source: str
    expected_pages: frozenset[int]


# Keep this production suite intentionally small and high-signal.
# Pages are 1-based human-visible PDF pages from Comp_Emp_Hand.pdf.
production_retrieval_goldens: tuple[ProductionRetrievalGolden, ...] = (
    ProductionRetrievalGolden(
        input="How many hours per week must a full-time employee work?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({3}),
    ),
    ProductionRetrievalGolden(
        input="What is the company's policy on discrimination?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({4}),
    ),
    ProductionRetrievalGolden(
        input="What are the main steps in the company's recruitment process?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({4, 5}),
    ),
    ProductionRetrievalGolden(
        input="What types of information are considered confidential?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({7}),
    ),
    ProductionRetrievalGolden(
        input="What behaviors can be considered workplace harassment?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({8}),
    ),
    ProductionRetrievalGolden(
        input="What measures does the company take to prevent workplace injuries?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({10}),
    ),
    ProductionRetrievalGolden(
        input="Can employees use the company's internet for personal purposes?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({13}),
    ),
    ProductionRetrievalGolden(
        input="What is the company's policy on working from home?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({23}),
    ),
    ProductionRetrievalGolden(
        input="How many paid time off days do employees receive per year?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({27}),
    ),
    ProductionRetrievalGolden(
        input="What does the company offer for parental leave?",
        expected_source=EXPECTED_SOURCE,
        expected_pages=frozenset({30}),
    ),
)


if len(production_retrieval_goldens) != 10:
    raise RuntimeError(
        "Production retrieval golden suite must contain exactly 10 queries."
    )


for golden in production_retrieval_goldens:
    if not golden.input.strip():
        raise RuntimeError("Production retrieval golden input cannot be empty.")
    if not golden.expected_pages:
        raise RuntimeError(
            f"Production retrieval golden has no expected pages: {golden.input!r}"
        )
