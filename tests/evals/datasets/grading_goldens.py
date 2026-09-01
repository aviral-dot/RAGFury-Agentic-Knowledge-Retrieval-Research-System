"""Golden dataset for RAG document-grader evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GradingGolden:
    """Single document-grader evaluation case."""

    input: str
    retrieved_context: list[str]
    expected_relevance: bool
    expected_reason: str


rag_grading_goldens = [
    # =========================================================
    # CLEARLY RELEVANT
    # =========================================================
    GradingGolden(
        input="How many paid time off days do employees receive per year?",
        retrieved_context=["Employees receive 20 days of paid time off per year."],
        expected_relevance=True,
        expected_reason=(
            "The retrieved document directly states the number "
            "of paid time off days employees receive."
        ),
    ),
    GradingGolden(
        input="Can employees work remotely?",
        retrieved_context=[
            "Employees may work remotely with manager approval "
            "under the remote work policy."
        ],
        expected_relevance=True,
        expected_reason=(
            "The retrieved document states that employees may "
            "work remotely with manager approval."
        ),
    ),
    GradingGolden(
        input="How far in advance should employees request leave?",
        retrieved_context=[
            "Employees should submit planned leave requests "
            "at least five working days before the requested date."
        ],
        expected_relevance=True,
        expected_reason=(
            "The retrieved document specifies the required "
            "advance notice for leave requests."
        ),
    ),
    # =========================================================
    # CLEARLY IRRELEVANT
    # =========================================================
    GradingGolden(
        input="How many paid time off days do employees receive per year?",
        retrieved_context=[
            "Corporate email accounts must only be used for "
            "legitimate business purposes."
        ],
        expected_relevance=False,
        expected_reason=(
            "The retrieved document discusses corporate email "
            "usage and contains no information about PTO entitlement."
        ),
    ),
    GradingGolden(
        input="Can employees work remotely?",
        retrieved_context=[
            "Employees must complete annual cybersecurity training "
            "before receiving access to internal systems."
        ],
        expected_relevance=False,
        expected_reason=(
            "The retrieved document discusses cybersecurity training "
            "and does not provide information about remote work."
        ),
    ),
    GradingGolden(
        input="How far in advance should employees request leave?",
        retrieved_context=[
            "Employees must use strong passwords and must not share "
            "their credentials with other employees."
        ],
        expected_relevance=False,
        expected_reason=(
            "The retrieved document discusses password security "
            "and contains no information about leave requests."
        ),
    ),
    # =========================================================
    # MIXED CONTEXT
    # =========================================================
    GradingGolden(
        input="How many paid time off days do employees receive per year?",
        retrieved_context=[
            "Corporate email accounts are intended for business use.",
            "Employees receive 20 days of paid time off per year.",
        ],
        expected_relevance=True,
        expected_reason=(
            "Although one document is unrelated, the second document "
            "contains the employee PTO entitlement."
        ),
    ),
    # =========================================================
    # SEMANTIC MATCH
    # =========================================================
    GradingGolden(
        input="How much vacation time does an employee receive?",
        retrieved_context=[
            "Full-time employees are entitled to 20 paid time-off "
            "days during each calendar year."
        ],
        expected_relevance=True,
        expected_reason=(
            "The retrieved document describes the employee's "
            "paid time-off entitlement using different wording."
        ),
    ),
    # =========================================================
    # RELATED TOPIC BUT NOT SUFFICIENT
    # =========================================================
    GradingGolden(
        input="How many paid time off days do employees receive per year?",
        retrieved_context=[
            "Employees must submit PTO requests through the company HR portal."
        ],
        expected_relevance=False,
        expected_reason=(
            "The document discusses how PTO requests are submitted "
            "but does not state the number of PTO days."
        ),
    ),
]
