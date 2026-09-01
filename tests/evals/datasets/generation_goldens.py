"""Golden dataset for RAG answer-generation evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationGolden:
    """Single answer-generation evaluation case."""

    input: str
    retrieved_context: list[str]
    expected_output: str


rag_generation_goldens = [
    # =========================================================
    # CLEAR FACTUAL ANSWER
    # =========================================================
    GenerationGolden(
        input="How many paid time off days do employees receive per year?",
        retrieved_context=["Employees receive 20 days of paid time off per year."],
        expected_output=("Employees receive 20 days of paid time off per year."),
    ),
    # =========================================================
    # SEMANTIC ANSWER
    # =========================================================
    GenerationGolden(
        input="How much vacation time does an employee receive?",
        retrieved_context=[
            "Full-time employees are entitled to 20 paid time-off "
            "days during each calendar year."
        ],
        expected_output=(
            "Full-time employees receive 20 paid time-off days per calendar year."
        ),
    ),
    # =========================================================
    # POLICY QUESTION
    # =========================================================
    GenerationGolden(
        input="Can employees work remotely?",
        retrieved_context=[
            "Employees may work remotely with manager approval "
            "under the remote work policy."
        ],
        expected_output=(
            "Yes. Employees may work remotely with manager "
            "approval under the remote work policy."
        ),
    ),
    # =========================================================
    # NOTICE PERIOD
    # =========================================================
    GenerationGolden(
        input="How far in advance should employees request leave?",
        retrieved_context=[
            "Employees should submit planned leave requests "
            "at least five working days before the requested date."
        ],
        expected_output=(
            "Employees should submit planned leave requests "
            "at least five working days in advance."
        ),
    ),
    # =========================================================
    # MULTI-DOCUMENT CONTEXT
    # =========================================================
    GenerationGolden(
        input="How many paid time off days do employees receive per year?",
        retrieved_context=[
            "Corporate email accounts are intended for business use.",
            "Employees receive 20 days of paid time off per year.",
        ],
        expected_output=("Employees receive 20 days of paid time off per year."),
    ),
    # =========================================================
    # INSUFFICIENT CONTEXT
    # =========================================================
    GenerationGolden(
        input="How many paid time off days can employees carry over?",
        retrieved_context=["Employees receive 20 days of paid time off per year."],
        expected_output=(
            "The provided documents do not contain information "
            "about how many unused paid time off days employees "
            "can carry over."
        ),
    ),
]
