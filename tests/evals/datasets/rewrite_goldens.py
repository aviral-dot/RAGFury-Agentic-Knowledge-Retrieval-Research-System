"""Golden dataset for RAG query-rewrite evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteGolden:
    """Golden test case for query rewriting."""

    input: str
    retrieved_context: list[str]
    grade_reason: str
    expected_output: str


rag_rewrite_goldens = [
    RewriteGolden(
        input="How many paid time off days do employees receive?",
        retrieved_context=[
            (
                "Employees may request different types of leave. "
                "Managers are responsible for approving leave requests."
            ),
        ],
        grade_reason=(
            "The retrieved document discusses leave requests but does "
            "not contain the number of paid time off days employees receive."
        ),
        expected_output=("number of paid time off days employees receive per year"),
    ),
    RewriteGolden(
        input="What is the company's remote work policy?",
        retrieved_context=[
            (
                "Employees must submit vacation requests through the "
                "HR portal at least two weeks before their planned leave."
            ),
        ],
        grade_reason=(
            "The retrieved document describes vacation requests rather "
            "than the company's remote work policy."
        ),
        expected_output=("company remote work policy for employees"),
    ),
    RewriteGolden(
        input="How much parental leave is available?",
        retrieved_context=[
            (
                "Employees receive twenty days of paid time off per year. "
                "Unused PTO may be carried forward according to company policy."
            ),
        ],
        grade_reason=(
            "The retrieved document contains information about PTO but "
            "does not provide the company's parental leave entitlement."
        ),
        expected_output=("parental leave entitlement and duration for employees"),
    ),
]
