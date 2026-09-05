"""Golden dataset for end-to-end RAGFury evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class E2EGolden:
    """Single end-to-end evaluation case."""

    input: str
    expected_output: str
    expected_route: str
    expected_sources: tuple[str, ...] = ()
    should_abstain: bool = False


rag_e2e_goldens = [
    E2EGolden(
        input="How many paid time off days do employees receive per year?",
        expected_output="Employees receive 20 days of paid time off per year.",
        expected_route="rag",
        expected_sources=("employee_handbook.pdf",),
    ),
    E2EGolden(
        input="How much vacation time does an employee receive?",
        expected_output=(
            "Full-time employees receive 20 paid time-off days per calendar year."
        ),
        expected_route="rag",
        expected_sources=("employee_handbook.pdf",),
    ),
    E2EGolden(
        input="Can employees work remotely?",
        expected_output=(
            "Yes. Employees may work remotely with manager approval under the "
            "remote work policy."
        ),
        expected_route="rag",
        expected_sources=("employee_handbook.pdf",),
    ),
    E2EGolden(
        input="How far in advance should employees request leave?",
        expected_output=(
            "Employees should submit planned leave requests at least five "
            "working days in advance."
        ),
        expected_route="rag",
        expected_sources=("employee_handbook.pdf",),
    ),
    E2EGolden(
        input="How many paid time off days can employees carry over?",
        expected_output=(
            "The provided documents do not contain information about how many "
            "unused paid time off days employees can carry over."
        ),
        expected_route="rag",
        should_abstain=True,
    ),
]
