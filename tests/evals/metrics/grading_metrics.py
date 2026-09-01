"""DeepEval metrics for RAG document-grader evaluation."""

from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.evals.helpers.eval_models import create_eval_model

eval_model = create_eval_model()


class GraderClassificationMetric(BaseMetric):
    """
    Deterministically evaluate the grader's True/False decision.

    Expected output:
        relevant=true
        reason=...

    Actual output:
        relevant=true
        reason=...
    """

    def __init__(
        self,
        threshold: float = 1.0,
    ):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""
        self.error = None

    @property
    def __name__(self) -> str:
        return "GraderClassification"

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = False,
        _in_component: bool = False,
    ) -> float:
        """Compare actual and expected relevance."""

        try:
            expected = self._extract_relevance(
                test_case.expected_output,
            )

            actual = self._extract_relevance(
                test_case.actual_output,
            )

            if expected is None:
                raise ValueError(
                    "Could not extract expected relevance "
                    f"from: {test_case.expected_output!r}"
                )

            if actual is None:
                raise ValueError(
                    "Could not extract actual relevance "
                    f"from: {test_case.actual_output!r}"
                )

            self.score = 1.0 if actual == expected else 0.0

            self.success = self.score >= self.threshold

            self.reason = f"Expected relevance={expected}; actual relevance={actual}."

            self.error = None

            return self.score

        except Exception as exc:
            self.error = str(exc)
            self.score = 0.0
            self.success = False
            self.reason = str(exc)
            raise

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = False,
        _in_component: bool = False,
    ) -> float:
        """Async version of the deterministic metric."""

        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def is_successful(self) -> bool:
        """Return whether the metric passed."""

        if self.error is not None:
            return False

        self.success = self.score >= self.threshold

        return self.success

    @staticmethod
    def _extract_relevance(
        output: str | None,
    ) -> bool | None:
        """Extract relevant=true/false from evaluator output."""

        if not output:
            return None

        text = str(output).strip().lower()

        for line in text.splitlines():
            line = line.strip()

            if line == "relevant=true":
                return True

            if line == "relevant=false":
                return False

        return None


def get_grading_metrics():
    """
    Return metrics used to evaluate the document grader.

    1. GraderClassification:
       Exact True/False correctness.

    2. GraderReasonCorrectness:
       LLM-based evaluation of the explanation.
    """

    classification_metric = GraderClassificationMetric(
        threshold=1.0,
    )

    reason_metric = GEval(
        name="GraderReasonCorrectness",
        criteria=(
            "Evaluate whether the document grader's explanation "
            "correctly justifies its relevance decision. "
            "The explanation must be grounded only in the supplied "
            "user question and retrieved context. "
            "If the expected relevance is true, the explanation "
            "should identify useful information in at least one "
            "retrieved document. "
            "If the expected relevance is false, the explanation "
            "should correctly explain why the retrieved documents "
            "do not contain useful information for answering the "
            "question. "
            "The explanation must not invent information that is "
            "not present in the retrieved context."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=eval_model,
    )

    return [
        classification_metric,
        reason_metric,
    ]
