"""DeepEval metrics for RAG retrieval evaluation."""

from deepeval.metrics import (
    ContextualRelevancyMetric,
)

from tests.evals.helpers.eval_models import create_eval_model

eval_model = create_eval_model()


def get_retrieval_metrics():
    """
    Return the metrics used to evaluate
    the RAG retrieval component.
    """

    return [
        ContextualRelevancyMetric(
            threshold=0.2,
            model=eval_model,
            include_reason=True,
        ),
        # ContextualRecallMetric(
        #     threshold=0.7,
        #     model = eval_model,
        #     include_reason=True,
        # ),
        # ContextualPrecisionMetric(
        #     threshold=0.7,
        #     model = eval_model,
        #     include_reason=True,
        # ),
    ]
