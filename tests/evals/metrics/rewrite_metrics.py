"""DeepEval metrics for RAG query-rewrite evaluation."""

from deepeval.metrics import GEval
from deepeval.test_case import (
    SingleTurnParams,
)

from tests.evals.helpers.eval_models import create_eval_model

eval_model = create_eval_model()


def get_rewrite_metrics():
    """
    Return metrics used to evaluate the RAG query-rewrite component.

    The metric evaluates whether the rewritten query improves the
    original search query while preserving the user's information need.
    """

    rewrite_quality = GEval(
        name="RewriteQuality",
        criteria=(
            "Evaluate whether the rewritten search query is a better "
            "retrieval query than the original query. "
            "The rewritten query must preserve the original information "
            "need and should address the reason the previous retrieval "
            "failed. "
            "It should be specific enough to improve retrieval and may "
            "use useful terminology or keywords from the retrieved "
            "candidates when appropriate. "
            "It must not answer the user's question. "
            "It must not introduce unsupported facts or change the "
            "user's intended question. "
            "Minor wording differences are acceptable."
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
        rewrite_quality,
    ]
