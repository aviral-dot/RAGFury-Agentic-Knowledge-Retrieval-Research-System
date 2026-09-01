"""DeepEval metrics for RAG answer-generation evaluation."""

from deepeval.metrics import (
    # AnswerRelevancyMetric,
    FaithfulnessMetric,
    # GEval,
)

# from deepeval.test_case import (
#     # SingleTurnParams,
# )
from tests.evals.helpers.eval_models import create_eval_model

eval_model = create_eval_model()


def get_generation_metrics():
    """
    Return metrics used to evaluate the RAG generation component.

    Metrics:

    1. AnswerCorrectness:
       Evaluates whether the generated answer correctly answers
       the question according to the expected answer.

    2. Faithfulness:
       Evaluates whether the generated answer is supported by
       the retrieved context.

    3. AnswerRelevancy:
       Evaluates whether the generated answer directly addresses
       the user's question.
    """

    # answer_correctness = GEval(
    #     name="AnswerCorrectness",
    #     criteria=(
    #         "Evaluate whether the generated answer correctly answers "
    #         "the user's question. Compare the generated answer with "
    #         "the expected answer. Minor differences in wording are "
    #         "acceptable when the meaning is equivalent. Penalize "
    #         "incorrect facts, missing important information, or "
    #         "answers that contradict the expected answer."
    #     ),
    #     evaluation_params=[
    #         SingleTurnParams.INPUT,
    #         SingleTurnParams.ACTUAL_OUTPUT,
    #         SingleTurnParams.EXPECTED_OUTPUT,
    #     ],
    #     threshold=0.7,
    #     model=eval_model,
    # )

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=eval_model,
        include_reason=True,
    )

    # answer_relevancy = AnswerRelevancyMetric(
    #     threshold=0.7,
    #     model=eval_model,
    #     include_reason=True,
    # )

    return [
        # answer_correctness,
        faithfulness,
        # answer_relevancy,
    ]
