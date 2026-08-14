"""Self-reflection nodes for the Wikipedia workflow."""

from pydantic import BaseModel, Field

from src.state.rag_state import RAGState


class ReflectionResult(BaseModel):
    """Structured output from the answer reflection model."""

    passed: bool = Field(
        description=(
            "Whether the generated answer adequately answers "
            "the user's question and is supported by the "
            "available information."
        )
    )

    reason: str = Field(
        description="Brief explanation of the reflection result."
    )


class ReflectionNodes:
    """Nodes responsible for evaluating generated answers."""

    def __init__(self, llm):
        """
        Initialize reflection nodes.

        Args:
            llm: Chat model used for answer reflection.
        """
        self.llm = llm

        self.reflector = llm.with_structured_output(
            ReflectionResult
        )

    def reflect_on_answer(self, state: RAGState) -> dict:
        """
        Evaluate the generated answer.

        The reflection checks:
            1. Whether the answer addresses the question.
            2. Whether the answer is supported by the available context.
            3. Whether the answer is sufficiently complete.
        """

        print("---REFLECTING ON ANSWER---")

        question = state["question"]
        answer = state.get("answer", "")

        prompt = f"""
You are an answer quality evaluator.

Evaluate the generated answer against the user's question.

User question:
{question}



Generated answer:
{answer}

Check:

1. Does the answer directly address the question?
2. Is the answer sufficiently complete?
3. Does the answer contain unsupported claims?

Return:
- passed=True if the answer is acceptable.
- passed=False if the answer needs improvement.

Provide a brief reason.
"""

        result = self.reflector.invoke(prompt)

        attempts = state.get("reflection_attempts", 0)

        print(f"Reflection passed: {result.passed}")
        print(f"Reason: {result.reason}")

        return {
            "reflection": result.reason,
            "reflection_passed": result.passed,
            "reflection_attempts": attempts + 1,
        }