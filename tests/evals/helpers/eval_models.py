import os

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


class GroqEvalModel(DeepEvalBaseLLM):
    """
    Groq LLM used by DeepEval as the evaluation/judge model.

    This model is separate from the application model configuration.
    """

    def __init__(self) -> None:
        self.model = ChatOpenAI(
            model=os.getenv(
                "DEEPEVAL_MODEL",
                "qwen/qwen3.8-27b",
            ),
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            max_tokens=1024,
            temperature=0,
            timeout=60,
            max_retries=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return "Groq qwen/qwen3.8-27b Evaluation Model"


def create_eval_model() -> DeepEvalBaseLLM:
    """
    Create the DeepEval evaluation/judge model.
    """

    return GroqEvalModel()
