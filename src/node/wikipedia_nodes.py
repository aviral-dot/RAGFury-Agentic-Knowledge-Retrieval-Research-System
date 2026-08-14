"""Wikipedia answer generation nodes."""

from src.state.rag_state import RAGState


class WikipediaNodes:
    """Nodes responsible for generating answers from Wikipedia context."""

    def __init__(self, llm):
        """
        Initialize Wikipedia nodes.

        Args:
            llm: Chat model used for answer generation.
        """
        self.llm = llm

    def generate_wikipedia_answer(self, state: RAGState) -> dict:

        print("---GENERATING WIKIPEDIA ANSWER---")

        question = state["question"]

        prompt = f"""
You are a knowledgeable assistant answering a user's question

User question:
{question}


Instructions:

1. Answer the user's question directly.
2. Do not mention internal tools, agents, retrieval, or workflow.
3. Keep the answer clear, concise, and useful.
"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
        }