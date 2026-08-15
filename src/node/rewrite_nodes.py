"""Query rewriting nodes for the Agentic RAG workflow."""

from src.state.rag_state import RAGState


class RewriteNodes:
    """Nodes responsible for improving failed retrieval queries."""

    def __init__(self, llm):
        """
        Initialize query rewriting nodes.

        Args:
            llm: Chat model used for query rewriting.
        """
        self.llm = llm

    def rewrite_query(self, state: RAGState) -> dict:
        """
        Rewrite the user's question to improve document retrieval.
        """

        print("---REWRITING QUERY---")

        question = state["question"]

        prompt = f"""
You are a query rewriting assistant for an Agentic RAG system.

The initial retrieval did not return sufficiently relevant
documents for the user's question.

Rewrite the question so that it is clearer, more specific,
and more suitable for semantic and keyword retrieval and use differt way to write so as to retrieve the documntes perfectly.


Do not answer the question.

Original question:
{question}

Return only the improved search query.
"""

        response = self.llm.invoke(prompt)

        rewritten_question = response.content.strip()

        print(f"Original query: {question}")
        print(f"Rewritten query: {rewritten_question}")

        return {
            "question": rewritten_question,
        }