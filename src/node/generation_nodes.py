"""RAG answer generation nodes."""

from src.state.rag_state import RAGState


class GenerationNodes:
    """Nodes responsible for generating answers from retrieved documents."""

    def __init__(self, llm):
        """
        Initialize generation nodes.

        Args:
            llm: Chat model used for answer generation.
        """
        self.llm = llm

    def generate_answer(self, state: RAGState) -> dict:

        print("---GENERATING RAG ANSWER---")

        question = state["question"]
        documents = state.get("retrieved_docs", [])

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        prompt = f"""
You are a knowledgeable assistant answering a user's question
using the provided retrieved documents.

User question:
{question}

Retrieved documents:
{context}

Instructions:

1. Answer the user's question directly.
2. Use the retrieved documents as the primary source of information.
3. Do not invent information that is not supported by the documents.
4. If the documents do not contain enough information to answer
   the question, clearly say so.
5. Do not mention internal tools, agents, retrieval, grading,
   or workflow.
6. Keep the answer clear, concise, and useful.
"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
        }

