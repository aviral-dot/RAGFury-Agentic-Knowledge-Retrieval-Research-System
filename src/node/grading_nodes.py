"""Document grading nodes for the Agentic RAG workflow."""

from pydantic import BaseModel, Field

from src.state.rag_state import RAGState


class DocumentGrade(BaseModel):
    """Structured output from the document relevance grader."""

    relevant: bool = Field(
        description="Whether the retrieved documents are relevant "
                    "to the user's question."
    )

    reason: str = Field(
        description="Brief explanation for the relevance decision."
    )


class GradingNodes:
    """Nodes responsible for grading retrieved documents."""

    def __init__(self, llm):
        """
        Initialize grading nodes.

        Args:
            llm: Chat model used for document grading.
        """
        self.llm = llm

        self.grader = llm.with_structured_output(
            DocumentGrade
        )

    def grade_documents(self, state: RAGState) -> dict:
        """
        Grade whether retrieved documents are relevant
        to the user's question.
        """

        print("---GRADING DOCUMENTS---")

        question = state["question"]
        documents = state.get("retrieved_docs", [])

        if not documents:
            return {
                "document_relevance": False,
                "grade_reason": "No documents were retrieved.",
            }

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        prompt = f"""
You are a document relevance grader for a Retrieval-Augmented
Generation (RAG) system.

Your task is to determine whether the retrieved document content
is relevant and useful for answering the user's question.

User question:
{question}

Retrieved documents:
{context}

Grading rules:

1. Return relevant=True if ANY retrieved document contains
   information that directly or indirectly helps answer the question.

2. Return relevant=True when the document contains the answer
   or contains specific facts needed to answer the question.

3. Semantic matches count as relevant. Do not require exact
   keyword matches.

4. Minor spelling or grammar mistakes in the user's question
   must NOT affect relevance.

5. Do NOT require the entire source document to be retrieved.
   A single relevant chunk is sufficient.

6. Return relevant=False only when the retrieved documents
   genuinely do not contain useful information for answering
   the question.

User questions may use different wording from the document.
Judge based on meaning, not exact wording.

Return a brief reason explaining your decision.
"""

        result = self.grader.invoke(prompt)

        print(f"Relevant: {result.relevant}")
        print(f"Reason: {result.reason}")

        return {
            "document_relevance": result.relevant,
            "grade_reason": result.reason,
        }