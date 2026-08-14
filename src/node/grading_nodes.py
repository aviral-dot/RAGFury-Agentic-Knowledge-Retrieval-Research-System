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
You are a document relevance grader.

Determine whether the retrieved documents contain
information that is relevant to answering the user's question.

User question:
{question}

Retrieved documents:
{context}

Evaluate the documents based on relevance to the question.

Return:
- relevant=True if the documents contain useful information
  for answering the question.
- relevant=False if the documents are unrelated, insufficient,
  or do not contain useful information.

Provide a brief reason for your decision.
"""

        result = self.grader.invoke(prompt)

        print(f"Relevant: {result.relevant}")
        print(f"Reason: {result.reason}")

        return {
            "document_relevance": result.relevant,
            "grade_reason": result.reason,
        }