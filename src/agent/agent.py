"""Routing agent for RAGFury."""

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured routing decision made by the agent."""

    next_step: Literal["rag", "wikipedia"] = Field(
        description=(
            "The workflow that should handle the user's question. "
            "Choose 'rag' for questions related to indexed company documents "
            "and 'wikipedia' for general external knowledge."
        )
    )


class Agent:
    """
    Routing agent responsible for selecting the next workflow.

    Available workflows:
        - rag: comapny document retrieval and RAG pipeline
        - wikipedia: external Wikipedia knowledge

    The agent only makes the routing decision.
    It does not retrieve documents or generate answers.
    """

    def __init__(self, llm):
        """
        Initialize the routing agent.

        Args:
            llm:
                Chat model used to make the routing decision.
        """

        self.llm = llm
        self.agent = None

    def build(self):
        """
        Build the structured-output routing agent.

        Returns:
            Configured routing model.
        """

        system_prompt = """
You are the routing agent of RAGFury.

Your ONLY responsibility is to decide which workflow
should handle the user's question.

You have exactly two workflows:

1. rag

Choose "rag" when the question is related to:
- the user's uploaded documents
- indexed documents
- information contained in the document corpus
- private or project-specific information
- company-specific information
- information that is expected to be available
  in the user's indexed documents

2. wikipedia

Choose "wikipedia" when the question requires:
- general knowledge
- public factual information
- history
- science
- people
- places
- general concepts
- information that is unlikely to be available
  in the user's indexed documents

Rules:

- Choose exactly ONE workflow.
- Do not answer the question.
- Do not retrieve documents.
- Do not search Wikipedia.
- Do not generate an answer.
- Do not rewrite the question.
- Do not execute any tools.

Return only a structured routing decision.

The allowed values are:

rag
wikipedia
"""

        self.agent = self.llm.with_structured_output(
            RouteDecision
        )

        self.system_prompt = system_prompt

        return self.agent

    def route(self, question: str) -> str:
        """
        Decide which workflow should handle the question.

        Args:
            question:
                User's question.

        Returns:
            "rag" or "wikipedia".
        """

        if self.agent is None:
            self.build()

        response = self.agent.invoke(
            [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
        )

        return response.next_step

    def get_agent(self):
        """
        Return the routing agent.

        Builds the agent if it has not already been built.
        """

        if self.agent is None:
            self.build()

        return self.agent