"""Routing agent for RAGFury."""

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured routing decision made by the agent."""

    next_step: Literal["rag", "chat"] = Field(
        description=(
            "The workflow that should handle the user's question. "
            "Choose 'rag' for questions that require information "
            "from the indexed private/company documents. "
            "Choose 'chat' for general conversation, general "
            "knowledge, follow-up questions, coding, writing, "
            "brainstorming, or questions that do not require "
            "the private document corpus."
        )
    )


class Agent:
    """
    Routing agent responsible for selecting the next workflow.

    Available workflows:

    - rag:
        Private/company document retrieval and RAG.

    - chat:
        General conversational AI with short-term and
        long-term memory.

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
        self.system_prompt = None

    def build(self):
        """
        Build the structured-output routing agent.

        Returns:
            Configured routing model.
        """

        system_prompt = """
You are the routing agent of RAGFury.

Your ONLY responsibility is to decide which workflow
should handle the user's current message.

You have exactly TWO workflows:

==================================================
1. rag
==================================================

Choose "rag" when the user is asking for information
that should come from the indexed private/company
document collection.

Examples:

- Questions about uploaded documents
- Questions about company policies
- Questions about internal/project documentation
- Questions asking "according to the document..."
- Questions about information expected to exist
  in the indexed document corpus
- Questions requiring retrieval from private documents

Examples:

"What is the company's leave policy?"

"How many sick leave days are allowed?"

"According to the remote work policy, what are
the working hours?"

"What does our security policy say about passwords?"


==================================================
2. chat
==================================================

Choose "chat" when the question does NOT require
retrieving information from the private document corpus.

This includes:

- General knowledge
- General factual questions
- Explanations
- Casual conversation
- Greetings
- Follow-up conversation
- Questions about previous conversation
- Personal conversational context
- Coding help
- Debugging help
- Writing help
- Brainstorming
- Opinions
- Everyday questions
- Creative tasks
- Questions that can be answered without
  the private document collection

Examples:

"Hello"

"How are you?"

"What is Python?"

"Explain Docker to me."

"Help me debug this code."

"What did I ask you earlier?"

"Can you explain that again?"

"Give me ideas for my project."


==================================================
IMPORTANT ROUTING RULES
==================================================

1. The current user message has the highest priority.

2. If the question explicitly depends on information
   inside private/company documents, choose "rag".

3. If the question is conversational or can be answered
   without the private document collection, choose "chat".

4. Follow-up questions should normally go to "chat"
   when they depend on previous conversation context.

5. Questions about the user's previous messages,
   preferences, or remembered information should go
   to "chat".

6. Do NOT choose "rag" simply because the question
   is factual.

7. Do NOT choose "chat" when the user explicitly asks
   about information contained in the indexed documents.

8. If the question is ambiguous and does not clearly
   require private document retrieval, choose "chat".

9. Do not answer the user's question.

10. Do not retrieve documents.

11. Do not generate an answer.

12. Do not rewrite the question.

13. Do not execute tools.

14. Return ONLY the structured routing decision.


==================================================
ALLOWED VALUES
==================================================

rag
chat
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
            "rag" or "chat".
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