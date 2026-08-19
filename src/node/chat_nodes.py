from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from src.memory.memory_manager import MemoryManager
from src.memory.memory_jobs import save_memory_turn
from src.memory.queue import memory_queue

class ChatNode:
    """
    General conversational chat node.

    Uses:
    - Redis for short-term conversation history
    - Mem0 for long-term semantic memory
    - Shared LLM passed from GraphBuilder
    """

    def __init__(self, llm):
        self.llm = llm
        self.memory_manager = MemoryManager()

    def run(self, state):
        

        user_id = state["user_id"]
        conversation_id = state["conversation_id"]
        question = state["question"]

     

        memory_context = self.memory_manager.get_context(
            user_id=user_id,
            conversation_id=conversation_id,
            query=question,
        )

        chat_history = memory_context["recent_history"]
        relevant_memories = memory_context["long_term_memories"]

       

        formatted_memories = self._format_memories(
            relevant_memories
        )

        

        history_messages = self._format_history(
            chat_history
        )

        

        system_prompt = f"""
You are a helpful AI assistant.

Your job is to answer the user's questions accurately,
clearly, and naturally.

You have access to:

1. Recent conversation history
2. Relevant long-term memories about the user

Use these memories only when they are relevant to
the current question.

Important rules:

- The current user message has the highest priority.
- Do not blindly trust stored memories if they conflict
  with the current conversation.
- Do not mention Redis, Mem0, memory retrieval, or
  internal system details.
- Do not invent facts.
- If you do not know something, say so clearly.
- Maintain conversational continuity when appropriate.

Relevant long-term memories:

{formatted_memories}
"""

        messages = [
            SystemMessage(content=system_prompt)
        ]

        messages.extend(history_messages)

        messages.append(
            HumanMessage(content=question)
        )


        response = self.llm.invoke(messages)

        answer = response.content


        self.memory_manager.redis.add_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=question,
            assistant_message=answer,
        )

        
        
        memory_queue.enqueue(
            save_memory_turn,
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=question,
            assistant_message=answer,
        )

       
        return {
            "answer": answer,
            "chat_history": chat_history,
            "relevant_memories": relevant_memories,
            "messages": [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ],
        }

  

    @staticmethod
    def _format_history(history: list) -> list[HumanMessage | AIMessage]:
        """
        Convert Redis history dictionaries into LangChain messages.
        """

        messages = []

        for message in history:

            role = message.get("role")
            content = message.get("content", "")

            if not content:
                continue

            if role == "user":
                messages.append(
                    HumanMessage(content=content)
                )

            elif role == "assistant":
                messages.append(
                    AIMessage(content=content)
                )

        return messages

    @staticmethod
    def _format_memories(memories: list) -> str:
        """
        Convert Mem0 search results into prompt-friendly text.
        """

        if not memories:
            return "No relevant long-term memories found."

        formatted = []

        for memory in memories:

            if isinstance(memory, dict):
                text = memory.get(
                    "memory",
                    memory.get("text", "")
                )
            else:
                text = str(memory)

            if text:
                formatted.append(f"- {text}")

        if not formatted:
            return "No relevant long-term memories found."

        return "\n".join(formatted)