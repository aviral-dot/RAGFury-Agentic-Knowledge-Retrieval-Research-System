"""LangGraph chat node with short-term and long-term memory."""

import logging
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from src.memory.memory_jobs import save_memory_turn
from src.memory.memory_manager import MemoryManager
from src.memory.queue import memory_queue
from src.state.rag_state import RAGState
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


class ChatNode:
    """
    General conversational chat node.

    Uses:
    - Redis for short-term conversation history
    - Mem0 for long-term semantic memory
    - Shared LLM passed from GraphBuilder
    """

    def __init__(self, llm):
        """Initialize the conversational chat node."""

        self.llm = llm
        self.memory_manager = MemoryManager()

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.node.initialized",
            component="ChatNode",
        )

    # =========================================================
    # CHAT EXECUTION
    # =========================================================

    def run(
        self,
        state: RAGState,
    ) -> dict:
        """
        Execute the conversational workflow.

        Memory contents and user-generated text are intentionally
        excluded from logs.
        """

        start_time = time.perf_counter()

        user_id = state["user_id"]
        conversation_id = state["conversation_id"]
        question = state["question"]

        log_event(
            logger,
            level=logging.INFO,
            event="chat.request.started",
        )

        # -----------------------------------------------------
        # MEMORY CONTEXT
        # -----------------------------------------------------

        memory_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.memory.context.started",
        )

        try:
            memory_context = self.memory_manager.get_context(
                user_id=user_id,
                conversation_id=conversation_id,
                query=question,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - memory_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="chat.memory.context.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to retrieve chat memory context",
            )

            raise

        memory_elapsed = (time.perf_counter() - memory_start_time) * 1000

        # -----------------------------------------------------
        # EXTRACT MEMORY CONTEXT
        # -----------------------------------------------------

        chat_history = memory_context.get(
            "recent_history",
            [],
        )

        relevant_memories = memory_context.get(
            "long_term_memories",
            [],
        )

        log_event(
            logger,
            level=logging.INFO,
            event="chat.memory.context.completed",
            recent_history_count=len(chat_history),
            long_term_memory_count=len(relevant_memories),
            duration_ms=round(
                memory_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # FORMAT MEMORY
        # -----------------------------------------------------

        formatting_start_time = time.perf_counter()

        formatted_memories = self._format_memories(relevant_memories)

        history_messages = self._format_history(chat_history)

        formatting_elapsed = (time.perf_counter() - formatting_start_time) * 1000

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.context.formatted",
            history_message_count=len(history_messages),
            long_term_memory_count=len(relevant_memories),
            duration_ms=round(
                formatting_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # SYSTEM PROMPT
        # -----------------------------------------------------

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

        messages = [SystemMessage(content=system_prompt)]

        messages.extend(history_messages)

        messages.append(HumanMessage(content=question))

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.llm.request.prepared",
            history_message_count=len(history_messages),
            long_term_memory_count=len(relevant_memories),
            total_message_count=len(messages),
        )

        # -----------------------------------------------------
        # LLM INVOCATION
        # -----------------------------------------------------

        llm_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="chat.llm.started",
        )

        try:
            response = self.llm.invoke(messages)

        except Exception as exc:
            llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="chat.llm.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            logger.exception(
                "Chat LLM invocation failed",
            )

            raise

        llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

        # -----------------------------------------------------
        # RESPONSE VALIDATION
        # -----------------------------------------------------

        answer = getattr(
            response,
            "content",
            None,
        )

        if answer is None:
            log_event(
                logger,
                level=logging.ERROR,
                event="chat.llm.invalid_response",
                response_type=type(response).__name__,
                duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            raise ValueError("Chat LLM returned a response without content.")

        answer = answer.strip()

        if not answer:
            log_event(
                logger,
                level=logging.ERROR,
                event="chat.llm.empty_response",
                duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            raise ValueError("Chat LLM returned an empty response.")

        log_event(
            logger,
            level=logging.INFO,
            event="chat.llm.completed",
            answer_length=len(answer),
            duration_ms=round(
                llm_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # REDIS SHORT-TERM MEMORY
        # -----------------------------------------------------

        redis_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.redis.write.started",
        )

        try:
            self.memory_manager.redis.add_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=question,
                assistant_message=answer,
            )

        except Exception as exc:
            redis_elapsed = (time.perf_counter() - redis_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="chat.redis.write.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    redis_elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to persist chat turn to Redis",
            )

            raise

        redis_elapsed = (time.perf_counter() - redis_start_time) * 1000

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.redis.write.completed",
            duration_ms=round(
                redis_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # MEM0 BACKGROUND MEMORY
        # -----------------------------------------------------

        queue_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.memory.enqueue.started",
        )

        try:
            memory_queue.enqueue(
                save_memory_turn,
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=question,
                assistant_message=answer,
            )

        except Exception as exc:
            queue_elapsed = (time.perf_counter() - queue_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="chat.memory.enqueue.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    queue_elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to enqueue long-term memory job",
            )

            raise

        queue_elapsed = (time.perf_counter() - queue_start_time) * 1000

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.memory.enqueue.completed",
            duration_ms=round(
                queue_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # REQUEST COMPLETED
        # -----------------------------------------------------

        total_elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="chat.request.completed",
            history_message_count=len(chat_history),
            long_term_memory_count=len(relevant_memories),
            answer_length=len(answer),
            duration_ms=round(
                total_elapsed,
                2,
            ),
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

    # =========================================================
    # HISTORY FORMATTING
    # =========================================================

    @staticmethod
    def _format_history(
        history: list,
    ) -> list[HumanMessage | AIMessage]:
        """
        Convert Redis history dictionaries into
        LangChain messages.
        """

        messages = []

        skipped_count = 0

        for message in history:
            role = message.get("role")

            content = message.get(
                "content",
                "",
            )

            if not content:
                skipped_count += 1

                continue

            if role == "user":
                messages.append(HumanMessage(content=content))

            elif role == "assistant":
                messages.append(AIMessage(content=content))

            else:
                skipped_count += 1

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.history.formatted",
            input_message_count=len(history),
            output_message_count=len(messages),
            skipped_message_count=skipped_count,
        )

        return messages

    # =========================================================
    # MEMORY FORMATTING
    # =========================================================

    @staticmethod
    def _format_memories(
        memories: list,
    ) -> str:
        """
        Convert Mem0 search results into
        prompt-friendly text.

        Memory contents are intentionally not logged.
        """

        if not memories:
            log_event(
                logger,
                level=logging.DEBUG,
                event="chat.memory.formatting.empty",
            )

            return "No relevant long-term memories found."

        formatted = []

        for memory in memories:
            if isinstance(
                memory,
                dict,
            ):
                text = memory.get(
                    "memory",
                    memory.get(
                        "text",
                        "",
                    ),
                )

            else:
                text = str(memory)

            if text:
                formatted.append(f"- {text}")

        if not formatted:
            log_event(
                logger,
                level=logging.DEBUG,
                event="chat.memory.formatting.empty",
                input_memory_count=len(memories),
            )

            return "No relevant long-term memories found."

        log_event(
            logger,
            level=logging.DEBUG,
            event="chat.memory.formatting.completed",
            input_memory_count=len(memories),
            formatted_memory_count=len(formatted),
        )

        return "\n".join(formatted)
