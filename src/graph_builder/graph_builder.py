"""LangGraph workflow builder for RAGFury."""

import logging
import time

from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.agent.agent import Agent
from src.node.chat_nodes import ChatNode
from src.node.generation_nodes import GenerationNodes
from src.node.grading_nodes import GradingNodes
from src.node.retrieval_nodes import RAGNodes
from src.node.rewrite_nodes import RewriteNodes
from src.state.rag_state import RAGState
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 3


class GraphBuilder:
    """
    Builds the complete RAGFury LangGraph workflow.

    Workflow:

                         START
                           |
                         Agent
                           |
                      next_step
                      /        \
                    RAG        Chat
                    |           |
                 Retrieve    ChatNode
                    |           |
                  Grade      Redis + Mem0
                 /    \          |
              YES      NO        LLM
               |        |         |
            Generate  Rewrite    END
               |        |
              END     Retrieve
                        |
                       Grade
    """

    def __init__(
        self,
        retriever,
        llm,
        checkpointer=None,
    ):
        """
        Initialize the graph builder.

        Args:
            retriever:
                Existing hybrid retriever.

            llm:
                Chat model used throughout the workflow.

            checkpointer:
                Optional LangGraph checkpointer used for
                persistent conversation state.
        """

        self.retriever = retriever
        self.llm = llm
        self.checkpointer = checkpointer

        # -----------------------------------------------------
        # Agent
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.started",
            component="agent",
        )

        try:
            self.agent = Agent(
                llm=self.llm,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.component.initialization.failed",
                component="agent",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Agent initialization failed",
            )

            raise

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.completed",
            component="agent",
        )

        # -----------------------------------------------------
        # Retrieval nodes
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.started",
            component="retrieval_nodes",
        )

        try:
            self.retrieval_nodes = RAGNodes(
                retriever=self.retriever,
                llm=self.llm,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.component.initialization.failed",
                component="retrieval_nodes",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "RAGNodes initialization failed",
            )

            raise

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.completed",
            component="retrieval_nodes",
        )

        # -----------------------------------------------------
        # Grading nodes
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.started",
            component="grading_nodes",
        )

        try:
            self.grading_nodes = GradingNodes(
                llm=self.llm,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.component.initialization.failed",
                component="grading_nodes",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "GradingNodes initialization failed",
            )

            raise

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.completed",
            component="grading_nodes",
        )

        # -----------------------------------------------------
        # Rewrite nodes
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.started",
            component="rewrite_nodes",
        )

        try:
            self.rewrite_nodes = RewriteNodes(
                llm=self.llm,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.component.initialization.failed",
                component="rewrite_nodes",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "RewriteNodes initialization failed",
            )

            raise

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.completed",
            component="rewrite_nodes",
        )

        # -----------------------------------------------------
        # Generation nodes
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.started",
            component="generation_nodes",
        )

        try:
            self.generation_nodes = GenerationNodes(
                llm=self.llm,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.component.initialization.failed",
                component="generation_nodes",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "GenerationNodes initialization failed",
            )

            raise

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.completed",
            component="generation_nodes",
        )

        # -----------------------------------------------------
        # Chat node
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.started",
            component="chat_node",
        )

        try:
            self.chat_node_instance = ChatNode(
                llm=self.llm,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.component.initialization.failed",
                component="chat_node",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "ChatNode initialization failed",
            )

            raise

        log_event(
            logger,
            level=logging.INFO,
            event="graph.component.initialization.completed",
            component="chat_node",
        )

    # =========================================================
    # AGENT NODE
    # =========================================================

    @traceable(
        name="RAGFury Router",
        run_type="chain",
    )
    async def agent_node(
        self,
        state: RAGState,
    ):
        """
        Run the routing agent while preserving the complete
        LangGraph state, including user and conversation IDs.
        """

        start_time = time.perf_counter()

        request_id = state.get("request_id")

        user_id = state.get("user_id")

        conversation_id = state.get("conversation_id")

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.started",
            node="agent",
            request_id=request_id,
            conversation_id=conversation_id,
        )

        question = state.get("question")

        if not question:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.validation.failed",
                node="agent",
                request_id=request_id,
                reason="question_missing",
            )

            raise ValueError("question is missing from LangGraph state.")

        if not user_id:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.validation.failed",
                node="agent",
                request_id=request_id,
                reason="user_id_missing",
            )

            raise ValueError("user_id is missing from LangGraph state.")

        if not conversation_id:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.validation.failed",
                node="agent",
                request_id=request_id,
                reason="conversation_id_missing",
            )

            raise ValueError("conversation_id is missing from LangGraph state.")

        try:
            next_step = await self.agent.route(question)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.failed",
                node="agent",
                request_id=request_id,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Agent routing failed",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.completed",
            node="agent",
            request_id=request_id,
            next_step=next_step,
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return {
            **state,
            "next_step": next_step,
        }

    # =========================================================
    # RETRIEVE NODE
    # =========================================================

    @traceable(
        name="RAGFury Retrieve",
        run_type="retriever",
    )
    async def retrieve_node(
        self,
        state: RAGState,
    ):
        """Execute document retrieval."""

        start_time = time.perf_counter()

        request_id = state.get("request_id")

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.started",
            node="retrieve",
            request_id=request_id,
        )

        try:
            result = await self.retrieval_nodes.retrieve_docs(state)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.failed",
                node="retrieve",
                request_id=request_id,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Document retrieval failed",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        retrieved_docs = result.get(
            "retrieved_docs",
            [],
        )

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.completed",
            node="retrieve",
            request_id=request_id,
            document_count=len(retrieved_docs),
            retrieval_attempts=result.get("retrieval_attempts"),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result

    # =========================================================
    # GRADE NODE
    # =========================================================

    @traceable(
        name="RAGFury Grade",
        run_type="chain",
    )
    async def grade_node(
        self,
        state: RAGState,
    ):
        """Execute document relevance grading."""

        start_time = time.perf_counter()

        request_id = state.get("request_id")

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.started",
            node="grade",
            request_id=request_id,
        )

        try:
            result = await self.grading_nodes.grade_documents(state)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.failed",
                node="grade",
                request_id=request_id,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Document grading failed",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.completed",
            node="grade",
            request_id=request_id,
            document_relevance=result.get("document_relevance"),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result

    # =========================================================
    # REWRITE NODE
    # =========================================================
    @traceable(
        name="RAGFury rewrite",
        run_type="chain",
    )
    async def rewrite_node(
        self,
        state: RAGState,
    ):
        """Rewrite the user's query."""

        start_time = time.perf_counter()

        request_id = state.get("request_id")

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.started",
            node="rewrite",
            request_id=request_id,
        )

        try:
            result = await self.rewrite_nodes.rewrite_query(state)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.failed",
                node="rewrite",
                request_id=request_id,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Query rewriting failed",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.completed",
            node="rewrite",
            request_id=request_id,
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result

    # =========================================================
    # GENERATE NODE
    # =========================================================
    @traceable(
        name="RAGFury Generate",
        run_type="chain",
    )
    async def generate_node(
        self,
        state: RAGState,
    ):
        """Generate the final answer from retrieved documents."""

        start_time = time.perf_counter()

        request_id = state.get("request_id")

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.started",
            node="generate",
            request_id=request_id,
        )

        try:
            result = await self.generation_nodes.generate_answer(state)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.failed",
                node="generate",
                request_id=request_id,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Answer generation failed",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        answer = result.get(
            "answer",
            "",
        )

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.completed",
            node="generate",
            request_id=request_id,
            answer_length=len(answer or ""),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result

    # =========================================================
    # CHAT NODE
    # =========================================================
    @traceable(
        name="RAGFury Chat",
        run_type="chain",
    )
    async def run_chat_node(
        self,
        state: RAGState,
    ):
        """
        Execute the conversational chat node.

        ChatNode handles:

        - Redis short-term memory
        - Mem0 long-term memory
        - Conversational context
        - LLM response generation
        """

        start_time = time.perf_counter()

        request_id = state.get("request_id")

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.started",
            node="chat",
            request_id=request_id,
        )

        try:
            result = await self.chat_node_instance.run(state)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.node.failed",
                node="chat",
                request_id=request_id,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Chat node execution failed",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        answer = result.get(
            "answer",
            "",
        )

        log_event(
            logger,
            level=logging.INFO,
            event="graph.node.completed",
            node="chat",
            request_id=request_id,
            answer_length=len(answer or ""),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result

    # =========================================================
    # ABSTAIN NODE
    # =========================================================

    def abstain_node(
        self,
        state: RAGState,
    ):
        """
        Stop the RAG workflow when no relevant information
        is available in the document collection.
        """

        request_id = state.get("request_id")

        retrieval_attempts = state.get(
            "retrieval_attempts",
            0,
        )

        retrieved_docs = state.get(
            "retrieved_docs",
            [],
        )

        if not retrieved_docs:
            answer = (
                "No documents related to your question were found "
                "in the available document collection."
            )

            abstention_reason = "no_documents_retrieved"

        else:
            answer = (
                "I’m sorry, but I couldn’t find sufficiently "
                "relevant information in the available documents "
                "to answer your question reliably."
            )

            abstention_reason = "maximum_retrieval_attempts_exhausted"

        log_event(
            logger,
            level=logging.WARNING,
            event="graph.node.abstained",
            node="abstain",
            request_id=request_id,
            retrieval_attempts=retrieval_attempts,
            reason=abstention_reason,
        )

        return {
            "answer": answer,
            "retrieval_abstained": True,
            "abstention_reason": abstention_reason,
        }

    # =========================================================
    # ROUTE AFTER AGENT
    # =========================================================

    @staticmethod
    def route_after_agent(
        state: RAGState,
    ):
        """
        Route the workflow according to the Agent's decision.

        Returns:

            "rag"  -> RAG workflow
            "chat" -> Chat workflow
        """

        request_id = state.get("request_id")

        next_step = state.get("next_step")

        if next_step == "rag":
            log_event(
                logger,
                level=logging.INFO,
                event="graph.routing.decision",
                node="agent",
                request_id=request_id,
                destination="rag",
            )

            return "rag"

        if next_step == "chat":
            log_event(
                logger,
                level=logging.INFO,
                event="graph.routing.decision",
                node="agent",
                request_id=request_id,
                destination="chat",
            )

            return "chat"

        log_event(
            logger,
            level=logging.ERROR,
            event="graph.routing.failed",
            node="agent",
            request_id=request_id,
            invalid_next_step=str(next_step),
        )

        raise ValueError(f"Invalid next_step selected by agent: {next_step}")

    @staticmethod
    def route_after_retrieval(state: RAGState):
        """
        Decide whether document grading is necessary.

        No retrieved documents:
            -> abstain immediately

        Retrieved documents:
            -> grade
        """

        request_id = state.get("request_id")

        retrieved_docs = state.get(
            "retrieved_docs",
            [],
        )

        if not retrieved_docs:
            log_event(
                logger,
                level=logging.WARNING,
                event="graph.routing.decision",
                node="retrieve",
                request_id=request_id,
                destination="abstain",
                reason="no_documents_retrieved",
            )

            return "abstain"

        log_event(
            logger,
            level=logging.INFO,
            event="graph.routing.decision",
            node="retrieve",
            request_id=request_id,
            destination="grade",
            document_count=len(retrieved_docs),
        )

        return "grade"

    # =========================================================
    # ROUTE AFTER GRADER
    # =========================================================

    @staticmethod
    def route_after_grader(
        state: RAGState,
    ):
        """
            Decide what to do after document relevance grading.

        Relevant documents:
            -> generate

        Irrelevant documents and retry budget remaining:
            -> rewrite

        Irrelevant documents and retry budget exhausted:
            -> abstain
        """

        request_id = state.get("request_id")

        relevant = state.get(
            "document_relevance",
            False,
        )

        retrieval_attempts = state.get(
            "retrieval_attempts",
            0,
        )

        if relevant:
            log_event(
                logger,
                level=logging.INFO,
                event="graph.routing.decision",
                node="grade",
                request_id=request_id,
                destination="generate",
                document_relevance=True,
                retrieval_attempts=retrieval_attempts,
            )

            return "generate"

        if retrieval_attempts >= MAX_RETRIEVAL_ATTEMPTS:
            log_event(
                logger,
                level=logging.WARNING,
                event="graph.routing.decision",
                node="grade",
                request_id=request_id,
                destination="abstain",
                document_relevance=False,
                retrieval_attempts=retrieval_attempts,
                max_retrieval_attempts=MAX_RETRIEVAL_ATTEMPTS,
            )

            return "abstain"

        log_event(
            logger,
            level=logging.INFO,
            event="graph.routing.decision",
            node="grade",
            request_id=request_id,
            destination="rewrite",
            document_relevance=False,
            retrieval_attempts=retrieval_attempts,
            max_retrieval_attempts=MAX_RETRIEVAL_ATTEMPTS,
        )

        return "rewrite"

    # =========================================================
    # BUILD
    # =========================================================

    def build(self):
        """
        Build and compile the complete LangGraph workflow.

        Returns:
            Compiled LangGraph application.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="graph.build.started",
        )

        try:
            workflow = StateGraph(RAGState)

            workflow.add_node(
                "agent",
                self.agent_node,
            )

            workflow.add_node(
                "retrieve",
                self.retrieve_node,
            )

            workflow.add_node(
                "grade",
                self.grade_node,
            )

            workflow.add_node(
                "abstain",
                self.abstain_node,
            )

            workflow.add_node(
                "rewrite",
                self.rewrite_node,
            )

            workflow.add_node(
                "generate",
                self.generate_node,
            )

            workflow.add_node(
                "chat",
                self.run_chat_node,
            )

            workflow.set_entry_point("agent")

            workflow.add_conditional_edges(
                "agent",
                self.route_after_agent,
                {
                    "rag": "retrieve",
                    "chat": "chat",
                },
            )

            workflow.add_conditional_edges(
                "retrieve",
                self.route_after_retrieval,
                {
                    "grade": "grade",
                    "abstain": "abstain",
                },
            )

            workflow.add_conditional_edges(
                "grade",
                self.route_after_grader,
                {
                    "generate": "generate",
                    "rewrite": "rewrite",
                    "abstain": "abstain",
                },
            )

            workflow.add_edge(
                "rewrite",
                "retrieve",
            )

            workflow.add_edge(
                "generate",
                END,
            )

            workflow.add_edge(
                "chat",
                END,
            )

            workflow.add_edge(
                "abstain",
                END,
            )

            graph = workflow.compile(checkpointer=self.checkpointer)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="graph.build.failed",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "LangGraph build failed",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="graph.build.completed",
            node_count=6,
            checkpointer_enabled=(self.checkpointer is not None),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return graph
