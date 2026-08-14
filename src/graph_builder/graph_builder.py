"""LangGraph workflow builder for RAGFury."""

from langgraph.graph import StateGraph, END

from src.state.rag_state import RAGState

from src.agent.agent import Agent

from src.node.retrieval_nodes import RAGNodes
from src.node.grading_nodes import GradingNodes
from src.node.rewrite_nodes import RewriteNodes
from src.node.wikipedia_nodes import WikipediaNodes
from src.node.reflection_nodes import ReflectionNodes
from src.node.generation_nodes import GenerationNodes


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
                    RAG       Wikipedia
                    |             |
                 Retrieve       Generate
                    |             |
                  Grade         Reflect
                 /    \         /     \
              YES      NO     PASS    FAIL
               |        |       |       |
            Generate  Rewrite   END    Retry
                        |
                     Retrieve
    """

    def __init__(self, retriever, llm):
        """
        Initialize the graph builder.

        Args:
            retriever:
                Existing hybrid retriever.

            llm:
                Chat model used throughout the workflow.
        """

        self.retriever = retriever
        self.llm = llm

        # =====================================================
        # INITIALIZE ROUTING AGENT
        # =====================================================

        self.agent = Agent(
            llm=self.llm
        )

        # =====================================================
        # INITIALIZE WORKFLOW NODES
        # =====================================================

        self.retrieval_nodes = RAGNodes(
            retriever=self.retriever,
            llm=self.llm
        )

        self.grading_nodes = GradingNodes(
            llm=self.llm
        )

        self.rewrite_nodes = RewriteNodes(
            llm=self.llm
        )

        self.generation_nodes = GenerationNodes(
            llm=self.llm
        )

        self.wikipedia_nodes = WikipediaNodes(
            llm=self.llm
        )

        self.reflection_nodes = ReflectionNodes(
            llm=self.llm
        )

    # =========================================================
    # NODE WRAPPERS
    # =========================================================

    def agent_node(self, state: RAGState):
        """
        Run the routing agent.

        The agent ONLY decides which workflow should handle
        the user's question.

        The decision is stored in:

            state["next_step"]

        Possible values:

            "rag"
            "wikipedia"
        """

        print("\n--- AGENT ---")

        question = state["question"]

        next_step = self.agent.route(
            question
        )

        print(
            f"Next step: {next_step}"
        )

        return {
            "next_step": next_step
        }

    def retrieve_node(self, state: RAGState):
        """Execute document retrieval."""

        print("\n--- RETRIEVE ---")

        return self.retrieval_nodes.retrieve_docs(
            state
        )

    def grade_node(self, state: RAGState):
        """Execute document relevance grading."""

        print("\n--- GRADE ---")

        return self.grading_nodes.grade_documents(
            state
        )

    def rewrite_node(self, state: RAGState):
        """Rewrite the user's query."""

        print("\n--- REWRITE ---")

        return self.rewrite_nodes.rewrite_query(
            state
        )

    def generate_node(self, state: RAGState):
        """Generate the final answer from retrieved documents."""

        print("\n--- GENERATE ---")

        return self.generation_nodes.generate_answer(
            state
        )

    def wikipedia_node(self, state: RAGState):
        """Generate an answer using Wikipedia."""

        print("\n--- WIKIPEDIA ---")

        return self.wikipedia_nodes.generate_wikipedia_answer(
            state
        )

    def reflection_node(self, state: RAGState):
        """Reflect on the generated Wikipedia answer."""

        print("\n--- REFLECTION ---")

        return self.reflection_nodes.reflect_on_answer(
            state
        )

    # =========================================================
    # CONDITIONAL ROUTING
    # =========================================================

    @staticmethod
    def route_after_agent(state: RAGState):
        """
        Route the workflow according to the Agent's decision.

        The Agent stores its decision in:

            state["next_step"]

        Returns:

            "rag"       -> RAG workflow
            "wikipedia" -> Wikipedia workflow
        """

        next_step = state.get(
            "next_step"
        )

        if next_step == "rag":
            return "rag"

        if next_step == "wikipedia":
            return "wikipedia"

        raise ValueError(
            f"Invalid next_step selected by agent: {next_step}"
        )

    @staticmethod
    def route_after_grader(state: RAGState):
        """
        Decide whether retrieved documents are relevant.

        Returns:

            "generate" -> documents are relevant
            "rewrite"  -> documents are not relevant
        """

        relevant = state.get(
            "document_relevance",
            False
        )

        if relevant:
            return "generate"

        return "rewrite"

    @staticmethod
    def route_after_reflection(state: RAGState):
        """
        Decide whether the Wikipedia answer passed reflection.

        Returns:

            "end"   -> reflection passed
            "retry" -> reflection failed
        """

        passed = state.get(
            "reflection_passed",
            False
        )

        if passed:
            return "end"

        return "retry"

    # =========================================================
    # BUILD GRAPH
    # =========================================================

    def build(self):
        """
        Build and compile the complete LangGraph workflow.

        Returns:
            Compiled LangGraph application.
        """

        workflow = StateGraph(
            RAGState
        )

        # =====================================================
        # ADD NODES
        # =====================================================

        workflow.add_node(
            "agent",
            self.agent_node
        )

        workflow.add_node(
            "retrieve",
            self.retrieve_node
        )

        workflow.add_node(
            "grade",
            self.grade_node
        )

        workflow.add_node(
            "rewrite",
            self.rewrite_node
        )

        workflow.add_node(
            "generate",
            self.generate_node
        )

        workflow.add_node(
            "wikipedia",
            self.wikipedia_node
        )

        workflow.add_node(
            "reflect",
            self.reflection_node
        )

        # =====================================================
        # START → AGENT
        # =====================================================

        workflow.set_entry_point(
            "agent"
        )

        # =====================================================
        # AGENT → RAG / WIKIPEDIA
        # =====================================================

        workflow.add_conditional_edges(
            "agent",
            self.route_after_agent,
            {
                "rag": "retrieve",
                "wikipedia": "wikipedia",
            },
        )

        # =====================================================
        # RAG WORKFLOW
        #
        # retrieve → grade
        # =====================================================

        workflow.add_edge(
            "retrieve",
            "grade"
        )

        # =====================================================
        # GRADE → GENERATE / REWRITE
        # =====================================================

        workflow.add_conditional_edges(
            "grade",
            self.route_after_grader,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            },
        )

        # =====================================================
        # REWRITE → RETRIEVE
        # =====================================================

        workflow.add_edge(
            "rewrite",
            "retrieve"
        )

        # =====================================================
        # GENERATE → END
        # =====================================================

        workflow.add_edge(
            "generate",
            END
        )

        # =====================================================
        # WIKIPEDIA → REFLECTION
        # =====================================================

        workflow.add_edge(
            "wikipedia",
            "reflect"
        )

        # =====================================================
        # REFLECTION → END / RETRY
        # =====================================================

        workflow.add_conditional_edges(
            "reflect",
            self.route_after_reflection,
            {
                "end": END,
                "retry": "wikipedia",
            },
        )

        # =====================================================
        # COMPILE
        # =====================================================

        return workflow.compile()