"""LangGraph workflow builder for RAGFury."""

from langgraph.graph import StateGraph, END

from src.state.rag_state import RAGState

from src.agent.agent import Agent

from src.node.retrieval_nodes import RAGNodes
from src.node.chat_nodes import ChatNode
from src.node.grading_nodes import GradingNodes
from src.node.rewrite_nodes import RewriteNodes
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

        
        print("1️⃣ Creating Agent...")
        self.agent = Agent(
        llm=self.llm
        )
        print("✅ Agent created")

        print("2️⃣ Creating RAGNodes...")
        self.retrieval_nodes = RAGNodes(
        retriever=self.retriever,
        llm=self.llm,
        )
        print("✅ RAGNodes created")

        print("3️⃣ Creating GradingNodes...")
        
        self.grading_nodes = GradingNodes(
           llm=self.llm
        )
        print("✅ GradingNodes created")

        print("4️⃣ Creating RewriteNodes...")
        self.rewrite_nodes = RewriteNodes(
          llm=self.llm
        )
        print("✅ RewriteNodes created")

        print("5️⃣ Creating GenerationNodes...")
        self.generation_nodes = GenerationNodes(
         llm=self.llm,
        )
        print("✅ GenerationNodes created")

        print("6️⃣ Creating ChatNode...")
        self.chat_node_instance = ChatNode(
           llm=self.llm
        )
        print("✅ ChatNode created")

    
       
    def agent_node(self, state: RAGState):
     """
     Run the routing agent while preserving the complete
     LangGraph state, including user and conversation IDs.
     """

     print("\n--- AGENT ---")

    
     question = state.get("question")
     user_id = state.get("user_id")
     conversation_id = state.get("conversation_id")

     print(f"Question: {question}")
     print(f"User ID: {user_id}")
     print(f"Conversation ID: {conversation_id}")

   

     if not question:
        raise ValueError(
            "question is missing from LangGraph state."
        )

     if not user_id:
        raise ValueError(
            "user_id is missing from LangGraph state."
        )

     if not conversation_id:
        raise ValueError(
            "conversation_id is missing from LangGraph state."
        )

    # ---------------------------------------------------------
    # Route question
    # ---------------------------------------------------------

     next_step = self.agent.route(
        question
     )

     print(
        f"Next step: {next_step}"
     )

    

     return {
        **state,
        "next_step": next_step,
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

    

    def run_chat_node(self, state: RAGState):
        """
        Execute the conversational chat node.

        ChatNode handles:

        - Redis short-term memory
        - Mem0 long-term memory
        - Conversational context
        - LLM response generation
        """

        print("\n--- CHAT ---")

        return self.chat_node_instance.run(
            state
        )

    
    @staticmethod
    def route_after_agent(state: RAGState):
        """
        Route the workflow according to the Agent's decision.

        Returns:

            "rag"  -> RAG workflow
            "chat" -> Chat workflow
        """

        next_step = state.get(
            "next_step"
        )

        if next_step == "rag":
            return "rag"

        if next_step == "chat":
            return "chat"

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


    def build(self):
        """
        Build and compile the complete LangGraph workflow.

        Returns:
            Compiled LangGraph application.
        """

        workflow = StateGraph(
            RAGState
        )

        
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
            "chat",
            self.run_chat_node
        )

        

        workflow.set_entry_point(
            "agent"
        )

        

        workflow.add_conditional_edges(
            "agent",
            self.route_after_agent,
            {
                "rag": "retrieve",
                "chat": "chat",
            },
        )

        

        workflow.add_edge(
            "retrieve",
            "grade"
        )

        workflow.add_conditional_edges(
            "grade",
            self.route_after_grader,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            },
        )

        workflow.add_edge(
            "rewrite",
            "retrieve"
        )

        workflow.add_edge(
            "generate",
            END
        )

        
        workflow.add_edge(
            "chat",
            END
        )

        

        return workflow.compile() 