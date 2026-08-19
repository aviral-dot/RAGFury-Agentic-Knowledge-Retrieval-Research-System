"""Main application entry point for Agentic RAG system."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


class AgenticRAG:
    """Main Agentic RAG application."""

    def __init__(self, directory=None):
        """Initialize the Agentic RAG system."""

        print("🚀 Initializing Agentic RAG System...")

        self.directory = directory or Path("data")

        # ---------------------------------------------------------
        # LLM
        # ---------------------------------------------------------

        self.llm = Config.get_llm()

        # ---------------------------------------------------------
        # Document processing
        # ---------------------------------------------------------

        self.doc_processor = DocumentProcessor(
            model_name="all-MiniLM-L6-v2",
            threshold=0.3
        )

        # ---------------------------------------------------------
        # Vector store
        # ---------------------------------------------------------

        self.vector_store = VectorStore()

        self._setup_vectorstore()

        # ---------------------------------------------------------
        # LangGraph
        # ---------------------------------------------------------

        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )

        self.graph = self.graph_builder.build()

        print("✅ System initialized successfully!\n")

    def _setup_vectorstore(self):
        """Setup vector store with processed documents."""

        print(f"📄 Processing documents from: {self.directory}")

        documents = self.doc_processor.process_pdfs(
            self.directory
        )

        print(
            f"📊 Created {len(documents)} document chunks"
        )

        print("🔍 Creating vector store...")

        self.vector_store.create_vectorstore(
            documents
        )

    def ask(
        self,
        question: str,
        user_id: str,
        conversation_id: str,
    ) -> str:
        """
        Ask a question to the Agentic RAG system.

        Args:
            question:
                User's question.

            user_id:
                Unique identifier for the user.

            conversation_id:
                Unique identifier for the conversation.

        Returns:
            Generated answer.
        """

        print(f"👤 User: {user_id}")
        print(f"💬 Conversation: {conversation_id}")
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")

        # ---------------------------------------------------------
        # Invoke LangGraph
        # ---------------------------------------------------------

        result = self.graph.invoke({
            "question": question,
            "user_id": user_id,
            "conversation_id": conversation_id,
        })

        # ---------------------------------------------------------
        # Get final answer
        # ---------------------------------------------------------

        answer = result.get(
            "answer",
            ""
        )

        print(f"✅ Answer: {answer}\n")

        return answer

    def interactive_mode(self):
        """Run the system in interactive mode."""

        # ---------------------------------------------------------
        # Local testing IDs
        # ---------------------------------------------------------

        user_id = "local_user"
        conversation_id = "local_conversation"

        print(
            "💬 Interactive Mode - Type 'quit' to exit\n"
        )

        print(
            f"👤 User ID: {user_id}"
        )

        print(
            f"💬 Conversation ID: {conversation_id}\n"
        )

        while True:

            question = input(
                "Enter your question: "
            ).strip()

            if question.lower() in [
                "quit",
                "exit",
                "q"
            ]:
                print("👋 Goodbye!")
                break

            if not question:
                continue

            self.ask(
                question=question,
                user_id=user_id,
                conversation_id=conversation_id,
            )

            print(
                "-" * 80 + "\n"
            )


def main():
    """Main function."""

    data_directory = Path("data")

    if not data_directory.exists():
        print(
            f"❌ Data directory not found: "
            f"{data_directory}"
        )
        return

    # ---------------------------------------------------------
    # Initialize application
    # ---------------------------------------------------------

    rag = AgenticRAG(
        directory=data_directory
    )

    # ---------------------------------------------------------
    # Example RAG questions
    # ---------------------------------------------------------

    example_questions = [
        "How much sick leave can an employee take?",
        "What time does the remote work day start and end?"
    ]

    demo_user_id = "demo_user"
    demo_conversation_id = "demo_conversation"

    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")

    for question in example_questions:

        rag.ask(
            question=question,
            user_id=demo_user_id,
            conversation_id=demo_conversation_id,
        )

        print("=" * 80 + "\n")

    # ---------------------------------------------------------
    # Interactive mode
    # ---------------------------------------------------------

    print("\n" + "=" * 80)

    user_input = input(
        "Would you like to enter interactive mode? (y/n): "
    ).strip()

    if user_input.lower() == "y":
        rag.interactive_mode()


if __name__ == "__main__":
    main()


