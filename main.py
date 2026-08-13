"""Main application entry point for Agentic RAG system"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


class AgenticRAG:
    """Main Agentic RAG application"""

    def __init__(self, directory=None):
        """
        Initialize Agentic RAG system

        Args:
            directory: Directory containing PDF documents
        """
        print("🚀 Initializing Agentic RAG System...")

        
        self.directory = directory or Path("data")

        
        self.llm = Config.get_llm()

        self.doc_processor = DocumentProcessor(
          model_name="all-MiniLM-L6-v2",
          threshold=0.3
        )

        self.vector_store = VectorStore()

        
        self._setup_vectorstore()

        
        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )

        self.graph_builder.build()

        print("✅ System initialized successfully!\n")

    def _setup_vectorstore(self):
        """Setup vector store with processed documents"""

        print(f"📄 Processing documents from: {self.directory}")

        documents = self.doc_processor.process_pdfs(self.directory)

        print(f"📊 Created {len(documents)} document chunks")

        print("🔍 Creating vector store...")

        self.vector_store.create_vectorstore(documents)

    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system

        Args:
            question: User question

        Returns:
            Generated answer
        """
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")

        result = self.graph_builder.run(question)
        answer = result["answer"]

        print(f"✅ Answer: {answer}\n")

        return answer

    def interactive_mode(self):
        """Run in interactive mode"""

        print("💬 Interactive Mode - Type 'quit' to exit\n")

        while True:
            question = input("Enter your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if question:
                self.ask(question)
                print("-" * 80 + "\n")


def main():
    """Main function"""

    
    data_directory = Path("data")

    pdf_files = [str(pdf) for pdf in data_directory.glob("*.pdf")]

    
    if not data_directory.exists():
        print(f"❌ Data directory not found: {data_directory}")
        return

    
    rag = AgenticRAG(directory=pdf_files)

    
    example_questions = [
        "What is the purpose of the company security policy?",
        "What are the main rules for remote work?",
        "What is Sample Company Remote Work Policy"
    ]

    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")

    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")

    
    print("\n" + "=" * 80)

    user_input = input(
        "Would you like to enter interactive mode? (y/n): "
    )

    if user_input.lower() == "y":
        rag.interactive_mode()


if __name__ == "__main__":
    main()