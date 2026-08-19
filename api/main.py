"""FastAPI backend for RAGFury."""

import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config.config import Config
from src.document_ingestion.document_processor import (
    DocumentProcessor,
)
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

from api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    HealthResponse,
    SystemInfoResponse,
)


# =============================================================
# RAG SERVICE
# =============================================================

class RAGService:
    """
    Application service responsible for initializing
    and executing the RAGFury pipeline.
    """

    def __init__(
        self,
        data_directory: Path,
    ) -> None:

        self.data_directory = data_directory

        self.llm = None
        self.doc_processor = None
        self.vector_store = None
        self.graph_builder = None
        self.graph = None

        self.num_chunks = 0
        self.initialized = False

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def initialize(self) -> None:
        """
        Initialize the complete RAGFury pipeline.
        """

        print("\n🚀 Initializing RAGFury API...")

        # -----------------------------------------------------
        # Validate data directory
        # -----------------------------------------------------

        if not self.data_directory.exists():

            raise FileNotFoundError(
                f"Data directory not found: "
                f"{self.data_directory}"
            )

        # -----------------------------------------------------
        # Load LLM
        # -----------------------------------------------------

        print("🧠 Loading LLM...")

        self.llm = Config.get_llm()

        # -----------------------------------------------------
        # Document processor
        # -----------------------------------------------------

        print(
            "📄 Initializing document processor..."
        )

        self.doc_processor = DocumentProcessor(
            model_name="all-MiniLM-L6-v2",
            threshold=0.3,
        )

        # -----------------------------------------------------
        # Vector store
        # -----------------------------------------------------

        print(
            "🔍 Initializing vector store..."
        )

        self.vector_store = VectorStore()

        # -----------------------------------------------------
        # Process documents
        # -----------------------------------------------------

        print(
            f"📚 Processing documents from: "
            f"{self.data_directory}"
        )

        documents = (
            self.doc_processor.process_pdfs(
                self.data_directory
            )
        )

        self.num_chunks = len(documents)

        print(
            f"📊 Created "
            f"{self.num_chunks} document chunks"
        )

        if not documents:

            raise ValueError(
                "No PDF documents found "
                "in the data directory."
            )

        # -----------------------------------------------------
        # Create vector store
        # -----------------------------------------------------

        print(
            "🗂️ Creating hybrid vector store..."
        )

        self.vector_store.create_vectorstore(
            documents
        )

        # -----------------------------------------------------
        # Build LangGraph
        # -----------------------------------------------------

        print(
            "🔗 Building LangGraph workflow..."
        )

        self.graph_builder = GraphBuilder(
            retriever=(
                self.vector_store.get_retriever()
            ),
            llm=self.llm,
        )

        self.graph = self.graph_builder.build()

        self.initialized = True

        print(
            "✅ RAGFury API initialized successfully!"
        )

    # =========================================================
    # QUERY
    # =========================================================

    def query(
        self,
        question: str,
        user_id: str,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """
        Execute a question through the compiled
        LangGraph workflow.

        Args:
            question:
                User's current question.

            user_id:
                Manually supplied user identifier.
                Used by Mem0 for long-term memory.

            conversation_id:
                Automatically generated conversation
                identifier used by Redis for short-term
                conversational memory.
        """

        if not self.initialized:

            raise RuntimeError(
                "RAGFury has not been initialized."
            )

        if self.graph is None:

            raise RuntimeError(
                "LangGraph is not available."
            )

        # -----------------------------------------------------
        # Initial LangGraph state
        # -----------------------------------------------------

        initial_state = {
            "question": question,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }

        return self.graph.invoke(
            initial_state
        )


# =============================================================
# GLOBAL SERVICE
# =============================================================

DATA_DIRECTORY = Path("data")

rag_service = RAGService(
    data_directory=DATA_DIRECTORY
)


# =============================================================
# FASTAPI LIFESPAN
# =============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Initialize RAGFury when FastAPI starts.
    """

    try:

        rag_service.initialize()

    except Exception as exc:

        print(
            f"❌ RAGFury initialization failed: "
            f"{exc}"
        )

    yield

    print(
        "🛑 RAGFury API shutting down..."
    )


# =============================================================
# FASTAPI APPLICATION
# =============================================================

app = FastAPI(
    title="RAGFury API",
    description=(
        "Agentic Knowledge Retrieval & "
        "Research System"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================
# CORS
# =============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "*",
    ],
)


# =============================================================
# ROOT
# =============================================================

@app.get("/")
async def root():
    """API root endpoint."""

    return {
        "name": "RAGFury API",
        "version": "1.0.0",
        "status": "running",
    }


# =============================================================
# HEALTH
# =============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():
    """Return service health."""

    if rag_service.initialized:

        return HealthResponse(
            status="healthy",
            rag_initialized=True,
        )

    return HealthResponse(
        status="unhealthy",
        rag_initialized=False,
    )


# =============================================================
# SYSTEM INFO
# =============================================================

@app.get(
    "/api/v1/info",
    response_model=SystemInfoResponse,
)
async def system_info():
    """Return information about RAGFury."""

    return SystemInfoResponse(
        name="RAGFury",
        version="1.0.0",
        rag_initialized=(
            rag_service.initialized
        ),
        document_chunks=(
            rag_service.num_chunks
        ),
    )


# =============================================================
# QUERY
# =============================================================

@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest,
):
    """
    Execute a question through RAGFury.

    USER ID
    -------
    Manually supplied by the user.

    Used for:
        Mem0 long-term memory
        Redis user identification

    CONVERSATION ID
    ---------------
    Generated automatically by FastAPI
    for a new conversation.

    Reused when the frontend sends the existing
    conversation ID.

    Used for:
        Redis short-term conversation memory

    The LangGraph routing agent decides between:

        rag
        chat

    RAG workflow:

        Agent
          ↓
        Retrieve
          ↓
        Grade
        ↙    ↘
    Generate  Rewrite
                 ↓
              Retrieve

    Chat workflow:

        Agent
          ↓
        ChatNode
          ↓
      Redis + Mem0
          ↓
          LLM
          ↓
         END
    """

    # =========================================================
    # VALIDATE QUESTION
    # =========================================================

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # =========================================================
    # VALIDATE USER ID
    # =========================================================

    user_id = request.user_id.strip()

    if not user_id:

        raise HTTPException(
            status_code=400,
            detail="User ID cannot be empty.",
        )

    # =========================================================
    # GENERATE / REUSE CONVERSATION ID
    # =========================================================

    if request.conversation_id:

        conversation_id = (
            request.conversation_id.strip()
        )

        if not conversation_id:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Conversation ID cannot "
                    "be empty."
                ),
            )

    else:

        conversation_id = (
            f"conversation_{uuid.uuid4().hex[:12]}"
        )

        print(
            f"🆕 New conversation created: "
            f"{conversation_id}"
        )

    # =========================================================
    # CHECK INITIALIZATION
    # =========================================================

    if not rag_service.initialized:

        raise HTTPException(
            status_code=503,
            detail=(
                "RAGFury is not initialized. "
                "Check the API startup logs."
            ),
        )

    # =========================================================
    # EXECUTE GRAPH
    # =========================================================

    start_time = time.perf_counter()

    try:

        result = rag_service.query(
            question=question,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    except Exception as exc:

     print("\n" + "=" * 70)
     print("❌ QUERY PROCESSING FAILED")
     print("=" * 70)

     print(f"Error type: {type(exc).__name__}")
     print(f"Error: {exc}")

     traceback.print_exc()

     print("=" * 70 + "\n")

     raise HTTPException(
         status_code=500,
         detail=str(exc),
     ) from exc

    elapsed = (
         time.perf_counter()
         - start_time
     ) 

    # =========================================================
    # RETRIEVED DOCUMENTS
    # =========================================================

    raw_documents = result.get(
        "retrieved_docs",
        [],
    )

    documents: List[
        RetrievedDocument
    ] = []

    for document in raw_documents:

        if hasattr(
            document,
            "page_content",
        ):

            content = (
                document.page_content
            )

            metadata = getattr(
                document,
                "metadata",
                {},
            )

        else:

            content = str(document)

            metadata = {}

        documents.append(
            RetrievedDocument(
                content=content,
                metadata=metadata or {},
            )
        )

    # =========================================================
    # RESPONSE
    # =========================================================

    return QueryResponse(

        question=question,

        answer=result.get(
            "answer",
            "No answer generated.",
        ),

        # -----------------------------------------------------
        # Conversation ID
        # -----------------------------------------------------
        # Frontend must store this ID and send it back
        # for subsequent messages in the same conversation.
        # -----------------------------------------------------

        conversation_id=conversation_id,

        # -----------------------------------------------------
        # Agent routing decision
        # -----------------------------------------------------

        next_step=result.get(
            "next_step"
        ),

        # -----------------------------------------------------
        # RAG information
        # -----------------------------------------------------

        documents=documents,

        document_relevance=result.get(
            "document_relevance"
        ),

        grade_reason=result.get(
            "grade_reason"
        ),

        reflection=result.get(
            "reflection"
        ),

        reflection_passed=result.get(
            "reflection_passed"
        ),

        retrieval_attempts=result.get(
            "retrieval_attempts"
        ),

        reflection_attempts=result.get(
            "reflection_attempts"
        ),

        # -----------------------------------------------------
        # Performance
        # -----------------------------------------------------

        response_time=round(
            elapsed,
            4,
           ),
        )  