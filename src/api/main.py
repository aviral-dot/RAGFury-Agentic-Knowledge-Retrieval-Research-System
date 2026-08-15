"""FastAPI backend for RAGFury."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config.config import Config
from src.document_ingestion.document_processor import (
    DocumentProcessor,
)
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    HealthResponse,
    SystemInfoResponse,
)


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

    def initialize(self) -> None:
        """
        Initialize the complete RAGFury pipeline.

        This follows the same initialization flow as
        the existing main.py.
        """

        print(
            "🚀 Initializing RAGFury API..."
        )

       

        if not self.data_directory.exists():

            raise FileNotFoundError(
                f"Data directory not found: "
                f"{self.data_directory}"
            )

       

        print("🧠 Loading LLM...")

        self.llm = Config.get_llm()

       

        print(
            "📄 Initializing document processor..."
        )

        self.doc_processor = DocumentProcessor(
            model_name="all-MiniLM-L6-v2",
            threshold=0.3,
        )

       
        print(
            "🔍 Initializing vector store..."
        )

        self.vector_store = VectorStore()

        

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

       

        print(
            "🗂️ Creating hybrid vector store..."
        )

        self.vector_store.create_vectorstore(
            documents
        )

       

        print(
            "🔗 Building LangGraph workflow..."
        )

        self.graph_builder = GraphBuilder(
            retriever=(
                self.vector_store.get_retriever()
            ),
            llm=self.llm,
        )

        self.graph = (
            self.graph_builder.build()
        )

        self.initialized = True

        print(
            "✅ RAGFury API initialized successfully!"
        )

    def query(
        self,
        question: str,
    ) -> Dict[str, Any]:

        """
        Execute a question through the
        compiled LangGraph workflow.
        """

        if not self.initialized:

            raise RuntimeError(
                "RAGFury has not been initialized."
            )

        if self.graph is None:

            raise RuntimeError(
                "LangGraph is not available."
            )

        return self.graph.invoke(
            {
                "question": question
            }
        )




DATA_DIRECTORY = Path("data")


rag_service = RAGService(
    data_directory=DATA_DIRECTORY
)




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



app = FastAPI(
    title="RAGFury API",
    description=(
        "Agentic Knowledge Retrieval & "
        "Research System"
    ),
    version="1.0.0",
    lifespan=lifespan,
)




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



@app.get("/")
async def root():
    """API root endpoint."""

    return {
        "name": "RAGFury API",
        "version": "1.0.0",
        "status": "running",
    }




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




@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest,
):
    """
    Execute a question through RAGFury.

    The existing LangGraph decides between:

        rag
        wikipedia

    and executes the appropriate workflow.
    """

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if not rag_service.initialized:

        raise HTTPException(
            status_code=503,
            detail=(
                "RAGFury is not initialized. "
                "Check the API startup logs."
            ),
        )

    start_time = time.perf_counter()

    try:

        result = rag_service.query(
            question
        )

    except Exception as exc:

        print(
            f"❌ Query processing failed: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process the question."
            ),
        ) from exc

    elapsed = (
        time.perf_counter()
        - start_time
    )

    

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

    

    return QueryResponse(

        question=question,

        answer=result.get(
            "answer",
            "No answer generated.",
        ),

        next_step=result.get(
            "next_step"
        ),

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

        response_time=round(
            elapsed,
            4,
        ),
    )