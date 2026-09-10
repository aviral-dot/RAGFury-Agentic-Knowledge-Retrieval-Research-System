

# 🚀 RAGFury — Agentic Knowledge Retrieval & Research System

> **A production-oriented Agentic RAG platform that routes requests between private knowledge retrieval and conversational AI, performs hybrid search with Qdrant Cloud, validates retrieved context with security and relevance guardrails, persists conversational state, caches repeated queries, applies distributed rate limiting, and exposes the system through a FastAPI API and Streamlit interface.**

<p align="center">

**Agentic RAG • LangGraph • Qdrant Cloud • Hybrid Search • Corrective Retrieval • Guardrails • Memory • Evaluation • Observability**

</p>

---

## ✨ What is RAGFury?

**RAGFury** is an agentic knowledge retrieval system built to go beyond a basic:

```text
Query → Retrieve → Generate
```

pipeline.

The system treats retrieval, generation, memory, security, and infrastructure as separate engineering concerns.

At query time, RAGFury can:

```text
                         ┌────────────────────┐
                         │     User Query     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │   Input Guardrail  │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │    Routing Agent   │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             Private RAG                      Chat
                    │                             │
                    ▼                             ▼
          Qdrant Cloud Hybrid          Redis + Mem0 Memory
              Retrieval                         │
                    │                           ▼
                    ▼                         LLM
          Retrieval Security                   │
             Guardrail                         ▼
                    │                         Answer
                    ▼
          Relevance Grading
                    │
             ┌──────┴──────┐
             │             │
          Relevant      Irrelevant
             │             │
             ▼             ▼
         Generate       Rewrite
                           │
                           ▼
                        Retrieve
                           │
                           ▼
                         Grade
```

The result is an application that combines **agentic orchestration, retrieval engineering, security controls, persistent memory, API infrastructure, caching, rate limiting, evaluation, and observability** in one system.

---

# 🧠 Core Architecture

RAGFury is divided into several major layers:

```text
                    ┌───────────────────────┐
                    │     Streamlit UI      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      FastAPI API      │
                    └───────────┬───────────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
                 ▼              ▼              ▼
           Guardrails       Rate Limit       Cache
                 │              │              │
                 └──────────────┼──────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      LangGraph        │
                    │   Agentic Workflow    │
                    └───────────┬───────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
          Private RAG                       Chat
                 │                             │
                 ▼                             ▼
         Qdrant Cloud                   Redis + Mem0
         Hybrid Search                      Memory
                 │                             │
                 ▼                             ▼
          Relevance Grade                     LLM
                 │
          ┌──────┴──────┐
          │             │
       Relevant      Retry
          │             │
          ▼             ▼
      Generate       Rewrite
```

Supporting infrastructure:

```text
                    ┌──────────────────────┐
                    │     LangSmith        │
                    │ Observability/Traces │
                    └──────────────────────┘

                    ┌──────────────────────┐
                    │     PostgreSQL       │
                    │ LangGraph Checkpoint │
                    └──────────────────────┘

                    ┌──────────────────────┐
                    │        Redis         │
                    │ Cache / Rate Limit / │
                    │ Conversation Memory │
                    └──────────────────────┘

                    ┌──────────────────────┐
                    │    Qdrant Cloud      │
                    │ Documents + Vectors  │
                    └──────────────────────┘
```

---

# 🔥 Why RAGFury?

Traditional RAG often assumes:

```text
Retrieved document = useful evidence
```

RAGFury explicitly separates these concepts:

```text
Retrieved
    ↓
Security validation
    ↓
Relevance validation
    ↓
Trusted context
    ↓
Generation
```

The central design principle is:

> **Retrieval is a candidate-generation step, not a guarantee of correctness.**

The system therefore introduces validation and recovery mechanisms around retrieval rather than blindly trusting the first search result.

---

# 🤖 1. Agentic Query Routing

Every request begins with a dedicated routing agent.

The router has exactly two workflow choices:

```text
rag
chat
```

The routing decision is represented using structured output:

```python
class RouteDecision(BaseModel):
    next_step: Literal["rag", "chat"]
```

The router selects:

### `rag`

For questions requiring information from the private/company document corpus.

Examples:

```text
What is the company's leave policy?

How many sick leave days are allowed?

According to the employee handbook, what are the working hours?
```

### `chat`

For requests that do not require private-document retrieval.

Examples:

```text
Hello

Explain Docker.

Help me debug this Python code.

What is machine learning?

Can you brainstorm ideas for my project?
```

The router itself does **not** retrieve documents or generate the final answer.

This keeps workflow selection separate from execution.

---

# 📚 2. Production RAG Pipeline

The private knowledge path follows:

```text
User Question
      ↓
Hybrid Retrieval
      ↓
Retrieved Documents
      ↓
Retrieval Security Guardrail
      ↓
Relevance Grading
      ↓
Relevant?
   ┌──┴──┐
   │     │
 YES     NO
   │     │
   ▼     ▼
Generate Rewrite
          │
          ▼
       Retrieve
          │
          ▼
        Grade
```

The workflow supports corrective retrieval rather than immediately generating from potentially irrelevant context.

---

# 🔎 3. Qdrant Cloud Hybrid Retrieval

The current production query path uses **Qdrant Cloud** rather than a local Chroma database.

The architecture is:

```text
                         Query
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Dense Embedding            Sparse BM25
       Qdrant Inference           Qdrant Inference
              │                         │
              └────────────┬────────────┘
                           ▼
                    Qdrant RRF Fusion
                           │
                           ▼
                      Top-K Results
```

RAGFury uses:

```text
Dense model:
sentence-transformers/all-MiniLM-L6-v2

Sparse model:
Qdrant/bm25
```

The existing production collection uses:

```text
Collection:
ragfury_documents

Dense vector:
""

Sparse vector:
"langchain-sparse"
```

The production query runtime does **not** load local embedding, reranking, PyTorch, Transformers, or CrossEncoder models.

Instead, embedding generation is delegated to Qdrant Cloud Inference.

This keeps the serverless query runtime significantly lighter.

---

# ⚡ 4. Qdrant Native RRF

Dense and sparse retrieval results are fused using Qdrant's native:

```text
Reciprocal Rank Fusion (RRF)
```

Conceptually:

```text
Dense Search ────────┐
                     ├──► RRF ───► Final Results
Sparse BM25 ─────────┘
```

This provides two complementary retrieval signals:

### Dense retrieval

Useful for:

* semantic similarity
* paraphrased questions
* conceptual matches
* meaning-based retrieval

### Sparse retrieval

Useful for:

* exact terminology
* policy names
* identifiers
* keyword-heavy questions
* rare terms

Combining the two improves robustness over relying on a single retrieval signal.

---

# 🛡️ 5. Retrieval Security Guardrails

Retrieved content is not automatically trusted.

RAGFury includes **NeMo Guardrails** around the retrieval pipeline.

The system provides:

```text
Input Guardrail
      ↓
RAG Workflow
      ↓
Retrieved Documents
      ↓
Retrieval Security Guardrail
      ↓
Relevance / Generation
```

The retrieval guardrail inspects retrieved document content before it is allowed to continue through the RAG pipeline.

This creates a security boundary between:

```text
External / indexed content
```

and:

```text
LLM reasoning / generation
```

The project also uses a fail-closed approach for input validation: if the input guardrail encounters an error, the request is not treated as automatically safe.

---

# 🚫 6. Input and Output Protection

The API integrates guardrail checks into the request lifecycle.

Conceptually:

```text
User Request
     ↓
Input Validation
     ↓
Agentic Workflow
     ↓
Generated Response
     ↓
Output Validation
     ↓
API Response
```

This allows the application to enforce safety checks independently from the core LangGraph workflow.

Guardrail configuration is maintained separately under:

```text
src/guardrails/
├── config.yml
├── prompts.yml
├── exceptions.py
└── guardrail_manager.py
```

---

# 🔄 7. Corrective Retrieval & Query Rewriting

When retrieved context is judged irrelevant, RAGFury does not immediately generate an answer.

Instead:

```text
Original Query
      ↓
Retrieve
      ↓
Grade
      ↓
Irrelevant
      ↓
Rewrite Query
      ↓
Retrieve Again
      ↓
Grade Again
```

The graph limits retrieval retries using a configured maximum retrieval-attempt mechanism.

The current graph builder defines:

```python
MAX_RETRIEVAL_ATTEMPTS = 3
```

This prevents an unsuccessful retrieval loop from becoming unbounded.

---

# 🧠 8. Stateful LangGraph Orchestration

The core workflow is implemented with:

```text
LangGraph StateGraph
```

The current graph is centered around:

```text
START
  ↓
Agent
  ↓
┌───────────────┐
│               │
RAG            Chat
│               │
├─ Retrieve     ├─ Memory
├─ Guardrail    ├─ Redis
├─ Grade        ├─ Mem0
├─ Rewrite      └─ LLM
└─ Generate
```

The graph is constructed through:

```text
src/graph_builder/graph_builder.py
```

Dedicated graph components include:

```text
Agent
RAGNodes
GradingNodes
RewriteNodes
GenerationNodes
ChatNode
```

This keeps orchestration separate from individual node responsibilities.

---

# 🧩 9. Typed Shared State

The workflow uses a shared typed:

```python
RAGState
```

Current state fields include:

```text
request_id
messages
question

user_id
conversation_id

chat_history
relevant_memories

next_step

retrieved_docs
citations

retrieval_attempts

document_relevance
grade_reason

answer

retrieval_abstained
abstention_reason

retrieval_metadata
```

This allows individual nodes to communicate through explicit state rather than global variables.

---

# 💬 10. Conversational AI with Persistent Memory

The `chat` workflow is not simply a stateless LLM call.

RAGFury combines two memory layers:

```text
                    Chat Request
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
           Redis                  Mem0
       Short-Term Memory      Long-Term Memory
              │                     │
              └──────────┬──────────┘
                         ▼
                   Memory Context
                         │
                         ▼
                        LLM
```

### Redis

Used for recent conversation history.

It provides short-term conversational continuity.

### Mem0

Used for semantic long-term memories associated with the user.

The memory manager retrieves both recent conversation history and relevant long-term memories before generating a response.

---

# 🧵 11. Persistent Conversation Checkpointing

RAGFury uses:

```text
PostgreSQL
+
LangGraph AsyncPostgresSaver
```

for persistent graph checkpoints.

Each conversation is mapped to a dedicated LangGraph thread:

```text
ragfury:{user_id}:{conversation_id}
```

Conceptually:

```text
User
 │
 ├── Conversation A
 │       └── PostgreSQL Thread A
 │
 ├── Conversation B
 │       └── PostgreSQL Thread B
 │
 └── Conversation C
         └── PostgreSQL Thread C
```

This allows different conversations to maintain isolated workflow state.

The FastAPI lifecycle initializes the PostgreSQL checkpointer before building the RAG graph and keeps it available while the application is running.

---

# ⚡ 12. Redis Query Caching

RAGFury includes a dedicated query-cache layer.

The API initializes:

```text
QueryCache
```

with configurable:

```text
TTL
lock TTL
wait timeout
key prefix
```

The default cache configuration includes:

```text
TTL:
300 seconds

Lock TTL:
180 seconds

Key prefix:
ragfury:query_cache:v2
```

Caching reduces unnecessary repeated LLM/retrieval work for identical or cache-equivalent requests.

The cache is implemented independently from the LangGraph workflow.

---

# 🚦 13. Distributed Rate Limiting

The API also includes a Redis-backed global rate limiter.

Default configuration:

```text
30 requests / minute
```

The limit is configurable through:

```text
GLOBAL_RATE_LIMIT_PER_MINUTE
```

The architecture is:

```text
Client
  ↓
FastAPI
  ↓
Rate Limiter
  ↓
Allowed?
 ┌─┴─┐
 │   │
No  Yes
 │    │
429   ▼
     Cache
       ↓
     LangGraph
```

This prevents uncontrolled request bursts from reaching the expensive parts of the pipeline.

---

# 🌐 14. FastAPI Production API

RAGFury exposes a dedicated FastAPI backend.

The application provides:

```text
GET  /
GET  /health
GET  /api/v1/info
POST /api/v1/query
```

The query endpoint accepts:

```json
{
  "question": "How much sick leave can an employee take?",
  "user_id": "demo-user",
  "conversation_id": "conversation-1"
}
```

`conversation_id` is optional and can be generated by the API when omitted.

The response can contain:

```text
question
answer
citations
documents
run_id
request_id
conversation_id

next_step

document_relevance
grade_reason

retrieval_attempts

response_time
```

The API schemas are defined in:

```text
api/schemas.py
```

and explicitly model query requests, citations, retrieved documents, health status, system information, errors, and feedback.

---

# 📚 15. Citation-Aware Responses

RAG responses are designed to preserve source information.

The API exposes citation objects containing:

```text
citation_id
source
chunk_id
page
```

This allows the frontend to present retrieved sources alongside the generated response rather than returning an opaque answer.

Conceptually:

```text
Answer
  │
  ├── Source 1
  │     ├── Document
  │     ├── Page
  │     └── Chunk
  │
  └── Source 2
        ├── Document
        ├── Page
        └── Chunk
```

This is particularly useful for document-grounded enterprise-style question answering.

---

# 📥 16. Dedicated Document Ingestion

Document ingestion has been separated from the production query runtime.

The ingestion architecture is:

```text
PDF
 │
 ▼
DocumentProcessor
 │
 ▼
Semantic Processing
 │
 ▼
Chunks
 │
 ▼
Qdrant Ingestion VectorStore
 │
 ▼
ragfury_documents
```

The dedicated:

```text
src/document_ingestion/
```

layer contains:

```text
document_processor.py
ingestion_service.py
worker.py
```

The ingestion service processes PDFs and writes their chunks into Qdrant.

This separation is important because:

```text
Ingestion workload
        ≠
Query workload
```

The production API therefore operates in:

```text
VectorStore(mode="query")
```

while indexing is handled separately.

---

# ☁️ 17. Serverless-Friendly Retrieval Runtime

One of the major architectural improvements in RAGFury is the removal of heavyweight local ML inference from the production query path.

The production vector store explicitly avoids loading:

```text
HuggingFaceEmbeddings
FastEmbedSparse
SentenceTransformer
CrossEncoder
PyTorch
Transformers
```

Instead:

```text
Application
    ↓
Qdrant Cloud Inference
    ↓
Dense + Sparse Embeddings
    ↓
Qdrant RRF
```

This makes the query layer significantly more appropriate for serverless deployment environments such as Vercel.

---

# 👁️ 18. LangSmith Observability

RAGFury uses **LangSmith** for application-level tracing and observability.

The API records:

```text
request_id
user_id
conversation_id
thread_id
workflow
```

as trace metadata.

Query executions are wrapped in a named trace:

```text
RAGFury Query
```

Retrieval is separately traceable:

```text
RAGFury Qdrant Cloud Hybrid Retrieval
```

The project also contains:

```text
src/utils/langsmith_observability.py
src/utils/trace_sanitizer.py
src/observability/langsmith.py
```

The intent is to make agent decisions, retrieval operations, and request lifecycle behavior observable without exposing raw sensitive conversation content in application logs.

---

# 🧪 19. Evaluation & Regression Testing

RAGFury contains a dedicated evaluation framework under:

```text
tests/evals/
```

The evaluation suite covers multiple RAG components.

```text
tests/evals/
│
├── components/
│   └── rag/
│       ├── test_generation_eval.py
│       ├── test_grader_eval.py
│       ├── test_retriever_eval.py
│       └── test_rewrite_eval.py
│
├── datasets/
│   ├── generation_goldens.py
│   ├── grading_goldens.py
│   ├── retrieval_goldens.py
│   ├── retrieval_ground_truth.py
│   └── rewrite_goldens.py
│
├── helpers/
│   ├── eval_models.py
│   ├── rag_eval_helpers.py
│   ├── regression_gate.py
│   └── ...
│
└── metrics/
    ├── generation_metrics.py
    ├── grading_metrics.py
    ├── retrieval_metrics.py
    └── rewrite_metrics.py
```

The retrieval evaluation includes deterministic:

```text
Recall@K
Precision@K
Hit Rate@K
```

and also supports LLM-judged retrieval evaluation.

Regression thresholds are centralized rather than embedding quality expectations directly inside individual tests.

---

# 🧪 20. Automated Test Coverage

The repository includes:

### Unit tests

```text
tests/unit/
```

covering areas such as:

```text
agent
configuration
document processing
guardrails
memory
memory jobs
nodes
query cache
RAG service
rate limiting
retrieval
schemas
semantic chunking
vector store
```

### Integration tests

```text
tests/integration/
```

covering:

```text
API behavior
API errors
output guardrails
feedback
RAG pipeline
```

This creates a testing structure around both individual components and end-to-end application behavior.

---

# 🖥️ 21. Streamlit Application

RAGFury also provides a dedicated Streamlit frontend.

The current frontend is a custom-designed interactive interface rather than a default Streamlit layout.

It communicates with the FastAPI backend through:

```text
RAGFURY_API_URL
```

The current default backend target is:

```text
https://ragfury.vercel.app
```

The frontend includes:

* conversational chat interface
* source cards
* citation display
* response metadata
* responsive layout
* custom visual styling
* animated interface elements
* API connectivity status
* conversation workspace

The frontend dependency set is intentionally separated through:

```text
requirements-streamlit.txt
```

which keeps the UI deployment lightweight.

---

# 🏗️ 22. Repository Structure

```text
RAGFury-Agentic-Knowledge-Retrieval-Research-System/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── Comp_Emp_Hand.pdf
│
├── src/
│   │
│   ├── agent/
│   │   └── agent.py
│   │
│   ├── cache/
│   │   └── query_cache.py
│   │
│   ├── checkpoint/
│   │   └── postgres.py
│   │
│   ├── config/
│   │   └── config.py
│   │
│   ├── document_ingestion/
│   │   ├── document_processor.py
│   │   ├── ingestion_service.py
│   │   └── worker.py
│   │
│   ├── graph_builder/
│   │   ├── conditional_edges.py
│   │   └── graph_builder.py
│   │
│   ├── guardrails/
│   │   ├── config.yml
│   │   ├── exceptions.py
│   │   ├── guardrail_manager.py
│   │   └── prompts.yml
│   │
│   ├── memory/
│   │   ├── memo_memory.py
│   │   ├── memory_jobs.py
│   │   ├── memory_manager.py
│   │   ├── queue.py
│   │   └── redis_memory.py
│   │
│   ├── models/
│   │   └── citation.py
│   │
│   ├── node/
│   │   ├── chat_nodes.py
│   │   ├── generation_nodes.py
│   │   ├── grading_nodes.py
│   │   ├── retrieval_nodes.py
│   │   └── rewrite_nodes.py
│   │
│   ├── observability/
│   │   └── langsmith.py
│   │
│   ├── rate_limit/
│   │   └── rate_limiter.py
│   │
│   ├── semantic_chunker/
│   │   └── semantic_chunker.py
│   │
│   ├── state/
│   │   └── rag_state.py
│   │
│   ├── utils/
│   │   ├── langsmith_observability.py
│   │   ├── loggers.py
│   │   └── trace_sanitizer.py
│   │
│   └── vectorstore/
│       ├── document_ids.py
│       ├── documents_ids.py
│       └── vectorstore.py
│
├── tests/
│   ├── evals/
│   ├── integration/
│   └── unit/
│
├── streamlit_app.py
├── run_server.py
├── docker-compose.yml
├── langgraph.json
├── requirements.txt
├── requirements-streamlit.txt
├── pyproject.toml
├── ruff.toml
├── uv.lock
├── .python-version
├── .gitignore
└── .vercelignore
```

---

# 📦 23. Component Responsibilities

| Component                    | Responsibility                                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `api/main.py`                | FastAPI application lifecycle, query execution, health, info, caching, rate limiting and API orchestration |
| `api/schemas.py`             | Request/response validation and public API contracts                                                       |
| `agent.py`                   | Structured `rag` vs `chat` workflow routing                                                                |
| `graph_builder.py`           | LangGraph workflow construction                                                                            |
| `conditional_edges.py`       | Graph routing logic                                                                                        |
| `rag_state.py`               | Shared typed workflow state                                                                                |
| `retrieval_nodes.py`         | Query-time document retrieval                                                                              |
| `grading_nodes.py`           | Retrieved-context relevance grading                                                                        |
| `rewrite_nodes.py`           | Query reformulation after poor retrieval                                                                   |
| `generation_nodes.py`        | Grounded answer generation                                                                                 |
| `chat_nodes.py`              | Conversational workflow with memory                                                                        |
| `vectorstore.py`             | Qdrant Cloud hybrid retrieval                                                                              |
| `document_processor.py`      | Document loading and semantic processing                                                                   |
| `ingestion_service.py`       | Document-to-Qdrant ingestion orchestration                                                                 |
| `worker.py`                  | Background ingestion/worker entry point                                                                    |
| `guardrail_manager.py`       | NeMo input, retrieval and output safety controls                                                           |
| `memory_manager.py`          | Redis + Mem0 memory orchestration                                                                          |
| `redis_memory.py`            | Short-term conversation history                                                                            |
| `memo_memory.py`             | Long-term semantic memory                                                                                  |
| `query_cache.py`             | Redis-backed query caching                                                                                 |
| `rate_limiter.py`            | Redis-backed global request limiting                                                                       |
| `postgres.py`                | PostgreSQL checkpoint configuration                                                                        |
| `citation.py`                | Citation data model                                                                                        |
| `langsmith_observability.py` | Trace metadata and tagging                                                                                 |
| `trace_sanitizer.py`         | Trace-data sanitization                                                                                    |
| `loggers.py`                 | Structured application logging                                                                             |
| `streamlit_app.py`           | Interactive web frontend                                                                                   |

---

# 🛠️ Technology Stack

| Technology                 | Role                                       |
| -------------------------- | ------------------------------------------ |
| **Python 3.13**            | Application/runtime language               |
| **FastAPI**                | Production API layer                       |
| **Uvicorn**                | ASGI server                                |
| **LangChain**              | LLM/retrieval abstractions                 |
| **LangGraph**              | Stateful agentic orchestration             |
| **Groq**                   | LLM inference                              |
| **OpenAI GPT-OSS 20B**     | Current configured Groq model              |
| **Qdrant Cloud**           | Production vector database                 |
| **Qdrant Cloud Inference** | Dense + sparse query embeddings            |
| **all-MiniLM-L6-v2**       | Dense embedding model                      |
| **Qdrant/bm25**            | Sparse retrieval model                     |
| **RRF**                    | Dense/sparse result fusion                 |
| **Redis**                  | Cache, rate limiting and short-term memory |
| **PostgreSQL**             | LangGraph persistent checkpoints           |
| **Mem0**                   | Long-term semantic memory                  |
| **NeMo Guardrails**        | Input/retrieval/output safety              |
| **LangSmith**              | Tracing and observability                  |
| **Streamlit**              | Interactive frontend                       |
| **PyPDF**                  | PDF processing                             |
| **Pydantic**               | Typed validation and structured outputs    |
| **RQ**                     | Background memory jobs                     |
| **uv**                     | Dependency/environment management          |
| **pytest**                 | Automated testing                          |
| **DeepEval**               | RAG component evaluation                   |
| **Ruff**                   | Linting/code quality                       |

The current project targets Python `>=3.13`.

---

# ⚙️ 24. Configuration

RAGFury reads runtime configuration from environment variables.

Core infrastructure variables include:

```env
GROQ_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=ragfury_documents

REDIS_URL=

CHECKPOINT_DATABASE_URL=

LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_PROJECT=

NVIDIA_API_KEY=
```

Additional application configuration includes:

```env
APP_ENV=
APP_VERSION=

GLOBAL_RATE_LIMIT_PER_MINUTE=

QUERY_CACHE_TTL_SECONDS=
QUERY_CACHE_LOCK_TTL_SECONDS=
QUERY_CACHE_WAIT_TIMEOUT_SECONDS=
QUERY_CACHE_KEY_PREFIX=

GRAPH_TIMEOUT_SECONDS=

CORS_ALLOWED_ORIGINS=

QDRANT_DENSE_MODEL=
QDRANT_SPARSE_MODEL=
QDRANT_DENSE_VECTOR_NAME=
QDRANT_SPARSE_VECTOR_NAME=
```

Never commit real credentials to Git.

---

# 🚀 25. Local Development

## Clone

```bash
git clone <repository-url>

cd RAGFury-Agentic-Knowledge-Retrieval-Research-System
```

## Create environment

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

# 📦 26. Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Using uv:

```bash
uv sync
```

The repository also contains:

```text
uv.lock
```

for reproducible dependency resolution.

---

# 🔐 27. Environment Setup

Create:

```text
.env
```

in the project root.

Configure the required infrastructure credentials.

At minimum, the application requires the relevant credentials for:

```text
Groq
Qdrant
Redis
PostgreSQL
NeMo Guardrails
LangSmith
Mem0 / supporting providers
```

Do not place secrets directly in source code.

---

# 🐳 28. Local Infrastructure with Docker Compose

The repository includes a local infrastructure composition for:

```text
Redis
Qdrant
PostgreSQL
```

Start the services:

```bash
docker compose up -d
```

The local services expose:

```text
Redis:
6379

Qdrant:
6333
6334

PostgreSQL:
5433
```

The PostgreSQL container is configured for the LangGraph checkpoint database.

The Compose setup is intended primarily for local development/testing; production deployments use managed infrastructure where appropriate.

---

# ▶️ 29. Start the FastAPI Backend

The API application lives in:

```text
api/main.py
```

A development server can be started with:

```bash
uvicorn api.main:app --reload
```

or through the repository's server launcher:

```bash
python run_server.py
```

The API exposes:

```text
/
 /health
/api/v1/info
/api/v1/query
```

---

# 💬 30. Run the Streamlit Frontend

Install the lightweight frontend dependencies:

```bash
pip install -r requirements-streamlit.txt
```

Then:

```bash
streamlit run streamlit_app.py
```

The frontend reads:

```env
RAGFURY_API_URL
```

to determine which FastAPI backend it should call.

For local development:

```env
RAGFURY_API_URL=http://127.0.0.1:8000
```

For a deployed backend, point it to the deployed API origin.

---

# 📥 31. Document Ingestion

Documents are indexed separately from the production query runtime.

The general process is:

```text
PDF
 ↓
DocumentProcessor
 ↓
Semantic Processing
 ↓
Chunks
 ↓
Qdrant
```

The ingestion service currently discovers PDF files from the configured data directory and processes them into Qdrant chunks.

Example:

```text
data/
└── Comp_Emp_Hand.pdf
```

Additional PDFs can be processed through the dedicated ingestion workflow.

---

# 🔄 32. Background Memory Jobs

Long-term memory operations can be handled through an RQ worker backed by Redis.

The queue is named:

```text
memory
```

The worker architecture is:

```text
FastAPI
   │
   ▼
Redis Queue
   │
   ▼
RQ Worker
   │
   ▼
Mem0
```

The queue is initialized lazily so importing the application does not require Redis connectivity during serverless startup.

---

# 🧪 33. Running Tests

Run the full test suite:

```bash
pytest
```

Unit tests:

```bash
pytest tests/unit
```

Integration tests:

```bash
pytest tests/integration
```

RAG evaluations:

```bash
pytest tests/evals
```

For retrieval evaluation, the project computes deterministic retrieval metrics such as:

```text
Recall@K
Precision@K
Hit Rate@K
```

and applies regression thresholds.

---

# 📊 34. Evaluation Philosophy

RAGFury treats evaluation as an engineering feedback loop:

```text
Code Change
    ↓
Run Tests
    ↓
Run RAG Evaluation
    ↓
Measure Retrieval / Grading / Rewrite / Generation
    ↓
Regression Gate
    ↓
Accept or Reject Change
```

This is designed to prevent retrieval improvements in one area from silently degrading another part of the system.

---

# 🔬 35. Example Private-Knowledge Questions

The current indexed company/employee handbook can be used for questions such as:

```text
How many hours per week must a full-time employee work?
```

```text
What does the employee handbook say about leave?
```

```text
What are the company's employee policies?
```

```text
According to the document, what benefits are available?
```

The response can contain both:

```text
Generated Answer
```

and:

```text
Source / Page / Chunk Citations
```

---

# 🧠 36. Example Chat Questions

The conversational workflow handles questions such as:

```text
Hello
```

```text
Explain Docker.
```

```text
What is Python?
```

```text
Help me debug this code.
```

```text
Can you explain that again?
```

The chat path can use:

```text
Recent Redis conversation history
+
Relevant Mem0 memories
+
Current user message
```

to provide continuity across interactions.

---

# 🧪 37. Retrieval-Failure Testing

RAGFury can be tested with questions that are unlikely to exist in the indexed corpus.

For example:

```text
What is the company's quantum computing policy?
```

The desired behavior is not:

```text
Retrieve unrelated document
        ↓
Generate confident answer
```

Instead:

```text
Retrieve
   ↓
Security Validation
   ↓
Relevance Grade
   ↓
Poor relevance
   ↓
Rewrite
   ↓
Retrieve Again
```

The system therefore has an explicit recovery mechanism around failed retrieval.

---

# 📈 38. Production-Oriented Engineering

RAGFury is no longer only a retrieval demo.

The current architecture addresses multiple production concerns:

### API

```text
FastAPI
```

### State

```text
LangGraph
+
PostgreSQL checkpoints
```

### Retrieval

```text
Qdrant Cloud
+
Dense/Sparse Hybrid Search
+
RRF
```

### Memory

```text
Redis
+
Mem0
```

### Performance

```text
Redis Query Cache
+
Distributed Rate Limiting
```

### Security

```text
NeMo Guardrails
+
Input Validation
+
Retrieved-Document Validation
+
Output Validation
```

### Observability

```text
LangSmith
+
Structured Logging
+
Trace Metadata
+
Trace Sanitization
```

### Quality

```text
pytest
+
component evaluations
+
golden datasets
+
retrieval metrics
+
regression gates
```

### Frontend

```text
Streamlit
```

---

# 🏆 39. What Makes RAGFury Different?

RAGFury is built around several engineering principles.

## 1. Route Before Expensive Processing

```text
Query
 ↓
Routing Agent
 ↓
Correct Workflow
```

---

## 2. Separate Ingestion from Query Serving

```text
Ingestion
    ↓
Qdrant

Query Serving
    ↓
Qdrant
```

The production API does not need to rebuild the document index during every startup.

---

## 3. Do Not Trust Retrieval Blindly

```text
Retrieved
    ↓
Security Check
    ↓
Relevance Check
    ↓
Generation
```

---

## 4. Recover From Retrieval Failure

```text
Bad Retrieval
     ↓
Rewrite
     ↓
Retrieve Again
```

---

## 5. Treat Conversation as State

```text
Conversation
     ↓
Redis
     +
Mem0
     +
PostgreSQL Checkpoint
```

---

## 6. Observe Before Optimizing

```text
Request
 ↓
Trace
 ↓
Measure
 ↓
Evaluate
 ↓
Improve
```

---

## 7. Keep the Production Runtime Lightweight

```text
Application
     ↓
Qdrant Cloud Inference
```

rather than loading large local ML stacks into the serverless query process.

---

# 🧱 40. Architectural Separation

The codebase deliberately separates:

```text
API
│
├── Request validation
├── Rate limiting
├── Cache
├── RAG service
└── Response serialization

Workflow
│
├── Agent
├── Retrieval
├── Grading
├── Rewrite
├── Generation
└── Chat

Infrastructure
│
├── Qdrant
├── Redis
├── PostgreSQL
└── Mem0

Safety
│
└── NeMo Guardrails

Observability
│
├── LangSmith
├── Structured logs
└── Trace sanitization

Quality
│
├── Unit tests
├── Integration tests
└── RAG evaluations
```

This makes individual subsystems replaceable without rewriting the entire application.

---

# 📊 41. Traditional RAG vs RAGFury

| Capability                   | Basic RAG |  RAGFury  |
| ---------------------------- | :-------: | :-------: |
| PDF ingestion                |     ✅     |     ✅     |
| Dense retrieval              |     ✅     |     ✅     |
| Sparse retrieval             | Sometimes |     ✅     |
| Hybrid retrieval             | Sometimes |     ✅     |
| Qdrant Cloud                 |     ❌     |     ✅     |
| Native RRF                   |     ❌     |     ✅     |
| Agentic routing              |     ❌     |     ✅     |
| Relevance grading            |    Rare   |     ✅     |
| Query rewriting              |    Rare   |     ✅     |
| Corrective retrieval         |    Rare   |     ✅     |
| Retrieval security guardrail |     ❌     |     ✅     |
| Input/output guardrails      |     ❌     |     ✅     |
| Redis short-term memory      |     ❌     |     ✅     |
| Mem0 long-term memory        |     ❌     |     ✅     |
| PostgreSQL checkpoints       |     ❌     |     ✅     |
| Query caching                |     ❌     |     ✅     |
| Distributed rate limiting    |     ❌     |     ✅     |
| API layer                    |  Optional |  FastAPI  |
| Citation-aware response      | Sometimes |     ✅     |
| LangSmith tracing            |  Optional |     ✅     |
| Unit testing                 | Sometimes |     ✅     |
| Integration testing          |    Rare   |     ✅     |
| RAG evaluation               |    Rare   |     ✅     |
| Regression gates             |    Rare   |     ✅     |
| Interactive UI               |  Optional | Streamlit |

---

# 🌍 42. Deployment Architecture

The production-oriented architecture separates the application from its managed stateful services:

```text
                         ┌───────────────────┐
                         │   Streamlit UI    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  Vercel / FastAPI │
                         │      Backend      │
                         └─────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
        Qdrant Cloud            Redis              PostgreSQL
        Documents + RRF       Cache / Memory       Checkpoints
              │                    │                    │
              │                    ▼                    │
              │                  Mem0                   │
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼
                              LangSmith
                              Observability
```

The design keeps persistent state outside the stateless application runtime.

---

# 🔐 43. Security Considerations

Never commit:

```text
.env
API keys
database credentials
Redis credentials
Qdrant credentials
private tokens
```

The application uses environment-driven configuration for infrastructure credentials.

Security-sensitive content is also intentionally excluded from normal structured application logs where possible.

---

# ⚠️ 44. Current Limitations

RAGFury is production-oriented, but it is not presented as an enterprise-complete platform.

Current boundaries include:

* Authentication/authorization is not yet a complete identity platform.
* The knowledge corpus is currently externally indexed rather than providing arbitrary end-user document uploads through the API.
* Qdrant ingestion is separated from query serving.
* External web research is not currently a first-class retrieval branch.
* The current router provides `rag` and `chat` workflows rather than the older Wikipedia branch.
* Production deployment depends on external managed infrastructure.
* Evaluation quality depends on the maintained golden datasets and evaluation configuration.
* The current system is designed primarily as a portfolio/interview-ready production-style AI engineering project rather than a fully managed enterprise SaaS product.

These boundaries are intentional and make the current architecture easier to reason about.

---

# 🗺️ 45. Roadmap

Future improvements can build naturally on the current architecture.

```text
                         RAGFury
                            │
             ┌──────────────┼──────────────┐
             │              │              │
         Retrieval        Agents         Safety
             │              │              │
         Reranking      Planning        Injection
         RRF tuning     Multi-step      Detection
         Multi-query    Research        PII
         HyDE           Tools           Policies
             │              │              │
             └──────────────┼──────────────┘
                            │
                         Quality
                            │
                 ┌──────────┼──────────┐
                 │          │          │
             Evaluation  Benchmarks  CI/CD
                 │          │          │
                 └──────────┼──────────┘
                            │
                        Production
                            │
                 ┌──────────┼──────────┐
                 │          │          │
             AuthN/AuthZ  Scaling   Document APIs
                 │          │          │
                 └──────────┼──────────┘
                            │
                         Platform
```

---

# 🎯 46. Engineering Skills Demonstrated

Building RAGFury demonstrates practical experience with:

### AI Engineering

* Retrieval-Augmented Generation
* Agentic workflows
* LangGraph
* LangChain
* Structured LLM outputs
* Prompt engineering
* Corrective retrieval
* Query rewriting

### Retrieval Engineering

* Dense retrieval
* Sparse retrieval
* Hybrid search
* BM25
* Qdrant
* Reciprocal Rank Fusion
* Semantic document processing
* Retrieval evaluation

### AI Reliability

* Relevance grading
* Retrieval security validation
* Input guardrails
* Output guardrails
* Abstention handling
* Retry boundaries
* Regression evaluation

### Memory Systems

* Redis short-term memory
* Mem0 long-term memory
* Persistent conversation state
* PostgreSQL LangGraph checkpoints

### Backend Engineering

* FastAPI
* Async execution
* API schemas
* Request validation
* Error handling
* Rate limiting
* Distributed caching
* Health checks

### MLOps / AI Observability

* LangSmith tracing
* Structured logging
* Trace metadata
* Trace sanitization
* Retrieval metrics
* Evaluation datasets
* Regression gates

### Application Engineering

* Streamlit
* Docker Compose
* Environment-based configuration
* Serverless-friendly architecture
* Background workers
* Dependency management with uv

---

# 📌 47. Key Design Decisions

### Qdrant Cloud instead of local Chroma

Production query-time retrieval uses managed Qdrant infrastructure.

### Cloud inference instead of local query embeddings

The query runtime avoids loading heavyweight ML inference dependencies.

### Redis + Mem0

Short-term conversational state and long-term semantic memory are handled as separate concerns.

### PostgreSQL checkpoints

LangGraph execution state is persisted independently from conversational memory.

### NeMo Guardrails

Security validation is placed around the LLM workflow rather than treated as an afterthought.

### LangSmith

Tracing provides visibility into the behavior of the agentic workflow and retrieval layer.

### Dedicated evaluation suite

Retrieval, grading, rewriting, and generation can be evaluated independently.

---

# 🚀 48. The RAGFury Workflow in One View

```text
                             USER
                              │
                              ▼
                       ┌─────────────┐
                       │ FastAPI API │
                       └──────┬──────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              Input Guardrail      Rate Limiter
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │ Query Cache │
                       └──────┬──────┘
                              │
                              ▼
                         LangGraph
                              │
                              ▼
                       Routing Agent
                         /       \
                        /         \
                       ▼           ▼
                    RAG           Chat
                     │              │
                     ▼              ▼
               Qdrant Cloud       Redis
               Dense + BM25        +
                    │             Mem0
                    ▼              │
                  RRF              ▼
                    │             LLM
                    ▼
             Security Guardrail
                    │
                    ▼
             Relevance Grader
                /        \
               /          \
          Relevant      Irrelevant
             │              │
             ▼              ▼
         Generate        Rewrite
                            │
                            ▼
                         Retrieve
                            │
                            ▼
                          Grade
                            │
                            ▼
                         Answer
                            │
                            ▼
                     Output Guardrail
                            │
                            ▼
                       Citations
                            │
                            ▼
                          USER
```

---

# 🏁 49. Final Summary

**RAGFury is an Agentic RAG platform engineered around the idea that reliable AI systems need more than retrieval and generation.**

It combines:

```text
Agentic Routing
       +
Hybrid Retrieval
       +
Qdrant Cloud
       +
Corrective Retrieval
       +
Security Guardrails
       +
Persistent Memory
       +
PostgreSQL Checkpointing
       +
Redis Caching
       +
Distributed Rate Limiting
       +
FastAPI
       +
Streamlit
       +
LangSmith
       +
Automated Evaluation
```

The architecture is intentionally modular:

```text
                RAGFury
                   │
      ┌────────────┼────────────┐
      │            │            │
    AI Core     Infra       Quality
      │            │            │
   LangGraph     Redis       Tests
   Retrieval     Qdrant      Evals
   Generation    Postgres    Metrics
   Memory        Mem0        Regression
      │            │            │
      └────────────┼────────────┘
                   │
              Production API
                   │
                Streamlit
```

The core philosophy is simple:

> **Don't just retrieve. Validate. Don't just generate. Observe. Don't just build a demo. Engineer the system around reliability, state, security, and measurable quality.**

---

# 👨‍💻 Author

**Aviral**

AI Engineer focused on:

```text
Agentic AI
RAG Systems
LLM Applications
AI Infrastructure
Retrieval Engineering
Production AI
```

---

# ⭐ Support

If RAGFury is useful or interesting, consider starring the repository and exploring the architecture.

---

## 📜 License

This project currently does not declare a specific open-source license.

If you intend to distribute it publicly under an open-source license, add the corresponding `LICENSE` file and update this section.
