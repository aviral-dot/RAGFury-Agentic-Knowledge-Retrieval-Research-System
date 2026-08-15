



# 🚀 RAGFury — Agentic Knowledge Retrieval & Research System

> **An agentic Retrieval-Augmented Generation system that intelligently routes questions between private document knowledge and external/general knowledge, validates retrieved context, rewrites failed queries, and uses reflection to improve answer quality.**

<p align="center">

**Agentic RAG • LangGraph • Hybrid Retrieval • Semantic Chunking • Corrective Retrieval • Self-Reflection**

</p>

---

## 🧠 Overview

**RAGFury** is an Agentic RAG system designed to address a common weakness in traditional Retrieval-Augmented Generation systems:

> **Retrieving a document does not necessarily mean retrieving the right document.**

A conventional RAG pipeline often follows:

```text
User Query
    ↓
Retriever
    ↓
Retrieved Context
    ↓
LLM
    ↓
Answer
```

The problem is that a retriever can return documents that are semantically or lexically similar to a query while still being irrelevant to the actual question.

RAGFury introduces an additional reasoning and validation layer:

```text
                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Routing Agent    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              Private RAG                 External Path
                    │                           │
                    ▼                           ▼
             Hybrid Retrieval              Generation
                    │                           │
                    ▼                           ▼
             Relevance Grading             Reflection
                    │                           │
              ┌─────┴─────┐                ┌────┴────┐
              │           │                │         │
           Relevant    Irrelevant         Pass      Fail
              │           │                │         │
              ▼           ▼                ▼         ▼
          Generate     Rewrite Query       END      Retry
                          │
                          ▼
                       Retrieve
```

The system therefore combines:

* 🤖 Agentic workflow routing
* 📄 Private PDF knowledge retrieval
* ✂️ Threshold-based semantic chunking
* 🔎 Dense + sparse hybrid retrieval
* 📊 LLM-based document relevance grading
* ✏️ Query rewriting
* 🔄 Corrective retrieval
* 🌐 External/general knowledge workflow
* 🪞 Answer reflection
* 🧠 Structured LangGraph state
* 💬 Interactive application interface

---

# ✨ Key Features

## 🤖 1. Agentic Query Routing

RAGFury begins every query with a dedicated routing agent.

The agent makes a structured decision:

```text
rag
```

or:

```text
wikipedia
```

The routing agent is intentionally separated from retrieval and answer generation.

Its responsibility is only:

```text
User Query
    ↓
Routing Agent
    ↓
next_step
    ├── rag
    └── wikipedia
```

This separation keeps the workflow modular and makes the routing decision explicit.

The routing decision is represented using a Pydantic structured output:

```python
class RouteDecision(BaseModel):
    next_step: Literal["rag", "wikipedia"]
```

This prevents the router from returning arbitrary free-form text.

---

# 📄 2. Private Document Knowledge Base

RAGFury can ingest documents from a local knowledge directory.

The current example knowledge base contains:

```text
data/
├── Leave_Policy.pdf
└── Remote_Work_Policy.pdf
```

The repository is structured so additional PDF documents can be added to the same directory.

The ingestion pipeline is:

```text
PDF Documents
      ↓
PDF Loader
      ↓
LangChain Documents
      ↓
Semantic Chunking
      ↓
Embeddings
      ↓
Hybrid Retrieval Index
```

The document processor currently supports:

* PDF directories
* Individual PDF files
* TXT files
* URLs

through dedicated loader methods.

---

# ✂️ 3. Threshold-Based Semantic Chunking

Instead of blindly splitting documents using fixed character or token sizes, RAGFury implements a lightweight semantic chunking strategy.

The process is:

```text
Document
    ↓
Sentence Splitting
    ↓
Sentence Embeddings
    ↓
Cosine Similarity
    ↓
Similarity Threshold
    ↓
Semantic Chunks
```

The project uses:

```text
Embedding Model:
sentence-transformers/all-MiniLM-L6-v2
```

The chunker compares consecutive sentence embeddings.

If their cosine similarity is above the configured threshold:

```text
Sentence A
      +
Sentence B
```

they remain in the same chunk.

If similarity falls below the threshold:

```text
Sentence A
      ↓
New semantic boundary
      ↓
Sentence B
```

a new chunk is created.

The current default threshold used by the document processor is:

```python
threshold = 0.3
```

This implementation is located in:

```text
src/semantic_chunker/semantic_chunker.py
```

and is used by the document processing pipeline.

---

# 🔎 4. Hybrid Retrieval

RAGFury combines two complementary retrieval strategies:

```text
                    Query
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
      Dense Retrieval      Sparse Retrieval
            │                   │
         Chroma                BM25
            │                   │
            └─────────┬─────────┘
                      │
                      ▼
              EnsembleRetriever
                      │
                      ▼
             Retrieved Documents
```

### Dense Retrieval

The dense retriever uses:

```text
Chroma
+
HuggingFace Embeddings
+
all-MiniLM-L6-v2
```

The current implementation persists the vector database under:

```text
./chroma_db
```

and retrieves the top candidates using Chroma's retriever interface.

### Sparse Retrieval

BM25 provides lexical matching for:

* exact terms
* names
* policy terminology
* technical vocabulary
* keyword-heavy queries

### Hybrid Retrieval

The two retrieval signals are combined using LangChain's:

```text
EnsembleRetriever
```

with the current weights:

```text
Dense / Chroma: 0.7
BM25:           0.3
```

This allows RAGFury to benefit from both semantic similarity and lexical matching.

---

# 📊 5. Document Relevance Grading

Retrieval is not automatically trusted.

After documents are retrieved, RAGFury sends the retrieved context to a structured relevance grader.

```text
User Query
    ↓
Retrieve
    ↓
Grade Documents
    ↓
Relevant?
```

The grader returns:

```python
class DocumentGrade(BaseModel):
    relevant: bool
    reason: str
```

This means the system gets both:

* a relevance decision
* an explanation for that decision

The grader considers semantic meaning rather than requiring exact keyword matches.

---

# ✏️ 6. Query Rewriting

If the retrieved documents are judged irrelevant, RAGFury does not immediately generate an answer.

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
```

The query rewriting node transforms the original question into a clearer retrieval-oriented query.

For example:

```text
Original:

"What is the company's security policy?"
```

can be transformed into a more retrieval-friendly formulation.

The goal is not to answer the question during rewriting.

The rewriting node returns only the improved search query.

---

# 🔄 7. Corrective RAG Workflow

The private-document workflow therefore behaves like a corrective RAG pipeline:

```text
                    ┌───────────────┐
                    │ User Question │
                    └───────┬───────┘
                            │
                            ▼
                       ┌─────────┐
                       │ Retrieve│
                       └────┬────┘
                            │
                            ▼
                        ┌───────┐
                        │ Grade │
                        └───┬───┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
              Relevant             Irrelevant
                 │                     │
                 ▼                     ▼
             Generate              Rewrite
                                       │
                                       ▼
                                   Retrieve
                                       │
                                       ▼
                                     Grade
```

This introduces an explicit quality checkpoint between retrieval and generation.

The important principle is:

> **Retrieved ≠ Relevant**

---

# 🪞 8. Answer Reflection

The external knowledge workflow includes an answer reflection stage.

The process is:

```text
Generate Answer
      ↓
   Reflect
      ↓
┌─────┴─────┐
│           │
PASS       FAIL
│           │
▼           ▼
END        Retry
```

The reflection model evaluates whether:

1. The answer addresses the user's question.
2. The answer is sufficiently complete.
3. The answer contains unsupported claims.

The reflection result is represented using structured output:

```python
class ReflectionResult(BaseModel):
    passed: bool
    reason: str
```

Reflection attempts are tracked in the graph state.

---

# 🧩 9. LangGraph Workflow Orchestration

The entire system is orchestrated using LangGraph's `StateGraph`.

The current graph is:

```text
                         START
                           │
                           ▼
                    ┌────────────┐
                    │   Agent    │
                    └─────┬──────┘
                          │
                    next_step
                     /          \
                    /            \
                   ▼              ▼
                RAG            Wikipedia
                 │                │
                 ▼                ▼
              Retrieve          Generate
                 │                │
                 ▼                ▼
               Grade           Reflect
              /     \          /      \
             /       \        /        \
            ▼         ▼      ▼          ▼
        Generate   Rewrite   END       Retry
                      │                 │
                      ▼                 │
                   Retrieve ◄──────────┘
```

The graph is built in:

```text
src/graph_builder/graph_builder.py
```

The graph contains dedicated nodes for:

* routing
* retrieval
* grading
* rewriting
* generation
* external generation
* reflection

and uses conditional edges to determine the next step.

---

# 🧠 State Management

All workflow information is represented using a typed `RAGState`.

The current state contains fields such as:

```python
question
rewritten_question
next_step
retrieved_docs
document_relevance
grade_reason
answer
reflection
reflection_passed
retrieval_attempts
reflection_attempts
```

This gives every graph node a shared structured state instead of relying on global variables.

The state definition lives in:

```text
src/state/rag_state.py
```

---

# 🏗️ Project Architecture

```text
RAGFury-Agentic-Knowledge-Retrieval-Research-System/
│
├── data/
│   ├── Leave_Policy.pdf
│   └── Remote_Work_Policy.pdf
│
├── src/
│   │
│   ├── agent/
│   │   └── agent.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── document_ingestion/
│   │   ├── __init__.py
│   │   └── document_processor.py
│   │
│   ├── semantic_chunker/
│   │   ├── __init__.py
│   │   └── semantic_chunker.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── vectorstore.py
│   │
│   ├── graph_builder/
│   │   ├── __init__.py
│   │   └── graph_builder.py
│   │
│   ├── node/
│   │   ├── __init__.py
│   │   ├── retrieval_nodes.py
│   │   ├── grading_nodes.py
│   │   ├── rewrite_nodes.py
│   │   ├── generation_nodes.py
│   │   ├── wikipedia_nodes.py
│   │   └── reflection_nodes.py
│   │
│   └── state/
│       ├── __init__.py
│       └── rag_state.py
│
├── main.py
├── streamlit_app.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

The repository follows a modular architecture separating configuration, ingestion, semantic processing, retrieval, graph orchestration, nodes, and state management.

---

# 📦 Component Responsibilities

| Component               | Responsibility                          |
| ----------------------- | --------------------------------------- |
| `agent.py`              | Structured query routing                |
| `config.py`             | LLM and environment configuration       |
| `document_processor.py` | Document loading and preprocessing      |
| `semantic_chunker.py`   | Semantic sentence-based chunking        |
| `vectorstore.py`        | Chroma + BM25 hybrid retrieval          |
| `rag_state.py`          | Shared LangGraph state                  |
| `retrieval_nodes.py`    | Document retrieval                      |
| `grading_nodes.py`      | Retrieved-document relevance evaluation |
| `rewrite_nodes.py`      | Failed-query reformulation              |
| `generation_nodes.py`   | Private-document answer generation      |
| `wikipedia_nodes.py`    | External/general answer generation      |
| `reflection_nodes.py`   | Generated-answer quality evaluation     |
| `graph_builder.py`      | LangGraph workflow construction         |
| `main.py`               | CLI/application entry point             |
| `streamlit_app.py`      | Web UI entry point                      |

---

# 🛠️ Technology Stack

| Technology                    | Role                                     |
| ----------------------------- | ---------------------------------------- |
| **Python 3.13**               | Core development language                |
| **LangChain**                 | LLM and retrieval abstractions           |
| **LangGraph**                 | Agentic workflow orchestration           |
| **Groq**                      | LLM inference                            |
| **Llama 3.1 8B Instant**      | Generation/routing model                 |
| **Sentence Transformers**     | Embedding generation                     |
| **all-MiniLM-L6-v2**          | Semantic embeddings                      |
| **Chroma**                    | Dense vector storage                     |
| **BM25**                      | Sparse lexical retrieval                 |
| **EnsembleRetriever**         | Hybrid retrieval                         |
| **Pydantic**                  | Structured model outputs                 |
| **PyPDF / LangChain loaders** | PDF ingestion                            |
| **Streamlit**                 | Interactive UI                           |
| **python-dotenv**             | Environment configuration                |
| **uv**                        | Python environment/dependency management |

The repository currently targets Python 3.13 and defines its dependencies through `pyproject.toml` and `requirements.txt`.

---

# ⚙️ How the System Works

## Step 1 — Load Documents

PDF files are loaded from:

```text
data/
```

The document processor converts them into LangChain `Document` objects.

---

## Step 2 — Semantic Chunking

Each document is divided into sentences.

Sentence embeddings are generated using:

```text
all-MiniLM-L6-v2
```

Consecutive sentences are compared using cosine similarity.

The configured threshold determines where semantic boundaries are created.

---

## Step 3 — Create the Retrieval Layer

The semantic chunks are indexed using:

```text
             Documents
                 │
        ┌────────┴────────┐
        ▼                 ▼
     Chroma              BM25
        │                 │
        └────────┬────────┘
                 ▼
         EnsembleRetriever
```

The current hybrid weighting is:

```text
Chroma: 0.7
BM25:   0.3
```

---

## Step 4 — Route the Query

A structured routing agent decides:

```text
rag
```

or:

```text
wikipedia
```

The routing agent itself does not retrieve documents or generate the final response.

---

## Step 5 — Private RAG

For document-oriented questions:

```text
Query
  ↓
Hybrid Retrieval
  ↓
Document Grading
  ↓
Relevant?
```

If relevant:

```text
Generate Answer
```

If irrelevant:

```text
Rewrite Query
      ↓
Retrieve Again
      ↓
Grade Again
```

---

## Step 6 — External Knowledge Path

For general-knowledge questions:

```text
Query
  ↓
Routing Agent
  ↓
External Workflow
  ↓
Generate
  ↓
Reflect
```

If reflection fails, the workflow retries the generation path.

---

# ▶️ Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/aviral-dot/RAGFury-Agentic-Knowledge-Retrieval-Research-System.git

cd RAGFury-Agentic-Knowledge-Retrieval-Research-System
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

# 📦 3. Install Dependencies

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv

If you use `uv`:

```bash
uv sync
```

The project also includes `uv.lock` for reproducible dependency resolution.

---

# 🔐 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The application loads the key through `python-dotenv`.

The configured LLM is:

```text
llama-3.1-8b-instant
```

with deterministic temperature:

```text
temperature = 0
```

### ⚠️ Security

Never commit:

```text
.env
API keys
tokens
credentials
private documents
```

The repository already ignores `.env` and the local Chroma database.

---

# 📄 5. Add Your Documents

Place your PDFs inside:

```text
data/
```

Example:

```text
data/
├── Leave_Policy.pdf
├── Remote_Work_Policy.pdf
├── Employee_Handbook.pdf
└── Company_Policies.pdf
```

Then the application will process them during initialization.

---

# ▶️ 6. Run the CLI Application

Run:

```bash
python main.py
```

The application:

1. Loads documents.
2. Performs semantic chunking.
3. Builds the hybrid retrieval system.
4. Creates the LangGraph workflow.
5. Runs example questions.
6. Optionally starts interactive mode.

The current CLI entry point is implemented in `main.py`.

---

# 💬 Interactive Mode

After startup, the application can enter interactive mode:

```text
💬 Interactive Mode - Type 'quit' to exit
```

Example:

```text
Enter your question: How much sick leave can an employee take?
```

Exit using:

```text
quit
```

or:

```text
exit
```

---

# 🌐 Streamlit Interface

The repository also contains:

```text
streamlit_app.py
```

Run:

```bash
python -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

> **Note:** The Streamlit file is currently a separate UI implementation and should be kept aligned with the current `AgenticRAG`/LangGraph API as the project evolves.

---

# 💡 Example Queries

## 📚 Private Knowledge Queries

Try questions such as:

```text
How much sick leave can an employee take?
```

```text
What are the remote working hours?
```

```text
What does the leave policy say about sick leave?
```

```text
What is the employee leave entitlement?
```

---

## 🔍 Retrieval-Failure Tests

These queries are useful for testing the relevance grader:

```text
What is the company's security policy?
```

If the indexed documents do not contain security information, the system should not blindly treat unrelated policy documents as evidence.

The intended behavior is:

```text
Retrieve
   ↓
Grade
   ↓
Irrelevant
   ↓
Rewrite
   ↓
Retrieve Again
   ↓
Grade Again
```

---

## 🌐 General Knowledge Queries

Examples:

```text
What is quantum computing?
```

```text
How does nuclear fusion work?
```

```text
Who is Brad Pitt?
```

These are intended to exercise the external/general-knowledge branch.

---

# 🧪 Engineering Concepts Demonstrated

RAGFury demonstrates several important AI engineering concepts.

### Retrieval Engineering

* Retrieval-Augmented Generation
* Dense retrieval
* Sparse retrieval
* Hybrid retrieval
* Chroma
* BM25
* Ensemble retrieval

### Document Processing

* PDF ingestion
* Sentence splitting
* Sentence embeddings
* Cosine similarity
* Threshold-based semantic chunking

### Agentic AI

* Structured routing
* LangGraph
* Conditional workflows
* Tool/workflow selection
* State-based orchestration

### RAG Reliability

* Document relevance grading
* Query rewriting
* Corrective retrieval
* Retrieval retry loops
* Answer reflection

### LLM Engineering

* Structured LLM output
* Prompt-based generation
* LLM routing
* State management
* Groq inference

### Application Engineering

* Modular Python architecture
* Configuration management
* CLI interface
* Streamlit interface
* Local vector persistence
* Response processing

---

# 🆚 Traditional RAG vs RAGFury

| Capability                  | Traditional RAG | RAGFury |
| --------------------------- | :-------------: | :-----: |
| PDF ingestion               |        ✅        |    ✅    |
| Semantic chunking           |    Sometimes    |    ✅    |
| Dense retrieval             |        ✅        |    ✅    |
| Sparse retrieval            |    Sometimes    |    ✅    |
| Hybrid retrieval            |    Sometimes    |    ✅    |
| Query routing               |        ❌        |    ✅    |
| Retrieval grading           |        ❌        |    ✅    |
| Query rewriting             |        ❌        |    ✅    |
| Corrective retrieval        |        ❌        |    ✅    |
| Answer reflection           |       Rare      |    ✅    |
| Stateful workflow           | Usually limited |    ✅    |
| Conditional graph execution |        ❌        |    ✅    |
| Interactive UI              |     Optional    |    ✅    |

---

# 🏆 What Makes RAGFury Different?

The project is intentionally built around the idea that:

```text
Retrieval
   ≠
Correct Retrieval
```

and:

```text
Generated Answer
   ≠
Reliable Answer
```

Therefore, RAGFury introduces checkpoints throughout the workflow:

```text
             Query
               │
               ▼
          Route Query
               │
       ┌───────┴────────┐
       │                │
     Private          External
       │                │
       ▼                ▼
    Retrieve          Generate
       │                │
       ▼                ▼
     Grade           Reflect
       │                │
    ┌──┴──┐        ┌───┴───┐
    │     │        │       │
  Good   Bad     Pass     Fail
    │     │        │       │
    ▼     ▼        ▼       ▼
 Generate Rewrite  END    Retry
             │
             ▼
          Retrieve
```

This turns a simple retrieval pipeline into a **stateful, corrective workflow**.

---

# 📈 Current Project Direction

RAGFury is structured so additional production capabilities can be added without rewriting the core retrieval workflow.

Potential next steps include:

### 🔎 Advanced Retrieval

* Cross-encoder reranking
* Reciprocal Rank Fusion
* Multi-query retrieval
* Query expansion
* HyDE
* Retrieval confidence scoring

### 🧠 Agentic Improvements

* Multi-step research planning
* Additional external sources
* More sophisticated routing
* Adaptive retrieval strategies
* Multi-agent research workflows

### 🛡️ AI Safety

* NVIDIA NeMo Guardrails
* Prompt-injection detection
* Input/output validation
* Sensitive-information filtering
* Tool-use restrictions

### 📊 Evaluation

* RAGAS
* DeepEval
* Retrieval precision/recall
* Faithfulness evaluation
* Answer relevancy
* Automated evaluation datasets
* Regression testing

### 👁️ Observability

* LangSmith tracing
* Structured logging
* Latency tracking
* Token/cost tracking
* Retrieval metrics
* Agent decision monitoring

### 🚀 Productionization

* FastAPI backend
* Docker deployment
* Persistent production vector database
* Authentication
* Authorization
* Rate limiting
* Redis caching
* Automated testing
* CI/CD
* Health checks

---

# 🔒 Design Principles

RAGFury follows three core principles.

## 1. Route Before Processing

The system first determines which knowledge path should handle the question.

```text
Question
   ↓
Routing Decision
   ↓
Workflow
```

---

## 2. Never Trust Retrieval Blindly

A retrieved document is not automatically considered valid evidence.

```text
Retrieved
    ↓
Graded
    ↓
Trusted for generation
```

---

## 3. Recover From Failed Retrieval

When retrieval fails:

```text
Bad Retrieval
      ↓
Query Rewrite
      ↓
Better Retrieval Attempt
```

Instead of immediately producing an unsupported answer.

---

# 🧱 Repository Design

The codebase follows separation of responsibilities:

```text
Configuration
      ↓
Document Processing
      ↓
Semantic Chunking
      ↓
Vector Retrieval
      ↓
Graph State
      ↓
LangGraph Nodes
      ↓
Graph Builder
      ↓
Application
```

This makes individual components easier to modify, test, and replace.

For example, the retrieval backend can evolve independently from the graph orchestration layer.

---

# 📌 Important Implementation Notes

### Local Vector Database

The current dense retrieval layer uses:

```text
Chroma
```

with local persistence:

```text
./chroma_db
```

The local database is intentionally excluded from Git tracking.

### Embeddings

The same embedding family is used for semantic processing and dense retrieval:

```text
all-MiniLM-L6-v2
```

### LLM

The current configured model is:

```text
llama-3.1-8b-instant
```

through Groq.

---

# ⚠️ Current Limitations

RAGFury is an engineering-focused Agentic RAG implementation and is not yet a fully productionized enterprise platform.

Current limitations include:

* Local Chroma persistence
* No authentication/authorization layer
* No multi-user isolation
* No production API layer
* No automated evaluation pipeline
* No formal observability stack
* No production-grade rate limiting
* No automated test suite
* No distributed deployment architecture
* External-knowledge workflow still requires further integration with a real external retrieval source
* Streamlit integration should be kept synchronized with the current LangGraph application API

These are intentional extension points rather than hidden limitations.

---

# 🚀 Roadmap

```text
                    RAGFury
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     Retrieval      Safety        Evaluation
        │              │              │
   Reranking       Guardrails      RAGAS
   RRF             Injection       DeepEval
   HyDE            Detection       Benchmarks
        │              │              │
        └──────────────┼──────────────┘
                       │
                  Production
                       │
          ┌────────────┼────────────┐
          │            │            │
        FastAPI      Redis       Docker
          │            │            │
          └────────────┼────────────┘
                       │
                  Deployment
```

---

# 📚 Learning Outcomes

Building RAGFury provides hands-on experience with:

* Retrieval-Augmented Generation
* Agentic AI
* LangGraph
* LangChain
* Vector databases
* Dense retrieval
* Sparse retrieval
* Hybrid search
* Semantic chunking
* Embeddings
* LLM structured outputs
* Corrective RAG
* Query rewriting
* Self-reflection
* Stateful AI workflows
* Streamlit application development
* AI system architecture

---

# 👨‍💻 Author

**Aviral**

GitHub:

https://github.com/aviral-dot

Project:

https://github.com/aviral-dot/RAGFury-Agentic-Knowledge-Retrieval-Research-System

---

# ⭐ Support

If you find the project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

Add your preferred open-source license here.

For example:

```text
MIT License
```

if you decide to release the project under MIT.

