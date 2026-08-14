# 🚀 RAGFury — Agentic Knowledge Retrieval & Research System

> **An agentic Retrieval-Augmented Generation (RAG) system that intelligently chooses between private document knowledge and external knowledge, while using semantic chunking, hybrid retrieval, document relevance grading, query rewriting, and LangGraph ReAct agents for more reliable question answering.**

RAGFury is an **Agentic RAG and knowledge-research system** built with Python, LangChain, LangGraph, ReAct agents, FAISS, BM25, Sentence Transformers, Groq, Wikipedia, and Streamlit.

The system is designed around an important problem in real-world RAG:

> **What should the system do when the user's question may belong to the private knowledge base, but the retrieved documents are not actually relevant?**

Instead of blindly generating an answer from the first retrieved documents, RAGFury introduces **agentic source selection and corrective retrieval**.

The system can:

* 🔎 Search a private PDF knowledge base.
* 🌐 Search Wikipedia for external knowledge.
* 🧠 Dynamically select the appropriate knowledge source using a ReAct agent.
* 📊 Grade retrieved documents for relevance.
* ✏️ Rewrite unsuccessful queries.
* 🔄 Retry retrieval when the retrieved context is not relevant.
* 🧩 Use semantic chunking instead of only fixed-size splitting.
* 🔗 Combine dense FAISS retrieval with sparse BM25 retrieval.
* 💬 Provide an interactive Streamlit interface.

---

# 🎯 Why RAGFury?

A conventional RAG pipeline usually looks like:

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

This approach assumes that:

1. the query belongs to the indexed knowledge base, and
2. the retriever will return relevant documents.

In practice, both assumptions can fail.

For example, suppose the indexed PDF contains information about:

```text
Leave Policy
Remote Work Policy
Employee Benefits
```

and the user asks:

```text
What is the company's security policy?
```

The retriever may still return documents because they contain words such as:

```text
company
employee
policy
```

But those documents may have **nothing to do with security**.

RAGFury addresses this problem through two levels of decision-making:

### Level 1 — Agentic Knowledge-Source Selection

A LangGraph ReAct agent decides whether the query should be handled using:

```text
Private Retriever Tool
```

or:

```text
Wikipedia Tool
```

### Level 2 — Corrective Retrieval

When the private retriever is selected, retrieved documents are evaluated for relevance.

```text
Retrieve
   ↓
Grade Documents
   ↓
Relevant?
 ┌─┴──────────────┐
 │                │
Yes               No
 │                │
 ▼                ▼
Generate       Rewrite Query
                  │
                  ▼
               Retrieve
                  │
                  ▼
              Grade Again
```

This makes the RAG pipeline more robust than simply retrieving once and generating immediately.

---

# 🧠 Core Architecture

The high-level architecture of RAGFury is:

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   LangGraph ReAct    │
                         │        Agent         │
                         └──────────┬───────────┘
                                    │
                     Agentic Tool Selection
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐           ┌───────────────────┐
          │  Retriever Tool   │           │  Wikipedia Tool   │
          │   Private PDFs    │           │ External Knowledge│
          └─────────┬─────────┘           └─────────┬─────────┘
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐               Wikipedia API
          │  Hybrid Retrieval │                     │
          │   FAISS + BM25    │                     │
          └─────────┬─────────┘                     │
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐              ┌────────────────┐
          │ Document Grader   │              │    Generate    │
          └─────────┬─────────┘              └───────┬────────┘
                    │                                │
              ┌─────┴─────┐                          │
              │           │                          │
           Relevant    Irrelevant                    │
              │           │                          │
              ▼           ▼                          │
          Generate   Rewrite Query                  │
                          │                          │
                          ▼                          │
                       Retrieve                     │
                          │                          │
                          └────────────┬─────────────┘
                                       ▼
                              ┌─────────────────┐
                              │   Final Answer  │
                              └─────────────────┘
```

The key architectural principle is:

> **The ReAct agent selects the knowledge source. The selected retrieval workflow then performs the appropriate retrieval and validation steps.**

There is no need for a separate deterministic `rag vs wikipedia` router.

---

# ✨ Key Features

## 🤖 1. Agentic Knowledge-Source Selection

RAGFury uses a **LangGraph ReAct agent** with multiple tools.

The agent has access to:

```text
┌──────────────────────────┐
│ Retriever Tool           │
│ Private PDF Knowledge    │
└──────────────────────────┘

┌──────────────────────────┐
│ Wikipedia Tool           │
│ External Knowledge       │
└──────────────────────────┘
```

The agent determines which tool is appropriate for the user's question.

For example:

```text
"What is the employee leave policy?"
```

can be handled by:

```text
Retriever Tool
```

while:

```text
"What is quantum computing?"
```

can be handled by:

```text
Wikipedia Tool
```

This makes the system more flexible than a fixed RAG pipeline.

---

# 📄 2. Private PDF Knowledge Base

RAGFury supports PDF-based private knowledge retrieval.

Documents placed in:

```text
data/
```

are processed and transformed into searchable knowledge.

The ingestion pipeline is:

```text
PDF Documents
      ↓
PDF Loading
      ↓
Document Processing
      ↓
Sentence Splitting
      ↓
Semantic Chunking
      ↓
Embeddings
      ↓
Hybrid Retrieval Index
```

The private knowledge base is completely separate from the external Wikipedia source.

---

# ✂️ 3. Threshold-Based Semantic Chunking

Instead of relying only on fixed-size chunks, RAGFury uses a **sentence-level semantic chunking strategy**.

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

Consecutive sentences are compared using their embeddings.

If the semantic similarity remains sufficiently high, the sentences remain in the same chunk.

When the similarity drops below the configured threshold, a new chunk is created.

This allows chunk boundaries to better follow **changes in semantic meaning**.

The project uses:

```text
Embedding Model:
all-MiniLM-L6-v2
```

with the configured semantic similarity threshold.

---

# 🔎 4. Hybrid Retrieval

The private document retriever combines two retrieval strategies:

```text
                    Query
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       Dense Search       Sparse Search
          FAISS                BM25
             │                 │
             └────────┬────────┘
                      ▼
              Hybrid Retrieval
                      │
                      ▼
              Candidate Documents
```

### FAISS

FAISS provides dense vector similarity search.

It is useful for finding documents that are **semantically similar** to the query even when the exact keywords are different.

### BM25

BM25 provides sparse lexical retrieval.

It is useful when exact words, phrases, identifiers, or domain-specific terminology matter.

### Combined Retrieval

The two retrieval signals are combined into a hybrid retriever.

This provides both:

```text
Semantic similarity
```

and:

```text
Lexical relevance
```

rather than relying on only one retrieval strategy.

---

# 📊 5. Document Relevance Grading

One of the important reliability features of RAGFury is **retrieved-document grading**.

A retriever can return documents that are technically similar to a query but irrelevant to the actual question.

Therefore, after retrieval, the system evaluates whether the retrieved context is relevant.

```text
Query
  ↓
Retrieve Documents
  ↓
Document Grader
  ↓
Is Context Relevant?
```

The grader produces a relevance decision.

### Relevant Documents

```text
Retrieve
   ↓
Grade
   ↓
Relevant
   ↓
Generate Answer
```

### Irrelevant Documents

```text
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

This prevents the system from immediately generating an answer from obviously unrelated retrieved context.

---

# ✏️ 6. Query Rewriting

When the retrieved documents are judged irrelevant, RAGFury does not simply give up after the first retrieval attempt.

Instead, the query can be rewritten to improve retrieval.

```text
Original Query
      ↓
Retrieve
      ↓
Grade
      ↓
Irrelevant
      ↓
Query Rewriter
      ↓
Improved Query
      ↓
Retrieve Again
```

The goal is to transform an unsuccessful query into one that is more useful for the document retriever.

For example:

```text
Original:
"What is the company's security policy?"
```

may be reformulated into a more retrieval-oriented query based on the available knowledge context.

The retry process is bounded so that the graph does not enter an uncontrolled retrieval loop.

---

# 🔄 7. Corrective RAG Workflow

The private-document path can therefore be viewed as a **Corrective RAG-style workflow**:

```text
                 User Query
                     │
                     ▼
                  Retrieve
                     │
                     ▼
               Grade Documents
                     │
              ┌──────┴──────┐
              │             │
          Relevant       Irrelevant
              │             │
              ▼             ▼
          Generate      Rewrite Query
                            │
                            ▼
                         Retrieve
                            │
                            ▼
                          Grade
                            │
                         bounded
                          retry
```

This is one of the main differences between RAGFury and a basic:

```text
Retrieve → Generate
```

system.

---

# 🌐 8. External Knowledge Retrieval

For questions that require general or external knowledge, the ReAct agent can use the Wikipedia tool.

```text
User Query
    ↓
ReAct Agent
    ↓
Wikipedia Tool
    ↓
Wikipedia API
    ↓
Retrieved Information
    ↓
LLM
    ↓
Final Answer
```

Example:

```text
"What is quantum computing?"
```

```text
"How does nuclear fusion work?"
```

These questions do not necessarily require the private PDF knowledge base.

---

# 💬 9. Streamlit Interface

RAGFury includes a Streamlit interface for interacting with the system.

The interface provides:

* Interactive question answering
* AI-generated responses
* Retrieved source inspection
* Search history
* Response-time tracking

This makes the retrieval and agentic reasoning system accessible through a simple web interface.

---

# 📚 10. Source Inspection

For private-document questions, retrieved document chunks can be inspected.

This provides visibility into the information that was retrieved from the knowledge base.

Instead of only seeing:

```text
Answer
```

the user can inspect:

```text
Answer
+
Retrieved Context
```

This is useful for debugging retrieval quality and understanding why a response was generated.

---

# ⚡ 11. Response-Time Tracking

The Streamlit application tracks query processing time.

This provides basic visibility into:

```text
Query
   ↓
Agent
   ↓
Retrieval
   ↓
Generation
   ↓
Response Time
```

This is useful when experimenting with different retrieval and agent configurations.

---

# 🏗️ System Architecture

```text
                         ┌───────────────────────┐
                         │      Streamlit UI     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     RAGFury Engine     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   LangGraph ReAct     │
                         │        Agent          │
                         └───────────┬───────────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                         ▼                       ▼
                ┌─────────────────┐     ┌─────────────────┐
                │ Retriever Tool  │     │ Wikipedia Tool  │
                └────────┬────────┘     └────────┬────────┘
                         │                       │
                         ▼                       ▼
                ┌─────────────────┐       Wikipedia API
                │ Hybrid Retrieval│
                │ FAISS + BM25    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Document Grader │
                └────────┬────────┘
                         │
                    ┌────┴────┐
                    │         │
                 Relevant   Irrelevant
                    │         │
                    ▼         ▼
                Generate   Rewrite
                              │
                              ▼
                           Retrieve
                              │
                              ▼
                             Grade
```

---

# 🔄 End-to-End Pipeline

## Phase 1 — Document Ingestion

```text
PDF
 ↓
Document Loader
 ↓
Document Objects
```

## Phase 2 — Semantic Processing

```text
Document
 ↓
Sentence Splitting
 ↓
Sentence Embeddings
 ↓
Cosine Similarity
 ↓
Threshold-Based Chunking
 ↓
Semantic Chunks
```

## Phase 3 — Indexing

```text
Semantic Chunks
       │
       ├──────────────► FAISS
       │
       └──────────────► BM25
```

## Phase 4 — Agentic Query Processing

```text
User Query
    ↓
LangGraph ReAct Agent
    ↓
Tool Selection
```

The agent chooses:

```text
Retriever Tool
```

or:

```text
Wikipedia Tool
```

## Phase 5 — Private RAG

```text
Retriever Tool
      ↓
FAISS + BM25
      ↓
Retrieved Documents
      ↓
Document Grader
      │
 ┌────┴─────┐
 │          │
Relevant  Irrelevant
 │          │
 ▼          ▼
Generate  Rewrite
             │
             ▼
          Retrieve
```

## Phase 6 — External Retrieval

```text
Wikipedia Tool
      ↓
Wikipedia API
      ↓
External Context
      ↓
Generate
```

---

# 🧠 Agentic RAG vs Traditional RAG

## Traditional RAG

```text
User Query
    ↓
Retriever
    ↓
Context
    ↓
LLM
    ↓
Answer
```

The retrieval path is predetermined.

---

## RAGFury

```text
User Query
    ↓
ReAct Agent
    ↓
Tool Selection
    │
    ├───────────────► Retriever Tool
    │                       ↓
    │                  Hybrid Retrieval
    │                       ↓
    │                  Document Grader
    │                       ↓
    │              ┌────────┴────────┐
    │              │                 │
    │          Relevant          Irrelevant
    │              │                 │
    │              ▼                 ▼
    │           Generate          Rewrite
    │                                │
    │                                ▼
    │                             Retrieve
    │
    └───────────────► Wikipedia Tool
                            ↓
                       Wikipedia
                            ↓
                         Generate
```

The important difference is that RAGFury has **decision-making at the knowledge-source level and corrective behavior inside the private RAG workflow**.

---

# 📂 Project Structure

```text
RAGFury-Agentic-Knowledge-Retrieval-Research-System/
│
├── data/
│   └── *.pdf
│
├── src/
│   ├── config/
│   │   └── config.py
│   │
│   ├── document_ingestion/
│   │   └── document_processor.py
│   │
│   ├── semantic_chunker/
│   │   └── semantic_chunker.py
│   │
│   ├── vectorstore/
│   │   └── vectorstore.py
│   │
│   ├── graph_builder/
│   │   └── graph_builder.py
│   │
│   ├── node/
│   │   └── reactnode.py
│   │
│   └── state/
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

The repository currently follows this modular organization, with separate modules for configuration, document ingestion, semantic chunking, vector storage, graph construction, nodes, and graph state.

---

# 🧱 Core Components

| Component               | Responsibility                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `document_processor.py` | Loads and prepares PDF source documents                                                 |
| `semantic_chunker.py`   | Creates semantically coherent chunks using sentence embeddings                          |
| `vectorstore.py`        | Builds dense/sparse retrieval using FAISS and BM25                                      |
| `reactnode.py`          | Defines the ReAct agent, retrieval tool, Wikipedia tool, and answer-generation behavior |
| `graph_builder.py`      | Builds and compiles the LangGraph workflow                                              |
| `rag_state.py`          | Defines the state passed through the LangGraph workflow                                 |
| `config.py`             | Centralizes model and environment configuration                                         |
| `main.py`               | Initializes and runs the application                                                    |
| `streamlit_app.py`      | Provides the interactive Streamlit interface                                            |

The current repository's `graph_builder.py` uses LangGraph `StateGraph` and the project state is passed through `RAGState`.

---

# 🛠️ Technology Stack

| Technology                    | Purpose                                   |
| ----------------------------- | ----------------------------------------- |
| **Python**                    | Core application development              |
| **LangChain**                 | LLM, retrieval, tools, and RAG components |
| **LangGraph**                 | Agent/workflow orchestration              |
| **ReAct**                     | Agentic reasoning and tool selection      |
| **Groq**                      | LLM inference                             |
| **Llama 3.1 8B Instant**      | Generation model                          |
| **Sentence Transformers**     | Sentence/document embeddings              |
| **all-MiniLM-L6-v2**          | Embedding model                           |
| **FAISS**                     | Dense vector retrieval                    |
| **BM25**                      | Sparse lexical retrieval                  |
| **PyPDF / LangChain loaders** | PDF ingestion                             |
| **Wikipedia API**             | External knowledge retrieval              |
| **Streamlit**                 | Interactive web interface                 |
| **Hugging Face**              | Transformer/embedding ecosystem           |
| **uv**                        | Python dependency/environment management  |
| **Git & GitHub**              | Version control                           |

---

# ⚙️ How It Works

## 1. Document Ingestion

PDF files placed inside:

```text
data/
```

are loaded and converted into document objects.

```text
PDF
 ↓
PDF Loader
 ↓
Document Objects
```

---

## 2. Semantic Chunking

```text
Document
 ↓
Sentences
 ↓
Sentence Embeddings
 ↓
Cosine Similarity
 ↓
Similarity Threshold
 ↓
Semantic Chunks
```

The semantic chunker groups sentences that are sufficiently related and creates new chunks when the semantic relationship changes.

---

## 3. Embedding Generation

```text
Text Chunk
    ↓
all-MiniLM-L6-v2
    ↓
Vector Representation
```

---

## 4. Hybrid Indexing

```text
Chunks
  │
  ├──► FAISS → Dense Semantic Search
  │
  └──► BM25  → Sparse Keyword Search
```

The two retrieval approaches complement each other:

```text
FAISS → semantic similarity
BM25  → lexical/exact-term matching
```

---

## 5. Agentic Query Processing

```text
User Query
    ↓
LangGraph ReAct Agent
    ↓
Tool Selection
```

The agent selects the most appropriate available knowledge source.

---

## 6. Private Knowledge Retrieval

```text
Query
 ↓
Retriever Tool
 ↓
FAISS + BM25
 ↓
Retrieved Documents
 ↓
Document Grader
```

If relevant:

```text
Relevant
   ↓
Generate
```

If irrelevant:

```text
Irrelevant
   ↓
Rewrite Query
   ↓
Retrieve Again
   ↓
Grade Again
```

---

## 7. External Knowledge Retrieval

```text
Query
 ↓
ReAct Agent
 ↓
Wikipedia Tool
 ↓
Wikipedia API
 ↓
External Information
 ↓
Generate
```

---

## 8. Final Answer

After the selected workflow retrieves and validates the required information, the LLM generates the final response.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/aviral-dot/RAGFury-Agentic-Knowledge-Retrieval-Research-System.git

cd RAGFury-Agentic-Knowledge-Retrieval-Research-System
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

# 📦 Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using `uv`:

```bash
uv sync
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

> ⚠️ Never commit `.env` files, API keys, tokens, or credentials to GitHub.

---

# 📄 Add Your Documents

Place PDF documents inside:

```text
data/
```

For example:

```text
data/
├── company_policy.pdf
├── employee_handbook.pdf
└── technical_documentation.pdf
```

The system builds its private knowledge base from these documents.

> **Important:** A query should only be answered from the private knowledge base when the indexed documents actually contain relevant information. If a document does not contain information about a requested topic, the retrieval grader can identify the mismatch and trigger query rewriting rather than treating unrelated documents as valid evidence.

---

# ▶️ Running the Application

## Run the Application

```bash
python main.py
```

## Launch the Streamlit UI

```bash
python -m streamlit run streamlit_app.py
```

The Streamlit application will be available at:

```text
http://localhost:8501
```

---

# 💡 Example Queries

## 📚 Private Knowledge Base

```text
What is the company's leave policy?
```

```text
What is the employee notice period?
```

```text
What benefits are mentioned in the employee handbook?
```

---

## 🔍 Queries That Test Corrective Retrieval

```text
What is the company's security policy?
```

If the indexed documents do not contain a security policy, the system should not blindly treat unrelated policy documents as evidence.

Instead:

```text
Query
 ↓
Retrieve
 ↓
Grade
 ↓
Irrelevant
 ↓
Rewrite
 ↓
Retrieve Again
```

This makes such queries useful for testing retrieval quality.

---

## 🌐 External Knowledge

```text
What is quantum computing?
```

```text
How does nuclear fusion work?
```

```text
Who is Brad Pitt?
```

These can be handled through the Wikipedia tool when the ReAct agent determines that external knowledge is appropriate.

---

# 🧪 Engineering Concepts Demonstrated

RAGFury demonstrates several practical AI engineering concepts:

### Retrieval

* Retrieval-Augmented Generation
* Dense retrieval
* Sparse retrieval
* Hybrid search
* FAISS
* BM25

### Document Processing

* PDF ingestion
* Sentence splitting
* Semantic chunking
* Sentence embeddings
* Cosine similarity

### Agentic AI

* LangGraph
* ReAct agents
* Tool calling
* Agentic tool selection
* Multi-source knowledge retrieval

### Retrieval Reliability

* Document relevance grading
* Corrective retrieval
* Query rewriting
* Bounded retrieval retries

### LLM Engineering

* LLM orchestration
* Groq inference
* Prompt-based generation
* State management

### Application Development

* Streamlit
* Modular Python architecture
* Configuration management
* Response-time tracking
* Source inspection

---

# 📊 Current Architecture vs Basic RAG

| Capability                 |   Basic RAG | RAGFury |
| -------------------------- | ----------: | ------: |
| PDF ingestion              |           ✅ |       ✅ |
| Semantic chunking          |   Sometimes |       ✅ |
| Dense retrieval            |           ✅ |       ✅ |
| Sparse retrieval           | Usually not |       ✅ |
| Hybrid retrieval           | Usually not |       ✅ |
| External knowledge         |           ❌ |       ✅ |
| Agentic tool selection     |           ❌ |       ✅ |
| Document relevance grading |           ❌ |       ✅ |
| Query rewriting            |           ❌ |       ✅ |
| Corrective retrieval       |           ❌ |       ✅ |
| Streamlit UI               |    Optional |       ✅ |
| Source inspection          |    Optional |       ✅ |
| Response-time tracking     |    Optional |       ✅ |

---

# 🏆 What Makes RAGFury Different?

RAGFury is more than:

```text
PDF
 ↓
Embeddings
 ↓
Vector Search
 ↓
LLM
```

It combines:

```text
Semantic Chunking
       +
Sentence Embeddings
       +
FAISS
       +
BM25
       +
Hybrid Retrieval
       +
ReAct Agent
       +
LangGraph
       +
Tool Calling
       +
Document Grading
       +
Query Rewriting
       +
Corrective Retrieval
       +
Wikipedia
       +
LLM Generation
```

The result is an **agentic knowledge-retrieval system that can select between private and external knowledge sources and recover from poor private-document retrieval through relevance grading and query rewriting.**

---

# 🔬 Retrieval Failure Handling

One of the important design goals of RAGFury is to avoid this failure pattern:

```text
User Query
    ↓
Bad Retrieval
    ↓
Irrelevant Context
    ↓
LLM
    ↓
Confident but Unsupported Answer
```

Instead, the private RAG workflow attempts:

```text
User Query
    ↓
Retrieve
    ↓
Grade
    ↓
Relevant?
   ┌┴───────────────┐
   │                │
  YES               NO
   │                │
   ▼                ▼
Generate         Rewrite
                    │
                    ▼
                 Retrieve
                    │
                    ▼
                   Grade
```

This introduces an explicit **retrieval-quality checkpoint** before generation.

---

# 🧩 Design Philosophy

RAGFury follows three important principles.

### 1. Let the Agent Choose the Knowledge Source

Instead of creating a large collection of hard-coded routing rules:

```text
if query_about_documents:
    use_rag()

elif general_question:
    use_wikipedia()
```

the ReAct agent is given tools and decides which tool is appropriate.

---

### 2. Retrieval Is Not Automatically Correct

A retriever returning documents does not mean that those documents answer the question.

Therefore:

```text
Retrieved ≠ Relevant
```

RAGFury explicitly grades retrieved documents before relying on them.

---

### 3. Failed Retrieval Should Trigger Recovery

When retrieval is poor:

```text
Bad Retrieval
     ↓
Query Rewrite
     ↓
Better Retrieval Attempt
```

rather than immediately generating an answer from weak context.

---

# 📈 Future Improvements

The following features are **not presented as currently implemented** and can be added in future iterations.

## 🔎 Advanced Retrieval

* [ ] Cross-encoder reranking
* [ ] Query expansion
* [ ] HyDE retrieval
* [ ] Multi-query retrieval
* [ ] Reciprocal Rank Fusion
* [ ] Retrieval confidence scoring

## 🧠 Advanced Agentic RAG

* [ ] Multi-step research planning
* [ ] Multi-agent research
* [ ] Adaptive retrieval strategies
* [ ] More sophisticated self-correction
* [ ] Multiple external knowledge sources

## 📊 Evaluation & Observability

* [ ] RAGAS evaluation
* [ ] DeepEval evaluation
* [ ] LangSmith tracing
* [ ] Retrieval precision/recall evaluation
* [ ] Answer faithfulness evaluation
* [ ] Automated evaluation datasets

## 💾 Application Features

* [ ] Conversation memory
* [ ] Streaming responses
* [ ] Direct document upload
* [ ] User authentication
* [ ] User-specific knowledge bases
* [ ] Persistent chat history
* [ ] Structured citations

## 🚀 Productionization

* [ ] FastAPI backend
* [ ] Docker deployment
* [ ] Production vector database
* [ ] API authentication
* [ ] Rate limiting
* [ ] Structured logging
* [ ] Automated testing
* [ ] CI/CD pipeline

---

# 📌 Resume-Ready Description

> **RAGFury — Agentic Knowledge Retrieval & Research System:** Built an Agentic RAG system using LangGraph ReAct agents to dynamically select between private PDF retrieval and Wikipedia-based external knowledge. Implemented threshold-based semantic chunking with `all-MiniLM-L6-v2`, hybrid FAISS + BM25 retrieval, document relevance grading, query rewriting, and corrective retrieval to improve robustness against irrelevant retrieved context. Integrated Groq/Llama 3.1 inference and Streamlit for interactive question answering and retrieval inspection.

### Core Technologies

```text
Python
LangChain
LangGraph
ReAct
FAISS
BM25
Sentence Transformers
all-MiniLM-L6-v2
Groq
Llama 3.1
Wikipedia API
Streamlit
```

---

# 👨‍💻 Author

**Aviral**

Focused on:

* 🤖 Agentic AI
* 🧠 Generative AI
* 📚 Retrieval-Augmented Generation
* 🔎 Semantic Search
* 🔗 LangChain
* 🕸️ LangGraph
* 🛠️ ReAct Agents
* 🔍 Hybrid Retrieval
* 🧩 LLM Tool Calling
* 📊 AI/RAG Evaluation

---

# ⭐ Support

If you find RAGFury useful or interesting, consider giving the repository a ⭐ on GitHub.

**Repository:**

https://github.com/aviral-dot/RAGFury-Agentic-Knowledge-Retrieval-Research-System

---

## 📜 License

This project is intended for educational and research purposes.



