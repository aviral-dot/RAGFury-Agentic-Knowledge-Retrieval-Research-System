# 🚀 RAGFury — Agentic Knowledge Retrieval & Research System

> **An agentic Retrieval-Augmented Generation (RAG) system that combines private document retrieval with external knowledge search, using semantic chunking, hybrid dense-sparse retrieval, and a LangGraph ReAct agent for intelligent tool selection.**

RAGFury is an AI-powered knowledge retrieval and research system built with **Python, LangChain, LangGraph, ReAct agents, FAISS, BM25, Sentence Transformers, Groq, Wikipedia, and Streamlit**.

Unlike a conventional RAG pipeline that always follows:

```text
Query → Retrieve → Generate
```

RAGFury introduces an **agentic reasoning layer** that can select between:

* 🔎 **Private Knowledge Retrieval** — searches indexed PDF documents.
* 🌐 **External Knowledge Retrieval** — searches Wikipedia when general/external knowledge is required.

The system also uses **threshold-based semantic chunking** and **hybrid retrieval** to improve document search.

---

## 🎯 Why RAGFury?

Traditional RAG systems generally depend on a predetermined retrieval pipeline. This works well when the answer exists inside the indexed knowledge base, but is less useful when the required information is outside that corpus.

RAGFury addresses this by giving a ReAct agent access to multiple tools:

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   LangGraph ReAct   │
                         │       Agent         │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │ Retriever Tool   │       │ Wikipedia Tool  │
              │ Private PDFs     │       │ External Web KB │
              └────────┬────────┘       └────────┬────────┘
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │ FAISS + BM25    │       │ Wikipedia API   │
              └────────┬────────┘       └────────┬────────┘
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │    Final Answer     │
                         └─────────────────────┘
```

The agent determines **which knowledge source is appropriate for the query instead of blindly relying on a single retrieval source**.

---

# ✨ Key Features

### 📄 Private PDF Knowledge Base

* Load multiple PDF documents from the `data/` directory.
* Extract and process document content.
* Build a searchable private knowledge base.
* Answer document-specific questions from indexed content.

### ✂️ Threshold-Based Semantic Chunking

Instead of relying only on fixed-size chunks, RAGFury uses a semantic chunking strategy based on sentence-level embeddings.

The chunker:

1. Splits documents into sentences.
2. Generates sentence embeddings.
3. Calculates cosine similarity between consecutive sentences.
4. Keeps semantically related sentences together.
5. Starts a new chunk when similarity falls below the configured threshold.

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

This allows chunk boundaries to follow changes in meaning rather than only character count.

### 🔎 Hybrid Retrieval

RAGFury combines **dense semantic retrieval** with **sparse lexical retrieval**.

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Dense Retrieval    Sparse Retrieval
       FAISS               BM25
          │                 │
          └────────┬────────┘
                   ▼
          Hybrid Retrieval
                   │
                   ▼
          Relevant Documents
```

* **FAISS** captures semantic similarity.
* **BM25** captures keyword and exact-term relevance.
* The retrieval signals are combined through LangChain's ensemble retrieval mechanism.

### 🤖 ReAct Agent

The system uses a **LangGraph-powered ReAct agent** with tool calling.

The agent can select between:

```text
Retriever Tool
      ↓
Private PDF Knowledge Base
```

or:

```text
Wikipedia Tool
      ↓
External Knowledge
```

The decision is made inside the agent workflow based on the user's question.

### 🌐 External Knowledge Fallback

When information is not expected to come from the private document corpus, the agent can use Wikipedia.

Example:

```text
"What is quantum computing?"
```

can use external knowledge, while:

```text
"What is the company's leave policy?"
```

can use the private PDF knowledge base.

### 💬 Streamlit Interface

The Streamlit application provides:

* Interactive question answering
* AI-generated responses
* Retrieved source inspection
* Search history
* Response-time tracking

### 📚 Source Inspection

Retrieved document chunks can be inspected to understand which pieces of the private knowledge base contributed to the response.

### ⚡ Response-Time Tracking

The application records query processing time, providing basic visibility into retrieval and generation latency.

### 🧩 Modular Architecture

The implementation separates:

* Configuration
* Document ingestion
* Semantic chunking
* Vector storage and retrieval
* Agent state
* ReAct tools
* LangGraph orchestration
* Streamlit UI

---

# 🏗️ System Architecture

```text
                         ┌───────────────────────┐
                         │      Streamlit UI     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    RAGFury Engine      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   LangGraph ReAct     │
                         │        Agent          │
                         └───────────┬───────────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                          ▼                     ▼
                 ┌─────────────────┐   ┌─────────────────┐
                 │ Retriever Tool   │   │ Wikipedia Tool  │
                 └────────┬────────┘   └────────┬────────┘
                          │                     │
                          ▼                     ▼
                 ┌─────────────────┐   ┌─────────────────┐
                 │ FAISS + BM25    │   │ Wikipedia API   │
                 └────────┬────────┘   └─────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   PDF Corpus    │
                 │ Private KB      │
                 └─────────────────┘
```

---

# 🔄 End-to-End RAG Pipeline

```text
PDF Documents
      │
      ▼
PDF Loader
      │
      ▼
Document Processing
      │
      ▼
Sentence Splitting
      │
      ▼
Semantic Chunking
      │
      ▼
Sentence-Transformer Embeddings
      │
      ▼
 ┌────┴────┐
 ▼         ▼
FAISS     BM25
 │         │
 └────┬────┘
      ▼
Hybrid Retriever
      │
      ▼
Retriever Tool
      │
      ▼
LangGraph ReAct Agent
      │
 ┌────┴──────────────┐
 ▼                   ▼
Private KB       Wikipedia
 ▼                   ▼
 └────────┬──────────┘
          ▼
     Final Answer
```

---

# 🧠 Agentic RAG vs Traditional RAG

## Traditional RAG

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

The retrieval path is mostly predetermined.

## RAGFury

```text
User Query
    ↓
ReAct Agent
    ↓
Reasoning / Tool Selection
    │
    ├──► Retriever Tool
    │       ↓
    │   Private PDFs
    │
    └──► Wikipedia Tool
            ↓
        External Knowledge
    ↓
Final Answer
```

The key difference is the **agentic tool-selection layer**.

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

---

# 🧱 Core Components

| Component               | Responsibility                                                  |
| ----------------------- | --------------------------------------------------------------- |
| `document_processor.py` | Loads and prepares source documents                             |
| `semantic_chunker.py`   | Creates semantically coherent chunks using embedding similarity |
| `vectorstore.py`        | Builds dense/sparse retrieval and combines FAISS + BM25         |
| `reactnode.py`          | Defines the ReAct agent and retrieval/Wikipedia tools           |
| `graph_builder.py`      | Builds the LangGraph workflow                                   |
| `rag_state.py`          | Defines the state passed through the graph                      |
| `config.py`             | Centralizes model and environment configuration                 |
| `main.py`               | Runs the RAG pipeline                                           |
| `streamlit_app.py`      | Provides the interactive UI                                     |

---

# 🛠️ Technology Stack

| Technology                    | Purpose                                   |
| ----------------------------- | ----------------------------------------- |
| **Python**                    | Core application development              |
| **LangChain**                 | LLM, retrieval, tools, and RAG components |
| **LangGraph**                 | Agent workflow orchestration              |
| **ReAct**                     | Reasoning and tool selection              |
| **Groq**                      | LLM inference                             |
| **Llama 3.1 8B Instant**      | Generation model                          |
| **Sentence Transformers**     | Sentence/document embeddings              |
| **all-MiniLM-L6-v2**          | Embedding model                           |
| **FAISS**                     | Dense vector similarity search            |
| **BM25**                      | Sparse lexical retrieval                  |
| **PyPDF / LangChain loaders** | PDF document ingestion                    |
| **Wikipedia API**             | External knowledge retrieval              |
| **Streamlit**                 | Interactive web interface                 |
| **Hugging Face**              | Transformer/embedding model ecosystem     |
| **Git & GitHub**              | Version control                           |

---

# ⚙️ How It Works

## 1. Document Ingestion

PDF files placed inside the `data/` directory are loaded and converted into LangChain document objects.

```text
PDF
 ↓
PDF Loader
 ↓
Document Objects
```

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
Threshold
 ↓
Semantic Chunks
```

Semantically related sentences are grouped together, while major semantic changes create new chunks.

## 3. Embedding Generation

```text
Text Chunk
    ↓
all-MiniLM-L6-v2
    ↓
Vector Representation
```

## 4. Hybrid Indexing

```text
Chunks
  │
  ├──► FAISS → Dense Semantic Search
  │
  └──► BM25  → Sparse Keyword Search
```

## 5. Query Processing

```text
User Query
    ↓
LangGraph Workflow
    ↓
ReAct Agent
    ↓
Tool Selection
```

## 6. Private Knowledge Retrieval

```text
Query
 ↓
Retriever Tool
 ↓
FAISS + BM25
 ↓
Relevant PDF Chunks
 ↓
Agent
```

## 7. External Knowledge Retrieval

```text
Query
 ↓
Wikipedia Tool
 ↓
Wikipedia API
 ↓
External Information
 ↓
Agent
```

## 8. Response Generation

The retrieved information is incorporated into the agent's context and the LLM generates the final response.

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

> ⚠️ **Never commit `.env` files, API keys, tokens, or credentials to GitHub.**

---

# 📄 Add Your Documents

Place your PDF files inside:

```text
data/
```

Example:

```text
data/
├── company_policy.pdf
├── employee_handbook.pdf
├── technical_documentation.pdf
└── security_policy.pdf
```

---

# ▶️ Running the Application

## Run the RAG Pipeline

```bash
python main.py
```

## Launch the Streamlit UI

```bash
python -m streamlit run streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 💡 Example Queries

### 📚 Private Knowledge Base

```text
What is the company's leave policy?
```

```text
What is the employee notice period?
```

```text
What security requirements are mentioned in the document?
```

### 🌐 External Knowledge

```text
What is quantum computing?
```

```text
How does nuclear fusion work?
```

```text
Who is Brad Pitt?
```

---

# 🧪 Engineering Concepts Demonstrated

* Retrieval-Augmented Generation
* Agentic RAG
* ReAct agents
* LangGraph orchestration
* Tool calling
* Semantic chunking
* Sentence embeddings
* Dense retrieval
* Sparse retrieval
* Hybrid search
* FAISS vector search
* BM25 retrieval
* Document ingestion
* PDF processing
* LLM orchestration
* External knowledge retrieval
* State management
* Modular AI architecture
* Streamlit application development

---

# 📈 Future Improvements

### 🔎 Advanced Retrieval

* [ ] Cross-encoder reranking
* [ ] Query expansion
* [ ] HyDE retrieval
* [ ] Multi-query retrieval
* [ ] Reciprocal Rank Fusion
* [ ] Retrieval confidence scoring

### 🧠 Advanced Agentic RAG

* [ ] Query planning
* [ ] Corrective RAG
* [ ] Self-RAG
* [ ] Adaptive RAG
* [ ] Multi-step research agent

### 📊 Evaluation & Observability

* [ ] RAGAS evaluation
* [ ] DeepEval evaluation
* [ ] LangSmith tracing
* [ ] Retrieval metrics
* [ ] Answer faithfulness evaluation
* [ ] Latency monitoring

### 💾 Application Features

* [ ] Conversation memory
* [ ] Streaming responses
* [ ] Direct document upload
* [ ] User authentication
* [ ] User-specific knowledge bases
* [ ] Persistent chat history
* [ ] Citation generation

### 🚀 Productionization

* [ ] FastAPI backend
* [ ] Docker deployment
* [ ] Production vector database
* [ ] API authentication
* [ ] Rate limiting
* [ ] Structured logging
* [ ] Automated testing
* [ ] CI/CD pipeline

---

# 🏆 What Makes RAGFury Different?

RAGFury is more than a basic:

```text
PDF → Embeddings → Vector Search → LLM
```

pipeline.

It combines:

```text
Semantic Chunking
       +
Hybrid Retrieval
       +
FAISS
       +
BM25
       +
LangGraph
       +
ReAct Agents
       +
Tool Calling
       +
External Knowledge
       +
LLM Generation
```

This creates an **agentic knowledge-retrieval system capable of selecting between private document knowledge and external knowledge sources**.

---

# 📌 Resume-Ready Description

> **RAGFury — Agentic Knowledge Retrieval & Research System:** Built an Agentic RAG system using LangGraph and ReAct agents, combining threshold-based semantic chunking with hybrid FAISS + BM25 retrieval for private PDF knowledge bases and Wikipedia-based external knowledge retrieval. Integrated Groq/Llama 3.1 inference and Streamlit for interactive, source-aware question answering.

### Core Technologies

`Python` `LangChain` `LangGraph` `ReAct` `FAISS` `BM25` `Sentence Transformers` `Groq` `Llama 3.1` `Wikipedia API` `Streamlit`

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

---

# ⭐ Support

If you find RAGFury useful or interesting, consider giving the repository a ⭐ on GitHub.

**Repository:**
https://github.com/aviral-dot/RAGFury-Agentic-Knowledge-Retrieval-Research-System

---

## 📜 License

This project is intended for educational and research purposes.


