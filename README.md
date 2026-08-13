🚀 RAGForge — Agentic RAG Knowledge Retrieval & Research System

An agentic Retrieval-Augmented Generation (RAG) system that intelligently decides how to answer a query by combining private PDF knowledge retrieval with external web-based knowledge.

RAGForge is an AI-powered knowledge retrieval and research system built with LangGraph, ReAct agents, LangChain, vector search, and Streamlit.

Unlike a traditional RAG pipeline that always follows a fixed:

Query → Retrieve → Generate

workflow, RAGForge introduces an agentic reasoning layer that dynamically decides whether a question should be answered using the user's private document knowledge base or external knowledge through Wikipedia.

🎯 Why RAGForge?

Traditional RAG systems are highly dependent on a single retrieval pipeline.

If the required information is not present in the indexed documents, the system may return an incomplete answer or hallucinate.

RAGForge addresses this by giving the LLM access to multiple tools:

                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   ReAct AI Agent    │
                         │   LangGraph         │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌───────────────────┐        ┌────────────────────┐
          │ Retriever Tool    │        │ Wikipedia Tool     │
          │                   │        │                    │
          │ Private PDFs      │        │ External Knowledge │
          └─────────┬─────────┘        └──────────┬─────────┘
                    │                             │
                    ▼                             ▼
          ┌───────────────────┐        ┌────────────────────┐
          │ Vector Store      │        │ Wikipedia API      │
          └─────────┬─────────┘        └──────────┬─────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │    Final Answer     │
                         └─────────────────────┘

The agent decides which source is appropriate for the query instead of blindly retrieving from one knowledge base.

✨ Key Features
📄 Private PDF Knowledge Base
Load multiple PDF documents from the data/ directory.
Automatically process and prepare documents for retrieval.
Supports document-centric question answering.
✂️ Intelligent Document Processing
PDF document loading
Text extraction
Recursive character-based chunking
Chunk generation for efficient semantic retrieval
🔍 Semantic Retrieval
Converts document chunks into embeddings.
Stores embeddings in a vector database.
Retrieves semantically relevant chunks for user queries.
🤖 ReAct Agent

Powered by LangGraph, the system uses an agentic workflow that can reason about the user's request and select the appropriate tool.

The agent can choose between:

Retriever Tool
      ↓
Private PDF Knowledge Base

or

Wikipedia Tool
      ↓
External Knowledge
🧠 Agentic Decision Making

Instead of using a fixed retrieval pipeline:

Query
 ↓
Retriever
 ↓
LLM
 ↓
Answer

RAGForge follows:

Query
 ↓
ReAct Agent
 ↓
Reason about the task
 ↓
Select appropriate tool
 ↓
Retrieve information
 ↓
Generate response
🌐 External Knowledge Fallback

When a question requires general knowledge outside the private document collection, the agent can use Wikipedia.

Example:

"What is quantum computing?"

can be handled using external knowledge.

While:

"What is the company's leave policy?"

can be answered from the indexed PDF knowledge base.

💬 Streamlit Interface

Provides an interactive interface for:

Asking questions
Receiving AI-generated responses
Inspecting retrieved source documents
Monitoring response processing time
📚 Source Inspection

Retrieved document chunks can be inspected to understand where the answer originated from.

⚡ Response Time Tracking

The application tracks query processing time, providing basic visibility into system performance.

🧩 Modular Architecture

The project separates:

Document ingestion
Vector storage
Agent state
LangGraph workflow
Configuration
User interface

This makes the system easier to extend and maintain.

🏗️ System Architecture
                         ┌───────────────────────┐
                         │      Streamlit UI     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    RAGForge Engine    │
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
              ┌──────────────────┐   ┌──────────────────┐
              │ Retriever Tool   │   │ Wikipedia Tool   │
              └────────┬─────────┘   └────────┬─────────┘
                       │                      │
                       ▼                      ▼
              ┌──────────────────┐   ┌──────────────────┐
              │  Vector Store    │   │  Wikipedia API   │
              └────────┬─────────┘   └──────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   PDF Documents  │
              │  Private KB      │
              └──────────────────┘
🔄 RAG Pipeline

RAGForge processes documents through the following pipeline:

PDF Documents
      │
      ▼
PDF Loader
      │
      ▼
Document Processing
      │
      ▼
Recursive Character Chunking
      │
      ▼
Embeddings
      │
      ▼
Vector Store
      │
      ▼
Retriever
      │
      ▼
Retriever Tool
      │
      ▼
LangGraph ReAct Agent
      │
      ├───────────────► Private Knowledge
      │
      └───────────────► Wikipedia
                       External Knowledge
      │
      ▼
Final Answer
🧠 Agentic RAG vs Traditional RAG
Traditional RAG
User Query
    ↓
Retriever
    ↓
Retrieved Context
    ↓
LLM
    ↓
Answer

The retrieval mechanism is mostly predetermined.

RAGForge
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

This allows the system to dynamically select the most appropriate knowledge source.

📂 Project Structure
RAGFury-Agentic-Knowledge-Retrieval-Research-System/
│
├── data/
│   └── *.pdf
│
├── src/
│   │
│   ├── config/
│   │   └── config.py
│   │
│   ├── document_ingestion/
│   │   └── document_processor.py
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
🛠️ Tech Stack
Technology	Purpose
Python	Core application development
LangChain	LLM and RAG components
LangGraph	Agent workflow orchestration
ReAct Agents	Reasoning and tool selection
Vector Database	Semantic document retrieval
Embeddings	Semantic representation of documents
PyPDF	PDF document processing
Wikipedia API	External knowledge retrieval
Streamlit	Interactive web interface
Hugging Face	Embedding / NLP model support
Git & GitHub	Version control
⚙️ How It Works
1. Document Ingestion

PDF files placed inside the data/ directory are loaded by the document ingestion pipeline.

PDF
 ↓
Text Extraction
 ↓
Document Objects
2. Document Chunking

Large documents are divided into smaller chunks using recursive character-based splitting.

Large Document
      ↓
 ┌────┼────┐
 ▼    ▼    ▼
Chunk Chunk Chunk

This allows the retrieval system to search for focused pieces of information rather than entire documents.

3. Embedding Generation

Each document chunk is converted into a numerical vector representation.

Text Chunk
    ↓
Embedding Model
    ↓
Vector Representation
4. Vector Storage

The generated embeddings are stored in a vector store for semantic similarity search.

5. Query Processing

When a user asks a question:

User Query
    ↓
ReAct Agent
    ↓
Tool Selection

The agent determines whether the query requires:

Private document retrieval
External Wikipedia knowledge
6. Retrieval

For document-based questions, the retriever searches the vector store and returns the most relevant chunks.

7. Response Generation

The retrieved information is passed back into the agent workflow, which generates the final answer.

💻 Installation
1. Clone the Repository
git clone https://github.com/aviral-dot/RAGFury-Agentic-Knowledge-Retrieval-Research-System.git

cd RAGFury-Agentic-Knowledge-Retrieval-Research-System
2. Create a Virtual Environment
Windows
python -m venv .venv

Activate it:

.venv\Scripts\activate
Linux / macOS
python -m venv .venv
source .venv/bin/activate
📦 Install Dependencies

Using pip:

pip install -r requirements.txt

Or, if using uv:

uv sync
🔐 Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token

Replace the values with your own API credentials.

⚠️ Never commit .env files, API keys, tokens, or credentials to GitHub.

📄 Add Your Documents

Place PDF files inside:

data/

Example:

data/
├── company_policy.pdf
├── employee_handbook.pdf
├── technical_documentation.pdf
└── security_policy.pdf

The ingestion pipeline processes the documents from this directory.

▶️ Running the Application
Run the RAG Pipeline
python main.py
Launch the Streamlit UI
python -m streamlit run streamlit_app.py

The application will be available at:

http://localhost:8501
💡 Example Queries
📚 Private Knowledge Base
What is the company's leave policy?
What is the employee notice period?
What security requirements are mentioned in the document?

The agent can use the Retriever Tool to search the private PDF knowledge base.

🌐 External Knowledge
Who is Brad Pitt?
What is quantum computing?
How does nuclear fusion work?

The agent can use the Wikipedia Tool when external knowledge is required.

🔬 Example Agent Flow

For a document-specific query:

User
 │
 │ "What is the leave policy?"
 ▼
ReAct Agent
 │
 │ Select Retriever Tool
 ▼
Vector Search
 │
 ▼
Relevant PDF Chunks
 │
 ▼
LLM
 │
 ▼
Final Answer

For a general knowledge query:

User
 │
 │ "What is quantum computing?"
 ▼
ReAct Agent
 │
 │ Select Wikipedia Tool
 ▼
Wikipedia
 │
 ▼
Retrieved Information
 │
 ▼
LLM
 │
 ▼
Final Answer
🧪 Engineering Concepts Demonstrated

This project demonstrates practical AI engineering concepts including:

Retrieval-Augmented Generation
Agentic RAG
ReAct Agents
LangGraph workflows
Tool calling
Semantic search
Vector embeddings
Vector databases
Document ingestion
Document chunking
LLM orchestration
External knowledge retrieval
State management
Modular AI architecture
Streamlit application development
📈 Future Improvements

The architecture is designed to support more advanced production-oriented RAG capabilities.

🔎 Advanced Retrieval

Hybrid Search — BM25 + Vector Search

Cross-Encoder Reranking

Semantic Chunking

Query Expansion

HyDE Retrieval

Multi-query Retrieval

🧠 Advanced Agentic RAG

Query Planning

Corrective RAG

Self-RAG

Adaptive RAG

Multi-step Research Agent

Retrieval confidence scoring

📊 Evaluation & Observability

RAGAS evaluation

DeepEval evaluation

LangSmith tracing

Retrieval metrics

Answer faithfulness evaluation

Latency monitoring

💾 Application Features

Conversation memory

Streaming responses

Direct document upload

User authentication

User-specific knowledge bases

Persistent chat history

Citation generation

🚀 Productionization

FastAPI backend

Docker deployment

Production vector database

API authentication

Rate limiting

Structured logging

Automated testing

CI/CD pipeline

🏆 What Makes This Project Different?

RAGForge is not just a basic:

PDF → Embeddings → Vector Search → LLM

application.

It introduces an agentic decision-making layer that allows the system to determine how information should be obtained.

The architecture combines:

RAG
 +
Vector Search
 +
LangGraph
 +
ReAct Agents
 +
Tool Calling
 +
External Knowledge

This makes the project a practical demonstration of how modern Agentic AI and RAG systems can be combined to build knowledge-intensive applications.

📊 High-Level Design
                    ┌─────────────────────┐
                    │       USER          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LangGraph Agent   │
                    │      ReAct          │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             ┌─────────────┐      ┌──────────────┐
             │  Retriever  │      │  Wikipedia   │
             │    Tool     │      │     Tool     │
             └──────┬──────┘      └──────┬───────┘
                    │                    │
                    ▼                    ▼
             ┌─────────────┐      ┌──────────────┐
             │ Vector DB   │      │ Wikipedia API│
             └──────┬──────┘      └──────────────┘
                    │
                    ▼
             ┌─────────────┐
             │ PDF Corpus  │
             └─────────────┘
                    │
                    ▼
             ┌─────────────┐
             │ Final Answer│
             └─────────────┘
🎓 Learning Outcomes

Through this project, the following AI engineering concepts were explored:

Retrieval

Understanding how documents can be transformed into embeddings and retrieved using semantic similarity.

Agentic Workflows

Understanding how LLM agents can reason about tasks and dynamically select tools.

LangGraph

Building stateful AI workflows using graph-based orchestration.

Tool Calling

Connecting LLM reasoning with external capabilities such as retrieval and Wikipedia.

RAG Architecture

Designing an end-to-end retrieval pipeline from document ingestion to response generation.

AI Application Development

Building an interactive AI application using Streamlit.

👨‍💻 Author

Aviral

AI Engineering Project focused on:

🤖 Agentic AI
🧠 Generative AI
📚 Retrieval-Augmented Generation
🔎 Semantic Search
🔗 LangChain
🕸️ LangGraph
🛠️ ReAct Agents
🧩 LLM Tool Calling
