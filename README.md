RAGForge
Agentic RAG Knowledge Retrieval System

RAGForge is an Agentic RAG system built with LangGraph, ReAct agents, vector search, and Streamlit. It processes PDF knowledge bases, retrieves relevant information using semantic search, and dynamically uses external tools such as Wikipedia when additional general knowledge is required.

🚀 Features
📄 PDF Knowledge Base — Load and process multiple PDF documents from a directory.
✂️ Document Chunking — Splits documents into smaller chunks for efficient retrieval.
🔍 Semantic Retrieval — Retrieves relevant document chunks from the vector store.
🤖 ReAct Agent — Uses LangGraph's ReAct architecture for tool-based reasoning.
🛠️ Tool Calling
retriever_tool — Searches the indexed PDF knowledge base.
wikipedia — Retrieves external general knowledge.
🧠 Agentic Decision Making — The agent decides which tool is appropriate for a question.
🌐 External Knowledge Fallback — Can answer general questions using Wikipedia.
💬 Interactive Streamlit UI — User-friendly interface for asking questions.
📚 Source Documents — Retrieved document chunks can be inspected from the UI.
⚡ Response Time Tracking — Displays query processing time.
🧩 Modular Architecture — Separate ingestion, vector store, graph, state, and UI components.
🏗️ Architecture
                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Agentic RAG App   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    LangGraph        │
                    │    ReAct Agent      │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Retriever Tool   │      │ Wikipedia Tool   │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  Vector Store    │      │   Wikipedia API  │
        └────────┬─────────┘      └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │   PDF Documents  │
        └──────────────────┘
🔄 RAG Pipeline
PDF Documents
      │
      ▼
PyPDFDirectoryLoader
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
ReAct Agent
      │
      ├──────────────► Retriever Tool ──► PDF Knowledge
      │
      └──────────────► Wikipedia Tool ──► External Knowledge
                              │
                              ▼
                         Final Answer
📂 Project Structure
RAGForge/
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
└── README.md
🛠️ Tech Stack
Technology	Purpose
Python	Core development
LangChain	LLM and RAG components
LangGraph	Agent workflow orchestration
ReAct Agent	Agentic reasoning and tool usage
Vector Database	Semantic document retrieval
PyPDF	PDF document processing
Wikipedia API	External knowledge retrieval
Streamlit	Web interface
Hugging Face	Embeddings / NLP models
Git & GitHub	Version control
📥 Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/RAGForge.git
cd RAGForge
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file:

GROQ_API_KEY=your_api_key
HF_TOKEN=your_huggingface_token

Never commit your .env file or API keys to GitHub.

📄 Add Documents

Place your PDF files inside:

data/

For example:

data/
├── company_policy.pdf
├── employee_handbook.pdf
└── security_policy.pdf

RAGForge automatically processes the PDFs from the directory.

▶️ Run the Application
CLI
python main.py
Streamlit
python -m streamlit run streamlit_app.py

Then open:

http://localhost:8501
💡 Example Queries
Questions about your documents
What is the company's leave policy?
What is the employee notice period?
What are the security requirements mentioned in the policy?

The agent can use the retriever tool to search the PDF knowledge base.

General knowledge
Who is Brad Pitt?
What is quantum computing?

For general external knowledge, the agent can use the Wikipedia tool.

🤖 Agentic RAG vs Traditional RAG

Traditional RAG generally follows:

Question
   ↓
Retriever
   ↓
Context
   ↓
LLM
   ↓
Answer

RAGForge introduces an agentic layer:

Question
   ↓
ReAct Agent
   ↓
Decide which tool to use
   │
   ├── Retriever → PDF Knowledge Base
   │
   └── Wikipedia → External Knowledge
   ↓
Final Answer

This allows the system to dynamically decide how to obtain information instead of relying on a fixed retrieval pipeline.

🔮 Future Improvements

Planned improvements include:

 Hybrid Search — BM25 + vector search
 Reranking
 Semantic Chunking
 Query Expansion
 HyDE retrieval
 RAG evaluation with RAGAS / DeepEval
 LangSmith observability
 Conversation memory
 Streaming responses
 Document upload directly through Streamlit
 Authentication and user-specific knowledge bases
 Retrieval confidence / citations
 Production deployment with Docker
🎯 Project Goal

RAGForge demonstrates how agentic workflows can be combined with Retrieval-Augmented Generation to build a more flexible knowledge assistant capable of working with private document collections while also accessing external information when required.

👨‍💻 Author

Aviral

Built as an AI Engineering project to explore:

Agentic RAG
LangGraph
ReAct Agents
Vector Search
LLM Tool Calling
Document Retrieval
Generative AI Applications
