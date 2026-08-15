"""Streamlit UI for Agentic RAG System."""

import streamlit as st
from pathlib import Path
import sys
import time



sys.path.append(str(Path(__file__).parent))


from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder




st.set_page_config(
    page_title="🤖 RAGFury",
    page_icon="🔍",
    layout="centered"
)


st.markdown(
    """
    <style>

    .stButton > button {
        width: 100%;
        font-weight: bold;
    }

    .route-box {
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)




def init_session_state():
    """Initialize Streamlit session state."""

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "history" not in st.session_state:
        st.session_state.history = []




@st.cache_resource
def initialize_rag(data_directory: str):
    """
    Initialize the Agentic RAG system.

    This follows the same initialization flow as main.py.
    """

    try:
        print("🚀 Initializing Agentic RAG System...")

       

        llm = Config.get_llm()

        

        doc_processor = DocumentProcessor(
            model_name="all-MiniLM-L6-v2",
            threshold=0.3
        )

        

        vector_store = VectorStore()

        

        print(
            f"📄 Processing documents from: {data_directory}"
        )

        documents = doc_processor.process_pdfs(
            Path(data_directory)
        )

        print(
            f"📊 Created {len(documents)} document chunks"
        )

        

        print("🔍 Creating vector store...")

        vector_store.create_vectorstore(documents)

        

        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )

        graph = graph_builder.build()

        print(
            "✅ Agentic RAG system initialized successfully!"
        )

        return graph, len(documents)

    except Exception as e:

        st.error(
            f"❌ Failed to initialize RAG system: {str(e)}"
        )

        return None, 0




def ask_question(question: str):
    """
    Send a question to the LangGraph workflow.

    The routing agent decides whether the question
    should use company documents or external knowledge.
    """

    if st.session_state.rag_system is None:
        return None

    result = st.session_state.rag_system.invoke(
        {
            "question": question
        }
    )

    return result




def display_route(result):
    """Display the route selected by the agent."""

    next_step = result.get("next_step")

    if next_step == "rag":

        st.info(
            "📄 **Source: Company Documents**\n\n"
            "The agent routed your question to the "
            "company document RAG pipeline."
        )

    elif next_step == "wikipedia":

        st.info(
            "🌐 **Source: External Knowledge**\n\n"
            "The agent routed your question to the "
            "Wikipedia knowledge source."
        )

    elif next_step:

        st.info(
            f"🤖 **Agent Route:** `{next_step}`"
        )




def display_documents(result):
    """Display retrieved company documents if available."""

    documents = result.get("documents")

    if documents is None:
        documents = result.get("retrieved_docs")

    if not documents:
        return

    with st.expander("📄 Retrieved Company Documents"):

        for i, doc in enumerate(documents, 1):

            st.markdown(
                f"### Document {i}"
            )

            if hasattr(doc, "page_content"):

                st.text(
                    doc.page_content
                )

            else:

                st.text(
                    str(doc)
                )

            st.markdown("---")




def display_workflow(result):
    """Display agent workflow information."""

    with st.expander("🔎 Workflow Details"):

        next_step = result.get("next_step")

        if next_step == "rag":

            st.write(
                "📄 **Selected Route:** Company Document RAG"
            )

        elif next_step == "wikipedia":

            st.write(
                "🌐 **Selected Route:** External Knowledge"
            )

        elif next_step:

            st.write(
                f"🤖 **Selected Route:** `{next_step}`"
            )

        if "question" in result:

            st.write(
                f"**Processed Question:** "
                f"`{result['question']}`"
            )




def main():
    """Main Streamlit application."""

    init_session_state()

   

    st.title(
        "🔍 RAGFury — Agentic Knowledge Retrieval"
    )

    st.markdown(
        """
        Ask questions about your **company documents**
        or **external knowledge**.

        🤖 The agent automatically decides whether to use
        your private company documents or external knowledge.
        """
    )

   
    data_directory = Path("data")

    if not data_directory.exists():

        st.error(
            f"❌ Data directory not found: `{data_directory}`"
        )

        st.info(
            "Please create a `data` directory and place your "
            "company PDF documents inside it."
        )

        return

   

    if not st.session_state.initialized:

        with st.spinner(
            "🚀 Initializing Agentic RAG system..."
        ):

            rag_system, num_chunks = initialize_rag(
                str(data_directory)
            )

            if rag_system is not None:

                st.session_state.rag_system = rag_system
                st.session_state.initialized = True

                st.success(
                    f"✅ System ready! "
                    f"{num_chunks} document chunks loaded."
                )

            else:

                st.error(
                    "❌ RAG system initialization failed."
                )

                return

   

    st.markdown("---")

    

    with st.form("search_form"):

        question = st.text_input(
            "Enter your question:",
            placeholder=(
                "e.g. How much sick leave can an employee take? "
                "or Who is the president of India?"
            )
        )

        submit = st.form_submit_button(
            "🤖 Ask Agent"
        )

    

    if submit and question.strip():

        question = question.strip()

        if st.session_state.rag_system is None:

            st.error(
                "❌ RAG system is not initialized."
            )

            return

        with st.spinner(
            "🤔 Agent is processing your question..."
        ):

            start_time = time.time()

            try:


                result = ask_question(question)

                elapsed_time = (
                    time.time() - start_time
                )

                

                answer = result.get(
                    "answer",
                    "No answer generated."
                )

              

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "time": elapsed_time,
                        "route": result.get(
                            "next_step",
                            "unknown"
                        )
                    }
                )

               

                display_route(result)

               
                st.markdown(
                    "### 💡 Answer"
                )

                st.success(answer)

               

                st.caption(
                    f"⏱️ Response time: "
                    f"{elapsed_time:.2f} seconds"
                )

               

                display_workflow(result)

                

                display_documents(result)

            except Exception as e:

                st.error(
                    f"❌ Error while processing "
                    f"question: {str(e)}"
                )

   

    if st.session_state.history:

        st.markdown("---")

        st.markdown(
            "### 📜 Recent Searches"
        )

        for item in reversed(
            st.session_state.history[-3:]
        ):

            with st.container():

              

                st.markdown(
                    f"**Q:** {item['question']}"
                )

                

                answer_preview = item["answer"]

                if len(answer_preview) > 200:

                    answer_preview = (
                        answer_preview[:200]
                        + "..."
                    )

                st.markdown(
                    f"**A:** {answer_preview}"
                )

                

                route = item.get(
                    "route",
                    "unknown"
                )

                if route == "rag":

                    st.caption(
                        "📄 Source: Company Documents"
                    )

                elif route == "wikipedia":

                    st.caption(
                        "🌐 Source: External Knowledge"
                    )

                else:

                    st.caption(
                        f"🤖 Route: {route}"
                    )

               

                st.caption(
                    f"⏱️ Response time: "
                    f"{item['time']:.2f}s"
                )




if __name__ == "__main__":
    main()