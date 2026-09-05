"""Production-style Streamlit frontend for RAGFury."""

import time

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAGFury",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LIGHTWEIGHT STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main content */
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.5rem;
    }

    /* Inputs */
    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }

    /* Chat spacing */
    div[data-testid="stChatMessage"] {
        margin-bottom: 0.75rem;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 12px;
        padding: 0.75rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================


def init_session_state():
    """Initialize Streamlit session state."""

    if "history" not in st.session_state:
        st.session_state.history = []

    if "user_id" not in st.session_state:
        st.session_state.user_id = ""

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None


def handle_user_id_change():
    """
    Reset the visible UI when the user changes.

    Backend Redis/Mem0 memory is NOT deleted.
    """

    new_user_id = st.session_state.user_id_input.strip()

    if new_user_id == st.session_state.user_id:
        return

    st.session_state.user_id = new_user_id

    # Start a fresh UI conversation.
    st.session_state.conversation_id = None

    # Remove previous user's visible conversation.
    st.session_state.history = []


def start_new_conversation():
    """Start a fresh conversation for the current user."""

    st.session_state.conversation_id = None
    st.session_state.history = []


# ============================================================
# API
# ============================================================


def check_api_health():
    """Check whether FastAPI is available."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def ask_backend(question: str):
    """Send a question to the FastAPI backend."""

    payload = {
        "question": question,
        "user_id": st.session_state.user_id,
    }

    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id

    response = requests.post(
        f"{API_URL}/api/v1/query",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    returned_conversation_id = result.get("conversation_id")

    if returned_conversation_id:
        st.session_state.conversation_id = returned_conversation_id

    return result


def submit_feedback(
    run_id: str,
    score: float,
):
    """Send feedback for a LangSmith run."""

    try:
        response = requests.post(
            f"{API_URL}/api/v1/feedback",
            json={
                "run_id": run_id,
                "score": score,
            },
            timeout=10,
        )

        response.raise_for_status()

        return True

    except requests.RequestException:
        return False


# ============================================================
# UI HELPERS
# ============================================================


def display_route(route: str):
    """Display the selected agent."""

    if route == "rag":
        st.caption("📄 **Document Research Agent**")

    elif route == "chat":
        st.caption("💬 **Conversational Agent**")

    else:
        st.caption(f"🤖 **Agent:** {route or 'Unknown'}")


def display_citations(citations):
    """Display citations using native Streamlit components."""

    if not citations:
        return

    st.markdown("##### 📚 Sources")

    for index, citation in enumerate(citations, start=1):
        citation_id = citation.get(
            "citation_id",
            f"S{index}",
        )

        source = citation.get(
            "source",
            "Unknown source",
        )

        page = citation.get("page")

        chunk_id = citation.get(
            "chunk_id",
            "unknown",
        )

        with st.container(border=True):
            st.markdown(f"**[{citation_id}] {source}**")

            metadata = []

            if page is not None:
                metadata.append(f"📄 Page {page}")

            if chunk_id:
                metadata.append(f"Chunk `{chunk_id}`")

            if metadata:
                st.caption(" · ".join(metadata))


def display_rag_details(result):
    """Display advanced RAG information."""

    if result.get("next_step") != "rag":
        return

    documents = result.get(
        "documents",
        [],
    )

    document_relevance = result.get(
        "document_relevance",
    )

    grade_reason = result.get(
        "grade_reason",
    )

    retrieval_attempts = result.get(
        "retrieval_attempts",
    )

    if not (
        documents
        or document_relevance is not None
        or grade_reason
        or retrieval_attempts is not None
    ):
        return

    with st.expander(
        "🔬 Research details",
        expanded=False,
    ):
        if retrieval_attempts is not None:
            st.metric(
                "Retrieval attempts",
                retrieval_attempts,
            )

        if document_relevance is not None:
            st.write(
                "**Document relevance:**",
                document_relevance,
            )

        if grade_reason:
            st.write(
                "**Grading reason:**",
                grade_reason,
            )

        if documents:
            st.markdown(f"**Retrieved context — {len(documents)} documents**")

            for index, document in enumerate(
                documents,
                start=1,
            ):
                with st.container(border=True):
                    st.markdown(f"**Document {index}**")

                    content = document.get(
                        "content",
                        "",
                    )

                    metadata = document.get(
                        "metadata",
                        {},
                    )

                    if content:
                        st.write(content)

                    if metadata:
                        st.caption(f"Metadata: {metadata}")


def display_chat_details(result):
    """Display advanced memory information."""

    if result.get("next_step") != "chat":
        return

    memories = result.get("relevant_memories")

    with st.expander(
        "🧠 Memory details",
        expanded=False,
    ):
        if not memories:
            st.caption("No relevant long-term memories were used.")

            return

        st.caption("Relevant long-term memories used for this response:")

        for memory in memories:
            if isinstance(memory, dict):
                text = memory.get(
                    "memory",
                    memory.get(
                        "text",
                        "",
                    ),
                )

            else:
                text = str(memory)

            if text:
                st.write(f"• {text}")


def display_feedback(run_id: str):
    """Display feedback controls."""

    if not run_id:
        return

    st.caption("Was this response helpful?")

    col1, col2, _ = st.columns([1, 1, 5])

    with col1:
        if st.button(
            "👍 Helpful",
            key=f"feedback_positive_{run_id}",
            use_container_width=True,
        ):
            if submit_feedback(
                run_id=run_id,
                score=1.0,
            ):
                st.success("Thanks for the feedback!")
            else:
                st.error("Could not submit feedback.")

    with col2:
        if st.button(
            "👎 Not helpful",
            key=f"feedback_negative_{run_id}",
            use_container_width=True,
        ):
            if submit_feedback(
                run_id=run_id,
                score=0.0,
            ):
                st.success("Thanks for the feedback!")
            else:
                st.error("Could not submit feedback.")


# ============================================================
# WELCOME SCREEN
# ============================================================


def display_welcome():
    """Display the initial workspace."""

    st.title("✦ RAGFury")

    st.subheader("Agentic Knowledge Assistant")

    st.write(
        """
        Ask questions about your knowledge base or
        start a natural conversation. RAGFury automatically
        chooses the appropriate agent.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 📄")
            st.markdown("**Document Research**")
            st.caption(
                "Hybrid retrieval, reranking and grounded "
                "answers from your indexed documents."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### 💬")
            st.markdown("**Natural Conversation**")
            st.caption("Conversational reasoning with short-term and long-term memory.")

    with col3:
        with st.container(border=True):
            st.markdown("### 🎯")
            st.markdown("**Grounded Answers**")
            st.caption(
                "Inspect citations and retrieval information "
                "when document knowledge is used."
            )

    st.divider()

    st.info("👈 Enter your User ID in the sidebar, then ask your first question below.")


# ============================================================
# CHAT HISTORY
# ============================================================


def display_history():
    """Display the current conversation."""

    for item in st.session_state.history:
        # ----------------------------------------------------
        # USER
        # ----------------------------------------------------

        with st.chat_message(
            "user",
        ):
            st.write(item["question"])

        # ----------------------------------------------------
        # ASSISTANT
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
        ):
            st.write(item["answer"])

            display_route(
                item.get(
                    "route",
                    "unknown",
                )
            )

            display_citations(
                item.get(
                    "citations",
                    [],
                )
            )

            stored_result = item.get(
                "result",
            )

            if stored_result:
                display_rag_details(stored_result)

                display_chat_details(stored_result)

            backend_time = item.get("response_time")

            if backend_time is not None:
                st.caption(f"Backend {backend_time:.2f}s · UI {item['time']:.2f}s")

            else:
                st.caption(f"Response time {item['time']:.2f}s")


# ============================================================
# SIDEBAR
# ============================================================


def display_sidebar(api_online: bool):
    """Render the application sidebar."""

    with st.sidebar:
        st.title("✦ RAGFury")

        st.caption("Agentic Knowledge & Research")

        st.divider()

        st.subheader("Workspace")

        st.text_input(
            "User ID",
            value=st.session_state.user_id,
            placeholder="e.g. user_12345",
            key="user_id_input",
            on_change=handle_user_id_change,
        )

        if st.session_state.user_id:
            st.success("Workspace active")

        else:
            st.info("Enter a User ID to begin.")

        st.divider()

        if st.button(
            "＋ New conversation",
            use_container_width=True,
        ):
            start_new_conversation()
            st.rerun()

        st.divider()

        st.subheader("Conversation")

        if st.session_state.conversation_id:
            st.caption("Conversation ID")

            st.code(
                st.session_state.conversation_id,
                language=None,
            )

        else:
            st.caption(
                "A conversation ID will be generated when you send your first message."
            )

        st.divider()

        if api_online:
            st.success("🟢 API Online")

        else:
            st.error("🔴 API Offline")

        with st.expander("🧠 Memory architecture"):
            st.markdown(
                """
                **Redis**

                Short-term conversation memory.

                **Mem0**

                Long-term semantic memory.

                **User ID**

                Identifies the long-term memory namespace.

                **Conversation ID**

                Identifies the current conversation.
                """
            )

        with st.expander("ℹ️ About RAGFury"):
            st.markdown(
                """
                RAGFury combines:

                - 🤖 Agentic routing
                - 🔎 Hybrid retrieval
                - 🧠 Document reranking
                - 💬 Conversational memory
                - 📚 Grounded citations
                - 🛡️ Production guardrails
                """
            )


# ============================================================
# MAIN APPLICATION
# ============================================================


def main():
    """Run the Streamlit application."""

    init_session_state()

    # ========================================================
    # API HEALTH
    # ========================================================

    api_online = check_api_health()

    display_sidebar(api_online)

    # ========================================================
    # MAIN HEADER
    # ========================================================

    st.title("Agentic Knowledge Assistant")

    st.caption(
        "Ask about your documents or have a natural conversation. "
        "RAGFury automatically chooses the right agent."
    )

    # ========================================================
    # API STATUS
    # ========================================================

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        if api_online:
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")

    with status_col2:
        st.info("✦ Agentic Routing")

    with status_col3:
        st.info("📚 Grounded Retrieval")

    # ========================================================
    # API OFFLINE
    # ========================================================

    if not api_online:
        st.warning("RAGFury API is currently unavailable.")

        st.code(
            "uvicorn api.main:app --reload",
            language="powershell",
        )

        st.caption("Start the FastAPI backend and refresh this page.")

        return

    # ========================================================
    # USER VALIDATION
    # ========================================================

    if not st.session_state.user_id:
        display_welcome()

    else:
        # ====================================================
        # EXISTING CHAT
        # ====================================================

        if st.session_state.history:
            display_history()

        else:
            st.markdown("### Start a conversation")

            st.caption(
                "Ask a question about your documents or start a general conversation."
            )

    # ========================================================
    # INPUT
    # ========================================================

    st.divider()

    with st.form(
        "search_form",
        clear_on_submit=True,
    ):
        question = st.text_input(
            "Message",
            placeholder=("Ask about your documents or start a conversation..."),
            label_visibility="collapsed",
        )

        submit = st.form_submit_button(
            "✦ Ask RAGFury",
            use_container_width=True,
        )

    # ========================================================
    # QUERY
    # ========================================================

    if submit:
        if not st.session_state.user_id:
            st.error("Please enter a User ID before asking a question.")

            return

        question = question.strip()

        if not question:
            st.warning("Please enter a question.")

            return

        start_time = time.time()

        with st.spinner("RAGFury is thinking..."):
            try:
                result = ask_backend(question)

                elapsed_time = time.time() - start_time

                answer = result.get(
                    "answer",
                    "No answer generated.",
                )

                route = result.get(
                    "next_step",
                    "unknown",
                )

                # ------------------------------------------------
                # SAVE FULL RESULT FOR RERENDERS
                # ------------------------------------------------

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "time": elapsed_time,
                        "route": route,
                        "run_id": result.get("run_id"),
                        "citations": result.get(
                            "citations",
                            [],
                        ),
                        "documents": result.get(
                            "documents",
                            [],
                        ),
                        "document_relevance": result.get("document_relevance"),
                        "grade_reason": result.get("grade_reason"),
                        "retrieval_attempts": result.get("retrieval_attempts"),
                        "relevant_memories": result.get("relevant_memories"),
                        "response_time": result.get("response_time"),
                        "result": result,
                    }
                )

                # ------------------------------------------------
                # CURRENT RESPONSE
                # ------------------------------------------------

                with st.chat_message(
                    "user",
                ):
                    st.write(question)

                with st.chat_message(
                    "assistant",
                ):
                    st.write(answer)

                    display_route(route)

                    display_citations(
                        result.get(
                            "citations",
                            [],
                        )
                    )

                    display_feedback(result.get("run_id"))

                    display_rag_details(result)

                    display_chat_details(result)

                    backend_time = result.get("response_time")

                    if backend_time is not None:
                        st.caption(
                            f"Backend {backend_time:.2f}s · UI {elapsed_time:.2f}s"
                        )

                    else:
                        st.caption(f"UI response time {elapsed_time:.2f}s")

            except requests.exceptions.Timeout:
                st.error("⏱️ The request timed out. The RAGFury pipeline took too long.")

            except requests.exceptions.ConnectionError:
                st.error("🔴 Could not connect to the FastAPI backend.")

            except requests.exceptions.HTTPError as exc:
                try:
                    error_data = exc.response.json()

                    detail = error_data.get(
                        "detail",
                        "The request was rejected by the API.",
                    )

                except Exception:
                    detail = "The request was rejected by the API."

                if exc.response.status_code == 400:
                    st.warning(f"🛡️ {detail}")

                elif exc.response.status_code == 503:
                    st.error(f"🚨 {detail}")

                else:
                    st.error(f"❌ {detail}")

            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")


if __name__ == "__main__":
    main()
