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
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.5rem;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }

    div[data-testid="stChatMessage"] {
        margin-bottom: 0.75rem;
    }

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
    st.session_state.conversation_id = None
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

    except requests.RequestException as exc:
        st.error(f"Feedback request failed: {exc}")

        if getattr(exc, "response", None) is not None:
            st.error(
                f"Backend response: {exc.response.status_code} — {exc.response.text}"
            )

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

    for index, citation in enumerate(
        citations,
        start=1,
    ):
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

    document_relevance = result.get("document_relevance")

    grade_reason = result.get("grade_reason")

    retrieval_attempts = result.get("retrieval_attempts")

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
    """Display the initial welcome screen."""

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
                "Hybrid retrieval, reranking and "
                "grounded answers from your indexed "
                "documents."
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
                "Inspect citations and retrieval "
                "information when document "
                "knowledge is used."
            )

    st.divider()

    st.info("👈 Enter your User ID in the sidebar, then ask your first message below.")


# ============================================================
# CHAT HISTORY
# ============================================================


def display_history():
    """Render the complete conversation."""

    for item in st.session_state.history:
        # ----------------------------------------------------
        # USER MESSAGE — LEFT
        # ----------------------------------------------------

        user_content, user_empty = st.columns([7, 3])

        with user_content:
            with st.chat_message("user"):
                st.write(item["question"])

        # ----------------------------------------------------
        # ASSISTANT MESSAGE — RIGHT
        # ----------------------------------------------------

        assistant_empty, assistant_content = st.columns([3, 7])

        with assistant_content:
            with st.chat_message("assistant"):
                st.write(item["answer"])

                display_route(
                    item.get(
                        "route",
                        "unknown",
                    )
                )

                if item.get("route") == "rag":
                    display_citations(
                        item.get(
                            "citations",
                            [],
                        )
                    )

                stored_result = item.get("result")

                if stored_result:
                    display_rag_details(stored_result)

                display_feedback(item.get("run_id"))

                backend_time = item.get("response_time")

                ui_time = item.get("time")

                if backend_time is not None:
                    st.caption(f"Backend {backend_time:.2f}s · UI {ui_time:.2f}s")

                elif ui_time is not None:
                    st.caption(f"UI response time {ui_time:.2f}s")


# ============================================================
# SIDEBAR
# ============================================================


def display_sidebar(
    api_online: bool,
):
    """Display the application sidebar."""

    with st.sidebar:
        st.title("✦ RAGFury")

        st.caption("Agentic Knowledge & Research")

        st.divider()

        st.subheader("Workspace")

        st.text_input(
            "EMP ID",
            value=st.session_state.user_id,
            placeholder="e.g. employee_12345",
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

    init_session_state()

    api_online = check_api_health()

    display_sidebar(api_online)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("Agentic Knowledge Assistant")

    st.caption(
        "Ask about your documents or have a natural "
        "conversation. RAGFury automatically chooses "
        "the right agent."
    )

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

    # --------------------------------------------------------
    # API OFFLINE
    # --------------------------------------------------------

    if not api_online:
        st.warning("RAGFury API is currently unavailable.")

        st.code(
            "uvicorn api.main:app --reload",
            language="powershell",
        )

        st.caption("Start the FastAPI backend and refresh this page.")

        return

    # --------------------------------------------------------
    # NO USER YET
    # --------------------------------------------------------

    if not st.session_state.user_id:
        display_welcome()

    # --------------------------------------------------------
    # EXISTING CONVERSATION
    # --------------------------------------------------------

    elif st.session_state.history:
        display_history()

    # --------------------------------------------------------
    # EMPTY CONVERSATION
    # --------------------------------------------------------

    else:
        st.markdown("### Start a conversation")

        st.caption(
            "Ask a question about your documents or start a general conversation."
        )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input("Message RAGFury...")

    if question is None:
        return

    # --------------------------------------------------------
    # VALIDATE USER
    # --------------------------------------------------------

    if not st.session_state.user_id:
        st.error("Please enter a User ID before asking a question.")

        return

    question = question.strip()

    if not question:
        st.warning("Please enter a question.")

        return

    # --------------------------------------------------------
    # SHOW USER MESSAGE IMMEDIATELY
    # --------------------------------------------------------

    user_content, user_empty = st.columns([7, 3])

    with user_content:
        with st.chat_message("user"):
            st.write(question)

    # --------------------------------------------------------
    # BACKEND REQUEST
    # --------------------------------------------------------

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
            # STORE COMPLETE TURN
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
                    "response_time": result.get("response_time"),
                    "result": result,
                }
            )

            # ------------------------------------------------
            # RERUN
            #
            # display_history() now renders the
            # complete conversation.
            # ------------------------------------------------

            st.rerun()

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


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()
