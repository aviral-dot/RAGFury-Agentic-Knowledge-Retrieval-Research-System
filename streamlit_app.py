"""Streamlit frontend for RAGFury."""

import time

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="RAGFury",
    page_icon="🔍",
    layout="centered",
)


st.markdown(
    """
    <style>

    .stButton > button {
        width: 100%;
        font-weight: bold;
    }

    .answer-box {
        padding: 1rem;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """
    Initialize Streamlit session state.

    user_id:
        Manually entered by the user.
        This identifies the user for Mem0 long-term memory.

    conversation_id:
        Generated automatically by FastAPI.
        This identifies the current Redis conversation.

    history:
        Stores responses locally for UI display.
    """

    if "history" not in st.session_state:
        st.session_state.history = []

    if "user_id" not in st.session_state:
        st.session_state.user_id = ""

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None


def handle_user_id_change():
    """
    Reset the UI when the user changes.

    The previous user's backend memory is NOT deleted.
    Only the current Streamlit conversation state is reset.
    """

    new_user_id = st.session_state.user_id_input.strip()

    if new_user_id == st.session_state.user_id:
        return

    # Switch active user
    st.session_state.user_id = new_user_id

    # Start a completely fresh UI conversation
    st.session_state.conversation_id = None

    # Clear previous user's visible conversation history
    st.session_state.history = []


def start_new_conversation():
    """
    Start a new conversation.

    The user ID stays the same.

    The current conversation ID is cleared.
    FastAPI will generate a new conversation ID
    when the next question is sent.
    """

    st.session_state.conversation_id = None
    st.session_state.history = []


def check_api_health():
    """
    Check whether the FastAPI backend is available.
    """

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def ask_backend(question: str):
    """
    Send the user's question to FastAPI.

    FastAPI expects:

        question
        user_id

    conversation_id is sent only when an existing
    conversation already exists.

    For the first message of a new conversation,
    conversation_id is omitted.

    FastAPI then generates it automatically.
    """

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
    """
    Send user feedback for a specific LangSmith run
    to the FastAPI backend.
    """

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


def display_route(route: str):
    """
    Display the route selected by the Agent.
    """

    if route == "rag":
        st.info(
            "📄 **Source: Company Documents**\n\n"
            "The routing agent selected the "
            "document RAG pipeline."
        )

    elif route == "chat":
        st.info(
            "💬 **Source: Chat Agent**\n\n"
            "The routing agent selected the "
            "conversational Chat agent."
        )

    elif route:
        st.info(f"🤖 **Agent Route:** `{route}`")


def display_citations(result):
    """
    Display citations returned by the RAG API.
    """

    citations = result.get(
        "citations",
        [],
    )

    if not citations:
        return

    st.markdown("### 📚 Sources")

    for citation in citations:
        citation_id = citation.get(
            "citation_id",
            "",
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

        if page is not None:
            page_text = f"Page {page}"
        else:
            page_text = "Page unavailable"

        st.markdown(
            f"**[{citation_id}] {source}**  \n📄 {page_text}  \n`Chunk: {chunk_id}`"
        )


def display_rag_details(result):
    """
    Display RAG-specific information.

    These fields are normally populated only
    when the Agent selects the RAG workflow.
    """

    route = result.get("next_step")

    if route != "rag":
        return

    documents = result.get(
        "documents",
        [],
    )

    document_relevance = result.get("document_relevance")

    grade_reason = result.get("grade_reason")

    retrieval_attempts = result.get("retrieval_attempts")

    if documents:
        with st.expander(f"📚 Retrieved Documents ({len(documents)})"):
            for index, document in enumerate(
                documents,
                start=1,
            ):
                st.markdown(f"### Document {index}")

                content = document.get(
                    "content",
                    "",
                )

                metadata = document.get(
                    "metadata",
                    {},
                )

                if content:
                    st.markdown(content)

                if metadata:
                    st.caption(f"Metadata: {metadata}")

                if index < len(documents):
                    st.divider()

    if document_relevance is not None or grade_reason:
        with st.expander("🧠 Document Grading"):
            if document_relevance is not None:
                st.write(
                    "Relevant:",
                    document_relevance,
                )

            if grade_reason:
                st.write(
                    "Reason:",
                    grade_reason,
                )

    if retrieval_attempts is not None:
        st.caption(f"🔄 Retrieval attempts: {retrieval_attempts}")


def display_chat_details(result):
    """
    Display information related to the Chat agent.
    """

    route = result.get("next_step")

    if route != "chat":
        return

    memories = result.get("relevant_memories")

    if memories:
        with st.expander("🧠 Long-Term Memory Used"):
            for memory in memories:
                if isinstance(
                    memory,
                    dict,
                ):
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
                    st.markdown(f"- {text}")

    else:
        st.caption("🧠 No relevant long-term memories found.")


def display_history():
    """
    Display recent conversation turns.
    """

    if not st.session_state.history:
        return

    st.markdown("---")

    st.markdown("### 📜 Recent Conversation")

    for item in reversed(st.session_state.history[-5:]):
        with st.container():
            st.markdown(f"**Q:** {item['question']}")

            answer_preview = item["answer"]

            if len(answer_preview) > 200:
                answer_preview = answer_preview[:200] + "..."

            st.markdown(f"**A:** {answer_preview}")

            route = item.get(
                "route",
                "unknown",
            )

            if route == "rag":
                st.caption("📄 Source: Company Documents")

            elif route == "chat":
                st.caption("💬 Source: Chat Agent")

            else:
                st.caption(f"🤖 Route: {route}")

            st.caption(f"⏱️ Response time: {item['time']:.2f}s")


def main():
    """
    Run the Streamlit application.
    """

    init_session_state()

    st.title("🔍 RAGFury")

    st.subheader("Agentic Knowledge Retrieval & Conversational AI")

    st.markdown(
        """
        Ask questions about your **company documents**
        or have a **general conversation** with the AI.

        🤖 The routing agent automatically decides whether
        to use the **RAG pipeline** or the **Chat agent**.
        """
    )

    if check_api_health():
        st.success("🟢 RAGFury API is online")

    else:
        st.error("🔴 RAGFury API is offline")

        st.info("Start the FastAPI backend with:\n\n`uvicorn api.main:app --reload`")

        return

    with st.sidebar:
        st.header("💬 Conversation")

        st.caption("User ID")

        st.text_input(
            "Enter your User ID",
            value=st.session_state.user_id,
            placeholder="e.g. user_12345",
            label_visibility="collapsed",
            key="user_id_input",
            on_change=handle_user_id_change,
        )

        st.caption("Current conversation")

        if st.session_state.conversation_id:
            st.code(st.session_state.conversation_id)

        else:
            st.info("Will be generated automatically when you send your first message.")

        st.divider()

        st.markdown(
            """
            **Memory Architecture**

            🔵 Redis  
            Short-term conversation memory

            🟣 Mem0  
            Long-term semantic memory

            **User ID**
            → identifies your long-term memory

            **Conversation ID**
            → automatically identifies the current Redis conversation
            """
        )

        st.divider()

        if st.button("🆕 New Conversation"):
            start_new_conversation()

            st.rerun()

    if not st.session_state.user_id:
        st.warning(
            "👤 Please enter your User ID in the sidebar before asking a question."
        )

    st.markdown("---")

    with st.form("search_form"):
        question = st.text_input(
            "Ask RAGFury",
            placeholder=("e.g. How much sick leave can an employee take?"),
        )

        submit = st.form_submit_button("🤖 Ask Agent")

    if submit:
        if not st.session_state.user_id:
            st.error("❌ Please enter a User ID in the sidebar first.")

            return

        question = question.strip()

        if not question:
            st.warning("Please enter a question.")

            return

        start_time = time.time()

        with st.spinner("🤔 Agent is processing your question..."):
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

                conversation_id = result.get("conversation_id")

                if conversation_id:
                    st.session_state.conversation_id = conversation_id

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "time": elapsed_time,
                        "route": route,
                        "run_id": result.get("run_id"),
                    }
                )

                st.markdown("### 💡 Answer")

                st.success(answer)

                display_citations(result)

                # =========================================================
                # USER FEEDBACK
                # =========================================================

                run_id = result.get("run_id")

                if run_id:
                    st.markdown("**Was this answer helpful?**")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(
                            "👍 Helpful",
                            key=f"feedback_positive_{run_id}",
                        ):
                            if submit_feedback(
                                run_id=run_id,
                                score=1.0,
                            ):
                                st.success("Thanks for your feedback! 👍")
                            else:
                                st.error("Could not submit feedback.")

                    with col2:
                        if st.button(
                            "👎 Not helpful",
                            key=f"feedback_negative_{run_id}",
                        ):
                            if submit_feedback(
                                run_id=run_id,
                                score=0.0,
                            ):
                                st.success("Thanks for your feedback! 👎")
                            else:
                                st.error("Could not submit feedback.")

                display_route(route)

                display_rag_details(result)

                display_chat_details(result)

                backend_time = result.get("response_time")

                if backend_time is not None:
                    st.caption(f"⏱️ Backend response time: {backend_time:.2f}s")

                st.caption(f"⏱️ Total UI response time: {elapsed_time:.2f}s")

                if st.session_state.conversation_id:
                    st.caption(f"💬 Conversation: {st.session_state.conversation_id}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The RAGFury pipeline took too long.")

            except requests.exceptions.ConnectionError:
                st.error("🔴 Could not connect to the FastAPI backend.")

            except requests.exceptions.HTTPError as exc:
                # ------------------------------------------------
                # Handle FastAPI HTTP errors cleanly
                # ------------------------------------------------

                try:
                    error_data = exc.response.json()

                    detail = error_data.get(
                        "detail",
                        "The request was rejected by the API.",
                    )

                except Exception:
                    detail = "The request was rejected by the API."

                # ------------------------------------------------
                # 400 = Safety / validation rejection
                # ------------------------------------------------

                if exc.response.status_code == 400:
                    st.warning(f"🛡️ {detail}")

                # ------------------------------------------------
                # 503 = Backend / security service failure
                # ------------------------------------------------

                elif exc.response.status_code == 503:
                    st.error(f"🚨 {detail}")

                # ------------------------------------------------
                # Other HTTP errors
                # ------------------------------------------------

                else:
                    st.error(f"❌ {detail}")

            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")

    display_history()


if __name__ == "__main__":
    main()
