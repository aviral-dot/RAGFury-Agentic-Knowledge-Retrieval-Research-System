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
    """Initialize Streamlit session state."""

    if "history" not in st.session_state:
        st.session_state.history = []




def check_api_health():
    """Check whether the FastAPI backend is available."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:
            return True

        return False

    except requests.RequestException:
        return False


def ask_backend(question: str):
    """Send a question to the FastAPI backend."""

    response = requests.post(
        f"{API_URL}/api/v1/query",
        json={
            "question": question,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()



def display_route(route: str):
    """Display the route selected by the agent."""

    if route == "rag":

        st.info(
            "📄 **Source: Company Documents**\n\n"
            "The agent routed your question to the "
            "company document RAG pipeline."
        )

    elif route == "wikipedia":

        st.info(
            "🌐 **Source: External Knowledge**\n\n"
            "The agent routed your question to "
            "external knowledge."
        )

    elif route:

        st.info(
            f"🤖 **Agent Route:** `{route}`"
        )


def display_history():
    """Display recent searches."""

    if not st.session_state.history:
        return

    st.markdown("---")

    st.markdown("### 📜 Recent Searches")

    for item in reversed(
        st.session_state.history[-5:]
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
                "unknown",
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




def main():
    """Run the Streamlit application."""

    init_session_state()


    st.title(
        "🔍 RAGFury"
    )

    st.subheader(
        "Agentic Knowledge Retrieval & Research System"
    )

    st.markdown(
        """
        Ask questions about your **company documents**
        or **external knowledge**.

        🤖 The agent automatically decides which knowledge
        source should be used.
        """
    )

   

    if check_api_health():

        st.success(
            "🟢 RAGFury API is online"
        )

    else:

        st.error(
            "🔴 RAGFury API is offline"
        )

        st.info(
            "Start the FastAPI backend with:\n\n"
            "`uvicorn src.api.main:app`"
        )

        return

   

    st.markdown("---")

    with st.form("search_form"):

        question = st.text_input(
            "Ask RAGFury",
            placeholder=(
                "e.g. How much sick leave can "
                "an employee take?"
            ),
        )

        submit = st.form_submit_button(
            "🤖 Ask Agent"
        )

    

    if submit:

        question = question.strip()

        if not question:

            st.warning(
                "Please enter a question."
            )

            return

        start_time = time.time()

        with st.spinner(
            "🤔 Agent is processing your question..."
        ):

            try:

                result = ask_backend(
                    question
                )

                elapsed_time = (
                    time.time()
                    - start_time
                )

                answer = result.get(
                    "answer",
                    "No answer generated.",
                )

                

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "time": elapsed_time,
                        "route": result.get(
                            "next_step",
                            "unknown",
                        ),
                    }
                )

               

                st.markdown(
                    "### 💡 Answer"
                )

                st.success(
                    answer
                )

                st.caption(
                    f"⏱️ Response time: "
                    f"{elapsed_time:.2f} seconds"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Request timed out. "
                    "The RAG pipeline took too long."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "🔴 Could not connect to "
                    "the FastAPI backend."
                )

            except requests.exceptions.HTTPError as exc:

                st.error(
                    f"❌ API error: {exc}"
                )

                try:

                    st.json(
                        exc.response.json()
                    )

                except Exception:
                    pass

            except Exception as exc:

                st.error(
                    f"❌ Unexpected error: {exc}"
                )

    

    display_history()




if __name__ == "__main__":
    main()