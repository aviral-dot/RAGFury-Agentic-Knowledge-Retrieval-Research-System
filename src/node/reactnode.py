"""LangGraph nodes for RAG workflow + ReAct Agent"""

from typing import List, Optional

from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper


class RAGNodes:
    """Contains node functions for RAG workflow."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents."""

        docs = self.retriever.invoke(state.question)

        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self):
        """Build retriever and Wikipedia tools."""

        @tool
        def retriever_tool(query: str) -> str:
            """Search the indexed document corpus for relevant passages."""

            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No documents found."

            merged = []

            for i, doc in enumerate(docs[:8], start=1):
                meta = doc.metadata if hasattr(doc, "metadata") else {}

                title = (
                    meta.get("title")
                    or meta.get("source")
                    or f"doc_{i}"
                )

                merged.append(
                    f"[{i}] {title}\n{doc.page_content}"
                )

            return "\n\n".join(merged)

        wiki_api = WikipediaAPIWrapper(
            top_k_results=3,
            lang="en"
        )

        @tool
        def wikipedia(query: str) -> str:
            """Search Wikipedia for general knowledge."""

            return wiki_api.run(query)

        return [retriever_tool, wikipedia]

    def _build_agent(self):
        """Create the ReAct agent."""

        tools = self._build_tools()

        system_prompt = (
            "You are a helpful Agentic RAG assistant. "
            "Use retriever_tool when the question relates to the user's indexed documents. "
            "Use wikipedia only when general external knowledge is required. "
            "After receiving useful information from a tool, answer the user directly. "
            "Do not call the same tool repeatedly with the same or similar query. "
            "Do not continue searching once you have enough information. "
            "If the tools do not provide useful information, answer that you could not find "
            "sufficient information. Always finish with a final answer."
        )

        self._agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=system_prompt
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate the final answer using the ReAct agent."""

        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke(
            {
                "messages": [
                    HumanMessage(content=state.question)
                ]
            },
            config={"recursion_limit": 10}
        )

        messages = result.get("messages", [])

        answer: Optional[str] = None

        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )
