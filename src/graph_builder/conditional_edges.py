"""Conditional routing functions for the RAGFury LangGraph workflow."""

from src.state.rag_state import RAGState


def route_after_agent(state: RAGState) -> str:
    """
    Route the workflow based on the tool selected by the ReAct agent.

    Returns:
        "rag"        -> retriever_tool was selected
        "wikipedia"  -> wikipedia tool was selected
        "end"        -> no tool was called / agent already answered
    """

    messages = state["messages"]

    if not messages:
        return "end"

    last_message = messages[-1]

    tool_calls = getattr(last_message, "tool_calls", None)

    if not tool_calls:
        return "end"

    tool_call = tool_calls[0]

    tool_name = tool_call.get("name")

    if tool_name == "retriever_tool":
        return "rag"

    if tool_name == "wikipedia":
        return "wikipedia"

    return "end"
