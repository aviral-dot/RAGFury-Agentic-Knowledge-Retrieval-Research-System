from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.agent.agent import Agent, RouteDecision

# ============================================================
# RouteDecision
# ============================================================


class TestRouteDecision:
    def test_accepts_rag(self):
        decision = RouteDecision(next_step="rag")

        assert decision.next_step == "rag"

    def test_accepts_chat(self):
        decision = RouteDecision(next_step="chat")

        assert decision.next_step == "chat"

    def test_rejects_invalid_route(self):
        with pytest.raises(ValidationError):
            RouteDecision(next_step="invalid")


# ============================================================
# Agent initialization
# ============================================================


class TestAgentInitialization:
    def test_initialization_stores_llm(self):
        llm = MagicMock()

        agent = Agent(llm=llm)

        assert agent.llm is llm
        assert agent.agent is None
        assert agent.system_prompt is None


# ============================================================
# Agent build
# ============================================================


class TestAgentBuild:
    def test_build_creates_structured_output_agent(self):
        llm = MagicMock()
        structured_agent = MagicMock()

        llm.with_structured_output.return_value = structured_agent

        agent = Agent(llm=llm)

        result = agent.build()

        assert result is structured_agent
        assert agent.agent is structured_agent
        assert agent.system_prompt is not None

        llm.with_structured_output.assert_called_once_with(
            RouteDecision,
        )

    def test_build_prompt_contains_routing_rules(self):
        llm = MagicMock()

        llm.with_structured_output.return_value = MagicMock()

        agent = Agent(llm=llm)

        agent.build()

        assert "rag" in agent.system_prompt
        assert "chat" in agent.system_prompt
        assert "private/company" in agent.system_prompt
        assert "Do not answer the user's question." in agent.system_prompt

    def test_build_propagates_structured_output_error(self):
        llm = MagicMock()

        llm.with_structured_output.side_effect = RuntimeError(
            "structured output failed"
        )

        agent = Agent(llm=llm)

        with pytest.raises(
            RuntimeError,
            match="structured output failed",
        ):
            agent.build()


# ============================================================
# Agent route
# ============================================================


class TestAgentRoute:
    @pytest.mark.asyncio
    async def test_route_returns_rag(self):
        llm = MagicMock()

        routing_agent = MagicMock()
        routing_agent.ainvoke = AsyncMock(return_value=RouteDecision(next_step="rag"))

        llm.with_structured_output.return_value = routing_agent

        agent = Agent(llm=llm)

        result = await agent.route("What is the company's leave policy?")

        assert result == "rag"

        routing_agent.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_returns_chat(self):
        llm = MagicMock()

        routing_agent = MagicMock()
        routing_agent.ainvoke = AsyncMock(return_value=RouteDecision(next_step="chat"))

        llm.with_structured_output.return_value = routing_agent

        agent = Agent(llm=llm)

        result = await agent.route("Explain Python decorators.")

        assert result == "chat"

        routing_agent.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_sends_system_and_user_messages(self):
        llm = MagicMock()

        routing_agent = MagicMock()
        routing_agent.ainvoke = AsyncMock(return_value=RouteDecision(next_step="rag"))

        llm.with_structured_output.return_value = routing_agent

        agent = Agent(llm=llm)

        question = "What does the company security policy say?"

        await agent.route(question)

        call_args = routing_agent.ainvoke.await_args

        messages = call_args.args[0]

        assert len(messages) == 2

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == agent.system_prompt

        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == question

    @pytest.mark.asyncio
    async def test_route_rejects_empty_question(self):
        llm = MagicMock()

        agent = Agent(llm=llm)

        with pytest.raises(
            ValueError,
            match="Question cannot be empty",
        ):
            await agent.route("")

        llm.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_builds_agent_lazily(self):
        llm = MagicMock()

        routing_agent = MagicMock()
        routing_agent.ainvoke = AsyncMock(return_value=RouteDecision(next_step="chat"))

        llm.with_structured_output.return_value = routing_agent

        agent = Agent(llm=llm)

        assert agent.agent is None

        result = await agent.route("Hello")

        assert result == "chat"
        assert agent.agent is routing_agent

        llm.with_structured_output.assert_called_once_with(
            RouteDecision,
        )

    @pytest.mark.asyncio
    async def test_route_propagates_llm_error(self):
        llm = MagicMock()

        routing_agent = MagicMock()
        routing_agent.ainvoke = AsyncMock(side_effect=RuntimeError("routing failed"))

        llm.with_structured_output.return_value = routing_agent

        agent = Agent(llm=llm)

        with pytest.raises(
            RuntimeError,
            match="routing failed",
        ):
            await agent.route("What is the leave policy?")

    @pytest.mark.asyncio
    async def test_route_rejects_invalid_structured_response(self):
        llm = MagicMock()

        routing_agent = MagicMock()
        routing_agent.ainvoke = AsyncMock(
            return_value={
                "next_step": "rag",
            }
        )

        llm.with_structured_output.return_value = routing_agent

        agent = Agent(llm=llm)

        with pytest.raises(
            ValueError,
            match="invalid structured response",
        ):
            await agent.route("What is the leave policy?")


# ============================================================
# get_agent
# ============================================================


class TestGetAgent:
    def test_get_agent_returns_existing_agent(self):
        llm = MagicMock()
        existing_agent = MagicMock()

        agent = Agent(llm=llm)
        agent.agent = existing_agent

        result = agent.get_agent()

        assert result is existing_agent

        llm.with_structured_output.assert_not_called()

    def test_get_agent_builds_when_missing(self):
        llm = MagicMock()
        structured_agent = MagicMock()

        llm.with_structured_output.return_value = structured_agent

        agent = Agent(llm=llm)

        result = agent.get_agent()

        assert result is structured_agent
        assert agent.agent is structured_agent

        llm.with_structured_output.assert_called_once_with(
            RouteDecision,
        )
