"""LLM observability helpers."""

import logging
import time
from typing import Any, Awaitable, Callable

from src.utils.loggers import get_logger, log_event
from src.utils.metrics import (
    LLM_CALLS,
    LLM_COST,
    LLM_LATENCY,
    LLM_TOKENS,
)

logger = get_logger(__name__)


# ============================================================
# MODEL PRICING
# ============================================================

MODEL_PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-oss-20b": {
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
    },
}


# ============================================================
# MODEL NAME
# ============================================================


def get_model_name(llm: Any) -> str:
    """Return a stable model identifier from a LangChain LLM."""

    model_name = getattr(llm, "model_name", None)

    if model_name:
        return str(model_name)

    model_name = getattr(llm, "model", None)

    if model_name:
        return str(model_name)

    model_name = getattr(llm, "model_id", None)

    if model_name:
        return str(model_name)

    return type(llm).__name__


# ============================================================
# USAGE EXTRACTION
# ============================================================


def extract_usage(response: Any) -> dict[str, int]:
    """Extract normalized token usage from an LLM response."""

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    if usage:
        input_tokens = int(usage.get("input_tokens", 0))

        output_tokens = int(usage.get("output_tokens", 0))

        total_tokens = int(
            usage.get(
                "total_tokens",
                input_tokens + output_tokens,
            )
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    response_metadata = getattr(
        response,
        "response_metadata",
        None,
    )

    if response_metadata:
        token_usage = response_metadata.get("token_usage")

        if token_usage:
            input_tokens = int(token_usage.get("prompt_tokens", 0))

            output_tokens = int(token_usage.get("completion_tokens", 0))

            total_tokens = int(
                token_usage.get(
                    "total_tokens",
                    input_tokens + output_tokens,
                )
            )

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


# ============================================================
# COST
# ============================================================


def calculate_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate estimated LLM cost in USD."""

    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return 0.0

    return (input_tokens / 1_000_000) * pricing["input_per_1m"] + (
        output_tokens / 1_000_000
    ) * pricing["output_per_1m"]


# ============================================================
# METRIC RECORDING
# ============================================================


def record_llm_metrics(
    *,
    operation: str,
    model: str,
    status: str,
    elapsed_seconds: float,
    usage: dict[str, int],
) -> float:
    """Record Prometheus metrics for an LLM call."""

    LLM_CALLS.labels(
        operation=operation,
        model=model,
        status=status,
    ).inc()

    LLM_LATENCY.labels(
        operation=operation,
        model=model,
    ).observe(elapsed_seconds)

    LLM_TOKENS.labels(
        operation=operation,
        model=model,
        direction="input",
    ).inc(usage["input_tokens"])

    LLM_TOKENS.labels(
        operation=operation,
        model=model,
        direction="output",
    ).inc(usage["output_tokens"])

    cost = calculate_cost(
        model=model,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
    )

    if cost > 0:
        LLM_COST.labels(
            model=model,
        ).inc(cost)

    return cost


# ============================================================
# CENTRAL LLM INVOCATION WRAPPER
# ============================================================


async def invoke_llm(
    *,
    llm: Any,
    operation: str,
    invoke: Callable[[], Awaitable[Any]],
) -> Any:
    """
    Execute an async LLM invocation with observability.

    Handles:
        - latency
        - token usage
        - estimated cost
        - Prometheus metrics
        - structured logging
    """

    model = get_model_name(llm)

    start_time = time.perf_counter()

    log_event(
        logger,
        level=logging.DEBUG,
        event="llm.call.started",
        operation=operation,
        model=model,
    )

    try:
        response = await invoke()

    except Exception as exc:
        elapsed = time.perf_counter() - start_time

        record_llm_metrics(
            operation=operation,
            model=model,
            status="error",
            elapsed_seconds=elapsed,
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )

        log_event(
            logger,
            level=logging.ERROR,
            event="llm.call.failed",
            operation=operation,
            model=model,
            duration_ms=round(
                elapsed * 1000,
                2,
            ),
            error_type=type(exc).__name__,
        )

        raise

    elapsed = time.perf_counter() - start_time

    usage = extract_usage(response)

    cost = record_llm_metrics(
        operation=operation,
        model=model,
        status="success",
        elapsed_seconds=elapsed,
        usage=usage,
    )

    log_event(
        logger,
        level=logging.INFO,
        event="llm.call.completed",
        operation=operation,
        model=model,
        duration_ms=round(
            elapsed * 1000,
            2,
        ),
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        cost_usd=round(cost, 8),
    )

    return response
