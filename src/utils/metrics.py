"""Prometheus metrics for RAGFury."""

from prometheus_client import Counter, Gauge, Histogram

# ============================================================
# API
# ============================================================

REQUEST_COUNT = Counter(
    "ragfury_requests_total",
    "Total number of HTTP requests.",
    ["route", "method", "status"],
)

ERROR_COUNT = Counter(
    "ragfury_errors_total",
    "Total application errors.",
    ["component", "error_type"],
)

REQUEST_LATENCY = Histogram(
    "ragfury_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["route", "method"],
)

REQUESTS_IN_PROGRESS = Gauge(
    "ragfury_requests_in_progress",
    "Number of requests currently being processed.",
)


# ============================================================
# GRAPH
# ============================================================

GRAPH_LATENCY = Histogram(
    "ragfury_graph_execution_seconds",
    "LangGraph execution latency.",
    ["outcome"],
)


# ============================================================
# RETRIEVAL
# ============================================================

RETRIEVAL_LATENCY = Histogram(
    "ragfury_retrieval_latency_seconds",
    "Retrieval latency.",
)

RETRIEVAL_ATTEMPTS = Counter(
    "ragfury_retrieval_attempts_total",
    "Total retrieval attempts.",
)

RETRIEVED_DOCUMENTS = Histogram(
    "ragfury_retrieved_documents",
    "Number of documents returned by retrieval.",
)


# ============================================================
# RERANKER
# ============================================================

RERANKER_LATENCY = Histogram(
    "ragfury_reranker_latency_seconds",
    "Cross encoder reranking latency.",
)

RERANKER_INPUT_DOCUMENTS = Histogram(
    "ragfury_reranker_input_documents",
    "Number of documents passed to reranker.",
)

RERANKER_OUTPUT_DOCUMENTS = Histogram(
    "ragfury_reranker_output_documents",
    "Number of documents returned after reranking.",
)


# ============================================================
# REWRITE
# ============================================================

REWRITE_COUNT = Counter(
    "ragfury_rewrites_total",
    "Total query rewrites.",
)


# ============================================================
# ABSTENTION
# ============================================================

ABSTENTION_COUNT = Counter(
    "ragfury_abstentions_total",
    "Total requests ending in abstention.",
)


LLM_CALLS = Counter(
    "ragfury_llm_calls_total",
    "Total LLM calls.",
    ["operation", "model", "status"],
)

LLM_LATENCY = Histogram(
    "ragfury_llm_latency_seconds",
    "LLM call latency.",
    ["operation", "model"],
)

LLM_TOKENS = Counter(
    "ragfury_llm_tokens_total",
    "Total LLM tokens.",
    ["operation", "model", "direction"],
)

LLM_COST = Counter(
    "ragfury_llm_cost_usd_total",
    "Estimated total LLM cost in USD.",
    ["model"],
)


GUARDRAIL_LATENCY = Histogram(
    "ragfury_guardrail_latency_seconds",
    "Guardrail execution latency.",
    ["stage"],
)

GUARDRAIL_BLOCKS = Counter(
    "ragfury_guardrail_blocks_total",
    "Total guardrail blocks.",
    ["stage"],
)

GUARDRAIL_ERRORS = Counter(
    "ragfury_guardrail_errors_total",
    "Total guardrail errors.",
    ["stage"],
)
