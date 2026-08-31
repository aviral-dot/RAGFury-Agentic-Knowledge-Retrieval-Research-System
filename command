uvicorn api.main:app --reload --reload-dir api --reload-dir src

uv run deepeval test run "tests/evals/components/rag/test_retriever_eval.py::test_retriever_component[golden0]" -v -s


uv run uvicorn api.main:app --reload --reload-dir api --reload-dir src

langgraph dev --no-reload

uv run uvicorn api.main:app

langgraph dev --allow-blocking
