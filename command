uvicorn api.main:app --reload --reload-dir api --reload-dir src

uv run deepeval test run "tests/evals/components/rag/test_retriever_eval.py::test_retriever_component[golden0]" -v -s
