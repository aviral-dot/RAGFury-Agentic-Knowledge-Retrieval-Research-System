uvicorn api.main:app --reload --reload-dir api --reload-dir src

uv run deepeval test run "tests/evals/components/rag/test_retriever_eval.py::test_retriever_component[golden0]" -v -s


uv run uvicorn api.main:app --reload --reload-dir api --reload-dir src

langgraph dev --no-reload

uv run uvicorn api.main:app

langgraph dev --allow-blocking

uv run deepeval test run "tests/evals/components/rag/test_grader_eval.py::test_grader_component[golden0]" -v -s

uv run deepeval test run "tests/evals/components/rag/test_generation_eval.py::test_generation_component[golden0]" -v -s

uv run deepeval test run "tests/evals/components/rag/test_rewrite_eval.py::test_rewrite_component[golden0]" -v -s

uv run pytest tests/evals/components/rag -v

rq worker memory --worker-class rq.worker.SimpleWorker

python -c "import asyncio, selectors, uvicorn; config=uvicorn.Config('api.main:app', host='0.0.0.0', port=8000); server=uvicorn.Server(config); asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())).run(server.serve())"
