FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install locked dependencies
RUN uv sync --frozen

# Copy application
COPY . .

# Start the RQ memory worker
CMD ["uv", "run", "rq", "worker", "memory", "--worker-class", "rq.worker.SimpleWorker"]