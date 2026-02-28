FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src/ src/

EXPOSE 5000
ENTRYPOINT ["/bin/sh", "-c", "uv run uvicorn stream.app.main:app --host 0.0.0.0 --port ${PORT:-5000} --log-level info"]
