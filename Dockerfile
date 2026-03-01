FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev --no-extra training

EXPOSE 5000
ENTRYPOINT []
CMD exec .venv/bin/uvicorn stream.app.main:app --host 0.0.0.0 --port ${PORT:-5000} --log-level info
