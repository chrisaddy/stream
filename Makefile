.PHONY: setup dev test lint format deploy-flows worker serve clean

setup:
	uv sync
	docker compose up -d

dev:
	uv run uvicorn stream.app.main:app --host 0.0.0.0 --port 5000 --reload

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# Deploy all training flows to Prefect Cloud
deploy-flows:
	uv run python scripts/deploy_flows.py

# Start a local worker to execute Prefect Cloud flow runs
worker:
	uv run python scripts/run_worker.py

serve:
	uv run uvicorn stream.app.main:app --host 0.0.0.0 --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
