set dotenv-load := true

default:
    @just --list

# Install dependencies and start local services
setup:
    uv sync
    docker compose up -d

# Run dashboard in dev mode with hot reload
dev:
    uv run uvicorn stream.app.main:app --host 0.0.0.0 --port 5001 --reload

# Run test suite
test:
    uv run pytest tests/ -v

# Lint and check formatting
lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

# Auto-fix lint and format
format:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/

# Run hourly data collection once (fee snapshots + lightning topology)
collect-data:
    uv run python -m stream.data.pipeline

# Deploy all flows to Prefect Cloud (collection hourly, training every 6h)
deploy-flows:
    uv run python scripts/deploy_flows.py

# Start a local Prefect worker to execute flow runs
worker:
    uv run python scripts/run_worker.py

# Set up Prefect Cloud email automations
setup-automations:
    uv run python scripts/setup_automations.py

# Run the app in production mode
serve:
    uv run uvicorn stream.app.main:app --host 0.0.0.0 --port ${PORT:-5000}

# Remove build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
