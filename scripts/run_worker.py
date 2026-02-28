"""Start a Prefect worker to execute deployed flows.

Run this on any machine that should execute training jobs:
    uv run python scripts/run_worker.py
"""

from prefect.workers.process import ProcessWorker


if __name__ == "__main__":
    worker = ProcessWorker(work_pool_name="stream-worker")
    worker.start()
