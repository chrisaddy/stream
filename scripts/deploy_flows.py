"""Deploy Prefect flows to Prefect Cloud with managed execution.

Creates deployments that pull code from GitHub and run on
Prefect's managed infrastructure (every 2h cron schedule).

    uv run python scripts/deploy_flows.py
"""

import asyncio
from prefect import flow

CRON_EVERY_2H = "0 */2 * * *"
WORK_POOL = "ml-work-pool"
REPO_URL = "https://github.com/chrisaddy/stream.git"

FLOWS = [
    {
        "entrypoint": "src/stream/illicit/pipeline.py:train_illicit_pipeline",
        "name": "illicit-detection-training",
        "tags": ["ml", "illicit", "training"],
    },
    {
        "entrypoint": "src/stream/fees/pipeline.py:train_fees_pipeline",
        "name": "fee-estimation-training",
        "tags": ["ml", "fees", "training"],
    },
    {
        "entrypoint": "src/stream/lightning/pipeline.py:train_lightning_pipeline",
        "name": "lightning-network-analysis",
        "tags": ["ml", "lightning", "training"],
    },
    {
        "entrypoint": "src/stream/onboarding/pipeline.py:train_onboarding_pipeline",
        "name": "onboarding-risk-scoring",
        "tags": ["ml", "onboarding", "training"],
    },
]


async def main():
    print("Deploying flows to Prefect Cloud (managed execution, every 2h)...\n")

    for f in FLOWS:
        loaded = await flow.from_source(source=REPO_URL, entrypoint=f["entrypoint"])
        deployment_id = await loaded.deploy(
            name=f["name"],
            work_pool_name=WORK_POOL,
            cron=CRON_EVERY_2H,
            tags=f["tags"],
            build=False,
            push=False,
        )
        print(f"  Deployed {f['name']} -> {deployment_id}")

    print("\nDone! All flows deployed.")


if __name__ == "__main__":
    asyncio.run(main())
