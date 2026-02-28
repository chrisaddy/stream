"""Deploy and serve Prefect flows.

Creates deployments with Prefect Cloud (cron every 2h) and starts
an in-process server to execute scheduled runs.

    uv run python scripts/deploy_flows.py
"""

from prefect import serve

from stream.illicit.pipeline import train_illicit_pipeline
from stream.fees.pipeline import train_fees_pipeline
from stream.lightning.pipeline import train_lightning_pipeline
from stream.onboarding.pipeline import train_onboarding_pipeline


CRON_EVERY_2H = "0 */2 * * *"
WORK_POOL = "ml-work-pool"


if __name__ == "__main__":
    print("Registering deployments with Prefect Cloud (every 2 hours)...")
    print("Starting in-process server to execute runs...\n")

    serve(
        train_illicit_pipeline.to_deployment(
            name="illicit-detection-training",
            cron=CRON_EVERY_2H,
            tags=["ml", "illicit", "training"],
        ),
        train_fees_pipeline.to_deployment(
            name="fee-estimation-training",
            cron=CRON_EVERY_2H,
            tags=["ml", "fees", "training"],
        ),
        train_lightning_pipeline.to_deployment(
            name="lightning-network-analysis",
            cron=CRON_EVERY_2H,
            tags=["ml", "lightning", "training"],
        ),
        train_onboarding_pipeline.to_deployment(
            name="onboarding-risk-scoring",
            cron=CRON_EVERY_2H,
            tags=["ml", "onboarding", "training"],
        ),
    )
