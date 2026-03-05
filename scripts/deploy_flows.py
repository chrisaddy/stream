"""Deploy and serve Prefect flows locally.

Creates deployments with Prefect Cloud and starts
an in-process server to execute scheduled runs.

- Data collection: every hour (lightweight, ~30s)
- Training: every 6 hours (uses accumulated data)

    uv run python scripts/deploy_flows.py
"""

from prefect import serve

from stream.data.pipeline import collect_data_pipeline
from stream.fees.pipeline import train_fees_pipeline
from stream.illicit.pipeline import train_illicit_pipeline
from stream.lightning.pipeline import train_lightning_pipeline
from stream.onboarding.pipeline import train_onboarding_pipeline

CRON_EVERY_1H = "0 * * * *"
CRON_EVERY_6H = "0 */6 * * *"


if __name__ == "__main__":
    print("Registering deployments with Prefect Cloud...")
    print("  Data collection: every 1 hour")
    print("  Training: every 6 hours")
    print("Starting in-process server to execute runs...\n")

    serve(
        collect_data_pipeline.to_deployment(
            name="hourly-data-collection",
            cron=CRON_EVERY_1H,
            tags=["data", "collection"],
        ),
        train_illicit_pipeline.to_deployment(
            name="illicit-detection-training",
            cron=CRON_EVERY_6H,
            tags=["ml", "illicit", "training"],
        ),
        train_fees_pipeline.to_deployment(
            name="fee-estimation-training",
            cron=CRON_EVERY_6H,
            tags=["ml", "fees", "training"],
        ),
        train_lightning_pipeline.to_deployment(
            name="lightning-network-analysis",
            cron=CRON_EVERY_6H,
            tags=["ml", "lightning", "training"],
        ),
        train_onboarding_pipeline.to_deployment(
            name="onboarding-risk-scoring",
            cron=CRON_EVERY_6H,
            tags=["ml", "onboarding", "training"],
        ),
    )
