"""Deploy Prefect flows to Prefect Cloud.

Creates deployments that can be triggered from the Prefect Cloud UI or API.
Runs are executed by a local worker process (scripts/run_worker.py).
"""

from prefect import deploy

from stream.illicit.pipeline import train_illicit_pipeline
from stream.fees.pipeline import train_fees_pipeline
from stream.lightning.pipeline import train_lightning_pipeline
from stream.onboarding.pipeline import train_onboarding_pipeline


if __name__ == "__main__":
    deploy(
        train_illicit_pipeline.to_deployment(
            name="illicit-detection-training",
            cron="0 */6 * * *",  # Every 6 hours
            tags=["ml", "illicit", "training"],
        ),
        train_fees_pipeline.to_deployment(
            name="fee-estimation-training",
            cron="0 * * * *",  # Every hour
            tags=["ml", "fees", "training"],
        ),
        train_lightning_pipeline.to_deployment(
            name="lightning-network-analysis",
            cron="0 */12 * * *",  # Every 12 hours
            tags=["ml", "lightning", "training"],
        ),
        train_onboarding_pipeline.to_deployment(
            name="onboarding-risk-scoring",
            cron="0 0 * * *",  # Daily
            tags=["ml", "onboarding", "training"],
        ),
        work_pool_name="stream-worker",
    )
    print("All flows deployed to Prefect Cloud.")
