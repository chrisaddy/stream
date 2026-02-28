"""Deploy Prefect flows to Prefect Cloud.

Creates deployments that run every 2 hours with email notifications.
Runs are executed by a local worker process (scripts/run_worker.py).
"""

from prefect import deploy
from prefect.events.actions import RunAutomation
from prefect.automations import Automation, EventTrigger, Posture
from prefect.events.schemas.automations import EventTrigger as EventTriggerSchema

from stream.illicit.pipeline import train_illicit_pipeline
from stream.fees.pipeline import train_fees_pipeline
from stream.lightning.pipeline import train_lightning_pipeline
from stream.onboarding.pipeline import train_onboarding_pipeline


CRON_EVERY_2H = "0 */2 * * *"


if __name__ == "__main__":
    deploy(
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
        work_pool_name="stream-worker",
    )
    print("All flows deployed to Prefect Cloud (every 2 hours).")
    print()
    print("IMPORTANT: Set up email notifications in Prefect Cloud UI:")
    print("  1. Go to https://app.prefect.cloud -> Automations")
    print("  2. Create automation: 'Email on flow run completion'")
    print("     Trigger: Flow run enters state 'Completed' OR 'Failed' OR 'Crashed'")
    print("     Action: Send email to chris.william.addy@gmail.com")
    print()
    print("  Or use the setup_automations.py script to configure via API.")
