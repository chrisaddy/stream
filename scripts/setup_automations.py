"""Set up Prefect Cloud automations for email notifications.

Sends email on every flow run completion (success or failure).
"""

from prefect.automations import Automation, EventTrigger, Posture
from prefect.events.actions import SendNotification


EMAIL = "chris.william.addy@gmail.com"


def create_email_automations():
    """Create automations that email on flow run success and failure."""

    # Email on any flow run completion (success)
    success_automation = Automation(
        name="Email on flow success",
        description="Send email when any training flow completes successfully",
        trigger=EventTrigger(
            expect={"prefect.flow-run.Completed"},
            match={
                "prefect.resource.id": "prefect.flow-run.*",
            },
            posture=Posture.Reactive,
            threshold=1,
            within=0,
        ),
        actions=[
            SendNotification(
                block_document_id=None,  # Will use default email block
                subject="[Stream ML] Flow Run Succeeded: {{ flow_run.name }}",
                body=(
                    "Flow: {{ flow.name }}\n"
                    "Run: {{ flow_run.name }}\n"
                    "State: {{ flow_run.state.name }}\n"
                    "Duration: {{ flow_run.total_run_time }}\n"
                    "Timestamp: {{ flow_run.state.timestamp }}\n"
                    "\n"
                    "View in Prefect Cloud: {{ flow_run|ui_url }}"
                ),
            )
        ],
    )
    success_automation.create()
    print(f"Created automation: {success_automation.name}")

    # Email on any flow run failure
    failure_automation = Automation(
        name="Email on flow failure",
        description="Send email when any training flow fails or crashes",
        trigger=EventTrigger(
            expect={"prefect.flow-run.Failed", "prefect.flow-run.Crashed"},
            match={
                "prefect.resource.id": "prefect.flow-run.*",
            },
            posture=Posture.Reactive,
            threshold=1,
            within=0,
        ),
        actions=[
            SendNotification(
                block_document_id=None,
                subject="[Stream ML] FAILURE: {{ flow_run.name }}",
                body=(
                    "ALERT: Flow run failed!\n\n"
                    "Flow: {{ flow.name }}\n"
                    "Run: {{ flow_run.name }}\n"
                    "State: {{ flow_run.state.name }}\n"
                    "Message: {{ flow_run.state.message }}\n"
                    "Timestamp: {{ flow_run.state.timestamp }}\n"
                    "\n"
                    "View in Prefect Cloud: {{ flow_run|ui_url }}"
                ),
            )
        ],
    )
    failure_automation.create()
    print(f"Created automation: {failure_automation.name}")


if __name__ == "__main__":
    print(f"Setting up email notifications to {EMAIL}...")
    print()
    print("NOTE: Prefect Cloud email notifications require an Email notification block.")
    print("Set this up first in Prefect Cloud UI:")
    print("  1. Go to Blocks -> Add Block -> Email")
    print("  2. Configure with your email settings")
    print("  3. Then run this script")
    print()
    print("Alternatively, configure automations directly in the Prefect Cloud UI:")
    print("  https://app.prefect.cloud -> Automations -> Create")
    print(f"  Trigger: Flow run state change (Completed, Failed, Crashed)")
    print(f"  Action: Send notification to {EMAIL}")
    print()

    try:
        create_email_automations()
        print("\nAutomations created successfully!")
    except Exception as e:
        print(f"\nFailed to create automations via API: {e}")
        print("Please create them manually in the Prefect Cloud UI.")
