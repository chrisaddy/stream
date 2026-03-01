"""Compliance Alert Queue page."""

from fasthtml.common import *
from stream.app.components import (
    DataGrid, DataReadout, DiagnosticFrame, Page, StatusBadge,
)


def alerts_page():
    return Page("Alert Queue", "/alerts",
        DiagnosticFrame(
            "COMPLIANCE ALERT QUEUE",

            Div(
                id="alert-stats",
                **{"hx-get": "/api/v1/alerts/stats", "hx-trigger": "load, every 10s", "hx-swap": "innerHTML"},
            ),

            # Alert feed
            Table(
                Thead(Tr(
                    Th("Time"), Th("TX ID"), Th("Risk Score"), Th("Label"), Th("Status"), Th("Action"),
                )),
                Tbody(
                    id="alert-feed",
                    **{"hx-get": "/api/v1/alerts/recent", "hx-trigger": "load, every 10s", "hx-swap": "innerHTML"},
                ),
                cls="spark-table",
            ),

            status="MONITORING",
            footer_left="[ALT-001] SAR PIPELINE",
            footer_right="LLM_NARRATIVES",
        ),

        # Alert detail / narrative
        DiagnosticFrame(
            "ALERT DETAIL",

            Div(
                P("Select an alert from the queue to view SHAP explanation and AI-generated compliance narrative.",
                  style="color: var(--fg-dim);"),
                id="alert-detail",
            ),

            status="SELECT_ALERT",
            footer_left="[ALT-002] SHAP + LLM",
            footer_right="AUDIT_TRAIL",
        ),
    )
