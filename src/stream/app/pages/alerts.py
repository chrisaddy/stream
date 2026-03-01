"""Compliance Alert Queue page."""

from fasthtml.common import *
from stream.app.components import (
    DataGrid, DataReadout, DiagnosticFrame, Page, StatusBadge, Tip,
)


def alerts_page():
    return Page("Alert Queue", "/alerts",
        DiagnosticFrame(
            "ALERT QUEUE",

            P("Transactions scored above the risk threshold are automatically queued here for compliance review. "
              "Each alert includes a SHAP explanation (why the model flagged it) and an AI-generated narrative "
              "for the compliance report.",
              style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;"),

            status="OVERVIEW",
            footer_left="COMPLIANCE PIPELINE",
            footer_right="SHAP + LLM",
        ),

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
            footer_left="SUSPICIOUS ACTIVITY REPORTS",
            footer_right="LLM NARRATIVES",
        ),

        # Alert detail / narrative
        DiagnosticFrame(
            "ALERT DETAIL",

            Div(
                P("Select an alert from the queue to view ",
                  Tip("SHAP"), " explanation and ",
                  Tip("LLM", "Large Language Model — AI that generates human-readable text. Used here to convert technical SHAP values into compliance narratives."),
                  "-generated compliance narrative.",
                  style="color: var(--fg-subtle);"),
                id="alert-detail",
            ),

            status="SELECT_ALERT",
            footer_left="SHAP + LLM EXPLANATION",
            footer_right="AUDIT TRAIL",
        ),
    )
