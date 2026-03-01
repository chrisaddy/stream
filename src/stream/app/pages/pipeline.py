"""System Status page — subsystem health + Prefect runs."""

from fasthtml.common import *
from stream.app.components import (
    DataGrid, DataReadout, DiagnosticFrame, Page,
)


def pipeline_page():
    return Page("System Status", "/pipeline",
        DiagnosticFrame(
            "SYSTEM STATUS",

            P("Live health of each STREAM subsystem. Pipeline DAGs and model freshness "
              "are shown on each model's page (Illicit, Fees, Lightning, Onboarding).",
              style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;"),

            Div(
                id="system-status",
                **{"hx-get": "/api/v1/system/status", "hx-trigger": "load, every 30s", "hx-swap": "innerHTML"},
            ),

            status="LIVE",
            footer_left="HEALTH CHECK",
            footer_right="ALL SUBSYSTEMS",
        ),

        # Recent Prefect runs (cross-model)
        DiagnosticFrame(
            "RECENT FLOW RUNS",

            P("Cross-model Prefect flow runs. Requires PREFECT_API_URL and PREFECT_API_KEY.",
              style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;"),

            Table(
                Thead(Tr(
                    Th("Flow"), Th("Started"), Th("Duration"), Th("Status"),
                )),
                Tbody(
                    id="recent-runs",
                    **{"hx-get": "/api/v1/pipeline/recent-runs", "hx-trigger": "load, every 60s", "hx-swap": "innerHTML"},
                ),
                cls="spark-table",
            ),

            status="PREFECT",
            footer_left="FLOW HISTORY",
            footer_right="PREFECT_API",
        ),

        # All-models freshness overview
        DiagnosticFrame(
            "MODEL FRESHNESS",

            Div(
                id="model-freshness",
                **{"hx-get": "/api/v1/pipeline/freshness", "hx-trigger": "load, every 60s", "hx-swap": "innerHTML"},
            ),

            status="OVERVIEW",
            footer_left="ALL MODELS",
            footer_right="R2 + MEMORY",
        ),
    )
