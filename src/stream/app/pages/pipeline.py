"""Pipeline visibility page — Live Prefect Cloud status."""

from fasthtml.common import *
from stream.app.components import (
    DiagnosticFrame, Page, StatusBadge,
)


def pipeline_page():
    return Page("Pipeline", "/pipeline",
        # DAG visualization
        DiagnosticFrame(
            "TRAINING PIPELINE DAG",

            # Illicit pipeline
            H4("Illicit Detection Pipeline", style="color: var(--fg-green); font-size: 11px; letter-spacing: 1px; margin-bottom: 12px;"),
            Div(
                Span("LOAD_DATA", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("TEMPORAL_SPLIT", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("TRAIN_XGBOOST", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("EVALUATE", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("SAVE_TO_R2", cls="flow-node active"),
                style="margin-bottom: 20px; overflow-x: auto; white-space: nowrap;",
            ),

            # Fee pipeline
            H4("Fee Estimation Pipeline", style="color: var(--fg-green); font-size: 11px; letter-spacing: 1px; margin-bottom: 12px;"),
            Div(
                Span("COLLECT_MEMPOOL", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("FEATURIZE", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("TRAIN_LGBM", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("SAVE_TO_R2", cls="flow-node active"),
                style="margin-bottom: 20px; overflow-x: auto; white-space: nowrap;",
            ),

            # Lightning pipeline
            H4("Lightning Analysis Pipeline", style="color: var(--fg-green); font-size: 11px; letter-spacing: 1px; margin-bottom: 12px;"),
            Div(
                Span("FETCH_TOPOLOGY", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("BUILD_GRAPH", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("COMPUTE_FEATURES", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("TRAIN_MODEL", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("SAVE_TO_R2", cls="flow-node active"),
                style="margin-bottom: 20px; overflow-x: auto; white-space: nowrap;",
            ),

            # Onboarding pipeline
            H4("Onboarding Risk Pipeline", style="color: var(--fg-green); font-size: 11px; letter-spacing: 1px; margin-bottom: 12px;"),
            Div(
                Span("GENERATE_DATA", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("TRAIN_LOGISTIC", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("TRAIN_XGBOOST", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("EVALUATE", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("SAVE_TO_R2", cls="flow-node active"),
                style="margin-bottom: 20px; overflow-x: auto; white-space: nowrap;",
            ),

            # Feedback pipeline
            H4("Feedback Retraining Pipeline", style="color: var(--fg-iris); font-size: 11px; letter-spacing: 1px; margin-bottom: 12px;"),
            Div(
                Span("LOAD_REVIEWS", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("JOIN_FEATURES", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("TRAIN_LGBM", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("EVALUATE", cls="flow-node active"), Span("->", cls="flow-arrow"),
                Span("DEPLOY_LIVE", cls="flow-node active"),
                style="margin-bottom: 20px; overflow-x: auto; white-space: nowrap;",
            ),

            status="5 FLOWS",
            footer_left="[PIP-001] PREFECT CLOUD",
            footer_right="ORCHESTRATION",
        ),

        # Recent runs
        DiagnosticFrame(
            "RECENT FLOW RUNS",

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

            status="LIVE",
            footer_left="[PIP-002] HISTORY",
            footer_right="PREFECT_API",
        ),

        # Model freshness
        DiagnosticFrame(
            "MODEL FRESHNESS",

            Div(
                id="model-freshness",
                **{"hx-get": "/api/v1/pipeline/freshness", "hx-trigger": "load, every 60s", "hx-swap": "innerHTML"},
            ),

            status="MONITORING",
            footer_left="[PIP-003] DRIFT_WATCH",
            footer_right="RETRAINING_SCHEDULE",
        ),
    )
