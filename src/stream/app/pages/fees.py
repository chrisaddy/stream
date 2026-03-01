"""Fee Estimation dashboard page."""

from fasthtml.common import *
from stream.app.components import (
    Card, DataGrid, DataReadout, DiagnosticFrame, MetricItem, MetricsRow, Page,
)
from stream.app.pages.models_page import render_model_card


def fees_page():
    return Page("Fee Estimation", "/fees",
        DiagnosticFrame(
            "FEE ESTIMATION ENGINE",

            # Live fee recommendations
            DataGrid(
                DataReadout("NEXT_BLOCK (1)", "--- sat/vB", highlight=True),
                DataReadout("30_MIN (3)", "--- sat/vB"),
                DataReadout("1_HOUR (6)", "--- sat/vB"),
                DataReadout("ECONOMY (12)", "--- sat/vB"),
            ),

            Div(
                id="fee-live",
                **{"hx-get": "/api/v1/fees/live", "hx-trigger": "load, every 30s", "hx-swap": "innerHTML"},
            ),

            status="LIVE",
            footer_left="[FEE-001] MEMPOOL ANALYSIS",
            footer_right="SAT/VB",
        ),

        # ML vs Bitcoin Core comparison
        DiagnosticFrame(
            "ML vs BITCOIN CORE",

            Table(
                Thead(Tr(
                    Th("Target"), Th("ML Estimate"), Th("Bitcoin Core"), Th("Difference"),
                )),
                Tbody(id="fee-comparison",
                    **{"hx-get": "/api/v1/fees/compare", "hx-trigger": "load, every 60s", "hx-swap": "innerHTML"},
                ),
                cls="spark-table",
            ),

            status="COMPARISON",
            footer_left="[FEE-002] BENCHMARK",
            footer_right="ADVANTAGE_ML",
        ),

        # Mempool state
        DiagnosticFrame(
            "MEMPOOL STATE",

            Div(
                id="mempool-state",
                **{"hx-get": "/api/v1/fees/mempool-state", "hx-trigger": "load, every 30s", "hx-swap": "innerHTML"},
                style="min-height: 100px;",
            ),

            status="MONITORING",
            footer_left="[FEE-003] LIVE DATA",
            footer_right="MEMPOOL.SPACE",
        ),

        render_model_card("fee-lgbm"),
    )
