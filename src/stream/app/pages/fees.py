"""Fee Estimation dashboard page."""

from fasthtml.common import *
from stream.app.components import (
    Card, DataGrid, DataReadout, DiagnosticFrame, MetricItem, MetricsRow, Page, Tip,
)
from stream.app.pages.models_page import render_model_card


def fees_page():
    return Page("Fee Estimation", "/fees",
        DiagnosticFrame(
            "FEE ESTIMATION",

            P("Bitcoin transactions compete for limited block space by offering fees. "
              "This page shows ML-predicted optimal fee rates compared to Bitcoin Core's built-in estimator, "
              "using live mempool data.",
              style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;"),

            status="OVERVIEW",
            footer_left="BITCOIN FEE PREDICTION",
            footer_right="LIVE_DATA",
        ),

        DiagnosticFrame(
            "FEE ESTIMATION ENGINE",

            # Live fee recommendations
            DataGrid(
                DataReadout(Tip("NEXT_BLOCK (1)", "Target: 1 block (~10 min). Highest fee tier for fastest confirmation."), "--- sat/vB", highlight=True),
                DataReadout(Tip("30_MIN (3)", "Target: 3 blocks (~30 min). Good balance of speed and cost."), "--- sat/vB"),
                DataReadout(Tip("1_HOUR (6)", "Target: 6 blocks (~1 hour). Standard priority."), "--- sat/vB"),
                DataReadout(Tip("ECONOMY (12)", "Target: 12 blocks (~2 hours). Lower priority, cheapest fee."), "--- sat/vB"),
            ),

            Div(
                id="fee-live",
                **{"hx-get": "/api/v1/fees/live", "hx-trigger": "load, every 30s", "hx-swap": "innerHTML"},
            ),

            status="LIVE",
            footer_left="LIVE FEE RECOMMENDATIONS",
            footer_right="MEMPOOL.SPACE",
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
            footer_left="ML vs CORE BENCHMARK",
            footer_right="COMPARISON",
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
            footer_left="LIVE MEMPOOL STATE",
            footer_right="MEMPOOL.SPACE API",
        ),

        render_model_card("fee-lgbm"),
    )
