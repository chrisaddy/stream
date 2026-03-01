"""Lightning Network analysis dashboard."""

from fasthtml.common import *
from stream.app.components import (
    Card, DataGrid, DataReadout, DiagnosticFrame, Page,
)
from stream.app.pages.models_page import render_model_card


def lightning_page():
    return Page("Lightning Network", "/lightning",
        DiagnosticFrame(
            "LIGHTNING NETWORK ANALYSIS",

            DataGrid(
                DataReadout("TOTAL_NODES", "---", highlight=True),
                DataReadout("TOTAL_CHANNELS", "---"),
                DataReadout("NETWORK_CAPACITY", "--- BTC"),
                DataReadout("AVG_CHANNEL_SIZE", "--- sats"),
            ),

            Div(
                id="ln-stats",
                **{"hx-get": "/api/v1/lightning/stats", "hx-trigger": "load, every 60s", "hx-swap": "innerHTML"},
            ),

            status="LIVE",
            footer_left="[LN-001] TOPOLOGY",
            footer_right="MEMPOOL.SPACE",
        ),

        # Top routing nodes
        DiagnosticFrame(
            "TOP ROUTING NODES",

            Table(
                Thead(Tr(
                    Th("Rank"), Th("Alias"), Th("Channels"), Th("Capacity (BTC)"), Th("Routing Score"),
                )),
                Tbody(
                    id="top-nodes",
                    **{"hx-get": "/api/v1/lightning/top-nodes", "hx-trigger": "load", "hx-swap": "innerHTML"},
                ),
                cls="spark-table",
            ),

            status="RANKED",
            footer_left="[LN-002] CENTRALITY",
            footer_right="BY_CONNECTIVITY",
        ),

        # Node evaluator
        DiagnosticFrame(
            "NODE EVALUATOR",

            P("Enter a Lightning node public key to evaluate its routing potential.",
              style="color: var(--fg-dim); margin-bottom: 12px;"),

            Form(
                Div(
                    Input(type="text", name="pubkey", placeholder="Node public key...",
                          cls="spark-input", style="margin-bottom: 8px;"),
                    Button("Evaluate Node", cls="spark-btn", type="submit"),
                ),
                **{"hx-post": "/api/v1/lightning/evaluate", "hx-target": "#node-result"},
            ),

            Div(id="node-result", style="margin-top: 16px;"),

            status="INTERACTIVE",
            footer_left="[LN-003] SCORING",
            footer_right="ENTER_PUBKEY",
        ),

        render_model_card("lightning-lgbm"),
    )
