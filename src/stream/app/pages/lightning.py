"""Lightning Network analysis dashboard."""

from fasthtml.common import *

from stream.app.components import (
    DataGrid,
    DataReadout,
    DiagnosticFrame,
    Page,
    PipelineDag,
    Tip,
)
from stream.app.pages.models_page import render_model_card


def lightning_page():
    return Page(
        "Lightning Network",
        "/lightning",
        DiagnosticFrame(
            "LIGHTNING NETWORK",
            P(
                "The Lightning Network is Bitcoin's layer-2 scaling solution. "
                "This page analyzes network topology — node connectivity, channel capacity, "
                "and routing potential — using live data from mempool.space.",
                style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;",
            ),
            status="OVERVIEW",
            footer_left="LAYER-2 ANALYSIS",
            footer_right="LIVE_DATA",
        ),
        DiagnosticFrame(
            "LIGHTNING NETWORK ANALYSIS",
            DataGrid(
                DataReadout(
                    Tip(
                        "TOTAL_NODES",
                        "Computers running Lightning Network software. They route payments and maintain channels.",
                    ),
                    "---",
                    highlight=True,
                ),
                DataReadout(
                    Tip(
                        "TOTAL_CHANNELS",
                        "Payment channels between Lightning nodes. Funds are locked in a channel to enable off-chain transactions.",
                    ),
                    "---",
                ),
                DataReadout(
                    Tip(
                        "NETWORK_CAPACITY",
                        "Total Bitcoin locked across all Lightning channels, available for routing payments.",
                    ),
                    "--- BTC",
                ),
                DataReadout(
                    Tip("AVG_CHANNEL_SIZE", "Average amount of Bitcoin locked per channel."),
                    "--- sats",
                ),
            ),
            Div(
                id="ln-stats",
                **{
                    "hx-get": "/api/v1/lightning/stats",
                    "hx-trigger": "load, every 60s",
                    "hx-swap": "innerHTML",
                },
            ),
            status="LIVE",
            footer_left="NETWORK TOPOLOGY",
            footer_right="MEMPOOL.SPACE API",
        ),
        # Top routing nodes
        DiagnosticFrame(
            "TOP ROUTING NODES",
            Table(
                Thead(
                    Tr(
                        Th("Rank"),
                        Th("Alias"),
                        Th(
                            Tip(
                                "Channels",
                                "Payment channels between Lightning nodes. Funds are locked in a channel to enable off-chain transactions.",
                            )
                        ),
                        Th(
                            Tip(
                                "Capacity (BTC)",
                                "Total Bitcoin locked in this node's channels, available for routing payments.",
                            )
                        ),
                        Th(
                            Tip(
                                "Capacity Score",
                                "ML-predicted channel capacity score from network topology features. Falls back to heuristic when model unavailable.",
                            )
                        ),
                    )
                ),
                Tbody(
                    id="top-nodes",
                    **{
                        "hx-get": "/api/v1/lightning/top-nodes",
                        "hx-trigger": "load",
                        "hx-swap": "innerHTML",
                    },
                ),
                cls="spark-table",
            ),
            P(
                "Scores predict channel capacity from network topology features. Source indicator shows ML or HEURISTIC per node.",
                style="color: var(--fg-dim); font-size: 10px; margin-top: 8px; letter-spacing: 0.5px;",
            ),
            status="RANKED",
            footer_left="NODE CENTRALITY RANKING",
            footer_right="BY CONNECTIVITY",
        ),
        # Node evaluator
        DiagnosticFrame(
            "NODE EVALUATOR",
            P(
                "Enter a Lightning node ",
                Tip("public key"),
                " to evaluate its ",
                Tip("routing"),
                " potential.",
                style="color: var(--fg-subtle); margin-bottom: 12px;",
            ),
            Form(
                Div(
                    Input(
                        type="text",
                        name="pubkey",
                        placeholder="Node public key...",
                        cls="spark-input",
                        style="margin-bottom: 8px;",
                    ),
                    Button("Evaluate Node", cls="spark-btn", type="submit"),
                ),
                **{"hx-post": "/api/v1/lightning/evaluate", "hx-target": "#node-result"},
            ),
            Div(id="node-result", style="margin-top: 16px;"),
            status="INTERACTIVE",
            footer_left="ROUTING POTENTIAL SCORER",
            footer_right="ENTER PUBLIC KEY",
        ),
        render_model_card("lightning-lgbm"),
        PipelineDag("lightning-lgbm"),
    )
