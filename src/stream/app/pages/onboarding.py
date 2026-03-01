"""Onboarding Risk Scoring demo page."""

from fasthtml.common import *
from stream.app.components import (
    Card, DataGrid, DataReadout, DiagnosticFrame, Page, PipelineDag,
)
from stream.app.pages.models_page import render_model_card


def onboarding_page():
    return Page("Onboarding Risk", "/onboarding",
        DiagnosticFrame(
            "ONBOARDING RISK",

            P("When new customers sign up at a regulated exchange, they go through KYC (Know Your Customer) verification. "
              "This model scores onboarding risk based on signals like email type, document verification, "
              "and initial behavior patterns.",
              style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;"),

            status="OVERVIEW",
            footer_left="KYC RISK SCORING",
            footer_right="SYNTHETIC_DATA",
        ),

        DiagnosticFrame(
            "ONBOARDING RISK SCORING",

            P("Interactive demo: fill out a mock signup form and get a risk score with explanation.",
              style="color: var(--fg-subtle); margin-bottom: 16px;"),

            Form(
                Div(
                    Div(
                        Label("Email Domain Type", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Select(
                            Option("Corporate", value="corporate"),
                            Option("Free (Gmail, etc)", value="free"),
                            Option("Disposable", value="disposable"),
                            name="email_domain_type", cls="spark-input",
                        ),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("Phone Verified", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Select(
                            Option("Yes", value="1"),
                            Option("No", value="0"),
                            name="phone_verified", cls="spark-input",
                        ),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("Document Verification Score", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Input(type="range", min="0", max="1", step="0.05", value="0.85",
                              name="doc_score", cls="spark-slider"),
                        Span(id="doc-score-display", style="color: var(--fg-green);"),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("IP Matches Document Country", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Select(
                            Option("Yes", value="1"),
                            Option("No", value="0"),
                            name="ip_match", cls="spark-input",
                        ),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("Transactions in First 24h", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Input(type="number", name="tx_velocity", value="2", min="0", max="50",
                              cls="spark-input"),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("Initial Deposit (USD)", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Input(type="number", name="initial_deposit", value="500", min="0",
                              cls="spark-input"),
                        style="margin-bottom: 12px;",
                    ),
                    cls="form-grid-2col",
                ),
                Button("Assess Risk", cls="spark-btn", type="submit", style="margin-top: 12px;"),
                **{"hx-post": "/api/v1/onboarding/score", "hx-target": "#onboarding-result"},
            ),

            Div(id="onboarding-result", style="margin-top: 20px;"),

            status="DEMO",
            footer_left="KYC RISK TRIAGE",
            footer_right="SYNTHETIC DATA",
        ),

        # Model comparison
        DiagnosticFrame(
            "MODEL COMPARISON",

            Div(
                id="onboarding-metrics",
                **{"hx-get": "/api/v1/onboarding/metrics", "hx-trigger": "load", "hx-swap": "innerHTML"},
            ),

            status="LR vs XGB",
            footer_left="MODEL CALIBRATION",
            footer_right="PLATT SCALING",
        ),

        render_model_card("onboarding-xgb"),

        PipelineDag("onboarding-xgb"),
    )
