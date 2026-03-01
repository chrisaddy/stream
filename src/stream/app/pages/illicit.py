"""Illicit Detection dashboard page."""

from fasthtml.common import *
from stream.app.components import (
    Card, DataGrid, DataReadout, DiagnosticFrame, MetricItem, MetricsRow, Page, PipelineDag, Tip, WalkthroughSection,
)
from stream.app.pages.models_page import render_model_card


def illicit_page():
    return Page("Illicit Detection", "/illicit",
        DiagnosticFrame(
            "ILLICIT DETECTION",

            P("Regulated Bitcoin exchanges must screen every transaction for potential illicit activity "
              "(money laundering, sanctions evasion). This page compares two ML approaches: "
              "XGBoost (fast, explainable) vs GCN (graph-aware, catches network patterns).",
              style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;"),

            status="OVERVIEW",
            footer_left="TRANSACTION SCREENING",
            footer_right="ML_COMPARISON",
        ),

        DiagnosticFrame(
            "ILLICIT TRANSACTION DETECTION",

            # Model comparison
            H3("Model Performance Comparison", style="color: var(--fg-green); font-size: 13px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;"),

            Div(
                Table(
                    Thead(Tr(
                        Th("Metric"), Th(Tip("XGBoost")), Th(Tip("GCN")), Th("Winner"),
                    )),
                    Tbody(
                        Tr(Td(Tip("PR-AUC")), Td(id="xgb-prauc"), Td(id="gcn-prauc"), Td(id="prauc-winner")),
                        Tr(Td(Tip("Precision")), Td(id="xgb-prec"), Td(id="gcn-prec"), Td(id="prec-winner")),
                        Tr(Td(Tip("Recall")), Td(id="xgb-rec"), Td(id="gcn-rec"), Td(id="rec-winner")),
                        Tr(Td(Tip("F1")), Td(id="xgb-f1"), Td(id="gcn-f1"), Td(id="f1-winner")),
                        Tr(Td(Tip("Inference")), Td("~2ms"), Td("~50ms"), Td("XGBoost")),
                    ),
                    cls="spark-table",
                ),

                Div(id="pr-curve-chart", style="height: 300px; margin-top: 20px;"),

                **{"hx-get": "/api/v1/illicit/metrics", "hx-trigger": "load", "hx-swap": "innerHTML"},
            ),

            status="MODEL_LOADED",
            footer_left="ILLICIT DETECTION",
            footer_right="ELLIPTIC BITCOIN DATASET",
        ),

        # Interactive scoring
        DiagnosticFrame(
            "INTERACTIVE SCORING",

            P("Submit transaction features for real-time risk scoring with ",
              Tip("SHAP"), " explanation.",
              style="color: var(--fg-subtle); margin-bottom: 16px;"),

            Form(
                Div(
                    Label("Feature Vector (comma-separated, 166 values)", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px;"),
                    Textarea(
                        placeholder="Enter 166 comma-separated feature values, or click 'Random Sample' for a test transaction...",
                        name="features",
                        cls="spark-input",
                        style="height: 80px; resize: none;",
                        id="feature-input",
                    ),
                    style="margin-bottom: 12px;",
                ),
                Div(
                    Button("Score Transaction", cls="spark-btn", type="submit"),
                    Button("Random Sample", cls="spark-btn", type="button",
                           style="margin-left: 8px;",
                           **{"hx-get": "/api/v1/illicit/random-sample", "hx-target": "#feature-input", "hx-swap": "innerHTML"}),
                    style="display: flex; gap: 8px;",
                ),
                **{"hx-post": "/api/v1/illicit/score", "hx-target": "#score-result"},
            ),

            Div(id="score-result", style="margin-top: 20px;"),

            status="READY",
            footer_left="SINGLE TRANSACTION SCORER",
            footer_right="SHAP EXPLANATIONS",
        ),

        # Threshold slider
        DiagnosticFrame(
            "THRESHOLD ANALYSIS",

            P("Adjust the classification threshold to see the tradeoff between false positives (manual reviews) and false negatives (missed illicit transactions).",
              style="color: var(--fg-subtle); margin-bottom: 16px;"),

            Div(
                Label("Classification Threshold", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                Input(type="range", min="0.1", max="0.9", step="0.05", value="0.5",
                      cls="spark-slider", id="threshold-slider",
                      **{"hx-get": "/api/v1/illicit/threshold-analysis", "hx-trigger": "change", "hx-target": "#threshold-result", "hx-include": "[name='threshold']"}),
                Input(type="hidden", name="threshold", id="threshold-value", value="0.5"),
                Div(id="threshold-display", style="color: var(--fg-green); font-size: 16px; margin-top: 8px;"),
            ),
            Div(id="threshold-result", style="margin-top: 16px;"),

            Script("""
            const slider = document.getElementById('threshold-slider');
            const display = document.getElementById('threshold-display');
            const hidden = document.getElementById('threshold-value');
            if (slider) {
                slider.addEventListener('input', (e) => {
                    display.textContent = 'Threshold: ' + e.target.value;
                    hidden.value = e.target.value;
                });
                display.textContent = 'Threshold: 0.5';
            }
            """),

            status="INTERACTIVE",
            footer_left="COST-SENSITIVE THRESHOLD",
            footer_right="MISSED THREATS COST MORE",
        ),

        render_model_card("illicit-xgboost"),

        PipelineDag("illicit-xgboost"),
    )
