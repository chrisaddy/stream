"""Illicit Detection dashboard page."""

from fasthtml.common import *
from stream.app.components import (
    Card, DataGrid, DataReadout, DiagnosticFrame, MetricItem, MetricsRow, Page, WalkthroughSection,
)


def illicit_page():
    return Page("Illicit Detection", "/illicit",
        DiagnosticFrame(
            "ILLICIT TRANSACTION DETECTION",

            # Model comparison
            H3("Model Performance Comparison", style="color: var(--fg-green); font-size: 13px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;"),

            Div(
                Table(
                    Thead(Tr(
                        Th("Metric"), Th("XGBoost"), Th("GCN"), Th("Winner"),
                    )),
                    Tbody(
                        Tr(Td("PR-AUC"), Td(id="xgb-prauc"), Td(id="gcn-prauc"), Td(id="prauc-winner")),
                        Tr(Td("Precision"), Td(id="xgb-prec"), Td(id="gcn-prec"), Td(id="prec-winner")),
                        Tr(Td("Recall"), Td(id="xgb-rec"), Td(id="gcn-rec"), Td(id="rec-winner")),
                        Tr(Td("F1"), Td(id="xgb-f1"), Td(id="gcn-f1"), Td(id="f1-winner")),
                        Tr(Td("Inference (ms)"), Td("~2ms"), Td("~50ms"), Td("XGBoost")),
                    ),
                    cls="spark-table",
                ),

                Div(id="pr-curve-chart", style="height: 300px; margin-top: 20px;"),

                **{"hx-get": "/api/v1/illicit/metrics", "hx-trigger": "load", "hx-swap": "innerHTML"},
            ),

            status="MODEL_LOADED",
            footer_left="[ILL-001] BINARY CLASSIFICATION",
            footer_right="ELLIPTIC_DATASET",
        ),

        # Interactive scoring
        DiagnosticFrame(
            "INTERACTIVE SCORING",

            P("Submit transaction features for real-time risk scoring with SHAP explanation.",
              style="color: var(--fg-dim); margin-bottom: 16px;"),

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
            footer_left="[ILL-002] SINGLE PREDICTION",
            footer_right="SHAP_ENABLED",
        ),

        # Threshold slider
        DiagnosticFrame(
            "THRESHOLD ANALYSIS",

            P("Adjust the classification threshold to see the tradeoff between false positives (manual reviews) and false negatives (missed illicit transactions).",
              style="color: var(--fg-dim); margin-bottom: 16px;"),

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
            footer_left="[ILL-003] COST-SENSITIVE",
            footer_right="FN >> FP",
        ),
    )
