"""Home page — Real-time scoring feed (Signal Extractor)."""

from fasthtml.common import *
from stream.app.components import (
    CounterBox, DataGrid, DataReadout, DiagnosticFrame, FeedHeader, FeedRow, Page, Tip,
)
from stream.services import get_risk_threshold


def home_page():
    return Page("Signal Feed", "/",
        # Intro
        DiagnosticFrame(
            "STREAM",

            P("Real-time Bitcoin transaction risk scoring. Live transactions are pulled from the mempool, "
              "scored by ML models for illicit activity signals, and logged to an audit trail. "
              "High-risk transactions are automatically flagged for compliance review.",
              style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;"),

            P("This page streams live Bitcoin mempool data, scores each transaction, "
              "and displays the results below. The feed updates automatically via SSE.",
              style="color: var(--fg-subtle); font-size: 11px;"),

            status="OVERVIEW",
            footer_left="BITCOIN ML COMPLIANCE",
            footer_right="LIVE_DATA",
        ),

        DiagnosticFrame(
            "MEMPOOL STATUS",

            # Oscilloscope canvas
            Canvas(id="oscilloscope", cls="oscilloscope"),

            # Data readouts (live via HTMX)
            Div(
                id="home-stats",
                **{"hx-get": "/api/v1/home/stats", "hx-trigger": "load, every 15s", "hx-swap": "innerHTML"},
            ),

            # Threshold slider
            Div(
                Span("RISK THRESHOLD", style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px; text-transform: uppercase;"),
                Div(
                    Span(f"{get_risk_threshold():.2f}", id="threshold-value"),
                    Input(type="range", name="threshold", min="0.05", max="0.9", step="0.05",
                          value=str(get_risk_threshold()), id="threshold-slider",
                          oninput="document.getElementById('threshold-value').textContent=parseFloat(this.value).toFixed(2)",
                          **{"hx-post": "/api/v1/settings/threshold", "hx-target": "#threshold-value",
                             "hx-trigger": "change", "hx-swap": "outerHTML", "hx-include": "this"}),
                    id="threshold-control",
                ),
                P("Lower the threshold to trigger more alerts for demo purposes.",
                  style="color: var(--fg-subtle); font-size: 10px; margin-top: 4px;"),
                style="margin-top: 16px; padding: 12px; border: 1px solid var(--highlight-med); border-radius: 4px; max-width: 360px;",
            ),

            status="LIVE",
            footer_left="REAL-TIME SCORING",
            footer_right="MEMPOOL ACTIVE",
        ),

        # Scoring feed + alert queue + alert detail — 3 columns on desktop
        Div(
            DiagnosticFrame(
                "SCORING FEED",

                P("Live mempool transactions scored in real-time.",
                  style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;"),

                FeedHeader(),
                Div(
                    id="score-feed",
                    cls="scroll-feed",
                    **{
                        "hx-ext": "sse",
                        "sse-connect": "/api/v1/stream/scores",
                        "sse-swap": "transaction_scored",
                        "hx-swap": "afterbegin",
                    },
                    style="min-height: 300px;",
                ),

                Div(
                    id="prediction-counter",
                    **{"hx-get": "/api/v1/home/prediction-count", "hx-trigger": "load, every 10s", "hx-swap": "innerHTML"},
                    style="text-align: center; margin-top: 20px;",
                ),

                status="STREAMING",
                footer_left="AUDIT TRAIL",
                footer_right="SSE CONNECTED",
            ),

            DiagnosticFrame(
                "ALERT QUEUE",

                P("Flagged transactions for compliance review.",
                  style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;"),

                Div(
                    id="alert-stats",
                    **{"hx-get": "/api/v1/alerts/stats", "hx-trigger": "load, every 10s", "hx-swap": "innerHTML"},
                ),

                Table(
                    Thead(Tr(
                        Th("Time"), Th("TX ID"), Th("Score"), Th("Status"), Th(""),
                    )),
                    Tbody(
                        id="alert-feed",
                        **{"hx-get": "/api/v1/alerts/recent", "hx-trigger": "load, every 10s", "hx-swap": "innerHTML"},
                    ),
                    cls="spark-table",
                ),

                status="MONITORING",
                footer_left="COMPLIANCE REVIEW",
                footer_right="SHAP + LLM",
            ),

            DiagnosticFrame(
                "ALERT DETAIL",

                Div(
                    P("Select an alert to view ",
                      Tip("SHAP"), " explanation and compliance narrative.",
                      style="color: var(--fg-subtle);"),
                    id="alert-detail",
                ),

                status="SELECT_ALERT",
                footer_left="SHAP + LLM EXPLANATION",
                footer_right="AUDIT TRAIL",
            ),

            cls="triple-row",
        ),

        # Oscilloscope JS
        Script("""
        const canvas = document.getElementById('oscilloscope');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            let width, height, time = 0;

            function resize() {
                width = canvas.parentElement.clientWidth;
                height = 200;
                canvas.width = width;
                canvas.height = height;
            }
            window.addEventListener('resize', resize);
            resize();

            const inputs = [
                {speed: 0.02, amp: 30, offset: 0},
                {speed: 0.03, amp: 20, offset: 100},
                {speed: 0.015, amp: 40, offset: 200},
            ];

            function draw() {
                ctx.clearRect(0, 0, width, height);
                const cy = height / 2;
                const cx = width / 2;
                time += 1;

                inputs.forEach((inp, i) => {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(255,255,255,${0.2 + Math.sin(time*0.05+i)*0.15})`;
                    ctx.lineWidth = 1;
                    for (let x = 0; x < cx; x += 4) {
                        const p = x / cx;
                        const m = 1 - Math.pow(p, 3);
                        const wave = Math.sin(x * inp.speed + time * inp.speed + inp.offset) * inp.amp;
                        const noise = (Math.random() - 0.5) * 20 * m;
                        const y = cy + (wave + noise) * m + (i-1) * 30 * m;
                        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                    }
                    ctx.stroke();
                });

                ctx.beginPath();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#9ccfd8';
                ctx.moveTo(cx, cy);
                ctx.lineTo(width, cy);
                ctx.stroke();

                requestAnimationFrame(draw);
            }
            draw();
        }
        """),
    )
