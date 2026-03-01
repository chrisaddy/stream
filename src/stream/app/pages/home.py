"""Home page — Real-time scoring feed (Signal Extractor)."""

from fasthtml.common import *
from stream.app.components import (
    CounterBox, DataGrid, DataReadout, DiagnosticFrame, FeedHeader, FeedRow, Page,
)


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

            status="LIVE",
            footer_left="REAL-TIME SCORING",
            footer_right="MEMPOOL ACTIVE",
        ),

        # Live scoring feed
        DiagnosticFrame(
            "TRANSACTION SCORING FEED",

            P("Live transactions from the Bitcoin mempool, scored in real-time. "
              "Rows flash red when risk exceeds the 0.7 threshold. "
              "Each prediction is logged to the audit trail.",
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
