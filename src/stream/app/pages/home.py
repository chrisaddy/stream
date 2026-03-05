"""Home page — Real-time scoring feed (Signal Extractor)."""

from fasthtml.common import *

from stream.app.components import (
    DiagnosticFrame,
    FeedHeader,
    Page,
    Tip,
)
from stream.services import get_risk_threshold


def home_page():
    return Page(
        "Signal Feed",
        "/",
        # Intro
        DiagnosticFrame(
            "STREAM",
            P(
                "Real-time Bitcoin transaction risk scoring. Live transactions are pulled from the mempool, "
                "scored by ML models for illicit activity signals, and logged to an audit trail. "
                "High-risk transactions are automatically flagged for compliance review.",
                style="color: var(--fg-white); opacity: 0.85; margin-bottom: 12px;",
            ),
            P(
                "This page streams live Bitcoin mempool data, scores each transaction, "
                "and displays the results below. The feed updates automatically via SSE.",
                style="color: var(--fg-subtle); font-size: 11px;",
            ),
            status="OVERVIEW",
            footer_left="BITCOIN ML COMPLIANCE",
            footer_right="LIVE_DATA",
        ),
        DiagnosticFrame(
            "MEMPOOL STATUS",
            # Oscilloscope canvas
            Canvas(id="oscilloscope", cls="oscilloscope"),
            P(
                "Live scoring uses a mempool-features heuristic. The trained XGBoost model is available at /illicit for 166-feature input.",
                style="color: var(--fg-dim); font-size: 10px; letter-spacing: 0.5px; margin-top: 8px; padding: 6px 10px; border: 1px dashed var(--highlight-med); background: rgba(57, 53, 82, 0.3);",
            ),
            # Data readouts (live via HTMX)
            Div(
                id="home-stats",
                **{
                    "hx-get": "/api/v1/home/stats",
                    "hx-trigger": "load, every 15s",
                    "hx-swap": "innerHTML",
                },
            ),
            # Threshold slider
            Div(
                Span(
                    "RISK THRESHOLD",
                    style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px; text-transform: uppercase;",
                ),
                Div(
                    Span(f"{get_risk_threshold():.2f}", id="threshold-value"),
                    Input(
                        type="range",
                        name="threshold",
                        min="0.05",
                        max="0.9",
                        step="0.05",
                        value=str(get_risk_threshold()),
                        id="threshold-slider",
                        oninput="document.getElementById('threshold-value').textContent=parseFloat(this.value).toFixed(2)",
                        **{
                            "hx-post": "/api/v1/settings/threshold",
                            "hx-target": "#threshold-value",
                            "hx-trigger": "change",
                            "hx-swap": "outerHTML",
                            "hx-include": "this",
                        },
                    ),
                    id="threshold-control",
                ),
                P(
                    "Lower the threshold to trigger more alerts for demo purposes.",
                    style="color: var(--fg-subtle); font-size: 10px; margin-top: 4px;",
                ),
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
                P(
                    "Live mempool transactions scored in real-time.",
                    style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;",
                ),
                FeedHeader(),
                Div(
                    id="score-feed",
                    cls="scroll-feed",
                    style="min-height: 300px;",
                ),
                Div(
                    id="prediction-counter",
                    **{
                        "hx-get": "/api/v1/home/prediction-count",
                        "hx-trigger": "load, every 10s",
                        "hx-swap": "innerHTML",
                    },
                    style="text-align: center; margin-top: 20px;",
                ),
                status="STREAMING",
                footer_left="AUDIT TRAIL",
                footer_right="SSE CONNECTED",
            ),
            DiagnosticFrame(
                "ALERT QUEUE",
                P(
                    "Flagged transactions for compliance review.",
                    style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;",
                ),
                Div(
                    id="alert-stats",
                    **{
                        "hx-get": "/api/v1/alerts/stats",
                        "hx-trigger": "load, every 10s",
                        "hx-swap": "innerHTML",
                    },
                ),
                Table(
                    Thead(
                        Tr(
                            Th("Time"),
                            Th("TX ID"),
                            Th("Score"),
                            Th("Status"),
                            Th(""),
                        )
                    ),
                    Tbody(
                        id="alert-feed",
                        **{
                            "hx-get": "/api/v1/alerts/recent",
                            "hx-trigger": "load, every 10s",
                            "hx-swap": "innerHTML",
                        },
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
                    P(
                        "Select an alert to view ",
                        Tip("SHAP"),
                        " explanation and compliance narrative.",
                        style="color: var(--fg-subtle);",
                    ),
                    id="alert-detail",
                ),
                status="SELECT_ALERT",
                footer_left="SHAP + LLM EXPLANATION",
                footer_right="AUDIT TRAIL",
            ),
            cls="triple-row",
        ),
        # Feedback loop + Drift monitor + Online metrics — 3 columns
        Div(
            DiagnosticFrame(
                "FEEDBACK LOOP",
                P(
                    "Review alerts as TP/FP above, then retrain the scoring model on your labels.",
                    style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;",
                ),
                Div(
                    id="feedback-panel",
                    **{
                        "hx-get": "/api/v1/feedback/stats",
                        "hx-trigger": "load, every 15s",
                        "hx-swap": "innerHTML",
                    },
                ),
                status="LEARNING",
                footer_left="HUMAN-IN-THE-LOOP",
                footer_right="RETRAIN_PIPELINE",
            ),
            DiagnosticFrame(
                "DRIFT MONITOR",
                P(
                    "Score distribution stability — detects when the mempool shifts away from baseline.",
                    style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;",
                ),
                Div(
                    id="drift-panel",
                    **{
                        "hx-get": "/api/v1/drift/status",
                        "hx-trigger": "load, every 10s",
                        "hx-swap": "innerHTML",
                    },
                ),
                status="WATCHING",
                footer_left="PSI + ADWIN / SELF_HEALING",
                footer_right="DISTRIBUTION_SHIFT",
            ),
            DiagnosticFrame(
                "ONLINE METRICS",
                P(
                    "Live precision, recall, F1, ROCAUC — cumulative and rolling window.",
                    style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;",
                ),
                Div(
                    id="online-metrics-panel",
                    **{
                        "hx-get": "/api/v1/online/metrics",
                        "hx-trigger": "load, every 10s",
                        "hx-swap": "innerHTML",
                    },
                ),
                status="TRACKING",
                footer_left="ONLINE ML",
                footer_right="CUMULATIVE + ROLLING",
            ),
            cls="triple-row",
        ),
        # Anomaly detector panel
        DiagnosticFrame(
            "ANOMALY DETECTOR",
            P(
                "Unsupervised anomaly detection using Half-Space Trees. Learns transaction patterns and flags outliers.",
                style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;",
            ),
            Div(
                id="anomaly-panel",
                **{
                    "hx-get": "/api/v1/anomaly/status",
                    "hx-trigger": "load, every 15s",
                    "hx-swap": "innerHTML",
                },
            ),
            status="SCANNING",
            footer_left="HALF-SPACE TREES",
            footer_right="UNSUPERVISED",
        ),
        # Model Race — full width
        DiagnosticFrame(
            "MODEL RACE",
            P(
                "3 online classifiers competing in real-time: LogisticRegression vs HoeffdingTree vs GaussianNB.",
                style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;",
            ),
            Div(
                Div(
                    H4(
                        "STANDINGS",
                        style="color: var(--fg-dim); font-size: 10px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                    ),
                    Div(
                        id="race-standings",
                        **{
                            "hx-get": "/api/v1/online/race/standings",
                            "hx-trigger": "load, every 5s",
                            "hx-swap": "innerHTML",
                        },
                    ),
                    style="flex: 1;",
                ),
                Div(
                    H4(
                        "CONVERGENCE",
                        style="color: var(--fg-dim); font-size: 10px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                    ),
                    Div(
                        id="race-convergence",
                        **{
                            "hx-get": "/api/v1/online/race/convergence",
                            "hx-trigger": "load, every 10s",
                            "hx-swap": "innerHTML",
                        },
                    ),
                    style="flex: 2;",
                ),
                cls="race-layout",
            ),
            status="RACING",
            footer_left="ONLINE ML",
            footer_right="ADAPTIVE_ENSEMBLE",
        ),
        # Oscilloscope + SSE feed JS
        Script("""
        const canvas = document.getElementById('oscilloscope');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            let width, height, time = 0;
            const scoreHistory = [];
            const anomalyHistory = [];
            const MAX_POINTS = 200;
            const MAX_FEED_ROWS = 50;

            function resize() {
                width = canvas.parentElement.clientWidth;
                height = 200;
                canvas.width = width;
                canvas.height = height;
            }
            window.addEventListener('resize', resize);
            resize();

            // Native EventSource connection (replaces HTMX SSE)
            const feed = document.getElementById('score-feed');
            const es = new EventSource('/api/v1/stream/scores');
            es.addEventListener('transaction_scored', function(e) {
                try {
                    // Insert into feed
                    if (feed) {
                        feed.insertAdjacentHTML('afterbegin', e.data);
                        while (feed.children.length > MAX_FEED_ROWS) {
                            feed.removeChild(feed.lastChild);
                        }
                    }
                    // Parse scores for oscilloscope
                    const el = document.createElement('div');
                    el.innerHTML = e.data;
                    const riskEl = el.querySelector('.risk-low, .risk-medium, .risk-high');
                    if (riskEl) {
                        const score = parseFloat(riskEl.textContent);
                        if (!isNaN(score)) {
                            scoreHistory.push(score);
                            if (scoreHistory.length > MAX_POINTS) scoreHistory.shift();
                        }
                    }
                    const anomalyEl = el.querySelector('.anomaly-low, .anomaly-med, .anomaly-high');
                    if (anomalyEl) {
                        const aScore = parseFloat(anomalyEl.textContent);
                        if (!isNaN(aScore)) {
                            anomalyHistory.push(aScore);
                            if (anomalyHistory.length > MAX_POINTS) anomalyHistory.shift();
                        }
                    }
                } catch(err) {}
            });

            function draw() {
                ctx.clearRect(0, 0, width, height);
                const cy = height / 2;
                time += 1;

                // Background wave — ambient signal
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(156, 207, 216, 0.15)';
                ctx.lineWidth = 1;
                for (let x = 0; x < width; x += 3) {
                    const wave = Math.sin(x * 0.02 + time * 0.02) * 20
                               + Math.sin(x * 0.035 + time * 0.015) * 15;
                    const noise = (Math.random() - 0.5) * 6;
                    const y = cy + wave + noise;
                    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();

                // Score trace — bright line showing recent risk scores
                if (scoreHistory.length > 1) {
                    ctx.beginPath();
                    ctx.strokeStyle = '#9ccfd8';
                    ctx.lineWidth = 2;
                    ctx.shadowColor = 'rgba(156, 207, 216, 0.6)';
                    ctx.shadowBlur = 8;
                    const step = width / MAX_POINTS;
                    const startX = width - scoreHistory.length * step;
                    scoreHistory.forEach((s, i) => {
                        const x = startX + i * step;
                        const y = height - (s * height * 0.9) - height * 0.05;
                        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    // Score dots for high-risk
                    scoreHistory.forEach((s, i) => {
                        if (s > 0.3) {
                            const x = startX + i * step;
                            const y = height - (s * height * 0.9) - height * 0.05;
                            ctx.beginPath();
                            ctx.fillStyle = s > 0.5 ? '#eb6f92' : '#f6c177';
                            ctx.arc(x, y, 3, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    });
                    // Anomaly trace — purple dashed line
                    if (anomalyHistory.length > 1) {
                        ctx.beginPath();
                        ctx.strokeStyle = 'rgba(196, 167, 231, 0.6)';
                        ctx.lineWidth = 1.5;
                        ctx.setLineDash([6, 4]);
                        const aStartX = width - anomalyHistory.length * step;
                        anomalyHistory.forEach((a, i) => {
                            const x = aStartX + i * step;
                            const y = height - (a * height * 0.9) - height * 0.05;
                            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                        });
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                } else {
                    // Waiting for data — show flat baseline with pulse
                    ctx.beginPath();
                    ctx.strokeStyle = 'rgba(156, 207, 216, 0.3)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([4, 8]);
                    ctx.moveTo(0, cy);
                    ctx.lineTo(width, cy);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                requestAnimationFrame(draw);
            }
            draw();
        }
        """),
    )
