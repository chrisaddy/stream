"""Reusable Spark-themed FastHTML components."""

import time as _time

from fasthtml.common import *

_CSS_VERSION = int(_time.time())


def NoiseOverlay():
    return Div(cls="noise-overlay")


def CrosshairCorners():
    return (
        Div(cls="crosshair ch-tl"),
        Div(cls="crosshair ch-tr"),
        Div(cls="crosshair ch-bl"),
        Div(cls="crosshair ch-br"),
    )


def ScanLine():
    return Div(cls="scan-line")


def DiagnosticFrame(title: str, *children, status: str = "ACTIVE", footer_left: str = "", footer_right: str = ""):
    return Div(
        Div(
            Span(title),
            Span(f"[ {status} ]", cls="status"),
            cls="frame-header",
        ),
        Div(ScanLine(), *children, cls="frame-body"),
        Div(
            Span(footer_left),
            Span(footer_right),
            cls="frame-footer",
        ),
        cls="diagnostic-frame",
    )


def DataReadout(label: str, value: str, highlight: bool = False, variant: str = ""):
    cls = "data-value"
    if highlight:
        cls += " highlight"
    if variant == "danger":
        cls += " danger"
    elif variant == "warning":
        cls += " warning"
    return Div(
        Div(label, cls="data-label"),
        Div(value, cls=cls),
        cls="data-readout",
    )


def DataGrid(*readouts):
    return Div(*readouts, cls="data-grid")


def StatusBadge(text: str, variant: str = ""):
    cls = "badge-sm"
    if variant == "green":
        cls += " badge-green"
    elif variant == "red":
        cls += " badge-red"
    elif variant == "yellow":
        cls += " badge-yellow"
    return Span(text, cls=cls)


def CounterBox(label: str, value: str):
    return Div(
        Span(label, cls="counter-label"),
        Span(value, cls="counter-value"),
        cls="counter-box",
    )


def FeedRow(timestamp, tx_id, size, fee_rate, risk_score, risk_label, threshold=0.7, model_name=""):
    risk_cls = "risk-low"
    if risk_score > threshold:
        risk_cls = "risk-high"
    elif risk_score > threshold * 0.6:
        risk_cls = "risk-medium"

    row_cls = "feed-row"
    if risk_score > threshold:
        row_cls += " high-risk"

    badge = ""
    if model_name:
        is_ml = "heuristic" not in model_name.lower()
        badge_cls = "model-badge model-badge-ml" if is_ml else "model-badge"
        badge_text = "ML" if is_ml else "HEURISTIC"
        badge = Span(badge_text, cls=badge_cls)

    return Div(
        Div(timestamp, style="color: var(--fg-dim);"),
        Div(tx_id, style="font-weight: bold;"),
        Div(f"{size} vB"),
        Div(f"{fee_rate:.1f} sat/vB"),
        Div(Span(f"{risk_score:.2f}"), Span(" "), badge, cls=risk_cls) if badge else Div(f"{risk_score:.2f}", cls=risk_cls),
        Div(risk_label),
        cls=row_cls,
    )


def DemoBanner(*children):
    """Wrap content in a visible demo-data indicator."""
    return Div(*children, cls="demo-banner")


def FeedHeader():
    return Div(
        Div("TIME"),
        Div("TX_ID"),
        Div("SIZE"),
        Div("FEE_RATE"),
        Div("RISK"),
        Div("LABEL"),
        Div("ANOMALY"),
        cls="feed-row",
        style="color: var(--fg-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid var(--fg-dim);",
    )


def MetricItem(label: str, value: str):
    return Div(
        Div(label, cls="metric-label"),
        Div(value, cls="metric-value"),
        cls="metric-item",
    )


def MetricsRow(*items):
    return Div(*items, cls="metrics-row")


def WalkthroughSection(title: str, content: str, code: str = None):
    children = [
        H3(title),
        P(content),
    ]
    if code:
        children.append(Pre(code, cls="walkthrough-code"))
    return Div(*children, cls="walkthrough-section")


TIPS = {
    # --- ML metric terms ---
    "PR-AUC": "Area under the Precision-Recall curve. Measures ranking quality for imbalanced classes — higher means better separation of rare positives.",
    "Precision": "Of all transactions flagged as illicit, what fraction actually were. High precision = fewer false alarms.",
    "Recall": "Of all truly illicit transactions, what fraction were caught. High recall = fewer missed threats.",
    "F1": "Harmonic mean of Precision and Recall. Balances the tradeoff between false alarms and missed detections.",
    "F1 (macro)": "Unweighted average of F1 across all classes. Treats each class equally regardless of size.",
    "Inference": "Time to score a single transaction. Lower is better for real-time systems.",
    "Accuracy": "Fraction of all predictions that were correct. Can be misleading with imbalanced classes.",
    "Log Loss": "Measures calibration of predicted probabilities. Lower means the model's confidence is better aligned with reality.",
    "MAE": "Mean Absolute Error — average prediction error in original units. Easy to interpret: 'off by X on average'.",
    "RMSE": "Root Mean Squared Error — penalizes large errors more than MAE. Sensitive to outliers.",
    "MAPE": "Mean Absolute Percentage Error — prediction error as a percentage. Scale-independent.",
    "Median AE": "Median Absolute Error — the middle prediction error. More robust to outliers than MAE.",
    "R²": "Coefficient of determination — fraction of variance explained by the model. 1.0 is perfect, 0 is baseline.",
    "Samples": "Number of data points used for evaluation.",
    "Threshold": "Classification cutoff — predictions above this score are flagged. Tuned for business cost.",
    "Total Cost": "Estimated business cost combining false negatives (missed illicit) and false positives (unnecessary reviews).",
    # --- Bitcoin terms ---
    "mempool": "Bitcoin's waiting room — unconfirmed transactions sit here until a miner includes them in a block.",
    "sat/vB": "Satoshis per virtual byte — the unit for Bitcoin transaction fees. Higher = faster confirmation.",
    "vsize": "Virtual size of a transaction in bytes. Determines how much block space it uses.",
    "BTC": "Bitcoin — the base currency unit. 1 BTC = 100,000,000 satoshis.",
    "sats": "Satoshis — the smallest unit of Bitcoin. 1 sat = 0.00000001 BTC.",
    "Lightning Network": "A layer-2 payment network built on Bitcoin. Enables instant, low-fee transactions through payment channels.",
    "channels": "Payment channels between Lightning nodes. Funds are locked in a channel to enable off-chain transactions.",
    "nodes": "Computers running Lightning Network software. They route payments and maintain channels.",
    "capacity": "Total Bitcoin locked in a Lightning channel or across the network, available for routing payments.",
    "routing": "Forwarding payments through intermediate nodes in the Lightning Network to reach the destination.",
    "public key": "A cryptographic identifier for a Lightning node. Like an address that other nodes use to find and connect to it.",
    # --- ML model terms ---
    "XGBoost": "Extreme Gradient Boosting — a fast, accurate ML algorithm for structured data. Industry standard for tabular prediction tasks.",
    "GCN": "Graph Convolutional Network — a neural network that operates on graph-structured data, capturing relationships between connected transactions.",
    "LightGBM": "Light Gradient Boosting Machine — similar to XGBoost but optimized for speed and memory efficiency.",
    "Logistic Regression": "A simple, interpretable ML model that estimates probabilities. Often used as a baseline to compare against more complex models.",
    "Calibrated": "A model whose predicted probabilities match real-world frequencies. If it says 80% risk, ~80% of those cases are actually risky.",
    # --- ML concept terms ---
    "SHAP": "SHapley Additive exPlanations — a method that shows how much each input feature contributed to a prediction. Makes ML decisions interpretable.",
    "KYC": "Know Your Customer — regulatory requirement for financial institutions to verify the identity of their clients.",
    "LLM": "Large Language Model — AI that generates human-readable text. Used here to convert technical SHAP values into compliance narratives.",
    "Platt Scaling": "A technique to calibrate model probabilities by fitting a logistic regression on top of raw model scores.",
    "Feature Importance": "Ranking of which input variables matter most for the model's predictions.",
}

# Backwards compat alias
METRIC_TIPS = TIPS


def Tip(text: str, tip: str = ""):
    """Wrap text in a tooltip span. Falls back to plain text if no tip provided."""
    desc = tip or TIPS.get(text, "")
    if not desc:
        return text
    return Span(text, cls="tip", **{"data-tip": desc})


_PIPELINE_DAGS = {
    "illicit-xgboost": [
        "LOAD_DATA", "TEMPORAL_SPLIT", "TRAIN_XGBOOST", "EVALUATE", "SAVE_TO_R2",
    ],
    "fee-lgbm": [
        "COLLECT_MEMPOOL", "FEATURIZE", "TRAIN_LGBM", "SAVE_TO_R2",
    ],
    "lightning-lgbm": [
        "FETCH_TOPOLOGY", "BUILD_GRAPH", "COMPUTE_FEATURES", "TRAIN_MODEL", "SAVE_TO_R2",
    ],
    "onboarding-xgb": [
        "GENERATE_DATA", "TRAIN_LOGISTIC", "TRAIN_XGBOOST", "EVALUATE", "SAVE_TO_R2",
    ],
}


def PipelineDag(model_name: str):
    """Render the training pipeline DAG + freshness for a specific model."""
    steps = _PIPELINE_DAGS.get(model_name, [])
    dag_nodes = []
    for i, step in enumerate(steps):
        dag_nodes.append(Span(step, cls="flow-node active"))
        if i < len(steps) - 1:
            dag_nodes.append(Span("->", cls="flow-arrow"))

    return DiagnosticFrame(
        "TRAINING PIPELINE",

        Div(*dag_nodes, style="overflow-x: auto; white-space: nowrap; margin-bottom: 16px;"),

        Div(
            id=f"freshness-{model_name}",
            **{"hx-get": f"/api/v1/pipeline/freshness/{model_name}",
               "hx-trigger": "load, every 60s", "hx-swap": "innerHTML"},
        ),

        status="PIPELINE",
        footer_left=f"[{model_name.upper()}] PREFECT",
        footer_right="ORCHESTRATION",
    )


def Card(title: str, *children):
    return Div(
        Div(title, cls="card-title"),
        *children,
        cls="card",
    )


def NavLink(text: str, href: str, active: bool = False):
    cls = "nav-link active" if active else "nav-link"
    return A(
        Span(cls="dot"),
        text,
        href=href,
        cls=cls,
    )


def NavSidebar(current_path: str = "/"):
    return Nav(
        Div(H1("STREAM"), cls="nav-brand"),
        Div(
            Div("DASHBOARD", cls="nav-section-title"),
            NavLink("Signal Feed", "/", active=current_path == "/"),
            NavLink("Illicit Detection", "/illicit", active=current_path == "/illicit"),
            NavLink("Fee Estimation", "/fees", active=current_path == "/fees"),
            NavLink("Lightning", "/lightning", active=current_path == "/lightning"),
            NavLink("Onboarding", "/onboarding", active=current_path == "/onboarding"),
            NavLink("System Status", "/pipeline", active=current_path == "/pipeline"),
            cls="nav-section",
        ),
        Div(
            Div("WALKTHROUGH", cls="nav-section-title"),
            NavLink("Overview", "/walkthrough", active=current_path == "/walkthrough"),
            NavLink("Illicit Deep Dive", "/walkthrough/illicit", active=current_path == "/walkthrough/illicit"),
            NavLink("Fee Deep Dive", "/walkthrough/fees", active=current_path == "/walkthrough/fees"),
            NavLink("Architecture", "/walkthrough/architecture", active=current_path == "/walkthrough/architecture"),
            cls="nav-section",
        ),
        cls="nav-sidebar",
    )


def Page(title: str, current_path: str, *children):
    return (
        Title(f"STREAM // {title}"),
        Link(rel="stylesheet", href=f"/style.css?v={_CSS_VERSION}"),
        Script(src="https://unpkg.com/htmx.org@2.0.4"),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"),
        NoiseOverlay(),
        Div(
            NavSidebar(current_path),
            Main(*children, cls="main-content"),
            cls="app-layout",
        ),
    )
