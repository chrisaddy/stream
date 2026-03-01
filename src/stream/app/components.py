"""Reusable Spark-themed FastHTML components."""

from fasthtml.common import *


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
            Span(title, Span(cls="badge-sm", style="margin-left: 8px;")),
            Span(f"[ {status} ]", cls="status"),
            cls="frame-header",
        ),
        Div(*CrosshairCorners(), ScanLine(), *children, cls="frame-body"),
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


def FeedRow(timestamp, tx_id, size, fee_rate, risk_score, risk_label):
    risk_cls = "risk-low"
    if risk_score > 0.7:
        risk_cls = "risk-high"
    elif risk_score > 0.4:
        risk_cls = "risk-medium"

    row_cls = "feed-row"
    if risk_score > 0.7:
        row_cls += " high-risk"

    return Div(
        Div(timestamp, style="color: var(--fg-dim);"),
        Div(tx_id, style="font-weight: bold;"),
        Div(f"{size} vB"),
        Div(f"{fee_rate:.1f} sat/vB"),
        Div(f"{risk_score:.2f}", cls=risk_cls),
        Div(risk_label),
        cls=row_cls,
    )


def FeedHeader():
    return Div(
        Div("TIME"),
        Div("TX_ID"),
        Div("SIZE"),
        Div("FEE_RATE"),
        Div("RISK"),
        Div("LABEL"),
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
        Div(
            H1("STREAM"),
            Div("Bitcoin ML Infrastructure", cls="badge"),
            cls="nav-brand",
        ),
        Div(
            Div("DASHBOARD", cls="nav-section-title"),
            NavLink("Signal Feed", "/", active=current_path == "/"),
            NavLink("Illicit Detection", "/illicit", active=current_path == "/illicit"),
            NavLink("Fee Estimation", "/fees", active=current_path == "/fees"),
            NavLink("Lightning", "/lightning", active=current_path == "/lightning"),
            NavLink("Onboarding", "/onboarding", active=current_path == "/onboarding"),
            NavLink("Alert Queue", "/alerts", active=current_path == "/alerts"),
            cls="nav-section",
        ),
        Div(
            Div("SYSTEM", cls="nav-section-title"),
            NavLink("Model Cards", "/models", active=current_path == "/models"),
            NavLink("Pipeline", "/pipeline", active=current_path == "/pipeline"),
            cls="nav-section",
        ),
        Div(
            Div("WALKTHROUGH", cls="nav-section-title"),
            NavLink("Overview", "/walkthrough", active=current_path == "/walkthrough"),
            NavLink("Illicit Deep Dive", "/walkthrough/illicit", active=current_path == "/walkthrough/illicit"),
            NavLink("Fee Deep Dive", "/walkthrough/fees", active=current_path == "/walkthrough/fees"),
            NavLink("Architecture", "/walkthrough/architecture", active=current_path == "/walkthrough/architecture"),
            NavLink("Integration", "/walkthrough/integration", active=current_path == "/walkthrough/integration"),
            NavLink("ML Roadmap", "/walkthrough/roadmap", active=current_path == "/walkthrough/roadmap"),
            cls="nav-section",
        ),
        cls="nav-sidebar",
    )


def Page(title: str, current_path: str, *children):
    return (
        Title(f"STREAM // {title}"),
        Link(rel="stylesheet", href="/style.css"),
        Script(src="https://unpkg.com/htmx.org@2.0.4"),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"),
        NoiseOverlay(),
        Div(
            NavSidebar(current_path),
            Main(*children, cls="main-content"),
            cls="app-layout",
        ),
    )
