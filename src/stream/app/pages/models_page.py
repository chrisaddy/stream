"""Model Cards page with dynamic data from R2 and Plotly charts."""

from fasthtml.common import *
from stream.app.components import (
    Card, DataGrid, DataReadout, DiagnosticFrame,
)
from stream.app.models import get_all_model_cards


# Fallback descriptions when no card exists in R2
_FALLBACK_CARDS = {
    "illicit-xgboost": {
        "name": "illicit-xgboost",
        "description": "XGBoost classifier for illicit Bitcoin transaction detection.",
    },
    "onboarding-xgb": {
        "name": "onboarding-xgb",
        "description": "Calibrated XGBoost for KYC onboarding risk scoring.",
    },
    "fee-lgbm": {
        "name": "fee-lgbm",
        "description": "LightGBM fee estimator for optimal sat/vB prediction.",
    },
    "lightning-lgbm": {
        "name": "lightning-lgbm",
        "description": "LightGBM regressor for Lightning node capacity prediction.",
    },
}


def _fmt(val, fmt=".4f"):
    """Format a numeric value, or return '—' if missing."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def _metrics_table(metrics: dict, model_name: str):
    """Render a metrics table based on model type."""
    if not metrics:
        return P("No metrics available. Run the training pipeline.", style="color: var(--fg-dim);")

    if model_name == "illicit-xgboost":
        cost = metrics.get("cost_analysis", {})
        return Table(
            Thead(Tr(Th("Metric"), Th("Value"))),
            Tbody(
                Tr(Td("PR-AUC"), Td(_fmt(metrics.get("pr_auc")), style="color: var(--fg-green);")),
                Tr(Td("Threshold"), Td(_fmt(cost.get("threshold"), ".3f"))),
                Tr(Td("Precision"), Td(_fmt(cost.get("precision"), ".3f"))),
                Tr(Td("Recall"), Td(_fmt(cost.get("recall"), ".3f"))),
                Tr(Td("F1"), Td(_fmt(cost.get("f1"), ".3f"))),
                Tr(Td("Total Cost"), Td(f"${cost.get('total_cost', 0):,.0f}")),
            ),
            cls="spark-table",
        )

    if model_name == "onboarding-xgb":
        lr = metrics.get("logistic_regression", {})
        xgb = metrics.get("calibrated_xgboost", {})
        return Table(
            Thead(Tr(Th("Metric"), Th("Logistic"), Th("XGBoost"))),
            Tbody(
                Tr(Td("Accuracy"), Td(_fmt(lr.get("accuracy"))), Td(_fmt(xgb.get("accuracy")), style="color: var(--fg-green);")),
                Tr(Td("F1 (macro)"), Td(_fmt(lr.get("f1_macro"))), Td(_fmt(xgb.get("f1_macro")), style="color: var(--fg-green);")),
                Tr(Td("Log Loss"), Td(_fmt(lr.get("log_loss"))), Td(_fmt(xgb.get("log_loss")), style="color: var(--fg-green);")),
            ),
            cls="spark-table",
        )

    if model_name == "fee-lgbm":
        return Table(
            Thead(Tr(Th("Metric"), Th("Value"))),
            Tbody(
                Tr(Td("MAE"), Td(f"{_fmt(metrics.get('mae'))} sat/vB", style="color: var(--fg-green);")),
                Tr(Td("RMSE"), Td(f"{_fmt(metrics.get('rmse'))} sat/vB")),
                Tr(Td("MAPE"), Td(_fmt(metrics.get("mape")))),
                Tr(Td("Median AE"), Td(f"{_fmt(metrics.get('median_ae'))} sat/vB")),
            ),
            cls="spark-table",
        )

    if model_name == "lightning-lgbm":
        return Table(
            Thead(Tr(Th("Metric"), Th("Value"))),
            Tbody(
                Tr(Td("MAE"), Td(_fmt(metrics.get("mae"), ".2f"), style="color: var(--fg-green);")),
                Tr(Td("R²"), Td(_fmt(metrics.get("r2")))),
                Tr(Td("Median AE"), Td(_fmt(metrics.get("median_ae"), ".2f"))),
                Tr(Td("Samples"), Td(str(metrics.get("n_samples", "—")))),
            ),
            cls="spark-table",
        )

    return P("Unknown model type", style="color: var(--fg-dim);")


def _data_summary(data: dict):
    """Render data summary section."""
    if not data:
        return None

    items = []
    for key, val in data.items():
        if key == "feature_names":
            continue  # Skip raw feature names
        if isinstance(val, dict):
            for k, v in val.items():
                items.append(DataReadout(f"{key}/{k}".upper(), str(v)))
        else:
            items.append(DataReadout(key.upper().replace("_", " "), str(val)))

    if not items:
        return None
    return Div(
        H4("Training Data", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;"),
        DataGrid(*items[:6]),
        style="margin-bottom: 12px;",
    )


def _chart_section(model_name: str, card: dict):
    """Render chart placeholders with HTMX lazy loading."""
    chart_types = {
        "illicit-xgboost": ["pr_curve", "feature_importance", "confusion_matrix", "temporal"],
        "onboarding-xgb": ["calibration", "feature_importance"],
        "fee-lgbm": ["feature_importance"],
        "lightning-lgbm": ["feature_importance"],
    }

    charts = chart_types.get(model_name, [])
    if not charts:
        return None

    divs = []
    for chart_type in charts:
        label = chart_type.replace("_", " ").title()
        divs.append(
            Div(
                H4(label, style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;"),
                Div(
                    id=f"chart-{model_name}-{chart_type}",
                    **{"hx-get": f"/api/v1/models/{model_name}/chart/{chart_type}", "hx-trigger": "load", "hx-swap": "innerHTML"},
                    style="min-height: 300px; display: flex; align-items: center; justify-content: center;",
                ),
                style="margin-bottom: 16px;",
            )
        )

    return Div(*divs)


def _model_card(model_name: str, card: dict):
    """Render a single model card from card data."""
    description = card.get("description", "")
    created = card.get("created_at", "")
    if created:
        created = created[:19].replace("T", " ")  # Trim to readable format

    metrics = card.get("metrics", {})
    data = card.get("data_summary", {})
    params = card.get("training_params", {})

    children = [
        P(description, style="margin-bottom: 12px; color: var(--fg-white); opacity: 0.85;"),
    ]

    if created:
        children.append(
            P(f"Last trained: {created} UTC", style="color: var(--fg-dim); font-size: 10px; margin-bottom: 12px;")
        )

    # Data summary
    ds = _data_summary(data)
    if ds:
        children.append(ds)

    # Training params
    if params:
        param_str = " | ".join(f"{k}: {v}" for k, v in list(params.items())[:4])
        children.append(
            Div(
                H4("Training Config", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;"),
                P(param_str, style="font-size: 11px; color: var(--fg-dim);"),
                style="margin-bottom: 12px;",
            )
        )

    # Metrics table
    children.append(
        Div(
            H4("Metrics", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;"),
            _metrics_table(metrics, model_name),
            style="margin-bottom: 12px;",
        )
    )

    # Charts
    charts = _chart_section(model_name, card)
    if charts:
        children.append(charts)

    return Card(card.get("name", model_name), *children)


def render_model_card(name: str):
    """Render a model card DiagnosticFrame for embedding in a model page."""
    cards = get_all_model_cards()
    card_data = cards.get(name, _FALLBACK_CARDS.get(name, {"name": name, "description": ""}))
    return DiagnosticFrame(
        "MODEL CARD",
        _model_card(name, card_data),
        status="LOADED" if name in cards else "FALLBACK",
        footer_left=f"[{name.upper()}] REGISTRY",
        footer_right="R2_BACKED",
    )
