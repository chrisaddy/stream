"""Model card schema, R2 persistence, and Plotly chart factories."""

import json
from datetime import datetime, timezone
from io import BytesIO

import plotly.graph_objects as go
import structlog

log = structlog.get_logger()

# Rosé Pine Moon — Plotly layout
_LAYOUT = dict(
    paper_bgcolor="#232136",
    plot_bgcolor="#232136",
    font=dict(family="monospace", color="#e0def4", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#393552", zerolinecolor="#393552"),
    yaxis=dict(gridcolor="#393552", zerolinecolor="#393552"),
)

GREEN = "#9ccfd8"   # Foam
RED = "#eb6f92"     # Love
AMBER = "#f6c177"   # Gold
DIM = "#6e6a86"     # Muted


def build_model_card(
    name: str,
    description: str,
    *,
    metrics: dict | None = None,
    data_summary: dict | None = None,
    training_params: dict | None = None,
    feature_importance: list[dict] | None = None,
    curves: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a model card dict."""
    card = {
        "name": name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
        "metrics": metrics or {},
        "data_summary": data_summary or {},
        "training_params": training_params or {},
        "feature_importance": feature_importance or [],
        "curves": curves or {},
    }
    if extra:
        card.update(extra)
    return card


def save_model_card(card: dict, name: str):
    """Save model card JSON to R2 (with local fallback)."""
    card_bytes = json.dumps(card, indent=2, default=str).encode()
    key = f"models/{name}/card.json"
    try:
        from prefect_aws.s3 import S3Bucket
        s3 = S3Bucket.load("model-store")
        s3.upload_from_file_object(BytesIO(card_bytes), key)
        log.info("Model card saved to R2", name=name)
    except Exception as e:
        log.warning("R2 card upload failed, saving locally", error=str(e))
        import os
        os.makedirs(f"models/{name}", exist_ok=True)
        with open(f"models/{name}/card.json", "w") as f:
            f.write(card_bytes.decode())


def load_model_card_from_r2(name: str) -> dict | None:
    """Load model card JSON from R2 (sync, for app startup)."""
    try:
        import boto3
        from stream.config import settings
        if not settings.R2_ENDPOINT_URL:
            return _load_local_card(name)
        s3 = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        )
        resp = s3.get_object(Bucket=settings.R2_BUCKET_NAME, Key=f"models/{name}/card.json")
        return json.loads(resp["Body"].read())
    except Exception as e:
        log.debug("R2 card load failed", name=name, error=str(e))
        return _load_local_card(name)


def _load_local_card(name: str) -> dict | None:
    import os
    path = f"models/{name}/card.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# === Plotly Chart Factories ===

def pr_curve_chart(precisions: list, recalls: list) -> go.Figure:
    """Precision-Recall curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recalls, y=precisions,
        mode="lines", line=dict(color=GREEN, width=2),
        name="PR Curve",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 0],
        mode="lines", line=dict(color=DIM, width=1, dash="dash"),
        name="Random",
    ))
    fig.update_layout(**_LAYOUT, title="Precision-Recall Curve",
                      xaxis_title="Recall", yaxis_title="Precision")
    return fig


def feature_importance_chart(names: list[str], scores: list[float], top_k: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)[:top_k]
    pairs.reverse()  # lowest at top for horizontal bar
    fig = go.Figure(go.Bar(
        x=[p[1] for p in pairs],
        y=[p[0] for p in pairs],
        orientation="h",
        marker_color=GREEN,
    ))
    fig.update_layout(**_LAYOUT, title="Feature Importance",
                      xaxis_title="Importance", yaxis_title="")
    return fig


def calibration_chart(prob_true: list, prob_pred: list, class_label: str = "") -> go.Figure:
    """Calibration curve (reliability diagram)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode="lines+markers", line=dict(color=GREEN, width=2),
        marker=dict(size=6, color=GREEN),
        name=f"Calibration {class_label}".strip(),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(color=DIM, width=1, dash="dash"),
        name="Perfect",
    ))
    fig.update_layout(**_LAYOUT, title=f"Calibration Curve {class_label}".strip(),
                      xaxis_title="Mean Predicted Probability",
                      yaxis_title="Fraction of Positives")
    return fig


def confusion_matrix_chart(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    """Confusion matrix heatmap."""
    z = [[tn, fp], [fn, tp]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{z[i][j]:,}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted Neg", "Predicted Pos"], y=["Actual Neg", "Actual Pos"],
        text=text, texttemplate="%{text}", textfont=dict(size=14, color="#232136"),
        colorscale=[[0, "#393552"], [1, GREEN]], showscale=False,
    ))
    fig.update_layout(**_LAYOUT, title="Confusion Matrix")
    return fig


def temporal_chart(timesteps: list[dict]) -> go.Figure:
    """Per-timestep metrics (PR-AUC over time)."""
    ts = [d["timestep"] for d in timesteps]
    pr_aucs = [d.get("pr_auc", 0) for d in timesteps]
    recalls = [d.get("recall", 0) for d in timesteps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=pr_aucs, mode="lines+markers",
        line=dict(color=GREEN, width=2), marker=dict(size=4),
        name="PR-AUC",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=recalls, mode="lines+markers",
        line=dict(color=AMBER, width=2), marker=dict(size=4),
        name="Recall",
    ))
    fig.update_layout(**_LAYOUT, title="Temporal Stability",
                      xaxis_title="Timestep", yaxis_title="Score")
    return fig


def regime_chart(regimes: dict) -> go.Figure:
    """Bar chart of MAE by congestion regime."""
    names = list(regimes.keys())
    maes = [regimes[n].get("mae", 0) for n in names]
    counts = [regimes[n].get("n_samples", 0) for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=maes,
        text=[f"n={c}" for c in counts], textposition="auto",
        marker_color=[GREEN, AMBER, RED][:len(names)],
    ))
    fig.update_layout(**_LAYOUT, title="MAE by Congestion Regime",
                      xaxis_title="Regime", yaxis_title="MAE (sat/vB)")
    return fig


def fig_to_png(fig: go.Figure, width: int = 800, height: int = 500) -> bytes:
    """Render Plotly figure to PNG bytes (for Prefect image artifacts)."""
    return fig.to_image(format="png", width=width, height=height)
