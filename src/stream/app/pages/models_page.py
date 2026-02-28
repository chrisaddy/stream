"""Model Cards page with ROI calculator."""

from fasthtml.common import *
from stream.app.components import (
    Card, DiagnosticFrame, MetricItem, MetricsRow, Page,
)


def _model_card(name, description, architecture, data_desc, loss, assumptions, live_stats_endpoint):
    return Card(name,
        P(description, style="margin-bottom: 12px; color: var(--fg-white); opacity: 0.85;"),

        Div(
            H4("Architecture", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;"),
            P(architecture, style="font-size: 12px; margin-bottom: 12px;"),

            H4("Training Data", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;"),
            P(data_desc, style="font-size: 12px; margin-bottom: 12px;"),

            H4("Loss Function", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;"),
            P(loss, style="font-size: 12px; margin-bottom: 12px;"),

            H4("Assumptions & Limitations", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;"),
            P(assumptions, style="font-size: 12px; margin-bottom: 12px;"),
        ),

        # Live stats from Prefect
        Div(
            H4("Live Stats", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;"),
            Div(id=f"stats-{name.lower().replace(' ', '-')}",
                **{"hx-get": live_stats_endpoint, "hx-trigger": "load", "hx-swap": "innerHTML"},
                style="min-height: 40px;"),
        ),
    )


def models_page():
    return Page("Model Cards", "/models",
        DiagnosticFrame(
            "MODEL REGISTRY",

            Div(
                _model_card(
                    "Illicit XGBoost",
                    "Detects illicit Bitcoin transactions using 166 transaction-level features. Production workhorse for real-time scoring.",
                    "An ensemble of decision trees that each learn to correct the mistakes of the previous tree. Think of it as a committee of experts voting on whether a transaction looks suspicious.",
                    "Elliptic Bitcoin Dataset: ~200K transactions across 49 timesteps, 166 features (1 timestep + 93 local transaction features + 72 aggregated neighbor features). ~10:1 class imbalance (licit vs illicit).",
                    "Binary cross-entropy with class weighting: we penalize missing an illicit transaction 10x more than falsely flagging a legitimate one, because the regulatory cost of a miss far exceeds the customer friction of a false alarm.",
                    "The Elliptic dataset covers transactions from 2018-2019. The model assumes illicit patterns are relatively stable — temporal evaluation on held-out timesteps tests this assumption.",
                    "/api/v1/models/stats/illicit-xgboost",
                ),
                _model_card(
                    "Illicit GCN",
                    "Graph Convolutional Network that leverages Bitcoin's transaction graph structure. Sees payment flows between transactions, not just individual features.",
                    "A neural network that looks not just at a transaction, but at the transactions it's connected to. It learns patterns in the flow of Bitcoin between addresses — 3 GCN layers (165->128->128->2).",
                    "Same Elliptic dataset, but uses the edge index (transaction graph) in addition to node features.",
                    "Class-weighted cross-entropy with temporal train/validation masks.",
                    "Graph convolutions aggregate over 1-hop neighbors per layer (3 layers = 3-hop receptive field). Very large or disconnected subgraphs may not propagate information effectively.",
                    "/api/v1/models/stats/illicit-gcn",
                ),
                _model_card(
                    "Fee LightGBM",
                    "Predicts optimal fee rate (sat/vB) for different confirmation targets using live mempool state.",
                    "Similar to XGBoost but optimized for speed. Uses histogram-based splitting to handle the high-dimensional mempool data.",
                    "Live mempool snapshots from mempool.space API: mempool size, fee distributions, projected blocks, time features.",
                    "MAE (L1) objective — robust to fee spikes that would distort MSE-based models.",
                    "Fee estimation in a non-stationary environment. Requires frequent retraining as mempool dynamics evolve. Model is blind to external events (halvings, protocol changes).",
                    "/api/v1/models/stats/fee-lgbm",
                ),
                _model_card(
                    "Onboarding XGBoost",
                    "Calibrated risk scoring for new customer onboarding. Platt-scaled probabilities for meaningful risk tiers.",
                    "XGBoost with Platt scaling (sigmoid calibration) so that predicted probabilities match actual risk rates. A 0.8 score means ~80% chance of being high-risk.",
                    "Synthetic KYC dataset: 10K records with realistic distributions of email type, phone verification, document scores, IP matching, transaction velocity.",
                    "Multi-class cross-entropy with balanced class weights + post-hoc Platt scaling.",
                    "Trained on synthetic data — real-world distributions would differ. Demonstrates the pattern, not production-ready without real data.",
                    "/api/v1/models/stats/onboarding-xgb",
                ),
                cls="card-grid",
            ),

            status="4 MODELS",
            footer_left="[MOD-001] DOCUMENTATION",
            footer_right="R2_BACKED",
        ),

        # ROI Calculator
        DiagnosticFrame(
            "ROI CALCULATOR",

            P("Adjust parameters to see the operational cost savings from ML-powered compliance.",
              style="color: var(--fg-dim); margin-bottom: 16px;"),

            Form(
                Div(
                    Div(
                        Label("Daily Transaction Volume", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Input(type="number", name="daily_volume", value="50000", cls="spark-input"),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("Current Manual Review Rate (%)", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Input(type="number", name="review_rate", value="5", cls="spark-input"),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("Analyst Cost ($/hour)", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Input(type="number", name="analyst_cost", value="45", cls="spark-input"),
                        style="margin-bottom: 12px;",
                    ),
                    Div(
                        Label("Minutes per Review", style="font-size: 10px; color: var(--fg-dim); text-transform: uppercase;"),
                        Input(type="number", name="review_minutes", value="5", cls="spark-input"),
                        style="margin-bottom: 12px;",
                    ),
                    style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;",
                ),
                Button("Calculate ROI", cls="spark-btn", type="submit"),
                **{"hx-post": "/api/v1/models/roi", "hx-target": "#roi-result"},
            ),

            Div(id="roi-result", style="margin-top: 20px;"),

            status="INTERACTIVE",
            footer_left="[MOD-002] BUSINESS CASE",
            footer_right="COST_ANALYSIS",
        ),
    )
