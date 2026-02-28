"""Walkthrough / demo pages for interviewers."""

from fasthtml.common import *
from stream.app.components import (
    Card, DiagnosticFrame, Page, WalkthroughSection,
)


def walkthrough_overview():
    return Page("Walkthrough", "/walkthrough",
        DiagnosticFrame(
            "STREAM // WALKTHROUGH",

            WalkthroughSection(
                "Why These Projects for River",
                "River Financial is building a dedicated ML team. They need fraud detection with compliance-ready explainability, real-time fee estimation, and the MLOps platform to run it all. This portfolio demonstrates exactly that infrastructure — deployed live, not in a notebook.",
            ),

            WalkthroughSection(
                "Architecture",
                "Prefect Cloud orchestrates training pipelines. Models are stored in Cloudflare R2 via S3-compatible API. FastHTML on Railway serves the dashboard and scoring API. Postgres stores audit trails, Redis caches predictions and LLM narratives. mempool.space provides live Bitcoin data.",
                """Prefect Cloud (training)
  -> Cloudflare R2 (model storage)
  -> FastHTML on Railway (serving + dashboard)
  <- Postgres + Redis on Railway (data + cache)
  <- mempool.space API (live Bitcoin data)""",
            ),

            WalkthroughSection(
                "Tech Stack Rationale",
                "Prefect over Airflow: Python-native, better for small teams, modern API. FastHTML over React: Python full-stack, matches Elixir/LiveView philosophy at River, HTMX for interactivity without JS frameworks. XGBoost for production speed + SHAP explainability. GCN to show when graph structure adds value.",
            ),

            WalkthroughSection(
                "Four Models, One Platform",
                "Illicit detection (XGBoost + GCN), fee estimation (LightGBM), Lightning network analysis (LightGBM), and onboarding risk scoring (Logistic Regression + Calibrated XGBoost). All sharing the same Prefect pipeline infrastructure, R2 model store, and FastHTML serving layer.",
            ),

            status="INTERACTIVE",
            footer_left="[WLK-001] ARCHITECTURE",
            footer_right="BITCOIN_ML_INFRA",
        ),
    )


def walkthrough_illicit():
    return Page("Illicit Deep Dive", "/walkthrough/illicit",
        DiagnosticFrame(
            "ILLICIT DETECTION // DEEP DIVE",

            WalkthroughSection(
                "The Problem: Bitcoin AML at a Regulated Exchange",
                "River holds a BitLicense and state money transmitter licenses. Every transaction must be screened for illicit activity. False negatives mean regulatory fines (potentially millions). False positives mean customer friction and support costs. ML automates the triage, not the decision.",
            ),

            WalkthroughSection(
                "The Data: Elliptic Bitcoin Dataset",
                "~200K Bitcoin transactions across 49 timesteps (2 weeks). 166 features per transaction: 1 timestep indicator, 93 local features (inputs, outputs, fees, size), and 72 aggregated features (statistics about neighboring transactions in the payment graph). Labels: licit, illicit, unknown (~10:1 imbalance).",
            ),

            WalkthroughSection(
                "Why Temporal Split (Not Random)",
                "Random train/test split causes data leakage: the model sees future transactions during training. In production, you only have past data. Temporal split (train on timesteps 1-34, test on 35-49) mimics real deployment. This typically reduces apparent performance by 5-15% but gives honest estimates.",
            ),

            WalkthroughSection(
                "Why PR-AUC Over ROC-AUC",
                "With 10:1 class imbalance, ROC-AUC overestimates performance because it credits the model for correctly classifying the abundant negative class. PR-AUC focuses entirely on the rare positive (illicit) class — which is what compliance cares about.",
            ),

            WalkthroughSection(
                "XGBoost: The Production Workhorse",
                "Fast inference (~2ms), SHAP-explainable (regulatory requirement), handles tabular data well. scale_pos_weight handles class imbalance. Early stopping on validation PR-AUC prevents overfitting.",
            ),

            WalkthroughSection(
                "GCN: When Graph Structure Matters",
                "XGBoost sees each transaction in isolation. The GCN sees the transaction graph — who paid whom. This captures money laundering patterns like layering (rapid fund movement through many addresses) that isolated features miss. Tradeoff: slower inference (~50ms), harder to explain to regulators.",
            ),

            WalkthroughSection(
                "Cost-Sensitive Threshold Selection",
                "The default 0.5 threshold minimizes classification error, but not business cost. A missed illicit transaction (FN) costs ~$50K in regulatory risk. A false alarm (FP) costs ~$50 in analyst review time. We optimize the threshold for minimum total business cost, not accuracy.",
            ),

            WalkthroughSection(
                "SHAP -> LLM Narrative Pipeline",
                "SHAP provides feature attributions. Claude API translates those into compliance-ready narratives: 'This transaction was flagged due to aggregated neighbor volume 3.2 SD above mean, combined with unusual output count suggesting fund splitting.' This reduces analyst review time from ~5 min to ~1 min per alert.",
            ),

            status="TECHNICAL",
            footer_left="[WLK-002] ELLIPTIC",
            footer_right="PR_AUC",
        ),
    )


def walkthrough_fees():
    return Page("Fee Deep Dive", "/walkthrough/fees",
        DiagnosticFrame(
            "FEE ESTIMATION // DEEP DIVE",

            WalkthroughSection(
                "The Problem: Bitcoin Core's Blind Spot",
                "Bitcoin Core's fee estimator (estimatesmartfee) looks at historical block confirmation times but is blind to the current mempool state. When congestion spikes, it reacts slowly. ML can see the current mempool and predict fees more accurately, especially during volatile periods.",
            ),

            WalkthroughSection(
                "Mempool Mechanics",
                "The mempool is Bitcoin's waiting room for unconfirmed transactions. Miners pick transactions with the highest fee rates (sat/vB) first. When the mempool is congested, fees spike. When it's empty, fees drop to minimums. Our model captures this dynamic state.",
            ),

            WalkthroughSection(
                "Feature Engineering",
                "Mempool size (count + vsize), fee distribution across projected blocks, median fee rates, time since last block, hour/day-of-week patterns, block fullness. These features capture signal that Bitcoin Core's historical approach misses.",
            ),

            WalkthroughSection(
                "Where ML Wins",
                "Congestion spikes: ML sees the mempool filling up before fees actually rise. Weekends: activity patterns differ. Post-halving: fee dynamics change. Bitcoin Core adapts slowly; ML can retrain on new patterns.",
            ),

            WalkthroughSection(
                "Non-Stationarity",
                "Fee estimation is a non-stationary problem — the data distribution shifts constantly. This requires frequent retraining (daily or on-demand via Prefect), monitoring for prediction drift, and fallback to conservative estimates when confidence is low.",
            ),

            status="TECHNICAL",
            footer_left="[WLK-003] MEMPOOL",
            footer_right="SAT/VB",
        ),
    )


def walkthrough_architecture():
    return Page("Architecture", "/walkthrough/architecture",
        DiagnosticFrame(
            "SYSTEMS ARCHITECTURE // DEEP DIVE",

            WalkthroughSection(
                "Pipeline: Prefect Cloud",
                "Prefect over Airflow: Python-native (no YAML DAGs), dynamic workflows, first-class Python support, better for small teams. Flows are just decorated Python functions. Artifacts track training metrics. S3 blocks provide cloud-agnostic storage.",
            ),

            WalkthroughSection(
                "Model Serving: R2 as Model Registry",
                "Cloudflare R2 (S3-compatible) stores serialized models. On app startup, models are loaded into memory. Hot-reload endpoint allows updating without restart. This is the 'model registry' pattern — simple, effective, no MLflow needed for this scale.",
            ),

            WalkthroughSection(
                "Integration with Elixir/Phoenix",
                "River's backend is Elixir/Phoenix. This Python ML service exposes a JSON API. The integration pattern: Elixir GenServer calls our scoring endpoint via HTTP. In production, this would go through RabbitMQ (AMQP): transaction published to scoring queue -> ML service consumes, scores, publishes result -> Elixir picks up risk score.",
                """# Elixir GenServer calling pattern:
defmodule River.ML.ScoringClient do
  use GenServer

  def score_transaction(features) do
    GenServer.call(__MODULE__, {:score, features})
  end

  def handle_call({:score, features}, _from, state) do
    {:ok, response} = HTTPoison.post(
      "#{state.ml_service_url}/api/v1/integration/webhook",
      Jason.encode!(%{features: features, correlation_id: UUID.uuid4()}),
      [{"Content-Type", "application/json"}]
    )
    {:reply, Jason.decode!(response.body), state}
  end
end""",
            ),

            WalkthroughSection(
                "Monitoring & Audit",
                "Every prediction is logged to Postgres with: model version, input hash (SHA256, not raw features for privacy), risk score, threshold used, top SHAP features, LLM narrative, inference time. Regulators can reconstruct any flagging decision by prediction ID.",
            ),

            WalkthroughSection(
                "Scaling Considerations",
                "10x volume: Add Redis caching for repeated feature patterns, batch scoring endpoint. 100x volume: Separate scoring service behind load balancer, async processing via AMQP, feature store for shared computation, model serving via TorchServe/Triton for GPU models.",
            ),

            status="TECHNICAL",
            footer_left="[WLK-004] DESIGN",
            footer_right="PRODUCTION",
        ),
    )


def walkthrough_integration():
    return Page("Integration", "/walkthrough/integration",
        DiagnosticFrame(
            "INTEGRATION ARCHITECTURE",

            WalkthroughSection(
                "API Contract",
                "The scoring endpoint accepts a JSON payload with transaction features and returns a risk assessment with all fields an Elixir GenServer would need to route the transaction: risk score, label, SHAP explanation, compliance narrative, audit ID, and recommended action.",
            ),

            P("Test the integration endpoint:", style="color: var(--fg-dim); margin-bottom: 12px;"),

            Form(
                Textarea(
                    '{"features": [0.5, -0.3, 1.2], "correlation_id": "test-001"}',
                    name="payload",
                    cls="spark-input",
                    style="height: 80px; resize: none; margin-bottom: 12px;",
                ),
                Button("Send Webhook", cls="spark-btn", type="submit"),
                **{"hx-post": "/api/v1/integration/webhook", "hx-target": "#webhook-result"},
            ),

            Div(id="webhook-result", style="margin-top: 16px;"),

            WalkthroughSection(
                "AMQP Flow",
                "In production at River: Transaction arrives via Elixir -> Published to AMQP scoring queue -> ML service consumes message -> Scores transaction -> Publishes result to response queue -> Elixir GenServer picks up risk score and routes accordingly (auto-clear, flag for review, or block).",
            ),

            status="LIVE_API",
            footer_left="[WLK-005] WEBHOOK",
            footer_right="ELIXIR_COMPAT",
        ),
    )


def walkthrough_roadmap():
    return Page("ML Roadmap", "/walkthrough/roadmap",
        DiagnosticFrame(
            "ML ROADMAP FOR RIVER",

            WalkthroughSection(
                "Phase 1 (Now): Fraud & Risk",
                "Improve existing models with graph features, add explainability for compliance, establish MLOps foundation. This portfolio demonstrates the infrastructure: Prefect pipelines, model registry, serving layer, audit trail, LLM narrative generation.",
            ),

            WalkthroughSection(
                "Phase 2 (Q2): Onboarding & KYC",
                "ML-powered risk scoring for new accounts. Document verification confidence scoring. Synthetic identity detection. The onboarding model in this portfolio demonstrates the pattern — logistic regression baseline, calibrated XGBoost, fairness analysis.",
            ),

            WalkthroughSection(
                "Phase 3 (Q3): Lightning Network",
                "Routing optimization to reduce payment failures. Channel balance prediction. Lightning-specific anomaly detection. The lightning analysis in this portfolio shows graph feature engineering on the real network topology.",
            ),

            WalkthroughSection(
                "Phase 4 (Q4): Operations & Product",
                "Smart DCA optimization (improve River's flagship product). Customer segmentation for support prioritization. Churn prediction. These require customer data we don't have, but the infrastructure (Prefect + feature store + serving) is ready.",
            ),

            WalkthroughSection(
                "ML Platform Vision",
                "Reusable infrastructure that any River engineer can use. Shared feature store (Postgres-backed, demonstrated here). Model registry (R2 + Prefect, demonstrated here). Serving layer with monitoring (FastHTML API + audit trail, demonstrated here). Self-service model training via Prefect deployments.",
            ),

            WalkthroughSection(
                "Team Building",
                "Testing culture: every model has temporal evaluation, not just random split metrics. Code review standards for ML: data leakage checks, feature engineering review, threshold selection rationale. On-call for model degradation: prediction distribution monitoring, automated retraining triggers.",
            ),

            status="STAFF_ENGINEER_VIEW",
            footer_left="[WLK-006] STRATEGY",
            footer_right="BUILDING_THE_FUNCTION",
        ),
    )
