"""Walkthrough / demo pages for interviewers."""

from fasthtml.common import *
from stream.app.components import (
    Card, DiagnosticFrame, Page, WalkthroughSection,
)


def _deep_dive_link(href, title, subtitle):
    return A(
        Div(
            Div(title, style="font-weight: bold; color: var(--fg-bright);"),
            Div(subtitle, style="color: var(--fg-subtle); font-size: 11px; margin-top: 2px;"),
            cls="walkthrough-section",
            style="padding: 10px 12px; border: 1px solid var(--highlight-dim); margin-bottom: 6px; transition: border-color 0.2s;",
        ),
        href=href,
        style="text-decoration: none;",
    )


def walkthrough_overview():
    return Page("Walkthrough", "/walkthrough",
        DiagnosticFrame(
            "DEMO",

            Video(
                Source(src="/demo.mp4", type="video/mp4"),
                controls=True,
                style="width: 100%; border-radius: 4px; border: 1px solid var(--highlight-med);",
                preload="metadata",
            ),

            P("4-minute walkthrough: live scoring, instant learning, drift detection, "
              "self-healing adaptation, and anomaly detection — all on real Bitcoin mempool data.",
              style="color: var(--fg-subtle); font-size: 11px; margin-top: 8px;"),

            status="WALKTHROUGH",
            footer_left="SCREEN CAPTURE + AI NARRATION",
            footer_right="4:20",
        ),

        DiagnosticFrame(
            "STREAM // WHAT THIS IS",

            WalkthroughSection(
                "The Problem",
                "A regulated Bitcoin exchange needs to screen every transaction for illicit activity, estimate optimal fees, analyze Lightning network health, and score onboarding risk — all with full explainability for regulators and an audit trail for every decision. Most ML projects stop at a notebook. This one is deployed live with real data flowing through it.",
            ),

            WalkthroughSection(
                "The System",
                "Four ML models share one platform: Prefect Cloud orchestrates training, Cloudflare R2 stores models, FastHTML on Railway serves the dashboard and JSON API, Postgres logs every prediction for audit. Live Bitcoin mempool data flows through continuously. Analysts review flagged transactions, and their verdicts feed back into online learning.",
                """mempool.space (live Bitcoin data)
  -> Scoring Service (cascade: online -> batch -> heuristic)
    -> SHAP explanation + Claude narrative
      -> Analyst review (TP / FP / escalate)
        -> Online model learns immediately
          -> Drift detection triggers self-healing""",
            ),

            status="OVERVIEW",
            footer_left="[WLK-001] START HERE",
            footer_right="BITCOIN_ML_INFRA",
        ),

        DiagnosticFrame(
            "DEEP DIVES",

            P("Each section explains one part of the system in detail — the problem, the approach, and the design decisions.",
              style="color: var(--fg-subtle); font-size: 11px; margin-bottom: 12px;"),

            _deep_dive_link("/walkthrough/illicit",
                "Illicit Detection",
                "XGBoost + GCN on Elliptic dataset. Temporal split, PR-AUC, cost-sensitive thresholds, SHAP explainability."),

            _deep_dive_link("/walkthrough/fees",
                "Fee Estimation",
                "LightGBM on live mempool data. Why ML beats Bitcoin Core's estimator during congestion spikes."),

            _deep_dive_link("/walkthrough/architecture",
                "Systems Architecture",
                "Prefect pipelines, R2 model registry, API integration patterns, monitoring, and scaling strategy."),

            _deep_dive_link("/walkthrough/online-ml",
                "Online ML & Adaptation",
                "Feedback loop, River incremental learning, model racing, ADWIN drift detection, self-healing state machine."),

            _deep_dive_link("/walkthrough/compliance",
                "Compliance & Explainability",
                "SHAP -> Claude narratives, investigation agent, analyst workflow, full audit trail, cost-sensitive decisions."),

            status="NAVIGATION",
            footer_left="6 SECTIONS",
            footer_right="CLICK TO EXPLORE",
        ),

        DiagnosticFrame(
            "TECH STACK",

            WalkthroughSection(
                "Key Decisions",
                "Prefect over Airflow: Python-native, no YAML, better for small teams. FastHTML over React: Python full-stack, HTMX for interactivity without JS build step. XGBoost for production speed + SHAP explainability. River for online learning from streaming analyst feedback. Claude API for translating SHAP values into compliance-ready narratives.",
            ),

            WalkthroughSection(
                "Infrastructure",
                "Railway (app hosting + Postgres), Cloudflare R2 (S3-compatible model storage), Prefect Cloud (pipeline orchestration), mempool.space (live Bitcoin data). Everything is Python 3.13 — training, serving, dashboard, API, and online learning.",
                """Training:  Prefect Cloud -> XGBoost / LightGBM / GCN
Storage:   Cloudflare R2 (S3 API)
Serving:   FastHTML + uvicorn on Railway
Database:  PostgreSQL (audit trail + feedback)
Live Data: mempool.space API
LLM:       Claude Haiku (compliance narratives)
Online ML: River (incremental learning)""",
            ),

            status="STACK",
            footer_left="[WLK-001] TECHNOLOGY",
            footer_right="PYTHON_FULLSTACK",
        ),
    )


def walkthrough_illicit():
    return Page("Illicit Deep Dive", "/walkthrough/illicit",
        DiagnosticFrame(
            "ILLICIT DETECTION // DEEP DIVE",

            WalkthroughSection(
                "The Problem: Bitcoin AML at a Regulated Exchange",
                "A regulated exchange with a BitLicense and state money transmitter licenses must screen every transaction for illicit activity. False negatives mean regulatory fines (potentially millions). False positives mean customer friction and support costs. ML automates the triage, not the decision.",
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
                "Integration Pattern",
                "This Python ML service exposes a JSON API. Backend services call the scoring endpoint via HTTP. In production, this would go through a message queue (AMQP): transaction published to scoring queue -> ML service consumes, scores, publishes result -> backend picks up risk score.",
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


def walkthrough_online_ml():
    return Page("Online ML", "/walkthrough/online-ml",
        DiagnosticFrame(
            "ONLINE ML & ADAPTATION // DEEP DIVE",

            WalkthroughSection(
                "The Cold Start Problem",
                "The trained XGBoost model needs 166 features per transaction. Live mempool transactions only have 3: vsize, fee, fee_rate. You can't just zero-pad 163 features — that produces meaningless scores. So the system starts with a heuristic scorer and learns from analyst feedback to build a production model on the features it actually has.",
            ),

            WalkthroughSection(
                "Cascade Scoring: Honest Degradation",
                "The scoring service tries three models in order: (1) the River online model if it has enough labels, (2) a batch model retrained on accumulated feedback, (3) a heuristic based on fee rate, transaction size, and fee disproportion. Each level is honest about what it is — the UI badges transactions as 'ML' or 'HEURISTIC' so analysts know the confidence level.",
                """score_transaction(vsize, fee, fee_rate):
  1. river_predict_one()    # online model (if labels exist)
  2. learned_model.predict() # batch feedback model
  3. heuristic_risk_score()  # sigmoid-based fallback""",
            ),

            WalkthroughSection(
                "River: Incremental Learning from Analyst Feedback",
                "River is an online ML library — models learn one sample at a time. When an analyst marks a flagged transaction as true positive or false positive, that label is immediately fed to the River model (StandardScaler | LogisticRegression). No batch retraining needed. The model improves with every review, creating a tight human-in-the-loop feedback cycle.",
            ),

            WalkthroughSection(
                "Model Racing: Let Them Compete",
                "Three online classifiers race in parallel: Logistic Regression, Hoeffding Tree, and Gaussian Naive Bayes. Each receives every label and makes predictions. Rolling F1 scores track performance. The system uses F1-weighted ensemble voting — the best-performing model gets the most influence. This answers 'which algorithm works best for this data?' empirically, not theoretically.",
            ),

            WalkthroughSection(
                "Drift Detection: ADWIN + PSI",
                "ADWIN (Adaptive Windowing) detects change points in the prediction score stream — it dynamically grows and shrinks a window to identify when the scoring distribution shifts. PSI (Population Stability Index) compares the current score distribution against a baseline. Together, they catch both sudden regime changes and gradual drift.",
                """Drift monitor (ring buffer of 1000 scores):
  ADWIN  -> change-point detection (sudden shifts)
  PSI    -> distribution comparison (gradual drift)
  Both   -> signal when scoring behavior changes""",
            ),

            WalkthroughSection(
                "Adaptation State Machine: Self-Healing",
                "When drift is detected, the system doesn't just alert — it adapts. A state machine governs the process: STABLE -> DRIFT_DETECTED (after 3 consecutive signals, to avoid false alarms) -> ADAPTING (resets ADWIN, PSI baseline, online metrics, model race) -> STABILIZING (waits for new scores to converge) -> RECOVERED -> STABLE. A 60-second cooldown prevents thrashing between states.",
                """STABLE
  -> DRIFT_DETECTED (3 consecutive drift signals)
    -> ADAPTING (reset all online components)
      -> STABILIZING (collect new baseline)
        -> RECOVERED (scores stabilized)
          -> STABLE (60s cooldown)""",
            ),

            WalkthroughSection(
                "Why This Matters",
                "Most ML systems are 'train once, deploy, pray.' This system closes the loop: models degrade, drift is detected, adaptation resets the online components, analysts provide ground truth, and the online model rebuilds itself. The entire cycle is automated and observable from the dashboard.",
            ),

            status="TECHNICAL",
            footer_left="[WLK-005] ONLINE_ML",
            footer_right="ADAPTATION",
        ),
    )


def walkthrough_compliance():
    return Page("Compliance", "/walkthrough/compliance",
        DiagnosticFrame(
            "COMPLIANCE & EXPLAINABILITY // DEEP DIVE",

            WalkthroughSection(
                "The Regulatory Requirement",
                "FinCEN requires that financial institutions file SARs (Suspicious Activity Reports) for transactions they know or suspect involve illicit funds. The report must explain why the transaction was flagged — 'the ML model said so' is not sufficient. Every flagging decision needs a human-readable justification backed by specific evidence.",
            ),

            WalkthroughSection(
                "SHAP: Per-Prediction Feature Attribution",
                "SHAP (SHapley Additive exPlanations) decomposes each prediction into feature contributions. For a flagged transaction, it might show: neighbor aggregate volume contributed +0.35 to the risk score, unusual output count contributed +0.22, while normal fee rate contributed -0.08. This is exact — the contributions sum to the prediction. TreeExplainer runs in ~2ms for XGBoost, fast enough for real-time use.",
            ),

            WalkthroughSection(
                "SHAP -> Claude -> SAR Narrative",
                "Raw SHAP values are numbers. Compliance analysts need prose. The narrator module sends the top SHAP features to Claude Haiku, which generates a 2-3 sentence narrative: 'This transaction was flagged primarily due to aggregated neighbor transaction volume 3.2 SD above mean, combined with output patterns consistent with fund splitting.' This reduces per-alert review time from ~5 minutes to ~1 minute.",
                """SHAP features:
  neighbor_agg_volume: +0.35
  output_count: +0.22
  fee_rate: -0.08
        |
        v
Claude Haiku API (with compliance prompt)
        |
        v
"Flagged due to elevated neighbor volume
 and output patterns suggesting fund splitting.
 Fee behavior within normal range." """,
            ),

            WalkthroughSection(
                "Investigation Agent",
                "For high-risk alerts, the system can launch an autonomous investigation using Claude with tool_use. The agent has access to tools: look up alert details, check on-chain data, review transaction history, and record analyst verdicts. It produces a structured report with SUMMARY, RISK_ASSESSMENT, EVIDENCE, and RECOMMENDATION sections — a first draft for the compliance officer.",
            ),

            WalkthroughSection(
                "The Analyst Workflow",
                "An alert lands in the compliance queue. The analyst clicks to expand it and sees: transaction details, the risk score, the model that scored it (ML or heuristic), SHAP feature breakdown, and the AI-generated narrative. They review and submit a verdict: true positive (genuinely suspicious), false positive (safe), or escalate. That verdict feeds back into online learning.",
                """Alert arrives (risk > threshold)
  -> Analyst opens alert
    -> Sees: SHAP breakdown + AI narrative
      -> Submits verdict: TP / FP / ESCALATE
        -> Verdict -> online model (immediate)
        -> Verdict -> batch retrain (scheduled)""",
            ),

            WalkthroughSection(
                "Full Audit Trail",
                "Every prediction is logged to the prediction_audit table: model name, model version, risk score, threshold at time of scoring, top SHAP features, LLM narrative, inference time in ms. Regulators can reconstruct any flagging decision months later by prediction ID. This is the difference between 'we use ML' and 'we can prove exactly why ML flagged this transaction.'",
            ),

            WalkthroughSection(
                "Cost-Sensitive Decision Making",
                "The alert threshold isn't arbitrary. A missed illicit transaction (false negative) carries ~$50K in regulatory risk. A false alarm (false positive) costs ~$50 in analyst review time. The system optimizes the classification threshold for minimum total business cost, not for maximum accuracy. The threshold is adjustable from the dashboard as business conditions change.",
            ),

            status="TECHNICAL",
            footer_left="[WLK-006] COMPLIANCE",
            footer_right="EXPLAINABILITY",
        ),
    )



