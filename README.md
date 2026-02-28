# Stream

Production ML infrastructure for Bitcoin: illicit transaction detection, real-time fee estimation, Lightning network analysis, and compliance automation — served as a live dashboard scoring real mempool transactions.

```
                    ┌─────────────────┐
                    │  Prefect Cloud   │
                    │  (orchestration) │
                    └────────┬────────┘
                             │ train every 2h
                             ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ mempool.space │───▶│  FastHTML App    │◀──▶│  Cloudflare R2   │
│  (live data)  │    │  (Railway)      │    │  (model store)   │
└──────────────┘    └────────┬────────┘    └──────────────────┘
                             │
                    ┌────────┴────────┐
                    │   Postgres      │
                    │   (audit trail) │
                    └─────────────────┘
```

## What This Is

Four ML models behind a single API and dashboard, built for a Bitcoin exchange compliance stack:

| Model | What It Does | How | Key Metric |
|-------|-------------|-----|------------|
| **Illicit XGBoost** | Flags suspicious Bitcoin transactions | 166 features from Elliptic dataset, temporal train/test split, SHAP explanations | PR-AUC |
| **Illicit GCN** | Same task, but uses the transaction graph | 3-layer Graph Convolutional Network over the UTXO payment graph | PR-AUC |
| **Fee LightGBM** | Predicts optimal fee rate (sat/vB) | Live mempool snapshots from mempool.space | MAE |
| **Onboarding XGBoost** | Risk-scores new customer signups | Calibrated probabilities (Platt scaling) on synthetic KYC data | Accuracy, calibration |

Every prediction is:
- **Explainable** — SHAP feature attributions on every score
- **Narrated** — Claude API turns SHAP into analyst-ready compliance text
- **Auditable** — full prediction trail in Postgres (model version, threshold, features, narrative)
- **Live** — real Bitcoin transactions from the mempool, scored via SSE in real-time

## Quick Start

```bash
uv sync                    # install deps
docker compose up -d       # postgres + redis (local dev)
just dev                   # http://localhost:5000
```

### Deploy Pipelines to Prefect Cloud

```bash
prefect cloud login        # authenticate
just deploy-flows          # registers 4 flows, cron every 2h
just worker                # starts local worker to execute runs
```

### Deploy App to Railway

```bash
railway link               # link to project
railway up                 # deploy (or push to main for auto-deploy)
```

Set these env vars in Railway:

```
DATABASE_URL, REDIS_URL, PREFECT_API_KEY, PREFECT_API_URL,
R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
R2_BUCKET_NAME, ANTHROPIC_API_KEY
```

## Project Structure

```
src/stream/
├── illicit/          # Project 1: illicit transaction detection
│   ├── data.py           PyG EllipticBitcoinDataset loader + pandas export
│   ├── preprocess.py     temporal split (1-34 train, 35-49 test), scaling
│   ├── xgboost_model.py  XGBoost + SHAP TreeExplainer
│   ├── gcn_model.py      3-layer GCNConv (165→128→128→2)
│   ├── evaluate.py       PR-AUC, cost-sensitive thresholds, per-timestep eval
│   └── pipeline.py       Prefect flow: load → split → train → evaluate → R2
│
├── fees/             # Project 2: fee estimation
│   ├── collector.py      mempool.space API poller (mempool, fees, blocks)
│   ├── features.py       mempool size, fee buckets, time-of-day, block fullness
│   ├── model.py          LightGBM regression (MAE objective)
│   ├── evaluate.py       MAE, MAPE, regime-based analysis
│   └── pipeline.py       Prefect flow: collect → featurize → train → R2
│
├── lightning/        # Project 3: Lightning network analysis
│   ├── data.py           mempool.space Lightning API → NetworkX graph
│   ├── features.py       degree centrality, betweenness, capacity metrics
│   ├── model.py          LightGBM capacity prediction
│   └── pipeline.py       Prefect flow: topology → graph → features → train
│
├── onboarding/       # Project 4: onboarding risk scoring
│   ├── data.py           synthetic KYC dataset (10K records)
│   ├── model.py          logistic regression baseline + calibrated XGBoost
│   ├── evaluate.py       calibration curves, per-tier analysis
│   └── pipeline.py       Prefect flow: generate → train LR → train XGB → eval
│
├── compliance/
│   └── narrator.py       Claude API: SHAP → SAR-ready compliance narrative
│
├── app/              # FastHTML dashboard + API
│   ├── main.py           routes for 14 pages
│   ├── api.py            /api/v1/* JSON endpoints + SSE scoring stream
│   ├── components.py     reusable Spark-themed FT components
│   ├── models.py         model loader (R2 → memory, local fallback)
│   ├── static/style.css  dark terminal aesthetic
│   └── pages/            home, illicit, fees, lightning, onboarding,
│                          alerts, models, pipeline, walkthrough (×6)
│
├── models/           # SQLAlchemy ORM
│   ├── predictions.py    prediction_audit table (full audit trail)
│   └── alerts.py         compliance alert queue
│
├── config.py         # Pydantic Settings (env-based)
└── db.py             # SQLAlchemy engine
```

## Dashboard Pages

| Route | What's There |
|-------|-------------|
| `/` | Real-time SSE feed — live mempool transactions scored with risk + fee estimate |
| `/illicit` | XGBoost vs GCN comparison table, interactive single-tx scoring, threshold slider |
| `/fees` | Live fee recommendations from mempool.space, ML vs Bitcoin Core side-by-side |
| `/lightning` | Network stats, top routing nodes ranked by centrality, node evaluator |
| `/onboarding` | Mock signup form → instant risk score with explanation |
| `/alerts` | Compliance alert queue, click-to-expand SHAP + AI narrative |
| `/models` | Model cards (architecture, data, loss, limitations) + interactive ROI calculator |
| `/pipeline` | DAG visualization, recent Prefect flow runs, model freshness badges |
| `/walkthrough` | Architecture overview, tech stack rationale |
| `/walkthrough/illicit` | Why PR-AUC, why temporal split, cost-sensitive thresholds, SHAP→LLM pipeline |
| `/walkthrough/fees` | Mempool mechanics, where ML beats Bitcoin Core, non-stationarity |
| `/walkthrough/architecture` | Prefect vs Airflow, R2 as registry, Elixir integration pattern |
| `/walkthrough/integration` | Live API tester, Elixir GenServer code, AMQP flow diagram |
| `/walkthrough/roadmap` | 4-quarter ML roadmap, platform vision, team building |

## API

```
GET  /api/v1/health
POST /api/v1/illicit/score          # risk score + SHAP for a feature vector
GET  /api/v1/illicit/random-sample  # random 166-feature test vector
GET  /api/v1/fees/estimate?target_blocks=6
GET  /api/v1/fees/live              # current mempool.space recommendations
GET  /api/v1/fees/compare           # ML vs Bitcoin Core
GET  /api/v1/lightning/stats        # network node/channel/capacity counts
GET  /api/v1/lightning/top-nodes    # top 10 by connectivity
POST /api/v1/lightning/evaluate     # score a node by pubkey
POST /api/v1/onboarding/score      # KYC risk assessment
POST /api/v1/integration/webhook   # full scoring response (Elixir-compatible)
GET  /api/v1/stream/scores          # SSE stream of live scored transactions
POST /api/v1/models/roi             # interactive ROI calculator
POST /api/v1/models/reload          # hot-reload models from R2
```

## Design Decisions

**PR-AUC over ROC-AUC** — With ~10:1 class imbalance (licit vs illicit), ROC-AUC inflates performance by crediting correct classification of the abundant negative class. PR-AUC focuses entirely on the rare positive class, which is what compliance actually cares about.

**Temporal split, not random** — Random train/test split leaks future transactions into training. Temporal split (train on timesteps 1–34, test on 35–49) mirrors production: you only ever have past data. This typically reduces apparent performance by 5–15% but gives honest estimates.

**Cost-sensitive threshold** — The default 0.5 threshold minimizes classification error, not business cost. A missed illicit transaction (FN) costs ~$50K in regulatory exposure. A false alarm (FP) costs ~$50 in analyst review time. We optimize the threshold for minimum total cost.

**SHAP + Claude API** — Regulatory explainability requirement. SHAP provides per-prediction feature attributions; Claude Haiku translates those into 2–3 sentence SAR-suitable narratives. Cached in Redis. Reduces analyst review time from ~5 min to ~1 min per alert.

**Prefect over Airflow** — Python-native (no YAML DAGs), first-class artifacts for metric tracking, S3 blocks for cloud-agnostic model storage. Better fit for a small ML team.

**FastHTML over React** — Python full-stack, HTMX for interactivity without a JS build step. Matches the Elixir/LiveView server-rendered philosophy.

## Tests

```bash
just test   # 26 tests
```

Covers temporal split correctness (no leakage), XGBoost probability validity, model serialization roundtrip, all API endpoints (status codes + content), and fee feature extraction.

## Stack

Python 3.13 · FastHTML · HTMX · XGBoost · PyTorch Geometric · LightGBM · scikit-learn · SHAP · Prefect · SQLAlchemy · Cloudflare R2 · Railway · Claude API · mempool.space API · Postgres · Redis · structlog
