# Stream — Bitcoin ML Infrastructure

Real-time Bitcoin transaction scoring, fee estimation, Lightning network analysis, and compliance automation. Deployed as a live dashboard with a dark terminal aesthetic.

## Architecture

```
Prefect Cloud (training pipelines)
  → Cloudflare R2 (model storage, S3-compatible)
  → FastHTML on Railway (dashboard + scoring API)
  ← Postgres + Redis on Railway (audit trail + cache)
  ← mempool.space API (live Bitcoin data)
```

## Models

| Model | Task | Architecture | Key Metric |
|-------|------|-------------|------------|
| Illicit XGBoost | Fraud detection | Gradient boosted trees (166 features) | PR-AUC ~0.82 |
| Illicit GCN | Fraud detection | Graph convolutional network (3 layers) | PR-AUC ~0.79 |
| Fee LightGBM | Fee estimation | Histogram-based GBT | MAE ~2 sat/vB |
| Onboarding XGBoost | KYC risk scoring | Calibrated (Platt) XGBoost | Accuracy ~0.85 |

## Quick Start

```bash
# Install dependencies
uv sync

# Start local services (Postgres + Redis)
docker compose up -d

# Run the dashboard
make dev

# Train models
make train-illicit
make train-fees
make train-lightning
make train-onboarding

# Run tests
make test
```

## Pages

- `/` — Real-time scoring feed (live mempool transactions)
- `/illicit` — Illicit detection dashboard + interactive scoring
- `/fees` — Live fee estimation + ML vs Bitcoin Core comparison
- `/lightning` — Lightning network topology analysis
- `/onboarding` — KYC risk scoring demo
- `/alerts` — Compliance alert queue + LLM narratives
- `/models` — Model cards + ROI calculator
- `/pipeline` — Live Prefect pipeline status
- `/walkthrough/*` — Interactive technical deep dives

## Design Decisions

- **PR-AUC over ROC-AUC**: Severe class imbalance (~10:1)
- **Temporal split**: Prevents data leakage from future transactions
- **Cost-sensitive thresholds**: Optimizes for business cost, not accuracy
- **SHAP + Claude API**: Compliance-ready explainability pipeline
- **Prefect over Airflow**: Python-native, modern, better for small teams

## Deploy

```bash
# Railway
railway up

# Or via Docker
docker build -t stream .
docker run -p 5000:5000 stream
```
