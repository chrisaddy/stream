# Stream — Interview Walkthrough & Quick Reference

> PRIVATE: This file is your interview cheat sheet. Not deployed.

## 30-Second Pitch

"I built the ML infrastructure that a new ML team at a Bitcoin exchange needs — fraud detection with compliance-ready explainability, real-time fee estimation, LLM-powered analyst narratives, and the MLOps platform to run it all. It's deployed live with real Bitcoin data flowing through it."

## Architecture (One Paragraph)

Prefect Cloud orchestrates training pipelines that produce XGBoost, GCN, LightGBM, and calibrated XGBoost models. Models are serialized and stored in Cloudflare R2 via S3-compatible API. A FastHTML application on Railway loads these models on startup and serves both an interactive dashboard (Spark Diagnostic dark terminal aesthetic) and a JSON scoring API. Postgres stores the full prediction audit trail; Redis caches LLM narratives and frequent predictions. Live Bitcoin data streams from mempool.space API, with real transactions scored in real-time via Server-Sent Events.

## Key Decisions and Why

| Decision | Rationale |
|----------|-----------|
| **PR-AUC over ROC-AUC** | 10:1 class imbalance. ROC-AUC overestimates by crediting correct negatives. PR-AUC focuses on the rare illicit class — what compliance cares about. |
| **Temporal split (not random)** | Random split leaks future data. Temporal split (train 1-34, test 35-49) mirrors production: you only ever have past data. |
| **XGBoost + GCN** | XGBoost = fast inference (~2ms), SHAP-explainable, production workhorse. GCN = leverages graph structure (payment flows), higher accuracy on connected components. Shows when to use tabular vs graph. |
| **SHAP for compliance** | Regulatory requirement: explainable AI. SHAP provides per-prediction feature attributions. "Feature 47 increased risk by 0.234 because aggregated neighbor volume is 3.2 SD above mean." |
| **Cost-sensitive thresholds** | Default 0.5 minimizes error, not cost. FN costs ~$50K (regulatory fine), FP costs ~$50 (manual review). We optimize for minimum business cost. |
| **Claude API for narratives** | Job posting requires "LLM-based systems." SHAP → Claude Haiku → analyst-ready SAR narrative. Reduces per-review time from ~5 min to ~1 min. |
| **Prefect over Airflow** | Python-native, no YAML DAGs, better for small teams, modern API, first-class artifacts. |
| **FastHTML over React** | Python full-stack, matches Elixir/LiveView philosophy at River, HTMX for interactivity without JS frameworks. |
| **Logistic regression baseline** | Screening interview asks about it. Show mastery by USING it, then improving with calibrated XGBoost. |

## Insider Context to Leverage

- River just created a dedicated ML team
- They already have fraud/risk models but want to expand
- Frame yourself as the person who **builds the ML function**, not just ships models
- The roadmap section of the site demonstrates this thinking
- River uses Elixir/Phoenix — the integration page shows you thought about how Python ML fits into their stack
- River's Lightning Service (RLS) — the Lightning analysis page shows direct relevance

## Preliminary Screening Prep

**Logistic regression** — be ready to explain from first principles:
- Sigmoid function: σ(z) = 1/(1+e^(-z)), maps any real number to (0,1)
- Log-loss: -[y·log(p) + (1-y)·log(1-p)], penalizes confident wrong predictions
- Gradient descent: update weights in direction of steepest descent of loss
- Regularization: L1 (sparse features) vs L2 (small weights)
- Your R library from grad school — mention it shows deep understanding

**XGBoost** — gradient boosted trees, each tree corrects previous trees' residuals, regularized objective

**GCN** — message passing: each node aggregates neighbor features, 3 layers = 3-hop receptive field

## Anticipated Interview Questions

**"How would this integrate with River's Elixir backend?"**
→ JSON API contract. In production: AMQP message pattern — Elixir publishes to scoring queue, ML service consumes, scores, publishes result, Elixir GenServer picks up risk score. The `/walkthrough/integration` page has working code examples.

**"How do you handle model drift?"**
→ Temporal evaluation shows model stability across timesteps. In production: monitor prediction distribution (KL divergence vs training), scheduled retraining via Prefect, alert on metric degradation. The pipeline page shows freshness monitoring.

**"What's the false positive rate and how would you reduce it?"**
→ Cost-sensitive threshold selection balances FP/FN based on business cost. Reduction: analyst feedback loop (TP/FP labels from review queue), retrain on feedback, active learning on uncertain predictions.

**"Why two models for illicit detection?"**
→ XGBoost for real-time (fast, explainable via SHAP). GCN for batch analysis (graph structure captures laundering patterns like layering). In production, XGBoost gates the fast path; GCN runs on flagged transactions for deeper analysis.

**"How would you scale this?"**
→ Separate scoring service behind load balancer. Batch vs real-time paths. Feature store for shared computation. Redis for high-frequency identical predictions. GPU serving (TorchServe/Triton) for GCN at scale.

**"What would you build next beyond fraud?"**
→ Show the roadmap page. Onboarding risk scoring, Lightning routing optimization, smart DCA, customer segmentation. The key: platform first (what this portfolio shows), then expand into new domains.

**"How would you set up an ML team from scratch?"**
→ MLOps foundation first (this portfolio). Shared tooling, testing culture (temporal eval, not random split), code review standards for ML (data leakage checks, threshold rationale), on-call for model degradation.

**"Tell me about the LLM integration"**
→ Claude Haiku generates analyst-ready narratives from SHAP explanations. Input: risk score + top 5 SHAP features. Output: 2-3 sentence SAR-suitable narrative. Cached in Redis (5 min TTL). Reduces per-review time. Full audit trail in Postgres.

## Metrics Quick Reference

| Model | PR-AUC | Precision | Recall | F1 | Inference |
|-------|--------|-----------|--------|----|-----------|
| XGBoost (Illicit) | ~0.82 | ~0.89 | ~0.73 | ~0.80 | ~2ms |
| GCN (Illicit) | ~0.79 | ~0.86 | ~0.79 | ~0.82 | ~50ms |
| LightGBM (Fees) | MAE ~2 sat/vB | - | - | - | ~1ms |
| XGBoost (Onboarding) | - | Acc 0.85 | F1m 0.77 | - | ~1ms |

## ROI Talking Points

- 50K daily transactions, 5% current manual review rate = 2,500 reviews/day
- ML auto-clears 84% → 400 reviews/day
- LLM narratives reduce per-review from 5 min to 1 min
- Annual savings: ~$1.5M in analyst time
- Infrastructure cost: ~$6K/year
- Breakeven: 2 days
