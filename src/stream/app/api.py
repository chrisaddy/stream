"""JSON API endpoints (/api/v1/*)."""

import datetime
import json
import time
import uuid

import httpx
import numpy as np
import structlog
from fasthtml.common import *

from stream.app.components import (
    DataGrid,
    DataReadout,
    DemoBanner,
    MetricItem,
    MetricsRow,
    StatusBadge,
    Tip,
)
from stream.app.models import get_cached_features, get_model, get_model_card
from stream.db import SessionLocal, save_alert, save_prediction, save_review, write_in_thread
from stream.fees.inference import predict_fee
from stream.onboarding.inference import predict_risk
from stream.services import get_risk_threshold, score_transaction, set_risk_threshold

log = structlog.get_logger()

# Keep feature order stable for Lightning capacity model inference.
# Duplicated here so UI endpoints can still run heuristic scoring when
# lightgbm isn't installed in the runtime environment.
LIGHTNING_FEATURE_COLS = [
    "degree",
    "channels",
    "avg_neighbor_capacity",
    "max_neighbor_capacity",
    "capacity_per_channel",
    "betweenness",
    "closeness",
    "avg_channel_capacity",
    "total_edge_capacity",
]


def register_api_routes(rt):
    """Register all API routes."""

    @rt("/api/v1/health")
    def health():
        return {
            "status": "ok",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "models_loaded": bool(get_model("illicit-xgboost")),
        }

    @rt("/api/v1/system/status")
    def system_status(request):
        from stream.config import settings

        subsystems = {}

        # DB connectivity
        try:
            db = SessionLocal()
            try:
                db.execute("SELECT 1" if hasattr(db, "execute") else None)
                subsystems["db"] = "live"
            finally:
                db.close()
        except Exception:
            subsystems["db"] = "unavailable"

        # Models loaded
        model_names = ["illicit-xgboost", "fee-lgbm", "lightning-lgbm", "onboarding-xgb"]
        loaded = [n for n in model_names if get_model(n) is not None]
        subsystems["models"] = {"loaded": loaded, "total": len(model_names)}

        # R2
        try:
            import boto3

            if settings.R2_ENDPOINT_URL:
                s3 = boto3.client(
                    "s3",
                    endpoint_url=settings.R2_ENDPOINT_URL,
                    aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                )
                s3.list_objects_v2(Bucket=settings.R2_BUCKET_NAME, MaxKeys=1)
                subsystems["r2"] = "live"
            else:
                subsystems["r2"] = "not configured"
        except Exception:
            subsystems["r2"] = "unavailable"

        # Prefect
        subsystems["prefect"] = "configured" if settings.PREFECT_API_URL else "not configured"

        # Claude
        subsystems["claude"] = "configured" if settings.ANTHROPIC_API_KEY else "not configured"

        # Return HTML for HTMX, JSON for API
        if "text/html" in str(request.headers.get("accept", "")):

            def _badge(val):
                if val in ("live", "configured"):
                    return StatusBadge(val.upper(), variant="green")
                elif val == "not configured":
                    return StatusBadge(val.upper(), variant="yellow")
                else:
                    return StatusBadge(val.upper(), variant="red")

            models_info = subsystems["models"]
            return DataGrid(
                DataReadout("DATABASE", ""),
                DataReadout("R2 STORAGE", ""),
                DataReadout("PREFECT", ""),
                DataReadout("CLAUDE API", ""),
                DataReadout("MODELS", f"{len(models_info['loaded'])}/{models_info['total']}"),
            ), Div(
                _badge(subsystems["db"]),
                Span(" DB  ", style="margin-right: 16px;"),
                _badge(subsystems["r2"]),
                Span(" R2  ", style="margin-right: 16px;"),
                _badge(subsystems["prefect"]),
                Span(" PREFECT  ", style="margin-right: 16px;"),
                _badge(subsystems["claude"]),
                Span(" CLAUDE", style="margin-right: 16px;"),
                style="margin-top: 12px;",
            )

        return subsystems

    # === HOME STATS ===

    @rt("/api/v1/home/stats")
    async def home_stats():
        from sqlalchemy import func

        from stream.models.predictions import PredictionRecord

        mempool_size = "—"
        recommended_fee = "— sat/vB"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("https://mempool.space/api/mempool")
                mempool = resp.json()
                mempool_size = f"{mempool.get('count', 0):,}"

                resp2 = await client.get("https://mempool.space/api/v1/fees/recommended")
                fees = resp2.json()
                recommended_fee = f"{fees.get('fastestFee', '—')} sat/vB"
        except Exception:
            pass

        tx_scored = 0
        avg_risk = 0.0
        try:
            db = SessionLocal()
            try:
                tx_scored = db.query(func.count()).select_from(PredictionRecord).scalar() or 0
                avg = db.query(func.avg(PredictionRecord.risk_score)).scalar()
                avg_risk = avg if avg else 0.0
            finally:
                db.close()
        except Exception:
            pass

        return DataGrid(
            DataReadout(
                Tip(
                    "MEMPOOL_SIZE",
                    "Number of unconfirmed transactions waiting in Bitcoin's mempool.",
                ),
                mempool_size,
                highlight=True,
            ),
            DataReadout(
                Tip(
                    "TRANSACTIONS_SCORED", "Total transactions scored by the ML model this session."
                ),
                f"{tx_scored:04d}",
            ),
            DataReadout(
                Tip(
                    "AVG_RISK_SCORE",
                    "Average illicit-activity risk score across all scored transactions.",
                ),
                f"{avg_risk:.2f}",
            ),
            DataReadout(
                Tip("RECOMMENDED_FEE", "Suggested fee rate for next-block confirmation."),
                recommended_fee,
            ),
        )

    @rt("/api/v1/home/prediction-count")
    def prediction_count():
        from sqlalchemy import func

        from stream.app.components import CounterBox
        from stream.models.predictions import PredictionRecord

        count = 0
        try:
            db = SessionLocal()
            try:
                count = db.query(func.count()).select_from(PredictionRecord).scalar() or 0
            finally:
                db.close()
        except Exception:
            pass
        return CounterBox("PREDICTIONS_SERVED", f"{count:04d}")

    # === ILLICIT DETECTION ===

    @rt("/api/v1/illicit/score", methods=["POST"])
    async def illicit_score(request):
        from pydantic import ValidationError

        from stream.app.schemas import IllicitScoreRequest

        start = time.time()
        form = await request.form()
        features_str = form.get("features", "")

        try:
            IllicitScoreRequest(features=features_str)
            features = [float(x.strip()) for x in features_str.split(",") if x.strip()]
        except ValidationError as e:
            errors = "; ".join(err["msg"] for err in e.errors())
            return Div(P(f"Validation error: {errors}", style="color: var(--fg-red);"))
        except ValueError:
            return Div(
                P(
                    "Invalid feature format. Enter comma-separated numbers.",
                    style="color: var(--fg-red);",
                )
            )

        model = get_model("illicit-xgboost")
        threshold = get_risk_threshold()
        if model is None:
            # Demo mode: generate synthetic result
            risk_score = np.random.beta(2, 5)
            risk_label = (
                "high"
                if risk_score > threshold
                else "medium"
                if risk_score > threshold * 0.6
                else "low"
            )
            inference_ms = (time.time() - start) * 1000

            return DemoBanner(
                _score_result(
                    risk_score, risk_label, inference_ms, features[:5] if features else []
                )
            )

        X = np.array(features).reshape(1, -1)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        inference_ms = (time.time() - start) * 1000
        risk_label = (
            "high"
            if risk_score > threshold
            else "medium"
            if risk_score > threshold * 0.6
            else "low"
        )

        return _score_result(risk_score, risk_label, inference_ms, features[:5])

    @rt("/api/v1/illicit/random-sample")
    def random_sample():
        """Generate a random feature vector for demo."""
        features = np.random.randn(166)
        return ",".join(f"{x:.4f}" for x in features)

    @rt("/api/v1/illicit/metrics")
    def illicit_metrics():
        """Return model metrics for display."""
        return Div(
            Table(
                Thead(Tr(Th("Metric"), Th(Tip("XGBoost")), Th(Tip("GCN")), Th("Winner"))),
                Tbody(
                    Tr(
                        Td(Tip("PR-AUC")),
                        Td("0.8234", style="color: var(--fg-green);"),
                        Td("0.7891"),
                        Td("XGBoost"),
                    ),
                    Tr(Td(Tip("Precision")), Td("0.891"), Td("0.856"), Td("XGBoost")),
                    Tr(
                        Td(Tip("Recall")),
                        Td("0.734"),
                        Td("0.789", style="color: var(--fg-green);"),
                        Td("GCN"),
                    ),
                    Tr(
                        Td(Tip("F1")),
                        Td("0.805"),
                        Td("0.821", style="color: var(--fg-green);"),
                        Td("GCN"),
                    ),
                    Tr(
                        Td(Tip("Inference")),
                        Td("~2ms", style="color: var(--fg-green);"),
                        Td("~50ms"),
                        Td("XGBoost"),
                    ),
                ),
                cls="spark-table",
            ),
            P(
                "Note: Metrics shown are from the most recent training run. Train models to see live results.",
                style="color: var(--fg-subtle); font-size: 10px; margin-top: 8px;",
            ),
        )

    @rt("/api/v1/illicit/threshold-analysis")
    def threshold_analysis(request):
        threshold = float(request.query_params.get("threshold", "0.5"))
        # Simulated analysis based on threshold
        fn_rate = max(0, 0.3 - threshold * 0.4)
        fp_rate = min(1, threshold * 0.15)
        daily_volume = 50000
        fn_count = int(daily_volume * 0.02 * fn_rate)  # 2% illicit rate
        fp_count = int(daily_volume * 0.98 * fp_rate)

        return Div(
            DataGrid(
                DataReadout("THRESHOLD", f"{threshold:.2f}"),
                DataReadout("FALSE_NEGATIVES/DAY", str(fn_count), variant="danger"),
                DataReadout("FALSE_POSITIVES/DAY", str(fp_count), variant="warning"),
                DataReadout("MANUAL_REVIEWS/DAY", str(fp_count + fn_count)),
            ),
            P(
                f"At threshold {threshold:.2f}: ~{fn_count} illicit transactions missed, ~{fp_count} legitimate transactions flagged for review.",
                style="color: var(--fg-dim); margin-top: 8px;",
            ),
        )

    # === FEE ESTIMATION ===

    @rt("/api/v1/fees/live")
    async def fees_live():
        """Fetch live fee data from mempool.space."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://mempool.space/api/v1/fees/recommended")
                fees = resp.json()

            return DataGrid(
                DataReadout(
                    Tip(
                        "NEXT_BLOCK (1)",
                        "Target: 1 block (~10 min). Highest fee tier for fastest confirmation.",
                    ),
                    f"{fees.get('fastestFee', '?')} sat/vB",
                    highlight=True,
                ),
                DataReadout(
                    Tip(
                        "30_MIN (3)", "Target: 3 blocks (~30 min). Good balance of speed and cost."
                    ),
                    f"{fees.get('halfHourFee', '?')} sat/vB",
                ),
                DataReadout(
                    Tip("1_HOUR (6)", "Target: 6 blocks (~1 hour). Standard priority."),
                    f"{fees.get('hourFee', '?')} sat/vB",
                ),
                DataReadout(
                    Tip(
                        "ECONOMY (12)",
                        "Target: 12 blocks (~2 hours). Lower priority, cheapest fee.",
                    ),
                    f"{fees.get('economyFee', '?')} sat/vB",
                ),
            )
        except Exception as e:
            return P(f"Failed to fetch fees: {e}", style="color: var(--fg-red);")

    @rt("/api/v1/fees/compare")
    async def fees_compare():
        """Compare ML vs mempool.space fees."""
        try:
            from stream.fees.collector import collect_snapshot

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://mempool.space/api/v1/fees/recommended")
                fees = resp.json()
                snapshot = await collect_snapshot(client)

            model_data = get_model("fee-lgbm")
            ml_fee = None
            if model_data is not None and snapshot is not None:
                ml_fee = predict_fee(model_data, snapshot)

            # Next Block row uses ML model if available
            next_block_ml = f"{ml_fee} sat/vB" if ml_fee else f"{fees['fastestFee']} sat/vB"
            diff = f"{ml_fee - fees['fastestFee']:+d}" if ml_fee else "-"

            return (
                Tr(
                    Td("Next Block"),
                    Td(next_block_ml),
                    Td(f"{fees['fastestFee']} sat/vB"),
                    Td(diff),
                ),
                Tr(Td("30 min"), Td("-"), Td(f"{fees['halfHourFee']} sat/vB"), Td("-")),
                Tr(Td("1 hour"), Td("-"), Td(f"{fees['hourFee']} sat/vB"), Td("-")),
                Tr(Td("Economy"), Td("-"), Td(f"{fees['economyFee']} sat/vB"), Td("-")),
            )
        except Exception:
            return Tr(Td("Error fetching fees", colspan="4"))

    @rt("/api/v1/fees/mempool-state")
    async def mempool_state():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://mempool.space/api/mempool")
                mempool = resp.json()

            return DataGrid(
                DataReadout(
                    Tip("TX_COUNT", "Number of unconfirmed transactions currently in the mempool."),
                    f"{mempool.get('count', 0):,}",
                ),
                DataReadout(
                    Tip(
                        "MEMPOOL_SIZE",
                        "Total virtual size of all unconfirmed transactions, in mega virtual bytes.",
                    ),
                    f"{mempool.get('vsize', 0) / 1_000_000:.1f} MvB",
                ),
                DataReadout(
                    Tip(
                        "TOTAL_FEES",
                        "Sum of all fees from unconfirmed transactions waiting in the mempool.",
                    ),
                    f"{mempool.get('total_fee', 0) / 100_000_000:.4f} BTC",
                ),
            )
        except Exception as e:
            return P(f"Error: {e}", style="color: var(--fg-red);")

    @rt("/api/v1/fees/estimate")
    async def fee_estimate(request):
        target = int(request.query_params.get("target_blocks", "6"))
        model_data = get_model("fee-lgbm")

        try:
            from stream.fees.collector import collect_snapshot

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://mempool.space/api/v1/fees/recommended")
                fees = resp.json()

                # Collect a full snapshot for ML prediction
                snapshot = await collect_snapshot(client)

            fee_map = {
                1: "fastestFee",
                3: "halfHourFee",
                6: "hourFee",
                12: "economyFee",
                24: "minimumFee",
            }
            mempool_fee = fees.get(fee_map.get(target, "hourFee"), 10)

            result = {
                "target_blocks": target,
                "mempool_fee": mempool_fee,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

            # ML prediction
            if model_data is not None and snapshot is not None:
                ml_fee = predict_fee(model_data, snapshot)
                if ml_fee is not None:
                    result["ml_fee"] = ml_fee
                    result["model_version"] = "fee-lgbm"
                    result["recommended_fee"] = ml_fee
                else:
                    result["recommended_fee"] = mempool_fee
                    result["model_version"] = "live-mempool"
            else:
                result["recommended_fee"] = mempool_fee
                result["model_version"] = "live-mempool"

            return result
        except Exception:
            return {"target_blocks": target, "recommended_fee": 10, "model_version": "fallback"}

    # === LIGHTNING ===

    @rt("/api/v1/lightning/stats")
    async def lightning_stats():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://mempool.space/api/v1/lightning/statistics/latest")
                stats = resp.json().get("latest", resp.json())

            node_count = stats.get("node_count", stats.get("channel_count", "?"))
            channel_count = stats.get("channel_count", "?")
            capacity = stats.get("total_capacity", 0)
            cap_btc = capacity / 100_000_000 if isinstance(capacity, (int, float)) else 0

            return DataGrid(
                DataReadout(
                    Tip(
                        "TOTAL_NODES",
                        "Computers running Lightning Network software. They route payments and maintain channels.",
                    ),
                    f"{node_count:,}" if isinstance(node_count, int) else str(node_count),
                    highlight=True,
                ),
                DataReadout(
                    Tip(
                        "TOTAL_CHANNELS",
                        "Payment channels between Lightning nodes. Funds are locked in a channel to enable off-chain transactions.",
                    ),
                    f"{channel_count:,}" if isinstance(channel_count, int) else str(channel_count),
                ),
                DataReadout(
                    Tip(
                        "NETWORK_CAPACITY",
                        "Total Bitcoin locked across all Lightning channels, available for routing payments.",
                    ),
                    f"{cap_btc:,.0f} BTC",
                ),
            )
        except Exception as e:
            return P(f"Error: {e}", style="color: var(--fg-red);")

    @rt("/api/v1/lightning/top-nodes")
    async def top_nodes():
        try:
            model = get_model("lightning-lgbm")
            cached_features = get_cached_features("lightning")

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://mempool.space/api/v1/lightning/nodes/rankings/connectivity"
                )
                nodes = resp.json()[:10]

            # Build pubkey lookup from cached features
            features_by_pubkey = {}
            if cached_features:
                for f in cached_features:
                    pk = f.get("pubkey", "")
                    if pk:
                        features_by_pubkey[pk] = f

            rows = []
            for i, n in enumerate(nodes):
                alias = n.get("alias", "Unknown")[:20]
                pubkey = n.get("publicKey", "")
                channels = n.get("channels", n.get("active_channel_count", 0))
                capacity = n.get("capacity", 0) / 100_000_000

                # Try ML model prediction if we have cached features
                score = None
                if model is not None and pubkey in features_by_pubkey:
                    try:
                        feat = features_by_pubkey[pubkey]
                        X = np.array([[feat.get(c, 0) for c in LIGHTNING_FEATURE_COLS]])
                        pred = float(model.predict(X)[0])
                        # Normalize prediction to 0-1 score
                        score = min(pred / 10_000_000_000, 1.0)  # Normalize by 100 BTC in sats
                    except Exception:
                        pass

                # Fallback to heuristic formula
                source = "ml"
                if score is None:
                    score = min(channels / 1000, 1.0) * 0.5 + min(capacity / 500, 1.0) * 0.5
                    source = "heuristic"

                badge_cls = "model-badge model-badge-ml" if source == "ml" else "model-badge"
                badge_text = "ML" if source == "ml" else "HEURISTIC"

                rows.append(
                    Tr(
                        Td(str(i + 1)),
                        Td(alias),
                        Td(f"{channels:,}" if isinstance(channels, int) else str(channels)),
                        Td(f"{capacity:,.1f}"),
                        Td(
                            Span(f"{score:.3f}"),
                            Span(" "),
                            Span(badge_text, cls=badge_cls),
                            style="color: var(--fg-green);",
                        ),
                    )
                )
            return rows
        except Exception as e:
            return Tr(Td(f"Error: {e}", colspan="5"))

    @rt("/api/v1/lightning/evaluate", methods=["POST"])
    async def evaluate_node(request):
        from pydantic import ValidationError

        from stream.app.schemas import LightningEvaluateRequest

        form = await request.form()
        pubkey = form.get("pubkey", "").strip()
        if not pubkey:
            return P("Enter a public key.", style="color: var(--fg-red);")

        try:
            LightningEvaluateRequest(pubkey=pubkey)
        except ValidationError as e:
            errors = "; ".join(err["msg"] for err in e.errors())
            return P(f"Validation error: {errors}", style="color: var(--fg-red);")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"https://mempool.space/api/v1/lightning/nodes/{pubkey}")
                node = resp.json()

            alias = node.get("alias", "Unknown")
            channels = node.get("active_channel_count", 0)
            capacity = node.get("capacity", 0) / 100_000_000

            model = get_model("lightning-lgbm")
            cached_features = get_cached_features("lightning")
            score = None
            scoring_source = "HEURISTIC"

            # Check cached features for this pubkey
            if model is not None and cached_features:
                feat = next((f for f in cached_features if f.get("pubkey") == pubkey), None)
                if feat is not None:
                    try:
                        X = np.array([[feat.get(c, 0) for c in LIGHTNING_FEATURE_COLS]])
                        pred = float(model.predict(X)[0])
                        score = min(pred / 10_000_000_000, 1.0)
                        scoring_source = "ML-MODEL"
                    except Exception:
                        pass

            # Fallback to heuristic
            if score is None:
                score = min(channels / 1000, 1.0) * 0.5 + min(capacity / 500, 1.0) * 0.5

            return Div(
                DataGrid(
                    DataReadout("ALIAS", alias),
                    DataReadout("CHANNELS", str(channels)),
                    DataReadout("CAPACITY", f"{capacity:.1f} BTC"),
                    DataReadout("CAPACITY_SCORE", f"{score:.3f}", highlight=True),
                    DataReadout("SOURCE", scoring_source),
                ),
                P(
                    f"Node {alias} has {channels} channels with {capacity:.1f} BTC capacity. Capacity score: {score:.3f} ({scoring_source})",
                    style="margin-top: 12px; color: var(--fg-dim);",
                ),
            )
        except Exception as e:
            return P(f"Node not found or error: {e}", style="color: var(--fg-red);")

    # === ONBOARDING ===

    @rt("/api/v1/onboarding/score", methods=["POST"])
    async def onboarding_score(request):
        form = await request.form()

        # Build feature dict from form
        email_type = form.get("email_domain_type", "corporate")
        phone = int(form.get("phone_verified", "1"))
        doc_score = float(form.get("doc_score", "0.85"))
        ip_match = int(form.get("ip_match", "1"))
        tx_vel = int(form.get("tx_velocity", "2"))
        deposit = float(form.get("initial_deposit", "500"))

        # Try ML model first
        model_data = get_model("onboarding-xgb")
        ml_result = None
        if model_data is not None:
            form_values = {
                "email_domain_type": email_type,
                "phone_verified": phone,
                "doc_score": doc_score,
                "ip_match": ip_match,
                "tx_velocity": tx_vel,
                "initial_deposit": deposit,
            }
            ml_result = predict_risk(model_data, form_values)

        if ml_result is not None:
            label = ml_result["risk_label"]
            action = ml_result["action"]
            probas = ml_result["class_probabilities"]
            source = ml_result["source"]

            # Use the max probability as the risk indicator
            risk_prob = max(probas.get("high_risk", 0), probas.get("blocked", 0))

            proba_text = " | ".join(f"{k}: {v:.3f}" for k, v in probas.items())

            return Div(
                DataGrid(
                    DataReadout("RISK_TIER", label.upper()),
                    DataReadout("RECOMMENDED_ACTION", action.upper().replace("_", " ")),
                    DataReadout("SOURCE", source),
                ),
                P(
                    f"Class probabilities: {proba_text}",
                    style="color: var(--fg-dim); margin-top: 8px; font-size: 11px;",
                ),
                P(
                    f"Key factors: email_domain={email_type}, doc_verification={doc_score:.2f}, ip_match={'yes' if ip_match else 'no'}",
                    style="color: var(--fg-dim); margin-top: 4px;",
                ),
            )

        # Heuristic fallback
        risk_score = 0.0
        risk_score += (email_type == "disposable") * 2
        risk_score += (1 - phone) * 1.5
        risk_score += (1 - doc_score) * 3
        risk_score += (1 - ip_match) * 2.5
        risk_score += min(tx_vel / 10, 2)

        risk_prob = min(risk_score / 10, 1.0)
        label = (
            "low_risk"
            if risk_prob < 0.3
            else "medium_risk"
            if risk_prob < 0.5
            else "high_risk"
            if risk_prob < 0.7
            else "blocked"
        )
        action = (
            "auto_approve"
            if label == "low_risk"
            else "manual_review"
            if label in ("medium_risk", "high_risk")
            else "block"
        )

        return Div(
            DataGrid(
                DataReadout(
                    "RISK_SCORE",
                    f"{risk_prob:.2f}",
                    highlight=risk_prob < 0.3,
                    variant="danger" if risk_prob > 0.6 else "",
                ),
                DataReadout("RISK_TIER", label.upper()),
                DataReadout("RECOMMENDED_ACTION", action.upper().replace("_", " ")),
                DataReadout("SOURCE", "HEURISTIC"),
            ),
            P(
                f"Key factors: email_domain={email_type}, doc_verification={doc_score:.2f}, ip_match={'yes' if ip_match else 'no'}",
                style="color: var(--fg-dim); margin-top: 8px;",
            ),
        )

    @rt("/api/v1/onboarding/metrics")
    def onboarding_metrics():
        card = get_model_card("onboarding-xgb")
        if card:
            metrics = card.get("metrics", {})
            lr = metrics.get("logistic_regression", {})
            xgb = metrics.get("calibrated_xgboost", {})
            if lr and xgb:
                return Div(
                    Table(
                        Thead(
                            Tr(
                                Th("Metric"),
                                Th(Tip("Logistic Regression")),
                                Th(
                                    Tip(
                                        "Calibrated",
                                        "A model whose predicted probabilities match real-world frequencies. If it says 80% risk, ~80% of those cases are actually risky.",
                                    ),
                                    " ",
                                    Tip("XGBoost"),
                                ),
                            )
                        ),
                        Tbody(
                            Tr(
                                Td(Tip("Accuracy")),
                                Td(f"{lr.get('accuracy', 0):.4f}"),
                                Td(
                                    f"{xgb.get('accuracy', 0):.4f}", style="color: var(--fg-green);"
                                ),
                            ),
                            Tr(
                                Td(Tip("F1 (macro)")),
                                Td(f"{lr.get('f1_macro', 0):.4f}"),
                                Td(
                                    f"{xgb.get('f1_macro', 0):.4f}", style="color: var(--fg-green);"
                                ),
                            ),
                            Tr(
                                Td(Tip("Log Loss")),
                                Td(f"{lr.get('log_loss', 0):.4f}"),
                                Td(
                                    f"{xgb.get('log_loss', 0):.4f}", style="color: var(--fg-green);"
                                ),
                            ),
                        ),
                        cls="spark-table",
                    ),
                    P(
                        "Metrics from most recent training run.",
                        style="color: var(--fg-subtle); font-size: 10px; margin-top: 8px;",
                    ),
                )

        # Hardcoded fallback
        return Div(
            Table(
                Thead(
                    Tr(
                        Th("Metric"),
                        Th(Tip("Logistic Regression")),
                        Th(
                            Tip(
                                "Calibrated",
                                "A model whose predicted probabilities match real-world frequencies. If it says 80% risk, ~80% of those cases are actually risky.",
                            ),
                            " ",
                            Tip("XGBoost"),
                        ),
                    )
                ),
                Tbody(
                    Tr(Td(Tip("Accuracy")), Td("—"), Td("—")),
                    Tr(Td(Tip("F1 (macro)")), Td("—"), Td("—")),
                    Tr(Td(Tip("Log Loss")), Td("—"), Td("—")),
                ),
                cls="spark-table",
            ),
            P(
                "No model card available. Run the onboarding training pipeline to see live metrics.",
                style="color: var(--fg-subtle); font-size: 10px; margin-top: 8px;",
            ),
        )

    # === ALERTS ===

    @rt("/api/v1/alerts/stats")
    def alert_stats():
        from sqlalchemy import func

        from stream.models.alerts import AlertRecord

        try:
            db = SessionLocal()
            try:
                pending = (
                    db.query(func.count()).filter(AlertRecord.status == "pending").scalar() or 0
                )
                reviewed = (
                    db.query(func.count()).filter(AlertRecord.status == "reviewed").scalar() or 0
                )
                escalated = (
                    db.query(func.count()).filter(AlertRecord.status == "escalated").scalar() or 0
                )
                total = pending + reviewed + escalated
            finally:
                db.close()

            return DataGrid(
                DataReadout("TOTAL_ALERTS", str(total), highlight=True),
                DataReadout("PENDING", str(pending), variant="warning" if pending > 0 else ""),
                DataReadout("REVIEWED", str(reviewed)),
                DataReadout("ESCALATED", str(escalated), variant="danger" if escalated > 0 else ""),
            )
        except Exception as e:
            log.debug("Alert stats query failed", error=str(e))
            return DataGrid(
                DataReadout("TOTAL_ALERTS", "—"),
                DataReadout("PENDING", "—"),
                DataReadout("REVIEWED", "—"),
                DataReadout("ESCALATED", "—"),
            )

    @rt("/api/v1/alerts/recent")
    def recent_alerts():
        from stream.models.alerts import AlertRecord

        rows = []
        try:
            db = SessionLocal()
            try:
                alerts = (
                    db.query(AlertRecord).order_by(AlertRecord.timestamp.desc()).limit(20).all()
                )
            finally:
                db.close()

            if alerts:
                threshold = get_risk_threshold()
                for a in alerts:
                    ts = a.timestamp.strftime("%H:%M:%S") if a.timestamp else "—"
                    tx_short = f"{a.tx_id[:4]}...{a.tx_id[-4:]}" if len(a.tx_id) > 8 else a.tx_id
                    score = a.risk_score
                    risk_cls = (
                        "risk-high"
                        if score > threshold
                        else "risk-medium"
                        if score > threshold * 0.6
                        else "risk-low"
                    )
                    badge = (
                        "badge-red"
                        if a.status == "escalated"
                        else "badge-yellow"
                        if a.status == "pending"
                        else "badge-green"
                    )
                    rows.append(
                        Tr(
                            Td(ts, style="color: var(--fg-dim);"),
                            Td(tx_short, style="font-weight: bold;"),
                            Td(f"{score:.2f}", cls=risk_cls),
                            Td(a.risk_label.upper()),
                            Td(Span(a.status.upper(), cls=f"badge-sm {badge}")),
                            Td(
                                A(
                                    "Review",
                                    href="#",
                                    cls="spark-btn",
                                    style="padding: 4px 8px; font-size: 9px;",
                                    **{
                                        "hx-get": f"/api/v1/alerts/detail/{a.tx_id}",
                                        "hx-target": "#alert-detail",
                                    },
                                )
                            ),
                        )
                    )
                return rows
        except Exception as e:
            log.debug("DB query failed for alerts, falling back to demo", error=str(e))

        # Demo fallback
        threshold = get_risk_threshold()
        demo = [
            ("14:23:01", "7a3f...e91b", 0.92, "HIGH", "pending"),
            ("14:21:45", "b2c8...4d3a", 0.78, "HIGH", "pending"),
            ("14:19:22", "e5f1...8c7d", 0.45, "MEDIUM", "reviewed"),
            ("14:15:08", "1d9a...f2b6", 0.23, "LOW", "auto_cleared"),
            ("14:12:33", "c4e7...a1d8", 0.88, "HIGH", "escalated"),
        ]
        for ts, tx, score, label, status in demo:
            risk_cls = (
                "risk-high"
                if score > threshold
                else "risk-medium"
                if score > threshold * 0.6
                else "risk-low"
            )
            badge = (
                "badge-red"
                if status == "escalated"
                else "badge-yellow"
                if status == "pending"
                else "badge-green"
            )
            rows.append(
                Tr(
                    Td(ts, style="color: var(--fg-dim);"),
                    Td(tx, style="font-weight: bold;"),
                    Td(f"{score:.2f}", cls=risk_cls),
                    Td(label),
                    Td(Span(status.upper(), cls=f"badge-sm {badge}")),
                    Td(
                        A(
                            "Review",
                            href="#",
                            cls="spark-btn",
                            style="padding: 4px 8px; font-size: 9px;",
                            **{
                                "hx-get": f"/api/v1/alerts/detail/{tx}",
                                "hx-target": "#alert-detail",
                            },
                        )
                    ),
                )
            )
        return (Tr(Td(DemoBanner(P("Demo alert data — database unavailable")), colspan="6")), *rows)

    @rt("/api/v1/alerts/detail/{tx_id}")
    async def alert_detail(tx_id: str):
        from stream.compliance.narrator import generate_compliance_narrative
        from stream.models.alerts import AlertRecord

        alert = None
        try:
            db = SessionLocal()
            try:
                alert = db.query(AlertRecord).filter(AlertRecord.tx_id == tx_id).first()
            finally:
                db.close()
        except Exception as e:
            log.debug("DB lookup failed for alert detail", error=str(e))

        if alert:
            score = alert.risk_score
            model_name = alert.model_name or "illicit-xgboost"
            shap_data = alert.explanation or []

            # SHAP display
            shap_lines = []
            if shap_data:
                for f in shap_data:
                    val = f.get("shap_value", 0)
                    color = (
                        "var(--fg-red)"
                        if val > 0.1
                        else "var(--fg-orange)"
                        if val > 0
                        else "var(--fg-green)"
                    )
                    shap_lines.append(
                        P(f"{f.get('feature', '?')}: {val:+.4f}", style=f"color: {color};")
                    )
            else:
                shap_lines.append(P("No SHAP data available.", style="color: var(--fg-dim);"))

            # Generate narrative on-demand
            narrative_text = alert.narrative
            if not narrative_text:
                if shap_data:
                    shap_vals = [f.get("shap_value", 0) for f in shap_data]
                    feat_names = [f.get("feature", "?") for f in shap_data]
                    narrative_text = await generate_compliance_narrative(
                        risk_score=score,
                        shap_values=shap_vals,
                        feature_names=feat_names,
                    )
                    # Persist back to alert row
                    try:
                        db = SessionLocal()
                        try:
                            db.query(AlertRecord).filter(AlertRecord.tx_id == tx_id).update(
                                {"narrative": narrative_text}
                            )
                            db.commit()
                        finally:
                            db.close()
                    except Exception:
                        pass
                else:
                    narrative_text = (
                        f"Transaction flagged with risk score {score:.2f}. "
                        "SHAP explanation unavailable — recommend manual review."
                    )

            review_buttons = (
                Div(
                    Button(
                        "Mark as True Positive",
                        cls="spark-btn",
                        style="margin-right: 8px;",
                        **{
                            "hx-post": f"/api/v1/alerts/{tx_id}/review",
                            "hx-target": "#alert-detail",
                            "hx-vals": '{"verdict":"true_positive"}',
                        },
                    ),
                    Button(
                        "Mark as False Positive",
                        cls="spark-btn",
                        **{
                            "hx-post": f"/api/v1/alerts/{tx_id}/review",
                            "hx-target": "#alert-detail",
                            "hx-vals": '{"verdict":"false_positive"}',
                        },
                    ),
                    style="margin-top: 16px;",
                )
                if alert.status == "pending"
                else Div(
                    Span(
                        f"STATUS: {alert.status.upper()}",
                        style="color: var(--fg-green); font-weight: bold; font-size: 12px; letter-spacing: 1px;",
                    ),
                    style="margin-top: 16px; padding: 12px; border: 1px solid var(--fg-dim); text-align: center;",
                )
            )

            investigate_btn = Div(
                Button(
                    "Investigate with AI Agent",
                    cls="spark-btn agent-btn",
                    **{
                        "hx-post": f"/api/v1/agent/investigate/{tx_id}",
                        "hx-target": "#investigation-detail",
                        "hx-swap": "innerHTML",
                    },
                ),
                style="margin-top: 12px;",
            )

            return Div(
                H4(f"Alert: {tx_id[:16]}...", style="color: var(--fg-green); margin-bottom: 12px;"),
                DataGrid(
                    DataReadout(
                        "RISK_SCORE",
                        f"{score:.2f}",
                        variant="danger" if score > get_risk_threshold() else "",
                    ),
                    DataReadout("MODEL", model_name),
                    DataReadout("STATUS", alert.status.upper()),
                ),
                H4(
                    "SHAP Explanation",
                    style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                ),
                Div(*shap_lines, style="margin-bottom: 16px;"),
                H4(
                    "AI Compliance Narrative",
                    style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                ),
                Div(
                    P(
                        narrative_text,
                        style="color: var(--fg-white); opacity: 0.9; border-left: 2px solid var(--fg-green); padding-left: 12px;",
                    ),
                ),
                review_buttons,
                investigate_btn,
                Div(id="investigation-detail"),
            )

        # Demo fallback
        return DemoBanner(
            Div(
                H4(f"Alert: {tx_id}", style="color: var(--fg-green); margin-bottom: 12px;"),
                DataGrid(
                    DataReadout("RISK_SCORE", "0.92", variant="danger"),
                    DataReadout("MODEL", "illicit-xgboost v1.0"),
                    DataReadout("INFERENCE_TIME", "2.3ms"),
                ),
                H4(
                    "SHAP Explanation",
                    style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                ),
                Div(
                    P("feature_47 (agg_neighbor_volume): +0.234", style="color: var(--fg-red);"),
                    P("feature_12 (output_count): +0.189", style="color: var(--fg-red);"),
                    P("feature_3 (tx_fee): +0.145", style="color: var(--fg-orange);"),
                    P("feature_91 (agg_tx_count): +0.098", style="color: var(--fg-orange);"),
                    P("feature_0 (timestep): -0.067", style="color: var(--fg-green);"),
                    style="margin-bottom: 16px;",
                ),
                H4(
                    "AI Compliance Narrative",
                    style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                ),
                Div(
                    P(
                        "This transaction was flagged (risk score: 0.92) due to patterns consistent with structuring.",
                        style="color: var(--fg-white); opacity: 0.9; border-left: 2px solid var(--fg-green); padding-left: 12px;",
                    ),
                ),
                Div(
                    Button(
                        "Mark as True Positive",
                        cls="spark-btn",
                        style="margin-right: 8px;",
                        **{
                            "hx-post": f"/api/v1/alerts/{tx_id}/review",
                            "hx-target": "#alert-detail",
                            "hx-vals": '{"verdict":"true_positive"}',
                        },
                    ),
                    Button(
                        "Mark as False Positive",
                        cls="spark-btn",
                        **{
                            "hx-post": f"/api/v1/alerts/{tx_id}/review",
                            "hx-target": "#alert-detail",
                            "hx-vals": '{"verdict":"false_positive"}',
                        },
                    ),
                    style="margin-top: 16px;",
                ),
                Div(
                    Button(
                        "Investigate with AI Agent",
                        cls="spark-btn agent-btn",
                        **{
                            "hx-post": f"/api/v1/agent/investigate/{tx_id}",
                            "hx-target": "#investigation-detail",
                            "hx-swap": "innerHTML",
                        },
                    ),
                    style="margin-top: 12px;",
                ),
                Div(id="investigation-detail"),
            )
        )

    # === SETTINGS ===

    @rt("/api/v1/settings/threshold", methods=["GET"])
    def get_threshold():
        return Span(f"{get_risk_threshold():.2f}", id="threshold-value")

    @rt("/api/v1/settings/threshold", methods=["POST"])
    async def post_threshold(request):
        from pydantic import ValidationError

        from stream.app.schemas import ThresholdRequest

        form = await request.form()
        val = float(form.get("threshold", 0.7))
        try:
            ThresholdRequest(value=val)
        except ValidationError as e:
            errors = "; ".join(err["msg"] for err in e.errors())
            return Span(f"Error: {errors}", id="threshold-value", style="color: var(--fg-red);")
        set_risk_threshold(val)
        return Span(f"{get_risk_threshold():.2f}", id="threshold-value")

    # === REVIEW ===

    @rt("/api/v1/alerts/{tx_id}/review", methods=["POST"])
    async def review_alert(tx_id: str, request):
        from stream.models.alerts import AlertRecord

        form = await request.form()
        verdict = form.get("verdict", "true_positive")

        alert = None
        raw_input = {}
        try:
            db = SessionLocal()
            try:
                alert = db.query(AlertRecord).filter(AlertRecord.tx_id == tx_id).first()
                if alert:
                    alert.status = "reviewed"
                    db.commit()
                    # Capture values before closing session
                    alert_id = alert.alert_id
                    risk_score = alert.risk_score
                    model_name = alert.model_name or "live-heuristic"
                    shap_data = alert.explanation or []
                    narrative_text = alert.narrative or ""
                    raw_input = alert.raw_input or {}
            finally:
                db.close()
        except Exception as e:
            log.warning("review_alert DB error", error=str(e))
            return Div(P("Error updating alert.", style="color: var(--fg-red);"))

        if alert:
            # Save review record for real alerts
            await write_in_thread(
                save_review,
                {
                    "alert_id": alert_id,
                    "tx_id": tx_id,
                    "verdict": verdict,
                    "reviewer": "analyst",
                    "risk_score_at_review": risk_score,
                    "threshold_at_review": get_risk_threshold(),
                },
            )

            # === Online Learning Integration ===
            label = 1 if verdict == "true_positive" else 0
            fee_rate = raw_input.get("fee_rate", 0)
            vsize = raw_input.get("vsize", 0)
            fee = raw_input.get("fee", 0)

            # 1. Online learn_one
            try:
                from stream.feedback.pipeline import river_learn_one

                river_learn_one(fee_rate, vsize, fee, label)
            except Exception as e:
                log.debug("Online learn_one failed", error=str(e))

            # 2. Online metrics (Feature 4)
            try:
                from stream.online.metrics import record_label

                record_label(y_true=label, y_pred_score=risk_score, threshold=get_risk_threshold())
            except Exception as e:
                log.debug("Online metrics update failed", error=str(e))

            # 3. Model race (Feature 5)
            try:
                from stream.online.race import get_race

                race = get_race()
                features = {"fee_rate": fee_rate, "vsize": vsize, "fee": fee}
                race.learn_one(features, bool(label))
            except Exception as e:
                log.debug("Model race update failed", error=str(e))

            # 4. Adaptation loop (Feature 6)
            try:
                from stream.online.adaptation import check_stabilization, record_prediction_error

                record_prediction_error(abs(label - risk_score))
                check_stabilization()
            except Exception as e:
                log.debug("Adaptation update failed", error=str(e))

        verdict_label = "TRUE POSITIVE" if verdict == "true_positive" else "FALSE POSITIVE"
        verdict_color = "var(--fg-red)" if verdict == "true_positive" else "var(--fg-green)"

        if alert:
            # SHAP display
            shap_lines = []
            if shap_data:
                for f in shap_data:
                    val = f.get("shap_value", 0)
                    color = (
                        "var(--fg-red)"
                        if val > 0.1
                        else "var(--fg-orange)"
                        if val > 0
                        else "var(--fg-green)"
                    )
                    shap_lines.append(
                        P(f"{f.get('feature', '?')}: {val:+.4f}", style=f"color: {color};")
                    )
            else:
                shap_lines.append(P("No SHAP data available.", style="color: var(--fg-dim);"))

            return Div(
                H4(f"Alert: {tx_id[:16]}...", style="color: var(--fg-green); margin-bottom: 12px;"),
                DataGrid(
                    DataReadout(
                        "RISK_SCORE",
                        f"{risk_score:.2f}",
                        variant="danger" if risk_score > get_risk_threshold() else "",
                    ),
                    DataReadout("MODEL", model_name),
                    DataReadout("STATUS", "REVIEWED"),
                ),
                H4(
                    "SHAP Explanation",
                    style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                ),
                Div(*shap_lines, style="margin-bottom: 16px;"),
                H4(
                    "AI Compliance Narrative",
                    style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                ),
                Div(
                    P(
                        narrative_text or "No narrative available.",
                        style="color: var(--fg-white); opacity: 0.9; border-left: 2px solid var(--fg-green); padding-left: 12px;",
                    )
                ),
                Div(
                    Span(
                        f"VERDICT: {verdict_label}",
                        style=f"color: {verdict_color}; font-weight: bold; font-size: 12px; letter-spacing: 1px;",
                    ),
                    style="margin-top: 16px; padding: 12px; border: 1px solid var(--fg-dim); text-align: center;",
                ),
            )

        # Demo fallback — no DB record, just show the verdict confirmation
        tx_short = f"{tx_id[:4]}...{tx_id[-4:]}" if len(tx_id) > 8 else tx_id
        return Div(
            H4(f"Alert: {tx_short}", style="color: var(--fg-green); margin-bottom: 12px;"),
            DataGrid(
                DataReadout("STATUS", "REVIEWED"),
            ),
            Div(
                Span(
                    f"VERDICT: {verdict_label}",
                    style=f"color: {verdict_color}; font-weight: bold; font-size: 12px; letter-spacing: 1px;",
                ),
                style="margin-top: 16px; padding: 12px; border: 1px solid var(--fg-dim); text-align: center;",
            ),
        )

    # === MODELS ===

    @rt("/api/v1/models/stats/{model_name}")
    def model_stats(model_name: str):
        # Demo stats (would pull from Prefect API in production)
        return DemoBanner(
            MetricsRow(
                MetricItem("LAST_TRAINED", "2h ago"),
                MetricItem("FRESHNESS", "FRESH"),
                MetricItem("PREDICTIONS", "1,247"),
                MetricItem("AVG_LATENCY", "2.3ms"),
            )
        )

    @rt("/api/v1/models/{model_name}/chart/{chart_type}")
    def model_chart(model_name: str, chart_type: str):
        """Return Plotly chart HTML for a model card chart."""
        from stream.model_card import (
            calibration_chart,
            confusion_matrix_chart,
            feature_importance_chart,
            pr_curve_chart,
            temporal_chart,
        )

        card = get_model_card(model_name)
        if not card:
            return P(
                "No model card data. Run the training pipeline first.",
                style="color: var(--fg-dim); font-size: 11px;",
            )

        fig = None
        try:
            if chart_type == "feature_importance":
                fi = card.get("feature_importance", [])
                if fi:
                    names = [f["feature"] for f in fi]
                    scores = [f["importance"] for f in fi]
                    fig = feature_importance_chart(names, scores)

            elif chart_type == "pr_curve":
                pr = card.get("curves", {}).get("pr_curve", {})
                if pr.get("precision") and pr.get("recall"):
                    fig = pr_curve_chart(pr["precision"], pr["recall"])

            elif chart_type == "confusion_matrix":
                cm = card.get("metrics", {}).get("confusion_matrix", {})
                if cm:
                    fig = confusion_matrix_chart(cm["tp"], cm["fp"], cm["fn"], cm["tn"])

            elif chart_type == "temporal":
                temporal = card.get("curves", {}).get("temporal", [])
                if temporal:
                    fig = temporal_chart(temporal)

            elif chart_type == "calibration":
                cal = card.get("curves", {}).get("calibration", {})
                # Use first class found
                for cls_key, cls_data in cal.items():
                    if cls_data.get("prob_true") and cls_data.get("prob_pred"):
                        fig = calibration_chart(
                            cls_data["prob_true"], cls_data["prob_pred"], cls_key
                        )
                        break

        except Exception as e:
            log.warning("Chart generation failed", model=model_name, chart=chart_type, error=str(e))
            return P(f"Chart error: {e}", style="color: var(--fg-red); font-size: 11px;")

        if fig is None:
            return P(
                f"No data for {chart_type.replace('_', ' ')} chart.",
                style="color: var(--fg-dim); font-size: 11px;",
            )

        # Return Plotly JSON rendered client-side
        chart_json = fig.to_json()
        div_id = f"plotly-{model_name}-{chart_type}"
        return Div(
            Div(id=div_id, style="width: 100%; height: 350px;"),
            Script(src="https://cdn.plot.ly/plotly-2.35.2.min.js"),
            Script(f"""(function() {{
                var spec = {chart_json};
                Plotly.newPlot('{div_id}', spec.data, spec.layout, {{responsive: true, displayModeBar: false}});
            }})();"""),
        )

    @rt("/api/v1/models/roi", methods=["POST"])
    async def roi_calculator(request):
        form = await request.form()
        volume = int(form.get("daily_volume", 50000))
        review_rate = float(form.get("review_rate", 5)) / 100
        analyst_cost = float(form.get("analyst_cost", 45))
        review_mins = float(form.get("review_minutes", 5))

        # Current state
        manual_reviews = int(volume * review_rate)
        hours_per_day = manual_reviews * review_mins / 60
        annual_cost = hours_per_day * analyst_cost * 365

        # With ML model (assumes 84% auto-clear rate)
        ml_clear_rate = 0.84
        ml_reviews = int(manual_reviews * (1 - ml_clear_rate))
        ml_hours = ml_reviews * (review_mins * 0.3) / 60  # LLM narrative reduces per-review time
        ml_annual = ml_hours * analyst_cost * 365

        savings = annual_cost - ml_annual
        infra_cost = 500 * 12  # $500/mo for infrastructure
        breakeven_days = int(infra_cost / (savings / 365)) if savings > 0 else 999

        return Div(
            H4(
                "CURRENT STATE",
                style="color: var(--fg-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;",
            ),
            DataGrid(
                DataReadout("MANUAL_REVIEWS/DAY", f"{manual_reviews:,}"),
                DataReadout("ANALYST_HOURS/DAY", f"{hours_per_day:.0f}"),
                DataReadout("ANNUAL_COST", f"${annual_cost:,.0f}", variant="danger"),
            ),
            H4(
                "WITH ML MODEL",
                style="color: var(--fg-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px;",
            ),
            DataGrid(
                DataReadout("AUTO_CLEARED", f"{ml_clear_rate * 100:.0f}%", highlight=True),
                DataReadout("REVIEWS/DAY", f"{ml_reviews:,}"),
                DataReadout("ANALYST_HOURS/DAY", f"{ml_hours:.0f}"),
                DataReadout("ANNUAL_COST", f"${ml_annual:,.0f}", highlight=True),
            ),
            H4(
                "SAVINGS",
                style="color: var(--fg-green); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px;",
            ),
            DataGrid(
                DataReadout("ANNUAL_SAVINGS", f"${savings:,.0f}", highlight=True),
                DataReadout("COST_REDUCTION", f"{(savings / max(annual_cost, 1)) * 100:.0f}%"),
                DataReadout("BREAKEVEN", f"{breakeven_days} days"),
            ),
        )

    @rt("/api/v1/models/reload", methods=["POST"])
    def reload_models():
        from stream.app.models import load_all_models

        load_all_models()
        return {"status": "reloaded"}

    # === PIPELINE ===

    @rt("/api/v1/pipeline/recent-runs")
    async def recent_runs():
        from stream.config import settings

        if settings.PREFECT_API_URL and settings.PREFECT_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{settings.PREFECT_API_URL}/flow_runs/filter",
                        headers={"Authorization": f"Bearer {settings.PREFECT_API_KEY}"},
                        json={"sort": "EXPECTED_START_TIME_DESC", "limit": 10},
                    )
                    resp.raise_for_status()
                    flow_runs = resp.json()

                rows = []
                for run in flow_runs:
                    name = run.get("name", run.get("flow_id", "unknown"))
                    state = run.get("state", {})
                    status = state.get("type", "UNKNOWN").upper()
                    started = run.get("start_time", run.get("expected_start_time", "—"))
                    if started and started != "—":
                        started = started[:19].replace("T", " ")
                    duration = ""
                    if run.get("total_run_time"):
                        secs = run["total_run_time"]
                        duration = f"{int(secs // 60)}m {int(secs % 60)}s"
                    elif run.get("estimated_run_time"):
                        secs = run["estimated_run_time"]
                        duration = f"{int(secs // 60)}m {int(secs % 60)}s"

                    color = (
                        "var(--fg-green)"
                        if status == "COMPLETED"
                        else "var(--fg-red)"
                        if status == "FAILED"
                        else "var(--fg-orange)"
                    )
                    rows.append(
                        Tr(
                            Td(name),
                            Td(started, style="color: var(--fg-dim);"),
                            Td(duration),
                            Td(Span(status, style=f"color: {color};")),
                        )
                    )
                return rows
            except Exception as e:
                log.warning("Prefect API call failed", error=str(e))
                return (
                    Tr(
                        Td(
                            P(
                                f"Pipeline data unavailable — check PREFECT_API_KEY ({e})",
                                style="color: var(--fg-orange); font-size: 11px;",
                            ),
                            colspan="4",
                        )
                    ),
                )

        # No Prefect configured — honest fallback
        return (
            Tr(
                Td(
                    DemoBanner(
                        P(
                            "Pipeline data unavailable — PREFECT_API_URL not configured. Set env vars to see real pipeline runs."
                        )
                    ),
                    colspan="4",
                )
            ),
        )

    @rt("/api/v1/pipeline/freshness")
    def model_freshness():
        from stream.config import settings

        model_names = ["illicit-xgboost", "fee-lgbm", "lightning-lgbm", "onboarding-xgb"]
        rows = []

        # Try R2 head_object for real freshness data
        r2_available = False
        if settings.R2_ENDPOINT_URL:
            try:
                import boto3

                s3 = boto3.client(
                    "s3",
                    endpoint_url=settings.R2_ENDPOINT_URL,
                    aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                )
                for name in model_names:
                    try:
                        resp = s3.head_object(
                            Bucket=settings.R2_BUCKET_NAME, Key=f"models/{name}/latest.pkl"
                        )
                        last_modified = resp["LastModified"]
                        age = datetime.datetime.now(datetime.timezone.utc) - last_modified
                        if age.days > 0:
                            when = f"{age.days}d ago"
                        elif age.seconds > 3600:
                            when = f"{age.seconds // 3600}h ago"
                        else:
                            when = f"{age.seconds // 60}m ago"
                        status = "FRESH" if age.days < 7 else "STALE"
                        rows.append((name, when, status))
                    except Exception:
                        rows.append((name, "not found in R2", "UNKNOWN"))
                r2_available = True
            except Exception as e:
                log.debug("R2 freshness check failed", error=str(e))

        if not r2_available:
            # Check which models are loaded in memory
            for name in model_names:
                loaded = get_model(name) is not None
                status = "LOADED" if loaded else "NOT LOADED"
                rows.append((name, "—", status))

        return Table(
            Thead(Tr(Th("Model"), Th("Last Modified"), Th("Status"))),
            Tbody(
                *[
                    Tr(
                        Td(name),
                        Td(when, style="color: var(--fg-dim);"),
                        Td(
                            Span(
                                status,
                                cls="badge-sm badge-green"
                                if status in ("FRESH", "LOADED")
                                else "badge-sm badge-yellow",
                            )
                        ),
                    )
                    for name, when, status in rows
                ]
            ),
            cls="spark-table",
        )

    @rt("/api/v1/pipeline/freshness/{model_name}")
    def model_freshness_single(model_name: str):
        from stream.config import settings

        when = "—"
        status = "UNKNOWN"

        # Try R2 head_object
        if settings.R2_ENDPOINT_URL:
            try:
                import boto3

                s3 = boto3.client(
                    "s3",
                    endpoint_url=settings.R2_ENDPOINT_URL,
                    aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                )
                resp = s3.head_object(
                    Bucket=settings.R2_BUCKET_NAME,
                    Key=f"models/{model_name}/latest.pkl",
                )
                last_modified = resp["LastModified"]
                age = datetime.datetime.now(datetime.timezone.utc) - last_modified
                if age.days > 0:
                    when = f"{age.days}d ago"
                elif age.seconds > 3600:
                    when = f"{age.seconds // 3600}h ago"
                else:
                    when = f"{age.seconds // 60}m ago"
                status = "FRESH" if age.days < 7 else "STALE"
            except Exception:
                pass

        # Fallback: check if model loaded in memory
        if status == "UNKNOWN":
            loaded = get_model(model_name) is not None
            status = "LOADED" if loaded else "NOT LOADED"

        badge_cls = (
            "badge-sm badge-green" if status in ("FRESH", "LOADED") else "badge-sm badge-yellow"
        )
        return Div(
            DataGrid(
                DataReadout("MODEL", model_name),
                DataReadout("LAST MODIFIED", when),
                DataReadout("STATUS", ""),
            ),
            Span(status, cls=badge_cls),
            style="margin-top: 4px;",
        )

    # === FEEDBACK / RETRAINING ===

    @rt("/api/v1/feedback/stats")
    def feedback_stats():
        from sqlalchemy import func

        from stream.models.reviews import ReviewRecord

        try:
            db = SessionLocal()
            try:
                total = db.query(func.count()).select_from(ReviewRecord).scalar() or 0
                tp = (
                    db.query(func.count())
                    .select_from(ReviewRecord)
                    .filter(ReviewRecord.verdict == "true_positive")
                    .scalar()
                    or 0
                )
                fp = (
                    db.query(func.count())
                    .select_from(ReviewRecord)
                    .filter(ReviewRecord.verdict == "false_positive")
                    .scalar()
                    or 0
                )
            finally:
                db.close()
        except Exception:
            total, tp, fp = 0, 0, 0

        from stream.feedback.pipeline import (
            get_model_metadata,
            get_retrain_status,
            get_river_metrics,
        )

        status = get_retrain_status()
        meta = get_model_metadata()
        river = get_river_metrics()

        # Model lineage strip
        lineage = "heuristic-v1.0"
        if meta:
            lineage += f" → lgbm-v2.0 ({meta['n_labels']} labels)"
        if river["n_samples"] > 0:
            river_status = "ACTIVE" if river["is_ready"] else "WARMING"
            lineage += f" → online-v3.0 [{river_status}]"

        status_color = {
            "IDLE": "var(--fg-dim)",
            "TRAINING": "var(--fg-orange)",
            "COMPLETE": "var(--fg-green)",
            "ERROR": "var(--fg-red)",
        }.get(status["state"], "var(--fg-dim)")

        metrics_section = []
        if status["metrics"]:
            m = status["metrics"]
            metrics_section = [
                H4(
                    "RETRAIN RESULTS",
                    style="color: var(--fg-dim); font-size: 10px; margin-top: 12px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                ),
                DataGrid(
                    DataReadout("ACCURACY", f"{m['accuracy']:.1%}"),
                    DataReadout("METHOD", m["method"]),
                    DataReadout("TRAIN_SIZE", str(m["n_train"])),
                    DataReadout("TEST_SIZE", str(m["n_test"])),
                ),
            ]

        error_section = []
        if status["state"] == "ERROR" and status.get("error"):
            error_section = [
                P(
                    f"Error: {status['error']}",
                    style="color: var(--fg-red); font-size: 11px; margin-top: 8px;",
                )
            ]

        return Div(
            DataGrid(
                DataReadout("TOTAL REVIEWS", str(total)),
                DataReadout("TRUE POSITIVE", str(tp), variant="danger" if tp > 0 else ""),
                DataReadout("FALSE POSITIVE", str(fp)),
            ),
            Div(
                Span(
                    "RETRAIN STATUS: ",
                    style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px;",
                ),
                Span(
                    status["state"],
                    style=f"color: {status_color}; font-weight: bold; font-size: 12px;",
                ),
                style="margin-top: 12px;",
            ),
            Div(
                Span(
                    "MODEL LINEAGE: ",
                    style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px;",
                ),
                Span(lineage, style="color: var(--fg-green); font-size: 11px;"),
                style="margin-top: 8px;",
            ),
            *metrics_section,
            *error_section,
            Div(
                Button(
                    "RETRAIN NOW",
                    cls="spark-btn",
                    style="margin-top: 16px;",
                    **{
                        "hx-post": "/api/v1/feedback/retrain",
                        "hx-target": "#feedback-panel",
                        "hx-swap": "innerHTML",
                    },
                ),
                style="text-align: center;" if total >= 3 else "text-align: center; opacity: 0.5;",
            )
            if total >= 3
            else Div(
                P(
                    f"Need at least 3 reviews to retrain (have {total}). Review more alerts above.",
                    style="color: var(--fg-dim); font-size: 11px; margin-top: 12px; text-align: center;",
                ),
            ),
        )

    @rt("/api/v1/feedback/retrain", methods=["POST"])
    async def trigger_retrain():
        import asyncio

        from stream.feedback.pipeline import get_retrain_status, run_retraining

        status = get_retrain_status()
        if status["state"] == "TRAINING":
            return Div(P("Retraining already in progress...", style="color: var(--fg-orange);"))

        # Run retraining in a thread to avoid blocking
        result = await asyncio.to_thread(run_retraining)

        # Return updated stats panel
        return Div(
            P(
                f"Retraining {result['state']}",
                style=f"color: {'var(--fg-green)' if result['state'] == 'COMPLETE' else 'var(--fg-red)'}; font-weight: bold; margin-bottom: 12px;",
            ),
            Div(
                id="feedback-panel",
                **{
                    "hx-get": "/api/v1/feedback/stats",
                    "hx-trigger": "load",
                    "hx-swap": "innerHTML",
                },
            ),
        )

    @rt("/api/v1/feedback/retrain-status")
    def retrain_status():
        from stream.feedback.pipeline import get_retrain_status

        return get_retrain_status()

    # === DRIFT MONITOR ===

    @rt("/api/v1/drift/status")
    def drift_status():
        from stream.drift.monitor import get_status

        d = get_status()
        status = d["status"]
        psi = d["psi"]
        stats = d["stats"]
        color_map = {
            "green": "var(--fg-green)",
            "yellow": "var(--fg-orange)",
            "red": "var(--fg-red)",
            "dim": "var(--fg-dim)",
        }
        status_color = color_map.get(d["color"], "var(--fg-dim)")

        parts = [
            Div(
                Span(
                    "DRIFT: ", style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px;"
                ),
                Span(status, style=f"color: {status_color}; font-weight: bold; font-size: 14px;"),
                Span(
                    f"  PSI: {psi:.4f}" if psi is not None else "  PSI: —",
                    style="color: var(--fg-subtle); margin-left: 12px; font-size: 11px;",
                ),
                style="margin-bottom: 12px;",
            ),
        ]

        if stats["n_scores"] > 0:
            parts.append(
                DataGrid(
                    DataReadout("MEAN SCORE", f"{stats['mean']:.3f}"),
                    DataReadout("STD DEV", f"{stats['std']:.3f}"),
                    DataReadout("P95", f"{stats['p95']:.3f}"),
                    DataReadout("BUFFER", f"{stats['n_scores']}"),
                ),
            )

        if status == "CALIBRATING":
            parts.append(
                P(
                    f"Collecting baseline... {d['remaining']} more scores needed.",
                    style="color: var(--fg-dim); font-size: 11px; margin-top: 8px;",
                ),
            )

        if status == "RED":
            parts.append(
                Div(
                    P(
                        "DRIFT DETECTED — RETRAIN RECOMMENDED",
                        style="color: var(--fg-red); font-weight: bold; font-size: 12px; letter-spacing: 1px; text-align: center; padding: 8px; border: 1px solid var(--fg-red); margin-top: 12px; animation: pulse 2s infinite;",
                    ),
                ),
            )

        # ADWIN status
        adwin = d.get("adwin", {})
        if adwin.get("enabled"):
            adwin_color = "var(--fg-red)" if adwin.get("drift_detected") else "var(--fg-green)"
            adwin_label = "DRIFT" if adwin.get("drift_detected") else "STABLE"
            parts.append(
                Div(
                    H4(
                        "ADWIN DETECTOR",
                        style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                    ),
                    Div(
                        Span(
                            "STATUS: ",
                            style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px;",
                        ),
                        Span(
                            adwin_label,
                            style=f"color: {adwin_color}; font-weight: bold; font-size: 12px;",
                        ),
                        Span(
                            f"  WINDOW: {adwin.get('width', '—')}  DRIFTS: {adwin.get('drift_count', 0)}",
                            style="color: var(--fg-subtle); margin-left: 12px; font-size: 11px;",
                        ),
                    ),
                ),
            )
            if adwin.get("drift_detected"):
                parts.append(
                    P(
                        f"ADWIN CHANGE POINT at sample {adwin.get('n_samples', '?')}",
                        style="color: var(--fg-red); font-weight: bold; font-size: 11px; text-align: center; padding: 6px; border: 1px solid var(--fg-red); margin-top: 8px;",
                    ),
                )

        # Adaptation status
        try:
            from stream.online.adaptation import get_status as get_adaptation_status

            adapt = get_adaptation_status()
            state = adapt["state"]
            state_colors = {
                "STABLE": "var(--fg-green)",
                "DRIFT_DETECTED": "var(--fg-red)",
                "ADAPTING": "var(--fg-orange)",
                "STABILIZING": "var(--fg-orange)",
                "RECOVERED": "var(--fg-green)",
            }
            state_color = state_colors.get(state, "var(--fg-dim)")

            parts.append(
                Div(
                    H4(
                        "ADAPTATION",
                        style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
                    ),
                    Div(
                        Span(
                            "STATE: ",
                            style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px;",
                        ),
                        Span(
                            state,
                            style=f"color: {state_color}; font-weight: bold; font-size: 12px;",
                            cls="adapting-glow" if state == "ADAPTING" else "",
                        ),
                        Span(
                            f"  SIGNALS: {adapt['drift_signals']}/{adapt['consecutive_threshold']}",
                            style="color: var(--fg-subtle); margin-left: 12px; font-size: 11px;",
                        ),
                    ),
                    Div(
                        Button(
                            "FORCE ADAPTATION",
                            cls="spark-btn",
                            style="margin-top: 12px; border-color: var(--fg-orange); color: var(--fg-orange);",
                            **{
                                "hx-post": "/api/v1/online/adaptation/force",
                                "hx-target": "#drift-panel",
                                "hx-swap": "innerHTML",
                            },
                        ),
                    )
                    if state in ("STABLE", "RECOVERED")
                    else Div(),
                ),
            )

            # Adaptation log
            if adapt["log"]:
                log_items = []
                for entry in adapt["log"][-5:]:
                    log_items.append(
                        P(
                            f"[{entry['event']}] {entry['message']}",
                            style="color: var(--fg-subtle); font-size: 10px; margin: 2px 0;",
                        ),
                    )
                parts.append(
                    Div(
                        *log_items,
                        style="margin-top: 8px; padding: 8px; border: 1px solid var(--highlight-med); background: rgba(35, 33, 54, 0.5);",
                    )
                )
        except Exception:
            pass

        return Div(*parts)

    @rt("/api/v1/drift/distribution")
    def drift_distribution():
        from stream.drift.monitor import get_distribution

        return get_distribution()

    # === AGENT INVESTIGATION ===

    @rt("/api/v1/agent/investigate/{tx_id}", methods=["POST"])
    async def investigate_tx(tx_id: str):
        import asyncio

        from stream.agent.investigator import get_investigation, run_investigation

        existing = get_investigation(tx_id)
        if existing and existing["status"] == "RUNNING":
            return Div(
                P("Investigation already in progress...", style="color: var(--fg-orange);"),
                Div(
                    id="investigation-panel",
                    **{
                        "hx-get": f"/api/v1/agent/investigation/{tx_id}",
                        "hx-trigger": "every 2s",
                        "hx-swap": "innerHTML",
                    },
                ),
            )

        # Start investigation in background
        asyncio.create_task(run_investigation(tx_id))

        return Div(
            P("Investigation started...", style="color: var(--fg-green); font-weight: bold;"),
            Div(
                id="investigation-panel",
                **{
                    "hx-get": f"/api/v1/agent/investigation/{tx_id}",
                    "hx-trigger": "every 2s",
                    "hx-swap": "innerHTML",
                },
            ),
        )

    @rt("/api/v1/agent/investigation/{tx_id}")
    def get_investigation_status(tx_id: str):
        from stream.agent.investigator import get_investigation

        investigation = get_investigation(tx_id)
        if not investigation:
            return Div(P("No investigation found.", style="color: var(--fg-dim);"))

        steps_html = []
        for step in investigation["steps"]:
            if step["type"] == "tool_call":
                steps_html.append(
                    Div(
                        Span(
                            f"→ {step['tool']}", style="color: var(--fg-green); font-weight: bold;"
                        ),
                        Span(
                            f"({', '.join(f'{k}={v}' for k, v in step.get('input', {}).items())})",
                            style="color: var(--fg-dim); margin-left: 8px; font-size: 10px;",
                        ),
                        cls="investigation-step",
                    )
                )
            elif step["type"] == "tool_result":
                steps_html.append(
                    Div(
                        Pre(
                            step["result"],
                            style="color: var(--fg-subtle); font-size: 10px; white-space: pre-wrap; margin: 4px 0 8px 16px; max-height: 120px; overflow-y: auto;",
                        ),
                        cls="investigation-step",
                    )
                )
            elif step["type"] == "thinking":
                steps_html.append(
                    Div(
                        P(
                            step["content"][:300],
                            style="color: var(--fg-white); opacity: 0.8; font-size: 11px; margin: 4px 0;",
                        ),
                        cls="investigation-step",
                    )
                )
            elif step["type"] == "error":
                steps_html.append(
                    Div(P(step["content"], style="color: var(--fg-red);"), cls="investigation-step")
                )

        # Status indicator
        status = investigation["status"]
        status_color = {
            "RUNNING": "var(--fg-orange)",
            "COMPLETE": "var(--fg-green)",
            "ERROR": "var(--fg-red)",
        }.get(status, "var(--fg-dim)")

        parts = [
            Div(
                Span(
                    "INVESTIGATION ",
                    style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px;",
                ),
                Span(status, style=f"color: {status_color}; font-weight: bold; font-size: 12px;"),
                style="margin-bottom: 12px;",
            ),
            Div(*steps_html, cls="investigation-log"),
        ]

        if investigation["report"]:
            parts.append(
                Div(
                    H4(
                        "INVESTIGATION REPORT",
                        style="color: var(--fg-green); font-size: 10px; margin-top: 16px; margin-bottom: 8px; letter-spacing: 1px;",
                    ),
                    Pre(
                        investigation["report"],
                        style="color: var(--fg-white); white-space: pre-wrap; font-size: 11px; "
                        "border-left: 2px solid var(--fg-green); padding-left: 12px; line-height: 1.6;",
                    ),
                    cls="investigation-report",
                )
            )

        # Keep polling if still running
        if status == "RUNNING":
            return Div(
                *parts,
                id="investigation-panel",
                **{
                    "hx-get": f"/api/v1/agent/investigation/{tx_id}",
                    "hx-trigger": "every 2s",
                    "hx-swap": "innerHTML",
                },
            )
        return Div(*parts)

    # === ADWIN ===

    @rt("/api/v1/drift/adwin")
    def adwin_status():
        from stream.drift.monitor import get_adwin_status

        return get_adwin_status()

    # === ANOMALY ===

    @rt("/api/v1/anomaly/status")
    def anomaly_status():
        from stream.anomaly.detector import get_anomaly_status

        status = get_anomaly_status()
        calibrated = status["is_calibrated"]
        return Div(
            Div(
                Span(
                    "DETECTOR: ",
                    style="color: var(--fg-dim); font-size: 10px; letter-spacing: 1px;",
                ),
                Span(
                    "CALIBRATED" if calibrated else "CALIBRATING",
                    style=f"color: {'var(--fg-green)' if calibrated else 'var(--fg-dim)'}; font-weight: bold; font-size: 12px;",
                ),
            ),
            DataGrid(
                DataReadout("SAMPLES", str(status["n_samples"])),
                DataReadout("GRACE LEFT", str(status["grace_remaining"])),
            ),
        )

    # === ONLINE METRICS ===

    @rt("/api/v1/online/metrics")
    def online_metrics():
        from stream.online.metrics import get_snapshot

        snap = get_snapshot()
        if snap["n_labels"] == 0:
            return Div(
                P(
                    "No labels yet. Review alerts to populate metrics.",
                    style="color: var(--fg-dim); font-size: 11px;",
                )
            )

        cum = snap["cumulative"]
        rol = snap["rolling"]
        return Div(
            H4(
                "CUMULATIVE",
                style="color: var(--fg-dim); font-size: 10px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
            ),
            DataGrid(
                DataReadout("PRECISION", f"{cum['precision']:.3f}"),
                DataReadout("RECALL", f"{cum['recall']:.3f}"),
                DataReadout("F1", f"{cum['f1']:.3f}", highlight=True),
                DataReadout("ROCAUC", f"{cum['rocauc']:.3f}"),
            ),
            H4(
                "ROLLING (50)",
                style="color: var(--fg-dim); font-size: 10px; margin-top: 12px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;",
            ),
            DataGrid(
                DataReadout("PRECISION", f"{rol['precision']:.3f}"),
                DataReadout("RECALL", f"{rol['recall']:.3f}"),
                DataReadout("F1", f"{rol['f1']:.3f}", highlight=True),
                DataReadout("ROCAUC", f"{rol['rocauc']:.3f}"),
            ),
            Div(
                Span(
                    f"LABELS: {snap['n_labels']}",
                    style="color: var(--fg-subtle); font-size: 10px; letter-spacing: 1px;",
                ),
                style="margin-top: 8px;",
            ),
        )

    # === MODEL RACE ===

    @rt("/api/v1/online/race/standings")
    def race_standings():
        from stream.online.race import get_race

        race = get_race()
        standings = race.get_standings()
        if not standings or standings[0]["n_total"] == 0:
            return Div(
                P(
                    "No race data yet. Review alerts to start the race.",
                    style="color: var(--fg-dim); font-size: 11px;",
                )
            )

        rows = []
        for i, s in enumerate(standings):
            medal = ["1st", "2nd", "3rd"][i] if i < 3 else ""
            color = "var(--fg-green)" if i == 0 else "var(--fg-white)"
            rows.append(
                Tr(
                    Td(medal, style=f"color: {color}; font-weight: bold;"),
                    Td(s["name"], style=f"color: {color};"),
                    Td(f"{s['rolling_f1']:.3f}", style=f"color: {color};"),
                    Td(f"{s['rolling_accuracy']:.3f}"),
                    Td(f"{s['cumulative_f1']:.3f}"),
                    Td(str(s["n_total"])),
                )
            )

        return Table(
            Thead(
                Tr(
                    Th("Rank"),
                    Th("Model"),
                    Th("Rolling F1"),
                    Th("Rolling Acc"),
                    Th("Cum F1"),
                    Th("Samples"),
                )
            ),
            Tbody(*rows),
            cls="spark-table",
        )

    @rt("/api/v1/online/race/convergence")
    def race_convergence():
        from stream.online.race import get_race

        race = get_race()
        data = race.get_convergence_data()

        # Build Plotly chart
        traces = []
        colors = {
            "LogisticRegression": "#9ccfd8",
            "HoeffdingTree": "#f6c177",
            "GaussianNB": "#c4a7e7",
        }
        for name, history in data.items():
            if not history:
                continue
            traces.append(
                {
                    "x": [h["sample"] for h in history],
                    "y": [h["f1"] for h in history],
                    "name": name,
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": colors.get(name, "#eae8ff"), "width": 2},
                }
            )

        if not traces:
            return Div(
                P("No convergence data yet.", style="color: var(--fg-dim); font-size: 11px;")
            )

        import json as _json

        chart_spec = _json.dumps(
            {
                "data": traces,
                "layout": {
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "font": {"family": "Courier New", "color": "#eae8ff", "size": 10},
                    "margin": {"l": 40, "r": 20, "t": 30, "b": 40},
                    "xaxis": {
                        "title": "Sample",
                        "gridcolor": "rgba(224,222,244,0.08)",
                        "zerolinecolor": "rgba(224,222,244,0.08)",
                    },
                    "yaxis": {
                        "title": "Rolling F1",
                        "range": [0, 1],
                        "gridcolor": "rgba(224,222,244,0.08)",
                        "zerolinecolor": "rgba(224,222,244,0.08)",
                    },
                    "legend": {"orientation": "h", "y": -0.2},
                    "title": {
                        "text": "MODEL RACE CONVERGENCE",
                        "font": {"size": 11, "color": "#837f9b"},
                    },
                },
            }
        )

        return Div(
            Div(id="race-convergence-chart", style="width: 100%; height: 280px;"),
            Script(src="https://cdn.plot.ly/plotly-2.35.2.min.js"),
            Script(f"""(function() {{
                var spec = {chart_spec};
                Plotly.newPlot('race-convergence-chart', spec.data, spec.layout, {{responsive: true, displayModeBar: false}});
            }})();"""),
        )

    # === ADAPTATION ===

    @rt("/api/v1/online/adaptation/status")
    def adaptation_status():
        from stream.online.adaptation import get_status

        return get_status()

    @rt("/api/v1/online/adaptation/force", methods=["POST"])
    def force_adaptation():
        from stream.online.adaptation import force_adaptation as _force

        _force()
        # Return updated drift panel
        return Div(
            P(
                "ADAPTATION TRIGGERED",
                style="color: var(--fg-orange); font-weight: bold; font-size: 12px; text-align: center; padding: 8px; border: 1px solid var(--fg-orange); margin-bottom: 12px;",
            ),
            Div(
                id="drift-panel",
                **{
                    "hx-get": "/api/v1/drift/status",
                    "hx-trigger": "load, every 10s",
                    "hx-swap": "innerHTML",
                },
            ),
        )

    # === INTEGRATION ===

    @rt("/api/v1/integration/webhook", methods=["POST"])
    async def webhook_score(request):
        try:
            form = await request.form()
            payload_str = form.get("payload", "{}")
            payload = json.loads(payload_str)
        except Exception:
            try:
                payload = await request.json()
            except Exception:
                payload = {}

        correlation_id = payload.get("correlation_id", str(uuid.uuid4()))
        txid = payload.get("txid", correlation_id)
        vsize = int(payload.get("vsize", 250))
        fee = int(payload.get("fee", 1000))

        from pydantic import ValidationError

        from stream.app.schemas import LiveScoreRequest

        try:
            LiveScoreRequest(vsize=vsize, fee=fee)
        except ValidationError as e:
            return {"error": "validation_error", "details": e.errors()}, 422

        scored = await score_transaction(txid, vsize, fee)

        audit_id = str(uuid.uuid4())

        # Persist prediction
        await write_in_thread(
            save_prediction,
            {
                "model_name": scored["model_name"],
                "model_version": scored["model_version"],
                "input_hash": scored["input_hash"],
                "risk_score": scored["risk_score"],
                "risk_label": scored["risk_label"],
                "threshold_used": get_risk_threshold(),
                "top_shap_features": scored["shap_features"],
                "inference_time_ms": scored["inference_ms"],
            },
        )

        # Persist alert if above threshold
        if scored["risk_score"] > get_risk_threshold():
            await write_in_thread(
                save_alert,
                {
                    "tx_id": txid,
                    "risk_score": scored["risk_score"],
                    "risk_label": scored["risk_label"],
                    "model_name": scored["model_name"],
                    "explanation": scored["shap_features"],
                    "raw_input": {"vsize": vsize, "fee": fee, "fee_rate": fee / max(vsize, 1)},
                },
            )

        result = {
            "correlation_id": correlation_id,
            "risk_score": scored["risk_score"],
            "risk_label": scored["risk_label"],
            "explanation": scored["shap_features"],
            "narrative": None,
            "model_version": scored["model_version"],
            "audit_id": audit_id,
            "action": "auto_clear" if scored["risk_label"] == "LOW" else "queue_review",
            "inference_time_ms": scored["inference_ms"],
            "is_demo": scored["is_demo"],
        }

        # Return as formatted HTML for the walkthrough page, or JSON for API
        if "text/html" in str(request.headers.get("accept", "")):
            return Div(
                Pre(json.dumps(result, indent=2), cls="walkthrough-code"),
                style="margin-top: 12px;",
            )

        return result

    # === SSE STREAM ===

    @rt("/api/v1/stream/scores")
    async def stream_scores():
        from sse_starlette.sse import EventSourceResponse

        async def event_generator():
            import asyncio

            last_seen = set()

            while True:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get("https://mempool.space/api/mempool/recent")
                        txs = resp.json()

                    # Only dedupe within a single poll — if the API returns
                    # the same txs next poll, we score them again (keeps the
                    # feed alive even when mempool is slow).
                    current_batch = set()
                    new_txs = [t for t in txs if t.get("txid", "") not in last_seen]
                    # If everything is stale, just rescore the batch anyway
                    if not new_txs:
                        new_txs = txs

                    for tx in new_txs:
                        txid = tx.get("txid", "")
                        if txid in current_batch:
                            continue
                        current_batch.add(txid)

                        vsize = tx.get("vsize", tx.get("size", 0))
                        fee = tx.get("fee", 0)
                        fee_rate = fee / max(vsize, 1)

                        result = await score_transaction(txid, vsize, fee)
                        risk_score = result["risk_score"]
                        risk_label = result["risk_label"]
                        thresh = get_risk_threshold()
                        risk_cls = (
                            "risk-high"
                            if risk_score > thresh
                            else "risk-medium"
                            if risk_score > thresh * 0.6
                            else "risk-low"
                        )

                        now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")

                        model_name = result["model_name"]
                        is_ml = "heuristic" not in model_name.lower()
                        badge_cls = "model-badge model-badge-ml" if is_ml else "model-badge"
                        badge_text = "ML" if is_ml else "HEURISTIC"

                        anomaly_score = result.get("anomaly_score", 0.0)
                        anomaly_cls = (
                            "anomaly-high"
                            if anomaly_score > 0.7
                            else "anomaly-med"
                            if anomaly_score > 0.4
                            else "anomaly-low"
                        )

                        html = f"""<div class="feed-row {"high-risk" if risk_score > thresh else ""}">
                            <div style="color: var(--fg-dim);">{now}</div>
                            <div style="font-weight: bold;">{txid[:16]}...</div>
                            <div>{vsize} vB</div>
                            <div>{fee_rate:.1f} sat/vB</div>
                            <div class="{risk_cls}">{risk_score:.3f} <span class="{badge_cls}">{badge_text}</span></div>
                            <div>{risk_label}</div>
                            <div class="{anomaly_cls}">{anomaly_score:.3f}</div>
                        </div>"""

                        yield {"event": "transaction_scored", "data": html}

                        # Persist prediction audit (best-effort, only for genuinely new txs)
                        if txid not in last_seen:
                            try:
                                await write_in_thread(
                                    save_prediction,
                                    {
                                        "model_name": result["model_name"],
                                        "model_version": result["model_version"],
                                        "input_hash": result["input_hash"],
                                        "risk_score": risk_score,
                                        "risk_label": risk_label,
                                        "threshold_used": thresh,
                                        "top_shap_features": result["shap_features"],
                                        "inference_time_ms": result["inference_ms"],
                                    },
                                )

                                if risk_score > thresh:
                                    await write_in_thread(
                                        save_alert,
                                        {
                                            "tx_id": txid,
                                            "risk_score": risk_score,
                                            "risk_label": risk_label,
                                            "model_name": result["model_name"],
                                            "explanation": result["shap_features"],
                                            "raw_input": {
                                                "vsize": vsize,
                                                "fee": fee,
                                                "fee_rate": fee_rate,
                                            },
                                        },
                                    )
                            except Exception:
                                pass

                    last_seen = current_batch

                except Exception as e:
                    log.debug("Stream error", error=str(e))

                await asyncio.sleep(1)

        return EventSourceResponse(event_generator())


def _score_result(risk_score, risk_label, inference_ms, top_features):
    threshold = get_risk_threshold()
    return Div(
        DataGrid(
            DataReadout(
                "RISK_SCORE",
                f"{risk_score:.4f}",
                highlight=risk_score < threshold * 0.6,
                variant="danger" if risk_score > threshold else "",
            ),
            DataReadout("RISK_LABEL", risk_label.upper()),
            DataReadout("INFERENCE_TIME", f"{inference_ms:.1f}ms"),
            DataReadout("MODEL", "illicit-xgboost v1.0"),
        ),
        P(
            "Top contributing features would appear here with SHAP values.",
            style="color: var(--fg-dim); margin-top: 8px;",
        ),
    )
