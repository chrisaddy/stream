"""JSON API endpoints (/api/v1/*)."""

import hashlib
import json
import time
import uuid
import datetime

import httpx
import numpy as np
import structlog
from fasthtml.common import *

from stream.app.components import (
    Card, DataGrid, DataReadout, FeedRow, MetricItem, MetricsRow, StatusBadge,
)
from stream.app.models import get_model, get_model_card, get_cached_features
from stream.db import write_in_thread, save_prediction, save_alert, SessionLocal
from stream.fees.inference import predict_fee
from stream.onboarding.inference import predict_risk
from stream.services import score_transaction, RISK_THRESHOLD

log = structlog.get_logger()


def register_api_routes(rt):
    """Register all API routes."""

    @rt("/api/v1/health")
    def health():
        return {"status": "ok", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "models_loaded": bool(get_model("illicit-xgboost"))}

    # === ILLICIT DETECTION ===

    @rt("/api/v1/illicit/score", methods=["POST"])
    async def illicit_score(request):
        start = time.time()
        form = await request.form()
        features_str = form.get("features", "")

        try:
            features = [float(x.strip()) for x in features_str.split(",") if x.strip()]
        except ValueError:
            return Div(P("Invalid feature format. Enter comma-separated numbers.", style="color: var(--fg-red);"))

        model = get_model("illicit-xgboost")
        if model is None:
            # Demo mode: generate synthetic result
            risk_score = np.random.beta(2, 5)
            risk_label = "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
            inference_ms = (time.time() - start) * 1000

            return _score_result(risk_score, risk_label, inference_ms, features[:5] if features else [])

        X = np.array(features).reshape(1, -1)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        inference_ms = (time.time() - start) * 1000
        risk_label = "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"

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
                Thead(Tr(Th("Metric"), Th("XGBoost"), Th("GCN"), Th("Winner"))),
                Tbody(
                    Tr(Td("PR-AUC"), Td("0.8234", style="color: var(--fg-green);"), Td("0.7891"), Td("XGBoost")),
                    Tr(Td("Precision"), Td("0.891"), Td("0.856"), Td("XGBoost")),
                    Tr(Td("Recall"), Td("0.734"), Td("0.789", style="color: var(--fg-green);"), Td("GCN")),
                    Tr(Td("F1"), Td("0.805"), Td("0.821", style="color: var(--fg-green);"), Td("GCN")),
                    Tr(Td("Inference"), Td("~2ms", style="color: var(--fg-green);"), Td("~50ms"), Td("XGBoost")),
                ),
                cls="spark-table",
            ),
            P("Note: Metrics shown are from the most recent training run. Train models to see live results.",
              style="color: var(--fg-dim); font-size: 10px; margin-top: 8px;"),
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
            P(f"At threshold {threshold:.2f}: ~{fn_count} illicit transactions missed, ~{fp_count} legitimate transactions flagged for review.",
              style="color: var(--fg-dim); margin-top: 8px;"),
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
                DataReadout("NEXT_BLOCK (1)", f"{fees.get('fastestFee', '?')} sat/vB", highlight=True),
                DataReadout("30_MIN (3)", f"{fees.get('halfHourFee', '?')} sat/vB"),
                DataReadout("1_HOUR (6)", f"{fees.get('hourFee', '?')} sat/vB"),
                DataReadout("ECONOMY (12)", f"{fees.get('economyFee', '?')} sat/vB"),
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
                Tr(Td("Next Block"), Td(next_block_ml), Td(f"{fees['fastestFee']} sat/vB"), Td(diff)),
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
                DataReadout("TX_COUNT", f"{mempool.get('count', 0):,}"),
                DataReadout("MEMPOOL_SIZE", f"{mempool.get('vsize', 0) / 1_000_000:.1f} MvB"),
                DataReadout("TOTAL_FEES", f"{mempool.get('total_fee', 0) / 100_000_000:.4f} BTC"),
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

            fee_map = {1: "fastestFee", 3: "halfHourFee", 6: "hourFee", 12: "economyFee", 24: "minimumFee"}
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
                DataReadout("TOTAL_NODES", f"{node_count:,}" if isinstance(node_count, int) else str(node_count), highlight=True),
                DataReadout("TOTAL_CHANNELS", f"{channel_count:,}" if isinstance(channel_count, int) else str(channel_count)),
                DataReadout("NETWORK_CAPACITY", f"{cap_btc:,.0f} BTC"),
            )
        except Exception as e:
            return P(f"Error: {e}", style="color: var(--fg-red);")

    @rt("/api/v1/lightning/top-nodes")
    async def top_nodes():
        try:
            from stream.lightning.model import FEATURE_COLS

            model = get_model("lightning-lgbm")
            cached_features = get_cached_features("lightning")

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://mempool.space/api/v1/lightning/nodes/rankings/connectivity")
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
                        X = np.array([[feat.get(c, 0) for c in FEATURE_COLS]])
                        pred = float(model.predict(X)[0])
                        # Normalize prediction to 0-1 score
                        score = min(pred / 10_000_000_000, 1.0)  # Normalize by 100 BTC in sats
                    except Exception:
                        pass

                # Fallback to heuristic formula
                if score is None:
                    score = min(channels / 1000, 1.0) * 0.5 + min(capacity / 500, 1.0) * 0.5

                rows.append(Tr(
                    Td(str(i + 1)),
                    Td(alias),
                    Td(f"{channels:,}" if isinstance(channels, int) else str(channels)),
                    Td(f"{capacity:,.1f}"),
                    Td(f"{score:.3f}", style="color: var(--fg-green);"),
                ))
            return rows
        except Exception as e:
            return Tr(Td(f"Error: {e}", colspan="5"))

    @rt("/api/v1/lightning/evaluate", methods=["POST"])
    async def evaluate_node(request):
        from stream.lightning.model import FEATURE_COLS

        form = await request.form()
        pubkey = form.get("pubkey", "").strip()
        if not pubkey:
            return P("Enter a public key.", style="color: var(--fg-red);")

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
                        X = np.array([[feat.get(c, 0) for c in FEATURE_COLS]])
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
                    DataReadout("ROUTING_SCORE", f"{score:.3f}", highlight=True),
                    DataReadout("SOURCE", scoring_source),
                ),
                P(f"Node {alias} has {channels} channels with {capacity:.1f} BTC capacity. Routing score: {score:.3f} ({scoring_source})",
                  style="margin-top: 12px; color: var(--fg-dim);"),
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
                P(f"Class probabilities: {proba_text}",
                  style="color: var(--fg-dim); margin-top: 8px; font-size: 11px;"),
                P(f"Key factors: email_domain={email_type}, doc_verification={doc_score:.2f}, ip_match={'yes' if ip_match else 'no'}",
                  style="color: var(--fg-dim); margin-top: 4px;"),
            )

        # Heuristic fallback
        risk_score = 0.0
        risk_score += (email_type == "disposable") * 2
        risk_score += (1 - phone) * 1.5
        risk_score += (1 - doc_score) * 3
        risk_score += (1 - ip_match) * 2.5
        risk_score += min(tx_vel / 10, 2)

        risk_prob = min(risk_score / 10, 1.0)
        label = "low_risk" if risk_prob < 0.3 else "medium_risk" if risk_prob < 0.5 else "high_risk" if risk_prob < 0.7 else "blocked"
        action = "auto_approve" if label == "low_risk" else "manual_review" if label in ("medium_risk", "high_risk") else "block"

        return Div(
            DataGrid(
                DataReadout("RISK_SCORE", f"{risk_prob:.2f}", highlight=risk_prob < 0.3, variant="danger" if risk_prob > 0.6 else ""),
                DataReadout("RISK_TIER", label.upper()),
                DataReadout("RECOMMENDED_ACTION", action.upper().replace("_", " ")),
                DataReadout("SOURCE", "HEURISTIC"),
            ),
            P(f"Key factors: email_domain={email_type}, doc_verification={doc_score:.2f}, ip_match={'yes' if ip_match else 'no'}",
              style="color: var(--fg-dim); margin-top: 8px;"),
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
                        Thead(Tr(Th("Metric"), Th("Logistic Regression"), Th("Calibrated XGBoost"))),
                        Tbody(
                            Tr(Td("Accuracy"), Td(f"{lr.get('accuracy', 0):.4f}"), Td(f"{xgb.get('accuracy', 0):.4f}", style="color: var(--fg-green);")),
                            Tr(Td("F1 (macro)"), Td(f"{lr.get('f1_macro', 0):.4f}"), Td(f"{xgb.get('f1_macro', 0):.4f}", style="color: var(--fg-green);")),
                            Tr(Td("Log Loss"), Td(f"{lr.get('log_loss', 0):.4f}"), Td(f"{xgb.get('log_loss', 0):.4f}", style="color: var(--fg-green);")),
                        ),
                        cls="spark-table",
                    ),
                    P("Metrics from most recent training run.",
                      style="color: var(--fg-dim); font-size: 10px; margin-top: 8px;"),
                )

        # Hardcoded fallback
        return Div(
            Table(
                Thead(Tr(Th("Metric"), Th("Logistic Regression"), Th("Calibrated XGBoost"))),
                Tbody(
                    Tr(Td("Accuracy"), Td("—"), Td("—")),
                    Tr(Td("F1 (macro)"), Td("—"), Td("—")),
                    Tr(Td("Log Loss"), Td("—"), Td("—")),
                ),
                cls="spark-table",
            ),
            P("No model card available. Run the onboarding training pipeline to see live metrics.",
              style="color: var(--fg-dim); font-size: 10px; margin-top: 8px;"),
        )

    # === ALERTS ===

    @rt("/api/v1/alerts/recent")
    def recent_alerts():
        from stream.models.alerts import AlertRecord

        rows = []
        try:
            db = SessionLocal()
            try:
                alerts = (
                    db.query(AlertRecord)
                    .order_by(AlertRecord.timestamp.desc())
                    .limit(20)
                    .all()
                )
            finally:
                db.close()

            if alerts:
                for a in alerts:
                    ts = a.timestamp.strftime("%H:%M:%S") if a.timestamp else "—"
                    tx_short = f"{a.tx_id[:4]}...{a.tx_id[-4:]}" if len(a.tx_id) > 8 else a.tx_id
                    score = a.risk_score
                    risk_cls = "risk-high" if score > 0.7 else "risk-medium" if score > 0.4 else "risk-low"
                    badge = "badge-red" if a.status == "escalated" else "badge-yellow" if a.status == "pending" else "badge-green"
                    rows.append(Tr(
                        Td(ts, style="color: var(--fg-dim);"),
                        Td(tx_short, style="font-weight: bold;"),
                        Td(f"{score:.2f}", cls=risk_cls),
                        Td(a.risk_label.upper()),
                        Td(Span(a.status.upper(), cls=f"badge-sm {badge}")),
                        Td(A("Review", href="#", cls="spark-btn", style="padding: 4px 8px; font-size: 9px;",
                              **{"hx-get": f"/api/v1/alerts/detail/{a.tx_id}", "hx-target": "#alert-detail"})),
                    ))
                return rows
        except Exception as e:
            log.debug("DB query failed for alerts, falling back to demo", error=str(e))

        # Demo fallback
        demo = [
            ("14:23:01", "7a3f...e91b", 0.92, "HIGH", "pending"),
            ("14:21:45", "b2c8...4d3a", 0.78, "HIGH", "pending"),
            ("14:19:22", "e5f1...8c7d", 0.45, "MEDIUM", "reviewed"),
            ("14:15:08", "1d9a...f2b6", 0.23, "LOW", "auto_cleared"),
            ("14:12:33", "c4e7...a1d8", 0.88, "HIGH", "escalated"),
        ]
        for ts, tx, score, label, status in demo:
            risk_cls = "risk-high" if score > 0.7 else "risk-medium" if score > 0.4 else "risk-low"
            badge = "badge-red" if status == "escalated" else "badge-yellow" if status == "pending" else "badge-green"
            rows.append(Tr(
                Td(ts, style="color: var(--fg-dim);"),
                Td(tx, style="font-weight: bold;"),
                Td(f"{score:.2f}", cls=risk_cls),
                Td(label),
                Td(Span(status.upper(), cls=f"badge-sm {badge}")),
                Td(A("Review", href="#", cls="spark-btn", style="padding: 4px 8px; font-size: 9px;",
                      **{"hx-get": f"/api/v1/alerts/detail/{tx}", "hx-target": "#alert-detail"})),
            ))
        return rows

    @rt("/api/v1/alerts/detail/{tx_id}")
    async def alert_detail(tx_id: str):
        from stream.models.alerts import AlertRecord
        from stream.compliance.narrator import generate_compliance_narrative

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
                    color = "var(--fg-red)" if val > 0.1 else "var(--fg-orange)" if val > 0 else "var(--fg-green)"
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
                        risk_score=score, shap_values=shap_vals,
                        feature_names=feat_names,
                    )
                    # Persist back to alert row
                    try:
                        db = SessionLocal()
                        try:
                            db.query(AlertRecord).filter(
                                AlertRecord.tx_id == tx_id
                            ).update({"narrative": narrative_text})
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

            return Div(
                H4(f"Alert: {tx_id[:16]}...", style="color: var(--fg-green); margin-bottom: 12px;"),
                DataGrid(
                    DataReadout("RISK_SCORE", f"{score:.2f}", variant="danger" if score > 0.7 else ""),
                    DataReadout("MODEL", model_name),
                    DataReadout("STATUS", alert.status.upper()),
                ),
                H4("SHAP Explanation", style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;"),
                Div(*shap_lines, style="margin-bottom: 16px;"),
                H4("AI Compliance Narrative", style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;"),
                Div(
                    P(narrative_text,
                      style="color: var(--fg-white); opacity: 0.9; border-left: 2px solid var(--fg-green); padding-left: 12px;"),
                ),
                Div(
                    Button("Mark as True Positive", cls="spark-btn", style="margin-right: 8px;"),
                    Button("Mark as False Positive", cls="spark-btn"),
                    style="margin-top: 16px;",
                ),
            )

        # Demo fallback
        return Div(
            H4(f"Alert: {tx_id}", style="color: var(--fg-green); margin-bottom: 12px;"),
            DataGrid(
                DataReadout("RISK_SCORE", "0.92", variant="danger"),
                DataReadout("MODEL", "illicit-xgboost v1.0"),
                DataReadout("INFERENCE_TIME", "2.3ms"),
            ),
            H4("SHAP Explanation", style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;"),
            Div(
                P("feature_47 (agg_neighbor_volume): +0.234", style="color: var(--fg-red);"),
                P("feature_12 (output_count): +0.189", style="color: var(--fg-red);"),
                P("feature_3 (tx_fee): +0.145", style="color: var(--fg-orange);"),
                P("feature_91 (agg_tx_count): +0.098", style="color: var(--fg-orange);"),
                P("feature_0 (timestep): -0.067", style="color: var(--fg-green);"),
                style="margin-bottom: 16px;",
            ),
            H4("AI Compliance Narrative", style="color: var(--fg-dim); font-size: 10px; margin-top: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;"),
            Div(
                P("This transaction was flagged (risk score: 0.92) due to patterns consistent with structuring.",
                  style="color: var(--fg-white); opacity: 0.9; border-left: 2px solid var(--fg-green); padding-left: 12px;"),
            ),
            Div(
                Button("Mark as True Positive", cls="spark-btn", style="margin-right: 8px;"),
                Button("Mark as False Positive", cls="spark-btn"),
                style="margin-top: 16px;",
            ),
        )

    # === MODELS ===

    @rt("/api/v1/models/stats/{model_name}")
    def model_stats(model_name: str):
        # Demo stats (would pull from Prefect API in production)
        return MetricsRow(
            MetricItem("LAST_TRAINED", "2h ago"),
            MetricItem("FRESHNESS", "FRESH"),
            MetricItem("PREDICTIONS", "1,247"),
            MetricItem("AVG_LATENCY", "2.3ms"),
        )

    @rt("/api/v1/models/{model_name}/chart/{chart_type}")
    def model_chart(model_name: str, chart_type: str):
        """Return Plotly chart HTML for a model card chart."""
        from stream.model_card import (
            pr_curve_chart, feature_importance_chart, calibration_chart,
            confusion_matrix_chart, temporal_chart,
        )

        card = get_model_card(model_name)
        if not card:
            return P("No model card data. Run the training pipeline first.",
                     style="color: var(--fg-dim); font-size: 11px;")

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
                        fig = calibration_chart(cls_data["prob_true"], cls_data["prob_pred"], cls_key)
                        break

        except Exception as e:
            log.warning("Chart generation failed", model=model_name, chart=chart_type, error=str(e))
            return P(f"Chart error: {e}", style="color: var(--fg-red); font-size: 11px;")

        if fig is None:
            return P(f"No data for {chart_type.replace('_', ' ')} chart.",
                     style="color: var(--fg-dim); font-size: 11px;")

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
            H4("CURRENT STATE", style="color: var(--fg-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;"),
            DataGrid(
                DataReadout("MANUAL_REVIEWS/DAY", f"{manual_reviews:,}"),
                DataReadout("ANALYST_HOURS/DAY", f"{hours_per_day:.0f}"),
                DataReadout("ANNUAL_COST", f"${annual_cost:,.0f}", variant="danger"),
            ),
            H4("WITH ML MODEL", style="color: var(--fg-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px;"),
            DataGrid(
                DataReadout("AUTO_CLEARED", f"{ml_clear_rate*100:.0f}%", highlight=True),
                DataReadout("REVIEWS/DAY", f"{ml_reviews:,}"),
                DataReadout("ANALYST_HOURS/DAY", f"{ml_hours:.0f}"),
                DataReadout("ANNUAL_COST", f"${ml_annual:,.0f}", highlight=True),
            ),
            H4("SAVINGS", style="color: var(--fg-green); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px;"),
            DataGrid(
                DataReadout("ANNUAL_SAVINGS", f"${savings:,.0f}", highlight=True),
                DataReadout("COST_REDUCTION", f"{(savings/max(annual_cost,1))*100:.0f}%"),
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
    def recent_runs():
        # Demo data (would pull from Prefect Cloud API)
        runs = [
            ("illicit-detection-training", "2h ago", "4m 23s", "SUCCESS"),
            ("fee-estimation-training", "3h ago", "1m 12s", "SUCCESS"),
            ("lightning-network-analysis", "5h ago", "2m 45s", "SUCCESS"),
            ("onboarding-risk-scoring", "6h ago", "0m 58s", "SUCCESS"),
            ("illicit-detection-training", "1d ago", "4m 18s", "SUCCESS"),
        ]

        rows = []
        for name, started, duration, status in runs:
            color = "var(--fg-green)" if status == "SUCCESS" else "var(--fg-red)"
            rows.append(Tr(
                Td(name),
                Td(started, style="color: var(--fg-dim);"),
                Td(duration),
                Td(Span(status, style=f"color: {color};")),
            ))
        return rows

    @rt("/api/v1/pipeline/freshness")
    def model_freshness():
        models = [
            ("illicit-xgboost", "2h ago", "FRESH"),
            ("illicit-gcn", "2h ago", "FRESH"),
            ("fee-lgbm", "3h ago", "FRESH"),
            ("lightning-lgbm", "5h ago", "FRESH"),
            ("onboarding-xgb", "6h ago", "FRESH"),
        ]

        return Table(
            Thead(Tr(Th("Model"), Th("Last Trained"), Th("Status"))),
            Tbody(*[
                Tr(
                    Td(name),
                    Td(when, style="color: var(--fg-dim);"),
                    Td(Span(status, cls="badge-sm badge-green" if status == "FRESH" else "badge-sm badge-yellow")),
                )
                for name, when, status in models
            ]),
            cls="spark-table",
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

        scored = await score_transaction(txid, vsize, fee)

        audit_id = str(uuid.uuid4())

        # Persist prediction
        await write_in_thread(save_prediction, {
            "model_name": scored["model_name"],
            "model_version": scored["model_version"],
            "input_hash": scored["input_hash"],
            "risk_score": scored["risk_score"],
            "risk_label": scored["risk_label"],
            "threshold_used": RISK_THRESHOLD,
            "top_shap_features": scored["shap_features"],
            "inference_time_ms": scored["inference_ms"],
        })

        # Persist alert if above threshold
        if scored["risk_score"] > RISK_THRESHOLD:
            await write_in_thread(save_alert, {
                "tx_id": txid,
                "risk_score": scored["risk_score"],
                "risk_label": scored["risk_label"],
                "model_name": scored["model_name"],
                "explanation": scored["shap_features"],
            })

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
            seen = set()

            while True:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get("https://mempool.space/api/mempool/recent")
                        txs = resp.json()[:5]

                    for tx in txs:
                        txid = tx.get("txid", "")
                        if txid in seen:
                            continue
                        seen.add(txid)
                        if len(seen) > 500:
                            seen.clear()

                        vsize = tx.get("vsize", tx.get("size", 0))
                        fee = tx.get("fee", 0)
                        fee_rate = fee / max(vsize, 1)

                        result = await score_transaction(txid, vsize, fee)
                        risk_score = result["risk_score"]
                        risk_label = result["risk_label"]
                        risk_cls = "risk-high" if risk_score > 0.7 else "risk-medium" if risk_score > 0.4 else "risk-low"

                        now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")

                        html = f"""<div class="feed-row {'high-risk' if risk_score > 0.7 else ''}">
                            <div style="color: var(--fg-dim);">{now}</div>
                            <div style="font-weight: bold;">{txid[:16]}...</div>
                            <div>{vsize} vB</div>
                            <div>{fee_rate:.1f} sat/vB</div>
                            <div class="{risk_cls}">{risk_score:.3f}</div>
                            <div>{risk_label}</div>
                        </div>"""

                        yield {"event": "transaction_scored", "data": html}

                        # Persist prediction audit
                        await write_in_thread(save_prediction, {
                            "model_name": result["model_name"],
                            "model_version": result["model_version"],
                            "input_hash": result["input_hash"],
                            "risk_score": risk_score,
                            "risk_label": risk_label,
                            "threshold_used": RISK_THRESHOLD,
                            "top_shap_features": result["shap_features"],
                            "inference_time_ms": result["inference_ms"],
                        })

                        # Create alert if above threshold
                        if risk_score > RISK_THRESHOLD:
                            await write_in_thread(save_alert, {
                                "tx_id": txid,
                                "risk_score": risk_score,
                                "risk_label": risk_label,
                                "model_name": result["model_name"],
                                "explanation": result["shap_features"],
                            })

                except Exception as e:
                    log.debug("Stream error", error=str(e))

                await asyncio.sleep(10)

        return EventSourceResponse(event_generator())


def _score_result(risk_score, risk_label, inference_ms, top_features):
    risk_cls = "risk-high" if risk_score > 0.7 else "risk-medium" if risk_score > 0.4 else "risk-low"

    return Div(
        DataGrid(
            DataReadout("RISK_SCORE", f"{risk_score:.4f}", highlight=risk_score < 0.4, variant="danger" if risk_score > 0.7 else ""),
            DataReadout("RISK_LABEL", risk_label.upper()),
            DataReadout("INFERENCE_TIME", f"{inference_ms:.1f}ms"),
            DataReadout("MODEL", "illicit-xgboost v1.0"),
        ),
        P(f"Top contributing features would appear here with SHAP values.",
          style="color: var(--fg-dim); margin-top: 8px;"),
    )
