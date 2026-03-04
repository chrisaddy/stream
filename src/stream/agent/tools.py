"""Tool implementations for the investigation agent.

Each tool queries real data from the DB or external APIs.
"""

import httpx
import structlog

from stream.db import SessionLocal

log = structlog.get_logger()

# Tool definitions for Claude's tool_use API
TOOL_DEFINITIONS = [
    {
        "name": "get_alert_details",
        "description": "Look up the full alert record for a transaction, including risk score, model name, SHAP explanation, and review status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tx_id": {"type": "string", "description": "The transaction ID to look up"}
            },
            "required": ["tx_id"],
        },
    },
    {
        "name": "lookup_transaction_onchain",
        "description": "Look up a Bitcoin transaction on-chain via mempool.space API. Returns inputs, outputs, fees, confirmation status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tx_id": {"type": "string", "description": "The Bitcoin transaction ID"}
            },
            "required": ["tx_id"],
        },
    },
    {
        "name": "query_prediction_history",
        "description": "Query recent prediction records above a risk threshold. Shows what the system has been scoring recently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_risk_score": {"type": "number", "description": "Minimum risk score to filter by (0-1)", "default": 0.5},
                "limit": {"type": "integer", "description": "Max results to return", "default": 10},
            },
            "required": [],
        },
    },
    {
        "name": "get_review_history",
        "description": "Query historical analyst reviews (TP/FP verdicts). Shows patterns of true vs false positives.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results to return", "default": 20},
            },
            "required": [],
        },
    },
    {
        "name": "check_similar_transactions",
        "description": "Find alerts with similar fee_rate and vsize profiles to the target transaction. Helps identify patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tx_id": {"type": "string", "description": "The reference transaction ID"},
                "tolerance": {"type": "number", "description": "How similar (0-1, where 0.2 = within 20%)", "default": 0.3},
            },
            "required": ["tx_id"],
        },
    },
]


async def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return a string result."""
    try:
        if name == "get_alert_details":
            return _get_alert_details(args["tx_id"])
        elif name == "lookup_transaction_onchain":
            return await _lookup_onchain(args["tx_id"])
        elif name == "query_prediction_history":
            return _query_predictions(args.get("min_risk_score", 0.5), args.get("limit", 10))
        elif name == "get_review_history":
            return _get_reviews(args.get("limit", 20))
        elif name == "check_similar_transactions":
            return _check_similar(args["tx_id"], args.get("tolerance", 0.3))
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def _get_alert_details(tx_id: str) -> str:
    """Look up a full alert record by transaction ID."""
    from stream.models.alerts import AlertRecord
    db = SessionLocal()
    try:
        alert = db.query(AlertRecord).filter(AlertRecord.tx_id == tx_id).first()
        if not alert:
            return f"No alert found for tx_id={tx_id}"
        raw = alert.raw_input or {}
        return (
            f"Alert for {tx_id}:\n"
            f"  Risk Score: {alert.risk_score:.4f}\n"
            f"  Risk Label: {alert.risk_label}\n"
            f"  Model: {alert.model_name}\n"
            f"  Status: {alert.status}\n"
            f"  Raw Input: vsize={raw.get('vsize', '?')}, fee={raw.get('fee', '?')}, fee_rate={raw.get('fee_rate', '?')}\n"
            f"  SHAP: {alert.explanation}\n"
            f"  Narrative: {alert.narrative or 'None generated yet'}"
        )
    finally:
        db.close()


async def _lookup_onchain(tx_id: str) -> str:
    """Fetch on-chain transaction data from mempool.space."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"https://mempool.space/api/tx/{tx_id}")
            if resp.status_code != 200:
                return f"mempool.space returned {resp.status_code} for {tx_id}. Transaction may be unconfirmed or invalid."
            tx = resp.json()

        n_inputs = len(tx.get("vin", []))
        n_outputs = len(tx.get("vout", []))
        fee = tx.get("fee", 0)
        size = tx.get("vsize", tx.get("size", 0))
        fee_rate = fee / max(size, 1)
        confirmed = tx.get("status", {}).get("confirmed", False)
        block_height = tx.get("status", {}).get("block_height", "unconfirmed")

        total_output = sum(v.get("value", 0) for v in tx.get("vout", []))
        output_addrs = []
        for v in tx.get("vout", [])[:5]:
            addr = v.get("scriptpubkey_address", "unknown")
            val = v.get("value", 0)
            output_addrs.append(f"  {addr}: {val} sats")

        return (
            f"On-chain data for {tx_id}:\n"
            f"  Confirmed: {confirmed} (block {block_height})\n"
            f"  Inputs: {n_inputs}, Outputs: {n_outputs}\n"
            f"  Size: {size} vB, Fee: {fee} sats ({fee_rate:.1f} sat/vB)\n"
            f"  Total Output: {total_output} sats ({total_output / 1e8:.8f} BTC)\n"
            f"  Top outputs:\n" + "\n".join(output_addrs)
        )
    except Exception as e:
        return f"On-chain lookup failed: {e}"


def _query_predictions(min_score: float, limit: int) -> str:
    """Query recent predictions above a minimum risk score."""
    from stream.models.predictions import PredictionRecord
    db = SessionLocal()
    try:
        preds = (
            db.query(PredictionRecord)
            .filter(PredictionRecord.risk_score >= min_score)
            .order_by(PredictionRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        if not preds:
            return f"No predictions found above {min_score}"
        lines = [f"Recent high-risk predictions (>= {min_score}):"]
        for p in preds:
            lines.append(f"  {p.timestamp}: score={p.risk_score:.4f} label={p.risk_label} model={p.model_name}")
        return "\n".join(lines)
    finally:
        db.close()


def _get_reviews(limit: int) -> str:
    """Retrieve recent analyst review verdicts (TP/FP counts)."""
    from stream.models.reviews import ReviewRecord
    db = SessionLocal()
    try:
        reviews = db.query(ReviewRecord).order_by(ReviewRecord.timestamp.desc()).limit(limit).all()
        if not reviews:
            return "No reviews recorded yet."
        tp_count = sum(1 for r in reviews if r.verdict == "true_positive")
        fp_count = sum(1 for r in reviews if r.verdict == "false_positive")
        lines = [
            f"Review history ({len(reviews)} total): {tp_count} TP, {fp_count} FP",
            f"Historical TP rate: {tp_count / max(len(reviews), 1):.0%}",
        ]
        for r in reviews[:5]:
            lines.append(f"  {r.timestamp}: tx={r.tx_id[:16]}... verdict={r.verdict} score={r.risk_score_at_review:.4f}")
        return "\n".join(lines)
    finally:
        db.close()


def _check_similar(tx_id: str, tolerance: float) -> str:
    """Find alerts with similar fee_rate/vsize profiles within tolerance."""
    from stream.models.alerts import AlertRecord
    db = SessionLocal()
    try:
        target = db.query(AlertRecord).filter(AlertRecord.tx_id == tx_id).first()
        if not target or not target.raw_input:
            return f"No raw_input data for {tx_id} — cannot compare."

        ref_fee_rate = target.raw_input.get("fee_rate", 0)
        ref_vsize = target.raw_input.get("vsize", 0)

        all_alerts = db.query(AlertRecord).filter(AlertRecord.tx_id != tx_id, AlertRecord.raw_input.isnot(None)).limit(100).all()

        similar = []
        for a in all_alerts:
            raw = a.raw_input or {}
            fr = raw.get("fee_rate", 0)
            vs = raw.get("vsize", 0)
            if ref_fee_rate > 0 and ref_vsize > 0:
                fr_diff = abs(fr - ref_fee_rate) / max(ref_fee_rate, 1)
                vs_diff = abs(vs - ref_vsize) / max(ref_vsize, 1)
                if fr_diff <= tolerance and vs_diff <= tolerance:
                    similar.append((a, fr_diff + vs_diff))

        similar.sort(key=lambda x: x[1])
        if not similar:
            return f"No similar transactions found within {tolerance:.0%} tolerance."

        lines = [f"Found {len(similar)} similar transactions:"]
        for a, dist in similar[:5]:
            raw = a.raw_input or {}
            reviewed = a.status != "pending"
            lines.append(
                f"  {a.tx_id[:16]}... score={a.risk_score:.4f} "
                f"fee_rate={raw.get('fee_rate', '?'):.1f} vsize={raw.get('vsize', '?')} "
                f"status={a.status}"
            )
        return "\n".join(lines)
    finally:
        db.close()
