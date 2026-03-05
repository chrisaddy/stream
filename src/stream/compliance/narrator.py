"""LLM compliance narrative generation via Claude API.

Translates SHAP explanations into analyst-ready SAR narratives.
This is the full compliance stack: model scores → SHAP explains → LLM narrates.
"""

import structlog

log = structlog.get_logger()


def get_top_shap_features(
    shap_values: list[float],
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Get top-k features by absolute SHAP value."""
    pairs = list(zip(feature_names, shap_values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": name, "shap_value": round(val, 4)} for name, val in pairs[:top_k]]


def format_features_for_prompt(top_features: list[dict]) -> str:
    lines = []
    for f in top_features:
        direction = "increases" if f["shap_value"] > 0 else "decreases"
        lines.append(f"- {f['feature']}: SHAP value {f['shap_value']:.4f} ({direction} risk)")
    return "\n".join(lines)


async def generate_compliance_narrative(
    risk_score: float,
    shap_values: list[float],
    feature_names: list[str],
    top_k: int = 5,
) -> str:
    """Generate a compliance-ready narrative from SHAP explanation.

    Uses Claude Haiku for speed + cost efficiency in production.
    """
    top_features = get_top_shap_features(shap_values, feature_names, top_k)

    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        feature_text = format_features_for_prompt(top_features)

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate a concise compliance analyst "
                        "narrative for this flagged Bitcoin "
                        "transaction.\n\n"
                        f"Risk score: {risk_score:.2f}\n"
                        "Top contributing features:\n"
                        f"{feature_text}\n\n"
                        "Write 2-3 sentences suitable for a SAR "
                        "filing. Be specific about which patterns "
                        "triggered the flag. Use financial "
                        "compliance terminology."
                    ),
                }
            ],
        )
        return response.content[0].text

    except Exception as e:
        log.warning("Narrative generation failed", error=str(e))
        return _template_narrative(risk_score, top_features)


def _template_narrative(risk_score: float, top_features: list[dict]) -> str:
    """Template fallback when Claude API is unavailable."""
    top = top_features[0] if top_features else {"feature": "unknown", "shap_value": 0}
    risk_level = "high" if risk_score > 0.8 else "medium" if risk_score > 0.5 else "elevated"

    return (
        f"This transaction was flagged with {risk_level} risk (score: {risk_score:.2f}). "
        f"The primary contributing factor was {top['feature']} "
        f"(SHAP contribution: {top['shap_value']:.4f}). "
        f"Recommend manual review by compliance analyst."
    )
