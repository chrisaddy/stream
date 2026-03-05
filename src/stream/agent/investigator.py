"""AI Investigation Agent — uses Claude tool_use to autonomously investigate alerts."""

import time

import structlog

from stream.agent.tools import TOOL_DEFINITIONS, execute_tool
from stream.config import settings

log = structlog.get_logger()

# Module-level investigation store
_investigations: dict[str, dict] = {}

SYSTEM_PROMPT = """You are a Bitcoin compliance investigation agent for the STREAM platform.

Your role is to investigate flagged Bitcoin transactions by gathering
evidence using the tools available to you, then producing a structured
investigation report.

Investigation protocol:
1. First, look up the alert details to understand why this transaction was flagged.
2. Look up the transaction on-chain to see its inputs, outputs, and confirmation status.
3. Check for similar transactions in the alert history to identify patterns.
4. Review the history of analyst verdicts to understand historical TP/FP patterns.
5. Based on all evidence, produce a final recommendation.

Your final message should be a structured report with:
- SUMMARY: 1-2 sentence overview
- RISK ASSESSMENT: LOW / MEDIUM / HIGH / CRITICAL with justification
- EVIDENCE: Key findings from your investigation
- RECOMMENDATION: ESCALATE / CLEAR / HOLD with confidence level

Be concise and analytical. Focus on facts from the data, not speculation."""


def get_investigation(tx_id: str) -> dict | None:
    return _investigations.get(tx_id)


async def run_investigation(tx_id: str) -> dict:
    """Run an autonomous investigation on a transaction.

    Uses Claude Haiku with tool_use to gather evidence and produce a report.
    Stores intermediate steps for real-time UI display.
    """
    investigation = {
        "tx_id": tx_id,
        "status": "RUNNING",
        "steps": [],
        "report": None,
        "started_at": time.time(),
        "completed_at": None,
    }
    _investigations[tx_id] = investigation

    try:
        import anthropic
    except ImportError:
        investigation["status"] = "ERROR"
        investigation["steps"].append(
            {"type": "error", "content": "anthropic package not installed"}
        )
        return investigation

    if not settings.ANTHROPIC_API_KEY:
        investigation["status"] = "ERROR"
        investigation["steps"].append(
            {"type": "error", "content": "ANTHROPIC_API_KEY not configured"}
        )
        return investigation

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    messages = [
        {
            "role": "user",
            "content": (
                f"Investigate alert for transaction {tx_id}."
                " Follow the investigation protocol."
            ),
        }
    ]

    max_turns = 8
    for turn in range(max_turns):
        try:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except Exception as e:
            investigation["steps"].append({"type": "error", "content": f"API error: {e}"})
            investigation["status"] = "ERROR"
            return investigation

        # Process response blocks
        assistant_content = response.content
        tool_uses = []
        text_blocks = []

        for block in assistant_content:
            if block.type == "text":
                text_blocks.append(block.text)
                investigation["steps"].append({"type": "thinking", "content": block.text})
            elif block.type == "tool_use":
                tool_uses.append(block)
                investigation["steps"].append(
                    {
                        "type": "tool_call",
                        "tool": block.name,
                        "input": block.input,
                    }
                )

        # If no tool use, this is the final response
        if response.stop_reason == "end_turn" or not tool_uses:
            final_text = (
                "\n".join(text_blocks)
                if text_blocks
                else "Investigation complete — no final report generated."
            )
            investigation["report"] = final_text
            investigation["status"] = "COMPLETE"
            investigation["completed_at"] = time.time()
            return investigation

        # Execute tool calls and build tool_result messages
        messages.append({"role": "assistant", "content": assistant_content})
        tool_results = []
        for tool_use in tool_uses:
            result = await execute_tool(tool_use.name, tool_use.input)
            investigation["steps"].append(
                {
                    "type": "tool_result",
                    "tool": tool_use.name,
                    "result": result[:500],  # truncate for display
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    # If we hit max turns without a final response
    investigation["status"] = "COMPLETE"
    investigation["completed_at"] = time.time()
    if not investigation["report"]:
        investigation["report"] = (
            "Investigation reached maximum depth. Review the gathered evidence above."
        )
    return investigation
