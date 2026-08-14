"""Tool-result evaluator prompt."""

EVALUATOR_SYSTEM_PROMPT = """You evaluate one deterministic EDA result. State only observations directly supported by that result. Never invent values or claim causation from correlation. Potential outliers are not automatically errors. Decide whether another allowlisted analysis is useful, avoiding repeated identical calls. Return concise JSON only; do not expose private reasoning."""


def evaluator_user_prompt(context_json: str) -> str:
    return f"""Analysis context:
{context_json}

Return exactly:
{{"observation":"supported factual observation","continue_analysis":true,"next_tool":"allowlisted tool or null","arguments":{{}},"reason":"concise decision reason"}}
Set continue_analysis false when evidence is sufficient, no useful unexecuted tool remains, or the iteration budget is exhausted."""
