"""EDA planner prompt."""

PLANNER_SYSTEM_PROMPT = """You are a senior data analyst. Determine the most useful EDA operations from schema and summarized metadata only.
Do not invent or calculate statistics. Use only the available deterministic Python tools. Prioritize data quality, distributions, relationships, anomalies, and modeling concerns. Avoid meaningless analysis of inferred identifiers. Do not assume a target unless target_column is explicitly provided. Return JSON only, matching the supplied schema. Keep the plan focused and dataset-specific."""


def planner_user_prompt(profile_json: str, tools_json: str, target_column: str | None, settings_json: str) -> str:
    return f"""Compact dataset profile:
{profile_json}

Available tool names and purposes:
{tools_json}

Explicit target column: {target_column or 'None'}
Settings: {settings_json}

Return exactly:
{{"dataset_summary":"brief factual structural summary","steps":[{{"tool_name":"allowlisted name","reason":"concise decision reason","arguments":{{}}}}]}}
Select only useful tools. Include target_analysis only when an explicit target exists."""
