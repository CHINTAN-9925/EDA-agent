"""Common types and helpers for deterministic EDA tools."""

from __future__ import annotations

from typing import Any

from utils.validators import json_safe


def tool_result(name: str, display_result: dict[str, Any], llm_summary: dict[str, Any], insights: list[str]) -> dict[str, Any]:
    """Build the standard result envelope returned by every tool."""
    return json_safe({
        "tool_name": name,
        "status": "success",
        "display_result": display_result,
        "llm_summary": llm_summary,
        "insights": insights,
    })
