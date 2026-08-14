"""Typed shared state passed through the LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class EDAState(TypedDict, total=False):
    dataframe: pd.DataFrame
    dataset_profile: dict[str, Any]
    analysis_plan: list[dict[str, Any]]
    dataset_summary: str
    completed_analyses: list[str]
    completed_tool_calls: list[str]
    current_tool: str | None
    current_tool_args: dict[str, Any]
    current_reason: str
    suggested_tool: str | None
    suggested_arguments: dict[str, Any]
    suggested_reason: str
    tool_results: list[dict[str, Any]]
    observations: list[str]
    iteration_count: int
    max_iterations: int
    continue_analysis: bool
    next_action: str
    final_report: str | None
    errors: list[str]
    execution_trace: list[dict[str, Any]]
    target_column: str | None
    settings: dict[str, Any]
