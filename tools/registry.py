"""Allowlisted EDA tool registry and parameter validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tools.categorical import cardinality_analysis, categorical_imbalance_analysis, categorical_summary
from tools.correlation import correlation_analysis
from tools.datetime_analysis import datetime_analysis
from tools.missing import missing_value_analysis
from tools.numerical import distribution_analysis, numerical_summary, outlier_analysis
from tools.overview import dataset_overview, duplicate_analysis
from tools.target import target_analysis


class ToolArguments(BaseModel):
    """Base for strict, tool-specific argument schemas."""
    model_config = ConfigDict(extra="forbid")


class NoArguments(ToolArguments):
    pass


class ColumnArguments(ToolArguments):
    columns: list[str] | None = None


class MissingArguments(ColumnArguments):
    significant_threshold: float | None = Field(default=None, ge=0, le=100)


class CategoricalArguments(ColumnArguments):
    top_n: int | None = Field(default=None, ge=1, le=25)


class CorrelationArguments(ColumnArguments):
    method: Literal["pearson", "spearman"] | None = None
    high_threshold: float | None = Field(default=None, ge=0, le=1)


class OutlierArguments(ColumnArguments):
    method: Literal["iqr"] | None = None


class ImbalanceArguments(ColumnArguments):
    dominance_threshold: float | None = Field(default=None, ge=50, le=100)


class TargetArguments(ToolArguments):
    target_column: str


TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    "dataset_overview": dataset_overview,
    "missing_value_analysis": missing_value_analysis,
    "numerical_summary": numerical_summary,
    "categorical_summary": categorical_summary,
    "correlation_analysis": correlation_analysis,
    "outlier_analysis": outlier_analysis,
    "duplicate_analysis": duplicate_analysis,
    "cardinality_analysis": cardinality_analysis,
    "distribution_analysis": distribution_analysis,
    "categorical_imbalance_analysis": categorical_imbalance_analysis,
    "datetime_analysis": datetime_analysis,
    "target_analysis": target_analysis,
}

TOOL_ARGUMENT_SCHEMAS: dict[str, type[ToolArguments]] = {
    "dataset_overview": NoArguments,
    "missing_value_analysis": MissingArguments,
    "numerical_summary": ColumnArguments,
    "categorical_summary": CategoricalArguments,
    "correlation_analysis": CorrelationArguments,
    "outlier_analysis": OutlierArguments,
    "duplicate_analysis": NoArguments,
    "cardinality_analysis": ColumnArguments,
    "distribution_analysis": ColumnArguments,
    "categorical_imbalance_analysis": ImbalanceArguments,
    "datetime_analysis": ColumnArguments,
    "target_analysis": TargetArguments,
}


def validate_tool_call(tool_name: str, arguments: dict[str, Any] | None, df: pd.DataFrame) -> dict[str, Any]:
    """Validate tool name, schema, and all requested column names."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool '{tool_name}'. Allowed tools: {', '.join(TOOL_REGISTRY)}")
    try:
        validated = TOOL_ARGUMENT_SCHEMAS[tool_name].model_validate(arguments or {}).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise ValueError(f"Invalid arguments for {tool_name}: {exc}") from exc
    for column in validated.get("columns", []):
        if column not in df.columns:
            raise ValueError(f"Unknown column '{column}'.")
    target = validated.get("target_column")
    if target is not None and target not in df.columns:
        raise ValueError(f"Unknown target column '{target}'.")
    return validated


def execute_tool(tool_name: str, df: pd.DataFrame, arguments: dict[str, Any] | None = None) -> dict:
    """Execute one validated allowlisted tool with safe error reporting."""
    try:
        clean_args = validate_tool_call(tool_name, arguments, df)
        return TOOL_REGISTRY[tool_name](df, **clean_args)
    except Exception as exc:  # Tool failures should not crash the graph.
        return {"tool_name": tool_name, "status": "error", "error": str(exc), "display_result": {}, "llm_summary": {"error": str(exc)}, "insights": []}
