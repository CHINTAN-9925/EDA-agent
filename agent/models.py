"""Validated schemas for intentionally exposed LLM decisions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalysisStep(BaseModel):
    tool_name: str
    reason: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AnalysisPlan(BaseModel):
    dataset_summary: str
    steps: list[AnalysisStep] = Field(min_length=1, max_length=12)


class EvaluationDecision(BaseModel):
    observation: str
    continue_analysis: bool
    next_tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str


class ToolSelection(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str


class DatasetAnswer(BaseModel):
    answer: str
