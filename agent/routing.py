"""Conditional edge functions for the agent loop."""

from __future__ import annotations

from typing import Literal

from agent.state import EDAState


def route_after_selection(state: EDAState) -> Literal["execute_analysis", "generate_report"]:
    """Execute a valid selection, otherwise proceed to reporting."""
    return "execute_analysis" if state.get("next_action") == "execute" and state.get("current_tool") else "generate_report"


def should_continue(state: EDAState) -> Literal["select_analysis", "generate_report"]:
    """Enforce both the evaluator decision and the hard iteration limit."""
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return "generate_report"
    if state.get("continue_analysis", False):
        return "select_analysis"
    return "generate_report"
