"""Explicit LangGraph construction for iterative agentic EDA."""

from __future__ import annotations

from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent.nodes import EDANodes
from agent.routing import route_after_selection, should_continue
from agent.state import EDAState
from services.llm import LLMClient


def build_eda_graph(llm: LLMClient):
    """Compile the profile → plan → select → execute → evaluate loop."""
    nodes = EDANodes(llm)
    builder = StateGraph(EDAState)
    builder.add_node("profile_dataset", nodes.profile_dataset)
    builder.add_node("plan_analysis", nodes.plan_analysis)
    builder.add_node("select_analysis", nodes.select_analysis)
    builder.add_node("execute_analysis", nodes.execute_analysis)
    builder.add_node("evaluate_analysis", nodes.evaluate_analysis)
    builder.add_node("generate_report", nodes.generate_report)
    builder.add_edge(START, "profile_dataset")
    builder.add_edge("profile_dataset", "plan_analysis")
    builder.add_edge("plan_analysis", "select_analysis")
    builder.add_conditional_edges("select_analysis", route_after_selection)
    builder.add_edge("execute_analysis", "evaluate_analysis")
    builder.add_conditional_edges("evaluate_analysis", should_continue)
    builder.add_edge("generate_report", END)
    return builder.compile()


def initial_state(df: pd.DataFrame, max_iterations: int = 10, target_column: str | None = None, settings: dict[str, Any] | None = None) -> EDAState:
    """Build a complete initial graph state for one uploaded DataFrame."""
    return {
        "dataframe": df,
        "analysis_plan": [], "completed_analyses": [], "completed_tool_calls": [],
        "tool_results": [], "observations": [], "iteration_count": 0,
        "max_iterations": min(max(int(max_iterations), 1), 15),
        "continue_analysis": True, "next_action": "start", "final_report": None,
        "errors": [], "execution_trace": [], "target_column": target_column,
        "settings": settings or {}, "suggested_tool": None, "suggested_arguments": {},
    }
