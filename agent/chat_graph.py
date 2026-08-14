"""A small safe LangGraph workflow for post-EDA dataset questions."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent.models import ToolSelection
from agent.nodes import TOOL_DESCRIPTIONS
from services.dataframe_service import compact_profile, profile_dataframe
from services.llm import LLMClient
from tools.registry import TOOL_REGISTRY, execute_tool, validate_tool_call
from utils.validators import json_safe


class ChatState(TypedDict, total=False):
    dataframe: pd.DataFrame
    question: str
    profile: dict[str, Any]
    target_column: str | None
    selection: dict[str, Any]
    tool_result: dict[str, Any]
    answer: str


def build_chat_graph(llm: LLMClient):
    """Compile question → safe tool choice → execution → grounded explanation."""
    def select_tool(state: ChatState) -> dict:
        profile = state.get("profile") or profile_dataframe(state["dataframe"])
        prompt = f"""Choose exactly one deterministic tool to answer the dataset question. Never request raw rows or code.
Question: {state['question']}
Profile: {json.dumps(compact_profile(profile))}
Available tools: {json.dumps(TOOL_DESCRIPTIONS)}
Explicit target: {state.get('target_column') or 'None'}
Return JSON: {{"tool_name":"allowlisted name","arguments":{{}},"reason":"brief reason"}}"""
        selection = llm.complete_structured("You route dataset questions to one safe deterministic EDA tool. Return JSON only.", prompt, ToolSelection)
        if selection.tool_name == "target_analysis" and not state.get("target_column"):
            selection = ToolSelection(tool_name="dataset_overview", reason="Fallback to a safe overview tool.")
        try:
            arguments = dict(selection.arguments)
            if selection.tool_name == "target_analysis":
                arguments["target_column"] = state["target_column"]
            selection.arguments = validate_tool_call(selection.tool_name, arguments, state["dataframe"])
        except ValueError:
            selection = ToolSelection(tool_name="dataset_overview", reason="The proposed call was invalid; using a safe overview.")
        return {"selection": selection.model_dump(), "profile": profile}

    def run_tool(state: ChatState) -> dict:
        selection = state["selection"]
        return {"tool_result": execute_tool(selection["tool_name"], state["dataframe"], selection.get("arguments", {}))}

    def explain(state: ChatState) -> dict:
        context = {"question": state["question"], "tool": state["selection"], "verified_result": state["tool_result"].get("llm_summary"), "deterministic_insights": state["tool_result"].get("insights")}
        answer = llm.complete_text("Answer using only the deterministic tool result. Never invent values. Clearly state limitations. Do not claim causation.", json.dumps(json_safe(context)))
        return {"answer": answer}

    builder = StateGraph(ChatState)
    builder.add_node("select_tool", select_tool)
    builder.add_node("execute_tool", run_tool)
    builder.add_node("explain_result", explain)
    builder.add_edge(START, "select_tool")
    builder.add_edge("select_tool", "execute_tool")
    builder.add_edge("execute_tool", "explain_result")
    builder.add_edge("explain_result", END)
    return builder.compile()
