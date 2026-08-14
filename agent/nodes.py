"""Meaningful LangGraph nodes for profiling, planning, tools, evaluation, and reporting."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent.models import AnalysisPlan, EvaluationDecision
from agent.state import EDAState
from prompts.evaluator import EVALUATOR_SYSTEM_PROMPT, evaluator_user_prompt
from prompts.planner import PLANNER_SYSTEM_PROMPT, planner_user_prompt
from prompts.reporter import REPORTER_SYSTEM_PROMPT, reporter_user_prompt
from services.dataframe_service import compact_profile, profile_dataframe
from services.llm import LLMClient, LLMServiceError
from tools.registry import TOOL_REGISTRY, execute_tool, validate_tool_call
from utils.validators import json_safe


TOOL_DESCRIPTIONS = {
    "dataset_overview": "shape, schema, memory, duplicates",
    "missing_value_analysis": "missing count and percentage by column",
    "numerical_summary": "full-data descriptive numerical statistics",
    "categorical_summary": "bounded top-category frequencies",
    "correlation_analysis": "Pearson or Spearman numerical relationships",
    "outlier_analysis": "potential IQR outliers",
    "duplicate_analysis": "exact duplicate row prevalence",
    "cardinality_analysis": "unique ratios and likely identifiers",
    "distribution_analysis": "skew, quantiles, zeros, and negatives",
    "categorical_imbalance_analysis": "dominant categorical levels",
    "datetime_analysis": "date ranges, monthly counts, observed gaps",
    "target_analysis": "user-selected classification/regression target",
}


def _call_key(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(json_safe(arguments), sort_keys=True, separators=(",", ":"))
    return f"{tool_name}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _normalized_arguments(tool: str, arguments: dict[str, Any], state: EDAState) -> dict[str, Any]:
    """Apply trusted UI settings and validate one model-proposed call."""
    normalized = dict(arguments)
    if tool == "correlation_analysis":
        normalized["method"] = state.get("settings", {}).get("correlation_method", normalized.get("method", "pearson"))
    if tool == "categorical_summary":
        normalized["top_n"] = state.get("settings", {}).get("top_categories", normalized.get("top_n", 10))
    if tool == "outlier_analysis":
        normalized["method"] = state.get("settings", {}).get("outlier_method", "iqr")
    if tool == "target_analysis" and state.get("target_column"):
        normalized["target_column"] = state["target_column"]
    return validate_tool_call(tool, normalized, state["dataframe"])


def _fallback_plan(profile: dict[str, Any], target_column: str | None, settings: dict[str, Any]) -> AnalysisPlan:
    """Create a safe dataset-dependent plan when model planning fails."""
    steps = [
        {"tool_name": "dataset_overview", "reason": "Establish verified dataset dimensions and duplicate prevalence.", "arguments": {}},
        {"tool_name": "missing_value_analysis", "reason": "Measure data completeness across the schema.", "arguments": {}},
    ]
    if profile.get("numerical_columns"):
        steps.extend([
            {"tool_name": "numerical_summary", "reason": "Numerical features are present and need deterministic summaries.", "arguments": {}},
            {"tool_name": "distribution_analysis", "reason": "Assess numerical shapes and skewness.", "arguments": {}},
            {"tool_name": "outlier_analysis", "reason": "Check numerical features for potential IQR extremes.", "arguments": {"method": settings.get("outlier_method", "iqr")}},
        ])
        if len(profile["numerical_columns"]) >= 2:
            steps.append({"tool_name": "correlation_analysis", "reason": "Multiple non-identifier numerical features permit relationship analysis.", "arguments": {"method": settings.get("correlation_method", "pearson")}})
    if profile.get("categorical_columns") or profile.get("boolean_columns"):
        steps.extend([
            {"tool_name": "categorical_summary", "reason": "Categorical features are present.", "arguments": {"top_n": settings.get("top_categories", 10)}},
            {"tool_name": "categorical_imbalance_analysis", "reason": "Check whether categorical levels are strongly dominant.", "arguments": {}},
        ])
    if profile.get("datetime_columns"):
        steps.append({"tool_name": "datetime_analysis", "reason": "Datetime-like columns support temporal range and gap analysis.", "arguments": {}})
    if profile.get("possible_identifier_columns") or profile.get("high_cardinality_text_columns"):
        steps.append({"tool_name": "cardinality_analysis", "reason": "Validate high-cardinality and identifier-like fields.", "arguments": {}})
    if target_column:
        steps.append({"tool_name": "target_analysis", "reason": "The user explicitly selected a target.", "arguments": {"target_column": target_column}})
    summary = f"Dataset with {profile['shape']['rows']:,} rows and {profile['shape']['columns']:,} columns."
    return AnalysisPlan.model_validate({"dataset_summary": summary, "steps": steps[:12]})


class EDANodes:
    """Node collection bound to a configured LLM client."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def profile_dataset(self, state: EDAState) -> dict[str, Any]:
        profile = profile_dataframe(state["dataframe"])
        trace = list(state.get("execution_trace", [])) + [{"node": "profile_dataset", "detail": f"Dataset: {profile['shape']['rows']:,} rows × {profile['shape']['columns']:,} columns"}]
        return {"dataset_profile": profile, "execution_trace": trace}

    def plan_analysis(self, state: EDAState) -> dict[str, Any]:
        profile = compact_profile(state["dataset_profile"])
        settings = state.get("settings", {})
        errors = list(state.get("errors", []))
        try:
            plan = self.llm.complete_structured(
                PLANNER_SYSTEM_PROMPT,
                planner_user_prompt(json.dumps(profile), json.dumps(TOOL_DESCRIPTIONS), state.get("target_column"), json.dumps(settings)),
                AnalysisPlan,
            )
            valid_steps = []
            for step in plan.steps:
                if step.tool_name not in TOOL_REGISTRY or (step.tool_name == "target_analysis" and not state.get("target_column")):
                    continue
                try:
                    step.arguments = _normalized_arguments(step.tool_name, step.arguments, state)
                    valid_steps.append(step)
                except ValueError as exc:
                    errors.append(f"Planner call skipped ({step.tool_name}): {exc}")
            if not valid_steps:
                raise LLMServiceError("Planner selected no valid tools.")
            plan = AnalysisPlan(dataset_summary=plan.dataset_summary, steps=valid_steps)
            detail = f"Agent planned {len(valid_steps)} dataset-specific analyses."
        except Exception as exc:
            errors.append(f"Planner fallback used: {exc}")
            plan = _fallback_plan(state["dataset_profile"], state.get("target_column"), settings)
            detail = f"Safe deterministic fallback planned {len(plan.steps)} analyses."
        trace = list(state.get("execution_trace", [])) + [{"node": "plan_analysis", "detail": detail}]
        return {"analysis_plan": [step.model_dump() for step in plan.steps], "dataset_summary": plan.dataset_summary, "errors": errors, "execution_trace": trace, "suggested_tool": None}

    def select_analysis(self, state: EDAState) -> dict[str, Any]:
        completed = set(state.get("completed_tool_calls", []))
        errors = list(state.get("errors", []))
        candidates: list[dict[str, Any]] = []
        if state.get("suggested_tool"):
            candidates.append({"tool_name": state["suggested_tool"], "arguments": state.get("suggested_arguments", {}), "reason": state.get("suggested_reason", "Evaluator requested follow-up.")})
        candidates.extend(state.get("analysis_plan", []))
        selected = None
        for candidate in candidates:
            tool = candidate.get("tool_name")
            if tool not in TOOL_REGISTRY or (tool == "target_analysis" and not state.get("target_column")):
                continue
            try:
                arguments = _normalized_arguments(tool, candidate.get("arguments") or {}, state)
            except ValueError as exc:
                errors.append(f"Agent call skipped ({tool}): {exc}")
                continue
            if _call_key(tool, arguments) not in completed:
                selected = {**candidate, "arguments": arguments}
                break
        if selected is None or state.get("iteration_count", 0) >= state.get("max_iterations", 10):
            return {"current_tool": None, "next_action": "finish", "suggested_tool": None, "errors": errors}
        trace = list(state.get("execution_trace", [])) + [{"node": "select_analysis", "tool": selected["tool_name"], "reason": selected.get("reason", "Selected from plan.")}]
        return {"current_tool": selected["tool_name"], "current_tool_args": selected["arguments"], "current_reason": selected.get("reason", ""), "next_action": "execute", "suggested_tool": None, "errors": errors, "execution_trace": trace}

    def execute_analysis(self, state: EDAState) -> dict[str, Any]:
        tool = state["current_tool"]
        arguments = state.get("current_tool_args", {})
        result = execute_tool(tool, state["dataframe"], arguments)
        call_key = _call_key(tool, arguments)
        errors = list(state.get("errors", []))
        if result.get("status") == "error":
            errors.append(f"{tool}: {result.get('error')}")
        trace = list(state.get("execution_trace", [])) + [{"node": "execute_analysis", "tool": tool, "status": result.get("status")}]
        return {
            "tool_results": list(state.get("tool_results", [])) + [result],
            "completed_analyses": list(state.get("completed_analyses", [])) + [tool],
            "completed_tool_calls": list(state.get("completed_tool_calls", [])) + [call_key],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "errors": errors,
            "execution_trace": trace,
        }

    def evaluate_analysis(self, state: EDAState) -> dict[str, Any]:
        latest = state["tool_results"][-1]
        remaining = state.get("max_iterations", 10) - state.get("iteration_count", 0)
        context = {
            "profile": compact_profile(state["dataset_profile"]),
            "original_plan": state.get("analysis_plan", []),
            "completed_tools": state.get("completed_analyses", []),
            "latest_tool": latest.get("tool_name"),
            "latest_result": latest.get("llm_summary", {}),
            "deterministic_insights": latest.get("insights", []),
            "remaining_iterations": remaining,
            "available_tools": list(TOOL_REGISTRY),
        }
        errors = list(state.get("errors", []))
        try:
            decision = self.llm.complete_structured(EVALUATOR_SYSTEM_PROMPT, evaluator_user_prompt(json.dumps(json_safe(context))), EvaluationDecision)
            if decision.next_tool not in TOOL_REGISTRY:
                decision.next_tool = None
                decision.continue_analysis = False
            if decision.next_tool and _call_key(decision.next_tool, decision.arguments) in set(state.get("completed_tool_calls", [])):
                decision.next_tool = None
                decision.continue_analysis = False
                decision.reason = "The suggested call was already completed with identical parameters."
        except Exception as exc:
            errors.append(f"Evaluator fallback used: {exc}")
            uncompleted = [step for step in state.get("analysis_plan", []) if _call_key(step["tool_name"], step.get("arguments", {})) not in set(state.get("completed_tool_calls", []))]
            next_step = uncompleted[0] if uncompleted and remaining > 0 else None
            facts = latest.get("insights", [])
            decision = EvaluationDecision(
                observation=" ".join(facts[:3]) or f"{latest.get('tool_name')} completed with status {latest.get('status')}.",
                continue_analysis=next_step is not None,
                next_tool=next_step["tool_name"] if next_step else None,
                arguments=next_step.get("arguments", {}) if next_step else {},
                reason=next_step.get("reason", "Analysis plan is complete.") if next_step else "No unexecuted planned analysis remains.",
            )
        if remaining <= 0:
            decision.continue_analysis = False
            decision.next_tool = None
            decision.reason = "Maximum agent iterations reached; generating the report."
        trace = list(state.get("execution_trace", [])) + [{"node": "evaluate_analysis", "observation": decision.observation, "decision": decision.reason}]
        return {"observations": list(state.get("observations", [])) + [decision.observation], "continue_analysis": decision.continue_analysis, "suggested_tool": decision.next_tool, "suggested_arguments": decision.arguments, "suggested_reason": decision.reason, "errors": errors, "execution_trace": trace}

    def generate_report(self, state: EDAState) -> dict[str, Any]:
        context = {
            "dataset_summary": state.get("dataset_summary"),
            "profile": compact_profile(state["dataset_profile"]),
            "performed_analyses": [result.get("tool_name") for result in state.get("tool_results", [])],
            "verified_tool_results": [{"tool": result.get("tool_name"), "facts": result.get("llm_summary"), "deterministic_insights": result.get("insights")} for result in state.get("tool_results", [])],
            "evaluator_observations": state.get("observations", []),
            "errors": state.get("errors", []),
        }
        errors = list(state.get("errors", []))
        try:
            report = self.llm.complete_text(REPORTER_SYSTEM_PROMPT, reporter_user_prompt(json.dumps(json_safe(context))))
        except Exception as exc:
            errors.append(f"Reporter fallback used: {exc}")
            report = _fallback_report(state)
        trace = list(state.get("execution_trace", [])) + [{"node": "generate_report", "detail": "Final report generated."}]
        return {"final_report": report, "next_action": "finished", "errors": errors, "execution_trace": trace}


def _fallback_report(state: EDAState) -> str:
    """Produce a useful facts-only report if the report LLM is unavailable."""
    profile = state["dataset_profile"]
    lines = ["# Dataset Overview", f"The dataset contains **{profile['shape']['rows']:,} rows** and **{profile['shape']['columns']:,} columns**.", "", "# Verified Findings"]
    insights = [fact for result in state.get("tool_results", []) for fact in result.get("insights", [])]
    lines.extend([f"- {fact}" for fact in insights] or ["- No deterministic findings were produced."])
    lines.extend(["", "# Recommended Next Steps", "Review the measured findings in context before changing or removing data. Validate missing-value treatment, suspected identifiers, and potential outliers with domain knowledge."])
    return "\n".join(lines)
