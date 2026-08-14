"""Offline integration test for the explicit LangGraph workflow."""

import pandas as pd

from agent.graph import build_eda_graph, initial_state
from agent.models import AnalysisPlan, EvaluationDecision


class FakeLLM:
    """Schema-aware fake that never makes a network request."""

    def complete_structured(self, _system: str, _user: str, schema: type):
        if schema is AnalysisPlan:
            return AnalysisPlan.model_validate({
                "dataset_summary": "A small mixed dataset.",
                "steps": [{"tool_name": "missing_value_analysis", "reason": "Measure completeness.", "arguments": {}}],
            })
        if schema is EvaluationDecision:
            return EvaluationDecision(observation="One value is missing.", continue_analysis=False, reason="The focused plan is complete.")
        raise AssertionError(f"Unexpected schema: {schema}")

    def complete_text(self, _system: str, _user: str) -> str:
        return "# Dataset Overview\n\nVerified offline test report."


def test_graph_profiles_plans_executes_evaluates_and_reports() -> None:
    frame = pd.DataFrame({"value": [1.0, None, 3.0], "kind": ["a", "b", "b"]})
    graph = build_eda_graph(FakeLLM())
    result = graph.invoke(initial_state(frame, max_iterations=3), {"recursion_limit": 20})
    assert result["completed_analyses"] == ["missing_value_analysis"]
    assert result["iteration_count"] == 1
    assert result["final_report"].startswith("# Dataset Overview")
    nodes = [item["node"] for item in result["execution_trace"]]
    assert nodes == ["profile_dataset", "plan_analysis", "select_analysis", "execute_analysis", "evaluate_analysis", "generate_report"]
