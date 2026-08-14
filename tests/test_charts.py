"""Tests for bounded deterministic charts."""

import pandas as pd

from tools.target import target_analysis
from visualization.charts import build_charts


def test_target_analysis_generates_classification_chart() -> None:
    frame = pd.DataFrame({"feature": [1, 2, 3, 4], "label": ["a", "a", "b", "b"]})
    charts = build_charts(frame, [target_analysis(frame, "label")])
    target_charts = [chart for chart in charts if chart["title"] == "Target distribution"]
    assert len(target_charts) == 1
    assert target_charts[0]["section"] == "Categorical"


def test_target_analysis_generates_regression_chart() -> None:
    frame = pd.DataFrame({"feature": range(100), "target": [value * 1.5 for value in range(100)]})
    charts = build_charts(frame, [target_analysis(frame, "target")])
    target_charts = [chart for chart in charts if chart["title"] == "Target distribution"]
    assert len(target_charts) == 1
    assert target_charts[0]["section"] == "Numerical"
