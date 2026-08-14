"""Tests for core deterministic tools and registry safeguards."""

import pandas as pd
import pytest

from tools.correlation import correlation_analysis
from tools.missing import missing_value_analysis
from tools.numerical import numerical_summary, outlier_analysis
from tools.registry import execute_tool, validate_tool_call


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 100, None],
        "y": [2, 4, 6, 8, 200, 12],
        "category": ["a", "a", "b", "b", "b", "b"],
    })


def test_missing_value_analysis(sample_frame: pd.DataFrame) -> None:
    result = missing_value_analysis(sample_frame)
    x = next(row for row in result["display_result"]["columns"] if row["column"] == "x")
    assert x["missing_count"] == 1
    assert x["missing_percentage"] == pytest.approx(16.667, abs=0.001)


def test_numerical_summary_uses_deterministic_values(sample_frame: pd.DataFrame) -> None:
    result = numerical_summary(sample_frame)
    assert result["display_result"]["statistics"]["x"]["median"] == 3
    assert result["display_result"]["statistics"]["x"]["max"] == 100


def test_correlation_extracts_pairs(sample_frame: pd.DataFrame) -> None:
    result = correlation_analysis(sample_frame)
    pairs = result["display_result"]["pairs"]
    assert len(pairs) == 1
    assert pairs[0]["column_1"] == "x"
    assert pairs[0]["column_2"] == "y"
    assert pairs[0]["correlation"] == pytest.approx(1.0)


def test_iqr_outlier_detection() -> None:
    frame = pd.DataFrame({"value": [10] * 20 + [1000]})
    result = outlier_analysis(frame)
    assert result["display_result"]["columns"]["value"]["outlier_count"] == 1


def test_registry_rejects_unknown_tool_and_columns(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        validate_tool_call("run_python", {}, sample_frame)
    with pytest.raises(ValueError, match="Unknown column"):
        validate_tool_call("numerical_summary", {"columns": ["secret"]}, sample_frame)
    error = execute_tool("run_python", sample_frame)
    assert error["status"] == "error"


def test_registry_rejects_extra_arguments(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Invalid arguments"):
        validate_tool_call("numerical_summary", {"code": "print('unsafe')"}, sample_frame)


def test_registry_uses_tool_specific_argument_schemas(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Invalid arguments"):
        validate_tool_call("dataset_overview", {"method": "pearson"}, sample_frame)
    with pytest.raises(ValueError, match="Invalid arguments"):
        validate_tool_call("target_analysis", {}, sample_frame)
    assert validate_tool_call("correlation_analysis", {"method": "spearman"}, sample_frame) == {"method": "spearman"}
