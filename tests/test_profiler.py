"""Tests for semantic DataFrame profiling."""

import pandas as pd

from services.dataframe_service import is_identifier_column, profile_dataframe


def test_profile_detects_types_quality_and_constant() -> None:
    frame = pd.DataFrame({
        "customer_id": [1001, 1002, 1003, 1004, 1005],
        "age": [20, 30, None, 40, 50],
        "segment": ["a", "a", "b", "b", "b"],
        "active": [True, False, True, True, False],
        "joined_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "constant": [1, 1, 1, 1, 1],
    })
    profile = profile_dataframe(frame)
    assert profile["shape"] == {"rows": 5, "columns": 6}
    assert "customer_id" in profile["possible_identifier_columns"]
    assert "age" in profile["numerical_columns"]
    assert "segment" in profile["categorical_columns"]
    assert "active" in profile["boolean_columns"]
    assert "joined_date" in profile["datetime_columns"]
    assert "constant" in profile["constant_columns"]
    assert profile["missing_counts"]["age"] == 1


def test_identifier_detection_is_heuristic_and_name_aware() -> None:
    assert is_identifier_column(pd.Series(["a", "b", "c", "d"]), "user_id")
    assert not is_identifier_column(pd.Series([1, 1, 2, 2]), "score")
