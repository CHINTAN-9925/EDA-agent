"""Dataset overview and duplicate analysis tools."""

from __future__ import annotations

import pandas as pd

from tools.common import tool_result


def dataset_overview(df: pd.DataFrame, **_: object) -> dict:
    """Return shape, schema, memory use, and duplicate count."""
    duplicates = int(df.duplicated().sum())
    result = {
        "rows": len(df), "columns": len(df.columns), "column_names": list(map(str, df.columns)),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "memory_bytes": int(df.memory_usage(deep=True).sum()), "duplicate_rows": duplicates,
    }
    insights = [f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns."]
    if duplicates:
        insights.append(f"{duplicates:,} rows are exact duplicates.")
    return tool_result("dataset_overview", result, result, insights)


def duplicate_analysis(df: pd.DataFrame, **_: object) -> dict:
    """Measure exact duplicated rows over the full dataset."""
    count = int(df.duplicated().sum())
    percentage = round(count / len(df) * 100, 3) if len(df) else 0.0
    result = {"duplicate_count": count, "duplicate_percentage": percentage}
    insights = [f"{count:,} rows ({percentage:.2f}%) are exact duplicates."]
    return tool_result("duplicate_analysis", result, result, insights)
