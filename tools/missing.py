"""Missing-value analysis."""

from __future__ import annotations

import pandas as pd

from tools.common import tool_result


def missing_value_analysis(df: pd.DataFrame, significant_threshold: float = 20.0, **_: object) -> dict:
    """Calculate missing counts and percentages for every column."""
    threshold = min(max(float(significant_threshold), 0.0), 100.0)
    rows = len(df)
    records = []
    for column in df.columns:
        count = int(df[column].isna().sum())
        records.append({"column": str(column), "missing_count": count, "missing_percentage": round(count / rows * 100, 3) if rows else 0.0})
    records.sort(key=lambda item: item["missing_percentage"], reverse=True)
    significant = [item for item in records if item["missing_percentage"] >= threshold and item["missing_count"]]
    insights = [f"{item['column']} has {item['missing_percentage']:.2f}% missing values." for item in records[:5] if item["missing_count"]]
    if not insights:
        insights = ["No missing values were detected."]
    return tool_result("missing_value_analysis", {"columns": records, "significant_columns": significant, "threshold": threshold}, {"columns_with_missing": sum(r["missing_count"] > 0 for r in records), "highest_missing": records[:10], "significant_columns": significant[:10]}, insights)
