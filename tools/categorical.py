"""Categorical frequency, cardinality, and imbalance analyses."""

from __future__ import annotations

import pandas as pd

from services.dataframe_service import is_identifier_column
from tools.common import tool_result
from utils.validators import require_columns


def _categorical_columns(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    available = [str(c) for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])) and not is_identifier_column(df[c], str(c))]
    if columns is None:
        return available
    require_columns(df, columns)
    return [column for column in columns if column in available]


def categorical_summary(df: pd.DataFrame, columns: list[str] | None = None, top_n: int = 10, **_: object) -> dict:
    """Summarize top category frequencies while bounding high-cardinality output."""
    selected = _categorical_columns(df, columns)
    top_n = min(max(int(top_n), 1), 25)
    records = {}
    for column in selected:
        counts = df[column].fillna("<MISSING>").astype(str).value_counts().head(top_n)
        total = len(df)
        items = [{"value": value, "count": int(count), "percentage": count / total * 100 if total else 0} for value, count in counts.items()]
        records[column] = {"unique_count": int(df[column].nunique(dropna=True)), "most_common": items[0] if items else None, "top_categories": items}
    insights = [f"In {column}, '{data['most_common']['value']}' represents {data['most_common']['percentage']:.2f}% of rows." for column, data in records.items() if data["most_common"]]
    return tool_result("categorical_summary", {"columns": records, "top_n": top_n}, {"columns": records}, insights[:10])


def cardinality_analysis(df: pd.DataFrame, columns: list[str] | None = None, **_: object) -> dict:
    """Measure category cardinality and flag heuristic identifier candidates."""
    selected = _categorical_columns(df, columns) if columns else [str(c) for c in df.columns]
    if columns:
        require_columns(df, columns)
    rows = len(df)
    records = {}
    for column in selected:
        unique = int(df[column].nunique(dropna=True))
        ratio = unique / rows if rows else 0
        records[column] = {"unique_count": unique, "unique_ratio": ratio, "level": "low" if unique <= 10 else "very_high" if ratio >= 0.5 else "moderate", "likely_identifier": is_identifier_column(df[column], column)}
    insights = [f"{column} is {data['unique_ratio'] * 100:.1f}% unique and may be an identifier." for column, data in records.items() if data["likely_identifier"]]
    return tool_result("cardinality_analysis", {"columns": records}, {"columns": records}, insights or ["No strong identifier pattern was found among analyzed columns."])


def categorical_imbalance_analysis(df: pd.DataFrame, columns: list[str] | None = None, dominance_threshold: float = 80.0, **_: object) -> dict:
    """Detect categorical columns dominated by one observed value."""
    selected = _categorical_columns(df, columns)
    threshold = min(max(float(dominance_threshold), 50.0), 100.0)
    records = {}
    for column in selected:
        counts = df[column].dropna().astype(str).value_counts()
        if counts.empty:
            continue
        percentage = float(counts.iloc[0] / counts.sum() * 100)
        records[column] = {"dominant_value": counts.index[0], "dominant_count": int(counts.iloc[0]), "dominant_percentage": percentage, "is_imbalanced": percentage >= threshold}
    insights = [f"{column}='{data['dominant_value']}' accounts for {data['dominant_percentage']:.2f}% of non-missing values." for column, data in records.items() if data["is_imbalanced"]]
    return tool_result("categorical_imbalance_analysis", {"threshold": threshold, "columns": records}, {"imbalanced_columns": {k: v for k, v in records.items() if v["is_imbalanced"]}}, insights or [f"No analyzed categorical column exceeds the {threshold:.0f}% dominance threshold."])
