"""Numerical summaries, distributions, and IQR outlier detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.dataframe_service import is_identifier_column
from tools.common import tool_result
from utils.validators import require_columns


def _numeric_columns(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    available = [str(c) for c in df.select_dtypes(include=np.number).columns if not is_identifier_column(df[c], str(c))]
    if columns is None:
        return available
    require_columns(df, columns)
    return [column for column in columns if column in available]


def numerical_summary(df: pd.DataFrame, columns: list[str] | None = None, **_: object) -> dict:
    """Calculate robust descriptive statistics on full numerical columns."""
    selected = _numeric_columns(df, columns)
    records: dict[str, dict] = {}
    for column in selected:
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        records[column] = {
            "count": int(values.count()), "mean": values.mean(), "median": values.median(),
            "std": values.std(), "min": values.min(), "25%": values.quantile(0.25),
            "50%": values.quantile(0.5), "75%": values.quantile(0.75), "max": values.max(),
            "skewness": values.skew(), "kurtosis": values.kurt(),
        }
    insights = []
    for column, stats in records.items():
        skew = stats["skewness"]
        if pd.notna(skew) and abs(skew) >= 1:
            insights.append(f"{column} has measured skewness of {skew:.2f}.")
    return tool_result("numerical_summary", {"statistics": records}, {"statistics": records}, insights or [f"Summarized {len(records)} numerical columns."])


def distribution_analysis(df: pd.DataFrame, columns: list[str] | None = None, **_: object) -> dict:
    """Measure numerical skew, quantiles, zeros, and negative values."""
    selected = _numeric_columns(df, columns)
    records = {}
    for column in selected:
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        count = len(values)
        records[column] = {
            "count": count, "skewness": values.skew(),
            "quantiles": {str(q): values.quantile(q) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
            "zero_percentage": float((values == 0).mean() * 100) if count else 0,
            "negative_percentage": float((values < 0).mean() * 100) if count else 0,
        }
    ranked = sorted(records.items(), key=lambda item: abs(item[1]["skewness"]) if pd.notna(item[1]["skewness"]) else -1, reverse=True)
    insights = [f"{name} has skewness {stats['skewness']:.2f}." for name, stats in ranked[:5] if pd.notna(stats["skewness"])]
    return tool_result("distribution_analysis", {"distributions": records, "chart_columns": [name for name, _ in ranked[:10]]}, {"most_skewed": dict(ranked[:10])}, insights)


def outlier_analysis(df: pd.DataFrame, columns: list[str] | None = None, method: str = "iqr", **_: object) -> dict:
    """Detect potential outliers with IQR bounds; detections are not labels of bad data."""
    if method.lower() != "iqr":
        raise ValueError("Only the safe 'iqr' outlier method is supported.")
    selected = _numeric_columns(df, columns)
    records = {}
    for column in selected:
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum()) if len(values) else 0
        records[column] = {"q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lower, "upper_bound": upper, "outlier_count": count, "outlier_percentage": count / len(values) * 100 if len(values) else 0.0}
    ranked = sorted(records.items(), key=lambda item: item[1]["outlier_percentage"], reverse=True)
    insights = [f"{name} has {stats['outlier_count']:,} potential IQR outliers ({stats['outlier_percentage']:.2f}%)." for name, stats in ranked[:5] if stats["outlier_count"]]
    return tool_result("outlier_analysis", {"method": "IQR (1.5×)", "columns": records}, {"highest_outlier_percentages": dict(ranked[:10]), "note": "Potential outliers are not automatically data errors."}, insights or ["No potential IQR outliers were detected in the analyzed numerical columns."])
