"""Numerical correlation analysis."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from services.dataframe_service import is_identifier_column
from tools.common import tool_result
from utils.validators import require_columns


def correlation_analysis(df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson", high_threshold: float = 0.7, **_: object) -> dict:
    """Calculate a correlation matrix and extract non-self variable pairs."""
    method = method.lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be 'pearson' or 'spearman'.")
    numeric = [str(c) for c in df.select_dtypes(include=np.number).columns if not is_identifier_column(df[c], str(c))]
    if columns is not None:
        require_columns(df, columns)
        numeric = [column for column in columns if column in numeric]
    matrix = df[numeric].replace([np.inf, -np.inf], np.nan).corr(method=method) if numeric else pd.DataFrame()
    pairs = []
    for left, right in itertools.combinations(numeric, 2):
        value = matrix.loc[left, right]
        if pd.notna(value):
            pairs.append({"column_1": left, "column_2": right, "correlation": float(value)})
    positive = sorted((p for p in pairs if p["correlation"] >= 0), key=lambda p: p["correlation"], reverse=True)
    negative = sorted((p for p in pairs if p["correlation"] < 0), key=lambda p: p["correlation"])
    high = sorted((p for p in pairs if abs(p["correlation"]) >= high_threshold), key=lambda p: abs(p["correlation"]), reverse=True)
    matrix_dict = {str(c): {str(i): value for i, value in matrix[c].items()} for c in matrix.columns}
    insights = [f"{p['column_1']} and {p['column_2']} have {method} correlation {p['correlation']:.3f}." for p in high[:5]]
    return tool_result("correlation_analysis", {"method": method, "matrix": matrix_dict, "pairs": pairs, "highly_correlated_pairs": high}, {"method": method, "strongest_positive": positive[:5], "strongest_negative": negative[:5], "high_correlation_pairs": len(high)}, insights or ["No analyzed numerical pair met the high-correlation threshold."])
