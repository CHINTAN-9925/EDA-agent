"""Datetime range, frequency, trend, and gap analysis."""

from __future__ import annotations

import pandas as pd

from services.dataframe_service import is_datetime_like
from tools.common import tool_result
from utils.validators import require_columns


def datetime_analysis(df: pd.DataFrame, columns: list[str] | None = None, **_: object) -> dict:
    """Analyze conservatively inferred dates without altering the source frame."""
    inferred = [str(c) for c in df.columns if is_datetime_like(df[c], str(c))]
    if columns is not None:
        require_columns(df, columns)
        inferred = [column for column in columns if column in inferred]
    records = {}
    for column in inferred:
        values = pd.to_datetime(df[column], errors="coerce", utc=True).dropna().sort_values()
        if values.empty:
            continue
        normalized = values.dt.normalize().drop_duplicates().sort_values()
        gaps = normalized.diff().dt.total_seconds().div(86_400).dropna()
        monthly = values.dt.to_period("M").astype(str).value_counts().sort_index().tail(36)
        records[column] = {
            "minimum": values.min(), "maximum": values.max(),
            "range_days": (values.max() - values.min()).total_seconds() / 86_400,
            "unique_dates": int(normalized.nunique()),
            "largest_observed_gap_days": gaps.max() if not gaps.empty else 0,
            "monthly_counts": {period: int(count) for period, count in monthly.items()},
        }
    insights = [f"{column} spans {data['minimum']} to {data['maximum']} ({data['range_days']:.0f} days)." for column, data in records.items()]
    return tool_result("datetime_analysis", {"columns": records}, {"columns": records}, insights)
