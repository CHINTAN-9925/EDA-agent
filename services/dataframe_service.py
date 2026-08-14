"""Dataset profiling and improved semantic type inference."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from utils.validators import json_safe


def is_identifier_column(series: pd.Series, name: str) -> bool:
    """Heuristically identify identifier-like columns; this is not a certainty."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    normalized_name = name.lower().strip().replace(" ", "_")
    name_hint = normalized_name == "id" or normalized_name.endswith("_id") or normalized_name.startswith("id_")
    sequence_like = False
    if is_numeric_dtype(non_null) and len(non_null) >= 20 and unique_ratio >= 0.98:
        sorted_values = pd.to_numeric(non_null, errors="coerce").dropna().sort_values()
        sequence_like = len(sorted_values) > 1 and sorted_values.diff().dropna().nunique() <= 2
    return bool(unique_ratio >= 0.98 and (name_hint or sequence_like))


def is_datetime_like(series: pd.Series, name: str) -> bool:
    """Conservatively detect datetime values without mutating the dataset."""
    if is_datetime64_any_dtype(series):
        return True
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    name_hint = any(token in name.lower() for token in ("date", "time", "timestamp", "created", "updated"))
    values = series.dropna().astype(str).head(100)
    if values.empty:
        return False
    format_signal = values.str.contains(r"[-/:]", regex=True).mean() >= 0.8
    if not name_hint and not format_signal:
        return False
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return bool(parsed.notna().mean() >= 0.9)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact full-dataset profile without involving an LLM."""
    rows = len(df)
    missing = df.isna().sum()
    unique = df.nunique(dropna=True)
    identifiers = [str(c) for c in df.columns if is_identifier_column(df[c], str(c))]
    booleans = [str(c) for c in df.columns if is_bool_dtype(df[c])]
    datetimes = [str(c) for c in df.columns if is_datetime_like(df[c], str(c))]
    numerical = [str(c) for c in df.columns if is_numeric_dtype(df[c]) and str(c) not in identifiers and str(c) not in booleans]
    categorical = [str(c) for c in df.columns if str(c) not in numerical + booleans + datetimes and str(c) not in identifiers]
    constants = [str(c) for c in df.columns if unique[c] <= 1]
    high_cardinality_text = [c for c in categorical if rows and unique[c] / rows > 0.5]

    profile = {
        "shape": {"rows": rows, "columns": len(df.columns)},
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "missing_counts": {str(c): int(missing[c]) for c in df.columns},
        "missing_percentages": {str(c): round(float(missing[c] / rows * 100), 3) if rows else 0.0 for c in df.columns},
        "unique_counts": {str(c): int(unique[c]) for c in df.columns},
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "boolean_columns": booleans,
        "datetime_columns": datetimes,
        "possible_identifier_columns": identifiers,
        "constant_columns": constants,
        "high_cardinality_text_columns": high_cardinality_text,
        "duplicated_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }
    return json_safe(profile)


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return only summarized metadata suitable for an LLM prompt."""
    return {
        "shape": profile["shape"],
        "dtypes": profile["dtypes"],
        "missing_percentages": profile["missing_percentages"],
        "unique_counts": profile["unique_counts"],
        "semantic_groups": {
            key: profile[key]
            for key in (
                "numerical_columns", "categorical_columns", "boolean_columns",
                "datetime_columns", "possible_identifier_columns", "constant_columns",
                "high_cardinality_text_columns",
            )
        },
        "duplicated_rows": profile["duplicated_rows"],
        "memory_bytes": profile["memory_bytes"],
    }
