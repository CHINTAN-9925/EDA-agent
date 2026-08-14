"""Validation and JSON-safety helpers shared by the application."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Validate and return requested DataFrame columns."""
    invalid = [column for column in columns if column not in df.columns]
    if invalid:
        raise ValueError(f"Unknown columns: {', '.join(invalid)}")
    return columns


def json_safe(value: Any) -> Any:
    """Recursively convert pandas/NumPy values to strict JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value
