"""Explicit target-column analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.common import tool_result
from utils.validators import require_columns


def target_analysis(df: pd.DataFrame, target_column: str | None = None, **_: object) -> dict:
    """Analyze a user-selected target; never assumes the last column is a target."""
    if not target_column:
        raise ValueError("target_analysis requires a user-selected target_column.")
    require_columns(df, [target_column])
    target = df[target_column]
    unique = target.nunique(dropna=True)
    is_classification = (not pd.api.types.is_numeric_dtype(target)) or unique <= min(20, max(2, int(len(df) * 0.05)))
    if is_classification:
        counts = target.fillna("<MISSING>").astype(str).value_counts().head(25)
        total = len(df)
        distribution = [{"value": k, "count": int(v), "percentage": v / total * 100 if total else 0} for k, v in counts.items()]
        display = {"target_column": target_column, "task_type": "classification", "class_distribution": distribution, "unique_classes": int(unique)}
        insights = [f"The largest observed class '{distribution[0]['value']}' represents {distribution[0]['percentage']:.2f}% of rows."] if distribution else []
    else:
        values = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric = df.select_dtypes(include=np.number).drop(columns=[target_column], errors="ignore")
        correlations = numeric.corrwith(values).dropna().sort_values(key=abs, ascending=False).head(10)
        display = {"target_column": target_column, "task_type": "regression", "distribution": {"count": int(values.count()), "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(), "skewness": values.skew()}, "feature_correlations": {str(k): float(v) for k, v in correlations.items()}}
        insights = [f"The strongest measured numerical relationship with {target_column} is {correlations.index[0]} ({correlations.iloc[0]:.3f})."] if len(correlations) else []
    return tool_result("target_analysis", display, display, insights)
