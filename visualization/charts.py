"""Bounded, deterministic Plotly visualizations for EDA results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MAX_CHART_COLUMNS = 10
MAX_CHART_ROWS = 10_000


def _sample(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(df) > MAX_CHART_ROWS:
        return df.sample(MAX_CHART_ROWS, random_state=42), True
    return df, False


def build_charts(df: pd.DataFrame, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create only useful charts for analyses that actually ran."""
    charts: list[dict[str, Any]] = []
    by_tool = {item.get("tool_name"): item for item in tool_results if item.get("status") == "success"}
    sampled, was_sampled = _sample(df)
    suffix = " (10,000-row visualization sample)" if was_sampled else ""

    missing = by_tool.get("missing_value_analysis", {}).get("display_result", {}).get("columns", [])
    missing = [item for item in missing if item.get("missing_count", 0) > 0][:MAX_CHART_COLUMNS]
    if missing:
        figure = px.bar(pd.DataFrame(missing), x="column", y="missing_percentage", title="Missing values by column")
        figure.update_yaxes(title="Missing (%)")
        charts.append({"section": "Data Quality", "title": "Missing values", "figure": figure, "note": "Calculated over the full dataset."})

    distribution = by_tool.get("distribution_analysis", {}).get("display_result", {})
    for column in distribution.get("chart_columns", [])[:4]:
        if column in sampled:
            figure = px.histogram(sampled, x=column, nbins=40, title=f"Distribution of {column}{suffix}", marginal="box")
            charts.append({"section": "Numerical", "title": f"{column} distribution", "figure": figure, "note": "Visualization sampled; reported statistics use all rows." if was_sampled else "Full-data visualization."})

    corr = by_tool.get("correlation_analysis", {}).get("display_result", {})
    matrix = corr.get("matrix", {})
    if matrix:
        columns = list(matrix)[:MAX_CHART_COLUMNS]
        z = [[matrix.get(column, {}).get(row) for column in columns] for row in columns]
        figure = go.Figure(go.Heatmap(z=z, x=columns, y=columns, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True, texttemplate="%{z:.2f}"))
        figure.update_layout(title=f"{corr.get('method', 'Pearson').title()} correlation heatmap")
        charts.append({"section": "Relationships", "title": "Correlation heatmap", "figure": figure, "note": "Calculated over the full dataset; correlation does not imply causation."})

    categories = by_tool.get("categorical_summary", {}).get("display_result", {}).get("columns", {})
    for column, summary in list(categories.items())[:4]:
        values = pd.DataFrame(summary.get("top_categories", []))
        if not values.empty:
            figure = px.bar(values, x="value", y="count", title=f"Top categories: {column}")
            charts.append({"section": "Categorical", "title": f"{column} categories", "figure": figure, "note": "Counts use the full dataset; only top categories are displayed."})

    outliers = by_tool.get("outlier_analysis", {}).get("display_result", {}).get("columns", {})
    ranked = sorted(outliers, key=lambda name: outliers[name].get("outlier_percentage", 0), reverse=True)
    for column in ranked[:3]:
        if column in sampled and pd.api.types.is_numeric_dtype(sampled[column]):
            values = sampled[[column]].replace([np.inf, -np.inf], np.nan)
            figure = px.box(values, y=column, points="outliers", title=f"Potential extremes: {column}{suffix}")
            charts.append({"section": "Outliers", "title": f"{column} box plot", "figure": figure, "note": "Potential extremes are not automatically data errors."})

    target = by_tool.get("target_analysis", {}).get("display_result", {})
    target_column = target.get("target_column")
    if target_column in sampled.columns and target.get("task_type") == "classification":
        values = pd.DataFrame(target.get("class_distribution", []))
        if not values.empty:
            figure = px.bar(values, x="value", y="count", title=f"Target distribution: {target_column}")
            charts.append({"section": "Categorical", "title": "Target distribution", "figure": figure, "note": "Class counts use the full dataset; at most 25 classes are displayed."})
    elif target_column in sampled.columns and target.get("task_type") == "regression":
        figure = px.histogram(sampled, x=target_column, nbins=40, title=f"Target distribution: {target_column}{suffix}", marginal="box")
        charts.append({"section": "Numerical", "title": "Target distribution", "figure": figure, "note": "Visualization sampled; target statistics use all rows." if was_sampled else "Full-data visualization."})
    return charts
