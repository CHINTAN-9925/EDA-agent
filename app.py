"""Streamlit entry point for the Agentic AI Data Analyst."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.chat_graph import build_chat_graph
from agent.graph import build_eda_graph, initial_state
from services.dataframe_service import profile_dataframe
from services.llm import LLMClient, LLMServiceError
from utils.csv_loader import CSVLoadError, load_csv
from visualization.charts import build_charts

load_dotenv()
st.set_page_config(page_title="Agentic AI Data Analyst", page_icon="📊", layout="wide")


def _initialize_session() -> None:
    defaults = {"file_hash": None, "dataframe": None, "filename": None, "profile": None, "eda_state": None, "chat_history": []}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _result_for(state: dict[str, Any], tool: str) -> dict[str, Any] | None:
    return next((item for item in state.get("tool_results", []) if item.get("tool_name") == tool and item.get("status") == "success"), None)


def _show_result_tables(state: dict[str, Any]) -> None:
    overview_tab, quality_tab, numerical_tab, categorical_tab, relationships_tab, outliers_tab, insights_tab = st.tabs(["Overview", "Data Quality", "Numerical", "Categorical", "Relationships", "Outliers", "AI Insights"])
    try:
        charts = build_charts(st.session_state.dataframe, state.get("tool_results", []))
    except Exception as exc:
        charts = []
        st.warning(f"Tables and the report are available, but charts could not be generated: {exc}")

    with overview_tab:
        st.subheader("Analysis performed")
        st.write(", ".join(state.get("completed_analyses", [])) or "None")
        result = _result_for(state, "dataset_overview")
        if result:
            st.json(result["display_result"], expanded=False)
    with quality_tab:
        missing = _result_for(state, "missing_value_analysis")
        if missing:
            st.dataframe(pd.DataFrame(missing["display_result"]["columns"]), use_container_width=True, hide_index=True)
        duplicates = _result_for(state, "duplicate_analysis")
        if duplicates:
            st.json(duplicates["display_result"])
        _render_charts(charts, "Data Quality")
    with numerical_tab:
        result = _result_for(state, "numerical_summary")
        if result:
            st.dataframe(pd.DataFrame(result["display_result"]["statistics"]).T, use_container_width=True)
        _render_charts(charts, "Numerical")
    with categorical_tab:
        result = _result_for(state, "categorical_summary")
        if result:
            for column, summary in result["display_result"]["columns"].items():
                with st.expander(column):
                    st.dataframe(pd.DataFrame(summary["top_categories"]), use_container_width=True, hide_index=True)
        _render_charts(charts, "Categorical")
    with relationships_tab:
        result = _result_for(state, "correlation_analysis")
        if result:
            st.dataframe(pd.DataFrame(result["display_result"]["highly_correlated_pairs"]), use_container_width=True, hide_index=True)
        _render_charts(charts, "Relationships")
    with outliers_tab:
        result = _result_for(state, "outlier_analysis")
        if result:
            st.caption("IQR detections are candidates for review, not automatic data errors.")
            st.dataframe(pd.DataFrame(result["display_result"]["columns"]).T, use_container_width=True)
        _render_charts(charts, "Outliers")
    with insights_tab:
        st.markdown(state.get("final_report") or "No report was generated.")
        st.download_button("Download EDA Report (.md)", data=state.get("final_report", ""), file_name="eda_report.md", mime="text/markdown")


def _render_charts(charts: list[dict[str, Any]], section: str) -> None:
    for chart in charts:
        if chart["section"] == section:
            st.plotly_chart(chart["figure"], use_container_width=True)
            st.caption(chart["note"])


def _run_eda(llm: LLMClient, settings: dict[str, Any], target: str | None) -> None:
    graph = build_eda_graph(llm)
    state = initial_state(st.session_state.dataframe, settings["max_iterations"], target, settings)
    progress = st.progress(0.0, text="Starting agentic workflow…")
    status_messages = {
        "profile_dataset": "✓ Dataset profiled",
        "plan_analysis": "✓ Agent created an analysis plan",
        "select_analysis": "Agent selected the next analysis",
        "execute_analysis": "✓ Deterministic Python analysis complete",
        "evaluate_analysis": "✓ Agent evaluated the result",
        "generate_report": "✓ Final report generated",
    }
    total_steps = max(6, settings["max_iterations"] * 3 + 3)
    event_count = 0
    with st.status("Agentic EDA in progress", expanded=True) as status:
        try:
            for event in graph.stream(state, config={"recursion_limit": total_steps + 5}, stream_mode="updates"):
                for node_name, update in event.items():
                    state.update(update)
                    event_count += 1
                    message = status_messages.get(node_name, node_name)
                    if node_name == "select_analysis" and update.get("current_tool"):
                        message = f"Running: {update['current_tool']} — {update.get('current_reason', '')}"
                    st.write(message)
                    progress.progress(min(event_count / total_steps, 0.95), text=message)
            status.update(label="Agentic EDA complete", state="complete", expanded=False)
            progress.progress(1.0, text="Analysis complete")
            st.session_state.eda_state = state
        except Exception as exc:
            status.update(label="EDA stopped", state="error")
            raise LLMServiceError(f"The workflow could not complete: {exc}") from exc


_initialize_session()
st.title("Agentic AI Data Analyst")
st.caption("Upload a CSV and let an AI agent autonomously explore your dataset.")

provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
key_variable = "GROQ_API_KEY" if provider == "groq" else "OPENROUTER_API_KEY"
model_variable = "GROQ_MODEL" if provider == "groq" else "OPENROUTER_MODEL"
default_model = "openai/gpt-oss-20b" if provider == "groq" else "openrouter/free"
api_key = os.getenv(key_variable, "")
model = os.getenv(model_variable, default_model)
with st.sidebar:
    st.header("Configuration")
    if api_key:
        st.success(f"{provider.title()} API key configured")
    else:
        st.error(f"{key_variable} is missing")
    provider_note = "Groq Free Plan compatible; account rate limits apply." if provider == "groq" else "OpenRouter provider"
    st.caption(f"Provider: {provider.title()} — {provider_note}")
    st.code(model, language=None)
    max_iterations = st.slider("Max agent iterations", 1, 15, 5, help="Five is the recommended default for Groq Free Plan usage.")
    correlation_method = st.selectbox("Correlation method", ["pearson", "spearman"])
    top_categories = st.slider("Number of top categories", 3, 25, 10)
    outlier_method = st.selectbox("Outlier detection method", ["iqr"], format_func=lambda value: "IQR (1.5×)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest != st.session_state.file_hash:
        try:
            loaded = load_csv(data, uploaded.name)
            st.session_state.update(file_hash=digest, dataframe=loaded.dataframe, filename=uploaded.name, profile=profile_dataframe(loaded.dataframe), eda_state=None, chat_history=[])
            for warning in loaded.warnings:
                st.warning(warning)
            st.caption(f"Loaded with {loaded.encoding} encoding.")
        except CSVLoadError as exc:
            st.error(str(exc))
            st.stop()

if st.session_state.dataframe is not None:
    df = st.session_state.dataframe
    profile = st.session_state.profile
    st.subheader(st.session_state.filename)
    cols = st.columns(5)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Columns", f"{len(df.columns):,}")
    cols[2].metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    cols[3].metric("Duplicates", f"{profile['duplicated_rows']:,}")
    cols[4].metric("Memory", _format_bytes(profile["memory_bytes"]))
    st.dataframe(df.head(10), use_container_width=True)
    with st.expander("Inferred column types"):
        st.json({key: profile[key] for key in ("dtypes", "numerical_columns", "categorical_columns", "boolean_columns", "datetime_columns", "possible_identifier_columns", "constant_columns")})

    target_options = ["None"] + [str(column) for column in df.columns]
    with st.sidebar:
        target_choice = st.selectbox("Target column", target_options, help="No target is assumed by default.")
    target = None if target_choice == "None" else target_choice
    settings = {"max_iterations": max_iterations, "correlation_method": correlation_method, "top_categories": top_categories, "outlier_method": outlier_method}

    if st.button("Run Agentic EDA", type="primary", disabled=not bool(api_key)):
        try:
            _run_eda(LLMClient(api_key=api_key, model=model, provider=provider), settings, target)
        except LLMServiceError as exc:
            st.error(str(exc))

    if st.session_state.eda_state:
        state = st.session_state.eda_state
        _show_result_tables(state)
        with st.expander("Agent Execution Trace"):
            for index, item in enumerate(state.get("execution_trace", []), 1):
                label = item.get("tool") or item.get("node")
                detail = item.get("reason") or item.get("observation") or item.get("detail") or item.get("status", "")
                st.markdown(f"**{index}. {label}**  \n{detail}")
        if state.get("errors"):
            with st.expander("Recoverable warnings"):
                for error in state["errors"]:
                    st.warning(error)

        st.divider()
        st.subheader("Ask Your Dataset")
        st.caption("Questions are routed through an allowlisted Python analysis tool; the CSV is never sent to the model.")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if question := st.chat_input("Ask about missing values, correlations, outliers, categories, or cleaning"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            try:
                chat = build_chat_graph(LLMClient(api_key=api_key, model=model, provider=provider))
                answer_state = chat.invoke({"dataframe": df, "question": question, "profile": profile, "target_column": target})
                answer = answer_state["answer"]
            except Exception as exc:
                answer = f"I could not answer safely: {exc}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
else:
    st.info("Upload a CSV file to begin.")
