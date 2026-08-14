You are a senior Python AI engineer. Build a complete, production-quality but beginner-readable **Agentic AI Exploratory Data Analysis (EDA) application** using:

* Python
* LangGraph
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Plotly where useful
* OpenRouter API
* A FREE OpenRouter LLM
* python-dotenv
* Pydantic / TypedDict where appropriate

The application should allow a user to upload a CSV file and then use an **agentic LangGraph workflow** to inspect the dataset, decide which EDA operations should be performed, execute those operations using deterministic Python tools, interpret the results, identify important patterns/problems, and produce a final EDA report.

The LLM should NOT directly calculate statistics or inspect the complete raw dataset itself. Python/Pandas tools must calculate statistics, correlations, missing values, outliers, distributions, etc. The LLM should act primarily as the **planner, reasoning/orchestration layer, and report generator**.

## 1. Core user flow

The Streamlit application should work like this:

1. User opens the application.
2. User enters or configures an OpenRouter API key through environment variables.
3. User uploads a `.csv` file.
4. Application validates the file.
5. Display:

   * filename
   * number of rows
   * number of columns
   * memory usage
   * first 5–10 rows
6. User clicks:

**Run Agentic EDA**

7. LangGraph starts an agentic workflow.
8. Agent first understands the dataset structure.
9. Agent creates an EDA plan.
10. Agent selects appropriate EDA tools.
11. Tools execute using Pandas/NumPy.
12. Agent evaluates the results.
13. If additional investigation is useful, the graph loops back and runs additional analysis tools.
14. When sufficient analysis has been performed, the agent generates a final structured EDA report.
15. Streamlit displays:

    * execution progress
    * analysis performed
    * statistics
    * detected issues
    * charts
    * final AI-generated observations
    * recommendations

The final application should feel like an **AI Data Analyst**.

---

# 2. Important architectural rule

Do NOT create a hardcoded sequential pipeline such as:

upload → describe → missing values → correlation → charts → report.

That would not be sufficiently agentic.

Instead implement:

Dataset
↓
Dataset Profiler
↓
EDA Planner Agent
↓
Tool Selection / Analysis Agent
↓
Execute Tool
↓
Evaluate Result
↓
Need more analysis?
↙           ↘
YES          NO
↓             ↓
Select Tool   Final Report
↓
Execute
↓
Evaluate
↓
...

Use **LangGraph StateGraph** with conditional edges.

The agent must dynamically determine which analyses are useful depending on the uploaded dataset.

For example:

A numerical dataset may trigger:

* descriptive statistics
* correlation
* distributions
* outlier analysis

A categorical-heavy dataset may trigger:

* categorical frequency analysis
* cardinality analysis
* imbalance analysis

A time-series-like dataset may trigger:

* date detection
* chronological trends
* missing periods

A dataset containing possible ID fields should avoid treating IDs as meaningful numerical variables.

---

# 3. LangGraph architecture

Create an explicit graph.

Use something conceptually similar to:

START
↓
profile_dataset
↓
plan_analysis
↓
select_analysis
↓
execute_analysis
↓
evaluate_analysis
↓
should_continue?
├── continue → select_analysis
└── finish → generate_report
↓
END

Do not simply hide everything inside one giant agent node.

Use meaningful LangGraph nodes.

Suggested nodes:

### Node 1 — `profile_dataset`

Automatically inspect basic metadata without LLM involvement.

Gather:

* shape
* columns
* inferred dtypes
* missing counts
* missing percentages
* unique counts
* numerical columns
* categorical columns
* boolean columns
* datetime-like columns
* possible identifier columns
* constant columns
* duplicated row count
* memory usage

Store this in graph state.

---

### Node 2 — `plan_analysis`

Send a compact dataset profile to the OpenRouter LLM.

Ask it to construct an EDA plan.

The planner should answer using structured output.

Example concept:

{
"dataset_summary": "...",
"analysis_plan": [
{
"tool": "missing_value_analysis",
"reason": "..."
},
{
"tool": "numerical_summary",
"reason": "..."
}
]
}

Do NOT send the full DataFrame to the LLM.

Only send:

* schema
* column names
* column types
* compact profile
* optionally a few sanitized sample rows

---

# 4. EDA tool system

Create deterministic Python tools.

Each tool should have:

* clear name
* docstring
* validated parameters
* safe error handling
* structured return result

Create at minimum the following tools.

## `dataset_overview`

Return:

* rows
* columns
* column names
* dtypes
* memory usage
* duplicates

## `missing_value_analysis`

Return for every column:

* missing count
* missing percentage

Highlight columns with significant missing values.

## `numerical_summary`

For numerical columns calculate:

* count
* mean
* median
* std
* min
* max
* 25%
* 50%
* 75%
* skewness
* optionally kurtosis

## `categorical_summary`

For categorical columns calculate:

* unique count
* most common value
* frequency
* top categories
* percentage distribution

Prevent huge output for high-cardinality columns.

## `correlation_analysis`

Calculate numerical correlation matrix.

Identify:

* strongest positive correlations
* strongest negative correlations
* highly correlated pairs

Ignore self correlations.

Allow selecting correlation method:

* Pearson
* Spearman

## `outlier_analysis`

Detect potential outliers.

Use IQR by default.

For each numerical column calculate:

Q1
Q3
IQR
lower bound
upper bound
outlier count
outlier percentage

Do not automatically claim that every detected outlier is bad data.

## `duplicate_analysis`

Return:

* duplicate row count
* duplicate percentage

## `cardinality_analysis`

For categorical columns calculate:

unique_count / total_rows

Identify:

* low-cardinality categorical features
* very high-cardinality features
* likely identifier columns

## `distribution_analysis`

Calculate useful distribution information for numerical columns:

* skewness
* quantiles
* zero percentage
* negative percentage where relevant

Create histogram information / chart metadata.

## `categorical_imbalance_analysis`

Detect categories where one class dominates heavily.

Especially useful for potential target-like columns.

## `datetime_analysis`

If datetime columns exist:

* minimum date
* maximum date
* date range
* number of unique dates
* possible gaps
* counts by month/year where sensible

## `target_analysis`

Only use when a likely target is specified or the user selects a target column.

If classification:

* class distribution
* imbalance

If regression:

* target distribution
* target correlations

Do NOT automatically assume the final column is the target.

---

# 5. Agent tool selection

The LLM should decide which tool should be executed next.

Do NOT allow arbitrary Python code generated by the LLM to execute.

The agent must only select from registered safe tools.

Implement tool calling if supported reliably by the selected OpenRouter model.

If native tool calling is unreliable for the free model, implement structured JSON tool selection with Pydantic validation.

Example:

{
"tool_name": "correlation_analysis",
"arguments": {
"method": "pearson"
},
"reason": "Dataset contains 8 continuous numerical variables."
}

Validate tool names against an allowlist.

Never execute strings using:

```python
eval()
exec()
```

---

# 6. Graph state

Create a clean typed LangGraph state.

Use TypedDict or Pydantic.

Conceptually include:

```python
class EDAState(TypedDict):
    dataset_profile: dict
    analysis_plan: list
    completed_analyses: list
    current_tool: str | None
    current_tool_args: dict
    tool_results: list
    observations: list
    iteration_count: int
    max_iterations: int
    next_action: str
    final_report: str | None
    errors: list
```

Do not store the complete pandas DataFrame inside prompts sent to the LLM.

You may maintain the DataFrame in application/session context or appropriate graph runtime state, but ensure it is never serialized unnecessarily into an LLM prompt.

---

# 7. Agent loop

Make the application genuinely iterative.

After every tool execution, call an evaluation node.

Provide the agent with:

* dataset profile
* original plan
* analyses already completed
* compact latest tool result

Ask:

1. What did we learn?
2. Is another analysis necessary?
3. If yes, which tool should run next?
4. If no, finish the analysis.

Return structured data.

Example:

```json
{
  "observation": "Age has a strongly right-skewed distribution.",
  "continue_analysis": true,
  "next_tool": "outlier_analysis",
  "reason": "The skew suggests extreme values may exist."
}
```

Use a LangGraph conditional edge to decide:

`continue_analysis == true`
→ select/execute next tool

otherwise:

→ generate final report

---

# 8. Infinite-loop protection

Very important.

Implement:

```python
MAX_AGENT_ITERATIONS = 10
```

or similar.

If iteration count reaches the maximum:

automatically route to final report generation.

Also prevent the same tool with identical parameters from being repeatedly executed unless there is a justified reason.

Track:

```python
completed_tool_calls
```

---

# 9. OpenRouter integration

Use OpenRouter's OpenAI-compatible API.

Use environment variables:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=some-free-model
```

Do not hardcode the API key.

Create:

```python
services/llm.py
```

responsible for LLM initialization.

Use the appropriate current OpenRouter endpoint/API configuration.

Prefer a free model.

However:

DO NOT blindly hardcode a model name that may disappear.

Implement a configurable model:

```python
OPENROUTER_MODEL
```

and provide a reasonable currently available free-model example in `.env.example`.

Document clearly that users can replace it with another OpenRouter model ending in `:free`.

Handle:

* rate limit errors
* unavailable models
* timeout
* malformed JSON
* API failures

Use retries with a small retry count.

Give helpful errors to the Streamlit interface.

---

# 10. Prompt engineering

Create prompts in a separate file:

```text
prompts/
    planner.py
    evaluator.py
    reporter.py
```

## Planner system prompt

The planner should behave as:

"You are a senior data analyst. Your job is to determine the most useful EDA operations for a dataset based only on its schema and summarized metadata.

Do not invent statistics.
Do not calculate values yourself.
Use available deterministic Python analysis tools.
Prioritize analyses that reveal data quality problems, distributions, relationships, anomalies, and modeling concerns.
Avoid meaningless analysis of identifier columns."

Require structured output.

---

## Evaluator prompt

The evaluator should behave as:

"You are evaluating results from one EDA operation.

Determine the meaningful observations supported by the result.

Never claim causation from correlation.

Never invent information.

Decide whether further analysis is useful.

Avoid repeatedly requesting analyses already completed."

Require structured output.

---

## Reporter prompt

Generate a structured report containing:

# Dataset Overview

# Data Quality

# Missing Values

# Numerical Features

# Categorical Features

# Relationships and Correlations

# Outliers and Anomalies

# Important Patterns

# Potential Modeling Concerns

# Recommended Data Cleaning Steps

# Suggested Next Analysis

Only include sections supported by analyses actually performed.

Clearly distinguish between:

* measured facts
* possible interpretations
* recommendations

Never invent values.

---

# 11. Streamlit interface

Create a polished Streamlit application.

Main page title:

**Agentic AI Data Analyst**

Subtitle:

"Upload a CSV and let an AI agent autonomously explore your dataset."

---

## Sidebar

Include:

OpenRouter configuration status.

Model display.

Settings:

* Max agent iterations
* Correlation method
* Number of top categories
* Outlier detection method

Optional:

Target column dropdown:

```text
None
[column names...]
```

Do not force user to choose target.

---

# 12. Upload area

Use:

```python
st.file_uploader(
    "Upload CSV",
    type=["csv"]
)
```

Validate:

* file extension
* empty dataset
* corrupted CSV
* unsupported encoding where possible
* extremely large datasets
* duplicate column names

Use Pandas robustly.

Try sensible encoding fallbacks when necessary.

Do not crash the app because of malformed CSV.

---

# 13. Dataset preview

After upload display metrics such as:

```text
Rows       Columns       Missing Values       Duplicates
10,000     15            237                  14
```

Display:

```python
st.dataframe(df.head(10))
```

Show inferred column types.

---

# 14. Agent execution UI

When user clicks:

**Run Agentic EDA**

show progress.

Example:

```text
✓ Dataset profiled

✓ Agent created analysis plan

Running: Missing Value Analysis
Reason: Several columns contain null values

✓ Missing value analysis complete

Running: Correlation Analysis
Reason: Dataset contains multiple numerical variables

✓ Correlation analysis complete

Running: Outlier Analysis
Reason: Income appears highly skewed

Generating final report...
```

Use Streamlit status/progress components where appropriate.

The user should be able to understand what the agent is doing.

---

# 15. Display agent reasoning safely

Do NOT expose hidden chain-of-thought.

Instead display concise agent decisions such as:

```text
Analysis Selected:
Correlation Analysis

Reason:
Multiple continuous numerical features are available, so examining relationships may reveal redundant or strongly related variables.
```

Store only concise explanations.

Call them:

* Decision
* Reason
* Observation

not "Chain of Thought".

---

# 16. Visualization system

Generate charts using deterministic Python code.

Do NOT have the LLM write plotting code dynamically.

Support useful plots including:

### Missing values

Bar chart:

column vs missing percentage

### Numerical distributions

Histogram

### Box plot

### Correlation

Heatmap

### Categorical

Bar chart of top categories

### Target distribution

Classification:
bar chart

Regression:
histogram

Charts should only be generated where useful.

Avoid generating 50 charts for datasets with many columns.

Limit chart count intelligently.

For example:

```python
MAX_CHART_COLUMNS = 10
```

Select the most informative columns.

---

# 17. EDA results interface

Use Streamlit tabs.

Example:

```text
Overview
Data Quality
Numerical
Categorical
Relationships
Outliers
AI Insights
```

Populate tabs dynamically.

---

# 18. Final AI report

Display report using:

```python
st.markdown(report)
```

The report should be concise but useful.

Every numerical statement should originate from deterministic tool results.

The LLM must not fabricate statistics.

---

# 19. Download report

Create a downloadable report.

At minimum allow:

```text
Download EDA Report (.md)
```

using Streamlit download button.

Bonus:

HTML report.

Do not add PDF unless it can be implemented cleanly without unnecessary dependencies.

---

# 20. Chat with dataset — bonus feature

After EDA completes, add:

**Ask Your Dataset**

User can type questions such as:

* Which columns contain the most missing values?
* Which numerical features are strongly correlated?
* Are there major outliers?
* Which category dominates?
* What should I clean before machine learning?
* Which columns may be useless?

Important:

The LLM should NOT receive the entire raw dataset.

Implement another LangGraph/tool-calling workflow.

Question
↓
Agent chooses relevant data-analysis tool
↓
Python tool executes
↓
LLM explains result
↓
Answer

Reuse the same safe analysis tools.

Maintain conversation messages using Streamlit session state.

---

# 21. Security requirements

Never execute arbitrary user or LLM-generated Python code.

Do NOT use:

```python
eval
exec
subprocess
os.system
```

for agent-generated commands.

The LLM may only invoke registered EDA tools.

Validate:

* tool name
* parameters
* column names

Never permit arbitrary filesystem paths.

Uploaded files should only be processed in memory where possible.

Do not send the whole CSV to OpenRouter.

---

# 22. Large dataset handling

Add safeguards.

If dataset is large, for example >100,000 rows:

Perform exact aggregate operations using Pandas where practical.

For expensive visualizations, sample:

```python
df.sample(min(len(df), 10000))
```

Clearly distinguish between:

* statistics calculated over the full dataset
* charts generated from a sample

Do not accidentally calculate important statistics from a visualization sample.

---

# 23. Code organization

Do NOT put everything inside `app.py`.

Use a clean architecture similar to:

```text
agentic-eda/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── agent/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   ├── nodes.py
│   └── routing.py
│
├── tools/
│   ├── __init__.py
│   ├── overview.py
│   ├── missing.py
│   ├── numerical.py
│   ├── categorical.py
│   ├── correlation.py
│   ├── outliers.py
│   ├── datetime_analysis.py
│   └── registry.py
│
├── services/
│   ├── __init__.py
│   ├── llm.py
│   └── dataframe_service.py
│
├── prompts/
│   ├── __init__.py
│   ├── planner.py
│   ├── evaluator.py
│   └── reporter.py
│
├── visualization/
│   ├── __init__.py
│   └── charts.py
│
└── utils/
    ├── __init__.py
    ├── csv_loader.py
    └── validators.py
```

You may modify this structure slightly if LangGraph architecture benefits from it, but keep clear separation of concerns.

---

# 24. Dependency requirements

Create a minimal `requirements.txt`.

Use only dependencies that are actually required.

Likely include:

```text
streamlit
pandas
numpy
langgraph
langchain-core
openai
python-dotenv
pydantic
plotly
matplotlib
```

If another dependency is necessary, explain why.

Avoid unnecessary framework dependencies.

---

# 25. Environment file

Create:

`.env.example`

Example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=replace-with-current-free-model
```

Do not expose secret keys.

---

# 26. README

Create a detailed README containing:

## Project description

## Architecture

Include ASCII diagram:

```text
CSV Upload
   |
   v
Dataset Profiler
   |
   v
EDA Planner Agent
   |
   v
Tool Selection
   |
   v
Python EDA Tool
   |
   v
Result Evaluator
   |
   +------ Need More Analysis? ------+
   |                                 |
  YES                               NO
   |                                 |
   +----> Tool Selection       Final Report
```

Explain why this is agentic.

## Installation

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

Create `.env`.

Run:

```bash
streamlit run app.py
```

## OpenRouter configuration

Explain how model configuration works.

Explain that free OpenRouter models can change and may have rate limits.

## How LangGraph works in this project

Explain:

* state
* nodes
* edges
* conditional edges
* agent loop

## Available EDA tools

## Security

## Limitations

---

# 27. Code quality requirements

Use:

* type hints
* docstrings
* clean functions
* meaningful variable names
* modular code
* reasonable comments
* exception handling

Avoid unnecessary abstractions.

Code should be understandable by an intermediate Python developer learning LangGraph.

Avoid deprecated LangGraph/LangChain APIs.

Before implementing LangGraph APIs, check the CURRENT official LangGraph documentation and use the latest supported Graph API patterns.

Do not copy outdated tutorials blindly.

Use:

* StateGraph
* explicit state
* explicit nodes
* conditional edges

where appropriate.

---

# 28. Structured outputs

Define Pydantic models such as:

```python
class AnalysisStep(BaseModel):
    tool_name: str
    reason: str
    arguments: dict = {}

class AnalysisPlan(BaseModel):
    dataset_summary: str
    steps: list[AnalysisStep]

class EvaluationDecision(BaseModel):
    observation: str
    continue_analysis: bool
    next_tool: str | None
    arguments: dict = {}
    reason: str
```

Adapt schemas where necessary.

Validate every LLM structured response.

If parsing fails:

1. retry once
2. use a safe fallback
3. do not crash the entire graph

---

# 29. Result compression

Some tool results may become large.

Do not send a huge correlation matrix or hundreds of categories to the LLM.

Create compact representations.

For example correlation output sent to LLM:

```json
{
    "strongest_positive": [
        ["income", "spending", 0.82]
    ],
    "strongest_negative": [
        ["age", "activity", -0.61]
    ],
    "high_correlation_pairs": 3
}
```

while the complete matrix can remain available to Streamlit for visualization.

Likewise categorical summaries should only send top categories.

Separate:

```python
display_result
```

from:

```python
llm_summary
```

where useful.

---

# 30. Data-type inference

Improve basic Pandas dtype detection.

Detect:

* numerical
* categorical
* boolean
* datetime-like
* possible IDs
* constant columns
* high-cardinality text fields

For example, a numeric column such as:

```text
customer_id
100001
100002
100003
```

should potentially be classified as identifier-like and excluded from correlation analysis unless explicitly requested.

Use heuristics, but clearly mark them as inferred.

---

# 31. Automatic insights

Create deterministic helper functions capable of identifying factual insights before asking the LLM.

Examples:

```text
"Column cabin has 77.1% missing values."

"fare has 14 potential IQR outliers."

"income and spending_score have correlation 0.81."

"customer_id contains 100% unique values and may be an identifier."

"status='active' represents 92% of rows."
```

Then let the LLM convert these facts into readable interpretation.

This reduces hallucination.

---

# 32. Important statistical behavior

Follow these rules:

Correlation does not imply causation.

Outliers are not automatically errors.

Missing values do not automatically require dropping rows.

High-cardinality columns are not automatically useless.

Skewness is not automatically a problem.

Identifier detection is heuristic.

Do not give unsupported machine-learning recommendations.

---

# 33. Target column behavior

In sidebar provide:

```text
Target column:
Auto / None
Column A
Column B
...
```

Default should effectively mean no explicitly defined target.

If a target is selected:

the planner should prioritize:

* target distribution
* feature-target relationships
* class imbalance for classification
* target correlation for regression

Do not leak the target into inappropriate analyses.

---

# 34. Streamlit state management

Use:

```python
st.session_state
```

where appropriate to preserve:

* uploaded DataFrame
* EDA results
* generated report
* tool execution history
* chat history

Be mindful that Streamlit reruns the Python script after interactions.

Do not unnecessarily rerun the entire EDA workflow every time the user interacts with another widget.

Cache safe deterministic computations where appropriate.

---

# 35. Optional execution history

Add an expandable section:

**Agent Execution Trace**

Example:

```text
1. profile_dataset
   Dataset: 8,912 rows × 14 columns

2. missing_value_analysis
   Reason: Four columns contain missing values.

3. categorical_summary
   Reason: Dataset contains five categorical features.

4. correlation_analysis
   Reason: Six meaningful numerical variables exist.

5. outlier_analysis
   Reason: Two features exhibit highly skewed distributions.

6. final_report
```

Do not expose hidden model chain-of-thought.

Only show sanitized decisions/reasons returned intentionally by structured output.

---

# 36. Error handling

Handle at minimum:

* OpenRouter API key missing
* API authentication failure
* API rate limit
* model unavailable
* invalid LLM JSON
* CSV parsing failure
* empty CSV
* dataset with zero rows
* dataset with only one column
* dataset with no numerical columns
* dataset with no categorical columns
* all-null column
* constant column
* extremely high cardinality
* NaN/Infinity serialization errors
* plotting errors

The application should gracefully continue where possible.

---

# 37. Testing

Add a small test suite where practical.

At minimum test:

* CSV profiler
* missing-value analysis
* numerical summary
* correlation extraction
* IQR outlier detection
* identifier detection
* tool registry validation

Use pytest if adding tests.

Example structure:

```text
tests/
    test_profiler.py
    test_tools.py
```

Do not spend excessive complexity on tests, but ensure the core deterministic EDA logic is verifiable independently of the LLM.

---

# 38. What I want from you

Do not merely explain how to build the project.

Actually BUILD the complete project.

Work in this sequence:

1. Inspect the current repository.
2. Determine whether any existing files should be reused.
3. Create the project structure.
4. Implement CSV loading and profiling.
5. Implement deterministic EDA tools.
6. Implement OpenRouter client.
7. Implement structured LLM responses.
8. Implement LangGraph state.
9. Implement graph nodes.
10. Implement conditional routing and agent loop.
11. Implement report generation.
12. Implement Streamlit UI.
13. Implement charts.
14. Add `.env.example`.
15. Add requirements.
16. Add tests.
17. Add README.
18. Run static/basic checks.
19. Fix import/runtime errors.
20. Provide final explanation.

Do not stop after scaffolding.

---

# 39. Verification before completion

Before saying the project is complete:

Run whatever local checks are available.

At minimum verify:

```bash
python -m compileall .
```

and, if tests exist:

```bash
pytest
```

Check imports.

Check obvious Streamlit runtime issues.

If API credentials are unavailable, mock or isolate the LLM-dependent portions and verify everything else.

Do not invent successful API results when an API key is unavailable.

---

# 40. Final response expected from you

After implementing, give me:

### Architecture

Short explanation of the LangGraph workflow.

### Files Created

Explain important files.

### Agent Flow

Show graph flow.

### Available Tools

List EDA tools.

### How to Configure OpenRouter

Explain `.env`.

### How to Run

Give exact commands.

### Testing

Tell me what tests/checks were run and their result.

### Important Design Decisions

Explain particularly:

* why calculations happen in Python instead of LLM
* why arbitrary Python execution is prohibited
* how the graph decides whether more analysis is needed
* how hallucination is minimized

Do NOT paste every source file into your final response unless necessary because the files should already have been created in the repository.

Most importantly: this must be a real **LangGraph-based agentic EDA system**, not a normal EDA script wrapped in an LLM.
