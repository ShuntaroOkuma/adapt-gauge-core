# adapt-gauge-core Usage Guide

Detailed guide for installing, configuring, running evaluations, and interpreting results.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations)
- [Example Selection Methods](#example-selection-methods)
- [Task Pack Format](#task-pack-format)
- [Scoring Methods](#scoring-methods)
- [Output Files](#output-files)
- [Viewing Results](#viewing-results)
- [Collapse Detection & Classification](#collapse-detection--classification)
- [Resuming Evaluations](#resuming-evaluations)
- [Supported Models](#supported-models)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.11+
- API access to at least one model provider:
  - **Google Cloud** (Vertex AI) for Gemini models
  - **Anthropic** for Claude models
  - **LMStudio** for local models

### Google Cloud Setup (for Gemini models)

If you plan to use Gemini models via Vertex AI, authenticate with Google Cloud first:

```bash
gcloud auth login
gcloud auth application-default login
```

Then set your project ID in `.env`:

```
GCP_PROJECT_ID=your-project-id
```

### Install

```bash
git clone https://github.com/ShuntaroOkuma/adapt-gauge-core.git
cd adapt-gauge-core
pip install -e ".[dev]"
```

To use the Streamlit viewer:

```bash
pip install -e ".[viewer]"
```

---

## Configuration

Copy the example environment file and edit it with your API keys:

```bash
cp .env.example .env
```

### API Keys (Required)

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID (for Gemini models) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude models) |
| `LMSTUDIO_BASE_URL` | LMStudio server URL (default: `http://localhost:1234/v1`) |
| `LMSTUDIO_API_KEY` | LMStudio API key (default: `lm-studio`) |

### Evaluation Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HARNESS_NUM_TRIALS` | 3 | Number of trials per evaluation |
| `HARNESS_AGGREGATION` | mean | Aggregation method: `mean` or `median` |
| `HARNESS_SUCCESS_THRESHOLD` | 0.8 | Target score for threshold_shots metric |
| `HARNESS_PASS_AT_K` | true | Calculate pass@k reliability metrics |
| `HARNESS_K_VALUES` | 1,3 | K values for pass@k (comma-separated) |

### Isolation Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HARNESS_NEW_CLIENT_PER_TRIAL` | true | Create a new client per trial (reduces state leakage) |
| `HARNESS_TIMEOUT_SECONDS` | 120 | Model response timeout in seconds |
| `HARNESS_MAX_RETRIES` | 3 | Max retry attempts on failure |
| `HARNESS_RETRY_DELAY_SECONDS` | 1.0 | Base delay between retries (seconds) |

### LLM Judge Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_JUDGE_ENABLED` | true | Enable LLM-based scoring |
| `LLM_JUDGE_GRADER_MODEL` | gemini-2.5-flash | Model used as grader |
| `LLM_JUDGE_TIMEOUT_SECONDS` | 30 | Grader response timeout |
| `LLM_JUDGE_MAX_RETRIES` | 2 | Grader retry attempts |
| `LLM_JUDGE_FALLBACK_METHOD` | f1 | Fallback scoring if grader fails |

---

## Running Evaluations

### CLI Options

```bash
python -m adapt_gauge_core.runner [OPTIONS]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--task-pack PATH` | Yes | - | Path to the task pack JSON file |
| `--models LIST` | No | DEFAULT_MODELS | Comma-separated model names |
| `--num-trials N` | No | from .env | Number of trials |
| `--run-id ID` | No | auto-generated | Run ID (also used for resuming) |
| `--output-dir DIR` | No | `results` | Output directory for CSV files |
| `--example-selection METHOD` | No | `fixed` | Example selection: `fixed` or `tfidf` |
| `--compare-selection` | No | false | Run both fixed and tfidf for comparison |

### Examples

```bash
# Basic run with default models
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json

# Specify models
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash,claude-haiku-4-5-20251001

# Use TF-IDF example selection
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --example-selection tfidf

# Compare fixed vs TF-IDF selection methods
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --compare-selection

# Custom output directory and trials
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --num-trials 5 \
  --output-dir results/experiment1
```

### Evaluation Flow

1. **Health Check** - Tests each model with a simple prompt
2. **Grader Health Check** - Tests the LLM judge grader (if llm_judge tasks exist)
3. **Run Evaluations** - Iterates over: selection method > trial > model > task > shot count > test case
4. **Aggregate Results** - Computes per-model, per-task summary metrics
5. **Collapse Detection** - Identifies negative learning, peak regression, and mid-curve dips
6. **Pattern Classification** - Classifies each model-task pair into a collapse pattern type
7. **Save Results** - Writes raw results and summary CSVs

---

## Example Selection Methods

### Fixed (default)

Uses examples in the order defined in the task pack JSON. The number of exemplars and distractors follows the shot configuration:

| Shot Count | Exemplars | Distractors |
|------------|-----------|-------------|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 1 | 1 |
| 4 | 2 | 2 |
| 8 | 6 | 2 |

### TF-IDF

Dynamically selects examples most similar to the test input using TF-IDF cosine similarity. Uses character n-grams (`char_wb`, 2-4 grams) for robust matching across languages including Japanese.

```bash
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --example-selection tfidf
```

### Compare Mode

Runs both `fixed` and `tfidf` methods sequentially for the same configuration, recording `example_selection` in the raw results CSV. This allows you to investigate whether the selection method affects negative learning occurrence.

```bash
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --compare-selection
```

---

## Task Pack Format

A task pack is a JSON file containing multiple evaluation tasks.

### Structure

```json
{
  "pack_id": "my_pack",
  "pack_name": "My Evaluation Pack",
  "description": "Description of the pack",
  "version": "1.0",
  "categories": ["classification", "summarization"],
  "tasks": [...]
}
```

### Task Definition

```json
{
  "task_id": "classification_001",
  "category": "classification",
  "version": "1.0",
  "difficulty": "medium",
  "description": "Short description of the task",
  "instruction": "Detailed instructions for the model",
  "measures": ["Acquisition", "Fidelity"],
  "examples": [
    {"input": "Example input", "output": "Expected output"}
  ],
  "test_cases": [
    {
      "input": "Test input",
      "expected_output": "Expected output",
      "scoring_method": "exact_match",
      "acceptable_variations": ["variation1", "variation2"]
    }
  ],
  "distractors": [
    {"input": "Misleading input", "output": "Misleading output"}
  ]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `task_id` | Yes | Unique identifier |
| `category` | Yes | Task category |
| `difficulty` | Yes | `low`, `medium`, or `hard` |
| `description` | Yes | Short task description |
| `instruction` | No | Detailed model instructions (used in prompt if present) |
| `examples` | Yes | Few-shot examples (minimum: 6 recommended for 8-shot) |
| `test_cases` | Yes | Evaluation cases with expected outputs |
| `distractors` | No | Noise examples to test robustness |
| `measures` | No | Evaluation axes: `Acquisition`, `Resilience-Noise`, `Resilience-Detect`, `Efficiency`, `Agency`, `Fidelity` |
| `scoring_method` | Yes (per test_case) | `exact_match`, `contains`, `f1`, or `llm_judge` |
| `acceptable_variations` | No (per test_case) | Alternative correct answers (used by llm_judge) |

---

## Scoring Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `exact_match` | Normalized string equality (case-insensitive, markdown stripped, Unicode NFKC) | Classification, labeling tasks |
| `contains` | Expected output found within actual output (normalized) | Code fix, keyword extraction |
| `f1` | Token-level F1 score with Japanese text support | Summarization, free-form text |
| `llm_judge` | LLM-based evaluation using a grader model | Complex reasoning, route optimization |

### LLM Judge Details

The LLM judge supports three `expected_output` formats:

1. **Keyword list** (string): Checks if key terms appear in output. Score: 0.0 or 1.0.
2. **Natural text** (string): Rubric-based evaluation on a 5-level scale (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
3. **Dict rubric** (dict): Custom criteria with scores (0.0, 0.5, 1.0).

---

## Output Files

### Raw Results CSV (`raw_results_{run_id}.csv`)

One row per individual evaluation.

| Column | Description |
|--------|-------------|
| `run_id` | Run identifier |
| `task_id` | Task identifier |
| `category` | Task category |
| `model_name` | Model used |
| `shot_count` | Number of shots (0, 1, 2, 4, 8) |
| `input` | Test case input |
| `expected_output` | Expected output |
| `actual_output` | Model's actual response |
| `score` | Evaluation score (0.0-1.0) |
| `scoring_method` | Scoring method used |
| `latency_ms` | Response latency in milliseconds |
| `timestamp` | ISO timestamp |
| `trial_id` | Trial number |
| `input_tokens` | Input token count |
| `output_tokens` | Output token count |
| `example_selection` | Selection method used (`fixed` or `tfidf`) |

### Summary CSV (`summary_{run_id}.csv`)

One row per model-task combination, aggregated across trials.

| Column | Description |
|--------|-------------|
| `task_id` | Task identifier |
| `category` | Task category |
| `model_name` | Model name |
| `score_0shot` - `score_8shot` | Mean/median score at each shot count |
| `improvement_rate` | (score_8shot - score_0shot) / 8 |
| `threshold_shots` | Minimum shots to reach success threshold (0.8) |
| `learning_curve_auc` | Area under the learning curve |
| `num_trials` | Number of trials |
| `score_variance` | Score variance across trials |
| `collapse_pattern` | Pattern classification (stable, immediate_collapse, gradual_decline, peak_regression) |
| `resilience_score` | Model resilience score (0.0-1.0) |
| `pass_@1`, `pass_@3` | pass@k reliability metrics (optional) |

---

## Viewing Results

### Streamlit Viewer

```bash
# View demo results (included)
streamlit run src/adapt_gauge_core/viewer.py -- --results-dir results/demo

# View your own results
streamlit run src/adapt_gauge_core/viewer.py -- --results-dir results
```

### Viewer Sections

1. **Learning Curves** - Interactive Plotly charts showing score progression across shot counts. Negative learning intervals are highlighted in red.
2. **Collapse Detection** - Warnings for three types of performance degradation:
   - **Negative Learning**: Final score drops below 0-shot baseline
   - **Peak Regression**: Score peaks at intermediate shot then regresses
   - **Mid-curve Dip**: Sharp drop between adjacent shot counts
3. **Collapse Pattern Classification** - Table showing each model-task pair classified as stable, immediate_collapse, gradual_decline, or peak_regression.
4. **Collapse Resilience Score** - Per-model score from 0.0 (always collapses) to 1.0 (fully stable).
5. **Metrics Summary** - Detailed table with all computed metrics.

---

## Collapse Detection & Classification

### Detection Types

| Type | Condition | Severity |
|------|-----------|----------|
| **Negative Learning** | Final score < 90% of 0-shot score | degradation (10-50% drop) / collapse (50%+ drop) |
| **Peak Regression** | Peak > 110% of 0-shot AND final < 80% of peak | - |
| **Mid-curve Dip** | Adjacent shot score drops > 30% | - |

### Collapse Pattern Classification

Each model-task pair is classified into one of four patterns:

| Pattern | Description |
|---------|-------------|
| `stable` | No significant degradation (monotonic increase, flat, or <10% overall drop) |
| `immediate_collapse` | Sharp drop right after 0-shot that persists (first drop accounts for 60%+ of total) |
| `gradual_decline` | Steady decrease across shot counts (10%+ overall drop, spread evenly) |
| `peak_regression` | Score improves at intermediate shots then regresses significantly |

### Resilience Score

Per-model score combining:
- **Pattern type penalty**: stable=0.0, gradual_decline=0.5, peak_regression=0.6, immediate_collapse=1.0
- **Drop magnitude**: Scaled by actual performance drop ratio

Score = 1.0 - (pattern_penalty * drop_ratio), averaged across all tasks for the model.

---

## Resuming Evaluations

If an evaluation is interrupted, you can resume from where it left off:

```bash
# First run (generates run_id like 20260205_143000)
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json

# Resume using the same run_id
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --run-id 20260205_143000
```

The runner loads existing `raw_results_{run_id}.csv` and skips completed evaluations. This works correctly with `--compare-selection` as the selection method is included in the skip key.

---

## Supported Models

### Default Models

```
gemini-3-flash-preview
gemini-2.5-flash
claude-haiku-4-5-20251001
```

### All Supported Models

| Provider | Models | Required Config |
|----------|--------|-----------------|
| **Vertex AI** | gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview | `GCP_PROJECT_ID` |
| **Anthropic** | claude-haiku-4-5-20251001, claude-sonnet-4-5-20250929, claude-opus-4-5-20251101 | `ANTHROPIC_API_KEY` |
| **LMStudio** | Any local model (prefix with `lmstudio/`) | `LMSTUDIO_BASE_URL` |

### Specifying Models

```bash
# Single model
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash

# Multiple models
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash,claude-haiku-4-5-20251001,gemini-3-flash-preview

# Local model via LMStudio
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json \
  --models lmstudio/llama-3.1-8b
```

---

## Troubleshooting

### Common Issues

**Model health check fails**
- Verify API keys in `.env`
- Check network connectivity
- For Vertex AI: ensure `gcloud auth application-default login` is configured

**LLM Judge scoring fails**
- The runner automatically falls back to f1 scoring
- Check `LLM_JUDGE_GRADER_MODEL` is accessible
- Increase `LLM_JUDGE_TIMEOUT_SECONDS` if timeouts occur

**Streamlit viewer not loading**
- Ensure `pip install -e ".[viewer]"` was run
- Check that result CSV files exist in the specified `--results-dir`

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_example_selector.py -v

# Run with coverage
python -m pytest tests/ --cov=adapt_gauge_core
```
