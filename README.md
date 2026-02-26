# adapt-gauge-core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Tests](https://github.com/ShuntaroOkuma/adapt-gauge-core/actions/workflows/test.yml/badge.svg)](https://github.com/ShuntaroOkuma/adapt-gauge-core/actions/workflows/test.yml)

[日本語](README_ja.md)

**Measure how fast LLMs learn from few-shot examples — and detect when they break.**

adapt-gauge-core is an open-source evaluation harness that measures **Adaptation Efficiency** — how quickly a language model improves with few-shot examples (0, 1, 2, 4, 8 shots) and whether it suffers from **negative learning** (performance degradation with more examples).

## Why Adaptation Efficiency?

Standard LLM benchmarks measure accuracy at a single point. But in production, teams often use few-shot prompting to adapt models to specific tasks. Two critical questions arise:

1. **How many examples does this model need?** Some models reach peak performance at 2 shots; others need 8.
2. **Does adding examples ever hurt?** For some model-task combinations, performance *drops* with more examples — a phenomenon we call **negative learning** or **collapse**.

adapt-gauge-core answers both questions automatically.

In our evaluations, we observed that **leaderboard rankings reverse** depending on shot count — a model that trails at 0-shot can overtake the leader at 4-shot. We also found models whose scores **collapse to near-zero** when given more examples. These are not edge cases; they are systematic patterns that standard benchmarks miss.

### See It in Action

**Few-shot learning curves across 4 tasks and 5 models:**

![Learning Curves Overview](docs/images/learning-curves-overview.png)

**Collapse detection** — gemini-3-flash-preview peaks at 4-shot then drops back to 0-shot level:

![Collapse Detection](docs/images/learning-curve-collapse.png)

## Quick Start

### Prerequisites

- Python 3.11+
- API access to at least one model provider:
  - **Google Cloud** (Vertex AI) for Gemini models
  - **Anthropic** for Claude models
  - **LMStudio** for local models

### Installation

```bash
git clone https://github.com/ShuntaroOkuma/adapt-gauge-core.git
cd adapt-gauge-core
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run Evaluation

```bash
# Run with default models (Gemini 3 Flash, Claude Haiku 4.5)
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json

# Specify models
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash,claude-haiku-4-5-20251001

# Use TF-IDF example selection (default) or fixed ordering
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --example-selection tfidf

# Compare both selection methods side by side
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --compare-selection

# Resume a previous run
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --run-id 20260101_120000
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--task-pack` | Path to the task pack JSON file (required) |
| `--models` | Comma-separated model names (default: built-in list) |
| `--num-trials` | Number of trials per evaluation |
| `--run-id` | Resume a previous run by its ID |
| `--output-dir` | Directory for output CSV files (default: `results`) |
| `--example-selection` | `tfidf` (default) or `fixed` ordering |
| `--compare-selection` | Run both selection methods for comparison |

### View Results

A demo evaluation result is included so you can explore the viewer without running an evaluation:

```bash
# Install viewer extras
pip install -e ".[viewer]"

# View demo results (included in results/demo/)
streamlit run src/adapt_gauge_core/viewer.py -- --results-dir results/demo

# View your own results after running an evaluation
streamlit run src/adapt_gauge_core/viewer.py
```

## What It Measures

For each model-task combination across shot counts (0, 1, 2, 4, 8):

| Metric | Description |
|--------|-------------|
| **Improvement Rate** | Score gain per additional shot |
| **Threshold Shots** | Minimum shots to reach target score (default: 0.8) |
| **Learning Curve AUC** | Area under the learning curve (higher = learns faster) |
| **Collapse Detection** | Three independent checks (see below) |
| **Collapse Pattern** | Classification: stable / immediate_collapse / gradual_decline / peak_regression |
| **Resilience Score** | Per-model collapse resilience on a 0–1 scale |
| **pass@k** | Reliability metric across multiple trials |
| **Token Usage** | Input/output tokens and latency per evaluation |

### Collapse Detection

The evaluation pipeline runs three independent collapse checks:

| Check | Triggers when |
|-------|---------------|
| **Negative Learning** | Final-shot score drops >10% below 0-shot |
| **Peak Regression** | Peak score drops >20% by final shot |
| **Mid-curve Dip** | >30% drop between consecutive shots |

Results are classified into a **collapse pattern** (stable, immediate_collapse, gradual_decline, peak_regression) and aggregated into a per-model **resilience score** (0.0–1.0).

## Demo Task Pack

The included `task_pack_core_demo.json` contains 4 tasks covering different scoring methods:

| Task | Scoring | Domain |
|------|---------|--------|
| Classification | exact_match | Email categorization |
| Code Fix | contains | Bug fixing |
| Summarization | f1 | Text summarization |
| Delivery Route | llm_judge | Route optimization |

## Project Structure

```
adapt-gauge-core/
├── src/adapt_gauge_core/
│   ├── runner.py              # CLI entry point
│   ├── viewer.py              # Streamlit results viewer
│   ├── prompt_builder.py      # Few-shot prompt construction
│   ├── example_selector.py    # TF-IDF / fixed example selection
│   ├── task_loader.py         # Task/pack JSON loading
│   ├── efficiency_calc.py     # AUC, improvement rate, threshold
│   ├── harness_config.py      # Configuration management
│   ├── domain/                # Entities, value objects, constants
│   ├── scoring/               # Scoring: exact_match, contains, f1, llm_judge
│   ├── infrastructure/        # Model clients: Vertex AI, Claude, LMStudio
│   └── use_cases/             # Evaluation, AEI/collapse analysis, health checks
├── tasks/                     # Task definitions and demo pack
├── results/                   # Evaluation output (CSV)
└── tests/                     # Test suite (264 tests)
```

## Scoring Methods

| Method | Description |
|--------|-------------|
| `exact_match` | Normalized string equality |
| `contains` | Expected output found within actual output |
| `f1` | Token-level F1 score (supports Japanese tokenization) |
| `llm_judge` | LLM-based evaluation using a grader model |

## Configuration

All settings can be configured via environment variables or `.env` file:

```bash
# Trials
HARNESS_NUM_TRIALS=3           # Number of trials per evaluation
HARNESS_AGGREGATION=mean       # mean or median

# LLM Judge
LLM_JUDGE_ENABLED=true
LLM_JUDGE_GRADER_MODEL=gemini-2.5-flash

# Reliability
HARNESS_PASS_AT_K=true
HARNESS_K_VALUES=1,3
```

See [.env.example](.env.example) for the full list.

For detailed instructions on installation, configuration, example selection methods, and interpreting results, see the [Usage Guide](docs/usage-guide.md).

## Development

```bash
make install         # Install package in dev mode
make test            # Run tests with current Python
make test-all        # Run tests with Python 3.11, 3.12, 3.13
make run             # Run evaluation with demo task pack
make help            # Show all available commands
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT
