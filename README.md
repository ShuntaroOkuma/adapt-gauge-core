# adapt-gauge-core

[日本語](README_ja.md)

**Measure how fast LLMs learn from few-shot examples — and detect when they break.**

adapt-gauge-core is an open-source evaluation harness that measures **Adaptation Efficiency** — how quickly a language model improves with few-shot examples (0, 1, 2, 4, 8 shots) and whether it suffers from **negative learning** (performance degradation with more examples).

## Why Adaptation Efficiency?

Standard LLM benchmarks measure accuracy at a single point. But in production, teams often use few-shot prompting to adapt models to specific tasks. Two critical questions arise:

1. **How many examples does this model need?** Some models reach peak performance at 2 shots; others need 8.
2. **Does adding examples ever hurt?** For some model-task combinations, performance *drops* with more examples — a phenomenon we call **negative learning** or **collapse**.

adapt-gauge-core answers both questions automatically.

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

# Resume a previous run
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --run-id 20260101_120000
```

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
| **Negative Learning** | Detects when 8-shot score drops >20% below 0-shot |
| **pass@k** | Reliability metric across multiple trials |

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
│   ├── task_loader.py         # Task/pack JSON loading
│   ├── efficiency_calc.py     # AUC, improvement rate, threshold
│   ├── harness_config.py      # Configuration management
│   ├── domain/                # Entities and value objects
│   ├── scoring/               # Scoring: exact_match, contains, f1, llm_judge
│   ├── infrastructure/        # Model clients: Vertex AI, Claude, LMStudio
│   └── use_cases/             # AEI computation, health checks
├── tasks/                     # Task definitions and demo pack
├── results/                   # Evaluation output (CSV)
└── tests/                     # Test suite (206+ tests)
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

## Running Tests

```bash
make test
# or
python -m pytest tests/ -v
```

## License

MIT
