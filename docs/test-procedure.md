# Test Procedure Manual

## Overview

This document describes how to verify **adapt-gauge-core** works correctly and that the extraction has caused **zero regression** in the original **adapt-gauge** repository.

| Repository | Location | Expected Tests | Expected Result |
|---|---|---|---|
| adapt-gauge-core | `/Users/s-ohkuma/code/pw/products/adapt-gauge-core` | 206 | **206 passed** |
| adapt-gauge (original) | `/Users/s-ohkuma/code/pw/products/adapt-gauge` | 567 | **564 passed, 3 failed** (pre-existing) |

---

## Prerequisites

- Python >= 3.11
- Both repositories cloned locally
- Virtual environments set up for each repository

---

## Step 1: Test adapt-gauge-core

### 1-1. Install dependencies

```bash
cd /Users/s-ohkuma/code/pw/products/adapt-gauge-core
pip install -e ".[dev]"
```

### 1-2. Run all tests

```bash
make test
# or equivalently:
python -m pytest tests/ -v
```

### 1-3. Expected output

```
======================= 206 passed, 4 warnings in ~1s ========================
```

**Warnings (safe to ignore):**
- `PytestCollectionWarning: cannot collect test class 'TestCase'` — This is the `@dataclass TestCase` in `task_loader.py`, not a real test class.

### 1-4. Test breakdown by module

| Module | Test File | Count | What it covers |
|---|---|---|---|
| Domain | `tests/domain/test_entities.py` | ~20 | EvaluationResult, AcquisitionMetrics, HealthCheckResult |
| Domain | `tests/domain/test_constants.py` | ~5 | SHOT_SCHEDULE, DEFAULT_MODELS, MODEL_PRICING |
| Scoring | `tests/scoring/test_text_scorers.py` | ~30 | exact_match, contains, f1 |
| Scoring | `tests/scoring/test_llm_judge.py` | ~15 | LLMJudgeScorer with mocked LLM |
| Scoring | `tests/scoring/test_scorer_core.py` | ~10 | score() dispatch, custom:* ValueError |
| Infrastructure | `tests/infrastructure/test_model_clients.py` | ~15 | Claude, VertexAI, LMStudio clients |
| Task Loader | `tests/test_task_loader.py` | ~25 | load_task, load_task_pack, load_all_tasks |
| Prompt Builder | `tests/test_prompt_builder.py` | ~20 | Prompt construction with examples |
| Efficiency | `tests/test_efficiency_calc.py` | ~25 | Learning curves, AUC, improvement rate |
| Config | `tests/test_harness_config.py` | ~15 | HarnessConfig loading and defaults |
| AEI | `tests/use_cases/test_aei.py` | ~15 | AEI computation, negative learning detection |
| Runner | `tests/test_runner_integration.py` | ~10 | End-to-end pipeline with mocks |
| Resume/Trials | `tests/test_resume_and_trials.py` | ~14 | trial_id, multi-trial aggregation, resume, intermediate save |

### 1-5. Run specific test subsets (for debugging)

```bash
# Scoring only
python -m pytest tests/scoring/ -v

# Domain only
python -m pytest tests/domain/ -v

# Single file
python -m pytest tests/test_runner_integration.py -v

# Single test
python -m pytest tests/scoring/test_scorer_core.py::TestScorerCore::test_custom_scoring_raises_error -v
```

### 1-6. Verify key behaviors unique to core

These tests verify the adaptations made during extraction:

```bash
# custom:* raises ValueError (not available in core)
python -m pytest tests/scoring/test_scorer_core.py::TestScorerCore::test_custom_scoring_raises_error -v

# llm_judge works via scorer dispatch
python -m pytest tests/scoring/test_scorer_core.py::TestScorerCore::test_llm_judge_scoring -v

# Aggregation output has NO axis_* columns
python -m pytest tests/test_runner_integration.py::TestRunnerAggregation -v

# Negative learning detection works
python -m pytest tests/test_runner_integration.py::TestNegativeLearningDetection -v
```

---

## Step 2: Verify zero regression in adapt-gauge (original)

### 2-1. Run all tests

```bash
cd /Users/s-ohkuma/code/pw/products/adapt-gauge
python -m pytest tests/ -v
```

### 2-2. Expected output

```
================== 3 failed, 564 passed, 5 warnings in ~1s ===================
```

### 2-3. Pre-existing failures (NOT caused by extraction)

These 3 failures existed **before** the core extraction and are unrelated:

| # | Test | Reason |
|---|------|--------|
| 1 | `tests/dashboard/services/test_analysis_service.py::TestBuildEfficiencyData::test_has_required_columns` | Dashboard analysis service test — unrelated to core pipeline |
| 2 | `tests/test_efficiency_calc.py::TestLearningCurveAuc::test_missing_shot_raises_error` | Pre-existing assertion mismatch |
| 3 | `tests/test_task_loader.py::TestTaskNewFields::test_task_invalid_measure` | Pre-existing assertion mismatch |

### 2-4. Regression check: confirm no NEW failures

Compare the failure list against the baseline above. If you see **any test NOT in the list above** failing, that is a regression caused by the extraction.

```bash
# Quick check: extract only FAILED lines
cd /Users/s-ohkuma/code/pw/products/adapt-gauge
python -m pytest tests/ --tb=line 2>&1 | grep "^FAILED"
```

Expected output (exactly these 3 lines):
```
FAILED tests/dashboard/services/test_analysis_service.py::TestBuildEfficiencyData::test_has_required_columns
FAILED tests/test_efficiency_calc.py::TestLearningCurveAuc::test_missing_shot_raises_error
FAILED tests/test_task_loader.py::TestTaskNewFields::test_task_invalid_measure
```

### 2-5. Verify original source files are untouched

The extraction used "Copy & Adapt" — no source files in adapt-gauge should be modified.

```bash
cd /Users/s-ohkuma/code/pw/products/adapt-gauge
git diff --name-only HEAD
```

The only file that should show changes is:
```
adapt-gauge-docs/sales/action-plan.md
```

If any `src/` or `tests/` file appears in the diff, that is unintended.

---

## Step 3: Cross-repo import isolation check

Verify that adapt-gauge-core does NOT accidentally import from `src.*` (the original adapt-gauge import path).

```bash
cd /Users/s-ohkuma/code/pw/products/adapt-gauge-core
grep -r "from src\." src/ tests/ || echo "OK: No 'from src.*' imports found"
grep -r "import src\." src/ tests/ || echo "OK: No 'import src.*' imports found"
```

Expected: Both commands print the "OK" message.

---

## Step 4: CLI smoke test (requires API keys)

This step requires live API credentials and is optional for CI.

### 4-1. Set up environment

```bash
cd /Users/s-ohkuma/code/pw/products/adapt-gauge-core
cp .env.example .env
# Edit .env and fill in at least one model's API key
```

### 4-2. Run the CLI

```bash
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json
```

### 4-3. Expected behavior

1. Health check passes for configured models
2. Grader health check passes (for llm_judge tasks)
3. Evaluations run for each task x model x shot combination
4. Learning curves are displayed
5. Collapse (negative learning) detection results are shown
6. CSV output is saved to `results/`

---

## Step 5: Streamlit viewer smoke test

After running an evaluation (Step 4), verify the Streamlit viewer renders results correctly.

### 5-1. Install viewer dependencies

```bash
cd /Users/s-ohkuma/code/pw/products/adapt-gauge-core
pip install -e ".[viewer]"
```

### 5-2. Launch the viewer

```bash
streamlit run src/adapt_gauge_core/viewer.py
```

To specify a custom results directory:

```bash
streamlit run src/adapt_gauge_core/viewer.py -- --results-dir /path/to/results
```

### 5-3. Expected behavior

1. Browser opens at `http://localhost:8501`
2. Sidebar shows a run selector with available run IDs
3. Sidebar shows task/model/trial counts
4. **Learning Curves**: Line chart per task with plotly, negative learning intervals highlighted in red, 80% target line
5. **Collapse Detection**: Green "No negative learning detected" or orange warnings per model/task
6. **Metrics Summary**: Table with scores per shot, improvement_rate, threshold_shots, learning_curve_auc

---

## Quick Reference: One-liner verification

Run both repos in sequence and check results:

```bash
# Core: expect 206 passed
cd /Users/s-ohkuma/code/pw/products/adapt-gauge-core && python -m pytest tests/ --tb=line 2>&1 | tail -3

# Original: expect 564 passed, 3 failed (pre-existing)
cd /Users/s-ohkuma/code/pw/products/adapt-gauge && python -m pytest tests/ --tb=line 2>&1 | tail -3

# Import isolation: expect no output
cd /Users/s-ohkuma/code/pw/products/adapt-gauge-core && grep -r "from src\." src/ tests/
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: adapt_gauge_core` | Package not installed | Run `pip install -e ".[dev]"` in adapt-gauge-core |
| New FAILED test in adapt-gauge | Possible regression | Check `git diff` for unintended source changes |
| `from src.*` found in core | Incomplete import migration | Replace `from src.X` with `from adapt_gauge_core.X` |
| `ValueError: Custom scoring is not available` | Expected behavior in core | custom:* scorers are intentionally excluded |
| Fewer than 206 tests collected in core | Missing test file | Compare test files against the list in Step 1-4 |
| Fewer than 567 tests collected in original | Test file accidentally deleted | Check `git status` in adapt-gauge |
