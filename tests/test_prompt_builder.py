"""
Unit tests for prompt_builder.py
"""

import pytest

from adapt_gauge_core.example_selector import ExampleSelectionMethod
from adapt_gauge_core.prompt_builder import (
    SHOT_CONFIG,
    SelectedExample,
    build_examples_section,
    build_prompt,
    select_examples_and_distractors,
)
from adapt_gauge_core.task_loader import Distractor, Example, Task, TestCase


@pytest.fixture
def sample_task():
    """Sample task for testing"""
    return Task(
        task_id="test_001",
        category="office",
        description="Test task description",
        difficulty="medium",
        examples=[
            Example(input="example input 1", output="example output 1"),
            Example(input="example input 2", output="example output 2"),
            Example(input="example input 3", output="example output 3"),
            Example(input="example input 4", output="example output 4"),
            Example(input="example input 5", output="example output 5"),
            Example(input="example input 6", output="example output 6"),
        ],
        test_cases=[TestCase(input="test input", expected_output="test output", scoring_method="exact_match")],
        distractors=[
            Distractor(input="noise input 1", output="noise output 1"),
            Distractor(input="noise input 2", output="noise output 2"),
        ],
        instruction="Detailed task instruction",
    )


@pytest.fixture
def task_without_distractors():
    """Task without distractors"""
    return Task(
        task_id="test_002",
        category="office",
        description="No-distractor task",
        difficulty="easy",
        examples=[
            Example(input="example input 1", output="example output 1"),
            Example(input="example input 2", output="example output 2"),
        ],
        test_cases=[],
    )


class TestShotConfig:
    """Tests for SHOT_CONFIG constant"""

    def test_shot_config_values(self):
        """Verify shot configuration rules"""
        assert SHOT_CONFIG[0] == (0, 0)  # 0-shot: 0 examples, 0 distractors
        assert SHOT_CONFIG[1] == (1, 0)  # 1-shot: 1 example, 0 distractors
        assert SHOT_CONFIG[2] == (1, 1)  # 2-shot: 1 example, 1 distractor
        assert SHOT_CONFIG[4] == (2, 2)  # 4-shot: 2 examples, 2 distractors
        assert SHOT_CONFIG[8] == (6, 2)  # 8-shot: 6 examples, 2 distractors


class TestSelectExamplesAndDistractors:
    """Tests for select_examples_and_distractors function"""

    def test_0_shot(self, sample_task):
        """0-shot: nothing selected"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 0
        )
        assert len(selected) == 0

    def test_1_shot(self, sample_task):
        """1-shot: only one example"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 1, shuffle=False
        )
        assert len(selected) == 1
        assert selected[0].is_distractor is False
        assert selected[0].input == "example input 1"

    def test_2_shot(self, sample_task):
        """2-shot: 1 example + 1 distractor"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 2, shuffle=False
        )
        assert len(selected) == 2

        examples = [s for s in selected if not s.is_distractor]
        distractors = [s for s in selected if s.is_distractor]

        assert len(examples) == 1
        assert len(distractors) == 1

    def test_4_shot(self, sample_task):
        """4-shot: 2 examples + 2 distractors"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 4, shuffle=False
        )
        assert len(selected) == 4

        examples = [s for s in selected if not s.is_distractor]
        distractors = [s for s in selected if s.is_distractor]

        assert len(examples) == 2
        assert len(distractors) == 2

    def test_8_shot(self, sample_task):
        """8-shot: 6 examples + 2 distractors"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 8, shuffle=False
        )
        assert len(selected) == 8

        examples = [s for s in selected if not s.is_distractor]
        distractors = [s for s in selected if s.is_distractor]

        assert len(examples) == 6
        assert len(distractors) == 2

    def test_shuffle_with_seed(self, sample_task):
        """Shuffle reproducibility with seed"""
        selected1 = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 4, shuffle=True, seed=42
        )
        selected2 = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 4, shuffle=True, seed=42
        )

        # Same seed should produce the same order
        assert [s.input for s in selected1] == [s.input for s in selected2]

    def test_no_distractors_available(self, task_without_distractors):
        """When no distractors available, only examples are returned"""
        selected = select_examples_and_distractors(
            task_without_distractors.examples,
            task_without_distractors.distractors,
            4,
            shuffle=False
        )

        # No distractors, so only 2 examples
        assert len(selected) == 2
        assert all(not s.is_distractor for s in selected)

    def test_invalid_shot_count(self, sample_task):
        """Undefined shot count should use nearest value"""
        # 3 is closest to 2 or 4, uses 2
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 3, shuffle=False
        )
        # 3 -> 2 (nearest): 1 example + 1 distractor = 2
        assert len(selected) == 2


class TestBuildExamplesSection:
    """Tests for build_examples_section function"""

    def test_0_shot_empty(self, sample_task):
        """0-shot should return empty string"""
        result = build_examples_section(sample_task, 0)
        assert result == ""

    def test_1_shot_format(self, sample_task):
        """1-shot format check"""
        result = build_examples_section(sample_task, 1, shuffle=False)

        assert "example input 1" in result
        assert "example output 1" in result
        assert "noise" not in result

    def test_2_shot_includes_distractor(self, sample_task):
        """2-shot should include distractor"""
        result = build_examples_section(sample_task, 2, shuffle=False)

        # Both example and distractor should be included
        assert "example output 1" in result
        assert "noise output 1" in result


class TestBuildPrompt:
    """Tests for build_prompt function"""

    def test_0_shot_prompt(self, sample_task):
        """0-shot prompt structure"""
        prompt = build_prompt(sample_task, "test input data", 0)

        assert "Detailed task instruction" in prompt  # instruction is used
        assert "test input data" in prompt

    def test_uses_instruction_over_description(self, sample_task):
        """Should use instruction when available"""
        prompt = build_prompt(sample_task, "test", 0)
        assert "Detailed task instruction" in prompt
        assert "Test task description" not in prompt

    def test_uses_description_when_no_instruction(self, task_without_distractors):
        """Should use description when instruction is absent"""
        prompt = build_prompt(task_without_distractors, "test", 0)
        assert "No-distractor task" in prompt

    def test_1_shot_prompt(self, sample_task):
        """1-shot prompt structure"""
        prompt = build_prompt(
            sample_task, "test input data", 1, shuffle=False,
            example_selection=ExampleSelectionMethod.FIXED,
        )

        assert "example input 1" in prompt

    def test_prompt_with_seed(self, sample_task):
        """Seed should ensure prompt reproducibility"""
        prompt1 = build_prompt(sample_task, "test", 4, shuffle=True, seed=123)
        prompt2 = build_prompt(sample_task, "test", 4, shuffle=True, seed=123)

        assert prompt1 == prompt2
