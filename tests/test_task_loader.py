"""
Unit tests for task_loader.py
"""

import json
import tempfile
from pathlib import Path

import pytest

from adapt_gauge_core.task_loader import (
    Distractor,
    Example,
    Task,
    TestCase,
    VALID_MEASURES,
    _parse_task_data,
    load_task,
)


class TestDistractor:
    """Tests for Distractor dataclass"""

    def test_create_distractor(self):
        """Create a Distractor instance"""
        distractor = Distractor(
            input="This is a chat log.",
            output="Chat summary"
        )
        assert distractor.input == "This is a chat log."
        assert distractor.output == "Chat summary"


class TestTaskNewFields:
    """Tests for Task dataclass new fields"""

    def test_task_with_new_fields(self):
        """Create Task with new fields"""
        task = Task(
            task_id="test_001",
            category="office",
            description="Test task",
            difficulty="medium",
            examples=[Example(input="input1", output="output1")],
            test_cases=[TestCase(input="test input", expected_output="test output", scoring_method="exact_match")],
            version="1.1",
            measures=["Acquisition", "Fidelity"],
            instruction="Task instruction",
            distractors=[Distractor(input="noise input", output="noise output")],
        )

        assert task.version == "1.1"
        assert task.measures == ["Acquisition", "Fidelity"]
        assert task.instruction == "Task instruction"
        assert len(task.distractors) == 1
        assert task.distractors[0].input == "noise input"

    def test_task_default_values(self):
        """Test default values (backward compatibility)"""
        task = Task(
            task_id="test_002",
            category="office",
            description="Test task",
            difficulty="easy",
            examples=[],
            test_cases=[],
        )

        assert task.version == "1.0"
        assert task.measures == ["Acquisition"]  # default
        assert task.instruction == ""
        assert task.distractors == []

    def test_task_invalid_measure(self):
        """Should raise error for invalid evaluation axis"""
        with pytest.raises(ValueError) as excinfo:
            Task(
                task_id="test_003",
                category="office",
                description="Test task",
                difficulty="easy",
                examples=[],
                test_cases=[],
                measures=["InvalidMeasure"],
            )
        assert "Invalid evaluation axis" in str(excinfo.value)

    def test_valid_measures_constant(self):
        """Verify VALID_MEASURES constant"""
        expected = [
            "Acquisition",
            "Resilience-Noise",
            "Resilience-Detect",
            "Efficiency",
            "Agency",
            "Fidelity",
        ]
        assert VALID_MEASURES == expected


class TestParseTaskData:
    """Tests for _parse_task_data function"""

    def test_parse_with_new_fields(self):
        """Parse data with new fields"""
        data = {
            "task_id": "parse_test_001",
            "category": "dev",
            "description": "Parse test",
            "difficulty": "hard",
            "examples": [{"input": "example input", "output": "example output"}],
            "test_cases": [{"input": "TC input", "expected_output": "TC output", "scoring_method": "f1"}],
            "version": "2.0",
            "measures": ["Acquisition", "Efficiency"],
            "instruction": "Parsing instruction",
            "distractors": [{"input": "noise", "output": "noise output"}],
        }

        task = _parse_task_data(data)

        assert task.task_id == "parse_test_001"
        assert task.version == "2.0"
        assert task.measures == ["Acquisition", "Efficiency"]
        assert task.instruction == "Parsing instruction"
        assert len(task.distractors) == 1

    def test_parse_without_new_fields(self):
        """Parse data without new fields (backward compatibility)"""
        data = {
            "task_id": "parse_test_002",
            "category": "office",
            "description": "Backward compat test",
            "difficulty": "easy",
            "examples": [],
            "test_cases": [],
        }

        task = _parse_task_data(data)

        assert task.task_id == "parse_test_002"
        assert task.version == "1.0"  # default
        assert task.measures == ["Acquisition"]  # default
        assert task.instruction == ""
        assert task.distractors == []

    def test_parse_empty_distractors(self):
        """Parse with empty distractors array"""
        data = {
            "task_id": "parse_test_003",
            "category": "office",
            "description": "Empty noise test",
            "difficulty": "easy",
            "examples": [],
            "test_cases": [],
            "distractors": [],
        }

        task = _parse_task_data(data)
        assert task.distractors == []

    def test_parse_with_agency_config(self):
        """agency_config should be correctly loaded from JSON"""
        data = {
            "task_id": "agency_test_001",
            "category": "dev",
            "description": "Agency config test",
            "difficulty": "hard",
            "examples": [{"input": "example input", "output": "example output"}],
            "test_cases": [{"input": "TC input", "expected_output": "TC output", "scoring_method": "exact_match"}],
            "agency_config": {
                "max_steps": 5,
                "tools": ["web_search", "code_exec"],
                "stop_condition": "task_complete"
            },
        }

        task = _parse_task_data(data)

        assert task.agency_config is not None
        assert task.agency_config["max_steps"] == 5
        assert "web_search" in task.agency_config["tools"]
        assert task.agency_config["stop_condition"] == "task_complete"

    def test_parse_without_agency_config(self):
        """agency_config should be None when not present"""
        data = {
            "task_id": "agency_test_002",
            "category": "office",
            "description": "No agency config test",
            "difficulty": "easy",
            "examples": [],
            "test_cases": [],
        }

        task = _parse_task_data(data)

        assert task.agency_config is None


class TestLoadTask:
    """Tests for load_task function (with new fields)"""

    def test_load_task_with_new_fields(self, tmp_path):
        """Load JSON with new fields"""
        task_data = {
            "task_id": "load_test_001",
            "category": "legal",
            "description": "Load test",
            "difficulty": "medium",
            "examples": [{"input": "Contract A", "output": "Summary A"}],
            "test_cases": [{"input": "Contract B", "expected_output": "Summary B", "scoring_method": "contains"}],
            "version": "1.5",
            "measures": ["Fidelity", "Acquisition"],
            "instruction": "Summarize the contract accurately",
            "distractors": [
                {"input": "Chat log", "output": "Chat summary"},
                {"input": "Memo notes", "output": "Memo summary"},
            ],
        }

        task_file = tmp_path / "test_task.json"
        task_file.write_text(json.dumps(task_data, ensure_ascii=False), encoding="utf-8")

        task = load_task(str(task_file))

        assert task.task_id == "load_test_001"
        assert task.version == "1.5"
        assert task.measures == ["Fidelity", "Acquisition"]
        assert task.instruction == "Summarize the contract accurately"
        assert len(task.distractors) == 2

    def test_load_legacy_task(self, tmp_path):
        """Load legacy format JSON (without new fields)"""
        task_data = {
            "task_id": "legacy_001",
            "category": "summarization",
            "description": "Legacy task",
            "difficulty": "easy",
            "examples": [{"input": "input", "output": "output"}],
            "test_cases": [{"input": "TC", "expected_output": "EO", "scoring_method": "exact_match"}],
        }

        task_file = tmp_path / "legacy_task.json"
        task_file.write_text(json.dumps(task_data, ensure_ascii=False), encoding="utf-8")

        task = load_task(str(task_file))

        assert task.task_id == "legacy_001"
        assert task.version == "1.0"
        assert task.measures == ["Acquisition"]
        assert task.distractors == []
