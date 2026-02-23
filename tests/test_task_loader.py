"""
task_loader.py の単体テスト
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
    """Distractor dataclass のテスト"""

    def test_create_distractor(self):
        """Distractorの作成"""
        distractor = Distractor(
            input="これは雑談ログです。",
            output="雑談の要約"
        )
        assert distractor.input == "これは雑談ログです。"
        assert distractor.output == "雑談の要約"


class TestTaskNewFields:
    """Task dataclass の新フィールドのテスト"""

    def test_task_with_new_fields(self):
        """新フィールド付きでTaskを作成"""
        task = Task(
            task_id="test_001",
            category="office",
            description="テストタスク",
            difficulty="medium",
            examples=[Example(input="入力1", output="出力1")],
            test_cases=[TestCase(input="テスト入力", expected_output="テスト出力", scoring_method="exact_match")],
            version="1.1",
            measures=["Acquisition", "Fidelity"],
            instruction="タスクの指示",
            distractors=[Distractor(input="ノイズ入力", output="ノイズ出力")],
        )

        assert task.version == "1.1"
        assert task.measures == ["Acquisition", "Fidelity"]
        assert task.instruction == "タスクの指示"
        assert len(task.distractors) == 1
        assert task.distractors[0].input == "ノイズ入力"

    def test_task_default_values(self):
        """デフォルト値のテスト（後方互換性）"""
        task = Task(
            task_id="test_002",
            category="office",
            description="テストタスク",
            difficulty="easy",
            examples=[],
            test_cases=[],
        )

        assert task.version == "1.0"
        assert task.measures == ["Acquisition"]  # デフォルト
        assert task.instruction == ""
        assert task.distractors == []

    def test_task_invalid_measure(self):
        """無効な評価軸でエラーが出ることを確認"""
        with pytest.raises(ValueError) as excinfo:
            Task(
                task_id="test_003",
                category="office",
                description="テストタスク",
                difficulty="easy",
                examples=[],
                test_cases=[],
                measures=["InvalidMeasure"],
            )
        assert "Invalid evaluation axis" in str(excinfo.value)

    def test_valid_measures_constant(self):
        """VALID_MEASURES定数の確認"""
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
    """_parse_task_data 関数のテスト"""

    def test_parse_with_new_fields(self):
        """新フィールドを含むデータのパース"""
        data = {
            "task_id": "parse_test_001",
            "category": "dev",
            "description": "パーステスト",
            "difficulty": "hard",
            "examples": [{"input": "例入力", "output": "例出力"}],
            "test_cases": [{"input": "TC入力", "expected_output": "TC出力", "scoring_method": "f1"}],
            "version": "2.0",
            "measures": ["Acquisition", "Efficiency"],
            "instruction": "パース用指示",
            "distractors": [{"input": "ノイズ", "output": "ノイズ出力"}],
        }

        task = _parse_task_data(data)

        assert task.task_id == "parse_test_001"
        assert task.version == "2.0"
        assert task.measures == ["Acquisition", "Efficiency"]
        assert task.instruction == "パース用指示"
        assert len(task.distractors) == 1

    def test_parse_without_new_fields(self):
        """新フィールドなし（後方互換性）"""
        data = {
            "task_id": "parse_test_002",
            "category": "office",
            "description": "後方互換テスト",
            "difficulty": "easy",
            "examples": [],
            "test_cases": [],
        }

        task = _parse_task_data(data)

        assert task.task_id == "parse_test_002"
        assert task.version == "1.0"  # デフォルト
        assert task.measures == ["Acquisition"]  # デフォルト
        assert task.instruction == ""
        assert task.distractors == []

    def test_parse_empty_distractors(self):
        """空のdistractors配列"""
        data = {
            "task_id": "parse_test_003",
            "category": "office",
            "description": "空ノイズテスト",
            "difficulty": "easy",
            "examples": [],
            "test_cases": [],
            "distractors": [],
        }

        task = _parse_task_data(data)
        assert task.distractors == []

    def test_parse_with_agency_config(self):
        """agency_config が JSON から正しく読み込まれること"""
        data = {
            "task_id": "agency_test_001",
            "category": "dev",
            "description": "Agency設定テスト",
            "difficulty": "hard",
            "examples": [{"input": "例入力", "output": "例出力"}],
            "test_cases": [{"input": "TC入力", "expected_output": "TC出力", "scoring_method": "exact_match"}],
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
        """agency_config がない場合に None になること"""
        data = {
            "task_id": "agency_test_002",
            "category": "office",
            "description": "Agency設定なしテスト",
            "difficulty": "easy",
            "examples": [],
            "test_cases": [],
        }

        task = _parse_task_data(data)

        assert task.agency_config is None


class TestLoadTask:
    """load_task 関数のテスト（新フィールド対応）"""

    def test_load_task_with_new_fields(self, tmp_path):
        """新フィールド付きJSONの読み込み"""
        task_data = {
            "task_id": "load_test_001",
            "category": "legal",
            "description": "読み込みテスト",
            "difficulty": "medium",
            "examples": [{"input": "契約書A", "output": "要約A"}],
            "test_cases": [{"input": "契約書B", "expected_output": "要約B", "scoring_method": "contains"}],
            "version": "1.5",
            "measures": ["Fidelity", "Acquisition"],
            "instruction": "契約書を正確に要約してください",
            "distractors": [
                {"input": "雑談ログ", "output": "雑談要約"},
                {"input": "メモ書き", "output": "メモ要約"},
            ],
        }

        task_file = tmp_path / "test_task.json"
        task_file.write_text(json.dumps(task_data, ensure_ascii=False), encoding="utf-8")

        task = load_task(str(task_file))

        assert task.task_id == "load_test_001"
        assert task.version == "1.5"
        assert task.measures == ["Fidelity", "Acquisition"]
        assert task.instruction == "契約書を正確に要約してください"
        assert len(task.distractors) == 2

    def test_load_legacy_task(self, tmp_path):
        """旧形式（新フィールドなし）JSONの読み込み"""
        task_data = {
            "task_id": "legacy_001",
            "category": "summarization",
            "description": "旧形式タスク",
            "difficulty": "easy",
            "examples": [{"input": "入力", "output": "出力"}],
            "test_cases": [{"input": "TC", "expected_output": "EO", "scoring_method": "exact_match"}],
        }

        task_file = tmp_path / "legacy_task.json"
        task_file.write_text(json.dumps(task_data, ensure_ascii=False), encoding="utf-8")

        task = load_task(str(task_file))

        assert task.task_id == "legacy_001"
        assert task.version == "1.0"
        assert task.measures == ["Acquisition"]
        assert task.distractors == []
