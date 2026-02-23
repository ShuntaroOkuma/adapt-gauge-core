"""
prompt_builder.py の単体テスト
"""

import pytest

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
    """テスト用のサンプルタスク"""
    return Task(
        task_id="test_001",
        category="office",
        description="テストタスクの説明",
        difficulty="medium",
        examples=[
            Example(input="手本入力1", output="手本出力1"),
            Example(input="手本入力2", output="手本出力2"),
            Example(input="手本入力3", output="手本出力3"),
            Example(input="手本入力4", output="手本出力4"),
            Example(input="手本入力5", output="手本出力5"),
            Example(input="手本入力6", output="手本出力6"),
        ],
        test_cases=[TestCase(input="テスト入力", expected_output="テスト出力", scoring_method="exact_match")],
        distractors=[
            Distractor(input="ノイズ入力1", output="ノイズ出力1"),
            Distractor(input="ノイズ入力2", output="ノイズ出力2"),
        ],
        instruction="詳細なタスク指示",
    )


@pytest.fixture
def task_without_distractors():
    """ノイズなしのタスク"""
    return Task(
        task_id="test_002",
        category="office",
        description="ノイズなしタスク",
        difficulty="easy",
        examples=[
            Example(input="手本入力1", output="手本出力1"),
            Example(input="手本入力2", output="手本出力2"),
        ],
        test_cases=[],
    )


class TestShotConfig:
    """SHOT_CONFIG定数のテスト"""

    def test_shot_config_values(self):
        """shot構成ルールの確認"""
        assert SHOT_CONFIG[0] == (0, 0)  # 0-shot: 手本0, ノイズ0
        assert SHOT_CONFIG[1] == (1, 0)  # 1-shot: 手本1, ノイズ0
        assert SHOT_CONFIG[2] == (1, 1)  # 2-shot: 手本1, ノイズ1
        assert SHOT_CONFIG[4] == (2, 2)  # 4-shot: 手本2, ノイズ2
        assert SHOT_CONFIG[8] == (6, 2)  # 8-shot: 手本6, ノイズ2


class TestSelectExamplesAndDistractors:
    """select_examples_and_distractors 関数のテスト"""

    def test_0_shot(self, sample_task):
        """0-shot: 何も選択されない"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 0
        )
        assert len(selected) == 0

    def test_1_shot(self, sample_task):
        """1-shot: 手本1つのみ"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 1, shuffle=False
        )
        assert len(selected) == 1
        assert selected[0].is_distractor is False
        assert selected[0].input == "手本入力1"

    def test_2_shot(self, sample_task):
        """2-shot: 手本1 + ノイズ1"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 2, shuffle=False
        )
        assert len(selected) == 2

        examples = [s for s in selected if not s.is_distractor]
        distractors = [s for s in selected if s.is_distractor]

        assert len(examples) == 1
        assert len(distractors) == 1

    def test_4_shot(self, sample_task):
        """4-shot: 手本2 + ノイズ2"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 4, shuffle=False
        )
        assert len(selected) == 4

        examples = [s for s in selected if not s.is_distractor]
        distractors = [s for s in selected if s.is_distractor]

        assert len(examples) == 2
        assert len(distractors) == 2

    def test_8_shot(self, sample_task):
        """8-shot: 手本6 + ノイズ2"""
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 8, shuffle=False
        )
        assert len(selected) == 8

        examples = [s for s in selected if not s.is_distractor]
        distractors = [s for s in selected if s.is_distractor]

        assert len(examples) == 6
        assert len(distractors) == 2

    def test_shuffle_with_seed(self, sample_task):
        """シャッフルの再現性（シード指定）"""
        selected1 = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 4, shuffle=True, seed=42
        )
        selected2 = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 4, shuffle=True, seed=42
        )

        # 同じシードなら同じ順序
        assert [s.input for s in selected1] == [s.input for s in selected2]

    def test_no_distractors_available(self, task_without_distractors):
        """ノイズがない場合は手本のみ"""
        selected = select_examples_and_distractors(
            task_without_distractors.examples,
            task_without_distractors.distractors,
            4,
            shuffle=False
        )

        # ノイズがないので手本2つのみ
        assert len(selected) == 2
        assert all(not s.is_distractor for s in selected)

    def test_invalid_shot_count(self, sample_task):
        """未定義のshot数は最も近い値に"""
        # 3は2か4に近いので2になる
        selected = select_examples_and_distractors(
            sample_task.examples, sample_task.distractors, 3, shuffle=False
        )
        # 3 -> 2（最も近い）: 手本1 + ノイズ1 = 2
        assert len(selected) == 2


class TestBuildExamplesSection:
    """build_examples_section 関数のテスト"""

    def test_0_shot_empty(self, sample_task):
        """0-shotは空文字列"""
        result = build_examples_section(sample_task, 0)
        assert result == ""

    def test_1_shot_format(self, sample_task):
        """1-shotのフォーマット確認"""
        result = build_examples_section(sample_task, 1, shuffle=False)

        assert "【例】" in result
        assert "入力: 手本入力1" in result
        assert "出力: 手本出力1" in result
        assert "ノイズ" not in result

    def test_2_shot_includes_distractor(self, sample_task):
        """2-shotにはノイズが含まれる"""
        result = build_examples_section(sample_task, 2, shuffle=False)

        assert "【例】" in result
        # 手本とノイズの両方が含まれる
        assert "手本出力1" in result
        assert "ノイズ出力1" in result


class TestBuildPrompt:
    """build_prompt 関数のテスト"""

    def test_0_shot_prompt(self, sample_task):
        """0-shotプロンプトの構造"""
        prompt = build_prompt(sample_task, "テスト入力データ", 0)

        assert "以下のタスクを実行してください。" in prompt
        assert "【タスク】詳細なタスク指示" in prompt  # instructionが使われる
        assert "【例】" not in prompt  # 例示なし
        assert "【あなたの番】" in prompt
        assert "入力: テスト入力データ" in prompt
        assert "出力:" in prompt

    def test_uses_instruction_over_description(self, sample_task):
        """instructionがある場合はそちらを優先"""
        prompt = build_prompt(sample_task, "テスト", 0)
        assert "詳細なタスク指示" in prompt
        assert "テストタスクの説明" not in prompt

    def test_uses_description_when_no_instruction(self, task_without_distractors):
        """instructionがない場合はdescriptionを使用"""
        prompt = build_prompt(task_without_distractors, "テスト", 0)
        assert "ノイズなしタスク" in prompt

    def test_1_shot_prompt(self, sample_task):
        """1-shotプロンプトの構造"""
        prompt = build_prompt(sample_task, "テスト入力データ", 1, shuffle=False)

        assert "【タスク】" in prompt
        assert "【例】" in prompt
        assert "手本入力1" in prompt
        assert "【あなたの番】" in prompt

    def test_prompt_with_seed(self, sample_task):
        """シード指定でプロンプトの再現性確保"""
        prompt1 = build_prompt(sample_task, "テスト", 4, shuffle=True, seed=123)
        prompt2 = build_prompt(sample_task, "テスト", 4, shuffle=True, seed=123)

        assert prompt1 == prompt2
