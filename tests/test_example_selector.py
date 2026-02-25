"""
Tests for TF-IDF dynamic example selection.
"""

import pytest

from adapt_gauge_core.task_loader import Example, Distractor
from adapt_gauge_core.example_selector import (
    select_examples_tfidf,
    ExampleSelectionMethod,
)


def _examples() -> list[Example]:
    return [
        Example(input="メールを分類してください: 会議の日程変更", output="社内連絡"),
        Example(input="メールを分類してください: 新製品のご案内", output="営業"),
        Example(input="メールを分類してください: サーバー障害のお知らせ", output="障害通知"),
        Example(input="メールを分類してください: 請求書を送付します", output="経理"),
        Example(input="メールを分類してください: 来週の出張について", output="社内連絡"),
        Example(input="メールを分類してください: セキュリティパッチの適用", output="IT"),
    ]


def _distractors() -> list[Distractor]:
    return [
        Distractor(input="天気予報を教えてください", output="東京は晴れ"),
        Distractor(input="レシピを教えてください", output="カレーの作り方"),
    ]


class TestSelectExamplesTfidf:
    """Tests for TF-IDF based example selection."""

    def test_returns_requested_count(self):
        """Should return exactly n_examples examples."""
        result = select_examples_tfidf(
            test_input="メールを分類してください: サーバーのエラー通知",
            examples=_examples(),
            n_examples=2,
        )
        assert len(result) == 2

    def test_selects_most_similar(self):
        """Should select examples most similar to the test input."""
        result = select_examples_tfidf(
            test_input="メールを分類してください: サーバーのエラー通知",
            examples=_examples(),
            n_examples=1,
        )
        # "サーバー障害のお知らせ" should be most similar to "サーバーのエラー通知"
        assert result[0].input == "メールを分類してください: サーバー障害のお知らせ"

    def test_handles_distractors(self):
        """Should include distractors when requested."""
        result = select_examples_tfidf(
            test_input="メールを分類してください: サーバーのエラー通知",
            examples=_examples(),
            n_examples=2,
            distractors=_distractors(),
            n_distractors=1,
        )
        assert len(result) == 3
        distractor_count = sum(1 for r in result if r.is_distractor)
        assert distractor_count == 1

    def test_n_examples_exceeds_available(self):
        """Should return all available when n exceeds count."""
        result = select_examples_tfidf(
            test_input="テスト",
            examples=_examples()[:2],
            n_examples=10,
        )
        assert len(result) == 2

    def test_empty_examples(self):
        """Should return empty list when no examples available."""
        result = select_examples_tfidf(
            test_input="テスト",
            examples=[],
            n_examples=3,
        )
        assert result == []

    def test_result_has_is_distractor_field(self):
        """Each result should have is_distractor field."""
        result = select_examples_tfidf(
            test_input="メールを分類してください: 会議の変更",
            examples=_examples(),
            n_examples=2,
            distractors=_distractors(),
            n_distractors=1,
        )
        for item in result:
            assert hasattr(item, "is_distractor")

    def test_no_distractors_when_none(self):
        """Should work without distractors."""
        result = select_examples_tfidf(
            test_input="テスト入力",
            examples=_examples(),
            n_examples=3,
            distractors=None,
            n_distractors=2,
        )
        assert len(result) == 3
        assert all(not r.is_distractor for r in result)

    def test_shuffle_randomizes_order(self):
        """With shuffle=True, order should be randomized (not always similarity order)."""
        # Run multiple times and check that order varies
        orders = set()
        for seed in range(10):
            result = select_examples_tfidf(
                test_input="メールを分類してください: 会議の変更",
                examples=_examples(),
                n_examples=3,
                shuffle=True,
                seed=seed,
            )
            orders.add(tuple(r.input for r in result))
        # With 10 seeds, we should see at least 2 different orderings
        assert len(orders) >= 2

    def test_no_shuffle_preserves_similarity_order(self):
        """With shuffle=False, results should be in similarity order (most similar first)."""
        result = select_examples_tfidf(
            test_input="メールを分類してください: サーバーのエラー通知",
            examples=_examples(),
            n_examples=3,
            shuffle=False,
        )
        # First result should be the most similar
        assert "サーバー" in result[0].input


class TestExampleSelectionMethod:
    """Tests for ExampleSelectionMethod enum."""

    def test_fixed_value(self):
        assert ExampleSelectionMethod.FIXED == "fixed"

    def test_tfidf_value(self):
        assert ExampleSelectionMethod.TFIDF == "tfidf"

    def test_from_string(self):
        assert ExampleSelectionMethod("fixed") == ExampleSelectionMethod.FIXED
        assert ExampleSelectionMethod("tfidf") == ExampleSelectionMethod.TFIDF
