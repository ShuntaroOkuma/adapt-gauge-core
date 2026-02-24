"""
Tests for text-based scoring functions

Tests for score_exact_match, score_contains, score_f1, normalize_text, and _tokenize.
"""

import pytest

from adapt_gauge_core.scoring.text_scorers import (
    remove_markdown,
    normalize_text,
    score_exact_match,
    score_contains,
    score_f1,
    _tokenize,
    _char_type,
)


class TestRemoveMarkdown:
    """Tests for remove_markdown"""

    def test_code_block_removal(self):
        text = "```python\nprint('hello')\n```"
        result = remove_markdown(text)
        assert "```" not in result
        assert "print('hello')" in result

    def test_inline_code_removal(self):
        text = "Use `print()` to output"
        result = remove_markdown(text)
        assert "`" not in result
        assert "print()" in result

    def test_bullet_list_removal(self):
        text = "- item1\n* item2\n- item3"
        result = remove_markdown(text)
        assert "- " not in result
        assert "* " not in result

    def test_numbered_list_removal(self):
        text = "1. first\n2. second"
        result = remove_markdown(text)
        assert "1. " not in result
        assert "first" in result

    def test_plain_text_unchanged(self):
        text = "Hello, world!"
        assert remove_markdown(text) == text


class TestNormalizeText:
    """Tests for normalize_text"""

    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_whitespace_normalization(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strip(self):
        assert normalize_text("  hello  ") == "hello"

    def test_unicode_normalization(self):
        # NFKC normalization: fullwidth alphanumeric -> halfwidth
        assert normalize_text("\uff21\uff22\uff23") == "abc"

    def test_markdown_removal_in_normalize(self):
        text = "```\ncode\n```"
        result = normalize_text(text)
        assert "```" not in result
        assert "code" in result

    def test_combined_normalization(self):
        text = "  `Hello`   WORLD  "
        assert normalize_text(text) == "hello world"


class TestScoreExactMatch:
    """Tests for score_exact_match"""

    def test_exact_match(self):
        assert score_exact_match("Hello", "hello") == 1.0

    def test_exact_match_with_whitespace(self):
        assert score_exact_match("hello  world", "hello world") == 1.0

    def test_mismatch(self):
        assert score_exact_match("hello", "world") == 0.0

    def test_empty_strings(self):
        assert score_exact_match("", "") == 1.0

    def test_case_insensitive(self):
        assert score_exact_match("ABC", "abc") == 1.0


class TestScoreContains:
    """Tests for score_contains"""

    def test_contains(self):
        assert score_contains("hello", "say hello world") == 1.0

    def test_not_contains(self):
        assert score_contains("hello", "goodbye world") == 0.0

    def test_exact_match_is_contains(self):
        assert score_contains("hello", "hello") == 1.0

    def test_case_insensitive(self):
        assert score_contains("Hello", "say hello") == 1.0


class TestTokenize:
    """Tests for _tokenize"""

    def test_whitespace_tokens(self):
        tokens = _tokenize("hello world foo bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_japanese_text_char_type_tokenize(self):
        # Japanese text: "abc商事売上500億円" (abc Corp. revenue 500 billion yen)
        tokens = _tokenize("abc商事売上500億円")
        assert "abc" in tokens
        assert "500" in tokens
        assert any("商事" in t for t in tokens)

    def test_mixed_japanese_with_spaces(self):
        # Japanese text: "支払い 100万円 月末" (Payment 1M yen end of month)
        tokens = _tokenize("支払い 100万円 月末")
        assert len(tokens) >= 3

    def test_empty_string(self):
        tokens = _tokenize("")
        assert tokens == []


class TestCharType:
    """Tests for _char_type"""

    def test_hiragana(self):
        assert _char_type("あ") == "hiragana"

    def test_katakana(self):
        assert _char_type("ア") == "katakana"

    def test_kanji(self):
        assert _char_type("漢") == "kanji"

    def test_ascii_alnum(self):
        assert _char_type("a") == "ascii_alnum"
        assert _char_type("1") == "ascii_alnum"

    def test_ascii_symbol(self):
        assert _char_type("!") == "ascii_symbol"


class TestScoreF1:
    """Tests for score_f1"""

    def test_exact_match_f1(self):
        assert score_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert score_f1("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_partial_overlap(self):
        result = score_f1("hello world foo", "hello world bar")
        assert 0.0 < result < 1.0

    def test_japanese_keyword_matching(self):
        # Japanese: "ABC Corp. revenue 500B yen +12%"
        expected = "ABC商事 売上 500億円 +12%"
        actual = "ABC商事：売上500億円（+12%）の業績報告です"
        result = score_f1(expected, actual)
        assert result > 0.0

    def test_empty_expected(self):
        assert score_f1("", "hello") == 0.0

    def test_empty_actual(self):
        assert score_f1("hello", "") == 0.0

    def test_both_empty(self):
        assert score_f1("", "") == 0.0

    def test_japanese_contract_keywords(self):
        # Japanese: "Payment 1M yen, consumption tax, end of month, next month end, bank transfer, handling fee"
        expected = "支払い 100万円 消費税 月末 締め 翌月末 振込 銀行 手数料 負担"
        actual = "支払いは100万円（消費税別）です。月末締めで翌月末に銀行振込でお支払いください。振込手数料はご負担ください。"
        result = score_f1(expected, actual)
        assert result > 0.0
