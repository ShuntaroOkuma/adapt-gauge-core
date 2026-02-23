"""
Text-based scoring functions

Implements scoring logic based on text comparison such as exact match, contains, and F1.
"""

from __future__ import annotations

import re
import unicodedata


def remove_markdown(text: str) -> str:
    """
    Remove markdown formatting

    - Remove code blocks (```...```)
    - Remove inline code (`...`)
    - Remove bullet point symbols (-, *, bullet)
    - Remove numbering from numbered lists (1. 2. etc.)

    Args:
        text: Text that may contain markdown

    Returns:
        Text with markdown removed
    """
    # Extract only the code portion from code blocks (```python ... ``` etc.)
    text = re.sub(r"```[\w]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    # Remove backticks from inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove bullet point symbols
    text = re.sub(r"^[\s]*[-*\u2022]\s+", "", text, flags=re.MULTILINE)
    # Remove numbering from numbered lists
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text


def normalize_text(text: str) -> str:
    """
    Normalize text

    - Remove markdown formatting
    - Unicode normalization (NFKC)
    - Convert to lowercase
    - Collapse consecutive whitespace to a single space
    - Strip leading and trailing whitespace

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Remove markdown formatting
    text = remove_markdown(text)
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Convert to lowercase
    text = text.lower()
    # Collapse consecutive whitespace to a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def score_exact_match(expected: str, actual: str) -> float:
    """
    Exact match evaluation

    Args:
        expected: Expected output
        actual: Actual output

    Returns:
        1.0 (match) or 0.0 (mismatch)
    """
    return 1.0 if normalize_text(expected) == normalize_text(actual) else 0.0


def score_contains(expected: str, actual: str) -> float:
    """
    Check whether the expected output is contained in the actual output

    Args:
        expected: Expected output
        actual: Actual output

    Returns:
        1.0 (contained) or 0.0 (not contained)
    """
    return 1.0 if normalize_text(expected) in normalize_text(actual) else 0.0


# Fallback threshold for whitespace tokenization. If whitespace splitting produces
# this many or more tokens, use them as-is; otherwise fall back to character-type-based tokenization.
_WHITESPACE_TOKEN_THRESHOLD = 3


def _tokenize(text: str) -> list[str]:
    """
    Split text into tokens

    Attempts tokenization by whitespace splitting, and falls back to character-level
    n-gram-based tokenization when the token count is low (e.g., Japanese text).

    Args:
        text: Normalized text

    Returns:
        List of tokens
    """
    # First, split by whitespace
    whitespace_tokens = text.split()

    # If there are enough whitespace tokens, use them as-is
    if len(whitespace_tokens) >= _WHITESPACE_TOKEN_THRESHOLD:
        return whitespace_tokens

    # For Japanese text etc.: character-level tokenization
    # Group consecutive characters of the same type (kanji, katakana, hiragana, ASCII) into tokens
    tokens = []
    current = []
    prev_type = None

    for char in text:
        if char == ' ':
            if current:
                tokens.append(''.join(current))
                current = []
                prev_type = None
            continue

        # Determine character type
        char_type = _char_type(char)
        if prev_type is not None and char_type != prev_type:
            if current:
                tokens.append(''.join(current))
            current = [char]
        else:
            current.append(char)
        prev_type = char_type

    if current:
        tokens.append(''.join(current))

    return tokens if tokens else whitespace_tokens


def _char_type(char: str) -> str:
    """Determine the character type"""
    cp = ord(char)
    # Hiragana
    if 0x3040 <= cp <= 0x309F:
        return 'hiragana'
    # Katakana
    if 0x30A0 <= cp <= 0x30FF:
        return 'katakana'
    # CJK Unified Ideographs (Kanji)
    if 0x4E00 <= cp <= 0x9FFF:
        return 'kanji'
    # ASCII alphanumeric
    if char.isascii() and char.isalnum():
        return 'ascii_alnum'
    # ASCII symbols
    if char.isascii():
        return 'ascii_symbol'
    # Other Unicode
    return 'other'


def score_f1(expected: str, actual: str) -> float:
    """
    Calculate token-based F1 score

    Calculates using whitespace-separated tokens, falling back to character-type-based
    tokenization when the token count is low (e.g., Japanese text).

    Args:
        expected: Expected output
        actual: Actual output

    Returns:
        F1 score (0.0 to 1.0)
    """
    expected_normalized = normalize_text(expected)
    actual_normalized = normalize_text(actual)

    expected_tokens = set(_tokenize(expected_normalized))
    actual_tokens = set(_tokenize(actual_normalized))

    if not expected_tokens or not actual_tokens:
        return 0.0

    # Common tokens (exact match)
    common = expected_tokens & actual_tokens

    # If there are no exact matches, try partial matching
    # Check whether each token from expected appears within the actual text
    if not common:
        expected_hits = sum(
            1 for et in expected_tokens if len(et) >= 2 and et in actual_normalized
        )
        actual_hits = sum(
            1 for at in actual_tokens if len(at) >= 2 and at in expected_normalized
        )
        if expected_hits > 0 or actual_hits > 0:
            # Partial-match-based F1 (bidirectional check for accurate precision/recall)
            precision = min(1.0, actual_hits / len(actual_tokens)) if actual_tokens else 0.0
            recall = min(1.0, expected_hits / len(expected_tokens)) if expected_tokens else 0.0
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
        return 0.0

    # Precision: Proportion of actual output tokens that match the expected output
    precision = len(common) / len(actual_tokens)
    # Recall: Proportion of expected output tokens that were actually produced
    recall = len(common) / len(expected_tokens)

    # F1 score
    f1 = 2 * precision * recall / (precision + recall)
    return f1
