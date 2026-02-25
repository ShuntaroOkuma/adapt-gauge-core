"""
Example Selection Strategies

Provides different strategies for selecting few-shot examples:
- fixed: Use examples in the order defined in the task pack (default)
- tfidf: Select examples most similar to the test input using TF-IDF cosine similarity
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from .task_loader import Distractor, Example


class ExampleSelectionMethod(str, Enum):
    """Example selection method."""
    FIXED = "fixed"
    TFIDF = "tfidf"


@dataclass
class SelectedExample:
    """A selected example (exemplar or distractor)."""
    input: str
    output: str
    is_distractor: bool


def select_examples_tfidf(
    test_input: str,
    examples: list[Example],
    n_examples: int,
    distractors: list[Distractor] | None = None,
    n_distractors: int = 0,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[SelectedExample]:
    """Select examples most similar to the test input using TF-IDF cosine similarity.

    Args:
        test_input: The test case input to find similar examples for.
        examples: Available exemplars to select from.
        n_examples: Number of exemplars to select.
        distractors: Available distractors (optional).
        n_distractors: Number of distractors to include.
        shuffle: Whether to shuffle the final selection order.
        seed: Random seed for reproducible shuffling.

    Returns:
        List of selected examples, sorted by similarity (or shuffled).
    """
    if not examples:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Build corpus: test_input + all example inputs
    corpus = [test_input] + [ex.input for ex in examples]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute similarity between test_input (index 0) and all examples
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Rank by similarity (descending) and select top n
    ranked_indices = similarities.argsort()[::-1]
    n_select = min(n_examples, len(examples))
    selected_indices = ranked_indices[:n_select]

    selected = [
        SelectedExample(
            input=examples[i].input,
            output=examples[i].output,
            is_distractor=False,
        )
        for i in selected_indices
    ]

    # Add distractors
    if distractors and n_distractors > 0:
        n_dist = min(n_distractors, len(distractors))
        selected.extend(
            SelectedExample(input=d.input, output=d.output, is_distractor=True)
            for d in distractors[:n_dist]
        )

    # Shuffle combined list
    if shuffle and len(selected) > 1:
        rng = random.Random(seed)
        rng.shuffle(selected)

    return selected
