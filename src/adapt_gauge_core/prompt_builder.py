"""
Prompt Builder

Generates prompts with examples based on the shot count.

Shot composition rules:
- 0-shot: instruction + input only
- 1-shot: 1 exemplar
- 2-shot: 1 exemplar + 1 distractor
- 4-shot: 2 exemplars + 2 distractors
- 8-shot: 6 exemplars + 2 distractors
"""

import random
from dataclasses import dataclass
from typing import Union

from .task_loader import Distractor, Example, Task
from .example_selector import (
    ExampleSelectionMethod,
    SelectedExample,
    select_examples_tfidf,
)


# Exemplar/distractor composition per shot count
SHOT_CONFIG: dict[int, tuple[int, int]] = {
    0: (0, 0),   # 0 exemplars, 0 distractors
    1: (1, 0),   # 1 exemplar, 0 distractors
    2: (1, 1),   # 1 exemplar, 1 distractor
    4: (2, 2),   # 2 exemplars, 2 distractors
    8: (6, 2),   # 6 exemplars, 2 distractors
}


def _resolve_shot_count(shot_count: int) -> int:
    """Resolve a shot count to the nearest valid value in SHOT_CONFIG."""
    if shot_count in SHOT_CONFIG:
        return shot_count
    valid_shots = sorted(SHOT_CONFIG.keys())
    return min(valid_shots, key=lambda x: abs(x - shot_count))


def select_examples_and_distractors(
    examples: list[Example],
    distractors: list[Distractor],
    shot_count: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[SelectedExample]:
    """
    Select exemplars and distractors based on the shot count

    Args:
        examples: List of exemplars
        distractors: List of distractors
        shot_count: Shot count (0, 1, 2, 4, 8)
        shuffle: Whether to shuffle exemplars and distractors (to eliminate order bias)
        seed: Random seed for shuffling (for reproducibility)

    Returns:
        List of selected examples
    """
    shot_count = _resolve_shot_count(shot_count)
    n_examples, n_distractors = SHOT_CONFIG[shot_count]

    # Select up to the available number
    selected_examples = [
        SelectedExample(input=ex.input, output=ex.output, is_distractor=False)
        for ex in examples[:n_examples]
    ]

    selected_distractors = [
        SelectedExample(input=d.input, output=d.output, is_distractor=True)
        for d in (distractors or [])[:n_distractors]
    ]

    combined = selected_examples + selected_distractors

    # Shuffle (to eliminate order bias)
    if shuffle and len(combined) > 1:
        if seed is not None:
            random.seed(seed)
        random.shuffle(combined)

    return combined


def build_examples_section(
    task: Task,
    shot_count: int,
    shuffle: bool = True,
    seed: int | None = None,
    example_selection: ExampleSelectionMethod = ExampleSelectionMethod.FIXED,
    test_input: str = "",
) -> str:
    """
    Build the examples section (with exemplar + distractor support)

    Args:
        task: Task definition
        shot_count: Shot count (0, 1, 2, 4, 8)
        shuffle: Whether to shuffle exemplars and distractors
        seed: Random seed for shuffling
        example_selection: Selection method ("fixed" or "tfidf")
        test_input: Test case input (required for tfidf selection)

    Returns:
        Examples section string (empty string for 0-shot)
    """
    if shot_count == 0:
        return ""

    if example_selection == ExampleSelectionMethod.TFIDF and test_input:
        shot_count = _resolve_shot_count(shot_count)
        n_examples, n_distractors = SHOT_CONFIG[shot_count]
        selected = select_examples_tfidf(
            test_input=test_input,
            examples=task.examples,
            n_examples=n_examples,
            distractors=task.distractors,
            n_distractors=n_distractors,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        selected = select_examples_and_distractors(
            task.examples,
            task.distractors,
            shot_count,
            shuffle=shuffle,
            seed=seed,
        )

    if not selected:
        return ""

    lines = ["\u3010\u4f8b\u3011"]
    for i, example in enumerate(selected):
        lines.append(f"\u5165\u529b: {example.input}")
        lines.append(f"\u51fa\u529b: {example.output}")
        if i < len(selected) - 1:
            lines.append("")  # Blank line between examples

    return "\n".join(lines)


def build_prompt(
    task: Task,
    test_input: str,
    shot_count: int,
    shuffle: bool = True,
    seed: int | None = None,
    example_selection: ExampleSelectionMethod = ExampleSelectionMethod.FIXED,
) -> str:
    """
    Build a prompt based on the shot count

    Args:
        task: Task definition
        test_input: Test case input
        shot_count: Shot count (0, 1, 2, 4, 8)
        shuffle: Whether to shuffle exemplars and distractors
        seed: Random seed for shuffling (for reproducibility)
        example_selection: Selection method ("fixed" or "tfidf")

    Returns:
        Constructed prompt string
    """
    examples_section = build_examples_section(
        task, shot_count, shuffle=shuffle, seed=seed,
        example_selection=example_selection, test_input=test_input,
    )

    # Task description (prefer instruction if available)
    task_description = task.instruction if task.instruction else task.description

    # Build prompt template
    prompt_parts = [
        "\u4ee5\u4e0b\u306e\u30bf\u30b9\u30af\u3092\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
        "",
        f"\u3010\u30bf\u30b9\u30af\u3011{task_description}",
    ]

    # Add examples section only if present
    if examples_section:
        prompt_parts.append("")
        prompt_parts.append(examples_section)

    prompt_parts.extend([
        "",
        "\u3010\u3042\u306a\u305f\u306e\u756a\u3011",
        f"\u5165\u529b: {test_input}",
        "\u51fa\u529b:"
    ])

    return "\n".join(prompt_parts)
