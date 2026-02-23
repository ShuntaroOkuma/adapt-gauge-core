"""
Task Loader

Loads task definitions from JSON files.
Supports both the unified format (task pack format) and individual format.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Example:
    """Task example (exemplar)"""
    input: str
    output: str


@dataclass
class Distractor:
    """Distractor (an example that resembles the correct answer but is essentially irrelevant)"""
    input: str
    output: str


@dataclass
class TestCase:
    """Test case"""
    input: str
    expected_output: str | dict  # Usually a string; a dict for custom scoring (custom:xxx)
    scoring_method: str
    acceptable_variations: list[str] | None = None  # Acceptable variations (referenced during LLM scoring)


# Valid evaluation axes
# - Acquisition, Efficiency, Resilience-Noise: Automatically measured for all tasks
# - Fidelity: Measured for tasks using the f1 score
# - Resilience-Detect, Agency: Measured only for dedicated tasks
VALID_MEASURES = [
    "Acquisition",       # Learning speed (automatically measured)
    "Resilience-Noise",  # Noise resilience (automatically measured when distractors are present)
    "Resilience-Detect", # Contradiction/anomaly detection (dedicated tasks)
    "Efficiency",        # Cost efficiency (automatically measured)
    "Agency",            # Task completion (dedicated tasks)
    "Fidelity",          # Accuracy (tasks using f1 score)
]


@dataclass
class Task:
    """Task definition"""
    task_id: str
    category: str
    description: str
    difficulty: str
    examples: list[Example]
    test_cases: list[TestCase]
    # Optional fields (with default values for backward compatibility)
    version: str = "1.0"
    measures: list[str] | None = None  # Evaluation axes to measure (e.g., ["Acquisition", "Fidelity"])
    instruction: str = ""  # Task purpose and constraints
    distractors: list[Distractor] | None = None  # Distractors (examples that resemble exemplars but are irrelevant)
    agency_config: dict | None = None  # Multi-step Agency configuration

    def __post_init__(self):
        """Post-initialization validation"""
        if self.measures is None:
            # Default to measuring Acquisition
            self.measures = ["Acquisition"]
        if self.distractors is None:
            self.distractors = []
        # Validate evaluation axes
        for measure in self.measures:
            if measure not in VALID_MEASURES:
                raise ValueError(f"Invalid evaluation axis: {measure}. Valid values: {VALID_MEASURES}")


@dataclass
class TaskPack:
    """Task pack definition"""
    pack_id: str
    pack_name: str
    description: str
    version: str
    categories: list[str]
    tasks: list[Task]


def _parse_task_data(data: dict) -> Task:
    """
    Create a Task object from dictionary data

    Args:
        data: Task data dictionary

    Returns:
        Task: Task object
    """
    # Create Example objects
    examples = [
        Example(input=ex["input"], output=ex["output"])
        for ex in data["examples"]
    ]

    # Create Distractor objects (optional)
    distractors = None
    if "distractors" in data and data["distractors"]:
        distractors = [
            Distractor(input=d["input"], output=d["output"])
            for d in data["distractors"]
        ]

    # Create TestCase objects
    test_cases = [
        TestCase(
            input=tc["input"],
            expected_output=tc["expected_output"],
            scoring_method=tc["scoring_method"],
            acceptable_variations=tc.get("acceptable_variations"),
        )
        for tc in data["test_cases"]
    ]

    return Task(
        task_id=data["task_id"],
        category=data["category"],
        description=data["description"],
        difficulty=data["difficulty"],
        examples=examples,
        test_cases=test_cases,
        # Optional fields (using get() with defaults for backward compatibility)
        version=data.get("version", "1.0"),
        measures=data.get("measures"),
        instruction=data.get("instruction", ""),
        distractors=distractors,
        agency_config=data.get("agency_config"),
    )


def load_task_pack(file_path: str) -> TaskPack:
    """
    Load a unified format (task pack format) JSON

    Args:
        file_path: Path to the task pack JSON file

    Returns:
        TaskPack: Task pack object

    Raises:
        FileNotFoundError: If the file does not exist
        KeyError: If a required field is missing
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["pack_id", "pack_name", "description", "version", "categories", "tasks"]
    for field in required_fields:
        if field not in data:
            raise KeyError(f"Required field '{field}' is missing: {file_path}")

    # Parse tasks
    tasks = [_parse_task_data(task_data) for task_data in data["tasks"]]

    return TaskPack(
        pack_id=data["pack_id"],
        pack_name=data["pack_name"],
        description=data["description"],
        version=data["version"],
        categories=data["categories"],
        tasks=tasks
    )


def load_tasks_from_pack(file_path: str) -> list[Task]:
    """
    Get only the task list from a task pack

    Args:
        file_path: Path to the task pack JSON file

    Returns:
        list[Task]: Task list
    """
    task_pack = load_task_pack(file_path)
    return task_pack.tasks


def load_task(file_path: str) -> Task:
    """
    Load a single task JSON (individual format)

    Args:
        file_path: Path to the task JSON file

    Returns:
        Task: Loaded task definition

    Raises:
        FileNotFoundError: If the file does not exist
        KeyError: If a required field is missing
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["task_id", "category", "description", "difficulty", "examples", "test_cases"]
    for field in required_fields:
        if field not in data:
            raise KeyError(f"Required field '{field}' is missing: {file_path}")

    return _parse_task_data(data)


def _is_task_pack(file_path: str) -> bool:
    """
    Determine whether a file is in task pack format

    Args:
        file_path: Path to the JSON file

    Returns:
        bool: True if in task pack format
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return "pack_id" in data and "tasks" in data


def load_all_tasks(tasks_dir: str = "tasks") -> list[Task]:
    """
    Load all tasks from the tasks directory
    Supports both individual format and task pack format

    Args:
        tasks_dir: Directory containing task JSON files

    Returns:
        list[Task]: List of all tasks
    """
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks directory does not exist: {tasks_dir}")

    tasks = []
    for json_file in sorted(tasks_path.glob("*.json")):
        # Skip task pack format files (used for individual loading)
        if json_file.name.startswith("task_pack_"):
            continue
        try:
            task = load_task(str(json_file))
            tasks.append(task)
        except KeyError:
            # Skip if it's a task pack format file
            continue

    return tasks


def get_available_task_packs(tasks_dir: str = "tasks") -> list[dict]:
    """
    Get a list of available task packs

    Args:
        tasks_dir: Directory containing task pack JSON files

    Returns:
        list[dict]: List of task pack information
            [{"pack_id": str, "pack_name": str, "description": str, "file_path": str}, ...]
    """
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        return []

    packs = []
    for json_file in sorted(tasks_path.glob("task_pack_*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "pack_id" in data:
                packs.append({
                    "pack_id": data["pack_id"],
                    "pack_name": data.get("pack_name", data["pack_id"]),
                    "description": data.get("description", ""),
                    "file_path": str(json_file),
                    "task_count": len(data.get("tasks", []))
                })
        except (json.JSONDecodeError, KeyError):
            continue

    return packs
