"""
Health Check

Performs connectivity checks for models and the Grader (LLM Judge).
"""

from pathlib import Path
from typing import Callable

from adapt_gauge_core.domain.entities import HealthCheckResult
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient


HEALTH_CHECK_PROMPT = "Reply with only 'OK' if you can read this message."


def health_check_model(
    model_name: str,
    create_client_fn: Callable[[str], ModelClient],
) -> HealthCheckResult:
    """
    Execute a health check for a single model.

    Args:
        model_name: Name of the model to check
        create_client_fn: Function to create a model client

    Returns:
        HealthCheckResult: Health check result
    """
    try:
        client = create_client_fn(model_name)
        response = client.generate(HEALTH_CHECK_PROMPT)
        return HealthCheckResult(
            model_name=model_name,
            success=True,
            latency_ms=response.latency_ms,
            error=None
        )
    except Exception as e:
        return HealthCheckResult(
            model_name=model_name,
            success=False,
            latency_ms=None,
            error=str(e)
        )


def health_check_all_models(
    models: list[str],
    create_client_fn: Callable[[str], ModelClient],
) -> tuple[list[str], list[HealthCheckResult]]:
    """
    Execute health checks for all models.

    Args:
        models: List of model names to check
        create_client_fn: Function to create a model client

    Returns:
        tuple: (list of available models, list of all check results)
    """
    print("=== Model Health Check ===\n")
    results = []
    available_models = []

    for model_name in models:
        print(f"  {model_name}... ", end="", flush=True)
        result = health_check_model(model_name, create_client_fn)
        results.append(result)

        if result.success:
            print(f"OK ({result.latency_ms}ms)")
            available_models.append(model_name)
        else:
            # Display only the first 100 characters of the error message
            error_short = result.error[:100] if result.error else "Unknown error"
            print(f"FAILED")
            print(f"    Error: {error_short}")

    print()
    return available_models, results


def run_health_check(
    models: list[str],
    create_client_fn: Callable[[str], ModelClient] | None = None,
) -> tuple[list[str], list[HealthCheckResult]]:
    """
    Execute health checks for all models (high-level function).

    Uses adapt_gauge_core.model_client.create_client if create_client_fn is not specified.

    Args:
        models: List of model names to check
        create_client_fn: Function to create a model client (optional)

    Returns:
        tuple: (list of available models, list of all check results)
    """
    if create_client_fn is None:
        from adapt_gauge_core.model_client import create_client
        create_client_fn = create_client

    return health_check_all_models(models, create_client_fn)


def get_llm_judge_tasks(task_pack_data) -> list[str]:
    """Get the list of task IDs that use llm_judge within a task pack.

    Args:
        task_pack_data: Task pack object (has .tasks[].test_cases[].scoring_method)

    Returns:
        List of task IDs that use llm_judge
    """
    llm_judge_tasks = []
    for task in task_pack_data.tasks:
        for tc in task.test_cases:
            if tc.scoring_method == "llm_judge":
                llm_judge_tasks.append(task.task_id)
                break
    return llm_judge_tasks


def run_grader_health_check(
    grader_model: str,
    create_client_fn: Callable[[str], ModelClient] | None = None,
) -> tuple[bool, str | None]:
    """Execute a health check for the Grader (LLM Judge).

    Args:
        grader_model: Grader model name
        create_client_fn: Function to create a model client (optional)

    Returns:
        (success, error_message): (True, None) on success, (False, error_message) on failure
    """
    if create_client_fn is None:
        from adapt_gauge_core.model_client import create_client
        create_client_fn = create_client

    try:
        client = create_client_fn(grader_model)
        response = client.generate("Reply with only 'OK'")
        if response.output:
            return True, None
        else:
            return False, f"Grader ({grader_model}) returned an empty response"
    except Exception as e:
        error_msg = (
            f"**Grader ({grader_model})** health check failed.\n"
            f"Error: `{str(e)[:200]}`\n\n"
            f"**Troubleshooting:**\n"
            f"- For Gemini: Run `gcloud auth application-default login`\n"
            f"- For LMStudio: Verify LMStudio is running. In Docker, set `LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1`\n"
            f"- For OpenAI: Set the `OPENAI_API_KEY` environment variable"
        )
        return False, error_msg
