"""Ollama model benchmarking and multi-configuration experimental testing."""

import json
import time
import urllib.error
import urllib.request

from core.logging import write_log
from core.ollama import model as model_api

COMPONENT = "ollama/experiment"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"


def _generate(
    model: str,
    prompt: str,
    options: dict | None = None,
) -> dict:
    """Run one temporary model configuration against a single prompt.

    The configuration options are applied only to this specific request.

    Args:
        model: Ollama model identifier tag.
        prompt: Text prompt string to evaluate.
        options: Optional generation parameters (e.g., temperature, num_ctx).

    Returns:
        dict: Raw JSON response parsed from Ollama generate API.

    Raises:
        RuntimeError: If connection fails or Ollama returns an error payload.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Failed to run model: {error}") from error

    try:
        result = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON") from error

    if "error" in result:
        raise RuntimeError(result["error"])

    return result


def run_test(
    model: str,
    prompts: list[str],
    config: dict | None = None,
    name: str = "test",
    include_output: bool = False,
) -> dict:
    """Run one temporary model configuration against multiple prompts.

    The model is checked and preloaded before each prompt.
    Model loading is not included in test timing or performance results.

    Args:
        model: Target model name or tag.
        prompts: List of prompt strings to execute.
        config: Optional model parameter dictionary.
        name: Identifier name for the test run. Defaults to 'test'.
        include_output: Whether to include the generated text in results. Defaults to False.

    Returns:
        dict: Test execution results and summary statistics.

    Raises:
        ValueError: If model name is empty, prompts list is empty.
        TypeError: If include_output is not boolean or any prompt is not a string.
    """
    if not model.strip():
        raise ValueError("Model name is required")

    if not prompts:
        raise ValueError("At least one prompt is required")

    if not isinstance(include_output, bool):
        raise TypeError("include_output must be a boolean")

    configuration = dict(config or {})

    write_log(
        level="INFO",
        component=COMPONENT,
        action="test",
        message="Test started",
        details={
            "name": name,
        },
    )

    results = []

    for index, prompt in enumerate(prompts, start=1):
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt {index} must be a string")

        try:
            # Ensure the model is loaded before the test.
            # This operation is intentionally outside the
            # benchmark timer and is not included in results.
            model_api.load_model(model)

            started_at = time.perf_counter()

            response = _generate(
                model=model,
                prompt=prompt,
                options=configuration,
            )

            duration = time.perf_counter() - started_at

            prompt_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)
            prompt_duration_ns = response.get("prompt_eval_duration", 0)
            output_duration_ns = response.get("eval_duration", 0)

            prompt_tokens_per_second = _tokens_per_second(
                prompt_tokens,
                prompt_duration_ns,
            )

            output_tokens_per_second = _tokens_per_second(
                output_tokens,
                output_duration_ns,
            )

            result = {
                "index": index,
                "success": True,
                "prompt": prompt,
                "duration_seconds": duration,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "prompt_tokens_per_second": prompt_tokens_per_second,
                "output_tokens_per_second": output_tokens_per_second,
                "done": response.get("done", True),
            }

            if include_output:
                result["response"] = response.get("response", "")

            results.append(result)

            write_log(
                level="INFO",
                component=COMPONENT,
                action="test",
                message="Prompt executed",
                details={
                    "name": name,
                    "prompt_index": index,
                    "success": result["success"],
                    "duration_seconds": result["duration_seconds"],
                    "prompt_tokens": result["prompt_tokens"],
                    "output_tokens": result["output_tokens"],
                    "prompt_tokens_per_second": result["prompt_tokens_per_second"],
                    "output_tokens_per_second": result["output_tokens_per_second"],
                    "done": result["done"],
                },
            )

        except Exception as error:
            duration = (
                time.perf_counter() - started_at
                if "started_at" in locals()
                else 0.0
            )

            result = {
                "index": index,
                "success": False,
                "prompt": prompt,
                "duration_seconds": duration,
                "error": str(error),
            }

            results.append(result)

            write_log(
                level="ERROR",
                component=COMPONENT,
                action="test",
                message="Prompt execution failed",
                details={
                    "name": name,
                    "prompt_index": index,
                    "success": False,
                    "duration_seconds": duration,
                    "error": str(error),
                },
            )

    successful = [result for result in results if result["success"]]
    summary = _build_summary(results=successful)

    result = {
        "name": name,
        "model": model,
        "configuration": configuration,
        "results": results,
        "summary": summary,
    }

    write_log(
        level="INFO",
        component=COMPONENT,
        action="test",
        message="Test completed",
        details={
            "name": name,
        },
    )

    return result


def compare_tests(
    model: str,
    prompts: list[str],
    configurations: list[dict],
    include_output: bool = False,
) -> dict:
    """Run the same prompts against multiple temporary model configurations.

    Args:
        model: Ollama model name.
        prompts: List of evaluation prompt strings.
        configurations: List of configuration dictionaries with 'name' and 'options'.
        include_output: Whether to include generated output in test results. Defaults to False.

    Returns:
        dict: Aggregated comparison results across all configurations.

    Raises:
        ValueError: If model, prompts, or configurations are empty.
        TypeError: If configurations or prompt items are invalid types.
    """
    if not model.strip():
        raise ValueError("Model name is required")

    if not prompts:
        raise ValueError("At least one prompt is required")

    if not configurations:
        raise ValueError("At least one configuration is required")

    if not isinstance(include_output, bool):
        raise TypeError("include_output must be a boolean")

    normalized_configurations = []

    for index, configuration in enumerate(configurations, start=1):
        if not isinstance(configuration, dict):
            raise TypeError(f"Configuration {index} must be a dictionary")

        name = configuration.get("name", f"configuration_{index}")
        options = configuration.get("options", {})

        if not isinstance(options, dict):
            raise TypeError(
                f"Configuration '{name}' options must be a dictionary"
            )

        normalized_configurations.append({
            "name": name,
            "options": options,
        })

    write_log(
        level="INFO",
        component=COMPONENT,
        action="compare",
        message="Tests started",
        details={
            "model": model,
            "prompts": prompts,
            "configurations": normalized_configurations,
        },
    )

    tests = []

    for configuration in normalized_configurations:
        result = run_test(
            model=model,
            prompts=prompts,
            config=configuration["options"],
            name=configuration["name"],
            include_output=include_output,
        )
        tests.append(result)

    result = {
        "model": model,
        "tests": tests,
    }

    write_log(
        level="INFO",
        component=COMPONENT,
        action="compare",
        message="Tests completed",
        details={
            "model": model,
        },
    )

    return result


def _build_summary(results: list[dict]) -> dict:
    """Build aggregate metrics from successful test results.

    Args:
        results: List of successful test result dictionaries.

    Returns:
        dict: Summary statistics including average durations, rates, and token counts.
    """
    if not results:
        return {
            "average_duration_seconds": None,
            "average_prompt_tokens_per_second": None,
            "average_output_tokens_per_second": None,
            "total_output_tokens": 0,
        }

    average_duration = (
        sum(result["duration_seconds"] for result in results) / len(results)
    )

    prompt_rates = [
        result["prompt_tokens_per_second"]
        for result in results
        if result.get("prompt_tokens_per_second") is not None
    ]

    output_rates = [
        result["output_tokens_per_second"]
        for result in results
        if result.get("output_tokens_per_second") is not None
    ]

    total_output_tokens = sum(
        result["output_tokens"] for result in results
    )

    return {
        "average_duration_seconds": average_duration,
        "average_prompt_tokens_per_second": (
            sum(prompt_rates) / len(prompt_rates)
            if prompt_rates
            else None
        ),
        "average_output_tokens_per_second": (
            sum(output_rates) / len(output_rates)
            if output_rates
            else None
        ),
        "total_output_tokens": total_output_tokens,
    }


def _tokens_per_second(
    token_count: int | float,
    duration_ns: int | float,
) -> float | None:
    """Calculate token generation rate in tokens per second.

    Args:
        token_count: Number of processed or evaluated tokens.
        duration_ns: Processing duration in nanoseconds.

    Returns:
        float | None: Tokens per second rate, or None if invalid.
    """
    if not token_count or not duration_ns:
        return None

    duration_seconds = duration_ns / 1_000_000_000

    if duration_seconds <= 0:
        return None

    return token_count / duration_seconds
