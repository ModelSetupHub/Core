"""Ollama model benchmarking and multi-configuration experimental testing."""

import json
import threading
import time
import urllib.error
import urllib.request

from core.cancellation import (
    POLL_INTERVAL,
    CancellationToken,
    OperationCancelled,
    log_cancelled,
)
from core.logging import write_log
from core.ollama import model as model_api

COMPONENT = "ollama/experiment"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"


def _generate(
    model: str,
    prompt: str,
    options: dict | None = None,
    cancellation: CancellationToken | None = None,
) -> dict:
    """Run one temporary model configuration against a single prompt.

    The configuration options are applied only to this specific request.

    The request is streamed so a cancellation can take effect mid-generation:
    with a single buffered response there is no point between sending the
    prompt and receiving the whole answer at which the operation could stop.
    The final streamed object carries the same timing and token fields a
    buffered response would, so the returned dictionary is unchanged.

    Args:
        model: Ollama model identifier tag.
        prompt: Text prompt string to evaluate.
        options: Optional generation parameters (e.g., temperature, num_ctx).
        cancellation: Optional token that stops the generation part-way.

    Returns:
        dict: Response fields parsed from the Ollama generate API, with the
            generated text collected under 'response'.

    Raises:
        RuntimeError: If connection fails or Ollama returns an error payload.
        OperationCancelled: If the token is cancelled during generation.
    """
    # A run without a token still has one, so every check below is a plain
    # token call rather than a None test guarding it. A token nobody holds is
    # never cancelled, which is exactly the uncancellable behaviour None asked
    # for.
    token = cancellation if cancellation is not None else CancellationToken()

    token.raise_if_cancelled()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
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
        response = urllib.request.urlopen(request, timeout=300)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Failed to run model: {error}") from error

    # Closing the response is what actually interrupts a generation in
    # progress: the reader below is blocked in a socket read that no flag check
    # can reach, so the close is done from a watcher thread and surfaces here
    # as a read failure, which the cancellation check just after turns into a
    # clean stop.
    watcher = _CancelWatcher(response=response, cancellation=token)
    watcher.start()

    chunks: list[str] = []
    final: dict = {}

    try:
        for line in response:
            token.raise_if_cancelled()

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError("Ollama returned invalid JSON") from error

            if "error" in event:
                raise RuntimeError(event["error"])

            text = event.get("response")
            if text:
                chunks.append(text)

            if event.get("done"):
                final = event
    except OperationCancelled:
        raise
    except RuntimeError:
        # Already a described failure — an error payload or malformed JSON.
        raise
    except Exception as error:
        # Interrupting a generation means closing the socket out from under
        # this reader, and what that surfaces as depends on how far the
        # response had got: a URLError, an OSError, or an AttributeError from
        # the emptied buffer. So the token is consulted before the error is
        # believed — otherwise a cancellation would be recorded as a failed
        # prompt.
        token.raise_if_cancelled()
        raise RuntimeError(f"Failed to run model: {error}") from error
    finally:
        watcher.stop()
        try:
            response.close()
        except Exception:
            pass

    if not final:
        token.raise_if_cancelled()
        raise RuntimeError("Ollama closed the stream before finishing")

    final["response"] = "".join(chunks)

    return final


class _CancelWatcher:
    """Closes an open response as soon as a cancellation token is set."""

    def __init__(
        self,
        response,
        cancellation: CancellationToken,
    ) -> None:
        """Prepare a watcher for one response.

        Args:
            response: Open HTTP response to close on cancellation.
            cancellation: Token to watch.
        """
        self._response = response
        self._cancellation = cancellation
        self._consumed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin watching the response."""
        self._thread = threading.Thread(
            target=self._watch,
            name="ollama-generate-cancel",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop watching once the response has been consumed."""
        self._consumed.set()

    def _watch(self) -> None:
        """Close the response when cancelled, or exit when it is consumed."""
        # Waits on the token rather than polling it, so a cancellation closes
        # the socket at once; the interval only bounds how long the watcher
        # takes to notice that the response was consumed and it can retire.
        while not self._consumed.is_set():
            if self._cancellation.wait(POLL_INTERVAL):
                try:
                    self._response.close()
                except Exception:
                    pass
                return


def run_test(
    model: str,
    prompts: list[str],
    config: dict | None = None,
    name: str = "test",
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
) -> dict:
    """Run one temporary model configuration against multiple prompts.

    The model is checked and preloaded before each prompt.
    Model loading is not included in test timing or performance results.

    Args:
        model: Target model name or tag.
        prompts: List of prompt strings to execute.
        config: Optional model parameter dictionary.
        name: Identifier name for the test run. Defaults to 'test'.
        include_output: Whether to include the generated text in results.
            Defaults to False.
        cancellation: Optional token that stops the run between or during
            prompts.

    Returns:
        dict: Test execution results and summary statistics.

    Raises:
        ValueError: If model name is empty or prompts list is empty.
        TypeError: If include_output is not boolean or any prompt is not a
            string.
        OperationCancelled: If the token is cancelled. Partial results are
            discarded and the model this run loaded is unloaded first, so a
            cancelled run leaves nothing behind but its log entry.
    """
    if not model.strip():
        raise ValueError("Model name is required")

    if not prompts:
        raise ValueError("At least one prompt is required")

    if not isinstance(include_output, bool):
        raise TypeError("include_output must be a boolean")

    # One token for the whole run, so the prompt loop, `_generate` and the
    # watcher thread all consult the same flag whether or not a caller supplied
    # one. An unsupplied token is simply never cancelled.
    token = cancellation if cancellation is not None else CancellationToken()

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

    try:
        for index, prompt in enumerate(prompts, start=1):
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt {index} must be a string")

            token.raise_if_cancelled()

            # Reset per prompt: a failure before the timer starts must not
            # report the previous prompt's start time.
            started_at: float | None = None

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
                    cancellation=token,
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
                        "prompt_tokens_per_second": (
                            result["prompt_tokens_per_second"]
                        ),
                        "output_tokens_per_second": (
                            result["output_tokens_per_second"]
                        ),
                        "done": result["done"],
                    },
                )

            except OperationCancelled:
                # Cancellation is not a prompt failure: it must not be recorded
                # as a result, and it stops the run rather than continuing.
                raise

            except Exception as error:
                duration = (
                    time.perf_counter() - started_at
                    if started_at is not None
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

    except OperationCancelled as error:
        _cleanup_cancelled_test(
            model=model,
            name=name,
            completed=len(results),
            total=len(prompts),
            reason=str(error),
        )
        results.clear()
        raise

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


def _cleanup_cancelled_test(
    model: str,
    name: str,
    completed: int,
    total: int,
    reason: str,
) -> None:
    """Undo a cancelled benchmark's side effects and record the cancellation.

    A benchmark writes nothing to disk, so its only side effect is the model it
    loaded into memory to run the prompts. Unloading it frees the VRAM the run
    was holding, which leaves the machine as it was before the run started; the
    log entry is the only thing that remains.

    Args:
        model: Model the run had loaded.
        name: Test label.
        completed: Prompts that had finished before the cancellation.
        total: Prompts the run was going to execute.
        reason: Why the run was cancelled.
    """
    unloaded = False
    unload_error: str | None = None

    try:
        model_api.stop_model(model)
        unloaded = True
    except Exception as error:
        # The model may not have loaded yet, or Ollama may already be gone;
        # neither is a reason to fail the cancellation.
        unload_error = str(error)

    log_cancelled(
        component=COMPONENT,
        action="test",
        message="Test cancelled",
        details={
            "name": name,
            "model": model,
            "prompts_completed": completed,
            "prompts_total": total,
            "partial_results_discarded": completed,
            "model_unloaded": unloaded,
            "unload_error": unload_error,
            "reason": reason,
        },
    )


def compare_tests(
    model: str,
    prompts: list[str],
    configurations: list[dict],
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
) -> dict:
    """Run the same prompts against multiple temporary model configurations.

    Args:
        model: Ollama model name.
        prompts: List of evaluation prompt strings.
        configurations: List of configuration dictionaries with 'name' and
            'options'.
        include_output: Whether to include generated output in test results.
            Defaults to False.
        cancellation: Optional token that stops the comparison part-way.

    Returns:
        dict: Aggregated comparison results across all configurations.

    Raises:
        ValueError: If model, prompts, or configurations are empty.
        TypeError: If configurations or prompt items are invalid types.
        OperationCancelled: If the token is cancelled. Results collected so far
            are discarded, so a cancelled comparison leaves nothing behind but
            its log entry.
    """
    if not model.strip():
        raise ValueError("Model name is required")

    if not prompts:
        raise ValueError("At least one prompt is required")

    if not configurations:
        raise ValueError("At least one configuration is required")

    if not isinstance(include_output, bool):
        raise TypeError("include_output must be a boolean")

    # Shared with every `run_test` below, so one cancellation stops the whole
    # comparison rather than only the configuration that was running.
    token = cancellation if cancellation is not None else CancellationToken()

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
        token.raise_if_cancelled()

        try:
            result = run_test(
                model=model,
                prompts=prompts,
                config=configuration["options"],
                name=configuration["name"],
                include_output=include_output,
                cancellation=token,
            )
        except OperationCancelled as error:
            # run_test has already unloaded the model and logged its own
            # cancellation; discard the finished configurations so no partial
            # comparison survives, and record what the comparison as a whole
            # lost.
            log_cancelled(
                component=COMPONENT,
                action="compare",
                message="Tests cancelled",
                details={
                    "model": model,
                    "configurations_completed": len(tests),
                    "configurations_total": len(normalized_configurations),
                    "partial_results_discarded": len(tests),
                    "reason": str(error),
                },
            )
            tests.clear()
            raise

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
        dict: Summary statistics including average durations, rates, and token
            counts.
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
