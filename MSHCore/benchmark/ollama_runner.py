"""Ollama model benchmarking and multi-configuration experimental testing."""

import json
import threading
import time
import urllib.error
import urllib.request

from MSHCore.cancellation import (
    POLL_INTERVAL,
    CancellationToken,
    OperationCancelled,
    log_cancelled,
)
from MSHCore.logging import write_log
from MSHCore.ollama import model as model_api
from MSHCore.system.hardware import get_gpu_thermal, get_vram_used

COMPONENT = "benchmark/ollama_runner"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"

# Per-run metrics averaged across a prompt's repetitions, and the subset whose
# spread (standard deviation, minimum, maximum) is reported alongside them.
# Every entry is optional on a run: ttft_seconds is absent when a generation
# produced no content, and the GPU figures when the machine has no NVIDIA GPU.
NUMERIC_METRICS = (
    "duration_seconds",
    "prompt_tokens",
    "output_tokens",
    "prompt_tokens_per_second",
    "output_tokens_per_second",
    "ttft_seconds",
    "vram_used_mb",
    "gpu_temperature_c",
    "gpu_clock_mhz",
)

SPREAD_METRICS = (
    "duration_seconds",
    "prompt_tokens_per_second",
    "output_tokens_per_second",
    "ttft_seconds",
    "gpu_temperature_c",
)


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
            generated text collected under 'response' and the time-to-first-
            token under 'ttft_seconds' (None when nothing was generated).

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

    # The time-to-first-token clock starts when the request is sent: the delay
    # a caller experiences covers the connection, the prompt's processing and
    # the first token's generation alike.
    request_started = time.perf_counter()

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
    ttft: float | None = None

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
                if ttft is None:
                    # First content of the answer: this is the latency a
                    # reader of a streamed response actually perceives.
                    ttft = time.perf_counter() - request_started

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
    # A generation that produced no content has no first token to time.
    final["ttft_seconds"] = ttft

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
    repetitions: int = 1,
) -> dict:
    """Run one temporary model configuration against multiple prompts.

    Each prompt runs ``repetitions`` times and the per-prompt result averages
    those runs, with the spread of the timing and rate metrics reported along
    the means. The default of one repetition reproduces exactly the behaviour
    this function had before averaging existed.

    Alongside the timing metrics, measurements are reported as taken: the time
    to the answer's first streamed token, the VRAM the driver reported right
    after the generation finished, and the hottest GPU's temperature and clock
    in that same moment. None is judged here — the numbers are offered to the
    caller, whose decision it is what they mean for the machine being
    benchmarked.

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
        repetitions: How many times every prompt is executed, from 1. The
            first repetition doubles as the warmup the following ones benefit
            from; results average them all.

    Returns:
        dict: Test execution results and summary statistics.

    Raises:
        ValueError: If model name is empty, prompts list is empty, or
            repetitions is not 1 or greater.
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

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")

    if repetitions < 1:
        raise ValueError(f"repetitions must be 1 or greater, got {repetitions}")

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

            # Successful repetitions only: a failed run contributes its error
            # but no timing, so averaging never mixes numbers with failures.
            successful: list[dict] = []
            errors: list[str] = []

            for repetition in range(1, repetitions + 1):
                token.raise_if_cancelled()

                # Reset per run: a failure before the timer starts must not
                # report the previous run's start time.
                started_at: float | None = None

                try:
                    # Ensure the model is loaded before the test.
                    # This operation is intentionally outside the
                    # benchmark timer and is not included in results.
                    # The check is cheap once the model is already loaded, so
                    # it runs before every repetition.
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

                    run_result = {
                        "index": index,
                        "success": True,
                        "prompt": prompt,
                        "duration_seconds": duration,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "prompt_tokens_per_second": prompt_tokens_per_second,
                        "output_tokens_per_second": output_tokens_per_second,
                        "ttft_seconds": response.get("ttft_seconds"),
                        "done": response.get("done", True),
                    }

                    # One snapshot right after the generation: the numbers
                    # describe the GPU as the driver saw it at that moment,
                    # model and everything else on the machine included. They
                    # are reported as measured — judging them is the caller's
                    # part.
                    vram_readings = get_vram_used()

                    if vram_readings:
                        run_result["vram_used_mb"] = max(vram_readings)

                    thermal_readings = get_gpu_thermal()

                    if thermal_readings:
                        # The hottest GPU is the one whose throttle, if any,
                        # shaped this run's tail; its figures are the ones
                        # worth carrying.
                        hottest = max(
                            thermal_readings,
                            key=lambda reading: (
                                reading.get("temperature_c") is not None,
                                reading.get("temperature_c") or 0,
                            ),
                        )

                        if hottest.get("temperature_c") is not None:
                            run_result["gpu_temperature_c"] = hottest[
                                "temperature_c"
                            ]

                        if hottest.get("sm_clock_mhz") is not None:
                            run_result["gpu_clock_mhz"] = hottest[
                                "sm_clock_mhz"
                            ]

                    if include_output:
                        run_result["response"] = response.get("response", "")

                    successful.append(run_result)

                    write_log(
                        level="INFO",
                        component=COMPONENT,
                        action="test",
                        message="Prompt executed",
                        details={
                            "name": name,
                            "prompt_index": index,
                            "repetition": repetition,
                            "success": run_result["success"],
                            "duration_seconds": run_result["duration_seconds"],
                            "prompt_tokens": run_result["prompt_tokens"],
                            "output_tokens": run_result["output_tokens"],
                            "prompt_tokens_per_second": (
                                run_result["prompt_tokens_per_second"]
                            ),
                            "output_tokens_per_second": (
                                run_result["output_tokens_per_second"]
                            ),
                            "ttft_seconds": run_result["ttft_seconds"],
                            "vram_used_mb": run_result.get("vram_used_mb"),
                            "gpu_temperature_c": run_result.get(
                                "gpu_temperature_c"
                            ),
                            "gpu_clock_mhz": run_result.get("gpu_clock_mhz"),
                            "done": run_result["done"],
                        },
                    )

                except OperationCancelled:
                    # Cancellation is not a prompt failure: it must not be
                    # recorded as a result, and it stops the run rather than
                    # continuing.
                    raise

                except Exception as error:
                    duration = (
                        time.perf_counter() - started_at
                        if started_at is not None
                        else 0.0
                    )

                    errors.append(str(error))

                    write_log(
                        level="ERROR",
                        component=COMPONENT,
                        action="test",
                        message="Prompt execution failed",
                        details={
                            "name": name,
                            "prompt_index": index,
                            "repetition": repetition,
                            "success": False,
                            "duration_seconds": duration,
                            "error": str(error),
                        },
                    )

            if successful:
                result = _summarize_repetitions(successful)
                result.update({
                    "index": index,
                    "success": True,
                    "prompt": prompt,
                })

                if include_output:
                    # The generated text of the last successful run stands for
                    # the prompt: every repetition sees the same seed unless the
                    # configuration asks otherwise, and the text is what a
                    # reader wants to eyeball rather than a per-run list.
                    result["response"] = successful[-1].get("response", "")

                if errors:
                    # Some repetitions failed while others succeeded: the
                    # averages stand on the successful runs alone, and the
                    # count says how much of the work did not report.
                    result["failed_repetitions"] = len(errors)
            else:
                # Every repetition failed: the prompt reports the last error,
                # which is the one a caller retrying would face again.
                result = {
                    "index": index,
                    "success": False,
                    "prompt": prompt,
                    "duration_seconds": 0.0,
                    "error": errors[-1] if errors else "Unknown error",
                }

                if repetitions > 1:
                    result["error_count"] = len(errors)

            results.append(result)

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
    repetitions: int = 1,
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
        repetitions: How many times every prompt runs per configuration, from
            1. Averaging runs the same way :func:`run_test` averages them, so
            every configuration's numbers carry a stddev a caller can compare
            differences against.

    Returns:
        dict: Aggregated comparison results across all configurations.

    Raises:
        ValueError: If model, prompts, or configurations are empty, or
            repetitions is not 1 or greater.
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

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")

    if repetitions < 1:
        raise ValueError(f"repetitions must be 1 or greater, got {repetitions}")

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
                repetitions=repetitions,
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
        "significance": _assess_significance(tests),
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


def _unload_model(model: str) -> bool:
    """Unload one model, treating a failure as a note rather than a stop.

    Args:
        model: Model name to unload.

    Returns:
        bool: True when the model was unloaded, False when Ollama refused or
        the model was not there to stop. Either way the caller carries on:
        the next model's load proceeds regardless, and its timings describe
        the machine as it actually was.
    """
    try:
        model_api.stop_model(model)
        return True
    except Exception as error:
        write_log(
            level="WARNING",
            component=COMPONENT,
            action="unload",
            message="Model unload failed between comparison steps",
            details={
                "model": model,
                "error": str(error),
            },
        )
        return False


def compare_models(
    models: list[str],
    prompts: list[str],
    config: dict | None = None,
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
    repetitions: int = 1,
) -> dict:
    """Run the same prompts and one shared configuration against several models.

    Models are benchmarked one after another, and the model measured before is
    unloaded before the next one loads: two models resident at once would
    compete for the same VRAM and make every timing in both meaningless. A
    model that was already loaded before the comparison began is not the
    comparison's to stop — only the models this call loaded are unloaded, the
    one just measured before the next loads and the last one when the run is
    over, so the machine is left as it was found.

    One configuration applies to every model, because a fair comparison
    changes one thing at a time; several configurations per model means one
    :func:`compare_tests` call per configuration instead.

    Args:
        models: Model names or tags to benchmark, in run order.
        prompts: Prompt strings every model answers.
        config: Optional generation parameters shared by every model.
        include_output: Whether to keep generated text in results. Defaults
            to False.
        cancellation: Optional token that stops the comparison part-way.
        repetitions: How many times every prompt runs per model, from 1.

    Returns:
        dict: Aggregated comparison across models. 'tests' holds one run_test
        result per model — its timings, noise spread and VRAM readings — with
        the model's name as the test name, and 'significance' judges the gap
        between the two fastest models against their own noise, exactly as
        :func:`compare_tests` judges configurations.

    Raises:
        ValueError: If models or prompts are empty, a model name is not a
            non-empty string, or repetitions is not 1 or greater.
        TypeError: If config is not a dictionary or repetitions is not an
            integer.
        OperationCancelled: If the token is cancelled. Results collected so
            far are discarded — a cancelled comparison leaves nothing behind
            but its log entry, and no model in memory.
    """
    if not isinstance(models, list) or not models:
        raise ValueError("At least one model is required")

    for position, model_name in enumerate(models, start=1):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"Model {position} must be a non-empty string")

    if not prompts:
        raise ValueError("At least one prompt is required")

    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary or None")

    if not isinstance(include_output, bool):
        raise TypeError("include_output must be a boolean")

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")

    if repetitions < 1:
        raise ValueError(f"repetitions must be 1 or greater, got {repetitions}")

    token = cancellation if cancellation is not None else CancellationToken()

    configuration = dict(config or {})

    write_log(
        level="INFO",
        component=COMPONENT,
        action="compare_models",
        message="Model comparison started",
        details={
            "models": models,
            "prompts": prompts,
            "config": configuration,
            "repetitions": repetitions,
        },
    )

    tests = []
    loaded: str | None = None

    try:
        for model_name in models:
            token.raise_if_cancelled()

            if loaded is not None:
                _unload_model(loaded)

            try:
                result = run_test(
                    model=model_name,
                    prompts=prompts,
                    config=configuration,
                    name=model_name,
                    include_output=include_output,
                    cancellation=token,
                    repetitions=repetitions,
                )
            except OperationCancelled as error:
                # run_test has already unloaded the model it had loaded; the
                # one before it was unloaded on the way out. Discard the
                # finished models so no partial comparison survives, and
                # record what the comparison as a whole lost.
                log_cancelled(
                    component=COMPONENT,
                    action="compare_models",
                    message="Model comparison cancelled",
                    details={
                        "models": models,
                        "models_completed": len(tests),
                        "models_total": len(models),
                        "partial_results_discarded": len(tests),
                        "reason": str(error),
                    },
                )
                tests.clear()
                raise

            tests.append(result)
            loaded = model_name

    finally:
        # run_test unloads its own model when a cancellation ends it; on every
        # other exit the last model measured is unloaded here, so the
        # comparison leaves no model in memory whatever its outcome.
        if loaded is not None:
            _unload_model(loaded)

    result = {
        "models": models,
        "config": configuration,
        "repetitions": repetitions,
        "tests": tests,
        "significance": _assess_significance(tests),
    }

    write_log(
        level="INFO",
        component=COMPONENT,
        action="compare_models",
        message="Model comparison completed",
        details={
            "models": models,
        },
    )

    return result


# How many combined standard deviations two configurations' averages must be
# apart before their difference counts as real. Two overlapping spreads are
# noise; one clear gap between them is a finding. Deliberately conservative —
# claiming a difference that does not exist sends a user tuning a parameter
# that never mattered.
SIGNIFICANCE_THRESHOLD = 2.0


def _assess_significance(tests: list[dict]) -> dict | None:
    """Compare the leading configurations' averages against their own noise.

    A comparison is only worth ranking when the top two are actually apart:
    two averages separated by less than their combined standard deviations
    describe the same speed, and naming a winner between them would be
    inventing a fact. The metric judged here is output generation rate, the
    figure the verdict turns on.

    With a single repetition there is no spread to argue from — the comparison
    then reports itself as unmeasured rather than pretending to a verdict. A
    configuration whose prompts all failed has no average and takes itself out
    of the running; the ranking stands among the configurations that have one.

    Args:
        tests: One run_test result per configuration, in run order.

    Returns:
        dict | None: The assessment, or None when fewer than two
        configurations produced a measurable average.
    """
    ranked = []

    for test in tests:
        rate = test["summary"].get("average_output_tokens_per_second")

        if rate is not None:
            ranked.append((
                test["name"],
                rate,
                test["summary"].get("output_tokens_per_second_stddev"),
            ))

    if len(ranked) < 2:
        return None

    ranked.sort(key=lambda item: item[1], reverse=True)

    leader_name, leader_rate, leader_spread = ranked[0]
    runner_name, runner_rate, runner_spread = ranked[1]

    spreads = [
        spread
        for spread in (leader_spread, runner_spread)
        if spread is not None
    ]

    gap = leader_rate - runner_rate

    if not spreads:
        # No repetition data anywhere: a difference is visible but nothing
        # measures whether it is real.
        return {
            "metric": "output_tokens_per_second",
            "leader": leader_name,
            "runner_up": runner_name,
            "difference": gap,
            "significant": None,
            "message": (
                "Measured once per prompt, so noise cannot be told apart "
                "from a real difference. Run again with repetitions set "
                "higher to learn whether this gap is real."
            ),
        }

    combined_spread = (sum(spread ** 2 for spread in spreads) / len(spreads)) ** 0.5

    # A zero spread on both sides makes the ratio infinite, which is correct:
    # two perfectly repeatable measurements a hair apart really do differ.
    ratio = gap / combined_spread if combined_spread > 0 else None

    if ratio is None:
        significant = True
        description = (
            f"{leader_name} is faster by {gap:.3g} tokens/s, and both "
            f"measurements repeated exactly — the difference is real."
        )
    elif ratio >= SIGNIFICANCE_THRESHOLD:
        significant = True
        description = (
            f"{leader_name} is faster by {gap:.3g} tokens/s "
            f"({ratio:.1f}x the noise level) — the difference is real."
        )
    else:
        significant = False
        description = (
            f"{leader_name} and {runner_name} are within noise of each other "
            f"(gap {gap:.3g} tokens/s is {ratio:.1f}x the noise level). "
            f"Either configuration is fine; pick on other grounds."
        )

    return {
        "metric": "output_tokens_per_second",
        "leader": leader_name,
        "runner_up": runner_name,
        "difference": gap,
        "noise_level": combined_spread,
        "difference_to_noise_ratio": ratio,
        "significant": significant,
        "message": description,
    }


def _build_summary(results: list[dict]) -> dict:
    """Build aggregate metrics from successful test results.

    The output-rate noise level combines every prompt's own run-to-run spread
    into one pooled figure. It exists for the comparison verdict: how far two
    configurations' averages sit apart only means something against the noise
    both of them carry. A single-repetition test has no spread to pool and
    reports None, so a caller never mistakes "measured once" for "perfectly
    repeatable".

    Args:
        results: List of successful test result dictionaries.

    Returns:
        dict: Summary statistics including average durations, rates, and token
            counts, plus the output-rate noise level when repetitions were run.
    """
    if not results:
        return {
            "average_duration_seconds": None,
            "average_prompt_tokens_per_second": None,
            "average_output_tokens_per_second": None,
            "total_output_tokens": 0,
            "output_tokens_per_second_stddev": None,
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

    # Only prompts that actually repeated carry a spread; averaging their
    # variances is the pooled estimate of the run-to-run noise the whole test
    # was subject to. Prompts measured once contribute nothing and say nothing.
    spreads = [
        result["output_tokens_per_second_stddev"]
        for result in results
        if result.get("repetitions", 1) > 1
        and result.get("output_tokens_per_second_stddev") is not None
    ]

    output_noise = (
        (sum(spread ** 2 for spread in spreads) / len(spreads)) ** 0.5
        if spreads
        else None
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
        "output_tokens_per_second_stddev": output_noise,
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


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None for an empty sample.

    Args:
        values: Sample of one measured metric across repetitions.

    Returns:
        float | None: The mean, or None when nothing was measured.
    """
    if not values:
        return None

    return sum(values) / len(values)


def _stddev(values: list[float]) -> float | None:
    """Return the population standard deviation of a sample.

    A single measurement has no spread to speak of, so it reports zero rather
    than None: callers can compare variability across prompts without special-
    casing the one-repetition runs that are the default.

    Args:
        values: Sample of one measured metric across repetitions.

    Returns:
        float | None: The standard deviation, or None for an empty sample.
    """
    if not values:
        return None

    if len(values) == 1:
        return 0.0

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2 for value in values
    ) / len(values)

    return variance ** 0.5


def _summarize_repetitions(
    repetitions: list[dict],
) -> dict | None:
    """Collapse one prompt's repetitions into averaged metrics with spread.

    Only successful repetitions carry timings, so an all-failed prompt has
    nothing to average and reports None — the caller then shows the error
    instead of invented numbers.

    Args:
        repetitions: Successful run results for one prompt, in execution order.

    Returns:
        dict | None: Mean for every numeric metric plus standard deviation,
        minimum and maximum for the timing and rate metrics, and the repetition
        count; or None when the list is empty.
    """
    if not repetitions:
        return None

    summary: dict = {"repetitions": len(repetitions)}

    for metric in NUMERIC_METRICS:
        values = [
            repetition[metric]
            for repetition in repetitions
            if repetition.get(metric) is not None
        ]

        summary[metric] = _mean(values)

        if metric in SPREAD_METRICS:
            summary[f"{metric}_stddev"] = _stddev(values)
            summary[f"{metric}_min"] = min(values) if values else None
            summary[f"{metric}_max"] = max(values) if values else None

    summary["done"] = all(
        repetition.get("done", True) for repetition in repetitions
    )

    return summary
