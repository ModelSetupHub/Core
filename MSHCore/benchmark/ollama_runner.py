"""Ollama model benchmarking and multi-configuration experimental testing."""

from collections.abc import Callable
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


def _emit_progress(
    on_progress: Callable[[dict], None],
    phase: str,
    prompt_index: int,
    prompt_count: int,
    repetition: int,
    repetition_count: int,
    completed: int,
) -> None:
    """Hand one progress step to a run's callback, absorbing its failures.

    A progress callback exists so a caller can show the user where a long run
    is; a bug in it must not be able to end a benchmark that has already been
    running for minutes. Anything it raises is logged and the run continues.

    Args:
        on_progress: The caller's callback.
        phase: Either 'prompt_start' (a prompt is about to run, no repetition
            has) or 'repetition_done' (one repetition just finished).
        prompt_index: 1-based prompt position within the test.
        prompt_count: Prompts the test will run.
        repetition: 1-based repetition that just finished, or 0 on
            'prompt_start'.
        repetition_count: Repetitions each prompt runs.
        completed: Repetitions finished so far in this test.
    """
    try:
        on_progress({
            "phase": phase,
            "prompt_index": prompt_index,
            "prompt_count": prompt_count,
            "repetition": repetition,
            "repetition_count": repetition_count,
            "completed": completed,
        })
    except Exception as error:
        write_log(
            level="WARNING",
            component=COMPONENT,
            action="progress",
            message="Progress callback failed",
            details={
                "phase": phase,
                "prompt_index": prompt_index,
                "repetition": repetition,
                "error": str(error),
            },
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
    on_progress: Callable[[dict], None] | None = None,
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
        on_progress: Optional callable called once with the configuration's
            identity before the first prompt runs, and once after every
            individual repetition finishes with the step the run is at:
            which prompt, which repetition, how many of both, and how many
            steps have completed. Progress is best-effort — an exception from
            the callback is logged and dropped rather than failing the run.

    Returns:
        dict: Test execution results and summary statistics, including the
            repetition count every prompt was averaged over.

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

            if on_progress is not None:
                _emit_progress(
                    on_progress,
                    phase="prompt_start",
                    prompt_index=index,
                    prompt_count=len(prompts),
                    repetition=0,
                    repetition_count=repetitions,
                    completed=0,
                )

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

                if on_progress is not None:
                    completed = len(successful) + len(errors)
                    _emit_progress(
                        on_progress,
                        phase="repetition_done",
                        prompt_index=index,
                        prompt_count=len(prompts),
                        repetition=repetition,
                        repetition_count=repetitions,
                        completed=completed,
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
        "repetitions": repetitions,
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


def run_benchmark(
    experiments: list[dict],
    shared_prompts: list[str] | None = None,
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
    repetitions: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Benchmark a matrix of models, configurations and prompts.

    Each experiment names one model and the temporary configurations it runs
    under, and every configuration answers ``shared_prompts`` — one that
    carries prompts of its own answers those in addition. Every leaf of the
    matrix is one :func:`run_test`, so every model-configuration pair is
    measured the same way: prompts averaged over ``repetitions`` runs, with
    the run-to-run spread reported beside the means.

    The two comparisons this module used to offer as separate functions —
    one model under several configurations, and several models under one —
    are plain shapes of this matrix, built by passing the experiments that
    name them.

    Models are benchmarked one after another, and the model measured before
    is unloaded before the next one loads: two models resident at once would
    compete for the same VRAM and make every timing in both meaningless. The
    configurations of one model run back to back while it stays loaded —
    unloading between them would only add the same load cost to each. A
    model that was already loaded before the comparison began is not the
    comparison's to stop — only the models this call loaded are unloaded, so
    the machine is left as it was found.

    Args:
        experiments: One dictionary per model, in run order. 'model' is the
            Ollama model name or tag, and 'configurations' — optional — is a
            list of dictionaries, defaulting to a single default
            configuration. Each configuration carries an optional 'name'
            (defaulting to its position), optional 'options' (the generation
            parameters, defaulting to empty) and optional 'prompts'
            (defaulting to ``shared_prompts``).
        shared_prompts: The prompts every configuration runs, before any
            prompts of its own. Optional when every configuration carries
            its own.
        include_output: Whether to keep generated text in results. Defaults
            to False.
        cancellation: Optional token that stops the comparison part-way.
        repetitions: How many times every prompt runs per configuration,
            from 1.
        on_progress: Optional callable receiving one progress dict per step
            of the comparison: which model and which configuration (name and
            position of each), which prompt and repetition, how many of both,
            and how many steps the whole comparison has completed and will
            run in total. See :func:`run_test` for the delivery and failure
            guarantees.

    Returns:
        dict: 'experiments' echoes the normalized matrix that ran, 'models'
        lists the model names in run order, and 'tests' holds one run_test
        result per model-configuration pair — its timings, noise spread and
        VRAM readings. 'significance' judges the matrix the two ways it can
        fairly be judged: 'by_model' assesses each model's configurations
        against one another, and 'across_models' assesses the models
        themselves, each represented by its fastest configuration — both by
        reading the gap between the top two averages against their own
        run-to-run noise.

    Raises:
        ValueError: If experiments is empty, an experiment names no model, a
            configuration ends up with no prompts, or repetitions is not 1 or
            greater.
        TypeError: If experiments, a configuration, prompts or repetitions
            have the wrong types.
        OperationCancelled: If the token is cancelled. Results collected so
            far are discarded — a cancelled comparison leaves nothing behind
            but its log entry, and no model in memory.
    """
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("At least one experiment is required")

    if shared_prompts is not None and not isinstance(
        shared_prompts, (list, tuple)
    ):
        raise TypeError("shared_prompts must be a list of strings or None")

    if not isinstance(include_output, bool):
        raise TypeError("include_output must be a boolean")

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")

    if repetitions < 1:
        raise ValueError(f"repetitions must be 1 or greater, got {repetitions}")

    # One token for every run_test below, so one cancellation stops the whole
    # comparison rather than only the experiment that was running.
    token = cancellation if cancellation is not None else CancellationToken()

    normalized_experiments = []

    for experiment_index, experiment in enumerate(experiments, start=1):
        if not isinstance(experiment, dict):
            raise TypeError(
                f"Experiment {experiment_index} must be a dictionary"
            )

        model = experiment.get("model")

        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"Experiment {experiment_index} must name a model"
            )

        raw_configurations = experiment.get("configurations")

        if raw_configurations is None:
            # A model with no configurations of its own runs once, under its
            # defaults, under the name a reader would recognize it by.
            raw_configurations = [{"name": "default", "options": {}}]

        if not isinstance(raw_configurations, list) or not raw_configurations:
            raise ValueError(
                f"Experiment '{model}' must carry at least one configuration"
            )

        configurations = []

        for configuration_index, configuration in enumerate(
            raw_configurations, start=1
        ):
            if not isinstance(configuration, dict):
                raise TypeError(
                    f"Configuration {configuration_index} of '{model}' must "
                    f"be a dictionary"
                )

            name = configuration.get(
                "name", f"configuration_{configuration_index}"
            )
            options = configuration.get("options", {})

            if not isinstance(options, dict):
                raise TypeError(
                    f"Configuration '{name}' options must be a dictionary"
                )

            own_prompts = configuration.get("prompts")

            if own_prompts is not None and not isinstance(
                own_prompts, (list, tuple)
            ):
                raise TypeError(
                    f"Configuration '{name}' prompts must be a list of "
                    f"strings"
                )

            # The shared prompts run for every configuration; a
            # configuration with its own adds those after them, so the
            # common baseline every configuration answers stays aligned
            # at the front of each run.
            prompts = list(shared_prompts or []) + list(own_prompts or [])

            if not prompts:
                raise ValueError(
                    f"Configuration '{name}' has no prompts to run: it "
                    f"carries none of its own and the benchmark was given "
                    f"no shared_prompts"
                )

            configurations.append({
                "name": name,
                "options": dict(options),
                "prompts": prompts,
            })

        normalized_experiments.append({
            "model": model,
            "configurations": configurations,
        })

    write_log(
        level="INFO",
        component=COMPONENT,
        action="compare",
        message="Comparison started",
        details={
            "experiments": normalized_experiments,
            "repetitions": repetitions,
        },
    )

    total_steps = sum(
        len(configuration["prompts"]) * repetitions
        for experiment in normalized_experiments
        for configuration in experiment["configurations"]
    )

    tests = []
    loaded: str | None = None
    steps_before = 0

    try:
        for model_position, experiment in enumerate(
            normalized_experiments, start=1
        ):
            token.raise_if_cancelled()

            if loaded is not None and loaded != experiment["model"]:
                _unload_model(loaded)

            for configuration_position, configuration in enumerate(
                experiment["configurations"], start=1
            ):
                token.raise_if_cancelled()

                # The comparison's steps are its leaves' steps, prefixed with
                # which model and configuration each belongs to, so one
                # counter covers the whole comparison for a caller drawing a
                # single progress bar. The loop variables are bound as
                # defaults, so each closure keeps the step it was made for.
                def _comparison_progress(
                    step: dict,
                    _model=experiment["model"],
                    _model_position=model_position,
                    _model_count=len(normalized_experiments),
                    _configuration=configuration["name"],
                    _configuration_position=configuration_position,
                    _configuration_count=len(experiment["configurations"]),
                    _steps_before=steps_before,
                ) -> None:
                    on_progress({
                        **step,
                        "model": _model,
                        "model_index": _model_position,
                        "model_count": _model_count,
                        "configuration": _configuration,
                        "configuration_index": _configuration_position,
                        "configuration_count": _configuration_count,
                        "completed": _steps_before + step.get("completed", 0),
                        "total": total_steps,
                    })

                try:
                    result = run_test(
                        model=experiment["model"],
                        prompts=configuration["prompts"],
                        config=configuration["options"],
                        name=configuration["name"],
                        include_output=include_output,
                        cancellation=token,
                        repetitions=repetitions,
                        on_progress=(
                            _comparison_progress
                            if on_progress is not None
                            else None
                        ),
                    )
                except OperationCancelled as error:
                    # run_test has already unloaded the model it had loaded
                    # and logged its own cancellation; discard the finished
                    # pairs so no partial comparison survives, and record
                    # what the comparison as a whole lost.
                    log_cancelled(
                        component=COMPONENT,
                        action="compare",
                        message="Comparison cancelled",
                        details={
                            "models": [
                                entry["model"]
                                for entry in normalized_experiments
                            ],
                            "experiments_completed": model_position - 1,
                            "experiments_total": len(normalized_experiments),
                            "partial_results_discarded": len(tests),
                            "reason": str(error),
                        },
                    )
                    tests.clear()
                    raise

                tests.append(result)
                steps_before += len(configuration["prompts"]) * repetitions

            loaded = experiment["model"]

    finally:
        # run_test unloads its own model when a cancellation ends it; on every
        # other exit the last model measured is unloaded here, so the
        # comparison leaves no model in memory whatever its outcome.
        if loaded is not None:
            _unload_model(loaded)

    result = {
        "experiments": normalized_experiments,
        "models": [
            experiment["model"] for experiment in normalized_experiments
        ],
        "tests": tests,
        "significance": _assess_comparison(
            tests,
            [experiment["model"] for experiment in normalized_experiments],
        ),
    }

    write_log(
        level="INFO",
        component=COMPONENT,
        action="compare",
        message="Comparison completed",
        details={
            "models": result["models"],
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


def _assess_comparison(tests: list[dict], models: list[str]) -> dict:
    """Judge a comparison's matrix per model and across models.

    A matrix holds two comparison questions at once — which of a model's
    configurations is faster, and which model is — and one flat ranking
    cannot answer both: the top two entries of the whole pile are usually
    configurations of different models, which answers neither question. So
    the matrix is judged the two ways it can fairly be: within each model,
    across that model's configurations; and across models, each represented
    by its fastest configuration.

    A model with fewer than two measurable averages cannot support a
    within-model verdict and carries None there; fewer than two measurable
    models and the cross-model verdict is None too.

    Args:
        tests: One run_test result per model-configuration pair, in run
            order.
        models: The model names the comparison ran, in run order.

    Returns:
        dict: 'by_model' maps each model's name to its within-model
        assessment (or None when it cannot support one), and 'across_models'
        holds the cross-model assessment or None.
    """
    by_model = {}

    for model in models:
        if model in by_model:
            # A model named twice runs twice; one verdict covers its tests.
            continue

        by_model[model] = _assess_significance(
            [test for test in tests if test.get("model") == model]
        )

    representatives = []

    for model in by_model:
        candidates = [
            test
            for test in tests
            if test.get("model") == model
            and test.get("summary", {}).get(
                "average_output_tokens_per_second"
            )
            is not None
        ]

        if candidates:
            fastest = max(
                candidates,
                key=lambda test: test["summary"][
                    "average_output_tokens_per_second"
                ],
            )

            # The cross-model verdict speaks of models, so each
            # representative stands in under its model's name rather than
            # its configuration's.
            representatives.append(dict(fastest, name=model))

    return {
        "by_model": by_model,
        "across_models": (
            _assess_significance(representatives)
            if len(representatives) >= 2
            else None
        ),
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
