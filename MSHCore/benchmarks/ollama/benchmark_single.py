"""Benchmark Mode 'single' — measure one model's speed.

With exactly one model and no more than one configuration, the plan is the
plain speed test. This module is that mode: the model answers every prompt
once under its own defaults, and the result carries the timings, rates and
GPU readings of that single run.

Everything hard about a benchmark — streaming, cancellation, averaging, the
significance verdicts — lives in the shared engine
(:func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`); this file
only names the matrix shape the mode stands for.
"""

from collections.abc import Callable

from MSHCore.benchmarks.ollama import ollama_benchmark
from MSHCore.cancellation import CancellationToken


def benchmark_single(
    model: str,
    prompts: list[str],
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
    repetitions: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Measure one model's raw speed over a list of prompts.

    The model runs once, under Ollama's own defaults — no configurations are
    compared, so there is nothing to rank against. Every prompt is answered
    ``repetitions`` times and averaged, with the run-to-run spread reported
    beside the means.

    Args:
        model: The Ollama model name or tag to measure.
        prompts: Prompt strings the model answers.
        include_output: Whether to keep generated text in results.
        cancellation: Optional token that stops the run part-way.
        repetitions: How many times every prompt runs, from 1.
        on_progress: Optional progress callback; see
            :func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`.

    Returns:
        dict: The comparison result for the single model-configuration pair
        — 'experiments', 'models', 'tests' (one entry) and 'significance'.
        A caller showing one test's numbers unwraps ``tests[0]`` itself.

    Raises:
        ValueError: If the model name is empty or there are no prompts.
        OperationCancelled: If the token is cancelled.
    """
    return ollama_benchmark.run_benchmark(
        experiments=[{"model": model}],
        shared_prompts=prompts,
        include_output=include_output,
        cancellation=cancellation,
        repetitions=repetitions,
        on_progress=on_progress,
    )
