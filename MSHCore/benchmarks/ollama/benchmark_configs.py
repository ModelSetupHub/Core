"""Benchmark Mode 'configs' — compare configurations of one model.

With one model and several configurations built, the plan is the
configuration test. This module is that mode: one model answers the same
prompts under each temporary configuration, and the result ranks the
configurations by their own run-to-run noise.

The configurations are applied per request and create nothing on disk; the
matrix execution itself — streaming, cancellation, averaging, the
significance verdicts — is the shared engine's
(:func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`).
"""

from collections.abc import Callable

from MSHCore.benchmarks.ollama import ollama_benchmark
from MSHCore.cancellation import CancellationToken


def benchmark_configs(
    model: str,
    configurations: list[dict],
    prompts: list[str],
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
    repetitions: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Compare several temporary configurations of one model.

    Each configuration is a dictionary carrying an optional 'name'
    (defaulting to its position) and an 'options' dict of generation
    parameters (temperature, num_ctx, …). Every configuration answers the
    same ``prompts``, which is what makes their numbers comparable: a gap
    between two configurations then describes the configurations, never the
    questions.

    Args:
        model: The Ollama model name or tag every configuration runs under.
        configurations: One dict per configuration, each with optional
            'name' and 'options'.
        prompts: Prompt strings every configuration answers.
        include_output: Whether to keep generated text in results.
        cancellation: Optional token that stops the run part-way.
        repetitions: How many times every prompt runs per configuration,
            from 1. Higher values turn the significance verdict real-or-noise
            from null into true/false.
        on_progress: Optional progress callback; see
            :func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`.

    Returns:
        dict: The comparison result — 'experiments', 'models', 'tests' (one
        entry per configuration) and 'significance'. The within-model
        verdict under ``significance['by_model'][model]`` is the ranking
        this mode exists to produce.

    Raises:
        ValueError: If the model name is empty, there are no configurations
            or no prompts.
        TypeError: If a configuration is malformed or carries its own
            'prompts' key.
        OperationCancelled: If the token is cancelled.
    """
    return ollama_benchmark.run_benchmark(
        experiments=[{"model": model, "configurations": configurations}],
        shared_prompts=prompts,
        include_output=include_output,
        cancellation=cancellation,
        repetitions=repetitions,
        on_progress=on_progress,
    )
