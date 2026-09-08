"""Benchmark Mode 'models' — compare several models under one configuration.

With several models and no more than one configuration, the plan is the
model comparison. This module is that mode: every model answers the same
prompts under the same shared configuration, each model unloaded before the
next loads so their timings never compete for VRAM.

The matrix execution itself — streaming, cancellation, averaging, the
significance verdicts — is the shared engine's
(:func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`).
"""

from collections.abc import Callable

from MSHCore.benchmarks.ollama import ollama_benchmark
from MSHCore.cancellation import CancellationToken


def benchmark_models(
    models: list[str],
    prompts: list[str],
    config: dict | None = None,
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
    repetitions: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Compare several models over one shared configuration.

    Every model runs under the same ``config`` options — or Ollama's defaults
    when it is empty — and answers the same ``prompts``, so a gap between two
    models' numbers describes the models. Each model is unloaded before the
    next loads; a model already loaded before this call is left resident.

    Args:
        models: Model names to compare, in run order.
        prompts: Prompt strings every model answers.
        config: Generation parameters shared by every model. Optional.
        include_output: Whether to keep generated text in results.
        cancellation: Optional token that stops the run part-way.
        repetitions: How many times every prompt runs per model, from 1.
        on_progress: Optional progress callback; see
            :func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`.

    Returns:
        dict: The comparison result — 'experiments', 'models', 'tests' (one
        entry per model) and 'significance'. The cross-model verdict under
        ``significance['across_models']`` is the ranking this mode exists to
        produce.

    Raises:
        ValueError: If a model name is empty or there are no prompts.
        OperationCancelled: If the token is cancelled.
    """
    options = dict(config or {})

    return ollama_benchmark.run_benchmark(
        experiments=[
            {
                "model": model,
                "configurations": [{"name": model, "options": dict(options)}],
            }
            for model in models
        ],
        shared_prompts=prompts,
        include_output=include_output,
        cancellation=cancellation,
        repetitions=repetitions,
        on_progress=on_progress,
    )
