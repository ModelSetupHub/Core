"""Benchmark Mode 'tournament' — each model races its own configuration.

The WebApp's Benchmark tab derives its test kind from what the user has set
up; with several models ticked and several configurations built, the plan is
the tournament (i18n key ``bench.plan.tournament``, "Tournament"): each model
races under the configuration paired with it rather than a single shared
one. This module is that mode.

The pairing arrives as ``model_configs`` — a dictionary keyed by model name,
exactly the payload the WebApp sends. A model named there runs under its own
options; a model missing from it falls back to the shared ``config``, which
reproduces the WebApp's positional fallback (an unpaired model takes the
configuration at its own position). The matrix execution itself — streaming,
cancellation, averaging, the significance verdicts — is the shared engine's
(:func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`).
"""

from collections.abc import Callable

from MSHCore.benchmarks.ollama import ollama_benchmark
from MSHCore.cancellation import CancellationToken


def benchmark_tournament(
    models: list[str],
    prompts: list[str],
    model_configs: dict[str, dict] | None = None,
    config: dict | None = None,
    include_output: bool = False,
    cancellation: CancellationToken | None = None,
    repetitions: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run a tournament: each model under the configuration paired with it.

    One experiment per model, in run order; the configuration a model races
    under is its own when ``model_configs`` names one, the shared ``config``
    otherwise. Every model still answers the same ``prompts`` — the questions
    stay constant across the field even though the settings do not.

    Args:
        models: Model names racing, in run order.
        prompts: Prompt strings every model answers.
        model_configs: Per-model overrides keyed by model name; a model
            named here runs under those options alone. Optional.
        config: Shared fallback options for models ``model_configs`` does
            not name. Optional.
        include_output: Whether to keep generated text in results.
        cancellation: Optional token that stops the run part-way.
        repetitions: How many times every prompt runs per model, from 1.
        on_progress: Optional progress callback; see
            :func:`MSHCore.benchmarks.ollama.ollama_benchmark.run_benchmark`.

    Returns:
        dict: The comparison result — 'experiments', 'models', 'tests' (one
        entry per model-configuration pair) and 'significance', whose
        'by_model' half judges each model's configurations and whose
        'across_models' half ranks the models against each other.

    Raises:
        ValueError: If a model name is empty or there are no prompts.
        OperationCancelled: If the token is cancelled.
    """
    overrides = dict(model_configs or {})
    shared = dict(config or {})

    experiments = []

    for model in models:
        if model in overrides:
            options = dict(overrides[model])
        else:
            options = dict(shared)

        experiments.append({
            "model": model,
            "configurations": [{"name": model, "options": options}],
        })

    return ollama_benchmark.run_benchmark(
        experiments=experiments,
        shared_prompts=prompts,
        include_output=include_output,
        cancellation=cancellation,
        repetitions=repetitions,
        on_progress=on_progress,
    )
