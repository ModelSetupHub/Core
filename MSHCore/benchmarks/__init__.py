"""Model benchmarking — per-runtime mode packages and the shared result store.

Each model runtime owns a subpackage of Benchmark Modes; Ollama's four live
under ``ollama``:

- ``ollama.benchmark_single`` — one model under its defaults: a plain speed test.
- ``ollama.benchmark_configs`` — one model under several configurations.
- ``ollama.benchmark_models`` — several models under one shared configuration.
- ``ollama.benchmark_tournament`` — several models, each racing its own configuration.
- ``ollama.ollama_benchmark`` — the matrix engine every Ollama mode runs on.

A future runtime adds its own subpackage beside ``ollama`` and shares the
same store:

- ``history`` — persisted benchmark runs under the toolkit's data root,
  written by every mode through one runner-agnostic place.
"""

from MSHCore.benchmarks.ollama import (
    benchmark_configs,
    benchmark_models,
    benchmark_single,
    benchmark_tournament,
)

__all__ = [
    "benchmark_configs",
    "benchmark_models",
    "benchmark_single",
    "benchmark_tournament",
]
