"""Ollama Benchmark Modes — the four shapes of one matrix, plus its engine.

Each model runtime owns a subpackage of benchmark modes; Ollama's four live
under ``ollama``, each in its own module with a function of the same name:

- ``benchmark_single`` — one model under its defaults: a plain speed test.
- ``benchmark_configs`` — one model under several configurations.
- ``benchmark_models`` — several models under one shared configuration.
- ``benchmark_tournament`` — several models, each racing its own configuration.

Shared underneath them all is ``ollama_benchmark``, the matrix engine every
mode runs on. The persisted run history lives one level up, in
``MSHCore.benchmarks.history`` — it is runner-agnostic, so a future runtime's
modes save there too.
"""

from MSHCore.benchmarks.ollama.benchmark_configs import benchmark_configs
from MSHCore.benchmarks.ollama.benchmark_models import benchmark_models
from MSHCore.benchmarks.ollama.benchmark_single import benchmark_single
from MSHCore.benchmarks.ollama.benchmark_tournament import benchmark_tournament

__all__ = [
    "benchmark_configs",
    "benchmark_models",
    "benchmark_single",
    "benchmark_tournament",
]
