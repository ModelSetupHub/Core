"""Model benchmarking — per-runtime mode packages and the shared result store.

Each model runtime owns a subpackage of benchmark modes; Ollama's four live
under ``ollama``. A future runtime adds its own subpackage beside ``ollama``
and shares the same store:

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
