"""Compatibility module for the benchmark engine's former home.

The benchmark engine lives in :mod:`MSHCore.ollama.benchmark.experiment`;
this module re-exports its public interface so every existing import path
keeps working unchanged::

    from MSHCore.ollama import experiment
    experiment.compare_tests(...)

New code may use either path; the two names are the same objects.
"""

from MSHCore.ollama.benchmark.experiment import (  # noqa: F401
    COMPONENT,
    compare_models,
    compare_tests,
    run_test,
)

__all__ = [
    "COMPONENT",
    "compare_models",
    "compare_tests",
    "run_test",
]
