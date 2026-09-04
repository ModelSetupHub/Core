"""Benchmark history package with saving, listing and reading past runs."""

from .history import (
    BenchmarkNotFoundError,
    delete,
    list_saved,
    load,
    save,
)

__all__ = [
    "BenchmarkNotFoundError",
    "delete",
    "list_saved",
    "load",
    "save",
]
