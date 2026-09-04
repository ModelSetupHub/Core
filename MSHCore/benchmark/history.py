"""Persisted benchmark history — every saved comparison is one JSON file.

A comparison run is the unit of history: one model, the prompts every
configuration answered, and one result per configuration. The three belong
together — a configuration's numbers only mean something against the others
it was compared with — so a run is stored whole in a single file rather than
split per configuration or per model.

The store is runner-agnostic: it validates only the shape of what arrives —
what was benchmarked, and at least one test — and never talks to a model
runtime, so a runner for any model server saves here unchanged.

Layout under the toolkit's data root (see :mod:`MSHCore.paths`)::

    %LOCALAPPDATA%\\MSH\\
    └── benchmarks/
        ├── index.json                  one small record per saved run
        └── 20260905T211530_a1b2c3.json the full result of one run

The file name carries the save time, so a plain directory listing is already
chronological. Each file holds the comparison result exactly as the caller
produced it — per-prompt timings, noise spreads, the significance assessment —
under an added header identifying what was compared. Nothing is recomputed or
judged at save time; the store only records and serves what it was given.

The index exists so listing runs never opens the result files: one record per
run with the identity fields a list shows (model, time, winner, noise
verdict), kept in the same write as the result so the two cannot disagree.
A run dropped from the index is dropped from disk with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import uuid

from MSHCore.logging import write_log
from MSHCore.paths import app_data_directory, ensure_directory

COMPONENT = "benchmarks/history"

DIRECTORY_NAME = "benchmarks"
INDEX_NAME = "index.json"

# Runs kept on disk. Oldest beyond the cap are removed — with their index
# records — so a long-running machine does not accumulate files without
# bound. A benchmark result can be a few hundred kilobytes, so the cap keeps
# the directory comfortably small while holding months of typical use.
MAX_SAVED_RUNS = 100

# A benchmark id is minted here and used as a file name, so anything arriving
# from outside is checked before it touches the filesystem.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class BenchmarkNotFoundError(FileNotFoundError):
    """Raised when no benchmark with the requested identifier is stored."""


def directory() -> Path:
    """Return the benchmarks directory, creating it when missing.

    Returns:
        Path: ``%LOCALAPPDATA%\\MSH\\benchmarks`` (or ``~/.msh`` on a platform
        without that variable), now existing.
    """
    return ensure_directory(app_data_directory() / DIRECTORY_NAME)


def _index_path() -> Path:
    """Return the index file's path.

    Returns:
        Path: ``benchmarks/index.json`` beside the result files.
    """
    return directory() / INDEX_NAME


def _result_path(benchmark_id: str) -> Path:
    """Return the file a benchmark id is stored under.

    Args:
        benchmark_id: Identifier of a saved run.

    Returns:
        Path: The run's file inside the benchmarks directory.
    """
    return directory() / f"{benchmark_id}.json"


def _write_atomically(path: Path, payload) -> None:
    """Write JSON to a path through a temporary file and an atomic replace.

    A reader either sees the previous file or the new one, never a partial
    write left behind by a crash between the two.

    Args:
        path: Destination file.
        payload: JSON-serialisable value.

    Raises:
        OSError: If the write or the replace fails.
    """
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def _read_json(path: Path):
    """Parse a JSON file, tolerating absence and corruption alike.

    A result file or index that cannot be read is treated as absent rather
    than failing a listing: the store's job is to serve what survived, and a
    damaged record is not worth taking the whole history down for.

    Args:
        path: File to read.

    Returns:
        The parsed value, or None when the file is missing or unreadable.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_index() -> list[dict]:
    """Read the index records, newest first.

    Returns:
        list[dict]: Index records, or an empty list when there is none yet.
    """
    records = _read_json(_index_path())

    if not isinstance(records, list):
        return []

    return [record for record in records if isinstance(record, dict)]


def _write_index(records: list[dict]) -> None:
    """Replace the whole index with the given records, newest first.

    Args:
        records: Index records in the order they should be listed.
    """
    _write_atomically(_index_path(), records)


def _summarize_result(result: dict) -> dict | None:
    """Read the winner's figures a list needs out of a comparison result.

    Args:
        result: A compare_tests result as produced by the toolkit.

    Returns:
        dict | None: Per-configuration averages keyed by name, the fastest
        configuration's name, and the significance assessment when one was
        made; None when the result holds no measurable configuration.
    """
    averages = {}

    for test in result.get("tests", []):
        rate = test.get("summary", {}).get("average_output_tokens_per_second")

        if rate is not None:
            averages[test.get("name", "unnamed")] = rate

    if not averages:
        return None

    winner = max(averages, key=averages.get)
    significance = result.get("significance")

    return {
        "average_output_tokens_per_second": averages,
        "winner": winner,
        "significant": (
            significance.get("significant") if significance else None
        ),
    }


def save(result: dict) -> str:
    """Store a comparison result as one history entry.

    The result is kept exactly as given — tests, per-prompt rows, noise
    spreads, significance — under a header naming what was compared. A
    single-model comparison carries its model under 'model'; a cross-model
    one names what it measured under 'models', and both are accepted. Two
    runs saved in the same second are told apart by a short random suffix,
    so an identifier is never reused and overwriting an older run by
    accident is not something that can happen.

    Args:
        result: A compare_tests or compare_models result dictionary, carrying
            'tests' plus either 'model' or a 'models' list, and optionally
            'significance'.

    Returns:
        str: The identifier the run was saved under.

    Raises:
        TypeError: If result is not a dictionary.
        ValueError: If the result names no model (or models) or holds no test
            results.
        OSError: If the run cannot be written to disk.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary")

    model = result.get("model")
    models = result.get("models")

    if models is None and isinstance(model, str) and model.strip():
        models = [model]

    if (
        not isinstance(models, list)
        or not models
        or not all(
            isinstance(name, str) and name.strip() for name in models
        )
    ):
        raise ValueError(
            "result must carry what it benchmarked: a 'model' name or a "
            "non-empty 'models' list"
        )

    model = model if isinstance(model, str) and model.strip() else None

    tests = result.get("tests")

    if not isinstance(tests, list) or not tests:
        raise ValueError("result must carry at least one test result")

    saved_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    benchmark_id = (
        time.strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )

    record = {
        "id": benchmark_id,
        "saved_at": saved_at,
        "model": model,
        "models": models,
        "prompts": [
            prompt.get("prompt")
            for prompt in tests[0].get("results", [])
            if isinstance(prompt, dict)
        ],
        "configurations": [
            {
                "name": test.get("name"),
                "options": test.get("configuration", {}),
            }
            for test in tests
        ],
        "repetitions": tests[0].get("repetitions", 1),
        "summary": _summarize_result(result),
        "result": result,
    }

    _write_atomically(_result_path(benchmark_id), record)

    records = _read_index()
    records.insert(
        0,
        {
            "id": benchmark_id,
            "saved_at": saved_at,
            "model": model,
            "models": models,
            "repetitions": record["repetitions"],
            "configuration_count": len(record["configurations"]),
            "prompt_count": len(record["prompts"]),
            "winner": (
                record["summary"]["winner"]
                if record["summary"]
                else None
            ),
            "significant": (
                record["summary"]["significant"]
                if record["summary"]
                else None
            ),
        },
    )
    records = records[:MAX_SAVED_RUNS]
    _write_index(records)

    # The cap applies to files as well: results that fell off the index are
    # deleted, so disk and index always agree.
    kept_ids = {record["id"] for record in records}

    for path in sorted(directory().glob("*.json")):
        if path.name == INDEX_NAME:
            continue

        if path.stem not in kept_ids:
            try:
                path.unlink()
            except OSError:
                continue

    write_log(
        level="INFO",
        component=COMPONENT,
        action="save",
        message="Benchmark saved to history",
        details={
            "id": benchmark_id,
            "model": model,
            "models": models,
            "configuration_count": len(record["configurations"]),
            "prompt_count": len(record["prompts"]),
        },
    )

    return benchmark_id


def list_saved() -> list[dict]:
    """List the saved benchmark runs, newest first.

    The listing comes from the index, so it costs one small file however many
    runs are stored. A record whose result file has gone missing on its own —
    deleted outside this module, say — is left out rather than listed and
    then failing to open.

    Returns:
        list[dict]: One record per saved run, each carrying 'id',
        'saved_at', 'model', 'repetitions', 'configuration_count',
        'prompt_count', 'winner' and 'significant'.
    """
    records = _read_index()
    available = [
        path.stem
        for path in directory().glob("*.json")
        if path.name != INDEX_NAME
    ]

    listed = []

    for record in records:
        if record.get("id") in available:
            listed.append(record)

    return listed


def load(benchmark_id: str) -> dict:
    """Read one saved benchmark run in full.

    Args:
        benchmark_id: Identifier as returned by :func:`save` or listed by
            :func:`list_saved`.

    Returns:
        dict: The stored record — the header, the summary for lists, and the
        complete comparison result under 'result'.

    Raises:
        ValueError: If the identifier is not a plain id.
        BenchmarkNotFoundError: If no run is stored under that identifier.
    """
    if not benchmark_id or not _ID_PATTERN.match(benchmark_id):
        raise ValueError(f"Not a benchmark id: '{benchmark_id}'")

    record = _read_json(_result_path(benchmark_id))

    if record is None:
        write_log(
            level="WARNING",
            component=COMPONENT,
            action="load",
            message="Benchmark record not found",
            details={"id": benchmark_id},
        )
        raise BenchmarkNotFoundError(
            f"No benchmark is stored under '{benchmark_id}'."
        )

    return record


def delete(benchmark_id: str) -> bool:
    """Remove one saved benchmark run from the history.

    Args:
        benchmark_id: Identifier of the run to remove.

    Returns:
        bool: True when a run was removed, False when nothing was stored
        under that identifier — removing a missing run is the state the
        caller asked for, not a failure.

    Raises:
        ValueError: If the identifier is not a plain id.
    """
    if not benchmark_id or not _ID_PATTERN.match(benchmark_id):
        raise ValueError(f"Not a benchmark id: '{benchmark_id}'")

    removed = False
    path = _result_path(benchmark_id)

    if path.exists():
        try:
            path.unlink()
            removed = True
        except OSError:
            return False

    records = _read_index()
    remaining = [
        record for record in records if record.get("id") != benchmark_id
    ]

    if len(remaining) != len(records):
        removed = True

        try:
            _write_index(remaining)
        except OSError:
            return False

    if removed:
        write_log(
            level="INFO",
            component=COMPONENT,
            action="delete",
            message="Benchmark removed from history",
            details={"id": benchmark_id},
        )

    return removed
