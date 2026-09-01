"""Execution logging module for recording and querying system events."""

from datetime import datetime
import json
from pathlib import Path


def _log_file_path() -> Path:
    """Return the execution log's path, creating its directory when missing.

    Kept separate from ``get_log_file_info`` so that appending a single entry
    does not have to read the whole log: ``write_log`` runs on every logged
    event and needs nothing but the path.

    Returns:
        Path: Absolute path to the execution log file.
    """
    # logging.py lives in MSHCore/, so the repository root is one level up
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    return data_dir / "executions.log"


def get_log_file_info() -> dict:
    """Describe the execution log file: where it is and how much it holds.

    The log file is stored inside the repository's data directory:
        Core/
        ├── MSHCore/
        │   └── logging.py
        └── data/
            └── executions.log

    Creates the data directory if it does not already exist. A log that has
    never been written to is reported at zero lines and zero bytes rather than
    treated as an error, so this can be called before the first event.

    Returns:
        dict: Mapping with 'path' as the absolute ``Path`` to the log file,
        'line_count' holding the number of entries it contains, and
        'size_bytes' its size on disk.

    Raises:
        OSError: If the log file exists but cannot be read.
    """
    log_file = _log_file_path()

    if not log_file.exists():
        return {
            "path": log_file,
            "line_count": 0,
            "size_bytes": 0,
        }

    with open(log_file, "r", encoding="utf-8") as file:
        line_count = sum(1 for _ in file)

    return {
        "path": log_file,
        "line_count": line_count,
        "size_bytes": log_file.stat().st_size,
    }


def write_log(
    level: str,
    component: str,
    action: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Append a new execution event to the log file.

    Each log entry is stored as a single line separated by pipe delimiters
    with structured details encoded in JSON format.

    Args:
        level: Log severity (e.g., 'INFO', 'WARNING', 'ERROR').
        component: System component originating the event (e.g., 'runtime').
        action: Operation being performed (e.g., 'install', 'start').
        message: Human-readable description of the log event.
        details: Optional dictionary containing additional event metadata.
    """
    if details is None:
        details = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Each log entry is stored in a single line for simple parsing
    entry = (
        f"{timestamp} | "
        f"{level} | "
        f"{component} | "
        f"{action} | "
        f"{message} | "
        f"{json.dumps(details, ensure_ascii=False)}\n"
    )

    log_file = _log_file_path()

    with open(log_file, "a", encoding="utf-8") as file:
        file.write(entry)


def _parse_log_line(line: str) -> dict | None:
    """Parse one raw log line into its fields.

    Args:
        line: Raw line as read from the log file.

    Returns:
        dict | None: Parsed entry, or None if the line is malformed or its
        details are not valid JSON.
    """
    # Only the four leading fields are delimiter-free, so the split stops there
    # and the message and details are separated afterwards.
    parts = line.strip().split(" | ", 4)

    if len(parts) != 5:
        return None

    # The message and the JSON details may both contain " | ", so the boundary
    # between them is found by taking the longest trailing segment that parses
    # as JSON. Everything before it is the message.
    segments = parts[4].split(" | ")

    for position in range(1, len(segments)):
        try:
            details = json.loads(" | ".join(segments[position:]))
        except json.JSONDecodeError:
            continue

        return {
            "timestamp": parts[0],
            "level": parts[1],
            "component": parts[2],
            "action": parts[3],
            "message": " | ".join(segments[:position]),
            "details": details,
        }

    return None


def read_logs(
    level: str | None = None,
    component: str | None = None,
    action: str | None = None,
    line_count: int | None = None,
) -> list[dict]:
    """Read execution logs with optional filtering.

    Filters are optional and can be combined to narrow down the query. The
    three value filters select which entries match; ``line_count`` then caps
    how many of those matches come back, keeping the most recent ones, since
    the log is appended in chronological order and the newest entries are what
    a capped read is usually after. Call ``get_log_file_info`` for the total
    the log holds before deciding on a cap.

    Args:
        level: Optional log severity level to filter by (e.g., 'ERROR').
        component: Optional component name to filter by.
        action: Optional action name to filter by.
        line_count: Optional maximum number of entries to return, counted back
            from the newest match. Defaults to returning every match.

    Returns:
        list[dict]: List of parsed log entry dictionaries matching the
        criteria, oldest first.

    Raises:
        ValueError: If line_count is given but is not 1 or greater.
    """
    if line_count is not None and line_count < 1:
        raise ValueError(f"line_count must be 1 or greater, got {line_count}")

    log_file = _log_file_path()

    if not log_file.exists():
        return []

    results = []

    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            log_data = _parse_log_line(line)

            # Ignore invalid or corrupted log lines
            if log_data is None:
                continue

            # Apply filters only when provided
            if level and log_data["level"] != level:
                continue

            if component and log_data["component"] != component:
                continue

            if action and log_data["action"] != action:
                continue

            results.append(log_data)

            # Older matches beyond the cap are dropped as the file is read, so
            # a long log costs no more memory than the number of entries asked
            # for.
            if line_count is not None and len(results) > line_count:
                results.pop(0)

    return results
