from pathlib import Path
from datetime import datetime
import json


def get_execution_log_path():
    """
    Returns the execution log file path.

    The log file is stored in the repository data folder:

    Core/
    ├── core/
    │   └── logging.py
    │
    └── data/
        └── executions.log

    Creates the data folder if it does not exist.
    """

    # logging.py is inside core/, so move one level up
    # to reach the repository root
    repo_root = Path(__file__).resolve().parent.parent

    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    return data_dir / "executions.log"


def write_log(
    level: str,
    component: str,
    action: str,
    message: str,
    details: dict | None = None
):
    """
    Append a new execution event to the log file.

    Args:
        level:
            Log severity.
            Examples: INFO, WARNING, ERROR

        component:
            Part of the system that created the event.
            Examples: runtime, download, mcp

        action:
            Operation being executed.
            Examples: install, start, download

        message:
            Human-readable description of the event.

        details:
            Additional structured data related to the event.
            Stored as JSON.
    """

    if details is None:
        details = {}

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Each log entry is stored in one line.
    # This keeps the file easy to read manually
    # and simple to parse later.
    entry = (
        f"{timestamp} | "
        f"{level} | "
        f"{component} | "
        f"{action} | "
        f"{message} | "
        f"{json.dumps(details, ensure_ascii=False)}\n"
    )

    log_file = get_execution_log_path()

    # Append mode keeps previous execution history.
    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(entry)


def read_logs(
    level: str | None = None,
    component: str | None = None,
    action: str | None = None
):
    """
    Read execution logs with optional filters.

    Filters are optional and can be combined.

    Examples:
        read_logs(level="ERROR")

        read_logs(
            component="runtime",
            action="install"
        )

    Returns:
        List of log entries as dictionaries.
    """

    log_file = get_execution_log_path()

    if not log_file.exists():
        return []

    results = []

    with open(
        log_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            parts = line.strip().split(" | ")

            # Ignore invalid log lines
            if len(parts) != 6:
                continue

            try:
                log_data = {
                    "timestamp": parts[0],
                    "level": parts[1],
                    "component": parts[2],
                    "action": parts[3],
                    "message": parts[4],
                    "details": json.loads(parts[5])
                }

            except json.JSONDecodeError:
                # Skip corrupted log entries
                continue


            # Apply filters only when provided
            if level and log_data["level"] != level:
                continue

            if component and log_data["component"] != component:
                continue

            if action and log_data["action"] != action:
                continue

            results.append(log_data)

    return results