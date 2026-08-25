from pathlib import Path
from datetime import datetime
import json


def get_log_file():
    """
    Returns the execution log path.
    """

    repo_root = Path(__file__).resolve().parent.parent

    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    return data_dir / "executions.log"


def log(
    level: str,
    component: str,
    action: str,
    message: str,
    details: dict | None = None
):
    """
    Append a log entry.
    """

    if details is None:
        details = {}

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    entry = (
        f"{timestamp} | "
        f"{level} | "
        f"{component} | "
        f"{action} | "
        f"{message} | "
        f"{json.dumps(details, ensure_ascii=False)}\n"
    )

    log_file = get_log_file()

    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(entry)