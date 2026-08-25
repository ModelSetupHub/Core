from pathlib import Path
from datetime import datetime
import json


def get_execution_log_path():
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

    log_file = get_execution_log_path()

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
                continue

            if level and log_data["level"] != level:
                continue

            if component and log_data["component"] != component:
                continue

            if action and log_data["action"] != action:
                continue

            results.append(log_data)

    return results