"""Ollama process lifecycle and service runtime management."""

import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from MSHCore.logging import write_log

COMPONENT = "ollama/runtime"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/tags"

START_TIMEOUT = 15.0
STOP_TIMEOUT = 10.0
CHECK_INTERVAL = 0.25

# Every file Ollama writes in its log directory, current and rotated alike.
LOG_FILE_GLOB = "*.log"


def _is_installed() -> bool:
    """Check whether the Ollama binary is available on the system PATH.

    Returns:
        bool: True if 'ollama' executable exists, False otherwise.
    """
    return shutil.which("ollama") is not None


def _is_running() -> bool:
    """Check whether the Ollama local HTTP API is responding.

    Returns:
        bool: True if Ollama service API returns HTTP 200, False otherwise.
    """
    try:
        with urllib.request.urlopen(
            OLLAMA_API_URL,
            timeout=2,
        ) as response:
            return response.status == 200
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ):
        return False


def _wait_for_running(
    expected: bool,
    timeout: float,
) -> bool:
    """Wait until Ollama reaches the target running state.

    Args:
        expected: Target boolean state (True for running, False for stopped).
        timeout: Maximum duration to wait in seconds.

    Returns:
        bool: True if the target state was achieved within the timeout.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if _is_running() == expected:
            return True
        time.sleep(CHECK_INTERVAL)

    return _is_running() == expected


def _get_version() -> str | None:
    """Query and parse the installed Ollama version string.

    Returns:
        str | None: Cleaned version string, or None if detection fails.
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            # This process may be an MCP server whose stdin carries the JSON-RPC
            # stream, and a child inheriting it could read the protocol's bytes.
            stdin=subprocess.DEVNULL,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return None

    output = result.stdout + result.stderr

    for line in output.splitlines():
        if "client version is" in line:
            return line.split("client version is", 1)[1].strip()

        if line.startswith("ollama version is"):
            return line.split("ollama version is", 1)[1].strip()

    return None


def get_status() -> dict:
    """Get the current Ollama installation and runtime health status.

    Returns:
        dict: Status dict containing 'installed', 'running', and 'version'.
    """
    installed = _is_installed()

    if not installed:
        return {
            "installed": False,
            "running": False,
            "version": None,
        }

    running = _is_running()
    version = _get_version()

    status = {
        "installed": True,
        "running": running,
        "version": version,
    }

    write_log(
        level="INFO",
        component=COMPONENT,
        action="status",
        message="Ollama status checked",
        details=status,
    )

    return status


def _collect_log_files() -> tuple[list[Path], dict[str, Path]]:
    """Locate Ollama's log directories and index the log files they hold.

    Ollama does not expose its log location, so the documented per-platform
    directories are probed instead: ``%LOCALAPPDATA%\\Ollama`` on Windows and
    ``~/.ollama/logs`` elsewhere. Both are probed on every platform because a
    manually placed installation can use either, and only directories that
    exist are reported.

    File names are unique within the index: when the same name appears in more
    than one directory, the first directory in search order wins.

    Returns:
        tuple[list[Path], dict[str, Path]]: Existing log directories, most
        likely first, and each log file name mapped to its full path, most
        recently modified first.

    Raises:
        RuntimeError: If no Ollama log directory exists on this machine.
    """
    candidates: list[Path] = []

    local_appdata = os.environ.get("LOCALAPPDATA")

    if local_appdata:
        candidates.append(Path(local_appdata) / "Ollama")

    candidates.append(Path.home() / ".ollama" / "logs")

    if os.name != "nt":
        candidates.append(Path("/var/log/ollama"))

    directories: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue

        if resolved in seen or not resolved.is_dir():
            continue

        seen.add(resolved)
        directories.append(resolved)

    if not directories:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="logs",
            message="No Ollama log directory found",
        )
        raise RuntimeError(
            "No Ollama log directory was found. Ollama logs to "
            "%LOCALAPPDATA%\\Ollama on Windows and ~/.ollama/logs "
            "elsewhere; neither exists, so Ollama has most likely never run "
            "on this machine."
        )

    found: dict[str, Path] = {}

    for directory in directories:
        for log_file in directory.glob(LOG_FILE_GLOB):
            if log_file.is_file() and log_file.name not in found:
                found[log_file.name] = log_file

    # Newest first, so the live app.log and server.log lead and the rotated
    # copies follow in the order they were retired.
    ordered = sorted(
        found.items(),
        key=lambda item: item[1].stat().st_mtime,
        reverse=True,
    )

    return directories, dict(ordered)


def list_ollama_logs() -> dict:
    """List the Ollama log files available on this machine.

    Names the ``*.log`` files Ollama keeps — the live ``app.log`` and
    ``server.log`` plus the rotated ``app-N.log`` and ``server-N.log`` copies —
    with the size and line count of each, returning no log content itself.
    Those two measures are what make the choice informed: ``server.log`` alone
    routinely runs past a megabyte, so a large file is worth reading through
    the line range of ``read_ollama_logs`` rather than whole, and the line
    count is the bound to aim that range at. These are Ollama's own logs,
    unrelated to this project's execution log that
    ``MSHCore.logging.read_logs`` serves.

    Returns:
        dict: Mapping with 'directories' searched, 'files' as a list of dicts
        ('name', 'path', 'size_bytes', 'line_count', 'modified') ordered most
        recently modified first, and 'total_bytes' summing every file.

    Raises:
        RuntimeError: If no Ollama log directory exists on this machine.
        OSError: If a listed log file cannot be read to count its lines.
    """
    directories, log_files = _collect_log_files()

    files = []

    for name, path in log_files.items():
        stats = path.stat()

        # Streamed rather than read whole, and counted the same way
        # read_ollama_logs numbers its lines, so a count reported here is the
        # bound a caller can pass straight back as end_line.
        with path.open(encoding="utf-8", errors="replace") as handle:
            line_count = sum(1 for _ in handle)

        files.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": stats.st_size,
                "line_count": line_count,
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(stats.st_mtime),
                ),
            }
        )

    searched = [str(directory) for directory in directories]
    total_bytes = sum(entry["size_bytes"] for entry in files)

    write_log(
        level="INFO",
        component=COMPONENT,
        action="list_logs",
        message="Ollama log files listed",
        details={
            "directories": searched,
            "file_count": len(files),
            "total_bytes": total_bytes,
        },
    )

    return {
        "directories": searched,
        "files": files,
        "total_bytes": total_bytes,
    }


def read_ollama_logs(
    file_name: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict:
    """Read one Ollama log file, whole or over a range of lines.

    The file is chosen by name from those ``list_ollama_logs`` reports; call
    that first. Only a bare file name is accepted, so a path cannot be used to
    reach outside Ollama's own log directories. Lines are numbered from 1 and
    the range is inclusive on both ends, so ``start_line=100, end_line=200``
    returns those 101 lines. Left at their defaults the whole file is returned.
    A range that starts past the end of the file yields empty content rather
    than an error, which is what makes 'total_lines' in the result worth
    checking when paging.

    Args:
        file_name: Log file name, for example 'server.log' or 'app-2.log'.
        start_line: First line to return, 1-based. Defaults to the first line.
        end_line: Last line to return, inclusive. Defaults to the final line.

    Returns:
        dict: Mapping with 'name', 'path', 'size_bytes', 'modified',
        'total_lines' counting the whole file, 'start_line' and 'end_line'
        bounding the lines actually returned (both None when the range matched
        nothing), and 'content' holding those lines.

    Raises:
        ValueError: If the name is empty or contains a path separator, or the
            line range is not a positive, non-descending pair.
        FileNotFoundError: If no log file of that name exists.
        RuntimeError: If no Ollama log directory exists on this machine.
        OSError: If the file exists but cannot be read.
    """
    requested = file_name.strip()

    if not requested:
        raise ValueError("A log file name is required")

    # A name is all that is accepted: anything path-shaped would let a caller
    # read a file outside the log directories this function is scoped to.
    if requested != Path(requested).name:
        raise ValueError(
            f"Expected a log file name, not a path: '{file_name}'. Call "
            f"list_ollama_logs for the available names."
        )

    if start_line < 1:
        raise ValueError(f"start_line must be 1 or greater, got {start_line}")

    if end_line is not None and end_line < start_line:
        raise ValueError(
            f"end_line ({end_line}) must not be before "
            f"start_line ({start_line})"
        )

    _, log_files = _collect_log_files()
    log_file = log_files.get(requested)

    if log_file is None:
        available = ", ".join(log_files) or "none"
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="read_logs",
            message="Ollama log file not found",
            details={
                "requested": requested,
                "available": list(log_files),
            },
        )
        raise FileNotFoundError(
            f"No Ollama log file named '{requested}' (available: {available})"
        )

    stats = log_file.stat()
    selected: list[str] = []
    total_lines = 0

    # Streamed a line at a time: the range exists so that a megabyte-sized
    # server.log need not be held in memory to read a hundred lines out of it.
    # The iteration continues past the range only to finish counting the lines,
    # which is what tells a caller paging through the file where it ends.
    with log_file.open(encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            total_lines = number

            if number < start_line:
                continue

            if end_line is not None and number > end_line:
                continue

            selected.append(line)

    first_returned = start_line if selected else None
    last_returned = start_line + len(selected) - 1 if selected else None

    write_log(
        level="INFO",
        component=COMPONENT,
        action="read_logs",
        message="Ollama log file read",
        details={
            "name": requested,
            "path": str(log_file),
            "size_bytes": stats.st_size,
            "total_lines": total_lines,
            "start_line": first_returned,
            "end_line": last_returned,
        },
    )

    return {
        "name": requested,
        "path": str(log_file),
        "size_bytes": stats.st_size,
        "modified": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(stats.st_mtime),
        ),
        "total_lines": total_lines,
        "start_line": first_returned,
        "end_line": last_returned,
        "content": "".join(selected),
    }


def install(
    installer_path: str,
) -> None:
    """Execute an Ollama standalone installer executable.

    Args:
        installer_path: Path to the installer file on disk.

    Raises:
        FileNotFoundError: If the installer binary is not found.
        RuntimeError: If the installer exits with a non-zero status.
        Exception: If running the installer fails.
    """
    installer = Path(installer_path).expanduser().resolve()

    if not installer.is_file():
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="install",
            message="Ollama installer not found",
            details={
                "path": str(installer),
            },
        )
        raise FileNotFoundError(f"Installer not found: {installer}")

    write_log(
        level="INFO",
        component=COMPONENT,
        action="install",
        message="Ollama installation started",
        details={
            "installer": str(installer),
        },
    )

    try:
        result = subprocess.run(
            [str(installer)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            # An installer that decides to prompt must not read this process's
            # stdin: under an MCP server that stream is the JSON-RPC transport.
            stdin=subprocess.DEVNULL,
        )
    except Exception as error:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="install",
            message="Ollama installation failed",
            details={
                "error": str(error),
            },
        )
        raise

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="install",
            message="Ollama installation failed",
            details={
                "returncode": result.returncode,
                "error": result.stderr.strip(),
            },
        )
        raise RuntimeError(
            result.stderr.strip()
            or f"Installer exited with code {result.returncode}"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="install",
        message="Ollama installation completed",
    )


def start(
    timeout: float = START_TIMEOUT,
) -> None:
    """Start Ollama background server process and wait until API becomes ready.

    Args:
        timeout: Maximum seconds to wait for API readiness. Defaults to 15.0.

    Raises:
        RuntimeError: If Ollama is not installed or fails to reach ready state.
    """
    if not _is_installed():
        raise RuntimeError("Ollama is not installed")

    if _is_running():
        write_log(
            level="INFO",
            component=COMPONENT,
            action="start",
            message="Ollama is already running",
        )
        return

    write_log(
        level="INFO",
        component=COMPONENT,
        action="start",
        message="Starting Ollama",
    )

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt"
                else 0
            ),
        )
    except Exception as error:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="start",
            message="Failed to start Ollama",
            details={
                "error": str(error),
            },
        )
        raise

    if not _wait_for_running(
        expected=True,
        timeout=timeout,
    ):
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="start",
            message="Ollama failed to start",
        )
        raise RuntimeError(f"Ollama did not start within {timeout} seconds")

    write_log(
        level="INFO",
        component=COMPONENT,
        action="start",
        message="Ollama started successfully",
    )


def stop(
    timeout: float = STOP_TIMEOUT,
) -> None:
    """Terminate the Ollama process and wait until API stops responding.

    Args:
        timeout: Maximum seconds to wait for shutdown. Defaults to 10.0.

    Raises:
        RuntimeError: If termination fails or the process does not stop in
            time.
    """
    if not _is_installed():
        write_log(
            level="WARNING",
            component=COMPONENT,
            action="stop",
            message="Ollama is not installed",
        )
        return

    if not _is_running():
        write_log(
            level="INFO",
            component=COMPONENT,
            action="stop",
            message="Ollama is already stopped",
        )
        return

    write_log(
        level="INFO",
        component=COMPONENT,
        action="stop",
        message="Stopping Ollama",
    )

    try:
        if os.name == "nt":
            result = subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/IM",
                    "ollama.exe",
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            result = subprocess.run(
                [
                    "pkill",
                    "-TERM",
                    "-x",
                    "ollama",
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"Failed to stop Ollama (exit code: {result.returncode})"
            )

    except Exception as error:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="stop",
            message="Failed to stop Ollama",
            details={
                "error": str(error),
            },
        )
        raise

    if not _wait_for_running(
        expected=False,
        timeout=timeout,
    ):
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="stop",
            message="Ollama failed to stop",
        )
        raise RuntimeError(f"Ollama did not stop within {timeout} seconds")

    write_log(
        level="INFO",
        component=COMPONENT,
        action="stop",
        message="Ollama stopped successfully",
    )
