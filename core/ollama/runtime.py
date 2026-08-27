import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from core.logging import write_log


COMPONENT = "ollama/runtime"

OLLAMA_API_URL = "http://127.0.0.1:11434/api/tags"

START_TIMEOUT = 15.0
STOP_TIMEOUT = 10.0
CHECK_INTERVAL = 0.25


def _is_installed() -> bool:
    """Check whether Ollama is installed."""
    return shutil.which("ollama") is not None


def _is_running() -> bool:
    """Check whether the Ollama API is running."""
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
    """Wait until Ollama reaches the expected running state."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if _is_running() == expected:
            return True

        time.sleep(CHECK_INTERVAL)

    return _is_running() == expected


def _get_version() -> str | None:
    """Get the installed Ollama version."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )

    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return None

    output = result.stdout + result.stderr

    for line in output.splitlines():
        if "client version is" in line:
            return line.split(
                "client version is",
                1,
            )[1].strip()

        if line.startswith("ollama version is"):
            return line.split(
                "ollama version is",
                1,
            )[1].strip()

    return None


def get_status() -> dict:
    """Get the current Ollama status."""

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


def install(installer_path: str) -> None:
    """Install Ollama from a local installer."""

    installer = (
        Path(installer_path)
        .expanduser()
        .resolve()
    )

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

        raise FileNotFoundError(
            f"Installer not found: {installer}"
        )

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
        subprocess.run(
            [str(installer)],
            check=True,
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

    write_log(
        level="INFO",
        component=COMPONENT,
        action="install",
        message="Ollama installation completed",
    )


def start(
    timeout: float = START_TIMEOUT,
) -> None:
    """Start Ollama and wait until the API is available."""

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

        raise RuntimeError(
            f"Ollama did not start within {timeout} seconds"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="start",
        message="Ollama started successfully",
    )


def stop(
    timeout: float = STOP_TIMEOUT,
) -> None:
    """Stop Ollama and wait until the API is unavailable."""

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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"Failed to stop Ollama "
                f"(exit code: {result.returncode})"
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

        raise RuntimeError(
            f"Ollama did not stop within {timeout} seconds"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="stop",
        message="Ollama stopped successfully",
    )
