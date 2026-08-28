"""Ollama process lifecycle and service runtime management."""

import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

try:
    import winreg
except ImportError:
    winreg = None

from core.cancellation import (
    CancellationToken,
    OperationCancelled,
    log_cancelled,
    run_cancellable,
)
from core.logging import write_log

COMPONENT = "ollama/runtime"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/tags"

START_TIMEOUT = 15.0
STOP_TIMEOUT = 10.0
CHECK_INTERVAL = 0.25

# Where a completed Ollama installation registers itself, read only to find the
# uninstaller when a cancelled installation has to be rolled back.
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Ollama"
UNINSTALL_TIMEOUT = 120.0


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


def install(
    installer_path: str,
    cancellation: CancellationToken | None = None,
) -> None:
    """Execute an Ollama standalone installer executable.

    Args:
        installer_path: Path to the installer file on disk.
        cancellation: Optional token that stops the installer part-way.

    Raises:
        FileNotFoundError: If the installer binary is not found.
        OperationCancelled: If the token is cancelled. The installer's process
            tree is terminated and, when it had already registered Ollama, that
            partial installation is removed, so a cancelled install leaves
            nothing behind but its log entry.
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

    # Recorded before the installer runs, so a cancellation can tell whether this
    # call is what put Ollama on the machine and therefore what to roll back.
    was_installed = _is_installed()

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
        result = run_cancellable(
            [str(installer)],
            cancellation=cancellation,
            component=COMPONENT,
            action="install",
            **_process_group_kwargs(),
        )
    except OperationCancelled as error:
        _cleanup_cancelled_install(
            installer=installer,
            was_installed=was_installed,
            reason=str(error),
        )
        raise
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


def _process_group_kwargs() -> dict:
    """Build the Popen arguments needed to terminate an installer's whole tree.

    An installer normally launches the real installation as a child process, so
    it has to be started in a way that lets the entire tree be signalled at once.

    Returns:
        dict: Platform-specific keyword arguments for ``subprocess.Popen``.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

    return {"start_new_session": True}


def _cleanup_cancelled_install(
    installer: Path,
    was_installed: bool,
    reason: str,
) -> None:
    """Remove what a cancelled Ollama installation left behind.

    The installer's process tree is already terminated by the time this runs. If
    Ollama was absent beforehand and the interrupted installer got far enough to
    register it, that half-finished installation is the leftover: the service is
    stopped and the registered uninstaller is run so the machine ends up as it
    was. An installation that was already there is left alone — it is not this
    operation's to remove.

    Args:
        installer: Installer that was interrupted.
        was_installed: Whether Ollama was present before this attempt.
        reason: Why the installation was cancelled.
    """
    partial_install = not was_installed and _is_installed()
    service_stopped = False
    removed = False
    cleanup_error: str | None = None

    if partial_install:
        try:
            if _is_running():
                stop()
                service_stopped = True
        except Exception as error:
            cleanup_error = str(error)

        uninstaller = _find_uninstaller()

        if uninstaller is not None:
            try:
                subprocess.run(
                    [str(uninstaller), "/S"],
                    capture_output=True,
                    check=False,
                    timeout=UNINSTALL_TIMEOUT,
                )
                removed = not _is_installed()
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_error = str(error)

    log_cancelled(
        component=COMPONENT,
        action="install",
        message="Ollama installation cancelled",
        details={
            "installer": str(installer),
            "was_installed_before": was_installed,
            "partial_install_detected": partial_install,
            "service_stopped": service_stopped,
            "partial_install_removed": removed,
            "cleanup_error": cleanup_error,
            "reason": reason,
        },
    )


def _find_uninstaller() -> Path | None:
    """Locate the uninstaller a partial Ollama installation registered.

    Returns:
        Path | None: Uninstaller executable, or None when none is registered, in
        which case there is nothing that can be removed safely.
    """
    if winreg is None:
        return None

    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, UNINSTALL_KEY) as key:
                value, _ = winreg.QueryValueEx(key, "UninstallString")
        except OSError:
            continue

        if not value:
            continue

        candidate = Path(str(value).strip().strip('"'))

        if candidate.is_file():
            return candidate

    return None


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
        RuntimeError: If termination command fails or process does not stop in time.
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

