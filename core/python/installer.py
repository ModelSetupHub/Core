"""Python system installer management and Windows registry detection."""

import os
from pathlib import Path
import subprocess
import sys

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

COMPONENT = "python"

# The Windows Python installer removes an installation it created when it is
# re-run with /uninstall, which is how a cancelled install is rolled back.
UNINSTALL_TIMEOUT = 300.0


def install_python(
    installer_path: str,
    all_users: bool = False,
    cancellation: CancellationToken | None = None,
) -> list[dict]:
    """Install Python from a local Windows installer executable in quiet mode.

    Args:
        installer_path: Path to the Python installer executable.
        all_users: Whether to install Python for all system users. Defaults to False.
        cancellation: Optional token that stops the installer part-way.

    Returns:
        list[dict]: List of detected Python versions and executable paths.

    Raises:
        FileNotFoundError: If the specified installer file does not exist.
        RuntimeError: If the installation process fails.
        OperationCancelled: If the token is cancelled. The installer's process
            tree is terminated and any interpreter this call had registered is
            uninstalled, so a cancelled install leaves nothing behind but its log
            entry.
    """
    installer = Path(installer_path).expanduser().resolve()

    if not installer.is_file():
        raise FileNotFoundError(
            f"Python installer not found: {installer}"
        )

    command = [
        str(installer),
        "/quiet",
        "PrependPath=1",
        "Include_test=0",
    ]

    if all_users:
        command.append("InstallAllUsers=1")
    else:
        command.append("InstallAllUsers=0")

    # Recorded before the installer runs, so a cancellation can tell which
    # interpreters this call added and therefore which ones to roll back.
    installed_before = {item["path"] for item in get_python_status()}

    try:
        result = run_cancellable(
            command,
            cancellation=cancellation,
            component=COMPONENT,
            action="install_python",
            **_process_group_kwargs(),
        )
    except OperationCancelled as error:
        _cleanup_cancelled_install(
            installer=installer,
            command=command,
            installed_before=installed_before,
            reason=str(error),
        )
        raise

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="install_python",
            message="Failed to install Python",
            details={
                "installer": str(installer),
                "all_users": all_users,
                "error": result.stderr.strip(),
            },
        )
        raise RuntimeError(
            result.stderr.strip() or "Failed to install Python"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="install_python",
        message="Python installed successfully",
        details={
            "installer": str(installer),
            "all_users": all_users,
        },
    )

    return get_python_status()


def _process_group_kwargs() -> dict:
    """Build the Popen arguments needed to terminate an installer's whole tree.

    The installer launches the actual installation as a child process, so it has
    to start in its own group for the whole tree to be stoppable at once.

    Returns:
        dict: Platform-specific keyword arguments for ``subprocess.Popen``.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

    return {"start_new_session": True}


def _cleanup_cancelled_install(
    installer: Path,
    command: list[str],
    installed_before: set[str],
    reason: str,
) -> None:
    """Remove what a cancelled Python installation left behind.

    The installer's process tree is already terminated by the time this runs. If
    it got far enough to register a new interpreter, that half-finished
    installation is the leftover, and the same installer removes it. Interpreters
    that were already present are left untouched — they are not this operation's
    to remove.

    Args:
        installer: Installer that was interrupted.
        command: Command line the interrupted installer was run with.
        installed_before: Interpreter paths present before this attempt.
        reason: Why the installation was cancelled.
    """
    added = [
        item["path"]
        for item in get_python_status()
        if item["path"] not in installed_before
    ]

    removed = False
    cleanup_error: str | None = None

    if added:
        try:
            subprocess.run(
                [str(installer), "/uninstall", "/quiet"],
                capture_output=True,
                check=False,
                timeout=UNINSTALL_TIMEOUT,
            )
            still_present = {item["path"] for item in get_python_status()}
            removed = not any(path in still_present for path in added)
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_error = str(error)

    log_cancelled(
        component=COMPONENT,
        action="install_python",
        message="Python installation cancelled",
        details={
            "installer": str(installer),
            "command": command,
            "partial_install_detected": bool(added),
            "interpreters_added": added,
            "partial_install_removed": removed,
            "cleanup_error": cleanup_error,
            "reason": reason,
        },
    )


def get_python_status() -> list[dict]:
    """Get detected installed Python versions from system and Windows registry.

    Returns:
        list[dict]: List of dictionaries containing 'version' and 'path'.
    """
    versions = []

    if winreg is not None:
        locations = [
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Python\PythonCore",
            ),
            (
                winreg.HKEY_CURRENT_USER,
                r"SOFTWARE\Python\PythonCore",
            ),
        ]

        for hive, key_path in locations:
            try:
                key = winreg.OpenKey(
                    hive,
                    key_path,
                )

                index = 0

                while True:
                    try:
                        version = winreg.EnumKey(
                            key,
                            index,
                        )

                        install_path_key = winreg.OpenKey(
                            key,
                            f"{version}\\InstallPath",
                        )

                        install_path, _ = winreg.QueryValueEx(
                            install_path_key,
                            None,
                        )

                        python_path = Path(install_path) / "python.exe"

                        if python_path.is_file():
                            versions.append({
                                "version": version,
                                "path": str(python_path),
                            })

                        index += 1

                    except OSError:
                        break

            except FileNotFoundError:
                continue

    current = {
        "version": sys.version.split()[0],
        "path": sys.executable,
    }

    if current not in versions:
        versions.append(current)

    return versions
