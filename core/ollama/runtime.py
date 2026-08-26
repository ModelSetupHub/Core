import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from core.logging import write_log


COMPONENT = "ollama/runtime"


def get_status() -> dict:
    """
    Get the current Ollama runtime status.

    Returns installation state, server state, health state,
    and the installed client version.
    """

    installed = shutil.which("ollama") is not None

    if not installed:
        write_log(
            level="WARNING",
            component=COMPONENT,
            action="status",
            message="Ollama is not installed",
        )

        return {
            "installed": False,
            "running": False,
            "healthy": False,
            "version": None,
        }

    version_result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    output = (
        version_result.stdout
        + version_result.stderr
    )

    version = None

    for line in output.splitlines():
        if "client version is" in line:
            version = line.split(
                "client version is",
                1
            )[1].strip()
            break

        if line.startswith("ollama version is"):
            version = line.split(
                "ollama version is",
                1
            )[1].strip()
            break

    # Check the local Ollama API instead of relying on
    # the CLI client to determine whether the server is running.
    running = False
    healthy = False

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags",
            timeout=2,
        ) as response:
            running = response.status == 200
            healthy = running

    except Exception:
        pass

    write_log(
        level="INFO",
        component=COMPONENT,
        action="status",
        message="Ollama status checked",
        details={
            "installed": installed,
            "running": running,
            "healthy": healthy,
            "version": version,
        },
    )

    return {
        "installed": True,
        "running": running,
        "healthy": healthy,
        "version": version,
    }


def install(installer_path: str) -> None:
    """
    Install Ollama from an existing local installer.

    The installer must already exist on the filesystem.
    """

    installer = Path(
        installer_path
    ).expanduser().resolve()

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

        write_log(
            level="INFO",
            component=COMPONENT,
            action="install",
            message="Ollama installation completed",
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


def start() -> None:
    """
    Start the Ollama server if it is not already running.
    """

    status = get_status()

    if status["running"]:
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

        write_log(
            level="INFO",
            component=COMPONENT,
            action="start",
            message="Ollama start command executed",
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


def stop() -> None:
    """
    Stop the Ollama server process.
    """

    write_log(
        level="INFO",
        component=COMPONENT,
        action="stop",
        message="Stopping Ollama",
    )

    try:
        # Use the native process command for each platform.
        if os.name == "nt":
            subprocess.run(
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
            subprocess.run(
                ["pkill", "ollama"],
                check=False,
            )

        write_log(
            level="INFO",
            component=COMPONENT,
            action="stop",
            message="Ollama stop command executed",
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


def restart() -> None:
    """
    Restart the Ollama server.
    """

    write_log(
        level="INFO",
        component=COMPONENT,
        action="restart",
        message="Restarting Ollama",
    )

    stop()
    start()

    write_log(
        level="INFO",
        component=COMPONENT,
        action="restart",
        message="Ollama restart completed",
    )