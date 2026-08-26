import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


def get_status() -> dict:
    installed = shutil.which("ollama") is not None

    if not installed:
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
            version = line.split("client version is", 1)[1].strip()
            break

        if line.startswith("ollama version is"):
            version = line.split("ollama version is", 1)[1].strip()
            break

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

    return {
        "installed": True,
        "running": running,
        "healthy": healthy,
        "version": version,
    }


def install(installer_path: str) -> None:
    installer = Path(installer_path).expanduser().resolve()

    if not installer.is_file():
        raise FileNotFoundError(
            f"Installer not found: {installer}"
        )

    subprocess.run(
        [str(installer)],
        check=True,
    )


def start() -> None:
    status = get_status()

    if status["running"]:
        return

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


def stop() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "ollama.exe"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            ["pkill", "ollama"],
            check=False,
        )


def restart() -> None:
    stop()
    start()