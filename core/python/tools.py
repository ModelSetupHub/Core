import subprocess
from pathlib import Path

from core.logging import write_log
from .environment import get_python_path


def install_packages(
    packages: list[str],
    environment: str | None = None,
):
    """Install Python packages."""

    if not packages:
        raise ValueError(
            "At least one package is required"
        )

    python_path = get_python_path(
        environment
    )

    result = subprocess.run(
        [
            python_path,
            "-m",
            "pip",
            "install",
            *packages,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or "Failed to install packages"
        )

    write_log(
        level="INFO",
        component="python",
        action="install_packages",
        message="Packages installed successfully",
        details={
            "packages": packages,
            "environment": environment,
        },
    )

    return result.stdout.strip()


def uninstall_packages(
    packages: list[str],
    environment: str | None = None,
):
    """Uninstall Python packages."""

    python_path = get_python_path(
        environment
    )

    result = subprocess.run(
        [
            python_path,
            "-m",
            "pip",
            "uninstall",
            "-y",
            *packages,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or "Failed to uninstall packages"
        )

    write_log(
        level="INFO",
        component="python",
        action="uninstall_packages",
        message="Packages uninstalled successfully",
        details={
            "packages": packages,
            "environment": environment,
        },
    )

    return result.stdout.strip()


def list_packages(
    environment: str | None = None,
):
    """List installed Python packages."""

    python_path = get_python_path(
        environment
    )

    result = subprocess.run(
        [
            python_path,
            "-m",
            "pip",
            "list",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or "Failed to list packages"
        )

    return result.stdout.strip()


def create_script(
    path: str,
    content: str,
):
    """Create Python script."""

    script = Path(
        path
    ).expanduser().resolve()

    if script.exists():
        raise FileExistsError(
            f"Script already exists: {script}"
        )

    script.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    script.write_text(
        content,
        encoding="utf-8",
    )

    write_log(
        level="INFO",
        component="python",
        action="create_script",
        message="Script created successfully",
        details={
            "path": str(script),
        },
    )

    return str(script)


def edit_script(
    path: str,
    content: str,
):
    """Edit Python script."""

    script = Path(
        path
    ).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    script.write_text(
        content,
        encoding="utf-8",
    )

    write_log(
        level="INFO",
        component="python",
        action="edit_script",
        message="Script updated successfully",
        details={
            "path": str(script),
        },
    )

    return str(script)


def delete_script(path: str):
    """Delete Python script."""

    script = Path(
        path
    ).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    script.unlink()

    write_log(
        level="INFO",
        component="python",
        action="delete_script",
        message="Script deleted successfully",
        details={
            "path": str(script),
        },
    )

    return str(script)


def run_script(
    path: str,
    environment: str | None = None,
):
    """Run Python script."""

    script = Path(
        path
    ).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    python_path = get_python_path(
        environment
    )

    result = subprocess.run(
        [
            python_path,
            str(script),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to run script: {script}"
        )

    write_log(
        level="INFO",
        component="python",
        action="run_script",
        message="Script executed successfully",
        details={
            "path": str(script),
            "environment": environment,
        },
    )

    return result.stdout.strip()