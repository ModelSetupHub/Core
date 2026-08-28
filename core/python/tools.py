"""Package management and Python script authoring and execution utilities."""

from pathlib import Path
import subprocess

from core.cancellation import (
    CancellationToken,
    OperationCancelled,
    log_cancelled,
    run_cancellable,
)
from core.logging import write_log
from core.python.environment import get_python_path

COMPONENT = "python"


def install_packages(
    packages: list[str],
    environment: str | None = None,
    cancellation: CancellationToken | None = None,
) -> str:
    """Install one or more Python packages via pip.

    Args:
        packages: List of package names or version specifiers.
        environment: Optional path to the virtual environment.
        cancellation: Optional token that stops the installation part-way.

    Returns:
        str: Output text from pip install.

    Raises:
        ValueError: If packages list is empty.
        RuntimeError: If pip install returns a non-zero exit code.
        OperationCancelled: If the token is cancelled. Pip is terminated and
            anything it had already installed is uninstalled, so a cancelled
            installation leaves nothing behind but its log entry.
    """
    if not packages:
        raise ValueError("At least one package is required")

    python_path = get_python_path(environment)

    # Recorded before pip runs, so a cancellation can tell what it added.
    installed_before = _installed_distributions(python_path)

    try:
        result = run_cancellable(
            [
                python_path,
                "-m",
                "pip",
                "install",
                *packages,
            ],
            cancellation=cancellation,
            component=COMPONENT,
            action="install_packages",
        )
    except OperationCancelled as error:
        _cleanup_cancelled_packages(
            python_path=python_path,
            packages=packages,
            environment=environment,
            installed_before=installed_before,
            reason=str(error),
        )
        raise

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Failed to install packages"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="install_packages",
        message="Packages installed successfully",
        details={
            "packages": packages,
            "environment": environment,
        },
    )

    return result.stdout.strip()


def _installed_distributions(python_path: str) -> set[str]:
    """List the distributions currently installed for an interpreter.

    Args:
        python_path: Interpreter to inspect.

    Returns:
        set[str]: Lower-cased distribution names, empty when pip cannot be
        queried — in which case a cancellation simply has nothing to compare
        against and removes nothing.
    """
    try:
        result = subprocess.run(
            [python_path, "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    if result.returncode != 0:
        return set()

    names = set()

    for line in result.stdout.splitlines():
        name = line.split("==")[0].strip()
        if name:
            names.add(name.lower())

    return names


def _cleanup_cancelled_packages(
    python_path: str,
    packages: list[str],
    environment: str | None,
    installed_before: set[str],
    reason: str,
) -> None:
    """Uninstall whatever a cancelled pip run had already installed.

    Pip installs one distribution at a time, so an interrupted run can leave some
    of the requested packages — and their dependencies — behind. Comparing the
    installed set against the one taken before the run identifies exactly what
    this call added, including dependencies, and those are removed. Packages that
    were already installed are left alone.

    Args:
        python_path: Interpreter pip was running against.
        packages: Packages the run was asked to install.
        environment: Environment that was targeted, for the log entry.
        installed_before: Distributions present before the run.
        reason: Why the installation was cancelled.
    """
    added = sorted(_installed_distributions(python_path) - installed_before)

    removed = False
    cleanup_error: str | None = None

    if added:
        try:
            result = subprocess.run(
                [python_path, "-m", "pip", "uninstall", "-y", *added],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=300,
            )
            removed = result.returncode == 0
            if not removed:
                cleanup_error = result.stderr.strip()
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_error = str(error)

    log_cancelled(
        component=COMPONENT,
        action="install_packages",
        message="Package installation cancelled",
        details={
            "packages": packages,
            "environment": environment,
            "distributions_added": added,
            "distributions_removed": removed,
            "cleanup_error": cleanup_error,
            "reason": reason,
        },
    )


def uninstall_packages(
    packages: list[str],
    environment: str | None = None,
) -> str:
    """Uninstall one or more Python packages via pip.

    Args:
        packages: List of package names to uninstall.
        environment: Optional path to the virtual environment.

    Returns:
        str: Output text from pip uninstall.

    Raises:
        RuntimeError: If pip uninstall fails.
    """
    python_path = get_python_path(environment)

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
            result.stderr.strip() or "Failed to uninstall packages"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
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
) -> str:
    """List installed Python packages in the selected environment.

    Args:
        environment: Optional path to the virtual environment.

    Returns:
        str: Output text from pip list.

    Raises:
        RuntimeError: If pip list fails.
    """
    python_path = get_python_path(environment)

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
            result.stderr.strip() or "Failed to list packages"
        )

    return result.stdout.strip()


def create_script(
    path: str,
    content: str,
) -> str:
    """Create a new Python script file on disk."""
    if Path(path).suffix.lower() != ".py":
        raise ValueError("Script path must have a .py extension")

    script = Path(path).expanduser().resolve()

    if script.exists():
        raise FileExistsError(f"Script already exists: {script}")

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
        component=COMPONENT,
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
) -> str:
    """Overwrite the contents of an existing Python script file."""
    if Path(path).suffix.lower() != ".py":
        raise ValueError("Script path must have a .py extension")

    script = Path(path).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    script.write_text(
        content,
        encoding="utf-8",
    )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="edit_script",
        message="Script updated successfully",
        details={
            "path": str(script),
        },
    )

    return str(script)


def delete_script(path: str) -> str:
    """Delete a Python script file from disk."""
    if Path(path).suffix.lower() != ".py":
        raise ValueError("Script path must have a .py extension")

    script = Path(path).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    script.unlink()

    write_log(
        level="INFO",
        component=COMPONENT,
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
) -> str:
    """Execute a Python script using the specified environment."""
    if Path(path).suffix.lower() != ".py":
        raise ValueError("Script path must have a .py extension")

    script = Path(path).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    python_path = get_python_path(environment)

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
            result.stderr.strip() or f"Failed to run script: {script}"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="run_script",
        message="Script executed successfully",
        details={
            "path": str(script),
            "environment": environment,
        },
    )

    return result.stdout.strip()
