"""Package management and Python script authoring and execution utilities."""

from pathlib import Path
import subprocess

from MSHCore.logging import write_log
from MSHCore.python.environment import get_python_path

COMPONENT = "python"


def install_packages(
    packages: list[str],
    environment: str | None = None,
) -> str:
    """Install one or more Python packages via pip.

    Args:
        packages: List of package names or version specifiers.
        environment: Optional path to the virtual environment.

    Returns:
        str: Output text from pip install.

    Raises:
        ValueError: If packages list is empty.
        RuntimeError: If pip install returns a non-zero exit code.
    """
    if not packages:
        raise ValueError("At least one package is required")

    python_path = get_python_path(environment)

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
    """Create a new Python script file on disk.

    Args:
        path: Target script path; must end in '.py'.
        content: Script source text to write.

    Returns:
        str: Resolved absolute path to the created script.

    Raises:
        ValueError: If the path does not have a .py extension.
        FileExistsError: If a file already exists at that path.
    """
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


def read_script(path: str) -> str:
    """Read the contents of an existing Python script file.

    Args:
        path: Path to the script; must end in '.py'.

    Returns:
        str: Source text of the script.

    Raises:
        ValueError: If the path does not have a .py extension.
        FileNotFoundError: If the script does not exist.
    """
    if Path(path).suffix.lower() != ".py":
        raise ValueError("Script path must have a .py extension")

    script = Path(path).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    content = script.read_text(encoding="utf-8")

    write_log(
        level="INFO",
        component=COMPONENT,
        action="read_script",
        message="Script read successfully",
        details={
            "path": str(script),
        },
    )

    return content


def edit_script(
    path: str,
    content: str,
) -> str:
    """Overwrite the contents of an existing Python script file.

    Args:
        path: Path to the existing script; must end in '.py'.
        content: Replacement source text.

    Returns:
        str: Resolved absolute path to the updated script.

    Raises:
        ValueError: If the path does not have a .py extension.
        FileNotFoundError: If the script does not exist.
    """
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
    """Delete a Python script file from disk.

    Args:
        path: Path to the script to delete; must end in '.py'.

    Returns:
        str: Resolved absolute path of the deleted script.

    Raises:
        ValueError: If the path does not have a .py extension.
        FileNotFoundError: If the script does not exist.
    """
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
    """Execute a Python script using the specified environment.

    Args:
        path: Path to the script to run; must end in '.py'.
        environment: Optional path to the virtual environment whose interpreter
            should run the script.

    Returns:
        str: Standard output produced by the script.

    Raises:
        ValueError: If the path does not have a .py extension.
        FileNotFoundError: If the script does not exist.
        RuntimeError: If the script exits with a non-zero status.
    """
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
