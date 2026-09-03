"""Python virtual environment lifecycle management."""

from pathlib import Path
import shutil
import subprocess
import sys

from MSHCore.logging import write_log

COMPONENT = "python"


def create_environment(path: str) -> str:
    """Create a new isolated Python virtual environment.

    Args:
        path: Filesystem path where the virtual environment will be created.

    Returns:
        str: Resolved absolute path to the created virtual environment.

    Raises:
        FileExistsError: If the target environment path already exists.
        RuntimeError: If virtual environment creation fails.
    """
    environment = Path(path).expanduser().resolve()

    if environment.exists():
        raise FileExistsError(
            f"Environment path already exists: {environment}"
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            str(environment),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        # This process may be an MCP server whose stdin carries the JSON-RPC
        # stream, and a child inheriting it could read the protocol's bytes.
        stdin=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="create_environment",
            message="Failed to create environment",
            details={
                "path": str(environment),
                "error": result.stderr.strip(),
            },
        )
        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to create environment: {environment}"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="create_environment",
        message="Environment created successfully",
        details={
            "path": str(environment),
        },
    )

    return str(environment)


def remove_environment(path: str) -> None:
    """Remove an existing Python virtual environment directory recursively.

    Args:
        path: Filesystem path to the virtual environment to delete.

    Raises:
        FileNotFoundError: If the virtual environment path does not exist.
    """
    environment = Path(path).expanduser().resolve()

    if not environment.exists():
        raise FileNotFoundError(f"Environment not found: {environment}")

    shutil.rmtree(environment)

    write_log(
        level="INFO",
        component=COMPONENT,
        action="remove_environment",
        message="Environment removed successfully",
        details={
            "path": str(environment),
        },
    )


def get_python_path(
    environment: str | None = None,
) -> str:
    """Get the absolute path to the Python executable.

    Args:
        environment: Optional path to a virtual environment. If None, the
            running interpreter's path is returned.

    Returns:
        str: Absolute path to the Python interpreter executable.

    Raises:
        FileNotFoundError: If the Python binary inside the environment cannot
            be found.
    """
    if environment is None:
        return sys.executable

    environment_path = Path(environment).expanduser().resolve()

    # Virtual environments place the interpreter in Scripts/ on Windows and
    # bin/ everywhere else.
    if sys.platform == "win32":
        python_path = environment_path / "Scripts" / "python.exe"
    else:
        python_path = environment_path / "bin" / "python"

    if not python_path.is_file():
        raise FileNotFoundError(
            f"Python executable not found: {python_path}"
        )

    return str(python_path)
