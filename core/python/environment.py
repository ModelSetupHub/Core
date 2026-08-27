import subprocess
import sys
from pathlib import Path

from core.logging import write_log


def create_environment(path: str):
    """Create Python virtual environment."""

    environment = Path(
        path
    ).expanduser().resolve()

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
    )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component="python",
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
        component="python",
        action="create_environment",
        message="Environment created successfully",
        details={
            "path": str(environment),
        },
    )

    return str(environment)


def remove_environment(path: str):
    """Remove Python virtual environment."""

    environment = Path(
        path
    ).expanduser().resolve()

    if not environment.exists():
        raise FileNotFoundError(
            f"Environment not found: {environment}"
        )

    import shutil

    shutil.rmtree(environment)

    write_log(
        level="INFO",
        component="python",
        action="remove_environment",
        message="Environment removed successfully",
        details={
            "path": str(environment),
        },
    )


def get_python_path(
    environment: str | None = None,
):
    """Get Python executable path."""

    if environment is None:
        return sys.executable

    environment_path = Path(
        environment
    ).expanduser().resolve()

    if sys.platform == "win32":
        python_path = (
            environment_path
            / "Scripts"
            / "python.exe"
        )
    else:
        python_path = (
            environment_path
            / "bin"
            / "python"
        )

    if not python_path.is_file():
        raise FileNotFoundError(
            f"Python executable not found: {python_path}"
        )

    return str(python_path)