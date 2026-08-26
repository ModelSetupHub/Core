import shutil
import subprocess
import sys
from pathlib import Path


def create_environment(path: str):
    environment = Path(path).expanduser().resolve()

    if environment.exists():
        raise FileExistsError(
            f"Environment path already exists: {environment}"
        )

    result = subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to create environment: {environment}"
        )

    return str(environment)


def remove_environment(path: str):
    environment = Path(path).expanduser().resolve()

    if not environment.exists():
        raise FileNotFoundError(
            f"Environment not found: {environment}"
        )

    shutil.rmtree(environment)


def _get_python_path(environment: str | None = None) -> str:
    if environment is None:
        return sys.executable

    environment_path = Path(
        environment
    ).expanduser().resolve()

    if sys.platform == "win32":
        python_path = environment_path / "Scripts" / "python.exe"
    else:
        python_path = environment_path / "bin" / "python"

    if not python_path.is_file():
        raise FileNotFoundError(
            f"Python executable not found: {python_path}"
        )

    return str(python_path)


def install_packages(
    packages: list[str],
    environment: str | None = None,
):
    if not packages:
        raise ValueError(
            "At least one package is required"
        )

    python_path = _get_python_path(environment)

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

    return result.stdout.strip()


def uninstall_packages(
    packages: list[str],
    environment: str | None = None,
):
    if not packages:
        raise ValueError(
            "At least one package is required"
        )

    python_path = _get_python_path(environment)

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

    return result.stdout.strip()


def list_packages(
    environment: str | None = None,
):
    python_path = _get_python_path(environment)

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
    script = Path(path).expanduser().resolve()

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

    return str(script)


def edit_script(
    path: str,
    content: str,
):
    script = Path(path).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    script.write_text(
        content,
        encoding="utf-8",
    )

    return str(script)


def delete_script(path: str):
    script = Path(path).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    script.unlink()

    return str(script)


def run_script(
    path: str,
    environment: str | None = None,
):
    script = Path(path).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    python_path = _get_python_path(environment)

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

    return result.stdout.strip()