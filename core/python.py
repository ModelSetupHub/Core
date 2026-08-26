import shutil
import subprocess
import sys
import winreg
from pathlib import Path

from core.logging import write_log


def install_python(
    installer_path: str,
    all_users: bool = False,
):
    installer = Path(
        installer_path
    ).expanduser().resolve()

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
        command.append(
            "InstallAllUsers=1"
        )
    else:
        command.append(
            "InstallAllUsers=0"
        )

    result = subprocess.run(
        command,
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
            action="install_python",
            message="Failed to install Python",
            details={
                "installer": str(installer),
                "all_users": all_users,
                "error": result.stderr.strip(),
            },
        )

        raise RuntimeError(
            result.stderr.strip()
            or "Failed to install Python"
        )

    write_log(
        level="INFO",
        component="python",
        action="install_python",
        message="Python installed successfully",
        details={
            "installer": str(installer),
            "all_users": all_users,
        },
    )

    return get_python_status()


def get_python_status():
    versions = []

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

                    python_path = Path(
                        install_path
                    ) / "python.exe"

                    if python_path.is_file():
                        versions.append(
                            {
                                "version": version,
                                "path": str(python_path),
                            }
                        )

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


def create_environment(path: str):
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
    environment = Path(path).expanduser().resolve()

    if not environment.exists():
        raise FileNotFoundError(
            f"Environment not found: {environment}"
        )

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


def _get_python_path(environment: str | None = None):
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
    script = Path(path).expanduser().resolve()

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
    script = Path(path).expanduser().resolve()

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