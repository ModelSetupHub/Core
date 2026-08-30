"""Python system installer management and Windows registry detection."""

from pathlib import Path
import subprocess
import sys

try:
    import winreg
except ImportError:
    winreg = None

from core.logging import write_log

COMPONENT = "python"


def install_python(
    installer_path: str,
    all_users: bool = False,
) -> list[dict]:
    """Install Python from a local Windows installer executable in quiet mode.

    Args:
        installer_path: Path to the Python installer executable.
        all_users: Whether to install Python for all system users. Defaults to False.

    Returns:
        list[dict]: List of detected Python versions and executable paths.

    Raises:
        FileNotFoundError: If the specified installer file does not exist.
        RuntimeError: If the installation process fails.
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
