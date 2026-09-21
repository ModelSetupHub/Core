"""Python system installer management and platform interpreter detection."""

from pathlib import Path
import shutil
import subprocess
import sys

try:
    import winreg
except ImportError:
    # Non-Windows platforms have no registry; detection falls back to the
    # running interpreter only.
    winreg = None

from MSHCore.logging import write_log

COMPONENT = "python"


def install_python(
    installer_path: str,
    all_users: bool = False,
) -> list[dict]:
    """Install Python from a local installer in quiet mode.

    The Windows installer executable is passed its own quiet flags. On Linux
    the argument is instead a script or package archive, which is run with a
    shell — sourced for a ``.sh`` script, extracted with tar otherwise — so a
    downloaded python.org source tarball and a distribution package both work.

    Args:
        installer_path: Path to the Python installer file.
        all_users: Whether to install Python for all system users. Applies to
            the Windows installer; a Linux install runs with the caller's own
            privileges, so this is ignored there. Defaults to False.

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

    if sys.platform == "win32":
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
    else:
        # A .sh script has to be sourced to mean anything; an archive is
        # unpacked into current directory, which the caller controls.
        if installer.suffix.lower() == ".sh":
            command = ["bash", "-c", f"source '{installer}'"]
        else:
            command = ["tar", "-xf", str(installer)]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        # The installer runs quietly, but an installer that decides to prompt
        # must not read this process's stdin: that stream may belong to the
        # caller, not to the installer.
        stdin=subprocess.DEVNULL,
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


def _windows_python_versions() -> list[dict]:
    """Enumerate registered Python installs from the Windows registry.

    Returns:
        list[dict]: One dictionary per registered version whose interpreter
        exists on disk.
    """
    versions = []

    if winreg is None:
        return versions

    # Per-machine installs live under HKLM and per-user ones under HKCU, so
    # both hives are enumerated.
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
            with winreg.OpenKey(hive, key_path) as key:
                index = 0

                while True:
                    try:
                        version = winreg.EnumKey(key, index)
                    except OSError:
                        # Raised once the versions are exhausted, which is
                        # how this loop terminates.
                        break

                    index += 1

                    # A version whose InstallPath is missing or unreadable
                    # is skipped on its own; enumeration continues so one
                    # malformed entry cannot hide the versions after it.
                    try:
                        with winreg.OpenKey(
                            key,
                            f"{version}\\InstallPath",
                        ) as install_path_key:
                            install_path, _ = winreg.QueryValueEx(
                                install_path_key,
                                None,
                            )
                    except OSError:
                        continue

                    python_path = Path(install_path) / "python.exe"

                    if python_path.is_file():
                        versions.append({
                            "version": version,
                            "path": str(python_path),
                        })

        except OSError:
            # The hive has no PythonCore key at all.
            continue

    return versions


def _linux_python_versions() -> list[dict]:
    """Find the Python interpreters this machine has on its PATH.

    The version each one reports is the only reliable source: the file name
    is a convention, not a fact, so a ``python3`` pointing at 3.13 is
    reported as 3.13.

    Returns:
        list[dict]: One dictionary per distinct interpreter, in PATH order.
    """
    versions = []
    seen: set[str] = set()

    for name in ("python3", "python"):
        found = shutil.which(name)

        if not found:
            continue

        resolved = str(Path(found).resolve())

        if resolved in seen:
            continue

        seen.add(resolved)

        try:
            result = subprocess.run(
                [
                    resolved,
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:3])))",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        version = result.stdout.strip()

        if result.returncode == 0 and version:
            versions.append({
                "version": version,
                "path": resolved,
            })

    return versions


def get_python_status() -> list[dict]:
    """Get detected installed Python versions for this platform.

    Windows installs are read from the registry; Linux interpreters are
    discovered on PATH.

    Returns:
        list[dict]: List of dictionaries containing 'version' and 'path'.
    """
    if sys.platform == "win32":
        versions = _windows_python_versions()
    else:
        versions = _linux_python_versions()

    # The running interpreter is always reported, even when it was not
    # registered in the registry (a venv, or a portable install).
    current = {
        "version": sys.version.split()[0],
        "path": sys.executable,
    }

    if current not in versions:
        versions.append(current)

    return versions
