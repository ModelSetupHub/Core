"""Filesystem locations MSHCore reads and writes.

Everything the toolkit stores lives under one root in the current user's local
application data::

    %LOCALAPPDATA%\\MSH
    ├── downloads/          completed downloads
    └── logs/
        └── executions.log  the execution log

The paths are collected here so the download manager and the logger cannot
drift apart, and so a caller that wants to know where something landed reads
the same constant the writer used.

The root is deliberately per-user rather than machine-wide: ``%LOCALAPPDATA%``
is writable without elevation, so nothing here depends on the server having
been started as an administrator. On a platform with no ``%LOCALAPPDATA%`` the
root is ``~/.msh``.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "MSH"


def app_data_directory() -> Path:
    """Return the root directory everything the toolkit stores lives under.

    Resolved on each call rather than at import, so a process that adjusts
    ``%LOCALAPPDATA%`` — a test harness, or a service running under a different
    profile — is not left writing to the profile that happened to be current
    when the module was first imported.

    Returns:
        Path: ``%LOCALAPPDATA%\\MSH`` when that variable is set, otherwise
        ``~/.msh``.
    """
    local_app_data = os.environ.get("LOCALAPPDATA")

    if local_app_data:
        return Path(local_app_data) / APP_NAME

    return Path.home() / f".{APP_NAME.lower()}"


def downloads_directory() -> Path:
    """Return where completed downloads land by default.

    Returns:
        Path: ``%LOCALAPPDATA%\\MSH\\downloads``. Not created; pass it to
        :func:`ensure_directory` when it has to exist.
    """
    return app_data_directory() / "downloads"


def logs_directory() -> Path:
    """Return the directory the execution log is written to.

    Returns:
        Path: ``%LOCALAPPDATA%\\MSH\\logs``. Not created; pass it to
        :func:`ensure_directory` when it has to exist.
    """
    return app_data_directory() / "logs"


# Resolved once for callers that want the default as a plain value — a function
# signature's default, or a tool schema. Prefer the functions above anywhere the
# directory is about to be written to.
APP_DATA_DIRECTORY = app_data_directory()
DOWNLOADS_DIRECTORY = downloads_directory()
LOGS_DIRECTORY = logs_directory()

LOG_FILE_NAME = "executions.log"


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and its parents, and return it.

    Args:
        path: Directory to create. Missing parents are created with it, and an
            existing directory is left as it is.

    Returns:
        Path: The directory, which now exists.

    Raises:
        PermissionError: If the process may not create the directory. The
            default root is per-user precisely so that this does not happen, so
            it means the caller named a location it cannot write to.
        OSError: If the directory could not be created for any other reason.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    return directory
