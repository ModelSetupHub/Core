"""Cooperative cancellation for the long-running operations.

Three operations here can run long enough that a user wants to stop them
mid-flight: a download queue, a benchmark run, and an installer. They are
cancelled the same way, through a token passed into the operation:

    token = CancellationToken()
    ...
    experiment.run_test(model="llama3", prompts=[...], cancellation=token)

    # from another thread
    token.cancel()

The operation checks the token at every point where it can stop safely and
raises :class:`OperationCancelled`. Cancelling is cooperative and therefore
never leaves a half-written file or a stray child process: each operation
cleans up its own partial work before the exception propagates, and records a
``cancelled`` entry in the execution log so the cancellation is visible
afterwards.

A subprocess is the one case where checking a flag is not enough, since the work
happens in another process. :func:`run_cancellable` runs it and, on
cancellation, terminates its whole process tree, so an installer that spawned
children does not keep running after the cancellation.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Any, Sequence

from core.logging import write_log

COMPONENT = "cancellation"

# How often a blocking wait re-checks the token.
POLL_INTERVAL = 0.2

# Grace period for a terminated process tree before it is killed outright.
TERMINATE_GRACE = 5.0


class OperationCancelled(Exception):
    """Raised by a long-running operation when its token has been cancelled."""


class CancellationToken:
    """Thread-safe cancel flag shared with a running operation.

    The operation and the caller requesting the cancellation are always on
    different threads, so the flag is an :class:`threading.Event`: setting it is
    atomic and a waiter can block on it instead of sleeping in a loop.
    """

    def __init__(self) -> None:
        """Create a token that has not been cancelled."""
        self._event = threading.Event()
        self._reason: str | None = None
        self._lock = threading.Lock()

    def cancel(self, reason: str | None = None) -> None:
        """Request cancellation.

        Safe to call more than once; the first reason given is kept.

        Args:
            reason: Optional explanation recorded with the cancellation.
        """
        with self._lock:
            if self._reason is None:
                self._reason = reason or "Cancelled by request"

        self._event.set()

    @property
    def cancelled(self) -> bool:
        """bool: Whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """str | None: Why the operation was cancelled, once it has been."""
        with self._lock:
            return self._reason

    def raise_if_cancelled(self) -> None:
        """Stop the operation if cancellation has been requested.

        Call this at every point where the operation can stop without leaving
        partial work behind.

        Raises:
            OperationCancelled: If the token has been cancelled.
        """
        if self._event.is_set():
            raise OperationCancelled(self.reason or "Cancelled by request")

    def wait(self, timeout: float) -> bool:
        """Sleep, but wake immediately if cancellation is requested.

        Args:
            timeout: Seconds to wait at most.

        Returns:
            bool: True if the token was cancelled during the wait.
        """
        return self._event.wait(timeout)


def run_cancellable(
    command: Sequence[str],
    cancellation: CancellationToken | None = None,
    component: str = COMPONENT,
    action: str = "run",
    **popen_kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run a subprocess that can be stopped part-way through.

    Behaves like ``subprocess.run(..., capture_output=True, text=True)`` when the
    token is never cancelled. When it is, the child's whole process tree is
    terminated — an installer typically spawns children, and killing only the
    launcher would leave them running — and the partial output is discarded.

    Args:
        command: Argument vector to execute.
        cancellation: Token that stops the process; None runs it to completion.
        component: Component name recorded if the process is terminated.
        action: Action name recorded if the process is terminated.
        **popen_kwargs: Extra arguments forwarded to ``subprocess.Popen``.

    Returns:
        subprocess.CompletedProcess: Completed process with captured output.

    Raises:
        OperationCancelled: If the token was cancelled before or during the run.
    """
    if cancellation is not None:
        cancellation.raise_if_cancelled()

    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **popen_kwargs,
    )

    if cancellation is None:
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    # Output has to be drained on another thread: a pipe that fills up blocks the
    # child, and this thread has to stay free to notice the cancellation.
    output: dict[str, str] = {}

    def drain() -> None:
        try:
            output["stdout"], output["stderr"] = process.communicate()
        except Exception as error:  # pragma: no cover - defensive
            output["stdout"] = ""
            output["stderr"] = str(error)

    reader = threading.Thread(target=drain, name="cancellable-output", daemon=True)
    reader.start()

    while reader.is_alive():
        if cancellation.wait(POLL_INTERVAL):
            terminate_tree(process)
            reader.join(timeout=TERMINATE_GRACE)

            write_log(
                level="WARNING",
                component=component,
                action=action,
                message="Process terminated after cancellation",
                details={
                    "command": list(command),
                    "reason": cancellation.reason,
                },
            )

            raise OperationCancelled(
                cancellation.reason or "Cancelled by request"
            )

    reader.join()

    return subprocess.CompletedProcess(
        args=list(command),
        returncode=process.returncode,
        stdout=output.get("stdout", ""),
        stderr=output.get("stderr", ""),
    )


def terminate_tree(process: subprocess.Popen) -> None:
    """Terminate a process and everything it spawned.

    An installer is usually a launcher that starts the real installer, so
    terminating just the process this module started would leave the actual work
    running. On Windows ``taskkill /T`` walks the tree; elsewhere the child is put
    in its own process group at launch and the group is signalled. Either way a
    plain terminate is the fallback, and a process still alive after the grace
    period is killed.

    Args:
        process: Process to stop.
    """
    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, AttributeError):
            pass

    deadline = time.monotonic() + TERMINATE_GRACE

    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)

    try:
        process.kill()
    except OSError:
        pass


def log_cancelled(
    component: str,
    action: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Record that an operation was cancelled.

    Cancelling is meant to leave nothing behind except this entry, so every
    operation calls it on the way out — that log line is the only lasting trace a
    cancelled operation is allowed to leave.

    Args:
        component: Component the operation belongs to.
        action: Action that was cancelled.
        message: Human-readable description of the cancellation.
        details: Optional metadata, for example what was cleaned up.
    """
    write_log(
        level="WARNING",
        component=component,
        action=action,
        message=message,
        details=details or {},
    )


