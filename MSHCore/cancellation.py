"""Cooperative cancellation for the long-running operations.

Two operations here can run long enough that a user wants to stop them
mid-flight: a download queue and a benchmark run. They are cancelled the same
way, through a token passed into the operation:

    token = CancellationToken()
    ...
    ollama_runner.run_test(model="llama3", prompts=[...], cancellation=token)

    # from another thread
    token.cancel()

The operation checks the token at every point where it can stop safely and
raises :class:`OperationCancelled`. Cancelling is cooperative and therefore
never leaves a half-written file behind: each operation cleans up its own
partial work before the exception propagates, and records a ``cancelled`` entry
in the execution log so the cancellation is visible afterwards.
"""

from __future__ import annotations

import threading

from MSHCore.logging import write_log

COMPONENT = "cancellation"

# How often a blocking wait re-checks the token.
POLL_INTERVAL = 0.2


class OperationCancelled(Exception):
    """Raised by a long-running operation when its token has been cancelled."""


class CancellationToken:
    """Thread-safe cancel flag shared with a running operation.

    The operation and the caller requesting the cancellation are always on
    different threads, so the flag is a :class:`threading.Event`: setting it
    is atomic and a waiter can block on it instead of sleeping in a loop.
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


def log_cancelled(
    component: str,
    action: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Record that an operation was cancelled.

    Cancelling is meant to leave nothing behind except this entry, so every
    operation calls it on the way out — that log line is the only lasting
    trace a cancelled operation is allowed to leave.

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
