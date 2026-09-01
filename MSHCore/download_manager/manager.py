"""Sequential download queue manager with pause, resume, skip, monitoring."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterable
from urllib.parse import urlparse

from MSHCore.cancellation import CancellationToken, log_cancelled
from MSHCore.logging import write_log
from .downloader import (
    DownloadCancelled,
    DownloadError,
    DownloadSkipped,
    Downloader,
)
from .sources import ALLOWED_DOMAINS

COMPONENT = "download_manager"

# Queue item states the worker must not process again on a later start. A
# failed item is deliberately absent: restarting a session is how a failure is
# retried, and `start` puts those items back to "waiting" itself.
CONSUMED_STATUSES = frozenset({"completed", "skipped", "cancelled"})

# Queue item states that mean the item will never run again.
FINAL_STATUSES = frozenset({"completed", "skipped", "failed"})

# How long `purge` waits for the worker to notice the cancellation and exit.
# The worker stops at a chunk boundary, so this only has to cover one chunk
# read.
WORKER_JOIN_TIMEOUT = 30.0


class SessionCancelled(RuntimeError):
    """Raised when a cancelled session is queued to or started again.

    A cancellation removes everything the session produced, so there is nothing
    left for a second run to continue from: the queue is gone and the files it
    had fetched are deleted. Reusing the object would silently re-download
    whatever it had been asked for before, which is why this is an error rather
    than a fresh start. Create a new manager instead.
    """


class DownloadManager:
    """Sequential download manager handling queueing, speed, and workers.

    The manager handles:
        - Download queue
        - Progress tracking and speed calculation
        - Automatic retry
        - Pause / Resume
        - Skip
        - Cancellation
        - Optional Windows keyboard shortcut listener
        - Event logging
    """

    def __init__(
        self,
        download_directory: str | Path = "data/downloads",
        max_retries: int = 3,
    ) -> None:
        """Initialize the DownloadManager.

        Args:
            download_directory: Target folder path for downloaded files.
            max_retries: Maximum download retry attempts per file. Defaults
                to 3.
        """
        self.download_directory = Path(download_directory)
        # Whether this manager created the directory, so a cancellation knows
        # if removing it again is its business.
        self._created_directory = not self.download_directory.exists()
        self.download_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.max_retries = max_retries
        self._queue: list[dict] = []
        self._current_downloader: Downloader | None = None
        self._current_index = -1

        self._running = False
        self._paused = False
        self._skip_requested = False

        self._cleanup_on_cancel = True
        self._cleanup_done = False
        # Set once the session has been cancelled or closed. A closed session
        # is finished for good: its queue is gone, so accepting a new file or a
        # second start would silently re-download what the cancellation
        # removed.
        self._closed = False

        self._worker_thread: threading.Thread | None = None
        self._keyboard_thread: threading.Thread | None = None

        # Guards the queue and every flag above. It is re-entrant because the
        # cancellation path reads the status it is about to tear down.
        self._lock = threading.RLock()
        # The cancel flag, its reason, and the event the worker and the
        # keyboard listener wake on, all of which CancellationToken already
        # provides. It is one-way by design, which suits a session that cannot
        # be restarted once cancelled.
        self._cancellation = CancellationToken()
        self._keyboard_stop = threading.Event()

    def _verify_download_source(self, url: str) -> None:
        """Verify that a URL domain is in the allowed whitelist.

        Args:
            url: URL string to validate.

        Raises:
            PermissionError: If the URL domain is not in ALLOWED_DOMAINS.
        """
        domain = urlparse(url).netloc.lower()
        if domain not in ALLOWED_DOMAINS:
            raise PermissionError(
                f"Access denied: domain '{domain}' is not allowed."
            )

    # ========================================================
    # Queue
    # ========================================================

    def add(
        self,
        url: str,
        filename: str | None = None,
    ) -> None:
        """Add a file URL to the sequential download queue.

        Args:
            url: Remote file HTTP/HTTPS URL.
            filename: Optional custom target filename. If not provided, it is
                inferred from the URL.

        Raises:
            ValueError: If the URL is empty.
            PermissionError: If the domain is not in the whitelist.
            SessionCancelled: If the session has already been cancelled or
                closed.
        """
        if not url:
            raise ValueError("URL cannot be empty.")
        self._verify_download_source(url)

        with self._lock:
            self._raise_if_closed("queue a file")

            if not filename:
                filename = (
                    Path(url.split("?")[0]).name
                    or f"download_{len(self._queue) + 1}"
                )

            item = {
                "url": url,
                "filename": filename,
                "status": "waiting",
                "downloaded": 0,
                "total": None,
                "speed": 0.0,
                "error": None,
            }

            self._queue.append(item)
            position = len(self._queue)

        write_log(
            level="INFO",
            component=COMPONENT,
            action="queue_add",
            message="File added to download queue",
            details={
                "filename": filename,
                "queue_position": position,
                "total_files": position,
                "url": url,
            },
        )

    def add_many(
        self,
        urls: Iterable[str],
    ) -> None:
        """Add multiple URLs sequentially to the download queue.

        Args:
            urls: Iterable of URL strings.
        """
        for url in urls:
            self.add(url)

    def _raise_if_closed(self, attempt: str) -> None:
        """Reject an operation on a cancelled or closed session.

        Called with the lock held.

        Args:
            attempt: What the caller was trying to do, for the message.

        Raises:
            SessionCancelled: If the session has been cancelled or closed.
        """
        if self._closed:
            raise SessionCancelled(
                f"Cannot {attempt}: this download session was "
                f"{'cancelled' if self._cancellation.cancelled else 'closed'} "
                f"and cannot be reused. Create a new session instead."
            )

    # ========================================================
    # Start
    # ========================================================

    def start(self) -> None:
        """Start background worker threads to process the download queue.

        Only items that have not already run are processed: a completed,
        skipped or cancelled item is left alone, so restarting a session that
        is part-finished never downloads the same file twice. A previously
        failed item is retried, which is the point of restarting one.

        Raises:
            RuntimeError: If the download queue is empty, or every item in it
                has already run.
            SessionCancelled: If the session has been cancelled or closed.
        """
        with self._lock:
            self._raise_if_closed("start the queue")

            if self._running:
                return

            if not self._queue:
                raise RuntimeError("Download queue is empty.")

            pending = [
                item
                for item in self._queue
                if item["status"] not in CONSUMED_STATUSES
            ]

            if not pending:
                raise RuntimeError(
                    "Every file in this queue has already been downloaded, "
                    "skipped or cancelled."
                )

            # A retry starts from "waiting" so the worker and any watcher see
            # one consistent set of states rather than a stale "failed" or
            # "paused".
            for item in pending:
                item["status"] = "waiting"
                item["error"] = None

            self._running = True
            self._paused = False
            self._skip_requested = False
            # The cancel flag is deliberately not reset: `_raise_if_closed`
            # above has already refused a cancelled session, so a start that
            # gets this far has never been cancelled.
            self._keyboard_stop.clear()

            total = len(self._queue)

            self._worker_thread = threading.Thread(
                target=self._worker,
                name="DownloadWorker",
                daemon=True,
            )
            worker = self._worker_thread

        write_log(
            level="INFO",
            component=COMPONENT,
            action="manager_start",
            message="Download manager started",
            details={
                "total_files": total,
            },
        )

        worker.start()

        if os.name == "nt" and self._console_input_available():
            with self._lock:
                self._keyboard_thread = threading.Thread(
                    target=self._keyboard_listener,
                    name="DownloadKeyboard",
                    daemon=True,
                )
                keyboard = self._keyboard_thread
            keyboard.start()

    @staticmethod
    def _console_input_available() -> bool:
        """Report whether this process has a console to read keys from.

        The listener exists for a user running the manager from a terminal.
        Under a server — where stdin is a pipe carrying the protocol — reading
        keys would consume that stream, so the listener is not started at all.

        Returns:
            bool: True when stdin is an interactive console.
        """
        try:
            return bool(sys.stdin) and sys.stdin.isatty()
        except (AttributeError, ValueError, OSError):
            return False

    # ========================================================
    # Worker
    # ========================================================

    def _worker(self) -> None:
        """Background worker loop executing sequential item downloads."""
        try:
            for index, item in enumerate(self._queue):
                if self._cancellation.cancelled:
                    break

                with self._lock:
                    self._current_index = index

                    # Anything already run is left as it is. Without this a
                    # restart would re-download a completed file and a
                    # cancelled item would come back to life.
                    if item["status"] in CONSUMED_STATUSES:
                        continue

                    total_files = len(self._queue)
                    filename = item["filename"]
                    url = item["url"]

                    item["status"] = "downloading"
                    item["error"] = None

                    destination = self.download_directory / filename

                    downloader = Downloader(
                        url=url,
                        destination=destination,
                        max_retries=self.max_retries,
                    )

                    # A cancellation between the checks above and this
                    # assignment would otherwise never reach the new
                    # downloader, leaving it transferring after the session was
                    # cancelled.
                    if self._cancellation.cancelled:
                        item["status"] = "cancelled"
                        break

                    self._current_downloader = downloader

                write_log(
                    level="INFO",
                    component=COMPONENT,
                    action="download_start",
                    message=f"Starting download {index + 1} of {total_files}",
                    details={
                        "file_index": index + 1,
                        "total_files": total_files,
                        "filename": filename,
                        "url": url,
                    },
                )

                try:
                    downloader.download(
                        progress_callback=(
                            lambda downloaded, total, i=index: self._progress(
                                i, downloaded, total
                            )
                        ),
                        status_callback=(
                            lambda status, details, i=index:
                            self._status_event(i, status, details)
                        ),
                    )

                    # Check if skip was requested
                    if self._skip_requested:
                        self._mark_skipped(index)
                        self._skip_requested = False
                        continue

                    with self._lock:
                        cancelled = self._cancellation.cancelled
                        fetched = bool(item.get("downloaded"))

                        # A cancellation can land in the moment between the
                        # last chunk and here. The cleanup is about to delete
                        # this file, so recording it as completed would leave
                        # the queue naming something no longer on disk. A file
                        # the session did not actually fetch keeps its
                        # completed status: that is how the cleanup recognises
                        # a pre-existing file it must not delete.
                        item["status"] = (
                            "cancelled"
                            if cancelled and fetched
                            else "completed"
                        )

                    if cancelled:
                        break

                    write_log(
                        level="INFO",
                        component=COMPONENT,
                        action="download_complete",
                        message=(
                            f"Download {index + 1} of {total_files} completed"
                        ),
                        details={
                            "file_index": index + 1,
                            "total_files": total_files,
                            "filename": filename,
                            "size": item["downloaded"],
                        },
                    )

                except DownloadSkipped:
                    self._mark_skipped(index)
                    self._skip_requested = False

                except DownloadCancelled:
                    if (
                        self._skip_requested
                        and not self._cancellation.cancelled
                    ):
                        self._mark_skipped(index)
                        self._skip_requested = False
                    else:
                        with self._lock:
                            item["status"] = "cancelled"
                        write_log(
                            level="WARNING",
                            component=COMPONENT,
                            action="download_cancelled",
                            message=(
                                f"Download {index + 1} of {total_files} "
                                f"cancelled"
                            ),
                            details={
                                "file_index": index + 1,
                                "filename": filename,
                            },
                        )
                        break

                except DownloadError as error:
                    with self._lock:
                        item["status"] = "failed"
                        item["error"] = str(error)

                    write_log(
                        level="ERROR",
                        component=COMPONENT,
                        action="download_failed",
                        message=(
                            f"Download {index + 1} of {total_files} failed"
                        ),
                        details={
                            "file_index": index + 1,
                            "total_files": total_files,
                            "filename": filename,
                            "error": str(error),
                        },
                    )

                except Exception as error:
                    with self._lock:
                        item["status"] = "failed"
                        item["error"] = str(error)

                    write_log(
                        level="ERROR",
                        component=COMPONENT,
                        action="download_unexpected_error",
                        message="Unexpected downloader error",
                        details={
                            "file_index": index + 1,
                            "filename": filename,
                            "error": str(error),
                        },
                    )

                finally:
                    with self._lock:
                        self._current_downloader = None

            # Queue finished. Anything still waiting when a cancellation
            # stopped the loop is marked cancelled, so no item is left claiming
            # it is about to run.
            if self._cancellation.cancelled:
                with self._lock:
                    for item in self._queue:
                        if item["status"] not in FINAL_STATUSES:
                            item["status"] = "cancelled"
                    cleanup = self._cleanup_on_cancel

                write_log(
                    level="WARNING",
                    component=COMPONENT,
                    action="manager_cancel",
                    message="Download manager cancelled",
                )
                if cleanup:
                    self._cleanup_cancelled()
            else:
                write_log(
                    level="INFO",
                    component=COMPONENT,
                    action="manager_complete",
                    message="Download queue completed",
                    details={
                        "total_files": len(self._queue),
                    },
                )

        finally:
            with self._lock:
                self._running = False
                self._paused = False
                self._current_downloader = None
            # The listener polls this to know when to exit, so it is set
            # whichever way the queue ended.
            self._keyboard_stop.set()

    # ========================================================
    # Status Events
    # ========================================================

    def _status_event(
        self,
        index: int,
        status: str,
        details: dict | None,
    ) -> None:
        """Handle status change callbacks from the active downloader.

        A callback can arrive after the session was cancelled — the downloader
        stops at its next chunk boundary, and an event already in flight lands
        after that. Those are dropped rather than applied, so a cancelled item
        cannot be moved back to a live status.

        Args:
            index: Queue item index.
            status: Status event string.
            details: Metadata associated with the event.
        """
        with self._lock:
            if self._cancellation.cancelled:
                return

            item = self._queue[index]

            if status == "connecting":
                item["status"] = "connecting"

            elif status == "downloading":
                item["status"] = "downloading"
                if details:
                    item["total"] = details.get("total")

            elif status == "paused":
                item["status"] = "paused"

            elif status == "retrying":
                item["status"] = "retrying"

            filename = item["filename"]

        if status == "connecting":
            write_log(
                level="INFO",
                component=COMPONENT,
                action="connecting",
                message=f"Connecting for download {index + 1}",
                details={
                    "file_index": index + 1,
                    "filename": filename,
                    **(details if details else {}),
                },
            )

        elif status == "connected":
            write_log(
                level="INFO",
                component=COMPONENT,
                action="connected",
                message=f"Server connected for download {index + 1}",
                details={
                    "file_index": index + 1,
                    "filename": filename,
                    **(details if details else {}),
                },
            )

        elif status == "paused":
            write_log(
                level="INFO",
                component=COMPONENT,
                action="download_paused",
                message=f"Download {index + 1} paused",
                details={
                    "file_index": index + 1,
                    "filename": filename,
                },
            )

        elif status == "retrying":
            write_log(
                level="WARNING",
                component=COMPONENT,
                action="download_retry",
                message=f"Retrying download {index + 1}",
                details={
                    "file_index": index + 1,
                    "filename": filename,
                    "attempt": details.get("attempt", 0) if details else 0,
                    "max_retries": (
                        details.get("max_retries", 0) if details else 0
                    ),
                    "retry_in": details.get("retry_in", 0) if details else 0,
                    "error": details.get("error") if details else None,
                },
            )

    # ========================================================
    # Progress
    # ========================================================

    def _progress(
        self,
        index: int,
        downloaded: int,
        total: int | None,
    ) -> None:
        """Update downloaded bytes and the item's transfer speed.

        Progress from a cancelled session is dropped: the bytes it reports are
        about to be deleted, and recording them would make a cancelled item
        look like it was still transferring.

        Args:
            index: Queue item index.
            downloaded: Bytes downloaded so far.
            total: Total expected bytes or None.
        """
        with self._lock:
            if self._cancellation.cancelled:
                return

            item = self._queue[index]
            item["downloaded"] = downloaded
            item["total"] = total

            now = time.monotonic()
            progress_state = item.get("_progress_state")

            if progress_state is None:
                progress_state = {
                    "start": now,
                    "speed": 0.0,
                }
                item["_progress_state"] = progress_state

            elapsed = now - progress_state["start"]
            if elapsed > 0:
                progress_state["speed"] = downloaded / elapsed

            # Mirrored onto the item itself so `get_status`, which hides the
            # underscore-prefixed bookkeeping, can report it.
            item["speed"] = progress_state["speed"]

    # ========================================================
    # Skip
    # ========================================================

    def _mark_skipped(
        self,
        index: int,
    ) -> None:
        """Mark a queue item as skipped and log the event.

        Args:
            index: Queue item index.
        """
        with self._lock:
            item = self._queue[index]
            item["status"] = "skipped"
            filename = item["filename"]

        write_log(
            level="INFO",
            component=COMPONENT,
            action="download_skip",
            message=f"Download {index + 1} skipped",
            details={
                "file_index": index + 1,
                "filename": filename,
            },
        )

    # ========================================================
    # Keyboard
    # ========================================================

    def _keyboard_listener(self) -> None:
        """Listen for interactive console keybindings on Windows.

        Exits as soon as the worker signals that the queue has ended, so a
        cancelled session leaves no thread behind reading the console.
        """
        if os.name != "nt":
            return

        import msvcrt

        while not self._keyboard_stop.is_set():
            if not msvcrt.kbhit():
                # Waits on the stop event rather than sleeping, so the thread
                # exits the moment the queue ends instead of up to 100ms later.
                self._keyboard_stop.wait(0.1)
                continue

            key = msvcrt.getwch().lower()
            if key == "p":
                if self._paused:
                    self.resume()
                else:
                    self.pause()
            elif key == "s":
                self.skip()
            elif key in ("c", "q"):
                self.cancel()

    # ========================================================
    # Pause
    # ========================================================

    def pause(self) -> None:
        """Pause the current download and active manager loop.

        Suspends the transfer without ending the task: the queue and the
        partial data are kept, and :meth:`resume` continues the active file
        from where it stopped. To end the task and remove what it produced, use
        :meth:`cancel`. A cancelled session cannot be paused — there is nothing
        left to suspend.
        """
        with self._lock:
            if (
                not self._running
                or self._paused
                or self._cancellation.cancelled
            ):
                return

            self._paused = True
            downloader = self._current_downloader
            file_index = self._current_index + 1

        if downloader:
            downloader.pause()

        write_log(
            level="INFO",
            component=COMPONENT,
            action="pause",
            message="Download manager paused",
            details={
                "file_index": file_index,
            },
        )

    # ========================================================
    # Resume
    # ========================================================

    def resume(self) -> None:
        """Resume a paused download."""
        with self._lock:
            if not self._running or not self._paused:
                return

            if self._cancellation.cancelled:
                # Resuming a cancelled session would restart a transfer whose
                # files have already been deleted.
                return

            self._paused = False
            downloader = self._current_downloader
            file_index = self._current_index + 1

        if downloader:
            downloader.resume()

        write_log(
            level="INFO",
            component=COMPONENT,
            action="resume",
            message="Download manager resumed",
            details={
                "file_index": file_index,
            },
        )

    # ========================================================
    # Skip
    # ========================================================

    def skip(self) -> None:
        """Request skipping the currently active download."""
        with self._lock:
            if not self._running or self._cancellation.cancelled:
                return

            self._skip_requested = True
            downloader = self._current_downloader
            file_index = self._current_index + 1

        if downloader:
            downloader.skip()

        write_log(
            level="WARNING",
            component=COMPONENT,
            action="skip_requested",
            message="Skip requested for current file",
            details={
                "file_index": file_index,
            },
        )

    # ========================================================
    # Cancel
    # ========================================================

    def cancel(
        self,
        reason: str | None = None,
        cleanup: bool = True,
    ) -> None:
        """Cancel the session and, by default, remove everything it produced.

        The active download stops at its next chunk boundary and the rest of
        the queue is abandoned. With ``cleanup`` left on, everything the
        session produced is then removed — partial files, and completed files
        too — so a cancelled download leaves nothing behind but its log entry.
        Pass ``cleanup=False`` to keep what had already finished.

        Either way the session is closed: its queue is emptied and it will not
        accept another file or another start, because the work it was tracking
        no longer exists. Safe to call more than once, and at any stage: before
        the queue was started, mid-transfer, or after it finished.

        To suspend a download and keep the task, use :meth:`pause` instead;
        this ends it.

        Args:
            reason: Optional explanation recorded with the cancellation.
            cleanup: Whether to delete the files this session produced.
        """
        with self._lock:
            if self._cancellation.cancelled:
                # Already cancelled. The first call owns the cleanup, so this
                # one must not repeat it or widen it from keep-files to delete.
                return

            finished = self._is_finished()
            running = self._running
            was_paused = self._paused

            self._closed = True
            self._cleanup_on_cancel = cleanup and not finished
            # Sets the flag, keeps the reason, and wakes the worker and the
            # keyboard listener, all in one call.
            self._cancellation.cancel(reason)
            # A paused transfer is waiting on its pause flag, so it has to be
            # released for the cancellation to reach the chunk loop.
            self._paused = False

            downloader = self._current_downloader
            file_index = self._current_index + 1

            if not running:
                # Nothing is transferring, so no worker will reach the terminal
                # states or the cleanup: both happen here instead.
                for item in self._queue:
                    if item["status"] not in FINAL_STATUSES:
                        item["status"] = "cancelled"

        if downloader:
            # Cancels the downloader outside the lock: it only sets a flag, but
            # the chunk loop calling back into _progress would otherwise
            # contend.
            downloader.cancel()

        if was_paused and downloader:
            downloader.resume()

        self._keyboard_stop.set()

        write_log(
            level="WARNING",
            component=COMPONENT,
            action="cancel",
            message="Download manager cancellation requested",
            details={
                "file_index": file_index,
                "reason": self._cancellation.reason,
                "cleanup": cleanup,
                "was_running": running,
            },
        )

        if not running:
            # A queue that already finished delivered its files, so cancelling
            # it has nothing to undo and must not delete them.
            if cleanup and not finished:
                self._cleanup_cancelled()
            else:
                self._cleanup_done = True

    def close(self, reason: str | None = None) -> None:
        """End the session without deleting what it downloaded.

        Bookkeeping counterpart to :meth:`cancel`: the transfer is stopped and
        the session is closed, but the files it produced are left on disk.

        Args:
            reason: Optional explanation recorded with the closure.
        """
        self.cancel(reason=reason or "Session closed", cleanup=False)

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        """Block until the worker thread has exited.

        A cancellation returns as soon as it has been signalled, so a caller
        that needs the session to be genuinely idle — before reporting the
        cleanup done, or before starting a replacement — waits here.

        Args:
            timeout: Seconds to wait at most; None waits indefinitely.

        Returns:
            bool: True when no worker is running any more.
        """
        with self._lock:
            worker = self._worker_thread

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)

        with self._lock:
            return not self._running

    def _is_finished(self) -> bool:
        """Report whether the queue ran to completion.

        Distinguishes a queue that finished from one that was never started or
        was abandoned part-way: cancelling the first has nothing to undo, while
        the others may have partial files to remove. Called with the lock held.

        Returns:
            bool: True when every queued item reached a final state and none is
            still waiting.
        """
        if not self._queue:
            return False

        return all(
            item["status"] in FINAL_STATUSES
            for item in self._queue
        )

    # ========================================================
    # Cancellation cleanup
    # ========================================================

    def _cleanup_cancelled(self) -> None:
        """Delete everything the cancelled session produced.

        A cancelled download is meant to leave no trace, so both the ``.part``
        files of interrupted transfers and the files that had finished are
        removed: the queue was a single unit of work that did not complete,
        and half a set of model files is not a useful thing to leave on disk.
        A file that was already there before the session started is never
        touched — ``Downloader`` reports those as completed without downloading
        them, so they are identified by having no partial file and no bytes
        recorded.

        The download directory itself is removed only when this manager created
        it and it is left empty.
        """
        with self._lock:
            if self._cleanup_done:
                return

            self._cleanup_done = True
            # Copied so the deletions below run without holding the lock: a
            # status poll must not block on the filesystem.
            queue = [dict(item) for item in self._queue]
            reason = self._cancellation.reason or "Cancelled by request"

        removed: list[str] = []
        kept: list[str] = []
        errors: list[str] = []

        for item in queue:
            filename = item["filename"]
            destination = self.download_directory / filename
            partial = Path(str(destination) + ".part")

            # Pre-existing file this session did not fetch: not ours to delete.
            preexisting = (
                item["status"] == "completed"
                and not item.get("downloaded")
                and not partial.exists()
            )

            if preexisting:
                kept.append(filename)
                continue

            for path in (partial, destination):
                try:
                    if path.is_file():
                        path.unlink()
                        removed.append(path.name)
                except OSError as error:
                    errors.append(f"{path.name}: {error}")

        directory_removed = False

        if self._created_directory:
            try:
                self.download_directory.rmdir()
                directory_removed = True
            except OSError:
                # Not empty, or in use: the files this session added are gone,
                # which is what cleanup promised.
                pass

        with self._lock:
            # The bytes recorded against each item described files that no
            # longer exist, so they are cleared along with them.
            for item in self._queue:
                if item["filename"] not in kept:
                    item["downloaded"] = 0
                    item["total"] = None
                    item["speed"] = 0.0
                item.pop("_progress_state", None)

        log_cancelled(
            component=COMPONENT,
            action="manager_cancel",
            message="Download cancelled",
            details={
                "download_directory": str(self.download_directory),
                "files_removed": removed,
                "preexisting_files_kept": kept,
                "directory_removed": directory_removed,
                "cleanup_errors": errors,
                "reason": reason,
            },
        )

    def purge(self) -> None:
        """Discard the session's remaining in-memory state.

        Called once nothing is reading this manager any more — after a
        cancellation has been reported, or when the session is dropped from
        whatever registry held it. The queue and the reference to the last
        downloader are released so no part of the cancelled task is still
        reachable; the cancellation's log entry is the only record left.
        """
        self.cancel(reason="Session purged", cleanup=False)
        self.wait_until_stopped(timeout=WORKER_JOIN_TIMEOUT)

        with self._lock:
            self._queue.clear()
            self._current_downloader = None
            self._current_index = -1
            self._worker_thread = None
            self._keyboard_thread = None

    # ========================================================
    # Status
    # ========================================================

    def get_status(self) -> dict:
        """Get the current manager state and queue details.

        Returns:
            dict: Current manager status including running/paused/cancelled
                states, current index, and download items, each carrying its
                'speed' in bytes per second. The queue entries are copies, so a
                caller can read them while the worker keeps writing.
        """
        with self._lock:
            downloads = [
                {
                    key: value
                    for key, value in item.items()
                    if not key.startswith("_")
                }
                for item in self._queue
            ]

            cancelled = self._cancellation.cancelled

            return {
                "running": self._running,
                "paused": self._paused,
                "cancelled": cancelled,
                "closed": self._closed,
                # Whether the cancellation deleted what the session produced. A
                # close keeps the files, so a caller describing the outcome —
                # or deciding whether a completed file is still on disk — needs
                # to tell the two apart.
                "files_deleted": cancelled and self._cleanup_on_cancel,
                "cancel_reason": self._cancellation.reason,
                "current_index": self._current_index,
                "total_files": len(self._queue),
                "downloads": downloads,
            }

    # ========================================================
    # Wait
    # ========================================================

    def wait(self) -> None:
        """Block until the download worker thread finishes."""
        self.wait_until_stopped()
