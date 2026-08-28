"""Sequential download queue manager with pause, resume, skip, and monitoring."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Iterable
from urllib.parse import urlparse

from core.cancellation import log_cancelled
from core.logging import write_log
from .downloader import (
    DownloadCancelled,
    DownloadError,
    DownloadSkipped,
    Downloader,
)

COMPONENT = "download_manager"
ALLOWED_DOMAINS = {
    "ollama.com",
    "www.ollama.com",
    "huggingface.co",
    "www.huggingface.co",
    "python.org",
    "www.python.org",
}


class DownloadManager:
    """Sequential download manager handling queueing, speed, ETA, and worker threads.

    The manager handles:
        - Download queue
        - Progress tracking and speed/ETA calculation
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
            max_retries: Maximum download retry attempts per file. Defaults to 3.
        """
        self.download_directory = Path(download_directory)
        # Whether this manager created the directory, so a cancellation knows if
        # removing it again is its business.
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
        self._cancelled = False
        self._skip_requested = False

        self._cancel_reason: str | None = None
        self._cleanup_on_cancel = True
        self._cleanup_done = False

        self._worker_thread: threading.Thread | None = None
        self._keyboard_thread: threading.Thread | None = None
        self._progress_lock = threading.Lock()

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
            filename: Optional custom target filename. If not provided, inferred from URL.

        Raises:
            ValueError: If the URL is empty.
            PermissionError: If the domain is not in the whitelist.
        """
        if not url:
            raise ValueError("URL cannot be empty.")
        self._verify_download_source(url)

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
            "error": None,
        }

        self._queue.append(item)

        write_log(
            level="INFO",
            component=COMPONENT,
            action="queue_add",
            message="File added to download queue",
            details={
                "filename": filename,
                "queue_position": len(self._queue),
                "total_files": len(self._queue),
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

    # ========================================================
    # Start
    # ========================================================

    def start(self) -> None:
        """Start background worker threads to process the download queue.

        Raises:
            RuntimeError: If the download queue is empty.
        """
        if self._running:
            return

        if not self._queue:
            raise RuntimeError("Download queue is empty.")

        self._running = True
        self._cancelled = False

        write_log(
            level="INFO",
            component=COMPONENT,
            action="manager_start",
            message="Download manager started",
            details={
                "total_files": len(self._queue),
            },
        )

        self._worker_thread = threading.Thread(
            target=self._worker,
            name="DownloadWorker",
            daemon=True,
        )
        self._worker_thread.start()

        if os.name == "nt":
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_listener,
                name="DownloadKeyboard",
                daemon=True,
            )
            self._keyboard_thread.start()

    # ========================================================
    # Worker
    # ========================================================

    def _worker(self) -> None:
        """Background worker loop executing sequential item downloads."""
        try:
            for index, item in enumerate(self._queue):
                self._current_index = index

                if self._cancelled:
                    break

                if item["status"] == "skipped":
                    continue

                total_files = len(self._queue)
                filename = item["filename"]

                item["status"] = "downloading"
                item["error"] = None

                write_log(
                    level="INFO",
                    component=COMPONENT,
                    action="download_start",
                    message=f"Starting download {index + 1} of {total_files}",
                    details={
                        "file_index": index + 1,
                        "total_files": total_files,
                        "filename": filename,
                        "url": item["url"],
                    },
                )

                destination = self.download_directory / filename

                downloader = Downloader(
                    url=item["url"],
                    destination=destination,
                    max_retries=self.max_retries,
                )
                self._current_downloader = downloader

                try:
                    downloader.download(
                        progress_callback=(
                            lambda downloaded, total, i=index: self._progress(
                                i, downloaded, total
                            )
                        ),
                        status_callback=(
                            lambda status, details, i=index: self._status_event(
                                i, status, details
                            )
                        ),
                    )

                    # Check if skip was requested
                    if self._skip_requested:
                        self._mark_skipped(index)
                        self._skip_requested = False
                        continue

                    # Completed successfully
                    item["status"] = "completed"

                    write_log(
                        level="INFO",
                        component=COMPONENT,
                        action="download_complete",
                        message=f"Download {index + 1} of {total_files} completed",
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
                    if self._skip_requested:
                        self._mark_skipped(index)
                        self._skip_requested = False
                    else:
                        item["status"] = "cancelled"
                        write_log(
                            level="WARNING",
                            component=COMPONENT,
                            action="download_cancelled",
                            message=f"Download {index + 1} of {total_files} cancelled",
                            details={
                                "file_index": index + 1,
                                "filename": filename,
                            },
                        )
                        break

                except DownloadError as error:
                    item["status"] = "failed"
                    item["error"] = str(error)

                    write_log(
                        level="ERROR",
                        component=COMPONENT,
                        action="download_failed",
                        message=f"Download {index + 1} of {total_files} failed",
                        details={
                            "file_index": index + 1,
                            "total_files": total_files,
                            "filename": filename,
                            "error": str(error),
                        },
                    )

                except Exception as error:
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
                    self._current_downloader = None

            # Queue finished
            if self._cancelled:
                write_log(
                    level="WARNING",
                    component=COMPONENT,
                    action="manager_cancel",
                    message="Download manager cancelled",
                )
                if self._cleanup_on_cancel:
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
            self._running = False

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

        Args:
            index: Queue item index.
            status: Status event string.
            details: Metadata associated with the event.
        """
        item = self._queue[index]

        if status == "connecting":
            item["status"] = "connecting"
            write_log(
                level="INFO",
                component=COMPONENT,
                action="connecting",
                message=f"Connecting for download {index + 1}",
                details={
                    "file_index": index + 1,
                    "filename": item["filename"],
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
                    "filename": item["filename"],
                    **(details if details else {}),
                },
            )

        elif status == "downloading":
            item["status"] = "downloading"
            if details:
                item["total"] = details.get("total")

        elif status == "paused":
            item["status"] = "paused"
            write_log(
                level="INFO",
                component=COMPONENT,
                action="download_paused",
                message=f"Download {index + 1} paused",
                details={
                    "file_index": index + 1,
                    "filename": item["filename"],
                },
            )

        elif status == "retrying":
            item["status"] = "retrying"
            attempt = details.get("attempt", 0) if details else 0
            max_retries = details.get("max_retries", 0) if details else 0
            retry_in = details.get("retry_in", 0) if details else 0
            error = details.get("error") if details else None

            write_log(
                level="WARNING",
                component=COMPONENT,
                action="download_retry",
                message=f"Retrying download {index + 1}",
                details={
                    "file_index": index + 1,
                    "filename": item["filename"],
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "retry_in": retry_in,
                    "error": error,
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
        """Update downloaded bytes, calculate transfer speed, and estimate ETA.

        Args:
            index: Queue item index.
            downloaded: Bytes downloaded so far.
            total: Total expected bytes or None.
        """
        item = self._queue[index]
        item["downloaded"] = downloaded
        item["total"] = total

        # Speed calculation
        now = time.monotonic()
        progress_state = item.get("_progress_state")

        if progress_state is None:
            progress_state = {
                "start": now,
                "last_time": now,
                "last_bytes": downloaded,
                "speed": 0.0,
            }
            item["_progress_state"] = progress_state

        elapsed = now - progress_state["start"]
        if elapsed > 0:
            speed = downloaded / elapsed
            progress_state["speed"] = speed

        speed = progress_state["speed"]

        # Formatting metrics
        if total and total > 0:
            remaining = total - downloaded
            if speed > 0:
                eta_seconds = remaining / speed
                _ = self._format_time(eta_seconds)

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
        """Listen for interactive console keybindings on Windows."""
        if os.name != "nt":
            return

        import msvcrt

        while self._running:
            if not msvcrt.kbhit():
                time.sleep(0.1)
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

        Suspends the transfer without ending the task: the queue and the partial
        data are kept, and :meth:`resume` continues the active file from where it
        stopped. To end the task and remove what it produced, use :meth:`cancel`.
        """
        if not self._running or self._paused:
            return

        self._paused = True
        if self._current_downloader:
            self._current_downloader.pause()

        write_log(
            level="INFO",
            component=COMPONENT,
            action="pause",
            message="Download manager paused",
            details={
                "file_index": self._current_index + 1,
            },
        )

    # ========================================================
    # Resume
    # ========================================================

    def resume(self) -> None:
        """Resume a paused download."""
        if not self._running or not self._paused:
            return

        self._paused = False
        if self._current_downloader:
            self._current_downloader.resume()

        write_log(
            level="INFO",
            component=COMPONENT,
            action="resume",
            message="Download manager resumed",
            details={
                "file_index": self._current_index + 1,
            },
        )

    # ========================================================
    # Skip
    # ========================================================

    def skip(self) -> None:
        """Request skipping the currently active download."""
        if not self._running:
            return

        self._skip_requested = True
        if self._current_downloader:
            self._current_downloader.skip()

        write_log(
            level="WARNING",
            component=COMPONENT,
            action="skip_requested",
            message="Skip requested for current file",
            details={
                "file_index": self._current_index + 1,
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
        """Cancel all pending downloads in the queue.

        The active download stops at its next chunk boundary and the rest of the
        queue is abandoned. With ``cleanup`` left on, everything the session
        produced is then removed — partial files, and completed files too — so a
        cancelled download leaves nothing behind but its log entry. Pass
        ``cleanup=False`` to keep what had already finished, which is what the
        older behaviour did.

        To suspend a download and keep the task, use :meth:`pause` instead; this
        ends it.

        Args:
            reason: Optional explanation recorded with the cancellation.
            cleanup: Whether to delete the files this session produced.
        """
        if not self._running:
            # Nothing is downloading. A queue that was started and abandoned
            # still has files to clean up, but one that already finished has
            # nothing to cancel — doing so must not delete what it delivered.
            if self._is_finished():
                return

            self._cancelled = True
            self._cancel_reason = reason or "Cancelled by request"

            if cleanup:
                self._cleanup_cancelled()

            return

        self._cancelled = True
        self._cancel_reason = reason or "Cancelled by request"
        self._cleanup_on_cancel = cleanup

        if self._current_downloader:
            self._current_downloader.cancel()

        write_log(
            level="WARNING",
            component=COMPONENT,
            action="cancel",
            message="Download manager cancellation requested",
            details={
                "file_index": self._current_index + 1,
                "reason": self._cancel_reason,
                "cleanup": cleanup,
            },
        )

    def _is_finished(self) -> bool:
        """Report whether the queue ran to completion.

        Distinguishes a queue that finished from one that was never started or was
        abandoned part-way: cancelling the first has nothing to undo, while the
        others may have partial files to remove.

        Returns:
            bool: True when every queued item reached a final state and none is
            still waiting.
        """
        if not self._queue:
            return False

        return all(
            item["status"] in ("completed", "skipped", "failed")
            for item in self._queue
        )

    # ========================================================
    # Cancellation cleanup
    # ========================================================

    def _cleanup_cancelled(self) -> None:
        """Delete everything the cancelled session produced.

        A cancelled download is meant to leave no trace, so both the ``.part``
        files of interrupted transfers and the files that had finished are
        removed: the queue was a single unit of work that did not complete, and
        half a set of model files is not a useful thing to leave on disk. A file
        that was already there before the session started is never touched —
        ``Downloader`` reports those as completed without downloading them, so
        they are identified by having no partial file and no bytes recorded.

        The download directory itself is removed only when this manager created
        it and it is left empty.
        """
        if self._cleanup_done:
            return

        self._cleanup_done = True

        removed: list[str] = []
        kept: list[str] = []
        errors: list[str] = []

        for item in self._queue:
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
                "reason": self._cancel_reason or "Cancelled by request",
            },
        )

    # ========================================================
    # Status
    # ========================================================

    def get_status(self) -> dict:
        """Get the current manager state and queue details.

        Returns:
            dict: Current manager status including running/paused/cancelled states,
                current index, and download items.
        """
        downloads = []
        for item in self._queue:
            clean_item = {
                key: value
                for key, value in item.items()
                if not key.startswith("_")
            }
            downloads.append(clean_item)

        return {
            "running": self._running,
            "paused": self._paused,
            "cancelled": self._cancelled,
            "cancel_reason": self._cancel_reason,
            "current_index": self._current_index,
            "total_files": len(self._queue),
            "downloads": downloads,
        }

    # ========================================================
    # Wait
    # ========================================================

    def wait(self) -> None:
        """Block until the download worker thread finishes."""
        if self._worker_thread:
            self._worker_thread.join()

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS format.

        Args:
            seconds: Duration in seconds.

        Returns:
            str: Formatted time string.
        """
        if seconds < 0:
            return "--:--"

        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return f"{minutes:02d}:{seconds:02d}"
