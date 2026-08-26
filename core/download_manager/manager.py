from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable

from core.logging import write_log

from .downloader import (
    DownloadCancelled,
    DownloadError,
    DownloadSkipped,
    Downloader,
)


COMPONENT = "download_manager"


class DownloadManager:
    """
    Sequential download manager.

    The manager handles:

        - Download queue
        - Progress display
        - Speed
        - ETA
        - Retry
        - Pause / Resume
        - Skip
        - Cancel
        - Keyboard shortcuts
        - Logging
    """

    def __init__(
        self,
        download_directory: str | Path = "data/downloads",
        max_retries: int = 3,
    ):

        self.download_directory = Path(
            download_directory
        )

        self.download_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.max_retries = max_retries

        self._queue: list[dict] = []

        self._current_downloader: (
            Downloader | None
        ) = None

        self._current_index = -1

        self._running = False
        self._paused = False
        self._cancelled = False
        self._skip_requested = False

        self._worker_thread: (
            threading.Thread | None
        ) = None

        self._keyboard_thread: (
            threading.Thread | None
        ) = None

        self._progress_lock = threading.Lock()

    # ========================================================
    # Queue
    # ========================================================

    def add(
        self,
        url: str,
        filename: str | None = None,
    ) -> None:
        """
        Add a file to the download queue.
        """

        if not url:
            raise ValueError(
                "URL cannot be empty."
            )

        if not filename:

            filename = (
                Path(
                    url.split("?")[0]
                ).name
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
                "queue_position": len(
                    self._queue
                ),
                "total_files": len(
                    self._queue
                ),
                "url": url,
            },
        )

    def add_many(
        self,
        urls: Iterable[str],
    ) -> None:
        """
        Add multiple URLs sequentially.
        """

        for url in urls:
            self.add(url)

    # ========================================================
    # Start
    # ========================================================

    def start(self) -> None:
        """
        Start the download manager.
        """

        if self._running:
            return

        if not self._queue:

            raise RuntimeError(
                "Download queue is empty."
            )

        self._running = True
        self._cancelled = False

        self._show_keyboard_help()

        write_log(
            level="INFO",
            component=COMPONENT,
            action="manager_start",
            message="Download manager started",
            details={
                "total_files": len(
                    self._queue
                ),
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
    # Help
    # ========================================================

    @staticmethod
    def _show_keyboard_help() -> None:

        print()
        print("=" * 68)
        print("                       DOWNLOAD MANAGER")
        print("=" * 68)
        print()
        print(" Keyboard shortcuts")
        print()
        print("   P  Pause / Resume current download")
        print("   S  Skip current file")
        print("   C  Cancel download manager")
        print("   Q  Quit / Cancel download manager")
        print()
        print(" Downloads are processed sequentially.")
        print(" A .part file is used for interrupted downloads.")
        print()
        print("=" * 68)
        print()

    # ========================================================
    # Worker
    # ========================================================

    def _worker(self) -> None:

        try:

            for index, item in enumerate(
                self._queue
            ):

                self._current_index = index

                if self._cancelled:
                    break

                if item["status"] == "skipped":
                    continue

                total_files = len(
                    self._queue
                )

                filename = item[
                    "filename"
                ]

                item["status"] = "downloading"
                item["error"] = None

                print()
                print(
                    f"[{index + 1}/{total_files}] "
                    f"Preparing: {filename}"
                )

                write_log(
                    level="INFO",
                    component=COMPONENT,
                    action="download_start",
                    message=(
                        f"Starting download "
                        f"{index + 1} of "
                        f"{total_files}"
                    ),
                    details={
                        "file_index": index + 1,
                        "total_files": total_files,
                        "filename": filename,
                        "url": item["url"],
                    },
                )

                destination = (
                    self.download_directory
                    / filename
                )

                downloader = Downloader(
                    url=item["url"],
                    destination=destination,
                    max_retries=self.max_retries,
                )

                self._current_downloader = (
                    downloader
                )

                try:

                    downloader.download(
                        progress_callback=(
                            lambda downloaded,
                            total,
                            i=index:
                            self._progress(
                                i,
                                downloaded,
                                total,
                            )
                        ),
                        status_callback=(
                            lambda status,
                            details,
                            i=index:
                            self._status_event(
                                i,
                                status,
                                details,
                            )
                        ),
                    )

                    # ------------------------------------------------
                    # Skip was requested
                    # ------------------------------------------------

                    if self._skip_requested:

                        self._mark_skipped(
                            index
                        )

                        self._skip_requested = False

                        continue

                    # ------------------------------------------------
                    # Completed
                    # ------------------------------------------------

                    item["status"] = "completed"

                    print(
                        f"\n[OK] Completed: "
                        f"{filename}"
                    )

                    write_log(
                        level="INFO",
                        component=COMPONENT,
                        action="download_complete",
                        message=(
                            f"Download {index + 1} "
                            f"of {total_files} "
                            f"completed"
                        ),
                        details={
                            "file_index": index + 1,
                            "total_files": total_files,
                            "filename": filename,
                            "size": item[
                                "downloaded"
                            ],
                        },
                    )

                except DownloadSkipped:

                    self._mark_skipped(
                        index
                    )

                    self._skip_requested = False

                except DownloadCancelled:

                    # Cancellation from skip is handled
                    # separately.
                    if self._skip_requested:

                        self._mark_skipped(
                            index
                        )

                        self._skip_requested = False

                    else:

                        item["status"] = (
                            "cancelled"
                        )

                        write_log(
                            level="WARNING",
                            component=COMPONENT,
                            action="download_cancelled",
                            message=(
                                f"Download {index + 1} "
                                f"of {total_files} "
                                f"cancelled"
                            ),
                            details={
                                "file_index": index + 1,
                                "filename": filename,
                            },
                        )

                        break

                except DownloadError as error:

                    item["status"] = "failed"
                    item["error"] = str(error)

                    print(
                        f"\n[ERROR] Failed: "
                        f"{filename}"
                    )

                    write_log(
                        level="ERROR",
                        component=COMPONENT,
                        action="download_failed",
                        message=(
                            f"Download {index + 1} "
                            f"of {total_files} "
                            f"failed"
                        ),
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
                        message=(
                            "Unexpected downloader error"
                        ),
                        details={
                            "file_index": index + 1,
                            "filename": filename,
                            "error": str(error),
                        },
                    )

                finally:

                    self._current_downloader = None

            # --------------------------------------------------------
            # Queue finished
            # --------------------------------------------------------

            if self._cancelled:

                write_log(
                    level="WARNING",
                    component=COMPONENT,
                    action="manager_cancel",
                    message=(
                        "Download manager cancelled"
                    ),
                )

            else:

                write_log(
                    level="INFO",
                    component=COMPONENT,
                    action="manager_complete",
                    message=(
                        "Download queue completed"
                    ),
                    details={
                        "total_files": len(
                            self._queue
                        ),
                    },
                )

        finally:

            self._running = False

            print()
            print("=" * 68)
            print("Download manager finished.")
            print("=" * 68)

    # ========================================================
    # Status Events
    # ========================================================

    def _status_event(
        self,
        index: int,
        status: str,
        details: dict | None,
    ) -> None:

        item = self._queue[index]

        if status == "connecting":

            item["status"] = "connecting"

            print(
                "  Connecting to server..."
            )

            write_log(
                level="INFO",
                component=COMPONENT,
                action="connecting",
                message=(
                    f"Connecting for "
                    f"download {index + 1}"
                ),
                details={
                    "file_index": index + 1,
                    "filename": item["filename"],
                    **(
                        details
                        if details
                        else {}
                    ),
                },
            )

        elif status == "connected":

            print(
                "  Server connected."
            )

            write_log(
                level="INFO",
                component=COMPONENT,
                action="connected",
                message=(
                    f"Server connected for "
                    f"download {index + 1}"
                ),
                details={
                    "file_index": index + 1,
                    "filename": item["filename"],
                    **(
                        details
                        if details
                        else {}
                    ),
                },
            )

        elif status == "downloading":

            item["status"] = "downloading"

            if details:

                item["total"] = details.get(
                    "total"
                )

        elif status == "paused":

            item["status"] = "paused"

            print(
                "\n  [PAUSED]"
            )

            write_log(
                level="INFO",
                component=COMPONENT,
                action="download_paused",
                message=(
                    f"Download {index + 1} "
                    f"paused"
                ),
                details={
                    "file_index": index + 1,
                    "filename": item["filename"],
                },
            )

        elif status == "retrying":

            item["status"] = "retrying"

            attempt = (
                details.get("attempt", 0)
                if details
                else 0
            )

            max_retries = (
                details.get("max_retries", 0)
                if details
                else 0
            )

            retry_in = (
                details.get("retry_in", 0)
                if details
                else 0
            )

            error = (
                details.get("error")
                if details
                else None
            )

            print()
            print(
                f"  [RETRY] "
                f"Attempt {attempt}/{max_retries}"
            )

            print(
                f"  Waiting {retry_in}s..."
            )

            write_log(
                level="WARNING",
                component=COMPONENT,
                action="download_retry",
                message=(
                    f"Retrying download "
                    f"{index + 1}"
                ),
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

        item = self._queue[index]

        item["downloaded"] = downloaded
        item["total"] = total

        # ----------------------------------------------------
        # Speed calculation
        # ----------------------------------------------------

        now = time.monotonic()

        progress_state = item.get(
            "_progress_state"
        )

        if progress_state is None:

            progress_state = {
                "start": now,
                "last_time": now,
                "last_bytes": downloaded,
                "speed": 0.0,
            }

            item["_progress_state"] = (
                progress_state
            )

        elapsed = (
            now
            - progress_state["start"]
        )

        if elapsed > 0:

            speed = (
                downloaded
                / elapsed
            )

            progress_state["speed"] = speed

        speed = progress_state["speed"]

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        if total and total > 0:

            percentage = (
                downloaded
                / total
                * 100
            )

            bar_width = 40

            filled = int(
                bar_width
                * percentage
                / 100
            )

            bar = (
                "#"
                * filled
                + "-"
                * (
                    bar_width
                    - filled
                )
            )

            downloaded_mb = (
                downloaded
                / 1024
                / 1024
            )

            total_mb = (
                total
                / 1024
                / 1024
            )

            speed_mb = (
                speed
                / 1024
                / 1024
            )

            # ------------------------------------------------
            # ETA
            # ------------------------------------------------

            remaining = (
                total - downloaded
            )

            if speed > 0:

                eta_seconds = (
                    remaining
                    / speed
                )

                eta = self._format_time(
                    eta_seconds
                )

            else:

                eta = "--:--"

            with self._progress_lock:

                print(
                    f"\r[{bar}] "
                    f"{percentage:6.2f}% "
                    f"{downloaded_mb:8.1f}/"
                    f"{total_mb:8.1f} MB "
                    f"{speed_mb:6.2f} MB/s "
                    f"ETA {eta}",
                    end="",
                    flush=True,
                )

        else:

            downloaded_mb = (
                downloaded
                / 1024
                / 1024
            )

            speed_mb = (
                speed
                / 1024
                / 1024
            )

            with self._progress_lock:

                print(
                    f"\rDownloaded: "
                    f"{downloaded_mb:.1f} MB "
                    f"at "
                    f"{speed_mb:.2f} MB/s",
                    end="",
                    flush=True,
                )

    # ========================================================
    # Skip
    # ========================================================

    def _mark_skipped(
        self,
        index: int,
    ) -> None:

        item = self._queue[index]

        item["status"] = "skipped"

        filename = item[
            "filename"
        ]

        print(
            f"\n[SKIPPED] {filename}"
        )

        write_log(
            level="INFO",
            component=COMPONENT,
            action="download_skip",
            message=(
                f"Download {index + 1} "
                f"skipped"
            ),
            details={
                "file_index": index + 1,
                "filename": filename,
            },
        )

    # ========================================================
    # Keyboard
    # ========================================================

    def _keyboard_listener(self) -> None:

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

            elif key == "c":

                self.cancel()

            elif key == "q":

                self.cancel()

    # ========================================================
    # Pause
    # ========================================================

    def pause(self) -> None:

        if not self._running:
            return

        if self._paused:
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
                "file_index":
                    self._current_index + 1,
            },
        )

        print(
            "\n[PAUSED] "
            "Press P to resume."
        )

    # ========================================================
    # Resume
    # ========================================================

    def resume(self) -> None:

        if not self._running:
            return

        if not self._paused:
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
                "file_index":
                    self._current_index + 1,
            },
        )

        print(
            "\n[RESUMED]"
        )

    # ========================================================
    # Skip
    # ========================================================

    def skip(self) -> None:

        if not self._running:
            return

        self._skip_requested = True

        if self._current_downloader:

            self._current_downloader.skip()

        write_log(
            level="WARNING",
            component=COMPONENT,
            action="skip_requested",
            message=(
                "Skip requested for current file"
            ),
            details={
                "file_index":
                    self._current_index + 1,
            },
        )

    # ========================================================
    # Cancel
    # ========================================================

    def cancel(self) -> None:

        if not self._running:
            return

        self._cancelled = True

        if self._current_downloader:

            self._current_downloader.cancel()

        write_log(
            level="WARNING",
            component=COMPONENT,
            action="cancel",
            message=(
                "Download manager cancellation "
                "requested"
            ),
            details={
                "file_index":
                    self._current_index + 1,
            },
        )

        print(
            "\n[CANCEL] "
            "Stopping download manager..."
        )

    # ========================================================
    # Status
    # ========================================================

    def get_status(self) -> dict:

        downloads = []

        for item in self._queue:

            clean_item = {
                key: value
                for key, value in item.items()
                if not key.startswith("_")
            }

            downloads.append(
                clean_item
            )

        return {
            "running": self._running,
            "paused": self._paused,
            "cancelled": self._cancelled,
            "current_index":
                self._current_index,
            "total_files":
                len(self._queue),
            "downloads": downloads,
        }

    # ========================================================
    # Wait
    # ========================================================

    def wait(self) -> None:
        """
        Wait until the download worker finishes.
        """

        if self._worker_thread:

            self._worker_thread.join()

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def _format_time(
        seconds: float,
    ) -> str:

        if seconds < 0:
            return "--:--"

        seconds = int(seconds)

        hours, remainder = divmod(
            seconds,
            3600,
        )

        minutes, seconds = divmod(
            remainder,
            60,
        )

        if hours:

            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )