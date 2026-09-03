"""Resumable and retryable HTTP/HTTPS file downloader."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable
import urllib.error
import urllib.request

from .sources import verify_download_source


class DownloadError(Exception):
    """Raised when a download cannot be completed."""


class DownloadConflict(DownloadError):
    """Raised when the destination file already exists.

    A file already at the destination is never overwritten and never taken as
    proof that this URL was downloaded: two repositories routinely publish the
    same file name — ``model.safetensors`` is the usual one — so accepting the
    existing file would report a download that never happened and leave the
    wrong bytes under the requested name. The caller chooses a free name and
    queues it again; ``DownloadManager`` does that automatically.

    A subclass of :class:`DownloadError` so a caller that already handles a
    failed download handles this too, and it is deliberately not retried: the
    file will still be there on the next attempt.
    """


class DownloadCancelled(Exception):
    """Raised when the current download is cancelled."""


class DownloadSkipped(Exception):
    """Raised when the current download is skipped."""


class Downloader:
    """Reliable resumable HTTP/HTTPS downloader.

    Responsibilities:
        - Connect to remote server
        - Download a single file
        - Resume interrupted downloads
        - Retry failed connections
        - Report progress
        - Support pause/resume
        - Support cancellation
        - Support skipping
    """

    def __init__(
        self,
        url: str,
        destination: Path,
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
        connect_timeout: int = 15,
        read_timeout: int = 30,
        retry_delay: int = 3,
    ) -> None:
        """Initialize the Downloader instance.

        Args:
            url: The HTTP/HTTPS download URL.
            destination: Local file path where the completed download is saved.
            chunk_size: Stream read chunk size in bytes. Defaults to 1 MB.
            max_retries: Maximum number of download attempts. Defaults to 3.
            connect_timeout: Seconds to wait for the connection. Defaults
                to 15.
            read_timeout: Seconds to wait on socket reads. Defaults to 30.
            retry_delay: Delay multiplier in seconds between retries. Defaults
                to 3.
        """
        self.url = url
        self.destination = destination
        self.partial_file = Path(str(destination) + ".part")
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.retry_delay = retry_delay

        self._paused = False
        self._cancelled = False
        self._skipped = False

    def _verify_download_source(self) -> None:
        """Validate that the target URL scheme and host are allowed.

        Delegates to :func:`MSHCore.download_manager.sources.verify_download_source`
        so this check and the manager's are the same comparison: the host comes
        from ``urlparse(...).hostname`` — no port, no userinfo — the scheme must
        be http or https, and Hugging Face's CDN subdomains are accepted.

        Raises:
            DownloadSourceRejected: If the scheme or the domain is not allowed.
                A ``PermissionError``, as before, whose message names the
                rejected domain and lists the allowed ones.
        """
        verify_download_source(self.url)

    # ========================================================
    # Control
    # ========================================================

    def pause(self) -> None:
        """Pause the active download loop."""
        self._paused = True

    def resume(self) -> None:
        """Resume the paused download loop."""
        self._paused = False

    def cancel(self) -> None:
        """Cancel the active download."""
        self._cancelled = True

    def skip(self) -> None:
        """Skip the active download and mark it as cancelled."""
        self._skipped = True
        self._cancelled = True

    # ========================================================
    # State
    # ========================================================

    @property
    def paused(self) -> bool:
        """bool: Whether the download is currently paused."""
        return self._paused

    @property
    def cancelled(self) -> bool:
        """bool: Whether the download was cancelled."""
        return self._cancelled

    @property
    def skipped(self) -> bool:
        """bool: Whether the download was skipped."""
        return self._skipped

    # ========================================================
    # Main download
    # ========================================================

    def download(
        self,
        progress_callback: Callable[[int, int | None], None] | None = None,
        status_callback: Callable[[str, dict | None], None] | None = None,
    ) -> None:
        """Download the remote file with retry, resume, and progress tracking.

        Args:
            progress_callback: Optional callable receiving (downloaded_bytes,
                total_bytes).
            status_callback: Optional callable receiving (status_string,
                details_dict).

        Raises:
            PermissionError: If the download URL domain is untrusted.
            DownloadConflict: If a file is already at the destination. Nothing
                is written and nothing is overwritten; the caller picks a free
                name.
            DownloadCancelled: If the download is cancelled.
            DownloadSkipped: If the download is skipped.
            DownloadError: If all retry attempts fail.
        """
        self._verify_download_source()

        self.destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # A file already at the destination is a conflict, not a completed
        # download. Reporting "completed" here was silently standing in for a
        # transfer that never ran, so a second repository's model.safetensors
        # resolved to the first one's bytes.
        if self.destination.exists():
            existing_size = self.destination.stat().st_size

            self._emit_status(
                status_callback,
                "conflict",
                {
                    "reason": "destination_exists",
                    "destination": str(self.destination),
                    "size": existing_size,
                },
            )

            raise DownloadConflict(
                f"A file is already at the destination and was not "
                f"overwritten: {self.destination} ({existing_size} bytes). "
                f"Nothing was downloaded from {self.url}. Choose a different "
                f"filename, or remove the existing file first."
            )

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self._download_once(
                    progress_callback=progress_callback,
                    status_callback=status_callback,
                )
                return
            except (DownloadCancelled, DownloadSkipped, DownloadConflict):
                raise
            except Exception as error:
                last_error = error

                if self.skipped:
                    raise DownloadSkipped()

                if self.cancelled:
                    raise DownloadCancelled()

                if attempt >= self.max_retries:
                    break

                retry_wait = self.retry_delay * attempt
                self._emit_status(
                    status_callback,
                    "retrying",
                    {
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error": str(error),
                        "retry_in": retry_wait,
                    },
                )
                time.sleep(retry_wait)

        raise DownloadError(
            str(last_error) if last_error else "Unknown download error"
        )

    # ========================================================
    # Single attempt
    # ========================================================

    def _download_once(
        self,
        progress_callback: Callable[[int, int | None], None] | None,
        status_callback: Callable[[str, dict | None], None] | None,
    ) -> None:
        """Execute a single download attempt supporting Range-based resume.

        Args:
            progress_callback: Optional callback for byte progress updates.
            status_callback: Optional callback for status transition events.

        Raises:
            DownloadCancelled: If cancellation occurs during download.
            DownloadSkipped: If skip is requested during download.
            DownloadError: If a network error occurs or an incomplete payload
                is received.
        """
        existing_size = 0
        if self.partial_file.exists():
            existing_size = self.partial_file.stat().st_size

        # Connection
        self._emit_status(
            status_callback,
            "connecting",
            {"resume_from": existing_size},
        )

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        request = urllib.request.Request(
            self.url,
            headers=headers,
            method="GET",
        )

        try:
            response = urllib.request.urlopen(
                request,
                timeout=self.connect_timeout,
            )
        except urllib.error.HTTPError as error:
            # HTTP 416 means the requested range is no longer valid. If the
            # server says the file is already complete, treat it as completed.
            if error.code == 416:
                remote_size = self._get_size_from_416(error)
                if remote_size is not None and existing_size >= remote_size:
                    self.partial_file.replace(self.destination)
                    self._emit_status(
                        status_callback,
                        "completed",
                        {"reason": "range_416_file_complete"},
                    )
                    return
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise DownloadError(f"Connection failed: {error}") from error

        # The connect timeout bounded the handshake; the chunk reads below get
        # their own, longer budget so a slow mirror is not mistaken for a dead
        # connection.
        self._apply_read_timeout(response)

        self._emit_status(
            status_callback,
            "connected",
            {"http_status": getattr(response, "status", None)},
        )

        # Determine response behavior
        response_status = getattr(response, "status", None)
        content_length = response.headers.get("Content-Length")
        if content_length:
            content_length = int(content_length)

        # Resume handling
        if existing_size > 0:
            if response_status == 206:
                total_size = (
                    existing_size + content_length
                    if content_length is not None
                    else None
                )
                mode = "ab"
            else:
                # Server ignored Range; restart from zero
                existing_size = 0
                total_size = content_length
                mode = "wb"
                try:
                    self.partial_file.unlink()
                except FileNotFoundError:
                    pass
        else:
            total_size = content_length
            mode = "wb"

        downloaded = existing_size
        self._emit_status(
            status_callback,
            "downloading",
            {
                "resume_from": existing_size,
                "total": total_size,
            },
        )

        # Read response stream
        try:
            with open(self.partial_file, mode) as file:
                while True:
                    if self.skipped:
                        raise DownloadSkipped()

                    if self.cancelled:
                        raise DownloadCancelled()

                    if self.paused:
                        self._emit_status(status_callback, "paused", None)

                    # Held here while paused so the partial file and the open
                    # response both survive until resume, cancel or skip.
                    while self.paused:
                        if self.skipped:
                            raise DownloadSkipped()
                        if self.cancelled:
                            raise DownloadCancelled()
                        time.sleep(0.2)

                    try:
                        chunk = response.read(self.chunk_size)
                    except (TimeoutError, OSError) as error:
                        raise DownloadError(
                            f"Read timeout/error: {error}"
                        ) from error

                    if not chunk:
                        break

                    file.write(chunk)
                    file.flush()

                    downloaded += len(chunk)

                    if progress_callback:
                        progress_callback(downloaded, total_size)
        finally:
            try:
                response.close()
            except Exception:
                pass

        # Validate file size
        if total_size is not None and downloaded < total_size:
            raise DownloadError(
                "Connection interrupted before the expected file size was "
                "received."
            )

        # Finalize file
        self.partial_file.replace(self.destination)

        self._emit_status(
            status_callback,
            "completed",
            {"size": downloaded},
        )

    # ========================================================
    # Timeouts
    # ========================================================

    def _apply_read_timeout(self, response) -> None:
        """Switch an open response from the connect timeout to the read one.

        ``urlopen`` takes a single timeout that covers both the handshake and
        every later read, so the socket is retimed once the response is open.

        Args:
            response: Open HTTP response whose socket should be retimed.
        """
        socket = getattr(response, "fp", None)
        socket = getattr(socket, "raw", socket)
        socket = getattr(socket, "_sock", None)

        if socket is None:
            return

        try:
            socket.settimeout(self.read_timeout)
        except OSError:
            # An already-closed socket cannot be retimed; the read below will
            # fail and be retried like any other transport error.
            pass

    # ========================================================
    # HTTP 416 helper
    # ========================================================

    @staticmethod
    def _get_size_from_416(error: urllib.error.HTTPError) -> int | None:
        """Extract total file size from a HTTP 416 Content-Range header.

        Example:
            Content-Range: bytes */123456789

        Args:
            error: HTTPError instance containing response headers.

        Returns:
            int | None: Total size in bytes if extractable, otherwise None.
        """
        content_range = error.headers.get("Content-Range")
        if not content_range:
            return None

        try:
            total = content_range.split("/")[1]
            return int(total)
        except (IndexError, ValueError):
            return None

    # ========================================================
    # Status callback
    # ========================================================

    @staticmethod
    def _emit_status(
        callback: Callable[[str, dict | None], None] | None,
        status: str,
        details: dict | None,
    ) -> None:
        """Emit a status event to the optional callback function.

        Args:
            callback: Callable receiving status string and details dict, or
                None.
            status: Status event name (e.g., 'connecting', 'completed').
            details: Event metadata payload or None.
        """
        if callback:
            callback(status, details)
