"""Download source whitelist shared by the downloader and the manager.

The list lives here rather than in either module so both validation points —
``DownloadManager.add`` when a file is queued and ``Downloader.download`` when
the transfer starts — always agree on which hosts are allowed. The comparison
lives here too, in :func:`verify_download_source`, so neither caller can drift
into matching hosts its own way.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Only real HTTP transports are downloadable. The scheme is checked because a
# ``file:``, ``data:`` or ``ftp:`` URL carries no host to compare, so a
# host-only test would wave one through on an empty hostname.
ALLOWED_SCHEMES = frozenset({"http", "https"})

# Hosts accepted by exact match, compared lowercased and without a trailing dot.
ALLOWED_DOMAINS = {
    "ollama.com",
    "www.ollama.com",
    "huggingface.co",
    "www.huggingface.co",
    "hf.co",
    "www.hf.co",
    "python.org",
    "www.python.org",
}

# Parents whose subdomains are accepted too. Hugging Face serves the file bodies
# from per-repository CDN hosts — ``cdn-lfs.huggingface.co``,
# ``cdn-lfs-us-1.huggingface.co``, the ``*.hf.co`` mirrors — and a download link
# redirects onto one of them, so the whitelist has to cover them without naming
# every host Hugging Face may add. Matching is on a label boundary, so
# ``nothuggingface.co`` is not a subdomain of ``huggingface.co``.
ALLOWED_DOMAIN_SUFFIXES = frozenset({"huggingface.co", "hf.co"})


class DownloadSourceRejected(PermissionError):
    """Raised when a URL's scheme or host is not on the whitelist.

    A ``PermissionError`` subclass, which is what both validation points
    documented and raised before this class existed, so anything catching that
    still catches this. The message always names what was rejected and lists
    every accepted host, since a caller that guessed a domain wrong has no other
    way to learn which ones are allowed.
    """


def allowed_sources() -> list[str]:
    """List every accepted host, wildcards included.

    Returns:
        list[str]: Exactly matched hosts followed by the ``*.domain`` wildcards,
        each group sorted, for display in an error or a tool result.
    """
    return sorted(ALLOWED_DOMAINS) + sorted(
        f"*.{suffix}" for suffix in ALLOWED_DOMAIN_SUFFIXES
    )


def _listed() -> str:
    """str: Every accepted host as one comma-separated line."""
    return ", ".join(allowed_sources())


def normalize_hostname(hostname: str | None) -> str:
    """Reduce a hostname to the form the whitelist is compared against.

    Args:
        hostname: Host as parsed from a URL, or None when it had none.

    Returns:
        str: Lowercased host without surrounding whitespace or a trailing root
        dot, or an empty string when there was no host.
    """
    if not hostname:
        return ""

    return hostname.strip().rstrip(".").lower()


def is_allowed_domain(hostname: str | None) -> bool:
    """Report whether a hostname is on the whitelist.

    Args:
        hostname: Host as parsed from a URL, without port or userinfo.

    Returns:
        bool: True when the host matches an allowed domain exactly or is a
        subdomain of one of the allowed parents.
    """
    host = normalize_hostname(hostname)

    if not host:
        return False

    if host in ALLOWED_DOMAINS:
        return True

    return any(
        host == suffix or host.endswith(f".{suffix}")
        for suffix in ALLOWED_DOMAIN_SUFFIXES
    )


def verify_download_source(url: str) -> str:
    """Validate a download URL's scheme and host against the whitelist.

    The host is taken from ``urlparse(...).hostname`` rather than ``netloc``, so
    a port and any ``user:password@`` prefix are dropped before the comparison:
    neither ``huggingface.co:8080`` nor ``huggingface.co@evil.example`` can be
    dressed up to look like an allowed host.

    Args:
        url: HTTP or HTTPS URL to validate.

    Returns:
        str: The accepted hostname, lowercased and without port or userinfo.

    Raises:
        DownloadSourceRejected: If the URL is malformed, its scheme is not
            ``http`` or ``https``, or its host is not on the whitelist. The
            message names the rejected value and lists every allowed domain.
    """
    try:
        parts = urlparse(url)
        scheme = (parts.scheme or "").lower()
        hostname = parts.hostname
    except ValueError as error:
        raise DownloadSourceRejected(
            f"Access denied: '{url}' is not a usable download URL ({error}). "
            f"Allowed domains: {_listed()}."
        ) from error

    if scheme not in ALLOWED_SCHEMES:
        raise DownloadSourceRejected(
            f"Access denied: scheme '{scheme or 'missing'}' is not allowed for "
            f"downloads (only http and https are). Allowed domains: "
            f"{_listed()}."
        )

    if not is_allowed_domain(hostname):
        raise DownloadSourceRejected(
            f"Access denied: domain '{normalize_hostname(hostname) or 'missing'}'"
            f" is not allowed. Allowed domains: {_listed()}."
        )

    return normalize_hostname(hostname)
