"""Download source whitelist shared by the downloader and the manager.

The list lives here rather than in either module so both validation points —
``DownloadManager.add`` when a file is queued and ``Downloader.download`` when
the transfer starts — always agree on which hosts are allowed.
"""

ALLOWED_DOMAINS = {
    "ollama.com",
    "www.ollama.com",
    "huggingface.co",
    "www.huggingface.co",
    "python.org",
    "www.python.org",
}
