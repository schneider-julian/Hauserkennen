"""Shared HTTP session with retries and a polite User-Agent.

`requests` honours HTTPS_PROXY / REQUESTS_CA_BUNDLE from the environment, so the
same code runs on a laptop and behind corporate or sandbox proxies.
"""

from __future__ import annotations

import io

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "hauserkennen/0.1 (building identification from listing photos)"

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST", "HEAD"),
        )
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
        s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
        s.headers["User-Agent"] = USER_AGENT
        _session = s
    return _session


def get(url: str, *, params: dict | None = None, headers: dict | None = None, timeout: float = 60) -> requests.Response:
    r = session().get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP range requests.

    Lets pyarrow read single row groups out of multi-hundred-megabyte parquet
    files without downloading them.
    """

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        data = get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=180).content
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
