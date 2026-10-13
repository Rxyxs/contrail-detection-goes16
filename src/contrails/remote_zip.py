"""Read members of the 451 GB competition archive without downloading it.

Kaggle serves the full archive through a single redirect to a signed storage
URL that honours HTTP Range requests. Wrapping that URL in a seekable
file-like object lets the standard ``zipfile`` module read the central
directory and then fetch only the members we need. This costs one Kaggle API
call per URL refresh instead of one call per file, which keeps the download
under the API rate limit.
"""
from __future__ import annotations

import io
import json
import os
import struct
import threading
import time
import zipfile
import zlib

import requests

DOWNLOAD_ALL = "https://www.kaggle.com/api/v1/competitions/data/download-all/{}"


def signed_url(competition: str) -> str:
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as fh:
        cred = json.load(fh)
    r = requests.get(
        DOWNLOAD_ALL.format(competition), auth=(cred["username"], cred["key"]),
        allow_redirects=False, timeout=60,
    )
    r.raise_for_status()
    if "Location" not in r.headers:
        raise RuntimeError(f"no redirect from Kaggle (status {r.status_code})")
    return r.headers["Location"]


class HttpRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote file. Each thread should use its own instance."""

    def __init__(self, competition: str, url: str | None = None, size: int | None = None,
                 url_lock: threading.Lock | None = None, shared: dict | None = None):
        self.competition = competition
        self.shared = shared if shared is not None else {}
        self.lock = url_lock or threading.Lock()
        if url:
            self.shared.setdefault("url", url)
        if "url" not in self.shared:
            self._refresh()
        self.session = requests.Session()
        self.pos = 0
        self.size = size if size is not None else self._fetch_size()

    def _refresh(self, stale: str | None = None) -> None:
        with self.lock:
            if stale is None or self.shared.get("url") == stale:
                self.shared["url"] = signed_url(self.competition)

    def _fetch_size(self) -> int:
        r = self._get(0, 0)
        return int(r.headers["Content-Range"].split("/")[-1])

    def _get(self, start: int, end: int) -> requests.Response:
        for attempt in range(8):
            url = self.shared["url"]
            try:
                r = self.session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            except requests.RequestException:
                time.sleep(2 * 2**attempt)
                continue
            if r.status_code == 206:
                return r
            if r.status_code in (400, 401, 403):  # signed URL expired
                self._refresh(stale=url)
            else:
                time.sleep(2 * 2**attempt)
        raise IOError(f"range {start}-{end} failed with status {r.status_code}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = self._get(self.pos, self.pos + n - 1).content
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


_LOCAL_HEADER = struct.Struct("<4s5H3L2H")  # 30 bytes
_EXTRA_SLACK = 1024  # local extra field may differ from the central one


def read_member(fh: HttpRangeFile, info: zipfile.ZipInfo) -> bytes:
    """Fetch and decompress one member with a single range request."""
    start = info.header_offset
    raw = fh._get(start, start + _LOCAL_HEADER.size + len(info.orig_filename.encode())
                  + _EXTRA_SLACK + info.compress_size).content
    sig, *_, name_len, extra_len = _LOCAL_HEADER.unpack_from(raw)
    if sig != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    off = _LOCAL_HEADER.size + name_len + extra_len
    if off + info.compress_size > len(raw):  # unusually large extra field
        raw = fh._get(start, start + off + info.compress_size - 1).content
    data = raw[off : off + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(out) != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return out


def central_directory(competition: str) -> tuple[dict[str, zipfile.ZipInfo], HttpRangeFile]:
    fh = HttpRangeFile(competition)
    with zipfile.ZipFile(fh) as zf:
        infos = {i.filename: i for i in zf.infolist()}
    return infos, fh
