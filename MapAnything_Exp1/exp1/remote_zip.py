"""Read single members of a large remote zip with HTTP range requests (E2-E data), so only the frames a
clip uses are downloaded. zipfile reads the central directory from the end of the file and then each
member's local header and data, so a seekable file object that fetches byte ranges is enough."""

import http.client
import io
import os
import time
import urllib.request
import zipfile

UA = {"User-Agent": "3D-CVPR_Exp/E2-E (research; range reads)"}


def _get(url, start=None, end=None, tries=5):
    hdr = dict(UA)
    if start is not None:
        hdr["Range"] = f"bytes={start}-{end}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
                if start is not None and r.status != 206:
                    raise IOError(f"server ignored the range request ({r.status}) for {url}")
                return r.read(), r.headers
        except (OSError, IOError, http.client.HTTPException):   # incl. IncompleteRead (dropped connection)
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def remote_size(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file; reads become range requests (blocks are cached)."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos = url, block, 0
        self.size = remote_size(url)
        self.cache = {}
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=os.SEEK_SET):
        self.pos = {os.SEEK_SET: off, os.SEEK_CUR: self.pos + off, os.SEEK_END: self.size + off}[whence]
        return self.pos

    def prefetch(self, start, end):
        """Fetch [start, end) in ONE request into the block cache (whole blocks, so reads stay exact)."""
        b0, b1 = start // self.block, (min(end, self.size) - 1) // self.block
        if all(b in self.cache for b in range(b0, b1 + 1)):
            return
        s, e = b0 * self.block, min((b1 + 1) * self.block, self.size)
        data, _ = _get(self.url, s, e - 1)
        self.bytes_fetched += len(data)
        for b in range(b0, b1 + 1):
            self.cache.setdefault(b, data[(b - b0) * self.block:(b - b0 + 1) * self.block])

    def _block(self, b):
        if b not in self.cache:
            s = b * self.block
            data, _ = _get(self.url, s, min(s + self.block, self.size) - 1)
            self.cache[b] = data
            self.bytes_fetched += len(data)
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            b, off = divmod(self.pos + len(out), self.block)
            out += self._block(b)[off:off + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def open_zip(url, block=1 << 20):
    return zipfile.ZipFile(RangeFile(url, block))


def read_member(z, name):
    """One member with ONE range request: local header (30 bytes + name + extra, <= 64 KiB slack) + data."""
    info = z.getinfo(name)
    z.fp.prefetch(info.header_offset, info.header_offset + 30 + len(info.orig_filename.encode()) + 65536
                  + info.compress_size)
    return z.read(name)


def fetch(url):
    """A whole (small) remote file."""
    return _get(url)[0]
