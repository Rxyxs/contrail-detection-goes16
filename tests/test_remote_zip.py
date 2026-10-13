import io
import zipfile

import numpy as np
import pytest

from contrails.remote_zip import read_member


class FakeRange:
    """Stands in for HttpRangeFile: serves byte ranges from an in-memory archive."""

    def __init__(self, blob: bytes):
        self.blob = blob

    def _get(self, start, end):
        return type("R", (), {"content": self.blob[start : end + 1]})()


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_read_member_matches_zipfile(method):
    arrays = {f"train/{i}/band_11.npy": np.random.default_rng(i).random((8, 8, 3)) for i in range(3)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name, a in arrays.items():
            b = io.BytesIO()
            np.save(b, a)
            zf.writestr(name, b.getvalue())
    blob = buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        infos = zf.infolist()
    fake = FakeRange(blob)
    for info in infos:
        out = np.load(io.BytesIO(read_member(fake, info)))
        assert np.array_equal(out, arrays[info.filename])
