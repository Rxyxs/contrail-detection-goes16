"""Download GOES-16 records from the Kaggle competition and compact them.

Only bands 11, 14 and 15 are fetched, straight out of the remote archive, (the "ash" false-colour combination used to
spot thin ice clouds). Each record is turned into an uint8 array of shape
(T, H, W, 3) plus the binary mask, and the raw float32 files are deleted, so a
~140 GB download ends up as ~35 GB on disk.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .remote_zip import HttpRangeFile, central_directory, read_member

COMPETITION = "google-research-identify-contrails-reduce-global-warming"
BANDS = ("band_11", "band_14", "band_15")
N_TIMES_BEFORE = 4  # labelled frame index inside the 8-frame sequence

# Ranges of the standard ash RGB recipe (brightness temperatures, K).
_T11_BOUNDS = (243.0, 303.0)
_CLOUD_TOP_TDIFF_BOUNDS = (-4.0, 5.0)
_TDIFF_BOUNDS = (-4.0, 2.0)


def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def ash_color(b11: np.ndarray, b14: np.ndarray, b15: np.ndarray) -> np.ndarray:
    """Return ash false colour in [0, 1] with channels last; inputs are (H, W, T)."""
    r = _normalize(b15 - b14, *_TDIFF_BOUNDS)
    g = _normalize(b14 - b11, *_CLOUD_TOP_TDIFF_BOUNDS)
    b = _normalize(b14, *_T11_BOUNDS)
    return np.stack([r, g, b], axis=-1)  # (H, W, T, 3)


def to_uint8_record(b11, b14, b15) -> np.ndarray:
    """(H, W, T) bands -> (T, H, W, 3) uint8."""
    img = ash_color(b11, b14, b15)
    return np.round(img * 255).astype(np.uint8).transpose(2, 0, 1, 3)


def _load_npy(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:2] == b"PK":  # Kaggle sometimes serves a zip under the original name
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            raw = zf.read(zf.namelist()[0])
    return np.load(io.BytesIO(raw))


_local = threading.local()


def _thread_file(base: HttpRangeFile) -> HttpRangeFile:
    """One HTTP session per worker thread, all sharing the signed URL."""
    if not hasattr(_local, "fh"):
        _local.fh = HttpRangeFile(COMPETITION, size=base.size, shared=base.shared, url_lock=base.lock)
    return _local.fh


def process_record(infos, base, split: str, record_id: str, out: Path) -> str:
    target = out / split / f"{record_id}.npz"
    if target.exists():
        return "skip"
    fh = _thread_file(base)
    arrays = {
        name: np.load(io.BytesIO(read_member(fh, infos[f"{split}/{record_id}/{name}.npy"])))
        for name in (*BANDS, "human_pixel_masks")
    }
    img = to_uint8_record(*(arrays[b] for b in BANDS))
    mask = arrays["human_pixel_masks"][..., 0].astype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with tmp.open("wb") as fh_out:
        np.savez_compressed(fh_out, image=img, mask=mask)
    tmp.replace(target)  # never leave a half-written record behind
    return "ok"


def record_ids(raw_dir: Path, split: str) -> list[str]:
    path = raw_dir / f"{split}_metadata.json"
    data = path.read_bytes()
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            data = zf.read(zf.namelist()[0])
    return [str(r["record_id"]) for r in json.loads(data)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    ids = record_ids(args.raw, args.split)[: args.limit]
    infos, base = central_directory(COMPETITION)
    done = failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_record, infos, base, args.split, i, args.out): i for i in ids}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as e:  # network hiccups: rerun the command to resume
                failed += 1
                print(f"FAIL {futs[fut]}: {e}", flush=True)
            if (done + failed) % 250 == 0:
                print(f"{args.split}: {done + failed}/{len(ids)} (failed {failed})", flush=True)
    print(f"{args.split}: done {done}, failed {failed}")


if __name__ == "__main__":
    main()
