"""Stream OpenContrails TFRecord shards from the public GCS bucket.

OpenContrails (Ng et al., 2023, CC BY 4.0) is the source dataset of the Kaggle
competition. Each record carries all nine bands as float64 .npy blobs, so a
shard is streamed and parsed on the fly and only the compact ash-colour version
is written to disk. Records whose id belongs to the Kaggle validation split are
skipped so the validation score stays honest.
"""
from __future__ import annotations

import argparse
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from .data import record_ids, to_uint8_record

BUCKET = "https://storage.googleapis.com/goes_contrails_dataset/20230419/tfrecords"
# GOES-16 ABI band -> OpenContrails feature name
BAND_FEATURES = {"band_11": "cloud_top", "band_14": "data_11um", "band_15": "data_12um"}


def _read_exact(stream, n: int) -> bytes:
    chunks, left = [], n
    while left:
        c = stream.read(left)
        if not c:
            raise EOFError
        chunks.append(c)
        left -= len(c)
    return b"".join(chunks)


def iter_tfrecords(stream):
    """Yield raw record payloads from a TFRecord byte stream (CRCs not checked)."""
    while True:
        try:
            header = _read_exact(stream, 12)
        except EOFError:
            return
        (length,) = struct.unpack("<Q", header[:8])
        payload = _read_exact(stream, length)
        _read_exact(stream, 4)
        yield payload


_TF_DTYPES = {1: np.float32, 2: np.float64, 3: np.int32, 4: np.uint8, 9: np.int64}


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        b = buf[i]
        out |= (b & 0x7F) << shift
        i += 1
        if b < 0x80:
            return out, i
        shift += 7


def decode_tensor_proto(buf: bytes) -> np.ndarray:
    """Decode a serialized tf.TensorProto holding dtype, shape and tensor_content.

    Only the three fields used by OpenContrails are handled, which avoids a
    TensorFlow dependency for what is a few lines of protobuf wire format.
    """
    i, dtype, shape, content = 0, None, [], None
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
            if field == 1:
                dtype = _TF_DTYPES[val]
        elif wire == 2:
            n, i = _varint(buf, i)
            chunk = buf[i : i + n]
            i += n
            if field == 2:  # TensorShapeProto: repeated Dim{size=1}
                j = 0
                while j < len(chunk):
                    _, j = _varint(chunk, j)  # dim tag
                    dn, j = _varint(chunk, j)
                    dim, k = chunk[j : j + dn], 0
                    while k < len(dim):
                        dkey, k = _varint(dim, k)
                        dval, k = _varint(dim, k)
                        if dkey >> 3 == 1:
                            shape.append(dval)
                    j += dn
            elif field == 4:
                content = chunk
        else:
            raise ValueError(f"unexpected wire type {wire}")
    return np.frombuffer(content, dtype=dtype).reshape(shape)


def parse_example(payload: bytes):
    from tfrecord import example_pb2

    ex = example_pb2.Example()
    ex.ParseFromString(payload)
    f = ex.features.feature
    rid = str(f["record_id"].int64_list.value[0])

    def arr(name):
        return decode_tensor_proto(f[name].bytes_list.value[0])

    bands = [arr(BAND_FEATURES[b]).astype(np.float32) for b in ("band_11", "band_14", "band_15")]
    mask = arr("human_pixel_masks")[..., 0].astype(np.uint8)
    return rid, bands, mask


def process_shard(split: str, shard: int, out: Path, skip: set[str]) -> tuple[int, int]:
    url = f"{BUCKET}/{split}.tfrecords-{shard:05d}-of-00100"
    written = skipped = 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        for payload in iter_tfrecords(r.raw):
            rid, bands, mask = parse_example(payload)
            target = out / f"{rid}.npz"
            if rid in skip or target.exists():
                skipped += 1
                continue
            tmp = target.with_suffix(".part")
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, image=to_uint8_record(*bands), mask=mask)
            tmp.replace(target)
            written += 1
    (out / f".shard-{shard:05d}.done").touch()
    print(f"shard {shard}: wrote {written}, skipped {skipped}", flush=True)
    return written, skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], default="train")
    p.add_argument("--shards", type=int, nargs="+", required=True)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    args.out = args.out or Path("data/processed") / args.split
    args.out.mkdir(parents=True, exist_ok=True)
    # Train records that are also Kaggle validation records would leak into the score.
    skip = set(record_ids(args.raw, "validation")) if args.split == "train" else set()
    todo = [s for s in args.shards if not (args.out / f".shard-{s:05d}.done").exists()]
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in [ex.submit(process_shard, args.split, s, args.out, skip) for s in todo]:
            fut.result()


if __name__ == "__main__":
    main()
