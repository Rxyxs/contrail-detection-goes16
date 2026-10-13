"""Run-length encoding in the competition format.

Pixels are numbered top to bottom, then left to right (column-major), starting
at 1. An empty mask is encoded as "-".
"""
from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    padded = np.concatenate([[0], flat, [0]])
    edges = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    if edges.size == 0:
        return "-"
    edges[1::2] -= edges[::2]
    return " ".join(map(str, edges))


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if rle.strip() not in ("", "-"):
        nums = np.asarray(rle.split(), dtype=np.int64)
        for start, length in zip(nums[::2], nums[1::2]):
            flat[start - 1 : start - 1 + length] = 1
    return flat.reshape(shape, order="F")
