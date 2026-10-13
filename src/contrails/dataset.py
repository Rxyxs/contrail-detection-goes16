"""PyTorch dataset over the compacted .npz records."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LABELLED = 4  # index of the labelled frame in the 8-frame sequence


def frame_indices(n_frames: int) -> list[int]:
    """Frames centred on the labelled one; n_frames=1 uses only the labelled frame."""
    if n_frames == 1:
        return [LABELLED]
    before = n_frames // 2
    return list(range(LABELLED - before, LABELLED - before + n_frames))


class ContrailDataset(Dataset):
    def __init__(self, files: list[Path], n_frames: int = 1, augment: bool = False):
        self.files = list(files)
        self.frames = frame_indices(n_frames)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        with np.load(self.files[i]) as d:
            img = d["image"][self.frames]  # (F, H, W, 3)
            mask = d["mask"]
        x = img.transpose(0, 3, 1, 2).reshape(-1, *img.shape[1:3]).astype(np.float32) / 255.0
        y = mask[None].astype(np.float32)
        if self.augment:
            # Horizontal flip only. The masks are reported to sit about half a
            # pixel off the images, so any flip moves that offset; the training
            # script runs with and without it to measure the effect.
            if np.random.rand() < 0.5:
                x, y = x[..., ::-1], y[..., ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def list_split(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.npz"))
