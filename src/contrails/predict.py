"""Write submission.csv from raw competition record folders (band_XX.npy files)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .data import BANDS, _load_npy, to_uint8_record
from .dataset import frame_indices
from .rle import rle_encode
from .train import ContrailNet


def load_record(folder: Path, frames: list[int]) -> np.ndarray:
    img = to_uint8_record(*(_load_npy(folder / f"{b}.npy") for b in BANDS))[frames]
    return img.transpose(0, 3, 1, 2).reshape(-1, *img.shape[1:3]).astype(np.float32) / 255.0


@torch.no_grad()
def predict(models, folders, frames, threshold, device="cuda", batch=16):
    rows = []
    for start in range(0, len(folders), batch):
        chunk = folders[start : start + batch]
        x = torch.from_numpy(np.stack([load_record(f, frames) for f in chunk])).to(device)
        prob = torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0)[:, 0].cpu().numpy()
        for f, p in zip(chunk, prob):
            rows.append((f.name, rle_encode(p > threshold)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True, help="folder holding one sub-folder per record")
    p.add_argument("--weights", type=Path, nargs="+", required=True)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for w in args.weights:
        m = ContrailNet(args.encoder, 3 * args.frames, args.upscale, pretrained=False)
        m.load_state_dict(torch.load(w, map_location=device))
        models.append(m.to(device).eval())
    folders = sorted(d for d in args.records.iterdir() if d.is_dir())
    rows = predict(models, folders, frame_indices(args.frames), args.threshold, device)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "encoded_pixels"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
