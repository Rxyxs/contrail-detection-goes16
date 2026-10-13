import numpy as np
import torch

from contrails.dataset import frame_indices
from contrails.predict import predict
from contrails.rle import rle_decode
from contrails.train import ContrailNet


def _fake_record(folder, seed):
    folder.mkdir()
    rng = np.random.default_rng(seed)
    for band in ("band_11", "band_14", "band_15"):
        np.save(folder / f"{band}.npy", rng.uniform(220, 300, (64, 64, 8)).astype(np.float32))


def test_predict_writes_valid_rle(tmp_path):
    folders = []
    for i in range(3):
        _fake_record(tmp_path / str(1000 + i), i)
        folders.append(tmp_path / str(1000 + i))
    model = ContrailNet("resnet18", 9, upscale=2, pretrained=False).eval()
    rows = predict([model], folders, frame_indices(3), threshold=0.5, device="cpu", batch=2)
    assert [r[0] for r in rows] == ["1000", "1001", "1002"]
    for _, rle in rows:
        assert rle == "-" or rle_decode(rle, (64, 64)).sum() > 0
