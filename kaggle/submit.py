# Kaggle inference script (internet disabled).
# Expects the "contrail-unet-weights" dataset attached at /kaggle/input/, holding:
#   src/contrails/...   the package from this repo
#   wheels/             segmentation_models_pytorch wheel
#   *.pt + config.json  trained weights and the settings they were trained with
import json
import subprocess
import sys
from pathlib import Path

# Mount points have changed across Kaggle versions, so locate inputs by content,
# looking only a few levels deep (the competition folder alone has 244k files).
INPUT = Path("/kaggle/input")


def shallow(pattern_depths):
    for depth in pattern_depths:
        if depth == 0:
            yield INPUT
            continue
        yield from INPUT.glob("/".join(["*"] * depth))


PKG = next(p for p in shallow(range(1, 5)) if (p / "config.json").is_file() and (p / "wheels").is_dir())
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
     "--find-links", str(PKG / "wheels"), "segmentation-models-pytorch"],
    check=True,
)
sys.path.insert(0, str(PKG / "src"))

import torch  # noqa: E402

from contrails.dataset import frame_indices  # noqa: E402
from contrails.predict import predict  # noqa: E402
from contrails.train import ContrailNet  # noqa: E402

cfg = json.loads((PKG / "config.json").read_text())
device = "cuda" if torch.cuda.is_available() else "cpu"
models = []
for w in sorted(PKG.glob("*.pt")):
    m = ContrailNet(cfg["encoder"], 3 * cfg["frames"], cfg["upscale"], pretrained=False)
    m.load_state_dict(torch.load(w, map_location=device))
    models.append(m.to(device).eval())

test = next(p / "test" for p in shallow(range(0, 4))
            if (p / "test").is_dir() and (p / "sample_submission.csv").is_file())
folders = sorted(d for d in test.iterdir() if d.is_dir())
rows = predict(models, folders, frame_indices(cfg["frames"]), cfg["threshold"], device)
with open("submission.csv", "w") as fh:
    fh.write("record_id,encoded_pixels\n")
    for rid, rle in rows:
        fh.write(f"{rid},{rle}\n")
print(f"{len(models)} model(s), {len(rows)} records")
