"""Figures for the README: validation curves per run and example predictions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .dataset import LABELLED, list_split  # noqa: E402
from .train import ContrailNet  # noqa: E402

RUNS = {
    "r34_f1_e40": "1 frame, 2,447 records",
    "r34_f3": "3 frames, 2,447 records",
    "r34_f1_up2": "1 frame, 2x input, 2,447 records",
    "r34_f1_up2_full": "1 frame, 2x input, 8,197 records",
}
OUT = Path("docs/img")


def curves() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label in RUNS.items():
        path = Path("reports") / f"{name}.json"
        if not path.exists():
            continue
        h = json.loads(path.read_text())["history"]
        ax.plot([r["epoch"] for r in h], [r["val_dice"] for r in h], label=label, lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("global Dice (validation, 1,856 records)")
    ax.set_title("Validation global Dice per epoch")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "val_curves.png", dpi=130)
    plt.close(fig)


@torch.no_grad()
def examples(name: str = "r34_f1_up2_full", n: int = 4, seed: int = 3) -> None:
    cfg = json.loads((Path("reports") / f"{name}_eval.json").read_text())
    model = ContrailNet("resnet34", 3, upscale=2, pretrained=False)
    model.load_state_dict(torch.load(Path("models") / name / "best.pt", map_location="cuda"))
    model = model.cuda().eval()
    files = list_split(Path("data/processed"), "validation")
    # Pick records that do contain contrails so the panels are informative.
    rng = np.random.default_rng(seed)
    picks = []
    for i in rng.permutation(len(files)):
        with np.load(files[i]) as d:
            if d["mask"].sum() > 300:
                picks.append((d["image"][LABELLED], d["mask"]))
        if len(picks) == n:
            break
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, (img, mask) in zip(axes, picks):
        x = torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32) / 255).cuda()
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        for ax, im, title in zip(row, (img, mask, prob > cfg["threshold"]),
                                 ("ash false colour", "human label", "prediction")):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=110)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves()
    examples()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
