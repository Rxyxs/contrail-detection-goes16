"""Score a saved checkpoint on the validation split over the full threshold grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ContrailDataset, list_split
from .train import ContrailNet, evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name under models/ and reports/")
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    report = Path("reports") / f"{args.name}.json"
    cfg = json.loads(report.read_text())["config"]
    frames, upscale = int(cfg["frames"]), int(cfg["upscale"])
    model = ContrailNet(cfg["encoder"], 3 * frames, upscale, pretrained=False)
    model.load_state_dict(torch.load(Path("models") / args.name / "best.pt", map_location="cuda"))
    model = model.cuda()
    loader = DataLoader(ContrailDataset(list_split(args.data, "validation"), frames),
                        batch_size=32, num_workers=6)
    meter = evaluate(model, loader, "cuda")
    th, dice = meter.best()
    curve = {f"{t:g}": round(float(s), 4) for t, s in zip(meter.thresholds, meter.scores())}
    out = {"name": args.name, "val_dice": dice, "threshold": th, "curve": curve}
    (Path("reports") / f"{args.name}_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("name", "val_dice", "threshold")}))


if __name__ == "__main__":
    main()
