"""Train a U-Net on the train split and score global Dice on the validation split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import ContrailDataset, list_split
from .metrics import GlobalDice


class ContrailNet(nn.Module):
    """U-Net that optionally works at a larger internal resolution.

    Contrails are often 1-3 pixels wide at 256 px; upsampling the input lets the
    encoder keep them alive through its first stride-2 layers. Logits are
    pooled back to 256 px so loss and metric are computed on the label grid.
    """

    def __init__(self, encoder: str, in_channels: int, upscale: int = 1, pretrained: bool = True):
        super().__init__()
        self.upscale = upscale
        self.net = smp.Unet(
            encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.upscale > 1:
            x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        y = self.net(x)
        if self.upscale > 1:
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return y


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meter = GlobalDice()
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            prob = torch.sigmoid(model(x.to(device, non_blocking=True))).float().cpu().numpy()
        meter.update(prob, y.numpy())
    return meter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--train-list", type=Path, default=None,
                   help="text file of record ids, to compare runs on a fixed training set")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    train_files = list_split(args.data, "train")
    if args.train_list:
        keep = set(args.train_list.read_text().split())
        train_files = [f for f in train_files if f.stem in keep]
    train_files = train_files[: args.train_limit]
    val_files = list_split(args.data, "validation")
    tr = DataLoader(
        ContrailDataset(train_files, args.frames, augment=args.flip),
        batch_size=args.batch, shuffle=True, num_workers=6, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    va = DataLoader(
        ContrailDataset(val_files, args.frames), batch_size=args.batch * 2,
        num_workers=6, pin_memory=True, persistent_workers=True,
    )

    model = ContrailNet(args.encoder, 3 * args.frames, args.upscale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.1
    )
    scaler = torch.amp.GradScaler()
    out = Path("models") / args.name
    out.mkdir(parents=True, exist_ok=True)

    history, best = [], (-1.0, 0.5)
    print(f"{args.name}: {len(train_files)} train / {len(val_files)} validation records", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        th, dice = evaluate(model, va, device).best()
        row = {"epoch": epoch, "loss": total / len(tr), "val_dice": dice, "threshold": th,
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        print(json.dumps(row), flush=True)
        if dice > best[0]:
            best = (dice, th)
            torch.save(model.state_dict(), out / "best.pt")

    summary = {"config": {k: str(v) for k, v in vars(args).items()},
               "best_val_dice": best[0], "best_threshold": best[1], "history": history}
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    print(f"best global Dice {best[0]:.4f} at threshold {best[1]}")


if __name__ == "__main__":
    main()
