"""Global Dice: pixels pooled over the whole set, not averaged per image."""
from __future__ import annotations

import numpy as np


class GlobalDice:
    """Accumulates intersection and sizes over many batches, for several thresholds."""

    # Log-spaced low end: with heavy class imbalance the best cut-off can sit
    # well below 0.05, which a linear grid would miss.
    def __init__(self, thresholds=np.unique(np.round(np.r_[np.geomspace(0.002, 0.05, 12),
                                                          np.arange(0.05, 0.96, 0.05)], 4))):
        self.thresholds = np.asarray(thresholds)
        self.inter = np.zeros(len(self.thresholds))
        self.pred = np.zeros(len(self.thresholds))
        self.true = 0.0

    def update(self, prob: np.ndarray, target: np.ndarray) -> None:
        t = target.astype(bool)
        self.true += t.sum()
        for i, th in enumerate(self.thresholds):
            p = prob > th
            self.inter[i] += (p & t).sum()
            self.pred[i] += p.sum()

    def scores(self) -> np.ndarray:
        denom = self.pred + self.true
        return np.where(denom > 0, 2 * self.inter / np.maximum(denom, 1), 1.0)

    def best(self) -> tuple[float, float]:
        s = self.scores()
        i = int(np.argmax(s))
        return float(self.thresholds[i]), float(s[i])


def global_dice(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = pred.astype(bool), target.astype(bool)
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else float(2 * (p & t).sum() / denom)
