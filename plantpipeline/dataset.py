"""Minimal PyTorch dataset over the processed npz files.

Each item: (points (n,3) float32, t float32) where
  points are randomly subsampled from the stored N and scaled by one global
  factor (so absolute size over time is preserved),
  t is the plant's days since first scan, normalised by `max_day`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PlantScans(Dataset):
    def __init__(self, processed: Path, split: str = "train", n: int = 2048, species: str | None = None, scale_mm: float = 500.0, max_day: float = 20.0, seed: int = 0):
        with open(processed / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split and (species is None or r["species"] == species)]
        self.processed, self.n, self.scale, self.max_day = processed, n, scale_mm, max_day
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[i]
        d = np.load(self.processed / r["plant"] / f"{r['scan_id']}.npz")
        pts = d["points"]
        idx = self.rng.choice(len(pts), self.n, replace=len(pts) < self.n)
        x = torch.from_numpy(pts[idx] / self.scale).float()
        t = torch.tensor(int(r["day"]) / self.max_day, dtype=torch.float32)
        return x, t
