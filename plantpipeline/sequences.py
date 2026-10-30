"""Sequence-level PyTorch dataset over data/unified (for 4D growth models).

Each item is one whole plant sequence:
    points  float32 (T, n, 3)  mm / scale, n random points per frame (with replacement if a frame has fewer)
    labels  int64   (T, n)     0 unknown, 1 stem, 2.. leaf instance (consistent over time where the source has it)
    days    float32 (T,)       days since the sequence's first scan
    t_norm  float32 (T,)       days / last day, in [0, 1]
    meta    dict               the sequences.json entry (dataset, species, split, ...)
T differs between sequences, so use batch_size=1 or a custom collate_fn (e.g. pad or sample a fixed window).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PlantSequences(Dataset):
    def __init__(self, unified: Path, split: str = "train", n: int = 2048, datasets: list[str] | None = None,
                 species: list[str] | None = None, scale_mm: float = 500.0, seed: int = 0):
        with open(unified / "sequences.json") as f:
            seqs = json.load(f)
        self.seqs = [s for s in seqs if s["split"] == split and (datasets is None or s["dataset"] in datasets)
                     and (species is None or s["species"] in species)]
        self.unified, self.n, self.scale = unified, n, scale_mm
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> dict:
        s = self.seqs[i]
        pts, lab, days, t_norm = [], [], [], []
        for scan_id in s["frames"]:
            d = np.load(self.unified / s["dataset"] / s["sequence_id"] / f"{scan_id}.npz")
            idx = self.rng.choice(len(d["points"]), self.n, replace=len(d["points"]) < self.n)
            pts.append(d["points"][idx] / self.scale)
            lab.append(d["labels"][idx])
            days.append(d["day"])
            t_norm.append(d["t_norm"])
        return {
            "points": torch.from_numpy(np.stack(pts)).float(), "labels": torch.from_numpy(np.stack(lab)).long(),
            "days": torch.tensor(np.array(days), dtype=torch.float32), "t_norm": torch.tensor(np.array(t_norm), dtype=torch.float32),
            "meta": s,
        }
