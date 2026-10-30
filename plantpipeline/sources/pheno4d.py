"""Pheno4D -> unified layout, from the already processed data/processed/<plant>/<scan>.npz (process.py is unchanged).

Processed files are mm, plant-centred (xy = plant centroid, z = soil surface), FPS to 8192, labels 1 stem / 2.. leaf
(GT on labelled days, 1-NN transfer from the nearest labelled day otherwise). Pheno4D leaf IDs persist across days.
Sequence = plant (M01..M07, T01..T07); split from the processed manifest (plants 01-05 train, 06 val, 07 test).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import Writer


def unify(processed: Path, w: Writer) -> None:
    with open(processed / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    plants: dict[str, list[dict]] = {}
    for r in rows:
        plants.setdefault(r["plant"], []).append(r)
    for plant, prow in plants.items():
        prow.sort(key=lambda r: int(r["day"]))
        frames = []
        for r in prow:
            d = np.load(processed / plant / f"{r['scan_id']}.npz")
            frames.append((r["scan_id"], d["points"], d["labels"], float(r["day"]), r["date"], r["labelled"] == "1"))
        w.add_sequence("pheno4d", prow[0]["species"], "control", plant, prow[0]["split"], frames, has_consistent_labels=True,
                       notes="source mm, scanner-arm frame shifted to plant centroid xy / soil z; unlabelled days carry 1-NN transferred labels; day = calendar days")
    print(f"pheno4d: {len(plants)} sequences")
