"""Sugar4D (Bonn / IfZ Goettingen, sugar beet) -> unified layout.

Zip layout: point_clouds/<DAS>_das/<DAS>_das_plant_NNN.ply, 16 time points (37..93 days after sowing) x 48 plants.
PLY fields: x y z (float, METRES, greenhouse-fixed frame, ground removed), nx ny nz,
label_semantic (0 taproot, 1 core, 2 leaf), label_instance (leaf 0..19 consistent over time; -1 core, -2 taproot).
We convert to mm, map taproot+core -> stem (1), leaf k -> k + 2, leaf with no instance -> 0 (unknown),
centre each sequence on the plant base of its first frame (xy = centroid of stem points, z = lowest stem point),
FPS to n_points, and set day = DAS - first DAS of that plant.
Splits: shipped point_clouds/splits/{train,val,test}.txt (by plant).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from ..sample import fps
from .common import Writer, read_ply, xyz

NAME_RE = re.compile(r"point_clouds/(?P<das>\d+)_das/\d+_das_(?P<plant>plant_\d{3})\.ply$")


def plant_base(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    stem = points[labels == 1] if (labels == 1).any() else points
    return np.array([*stem[:, :2].mean(axis=0), stem[:, 2].min()], dtype=np.float32)


def unify(raw_zip: Path, w: Writer, n_points: int = 8192) -> None:
    zf = zipfile.ZipFile(raw_zip)
    split_of = {}
    for s in ("train", "val", "test"):
        for line in zf.read(f"point_clouds/splits/{s}.txt").decode().split():
            split_of[line.split("_das_")[1].removesuffix(".ply")] = s
    seqs: dict[str, list] = {}
    for name in zf.namelist():
        m = NAME_RE.match(name)
        if m:
            seqs.setdefault(m["plant"], []).append((int(m["das"]), name))
    for plant, items in sorted(seqs.items()):
        items.sort()
        frames, origin = [], None
        for das, name in items:
            v = read_ply(zf.read(name))
            pts = xyz(v) * 1000.0  # m -> mm
            sem, inst = v["label_semantic"], v["label_instance"]
            lab = np.where(sem == 2, np.where(inst >= 0, inst + 2, 0), 1).astype(np.int16)
            if origin is None:
                origin = plant_base(pts, lab)
            idx = fps(pts, n_points)
            frames.append((f"{plant}_das{das}", pts[idx] - origin, lab[idx], float(das - items[0][0]), f"DAS{das}", True))
        w.add_sequence("sugar4d", "sugar_beet", "control", plant, split_of[plant], frames, has_consistent_labels=True,
                       notes=f"source metres -> mm, greenhouse frame shifted by {[round(float(x), 1) for x in -origin]} (first-frame plant base); day = DAS - {items[0][0]}; 12 genotypes (not machine-readable)")
        print(f"sugar4d {plant}: {len(frames)} frames")
    print(f"sugar4d: {len(seqs)} sequences")
