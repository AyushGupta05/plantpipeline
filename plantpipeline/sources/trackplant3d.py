"""TrackPlant3D v2 (ETH, GrowFields processed version) -> unified layout.

Zip layout: v2/gt_corrected_v2/<crop>/<index>_<crop>_<treatment>_plant<k>_D<day>.ply
PLY fields: x y z (double, mm), nx ny nz, organ_label (0 = stem, 1.. = leaf, consistent over time), is_leaf_tip.
Clouds are already temporally aligned, z up, stem base at z = 0 (~6.7k-11k points; kept as is, no FPS).
We only shift xy so the first frame's stem base sits at the origin (same shift for every frame of a sequence),
and remap labels: stem 0 -> 1, leaf k -> k + 1.
Splits: splits/data_splits.json (GrowFields) gives val/test plants; all other plants are train.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from .common import Writer, read_ply, xyz

NAME_RE = re.compile(r"v2/gt_corrected_v2/(?P<crop>\w+)/(?P<idx>\d+)_(?P<seq>(?P<cultivar>[a-z]+\d?)_(?P<treat>[a-z]+\d?)_plant\d+)_D(?P<day>\d+)\.ply$")
SPECIES = {"tomato1": "tomato", "tomato2": "tomato"}


def stem_base_xy(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    stem = points[labels == 1] if (labels == 1).any() else points
    low = stem[stem[:, 2] <= np.percentile(stem[:, 2], 10)]
    return low[:, :2].mean(axis=0)


def unify(raw_zip: Path, w: Writer) -> None:
    zf = zipfile.ZipFile(raw_zip)
    splits = json.loads(zf.read("v2/splits/data_splits.json"))["splits"]
    split_of = {p: s for s in ("val", "test") for p in splits[s]["plants"]}
    seqs: dict[str, list] = {}
    for name in zf.namelist():
        m = NAME_RE.match(name)
        if m:
            seqs.setdefault(m["seq"], []).append((int(m["day"]), name, m))
    for seq, items in sorted(seqs.items()):
        items.sort()
        frames, shift = [], None
        for day, name, m in items:
            v = read_ply(zf.read(name))
            pts = xyz(v)
            lab = (v["organ_label"].astype(np.int16) + 1)  # 0 stem -> 1, leaf k -> k+1
            if shift is None:
                shift = np.array([*stem_base_xy(pts, lab), 0.0], dtype=np.float32)
            frames.append((f"{seq}_D{day:02d}", pts - shift, lab, float(day), f"D{day:02d}", True))
        m = items[0][2]
        w.add_sequence("trackplant3d", SPECIES.get(m["cultivar"], m["cultivar"]), m["treat"].rstrip("12"), seq, split_of.get(seq, "train"),
                       frames, has_consistent_labels=True,
                       notes=f"cultivar={m['cultivar']}; source mm, ETH v2 stem-aligned frame, xy shifted by {[round(float(x), 2) for x in -shift[:2]]}; day = D index")
    print(f"trackplant3d: {len(seqs)} sequences")
