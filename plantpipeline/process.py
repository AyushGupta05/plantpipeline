"""Raw Pheno4D -> processed per-scan .npz files + manifest.csv.

For each plant:
  1. read every scan (labelled or not),
  2. get a soil mask (labels, or transferred from the nearest labelled days),
  3. drop soil, shift to a per-plant origin (same for all days),
  4. farthest-point-sample to N points,
  5. save  processed/<plant>/<scan_id>.npz  with points, labels, metadata.

Output npz keys:
  points  float32 (N,3)  mm, plant-centred, z up from the soil surface
  labels  int16   (N,)   1 stem, 2.. leaf instance (from GT or transferred)
  origin  float32 (3,)   subtracted offset in the raw arm frame
  day     int            calendar days since this plant's first scan
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .pheno4d import ScanInfo, iter_plants, list_scans, read_scan
from .sample import fps
from .soil import plant_origin, soil_mask_from_labels, transfer_labels

SPLITS = {"train": {"01", "02", "03", "04", "05"}, "val": {"06"}, "test": {"07"}}
MANIFEST_COLS = ["scan_id", "plant", "species", "date", "day", "split", "labelled", "soil_method", "n_raw", "n_plant", "n_out", "height_mm", "n_leaves"]


def split_of(plant: str) -> str:
    for name, ids in SPLITS.items():
        if plant[1:] in ids:
            return name
    return "train"


def process(raw: Path, out: Path, n_points: int = 8192, plants: list[str] | None = None) -> list[dict]:
    scans = list_scans(raw)
    if plants:
        scans = [s for s in scans if s.plant in plants]
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for plant, plant_scans in iter_plants(scans):
        rows.extend(_process_plant(raw, out, plant, plant_scans, n_points))
    _write_manifest(out / "manifest.csv", rows)
    return rows


def _process_plant(raw: Path, out: Path, plant: str, scans: list[ScanInfo], n_points: int) -> list[dict]:
    t0 = time.time()
    clouds = {s.scan_id: read_scan(raw, s) for s in scans}
    labelled = [s for s in scans if s.labelled]
    first_day = scans[0].scan_date

    # soil masks + labels for every scan
    masks: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for s in scans:
        pts, lab = clouds[s.scan_id]
        if lab is not None:
            masks[s.scan_id] = (lab, soil_mask_from_labels(lab), "gt")
        else:
            nb = _nearest_labelled(s, labelled)
            lab_t, soil = transfer_labels(pts, [clouds[n.scan_id] for n in nb])
            masks[s.scan_id] = (lab_t, soil, "transfer")

    # one origin per plant, from the first labelled scan (or first scan)
    ref = (labelled or scans)[0]
    origin = plant_origin(clouds[ref.scan_id][0], masks[ref.scan_id][1])

    rows = []
    (out / plant).mkdir(exist_ok=True)
    for s in scans:
        pts, _ = clouds[s.scan_id]
        lab, soil, method = masks[s.scan_id]
        plant_pts = pts[~soil] - origin
        plant_lab = lab[~soil]
        idx = fps(plant_pts, n_points)
        day = (s.scan_date - first_day).days
        np.savez(out / plant / f"{s.scan_id}.npz", points=plant_pts[idx], labels=plant_lab[idx], origin=origin, day=day)
        rows.append({
            "scan_id": s.scan_id, "plant": plant, "species": s.species, "date": s.scan_date.isoformat(), "day": day,
            "split": split_of(plant), "labelled": int(s.labelled), "soil_method": method,
            "n_raw": len(pts), "n_plant": len(plant_pts), "n_out": n_points,
            "height_mm": round(float(plant_pts[:, 2].max()), 1) if len(plant_pts) else 0.0,
            "n_leaves": int(len(np.unique(plant_lab[plant_lab >= 2]))),
        })
    print(f"{plant}: {len(scans)} scans in {time.time() - t0:.1f}s")
    return rows


def _nearest_labelled(s: ScanInfo, labelled: list[ScanInfo]) -> list[ScanInfo]:
    """Closest labelled scan before and after `s` (1 or 2 scans)."""
    before = [l for l in labelled if l.scan_date < s.scan_date]
    after = [l for l in labelled if l.scan_date > s.scan_date]
    nb = []
    if before:
        nb.append(before[-1])
    if after:
        nb.append(after[0])
    if not nb:
        raise ValueError(f"no labelled scan to transfer soil from for {s.scan_id}")
    return nb


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
