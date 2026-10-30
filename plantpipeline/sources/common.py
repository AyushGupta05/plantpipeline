"""Shared helpers for the unified layout: a tiny binary-PLY reader and the per-scan / per-sequence writers.

Unified layout (see README.md in this folder):
    data/unified/<dataset>/<sequence_id>/<scan_id>.npz   points, labels, day, t_norm
    data/unified/manifest.csv                             one row per scan
    data/unified/sequences.json                           one entry per sequence
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
             "float": "f4", "double": "f8", "int32": "i4", "uint8": "u1", "float32": "f4", "float64": "f8"}

MANIFEST_COLS = ["scan_id", "dataset", "species", "treatment", "sequence_id", "frame", "day", "t_norm", "source_time",
                 "split", "labelled", "n_points", "n_leaves", "height_mm", "path"]


def read_ply(data: bytes) -> dict[str, np.ndarray]:
    """Parse a binary little-endian PLY with a single `vertex` element (no list properties) into {property: array}."""
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    assert "format binary_little_endian 1.0" in header, header[:3]
    n, fields = 0, []
    for line in header:
        tok = line.split()
        if tok[:2] == ["element", "vertex"]:
            n = int(tok[2])
        elif tok[0] == "property" and n:  # only the vertex element's properties
            fields.append((tok[2], "<" + PLY_TYPES[tok[1]]))
        elif tok[0] == "element" and n:
            break
    arr = np.frombuffer(data, dtype=np.dtype(fields), count=n, offset=end)
    return {name: arr[name] for name, _ in fields}


def xyz(v: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


class Writer:
    """Collects manifest rows and sequence entries while sources write npz files."""

    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self.sequences: list[dict] = []

    def add_sequence(self, dataset: str, species: str, treatment: str, sequence_id: str, split: str,
                     frames: list[tuple[str, np.ndarray, np.ndarray, float, str, bool]],
                     has_consistent_labels: bool, notes: str) -> None:
        """frames: ordered list of (scan_id, points float32 (N,3) mm, labels int16 (N,), day, source_time, labelled)."""
        folder = self.out / dataset / sequence_id
        folder.mkdir(parents=True, exist_ok=True)
        days = np.array([f[3] for f in frames], dtype=np.float64)
        span = days.max() - days.min()
        t_norm = (days - days.min()) / span if span > 0 else np.zeros_like(days)
        for i, (scan_id, pts, lab, day, source_time, labelled) in enumerate(frames):
            path = folder / f"{scan_id}.npz"
            np.savez(path, points=pts.astype(np.float32), labels=lab.astype(np.int16), day=np.float32(day), t_norm=np.float32(t_norm[i]))
            self.rows.append({
                "scan_id": scan_id, "dataset": dataset, "species": species, "treatment": treatment, "sequence_id": sequence_id,
                "frame": i, "day": round(float(day), 3), "t_norm": round(float(t_norm[i]), 4), "source_time": source_time,
                "split": split, "labelled": int(labelled), "n_points": len(pts), "n_leaves": int(len(np.unique(lab[lab >= 2]))),
                "height_mm": round(float(pts[:, 2].max()), 1) if len(pts) else 0.0, "path": str(path.relative_to(self.out)),
            })
        self.sequences.append({
            "dataset": dataset, "species": species, "treatment": treatment, "sequence_id": sequence_id, "split": split,
            "frames": [f[0] for f in frames], "days": [round(float(d), 3) for d in days], "n_frames": len(frames),
            "has_consistent_labels": has_consistent_labels, "notes": notes,
        })

    def save(self) -> None:
        with open(self.out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
            w.writeheader()
            w.writerows(self.rows)
        with open(self.out / "sequences.json", "w") as f:
            json.dump(self.sequences, f, indent=1)
