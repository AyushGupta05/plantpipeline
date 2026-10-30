"""Run every source and write data/unified/{manifest.csv,sequences.json}."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import pheno4d, sugar4d, trackplant3d
from .common import Writer


def unify(raw: Path, processed: Path, out: Path, datasets: list[str], n_points: int = 8192) -> Writer:
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(out)
    if "pheno4d" in datasets:
        pheno4d.unify(processed, w)
    if "trackplant3d" in datasets:
        trackplant3d.unify(raw / "TrackPlant3D_v2.zip", w)
    if "sugar4d" in datasets:
        sugar4d.unify(raw / "sugar4d.zip", w, n_points)
    w.save()
    return w


def summary(rows: list[dict], sequences: list[dict]) -> str:
    lines = [f"{len(rows)} scans, {len(sequences)} sequences"]
    for key in ("dataset", "species", "split"):
        lines.append(f"  by {key}: scans {dict(Counter(r[key] for r in rows))} | sequences {dict(Counter(s[key] for s in sequences))}")
    for ds in sorted({r["dataset"] for r in rows}):
        n = [int(r["n_points"]) for r in rows if r["dataset"] == ds]
        nf = [s["n_frames"] for s in sequences if s["dataset"] == ds]
        d = [float(r["day"]) for r in rows if r["dataset"] == ds]
        lines.append(f"  {ds}: points/scan {min(n)}-{max(n)}, frames/seq {min(nf)}-{max(nf)}, day range 0-{max(d):g}")
    return "\n".join(lines)
