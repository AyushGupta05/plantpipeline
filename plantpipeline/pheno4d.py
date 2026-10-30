"""Read raw Pheno4D scans.

Raw layout (inside Pheno4D.zip or an extracted folder):
    Pheno4D/Maize01/M01_0313_a.txt      labelled   -> x y z label_collar label_tip
    Pheno4D/Tomato01/T01_0305_a.txt     labelled   -> x y z label
    Pheno4D/Tomato01/T01_0306.txt       unlabelled -> x y z

Coordinates are millimetres in the scanner arm's fixed frame; all days of one
plant share that frame. Labels: 0 = soil + pot, 1 = stem, 2.. = leaf instances.
Maize has two label columns; the *collar* column has a stem class, the *tip*
column does not (0 soil, 1.. leaves). We keep the collar column.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

NAME_RE = re.compile(r"^(?P<plant>[MT]\d{2})_(?P<mmdd>\d{4})(?P<lab>_a)?\.txt$")
SPECIES = {"M": "maize", "T": "tomato"}
YEAR = 2020  # scan year is not documented; only used to compute day gaps


@dataclass(frozen=True)
class ScanInfo:
    plant: str  # e.g. "M01"
    species: str  # "maize" | "tomato"
    scan_date: date
    labelled: bool
    source: str  # zip member name or file path

    @property
    def scan_id(self) -> str:
        return f"{self.plant}_{self.scan_date:%m%d}"


def parse_name(name: str) -> ScanInfo | None:
    m = NAME_RE.match(Path(name).name)
    if not m:
        return None
    mmdd = m["mmdd"]
    return ScanInfo(
        plant=m["plant"],
        species=SPECIES[m["plant"][0]],
        scan_date=date(YEAR, int(mmdd[:2]), int(mmdd[2:])),
        labelled=m["lab"] is not None,
        source=name,
    )


def list_scans(raw: Path) -> list[ScanInfo]:
    """List all scans in a Pheno4D.zip or an extracted folder, sorted by plant and date."""
    if raw.is_file():
        with zipfile.ZipFile(raw) as zf:
            names = zf.namelist()
    else:
        names = [str(p) for p in raw.rglob("*.txt")]
    scans = [s for s in (parse_name(n) for n in names) if s is not None]
    return sorted(scans, key=lambda s: (s.plant, s.scan_date))


def read_scan(raw: Path, info: ScanInfo) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (points float32 (N,3) in mm, labels int16 (N,) or None). Exact duplicate points are dropped."""
    if raw.is_file():
        with zipfile.ZipFile(raw) as zf:
            data = io.BytesIO(zf.read(info.source))
    else:
        data = Path(info.source)
    first = _first_line(raw, info)
    ncol = len(first.split())
    schema = [pl.Float32] * 3 + [pl.Int16] * (ncol - 3)
    df = pl.read_csv(data, separator=" ", has_header=False, schema_overrides=schema, new_columns=[f"c{i}" for i in range(ncol)])
    arr = df.to_numpy()
    arr = np.unique(arr, axis=0)  # drop exact duplicates (0.1-0.5 % of points)
    points = np.ascontiguousarray(arr[:, :3], dtype=np.float32)
    labels = arr[:, 3].astype(np.int16) if ncol > 3 else None  # column 4 = collar labels for maize
    return points, labels


def _first_line(raw: Path, info: ScanInfo) -> str:
    if raw.is_file():
        with zipfile.ZipFile(raw) as zf, zf.open(info.source) as f:
            return f.readline().decode()
    with open(info.source) as f:
        return f.readline()


def iter_plants(scans: list[ScanInfo]) -> Iterator[tuple[str, list[ScanInfo]]]:
    plants: dict[str, list[ScanInfo]] = {}
    for s in scans:
        plants.setdefault(s.plant, []).append(s)
    yield from plants.items()
