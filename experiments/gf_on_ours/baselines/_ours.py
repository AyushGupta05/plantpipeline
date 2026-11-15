"""Shared shim: point monogrow/data.py at our converted TrackPlant3D-style tree and teach it our sequence names.
Nothing under experiments/monogrow or experiments/richards is modified; we patch module attributes at import time."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MONO = HERE.parent / "monogrow"; RICH = HERE.parent / "richards"
TP3D = HERE / "data" / "TrackPlant3D" / "versions" / "v2" / "gt_corrected_v2"
SPLITS = HERE / "point_splits"
_RE = re.compile(r"^(?:pheno4d|sugar4d|trackplant3d)_(maize|tomato|sorghum|tobacco|sugar_beet)_")


def species_of(seq: str) -> str:
    """'pheno4d_maize_M01' -> 'maize', 'sugar4d_sugar_beet_plant_002' -> 'sugar_beet'; TP3D names unchanged."""
    m = _RE.match(seq)
    return m.group(1) if m else {"tomato1": "tomato", "tomato2": "tomato"}.get(seq.split("_")[0], seq.split("_")[0])


def patch(paths):
    for p in reversed(paths):
        sys.path.insert(0, str(p))
    import data  # noqa: E402  (monogrow/data.py)
    data.tp3d_species = species_of   # load_benchmark_sequence looks this up in data's globals
    return data
