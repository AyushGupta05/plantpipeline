"""Checks that need the GrowFields/PSA environment (run with the gf_smoke venv python):

  python -I tests.py canon <plant>   canonicalisation parity with plant-shape-analysis (points, matrices, tips)
  python -I tests.py ply <file.ply>  our PLY writer round-trips through growfields.utils.io_utils.load_ply
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig  # noqa: E402
from data import load_benchmark_sequence  # noqa: E402


def check_canon(seq: str) -> None:
    from plant_shape_analysis import LeafSequencesDataset
    cfg = DataConfig()
    ds = LeafSequencesDataset(dataset_path=str(cfg.tp3d_dir.parents[1]), version="v2", min_timepoints=1,
                              use_ply=True, apply_alignment=True, plant_alignment_method=None, estimate_normals=False,
                              auto_download=False)
    _, tracks, _ = load_benchmark_sequence(cfg, seq)
    worst_pts = worst_M = worst_tip = 0.0
    n = 0
    for leaf in ds.get_timeseries_by_plant_sequence(seq):
        ours = next((t for t in tracks if t.organ_id == leaf["leaf_id"]), None)
        if ours is None or not leaf.get("is_aligned", False):
            print(f"  leaf {leaf['leaf_id']}: skipped (PSA aligned={leaf.get('is_aligned')}, ours={'yes' if ours else 'no'})")
            continue
        comp = LeafSequencesDataset.compose_transformations(leaf["transformation_stages"])
        assert len(comp) == len(ours.points) == len(leaf["timepoints"]), (len(comp), len(ours.points))
        for t, (tp, c) in enumerate(zip(leaf["timepoints"], comp)):
            assert tp["day"] == ours.days[t]
            worst_pts = max(worst_pts, np.abs(tp["points"] - ours.points[t]).max())
            worst_M = max(worst_M, np.abs(c["composed_matrix"] - ours.M[t]).max())
            if tp.get("leaf_tip") is not None:
                worst_tip = max(worst_tip, np.abs(np.asarray(tp["leaf_tip"]) - ours.tips[t]).max())
            n += 1
    print(f"{seq}: {n} leaf frames compared; max |dpoints| = {worst_pts:.2e} mm (float32 storage), "
          f"max |dM| = {worst_M:.2e}, max |dtip| = {worst_tip:.2e}")
    assert worst_pts < 1e-2 and worst_M < 1e-6 and worst_tip < 1e-2


def check_ply(path: str) -> None:
    from growfields.utils.io_utils import load_ply
    from ply import read_ply
    a, b = load_ply(path), read_ply(path)
    for k in ("points", "organ_label", "is_train", "is_leaf_tip", "tip_organ_label"):
        assert k in a and k in b, k
        assert np.allclose(a[k].reshape(b[k].shape), b[k]), k
    print(f"{path}: GrowFields load_ply == ours on {list(a)}; {len(a['points'])} points, {int(a['is_leaf_tip'].sum())} tips")


if __name__ == "__main__":
    {"canon": check_canon, "ply": check_ply}[sys.argv[1]](sys.argv[2])
