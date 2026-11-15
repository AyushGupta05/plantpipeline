"""data/unified (Pheno4D, Sugar4D) -> GrowFields / plant-shape-analysis TrackPlant3D-v2 PLY layout.

  python convert.py [--datasets pheno4d sugar4d] [--out data/TrackPlant3D/versions/v2/gt_corrected_v2]

Writes, for every sequence of the chosen datasets,
  <out>/<dataset>_<species>/<N>_<dataset>_<species>_<sequence_id>_D<dd>.ply
    binary PLY with the exact v2 field layout (x y z double, nx ny nz float, organ_label int, is_leaf_tip uchar):
    organ_label = unified label - 1 (stem 1 -> 0, leaf k -> k-1); unified 0 (unknown) points are dropped;
    normals = Open3D estimate oriented towards +Z (monogrow/canon.py estimate_normals); is_leaf_tip = 0 everywhere
    (no tip annotations -> EPE is undefined, only CD is scored); D<dd> = int(day) of the unified npz; the unified
    plant-centred mm frame is kept (GrowFields canonicalises per organ itself).
  point_splits/<sequence>_split.json   GrowFields' scripts/create_train_test_splits.py re-implemented exactly
    (RandomState(42) per sequence, per frame rng.permutation(n), first int(0.5 n) = train; no tips to move).
  data_splits.json                     {"splits": {"val": {"plants": [...]}, "test": {...}, "train": {...}}} from
    sequences.json (GrowFields only reads val/test, and only via its own repo copy; ours is for our scripts).
  MANIFEST.md                          sequences, frames, organ counts, prior-leakage flag.
GrowFields sequence name = "<dataset>_<species>_<sequence_id>" (PSA regex `\\d+_(.+)_D\\d+\\.ply`).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
UNIFIED = HERE.parents[1] / "data" / "unified"
# MonoGrow's 95k prior (experiments/results/priors/ckpt_cpu95k.pt, config.data) trained on Pheno4D minus these five
# plants and on ALL Sugar4D splits (other_splits = train, val, test) -> only these five are fair for MonoGrow/RichardsNet.
PRIOR_FAIR = {"M01", "M02", "M04", "T01", "T05"}

PLY_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                      ("organ_label", "<i4"), ("is_leaf_tip", "u1")])
HEADER = ("ply\nformat binary_little_endian 1.0\ncomment converted from data/unified by experiments/gf_on_ours/convert.py\n"
          "element vertex {n}\nproperty double x\nproperty double y\nproperty double z\nproperty float nx\nproperty float ny\n"
          "property float nz\nproperty int organ_label\nproperty uchar is_leaf_tip\nend_header\n")


def estimate_normals(points: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()   # +Z, as monogrow/canon.py
    return np.asarray(pcd.normals, dtype=np.float32)


def write_ply(path: Path, points: np.ndarray, normals: np.ndarray, labels: np.ndarray) -> None:
    arr = np.empty(len(points), dtype=PLY_DTYPE)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    arr["organ_label"] = labels; arr["is_leaf_tip"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(HEADER.format(n=len(points)).encode()); f.write(arr.tobytes())


def point_split(seq_name: str, n_per_frame: list[int], labels_per_frame: list[np.ndarray], seed=42, ratio=0.5) -> dict:
    rng = np.random.RandomState(seed)
    out = {"sequence_name": seq_name, "random_seed": seed, "train_ratio": ratio, "created_date": datetime.now().isoformat(),
           "dataset_version": "v2", "n_timepoints": len(n_per_frame), "timepoints": []}
    for t, (n, lab) in enumerate(zip(n_per_frame, labels_per_frame)):
        idx = rng.permutation(n); tr, te = idx[: int(n * ratio)], idx[int(n * ratio):]
        out["timepoints"].append({
            "timepoint_idx": t, "n_total_points": int(n), "n_train": len(tr), "n_test": len(te),
            "train_indices": tr.tolist(), "test_indices": te.tolist(),
            "organ_counts_train": {str(int(o)): int((lab[tr] == o).sum()) for o in np.unique(lab)},
            "organ_counts_test": {str(int(o)): int((lab[te] == o).sum()) for o in np.unique(lab)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pheno4d", "sugar4d"])
    ap.add_argument("--out", type=Path, default=HERE / "data" / "TrackPlant3D" / "versions" / "v2" / "gt_corrected_v2")
    a = ap.parse_args()
    seqs = [e for e in json.load(open(UNIFIED / "sequences.json")) if e["dataset"] in a.datasets]
    (HERE / "point_splits").mkdir(exist_ok=True)
    rows, n_file, splits = [], 0, {"train": [], "val": [], "test": []}
    for e in seqs:
        name = f"{e['dataset']}_{e['species']}_{e['sequence_id']}"
        ns, labs, organs, n_dropped, days, skipped = [], [], set(), 0, [], []
        for scan in e["frames"]:
            d = np.load(UNIFIED / e["dataset"] / e["sequence_id"] / f"{scan}.npz")
            keep = d["labels"] >= 1; n_dropped += int((~keep).sum())
            pts = d["points"][keep].astype(np.float64); lab = (d["labels"][keep].astype(np.int32) - 1)
            if (lab == 0).sum() < 3:     # PSA raises on a day without stem points (needed to orient its leaves)
                skipped.append(scan); continue
            day = int(round(float(d["day"]))); assert abs(day - float(d["day"])) < 1e-6, f"{scan}: non-integer day {d['day']}"
            n_file += 1
            write_ply(a.out / f"{e['dataset']}_{e['species']}" / f"{n_file}_{name}_D{day:02d}.ply", pts, estimate_normals(pts), lab)
            ns.append(len(pts)); labs.append(lab); organs |= set(np.unique(lab).tolist()); days.append(day)
        assert len(set(days)) == len(days), f"{name}: duplicate integer days {days}"
        json.dump(point_split(name, ns, labs), open(HERE / "point_splits" / f"{name}_split.json", "w"), indent=2)
        splits[e["split"]].append(name)
        rows.append((name, e["dataset"], e["species"], e["split"], len(ns), f"{days[0]}-{days[-1]}", len(organs) - 1,
                     f"{min(ns)}-{max(ns)}", n_dropped, ", ".join(skipped) or "-", "yes" if e["sequence_id"] in PRIOR_FAIR else "no"))
        print(f"{name:34s} {len(ns):3d} frames  {len(organs) - 1:3d} leaves  dropped pts {n_dropped}  skipped frames {skipped}")
    json.dump({"description": "Plant-level splits of the converted data/unified sequences (from data/unified/sequences.json). "
                              "GrowFields fits per plant, so only val/test matter; train is listed for completeness.",
               "splits": {k: {"plants": v, "n_plants": len(v)} for k, v in splits.items()}},
              open(HERE / "data_splits.json", "w"), indent=2)
    with open(HERE / "MANIFEST.md", "w") as f:
        f.write(f"# Converted sequences ({datetime.now():%Y-%m-%d}), {n_file} PLYs under `{a.out.relative_to(HERE)}`\n\n"
                "Labels: unified stem 1 -> 0, leaf k -> k-1, unknown 0 dropped. `leaves` = distinct leaf ids over the sequence. "
                "`skipped frames` = scans without stem points (PSA cannot orient their leaves), left out of the sequence. `prior-fair` = not in MonoGrow's 95k prior training set (only Pheno4D M01/M02/M04/T01/T05; every Sugar4D plant was used).\n\n"
                "| sequence | dataset | species | split | frames | days | leaves | pts/frame | dropped pts | skipped frames (no stem) | prior-fair |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"{len(rows)} sequences, {n_file} PLYs; val {len(splits['val'])}, test {len(splits['test'])}, train {len(splits['train'])}")


if __name__ == "__main__":
    main()
