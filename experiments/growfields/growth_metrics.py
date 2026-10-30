#!/usr/bin/env python
"""Growth-specific metrics from a GrowFields experiment directory (DESIGN_MEMO.md §7).

For every organ of every sequence/run, the organ length per frame is the 99th-percentile
axial extent in the canonical organ frame: L = p99(z) over the organ's points
(leaves: PSA canonical frame, base at origin, principal axis = +Z; stem: plant space, z up).
Read from the exported organ PLYs `organs/<organ>/{predicted,ground_truth}/*_D<dd>_*.ply`,
which are in mm and in that frame. The organ's first observed frame is never exported
(rollout starts there), so lengths cover frames 2..T of each organ.

Reported per sequence (mean over organs, then over runs) and as the unweighted mean over
sequences:
  viol_rate      fraction of consecutive predicted-frame pairs with L(t+1) < L(t) - tol (tol = 2 mm)
  viol_rate_gt   same on the ground truth (measurement-noise floor)
  neg_mass       sum over pairs of max(0, L(t) - L(t+1)) in mm (prediction)
  len_mae        mean |L_pred - L_gt| over all exported frames, mm

Usage: python growth_metrics.py --experiment_dir <exp> [--tol 2.0] [--output <csv>]
Writes <exp>/growth_metrics.csv (one row per organ frame) and prints the summary table.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

DAY_RE = re.compile(r"_D(\d+)_")


def p99_axial(ply_path):
    pts = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
    return float(np.percentile(pts[:, 2], 99)) if len(pts) else np.nan


def collect(exp_dir):
    rows = []
    for seq_dir in sorted((Path(exp_dir) / "sequences").iterdir()):
        if not seq_dir.is_dir():
            continue
        run_dirs = sorted(seq_dir.glob("run_*")) or [seq_dir]
        for run_dir in run_dirs:
            run = int(run_dir.name.split("_")[1]) if run_dir.name.startswith("run_") else 1
            for organ_dir in sorted((run_dir / "organs").glob("*")):
                pred_dir, gt_dir = organ_dir / "predicted", organ_dir / "ground_truth"
                if not pred_dir.is_dir():
                    continue
                for pred in sorted(pred_dir.glob("*.ply")):
                    m = DAY_RE.search(pred.name)
                    if not m:
                        continue
                    day = int(m.group(1))
                    gts = list(gt_dir.glob(f"*_D{day:02d}_*.ply")) if gt_dir.is_dir() else []
                    rows.append(dict(
                        sequence=seq_dir.name, run=run, organ=organ_dir.name, day=day,
                        L_pred=p99_axial(pred), L_gt=p99_axial(gts[0]) if gts else np.nan,
                    ))
    return pd.DataFrame(rows)


def summarise(df, tol):
    recs = []
    for (seq, run, organ), g in df.groupby(["sequence", "run", "organ"]):
        g = g.sort_values("day")
        d_pred = np.diff(g["L_pred"].to_numpy())
        d_gt = np.diff(g["L_gt"].to_numpy())
        recs.append(dict(
            sequence=seq, run=run, organ=organ, n_pairs=len(d_pred),
            viol=(d_pred < -tol).sum(), viol_gt=(d_gt < -tol).sum(),
            neg_mass=float(np.clip(-d_pred, 0, None).sum()),
            len_mae=float((g["L_pred"] - g["L_gt"]).abs().mean()),
        ))
    per_organ = pd.DataFrame(recs)
    out = []
    for seq, g in per_organ.groupby("sequence"):
        per_run = []
        for run, gr in g.groupby("run"):
            n_pairs = gr["n_pairs"].sum()
            per_run.append(dict(
                viol_rate=gr["viol"].sum() / n_pairs if n_pairs else np.nan,
                viol_rate_gt=gr["viol_gt"].sum() / n_pairs if n_pairs else np.nan,
                neg_mass=gr["neg_mass"].mean(),
                len_mae=gr["len_mae"].mean(),
                n_pairs=n_pairs, n_organs=len(gr),
            ))
        out.append(dict(sequence=seq, **pd.DataFrame(per_run).mean().to_dict()))
    summ = pd.DataFrame(out).set_index("sequence")
    summ.loc["Mean"] = summ.mean()
    return per_organ, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True)
    ap.add_argument("--tol", type=float, default=2.0, help="violation tolerance in mm")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    df = collect(args.experiment_dir)
    if df.empty:
        raise SystemExit(f"no organ predictions under {args.experiment_dir}")
    out = args.output or str(Path(args.experiment_dir) / "growth_metrics.csv")
    df.to_csv(out, index=False)
    _, summ = summarise(df, args.tol)
    summ.to_csv(out.replace(".csv", "_summary.csv"))
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    print(f"\nrows: {len(df)} -> {out}")


if __name__ == "__main__":
    main()
