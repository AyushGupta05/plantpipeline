"""Growth metrics beyond Chamfer (DESIGN_MEMO section 7), from predicted and GT organ point sets in the canonical frame.

  python growth_metrics.py --experiment_dir runs/full/fit_extrap [--mode normal|interpolation|extrapolation|all]

Length L(t) = 99th-percentile axial (canonical Z) extent of the organ's points in frame t. The predicted trajectory
starts at the GT length of the organ's first observed frame (the rollout starts from those points); with
`fit.source=nearest` (results.json "source") every prediction is compared with the GT length of its own source frame,
the latest observed frame before it, so dL / violations measure one rollout each rather than a chain.
  * monotonicity violation rate: fraction of consecutive predicted frames with L(t+1) < L(t) - 2 mm
  * negative growth mass: sum max(0, L(t) - L(t+1)) in mm (GT values are the measurement-noise floor)
  * length MAE |L_pred - L_gt| per frame, split into observed / held-out frames and by extrapolation horizon
    (held-out frame +1, +2, +3 after the last observed frame)
Writes <exp>/growth_metrics[_mode].csv (one row per organ frame) and prints the summary. GrowFields' multi-seed layout
(<plant>/run_<k>/...) adds a "run" column and reports each summary value as mean ± std over runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from config import DataConfig
from data import load_benchmark_sequence
from metrics import run_dirs
from ply import read_ply


def length(points: np.ndarray) -> float:
    return float(np.percentile(points[:, 2], 99))


def organ_rows(exp_dir: Path, cfg: DataConfig, modes: list[str], tol_mm: float = 2.0):
    rows = []
    exp_cfg = exp_dir / "experiment_config.json"      # GrowFields exports mark the source here ("rollout_source")
    default_source = json.load(open(exp_cfg)).get("rollout_source", "first") if exp_cfg.exists() else "first"
    for seq_dir in sorted((exp_dir / "sequences").iterdir()):
        seq = seq_dir.name
        frames, tracks, _ = load_benchmark_sequence(cfg, seq)
        for run, rdir in run_dirs(seq_dir):   # GrowFields multi-seed layout run_<k>/, or the plant dir itself
            res = json.load(open(rdir / "results.json"))
            # GrowFields' results.json has no observed_frames: all frames, or every stride-th one for interpolation
            observed = set(res.get("observed_frames", range(0, len(frames), res.get("train_timepoint_stride") or 1)))
            nearest = res.get("source", default_source) == "nearest"
            for t in tracks:
                for r in track_rows(rdir, t, modes, observed, nearest, tol_mm):
                    rows.append({"sequence": seq, "organ_id": t.organ_id, "organ_type": r.pop("organ_type"), **run, **r})
    return rows


def track_rows(rdir: Path, t, modes: list[str], observed: set, nearest: bool, tol_mm: float) -> list[dict]:
    oname = "stem" if t.organ_id == 0 else f"leaf_{t.organ_id}"
    pred_by_frame = {}
    for mode in modes:
        pdir = rdir / "organs" / oname / ("" if mode == "normal" else mode) / "predicted"
        for pf in sorted(pdir.glob("pred_*.ply")) if pdir.exists() else []:
            day = int(re.search(r"_D(\d+)_", pf.name).group(1))
            pred_by_frame[int(np.where(t.days == day)[0][0])] = length(read_ply(pf)["points"])
    rows = []
    if not pred_by_frame:
        return rows
    s0 = min(pred_by_frame) - 1                      # rollout source frame: prediction == GT there
    L_gt = [length(p) for p in t.points]
    last_obs = max(i for i, g in enumerate(t.frame_idx) if g in observed)
    prev = L_gt[s0]
    for lt in range(s0 + 1, len(t.days)):
        if lt not in pred_by_frame:
            continue
        Lp = pred_by_frame[lt]; g = t.frame_idx[lt]; ref = lt - 1
        if nearest:   # reference = the prediction's own source frame (GT there)
            ref = max(i for i, gg in enumerate(t.frame_idx) if gg in observed and i < lt); prev = L_gt[ref]
        rows.append({"organ_type": oname, "timepoint": int(t.days[lt]),
                     "observed": int(g in observed), "horizon": 0 if g in observed else lt - last_obs,
                     "L_pred": Lp, "L_gt": L_gt[lt], "abs_err": abs(Lp - L_gt[lt]),
                     "dL_pred": Lp - prev, "dL_gt": L_gt[lt] - L_gt[ref],
                     "violation_pred": int(Lp < prev - tol_mm), "violation_gt": int(L_gt[lt] < L_gt[ref] - tol_mm),
                     "neg_mass_pred": max(0.0, prev - Lp), "neg_mass_gt": max(0.0, L_gt[ref] - L_gt[lt])})
        prev = Lp
    return rows


def summarise(rows: list[dict]) -> dict:
    def m(key, sel=lambda r: True):
        v = [r[key] for r in rows if sel(r)]
        return float(np.mean(v)) if v else float("nan")
    s = {"n_organ_frames": len(rows),
         "violation_rate_pred": m("violation_pred"), "violation_rate_gt": m("violation_gt"),
         "neg_growth_mass_mm_pred": float(sum(r["neg_mass_pred"] for r in rows)),
         "neg_growth_mass_mm_gt": float(sum(r["neg_mass_gt"] for r in rows)),
         "length_mae_mm_all": m("abs_err"), "length_mae_mm_observed": m("abs_err", lambda r: r["observed"] == 1),
         "length_mae_mm_heldout": m("abs_err", lambda r: r["observed"] == 0),
         "rate_mae_mm_per_frame": float(np.mean([abs(r["dL_pred"] - r["dL_gt"]) for r in rows])) if rows else float("nan")}
    for h in (1, 2, 3):
        s[f"length_mae_mm_horizon+{h}"] = m("abs_err", lambda r, h=h: r["horizon"] == h)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True, type=Path)
    ap.add_argument("--mode", default="all", choices=["normal", "interpolation", "extrapolation", "all"])
    a = ap.parse_args()
    modes = ["normal", "interpolation", "extrapolation"] if a.mode == "all" else [a.mode]
    rows = organ_rows(a.experiment_dir, DataConfig(), modes)
    suffix = "" if a.mode == "all" else f"_{a.mode}"
    with open(a.experiment_dir / f"growth_metrics{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    runs = sorted({r["run"] for r in rows if "run" in r})
    if runs:   # GrowFields multi-seed layout: summarise each run, report mean ± std (sample) over runs
        per_run = [summarise([r for r in rows if r["run"] == k]) for k in runs]
        s = {"runs": len(runs)}
        for k in per_run[0]:
            v = np.array([p[k] for p in per_run], dtype=float); v = v[~np.isnan(v)]
            s[k], s[k + "_std"] = (float(v.mean()) if len(v) else float("nan")), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            print(f"{k:28s} {s[k]:.4f} ± {s[k + '_std']:.4f}  (n={len(runs)} runs)")
    else:
        s = summarise(rows)
        for k, v in s.items():
            print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")
    json.dump(s, open(a.experiment_dir / f"growth_summary{suffix}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
