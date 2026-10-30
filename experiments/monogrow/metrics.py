"""Self-contained re-implementation of GrowFields' evaluation (compute_metrics_{fullplant,per_organ}.py +
create_latex_table_*.py --per_sequence), so we are not blocked on their scripts.

  python metrics.py --experiment_dir runs/full/fit_headline [--mode normal|interpolation|extrapolation] [--compare_gf]

CD  = mean_p min_g ||p-g||^2 + mean_g min_p ||p-g||^2 (mm^2) on test points only (is_train == 0 on both sides).
EPE = mean ||pred_tip - gt_tip|| (mm) over tips matched by organ id, skipping each tip's first-appearance frame
      (first frame of the plant in which that organ has an annotated tip, frame 0 included).
Aggregation: full plant: frames -> plant -> unweighted mean over plants. Per organ: frames -> organ -> plant -> mean
over plants, dropping organs with < 2 scored rows in the CSV being aggregated (GrowFields' table scripts apply
`--min_timepoints_per_organ 2` to the training-frame, interpolation and extrapolation CSVs alike). Checks n_tips sums
(192 / 189 on the 8 test plants, headline). `--compare_gf` diffs against GrowFields' CSVs if they exist in the experiment dir.
Multi-seed GrowFields layout (<plant>/run_<k>/...) is aggregated like their table scripts: per run, mean ± std over runs
per plant (CSV rows get a "run" column, the table a "runs" column); the Mean row's ± is the std over runs of the per-run mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from config import DataConfig
from data import load_tp3d_frames
from model import chamfer
from ply import read_ply


def cd_test(pred: dict, gt: dict) -> tuple[float | None, int]:
    pm, gm = pred["is_train"] == 0, gt["is_train"] == 0
    if not (pm.any() and gm.any()):
        return None, 0
    a, b = torch.from_numpy(pred["points"][pm]).float(), torch.from_numpy(gt["points"][gm]).float()
    return float(chamfer(a, b)), int(gm.sum())


def tip_epe(pred: dict, gt: dict, skip_ids: set[int]) -> tuple[float | None, int | None]:
    """Matched-by-organ EPE; skip_ids = organs whose tip first appears in this frame."""
    if "is_leaf_tip" not in pred or "is_leaf_tip" not in gt:
        return None, None
    pt, gtp = pred["is_leaf_tip"].astype(bool), gt["is_leaf_tip"].astype(bool)
    if not (pt.any() and gtp.any()):
        return None, None
    pl, gl = pred["tip_organ_label"][pt], gt["tip_organ_label"][gtp]
    common = sorted((set(pl) & set(gl)) - skip_ids)
    if not common:
        return None, 0 if (set(pl) & set(gl)) == set() else None
    P = np.stack([pred["points"][pt][np.where(pl == o)[0][0]] for o in common])
    G = np.stack([gt["points"][gtp][np.where(gl == o)[0][0]] for o in common])
    return float(np.linalg.norm(P - G, axis=1).mean()), len(common)


def first_tip_frame(frames: list[dict]) -> tuple[dict[int, int], dict[int, int]]:
    first, day2frame = {}, {}
    for g, f in enumerate(frames):
        day2frame[int(f["day"])] = g
        for o in f["labels"][f["tip_idx"]]:
            first.setdefault(int(o), g)
    return first, day2frame


def day_of(name: str) -> int:
    return int(re.search(r"_D(\d+)(?:_|\.)", name).group(1))


def run_dirs(seq_dir: Path) -> list[tuple[dict, Path]]:
    """GrowFields multi-seed layout <plant>/run_<k>/{full_plant,organs,results.json} -> [({"run": k}, dir)];
    the single-run layout (full_plant directly under the plant dir) gives [({}, seq_dir)], so its rows/CSVs are unchanged."""
    runs = sorted(seq_dir.glob("run_*"), key=lambda d: int(d.name.split("_")[1]))
    return [({"run": int(d.name.split("_")[1])}, d) for d in runs] or [({}, seq_dir)]


def evaluate(exp_dir: Path, tp3d_dir: Path, mode: str, min_tp: int = 2):
    sub = "" if mode == "normal" else mode
    full_rows, organ_rows = [], []
    for seq_dir in sorted((exp_dir / "sequences").iterdir()):
        seq = seq_dir.name
        frames = load_tp3d_frames(tp3d_dir, seq)
        first, day2frame = first_tip_frame(frames)
        for run, rdir in run_dirs(seq_dir):
            pdir = rdir / "full_plant" / sub / "predicted"
            for pf in sorted(pdir.glob("pred_*.ply")) if pdir.exists() else []:
                pred, gt = read_ply(pf), read_ply(pdir.parent / "ground_truth" / pf.name.replace("pred_", "gt_"))
                day = day_of(pf.name); g = day2frame[day]
                cd, n_test = cd_test(pred, gt)
                epe, n_tips = tip_epe(pred, gt, {o for o, fg in first.items() if fg == g})
                full_rows.append({"sequence": seq, **run, "timepoint": day, "cd_test": cd, "n_points_test": n_test, "epe": epe, "n_tips": n_tips})
            odir = rdir / "organs"
            for org in sorted(odir.iterdir()) if odir.exists() else []:
                oid = 0 if org.name == "stem" else int(org.name.split("_")[1])
                pdir = org / sub / "predicted"
                for pf in sorted(pdir.glob("pred_*.ply")) if pdir.exists() else []:
                    pred, gt = read_ply(pf), read_ply(pdir.parent / "ground_truth" / pf.name.replace("pred_", "gt_"))
                    day = day_of(pf.name); g = day2frame[day]
                    cd, n_test = cd_test(pred, gt)
                    epe, n_tips = (None, None)
                    if oid > 0 and first.get(oid) != g:
                        epe, n_tips = tip_epe(pred, gt, set())
                    organ_rows.append({"sequence": seq, "organ_id": oid, "organ_type": org.name, **run, "timepoint": day, "cd_test": cd,
                                       "n_points_test": n_test, "epe": epe, "n_tips": n_tips})
    return full_rows, organ_rows


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _std(xs):   # over runs, sample std like pandas (GF's create_latex_table_*: 0 for a single run)
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def aggregate(full_rows, organ_rows, mode: str, min_tp: int = 2) -> dict:
    """GF aggregation: frames -> (plant, run) -> mean over runs per plant -> unweighted mean over plants. Per organ the
    >=2-frame filter and the organ mean are applied per run. With several runs the summary also carries per-plant std
    over runs ("full_std"/"organ_std"), the run count ("runs") and, for the Mean row, the std over runs of the per-run
    mean over plants (GF compute_mean_column_run_std). n_tips are checked on the first run."""
    plants = sorted({r["sequence"] for r in full_rows} | {r["sequence"] for r in organ_rows})
    out = {"full": {}, "organ": {}}
    per_run = {"full": {}, "organ": {}}          # (plant, run) -> (cd, epe)
    multi = any("run" in r for r in full_rows + organ_rows)
    for p in plants:
        runs = sorted({r.get("run", 1) for r in full_rows + organ_rows if r["sequence"] == p})
        for k in runs:
            rows = [r for r in full_rows if r["sequence"] == p and r.get("run", 1) == k]
            per_run["full"][p, k] = (_mean([r["cd_test"] for r in rows]), _mean([r["epe"] for r in rows]))
            per_organ = {}
            for r in organ_rows:
                if r["sequence"] == p and r.get("run", 1) == k:
                    per_organ.setdefault(r["organ_id"], []).append(r)
            kept = {o: rs for o, rs in per_organ.items() if len(rs) >= min_tp}   # GF applies this in every mode, per run
            per_run["organ"][p, k] = (_mean([_mean([r["cd_test"] for r in rs]) for rs in kept.values()]),
                                      _mean([_mean([r["epe"] for r in rs]) for rs in kept.values()]))
            if k == runs[0]:
                n_tips_first = sum(int(r["n_tips"] or 0) for rs in kept.values() for r in rs)
        for t in ("full", "organ"):
            vals = [per_run[t][p, k] for k in runs]
            out[t][p] = (_mean([v[0] for v in vals]), _mean([v[1] for v in vals]))
            if multi:
                out.setdefault(t + "_std", {})[p] = (_std([v[0] for v in vals]), _std([v[1] for v in vals]))
        out["organ"][p + "_n_tips"] = n_tips_first
        if multi:
            out.setdefault("runs", {})[p] = len(runs)
    out["full"]["Mean"] = (_mean([v[0] for k, v in out["full"].items()]), _mean([v[1] for k, v in out["full"].items()]))
    out["organ"]["Mean"] = (_mean([v[0] for k, v in out["organ"].items() if not k.endswith("_n_tips")]),
                            _mean([v[1] for k, v in out["organ"].items() if not k.endswith("_n_tips")]))
    if multi:
        for t in ("full", "organ"):
            by_run = {}
            for (p, k), v in per_run[t].items():
                by_run.setdefault(k, []).append(v)
            out[t + "_std"]["Mean"] = (_std([_mean([v[0] for v in vs]) for vs in by_run.values()]),
                                       _std([_mean([v[1] for v in vs]) for vs in by_run.values()]))
    first_run = {p: min(r.get("run", 1) for r in full_rows if r["sequence"] == p) for p in plants if any(r["sequence"] == p for r in full_rows)}
    out["n_tips_full"] = sum(int(r["n_tips"] or 0) for r in full_rows if r.get("run", 1) == first_run[r["sequence"]])
    out["n_tips_organ"] = sum(v for k, v in out["organ"].items() if k.endswith("_n_tips"))
    return out


def write_csv(rows, path):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def compare_gf(exp_dir: Path, full_rows, organ_rows, mode: str):
    suffix = "" if mode == "normal" else f"_{mode}"
    for name, rows, key in (("fullplant_metrics", full_rows, ("sequence", "timepoint")),
                            ("all_metrics", organ_rows, ("sequence", "organ_id", "timepoint"))):
        path = exp_dir / f"{name}{suffix}.csv"
        if not path.exists():
            print(f"  (no GrowFields {path.name} to compare)"); continue
        gf = list(csv.DictReader(open(path)))
        if gf and "run" in gf[0] and rows and "run" in rows[0]:
            key = key + ("run",)
        gf = {tuple(str(r[k]) for k in key): r for r in gf}
        ours = {tuple(str(r[k]) for k in key): r for r in rows}
        assert set(gf) == set(ours), f"{name}: row sets differ: {set(gf) ^ set(ours)}"
        for col in ("cd_test", "epe", "n_tips"):
            d = [abs(float(gf[k][col]) - float(ours[k][col])) for k in gf
                 if gf[k][col] not in ("", "nan") and ours[k][col] is not None]
            n_none = sum((gf[k][col] in ("", "nan")) != (ours[k][col] is None) for k in gf)
            print(f"  {name}.{col}: {len(d)} rows, max |GF - ours| = {max(d) if d else 0:.2e}, None-mismatch rows = {n_none}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True, type=Path)
    ap.add_argument("--tp3d_dir", type=Path, default=DataConfig().tp3d_dir)
    ap.add_argument("--mode", default="normal", choices=["normal", "interpolation", "extrapolation"])
    ap.add_argument("--compare_gf", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    full_rows, organ_rows = evaluate(a.experiment_dir, a.tp3d_dir, a.mode)
    suffix = "" if a.mode == "normal" else f"_{a.mode}"
    write_csv(full_rows, a.experiment_dir / f"monogrow_fullplant{suffix}.csv")
    write_csv(organ_rows, a.experiment_dir / f"monogrow_per_organ{suffix}.csv")
    agg = aggregate(full_rows, organ_rows, a.mode)
    multi = "runs" in agg
    w = 14 if multi else 9                                   # "  1.52±0.07" needs the wider column
    print(f"{'plant':28s} " + ("runs " if multi else "") + " ".join(f"{h:>{w}s}" for h in ("full CD", "full EPE", "organ CD", "organ EPE")))
    for p in agg["full"]:
        cells = [(agg[t][p][i], agg[t + "_std"][p][i] if multi else None) for t in ("full", "organ") for i in (0, 1)]
        print(f"{p:28s} " + (f"{agg['runs'].get(p, ''):>4} " if multi else "")
              + " ".join(f"{v:{w}.2f}" if s is None else f"{v:9.2f}±{s:4.2f}" for v, s in cells))
    print(f"n_tips: full plant {agg['n_tips_full']} (192 expected on the 8 test plants, headline), "
          f"per organ after the >=2-frame filter {agg['n_tips_organ']} (189 expected)"
          + ("; n_tips and the >=2-frame filter counted on the first run, ± = std over runs" if multi else ""))
    json.dump(agg, open(a.experiment_dir / f"monogrow_summary{suffix}.json", "w"), indent=1)
    if a.compare_gf:
        compare_gf(a.experiment_dir, full_rows, organ_rows, a.mode)


if __name__ == "__main__":
    main()
