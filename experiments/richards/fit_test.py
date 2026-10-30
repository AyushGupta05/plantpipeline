"""RichardsNet on the GrowFields benchmark: MAP curve fit per organ, closed-form decoding, GF-layout PLY export.

  python fit_test.py --variant headline|interpolation|extrapolation [--split test|val] [--family richards|beta]
                     [--template first|last|nearest] [--out runs/<name>] [--sequences a,b] [--lambda_prior 1] [--sigma_L 5]
                     [--log_space] [--no_rank] [--no_gt_fill_single_frame] [--residual_mlp]

Per plant and organ: observed frames = all / every 2nd global frame / all but the last 3 (GrowFields' protocols);
lengths L and widths W of the observed frames (99th-percentile axial / radial extent in the canonical frame, only
the GF point split's train points) -> MAP fit of the 4 curve parameters with the species(/rank) Gaussian prior as a
quadratic penalty (common.fit_theta); organs with a single observed frame take the prior mean; per-organ
allometry exponent beta_i = shrunk within-organ slope of log W on log L. Every frame after the organ's first
observed frame is decoded in closed form from the template frame's points (the first observed frame, the last
observed frame with inverse scaling, or `nearest`: the organ's latest observed frame s < t, as MonoGrow's
fit.source=nearest), is_train inherited from the template, the GT tip scaled the same way (first / last: the organ's
first tip-annotated frame, as GrowFields advects it; nearest: MonoGrow's rule, the tip of s or of the latest earlier
annotated frame, else the first annotated frame after s), re-posed with the GT transform of the target day and written
exactly like monogrow/fit_test.py:
  <out>/sequences/<plant>/full_plant/[<mode>/]{predicted,ground_truth}/{pred,gt}_<plant>_D<dd>.ply
  <out>/sequences/<plant>/organs/<stem|leaf_k>/[<mode>/]{predicted,ground_truth}/..._D<dd>_<stem|leafk>.ply
plus experiment_config.json / results.json, so monogrow/metrics.py, monogrow/growth_metrics.py and GrowFields'
compute_metrics_*.py score it unchanged. GT fill rule as GrowFields: organs with < 2 observed frames are copied
from GT in the full-plant files (--no_gt_fill_single_frame predicts them from the prior mean instead).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import HERE, fit_theta, group_of, leaf_ranks, length, width  # noqa: F401  (threads / sys.path)
from canon import apply_transform
from config import DataConfig
from curves import CURVE, decode
from data import gf_split_plants, load_benchmark_sequence
from ply import write_ply
from prior import load_prior


def observed_frames(variant: str, n: int) -> list[int]:
    return {"headline": list(range(n)), "interpolation": list(range(0, n, 2)), "extrapolation": list(range(0, n - 3))}[variant]


def fit_organ(track, obs: list[int], rank, prior: dict, a) -> tuple[np.ndarray, float]:
    """MAP curve parameters theta and allometry exponent beta_i from the observed frames' train points."""
    g = group_of(track); P = prior["groups"][a.family][g]; A = prior["allometry"][g]
    mu = P["mu"].copy()
    if a.use_rank and rank is not None:
        mu += P["rank_slope"] * (rank - P["rank_mean"])
    pts = [track.points[t][track.is_train[t]] if track.is_train[t].any() else track.points[t] for t in obs]
    ages = track.days[obs] - track.days[obs[0]]
    L = np.array([length(p) for p in pts]); W = np.array([width(p) for p in pts])
    theta = fit_theta(a.family, ages, L, mu, np.linalg.inv(P["cov"]), lam=a.lambda_prior, sigma_L=a.sigma_L, log_space=a.log_space)
    x, y = np.log(np.maximum(L, 1e-3)), np.log(np.maximum(W, 1e-3)); x -= x.mean(); y -= y.mean()
    lam_b = (A["sigma_eps"] / max(A["sigma_beta"], 1e-3)) ** 2
    beta = float((x @ y + lam_b * A["beta"]) / (x @ x + lam_b))
    return theta, beta


def predict_track(track, obs: list[int], theta, beta: float, a) -> dict[int, list]:
    """{local frame lt > first observed: [canonical points, canonical tip | None, local template frame s]}.
    Template s = the first observed frame, the last observed frame, or (nearest) the latest observed frame s < lt;
    points and tip are scaled by the fitted curve's ratio L(a_lt) / L(a_s) (and its allometric width ratio)."""
    s0 = obs[0]; n = len(track.days)
    src = {lt: obs[0] if a.template == "first" else obs[-1] if a.template == "last" else max(o for o in obs if o < lt)
           for lt in range(s0 + 1, n)}
    Lc = CURVE[a.family](theta, track.days - track.days[s0])            # curve length at every frame of the track
    out = {lt: [decode(track.points[s].astype(np.float64), Lc[lt] / Lc[s], beta), None, s] for lt, s in src.items()}
    for s in sorted(set(src.values())):
        tg = [lt for lt, ss in src.items() if ss == s]
        if a.template == "nearest":   # MonoGrow: tip of s if annotated, else of the latest earlier annotated frame
            ta = next((i for i in range(s, -1, -1) if track.tips[i] is not None), None)
            if ta is None:            # else GrowFields' rule: the first annotated frame after s, even if held out
                ta = next((i for i in range(s, tg[-1]) if track.tips[i] is not None), None)
        else:                         # GrowFields: the organ's first tip-annotated frame from the first observed frame on
            ta = next((i for i in range(s0, n) if track.tips[i] is not None), None)
        for lt in tg if ta is not None else []:
            if lt > ta:
                out[lt][1] = decode(np.asarray(track.tips[ta], np.float64)[None], Lc[lt] / Lc[ta], beta)[0]
    return out


def fit_sequence(seq: str, prior: dict, a, exp_dir: Path, log):
    frames, tracks, _ = load_benchmark_sequence(DataConfig(), seq)
    obs_g = observed_frames(a.variant, len(frames))
    mode = {"headline": None, "interpolation": "interpolation", "extrapolation": "extrapolation"}[a.variant]
    obs = [[i for i, g in enumerate(t.frame_idx) if g in obs_g] for t in tracks]
    fit_idx = [i for i, o in enumerate(obs) if len(o) >= (2 if a.gt_fill_single_frame else 1)]
    seen = [i for i, o in enumerate(obs) if o]
    ranks = leaf_ranks([tracks[i] for i in seen], [tracks[i].frame_idx[obs[i][0]] for i in seen])
    log(f"{seq}: {len(frames)} frames, observed {obs_g}, {len(tracks)} organ tracks, {len(fit_idx)} fitted")

    t0 = time.time()
    seq_dir = exp_dir / "sequences" / seq
    preds, params = {}, []
    for i in fit_idx:
        t = tracks[i]
        theta, beta = fit_organ(t, obs[i], ranks.get(t.organ_id), prior, a)
        preds[t.organ_id] = predict_track(t, obs[i], theta, beta, a)
        params.append({"organ_id": t.organ_id, "theta": theta.tolist(), "beta": beta, "rank": ranks.get(t.organ_id), "n_obs": len(obs[i])})
    fit_time = time.time() - t0

    for i in fit_idx:
        t = tracks[i]; oname = "stem" if t.organ_id == 0 else f"leaf_{t.organ_id}"; odir = seq_dir / "organs" / oname
        odir.mkdir(parents=True, exist_ok=True)
        json.dump({"organ_id": t.organ_id, "organ_type": oname, "model_type": "richards", "n_timepoints_present": len(t.days)}, open(odir / "metrics.json", "w"))
        for lt, (pc, tip, s) in preds[t.organ_id].items():
            sub = odir if t.frame_idx[lt] in obs_g else odir / mode
            fn = f"{seq}_D{int(t.days[lt]):02d}_{oname.replace('_', '')}.ply"
            write_ply(sub / "predicted" / f"pred_{fn}", pc, np.full(len(pc), t.organ_id, np.int32), t.is_train[s].astype(np.int32),
                      None if tip is None else tip[None], None if tip is None else np.array([t.organ_id], np.int32))
            gt_tip = t.tips[lt]
            write_ply(sub / "ground_truth" / f"gt_{fn}", t.points[lt].astype(np.float64), np.full(len(t.points[lt]), t.organ_id, np.int32),
                      t.is_train[lt].astype(np.int32), None if gt_tip is None else gt_tip[None], None if gt_tip is None else np.array([t.organ_id], np.int32))

    by_id = {t.organ_id: t for t in tracks}
    for g in range(1, len(frames)):
        f = frames[g]; day = int(f["day"]); sub = seq_dir / "full_plant" / (mode if g not in obs_g else "")
        P, L, IT, tips, tlab = [], [], [], [], []
        for oid in sorted(set(int(o) for o in np.unique(f["labels"]) if o >= 0)):
            t = by_id.get(oid); lt = t.frame_idx.index(g) if t is not None and g in t.frame_idx else None
            if oid in preds and lt in preds[oid]:
                pc, tip, s = preds[oid][lt]; Minv = np.linalg.inv(t.M[lt])
                P.append(apply_transform(pc, Minv)); L.append(np.full(len(pc), oid, np.int32)); IT.append(t.is_train[s].astype(np.int32))
                if tip is not None:
                    tips.append(apply_transform(tip[None], Minv)[0]); tlab.append(oid)
            else:   # GrowFields GT fill: organ without a model (first appearance, or seen < 2 times)
                m = f["labels"] == oid
                P.append(f["points"][m]); L.append(np.full(m.sum(), oid, np.int32)); IT.append(f["is_train"][m].astype(np.int32))
        write_ply(sub / "predicted" / f"pred_{seq}_D{day:02d}.ply", np.concatenate(P), np.concatenate(L), np.concatenate(IT),
                  np.array(tips) if tips else None, np.array(tlab, np.int32) if tips else None)
        gt_tips = f["points"][f["tip_idx"]] if len(f["tip_idx"]) else None
        write_ply(sub / "ground_truth" / f"gt_{seq}_D{day:02d}.ply", f["points"], f["labels"].astype(np.int32), f["is_train"].astype(np.int32),
                  gt_tips, f["labels"][f["tip_idx"]].astype(np.int32) if gt_tips is not None else None)

    json.dump({"sequence_name": seq, "model_type": "richards", "variant": a.variant, "observed_frames": obs_g, "source": a.template,
               "aggregate_stats": {"training_time_seconds": fit_time, "total_model_size_mb": 5 * 4 * len(fit_idx) / 2 ** 20, "n_organs_fitted": len(fit_idx)}},
              open(seq_dir / "results.json", "w"), indent=1)
    json.dump(params, open(seq_dir / "curve_params.json", "w"), indent=1)
    log(f"  fitted {len(fit_idx)} organs in {fit_time:.2f}s, wrote {seq_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="headline", choices=["headline", "interpolation", "extrapolation"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--sequences", default="", help="comma-separated subset of the split")
    ap.add_argument("--family", default="richards", choices=["richards", "beta"])
    ap.add_argument("--template", default="first", choices=["first", "last", "nearest"])
    ap.add_argument("--prior", default=str(HERE / "prior.npz"))
    ap.add_argument("--out", default="")
    ap.add_argument("--lambda_prior", type=float, default=1.0, help="weight of the Gaussian prior penalty (inf = prior mean only)")
    ap.add_argument("--sigma_L", type=float, default=5.0, help="length noise scale, mm (or log units with --log_space)")
    ap.add_argument("--log_space", action="store_true")
    ap.add_argument("--no_rank", dest="use_rank", action="store_false")
    ap.add_argument("--no_gt_fill_single_frame", dest="gt_fill_single_frame", action="store_false")
    ap.add_argument("--residual_mlp", action="store_true", help="shared residual eps_psi (not implemented in this pass)")
    a = ap.parse_args()
    if a.residual_mlp:
        raise NotImplementedError("eps_psi residual MLP is left for a later pass (flag reserved)")
    exp_dir = Path(a.out or HERE / "runs" / f"{a.split}_{a.variant}_{a.family}_{a.template}").resolve(); exp_dir.mkdir(parents=True, exist_ok=True)
    logf = open(exp_dir / "fit_log.txt", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    seqs = [s for s in a.sequences.split(",") if s] or gf_split_plants(DataConfig(), a.split)
    dc = DataConfig()
    json.dump({"config": {"dataset": {"dataset_path": str(dc.tp3d_dir.parents[1]), "version": "v2", "label_field": "labels", "aligned_leaves": True},
                          "default_model": {"model_type": "richards"}}, "richards": vars(a), "sequence_names": seqs},
              open(exp_dir / "experiment_config.json", "w"), indent=1)
    prior = load_prior(a.prior)
    t0 = time.time()
    for seq in seqs:
        fit_sequence(seq, prior, a, exp_dir, log)
    log(f"total {time.time() - t0:.1f}s for {len(seqs)} plants -> {exp_dir}")


if __name__ == "__main__":
    main()
