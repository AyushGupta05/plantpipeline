"""Test-time fitting on the GrowFields benchmark and export in GrowFields' exact PLY layout.

  python fit_test.py smoke                                  # SoH2, 100 z-steps
  python fit_test.py full fit.variant=headline              # 8 test plants, all frames observed
  python fit_test.py full fit.variant=interpolation fit.out=runs/full/fit_interp
  python fit_test.py full fit.variant=extrapolation fit.out=runs/full/fit_extrap
  python fit_test.py full fit.split=val ...                 # the 4 GrowFields val plants (tuning)

Per organ: z <- mu_species, then N_z Adam steps on multi-step Chamfer over the OBSERVED frames using only the
point split's train points, + lambda (z-mu)^T Sigma^-1 (z-mu); optional stage 2 fine-tunes theta (lr 1e-6).
Then every organ is rolled out (1-day Euler grid) to every later frame, re-posed with the GT transform of the
target day and written as
  <out>/sequences/<plant>/full_plant/[<mode>/]{predicted,ground_truth}/{pred,gt}_<plant>_D<dd>.ply
  <out>/sequences/<plant>/organs/<stem|leaf_k>/[<mode>/]{predicted,ground_truth}/..._D<dd>_<stem|leafk>.ply
(mode = interpolation | extrapolation for frames the variant held out), plus experiment_config.json and
results.json, so GrowFields' compute_metrics_{fullplant,per_organ}.py and create_latex_table_*.py run unchanged.
Protocol parity with GrowFields: organs with < 2 observed frames get no model and are copied from GT in the
full-plant files (`fit.gt_fill_single_frame=false` predicts them from the prior instead).
Rollout source (`fit.source`, inference only; fitting is unchanged): "first" = GrowFields' protocol, every target
frame is predicted from the organ's first observed frame; "nearest" = from the latest observed frame before the
target (one-day predictions for headline / interpolation, the last observed frame for the 3 extrapolated frames).
Source points = all points of that frame with its is_train mask inherited; a source frame with < 3 train points
gets the GT fill. Tip for "nearest": the source frame's GT tip if annotated, else advected from the latest earlier
annotated frame; if none, GrowFields' rule (first annotated frame after the source, even if held out).
`fit.reuse=<other fit.out>` skips fitting and re-exports that run's fitted codes / fine-tuned theta
(sequences/<plant>/fit_state.pt) with the current source: fitting is identical for both sources, so one fit serves both.
`fit.lambda_w` adds lambda_w * mean ||P_perp w||^2 (the model's radial term) to the fitting loss.
Fitting objective (`fit.objective`): "first" = GrowFields' multi-step CD of one rollout from the first observed frame
through all observed frames (mean over frames); "nearest" = mean over consecutive observed pairs (s, t) of CD between
frame s's train points rolled out to day(t) and frame t's train points, i.e. exactly what `fit.source=nearest` is scored
on. Same chunked Chamfer, prior term, optimiser and steps; export and metrics are untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config as C
from canon import apply_transform
from data import OrganTrack, gf_split_plants, load_benchmark_sequence, to_tensor
from model import MonoGrow, chamfer, multistep_chamfer
from ply import write_ply


def source_frames(obs_i: list[int], n: int, source: str) -> dict[int, int]:
    """Local target frame -> local source frame for one organ with observed local frames `obs_i`."""
    if source == "first":
        return {lt: obs_i[0] for lt in range(obs_i[0] + 1, n)}
    if source == "nearest":
        return {lt: max(o for o in obs_i if o < lt) for lt in range(1, n) if any(o < lt for o in obs_i)}
    raise ValueError(source)


def observed_frames(variant: str, n_frames: int, fc: C.FitConfig) -> list[int]:
    if variant == "headline":
        return list(range(n_frames))
    if variant == "interpolation":
        return list(range(0, n_frames, fc.stride))
    if variant == "extrapolation":
        return list(range(0, n_frames - fc.holdout_last_n))
    raise ValueError(variant)


def load_prior_model(cfg: C.Config, device):
    ck = torch.load(cfg.fit.ckpt, map_location=device, weights_only=False)
    mcfg = C.ModelConfig(**{k: v for k, v in ck["config"]["model"].items()})
    mcfg.step_per_frame = cfg.model.step_per_frame  # integration scheme may be changed at test time
    model = MonoGrow(mcfg).to(device); model.load_state_dict(ck["model"])
    prior = torch.load(cfg.fit.prior, map_location=device, weights_only=False)
    prior["prec"] = torch.linalg.inv(prior["cov"].to(device)); prior["mu"] = prior["mu"].to(device)
    return model, prior


def fit_codes(model, prior, tracks: list[OrganTrack], obs: list[list[int]], fc: C.FitConfig, device, rng, log):
    """Optimise one z per track (then optionally theta) on observed frames / train points. Returns z (n,32)."""
    sp = [C.SPECIES_ID[t.species] for t in tracks]
    mu = torch.stack([prior["mu"][s] for s in sp]); prec = torch.stack([prior["prec"][s] for s in sp])
    z = torch.nn.Parameter(mu.clone())

    def loss_fn():
        total = 0.0
        for i, t in enumerate(tracks):
            o = obs[i]
            def pick(f, n):
                pts = t.points[f][t.is_train[f]] if t.is_train[f].any() else t.points[f]
                return pts[rng.integers(0, len(pts), n)]
            x0 = to_tensor(pick(o[0], fc.points_per_organ), device)   # (sampled first: keeps objective=first bit-identical to before)
            tg = [to_tensor(pick(f, fc.points_per_organ), device) / model.cfg.scale_mm for f in o[1:]]
            if fc.objective == "first":   # GrowFields: one rollout from the first observed frame through every observed frame
                preds, reg = model.rollout(x0, t.days[o], float(t.ages[o[0]]), float(t.plant_ages[o[0]]), z[i], sp[i], collect_reg=fc.lambda_w > 0)
                cd, reg_w = multistep_chamfer([p / model.cfg.scale_mm for p in preds], tg), reg["w"]
            elif fc.objective == "nearest":   # what source=nearest scores: frame s rolled out to the next observed frame t, one CD per pair
                cds, regs = [], []
                for s, (k, tk) in zip(o[:-1], zip(o[1:], tg)):
                    xs = x0 if s == o[0] else to_tensor(pick(s, fc.points_per_organ), device)
                    preds, reg = model.rollout(xs, t.days[[s, k]], float(t.ages[s]), float(t.plant_ages[s]), z[i], sp[i], collect_reg=fc.lambda_w > 0)
                    cds.append(chamfer(preds[0] / model.cfg.scale_mm, tk)); regs.append(reg["w"])
                cd, reg_w = torch.stack(cds).mean(), torch.stack(regs).mean()
            else:
                raise ValueError(fc.objective)
            d = z[i] - mu[i]
            total = total + cd + fc.lambda_prior * (d @ prec[i] @ d) + fc.lambda_w * reg_w
        return total / len(tracks)

    for stage, steps, params in (("z", fc.n_z, [{"params": [z], "lr": fc.lr_z}]),
                                 ("theta", fc.n_theta, [{"params": [z], "lr": fc.lr_z}, {"params": model.parameters(), "lr": fc.lr_theta}])):
        if steps == 0:
            continue
        opt = torch.optim.Adam(params)
        for step in range(1, steps + 1):
            loss = loss_fn(); opt.zero_grad(); loss.backward(); opt.step()
            if step % max(steps // 10, 1) == 0 or step == 1:
                log(f"    stage {stage} step {step:5d} loss {loss.item():.6f} (CD {loss.item() * model.cfg.scale_mm ** 2:.3f} mm^2-ish)")
    return z.detach()


@torch.no_grad()
def predict_track(model, t: OrganTrack, src: dict[int, int], z, species: int, device, source: str, min_train: int = 3):
    """Roll out all points of each source frame to its target frames; returns {local t: [pred_mm canonical, tip|None, s]}."""
    out = {}
    for s in sorted(set(src.values())):
        tg = sorted(lt for lt, ss in src.items() if ss == s)
        if t.is_train[s].sum() < min_train:   # too few train points to be a source: GT fill downstream
            continue
        preds, _ = model.rollout(to_tensor(t.points[s], device), t.days[s:tg[-1] + 1], float(t.ages[s]), float(t.plant_ages[s]), z, species)
        for lt in tg:
            out[lt] = [preds[lt - s - 1].cpu().numpy().astype(np.float64), None, s]
        a = next((i for i in range(s, -1, -1) if t.tips[i] is not None), None) if source == "nearest" else None
        if a is None:   # GrowFields: the tip starts at the organ's first tip-annotated frame (>= s), even if held out
            a = next((i for i in range(s, tg[-1]) if t.tips[i] is not None), None)
        if a is not None:
            tips, _ = model.rollout(to_tensor(t.tips[a][None], device), t.days[a:tg[-1] + 1], float(t.ages[a]), float(t.plant_ages[a]), z, species)
            for lt in tg:
                if lt > a:
                    out[lt][1] = tips[lt - a - 1].cpu().numpy()[0].astype(np.float64)
    return out


def fit_sequence(cfg: C.Config, seq: str, model, prior, device, exp_dir, log):
    fc = cfg.fit
    frames, tracks, split = load_benchmark_sequence(cfg.data, seq)
    obs_g = observed_frames(fc.variant, len(frames), fc)
    mode = {"headline": None, "interpolation": "interpolation", "extrapolation": "extrapolation"}[fc.variant]
    obs = [[i for i, g in enumerate(t.frame_idx) if g in obs_g] for t in tracks]
    min_obs = 2 if fc.gt_fill_single_frame else 1
    fit_idx = [i for i, o in enumerate(obs) if len(o) >= min_obs]
    log(f"{seq}: {len(frames)} frames, observed {obs_g}, {len(tracks)} organ tracks, {len(fit_idx)} fitted")

    seq_dir = exp_dir / "sequences" / seq
    if fc.reuse:   # re-export a previous fit of this plant (same recipe) with the current fit.source
        st = torch.load(Path(fc.reuse) / "sequences" / seq / "fit_state.pt", map_location=device, weights_only=False)
        assert st["organ_id"] == [tracks[i].organ_id for i in fit_idx], f"{seq}: fitted organs differ from {fc.reuse}"
        z, fit_time = st["z"], st["fit_time"]; model.load_state_dict(st["model"])
        log(f"  reusing codes/theta from {fc.reuse}")
    else:
        t0 = time.time()
        rng = np.random.default_rng(fc.seed); torch.manual_seed(fc.seed)
        fit_tracks = [tracks[i] for i in fit_idx]
        z = fit_codes(model, prior, fit_tracks, [obs[i] for i in fit_idx], fc, device, rng, log) if fit_idx else None
        fit_time = time.time() - t0
        seq_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"z": z, "model": model.state_dict(), "fit_time": fit_time, "organ_id": [tracks[i].organ_id for i in fit_idx]}, seq_dir / "fit_state.pt")

    # predictions per organ (canonical) and per-organ PLYs
    preds = {}   # organ_id -> {local t: (canonical pred, canonical tip, local source frame)}
    for j, i in enumerate(fit_idx):
        t = tracks[i]
        preds[t.organ_id] = predict_track(model, t, source_frames(obs[i], len(t.days), fc.source), z[j], C.SPECIES_ID[t.species], device, fc.source)
        oname = "stem" if t.organ_id == 0 else f"leaf_{t.organ_id}"
        odir = seq_dir / "organs" / oname
        odir.mkdir(parents=True, exist_ok=True)
        json.dump({"organ_id": t.organ_id, "organ_type": oname, "model_type": "monogrow", "n_timepoints_present": len(t.days)},
                  open(odir / "metrics.json", "w"))
        for lt, (pc, tip, s) in preds[t.organ_id].items():
            g = t.frame_idx[lt]; sub = odir if g in obs_g else odir / mode
            day = int(t.days[lt]); fn = f"{seq}_D{day:02d}_{oname.replace('_', '')}.ply"
            write_ply(sub / "predicted" / f"pred_{fn}", pc, np.full(len(pc), t.organ_id, np.int32), t.is_train[s].astype(np.int32),
                      None if tip is None else tip[None], None if tip is None else np.array([t.organ_id], np.int32))
            gt_tip = t.tips[lt]
            write_ply(sub / "ground_truth" / f"gt_{fn}", t.points[lt].astype(np.float64), np.full(len(t.points[lt]), t.organ_id, np.int32),
                      t.is_train[lt].astype(np.int32), None if gt_tip is None else gt_tip[None], None if gt_tip is None else np.array([t.organ_id], np.int32))

    # full plant: every global frame >= 1, organs re-posed with the GT transform of the target day
    by_id = {t.organ_id: t for t in tracks}
    for g in range(1, len(frames)):
        f = frames[g]; day = int(f["day"]); sub = seq_dir / "full_plant" / (mode if g not in obs_g else "")
        P, L, IT, tips, tlab = [], [], [], [], []
        for oid in sorted(set(int(o) for o in np.unique(f["labels"]) if o >= 0)):
            t = by_id.get(oid); lt = t.frame_idx.index(g) if t is not None and g in t.frame_idx else None
            if t is not None and oid in preds and lt in preds[oid]:
                pc, tip, s = preds[oid][lt]; Minv = np.linalg.inv(t.M[lt])
                P.append(apply_transform(pc, Minv)); L.append(np.full(len(pc), oid, np.int32)); IT.append(t.is_train[s].astype(np.int32))
                if tip is not None:
                    tips.append(apply_transform(tip[None], Minv)[0]); tlab.append(oid)
            else:   # GrowFields GT fill: organ without a model (first appearance, seen < 2 times, or source frame too sparse)
                m = f["labels"] == oid
                P.append(f["points"][m]); L.append(np.full(m.sum(), oid, np.int32)); IT.append(f["is_train"][m].astype(np.int32))
        write_ply(sub / "predicted" / f"pred_{seq}_D{day:02d}.ply", np.concatenate(P), np.concatenate(L), np.concatenate(IT),
                  np.array(tips) if tips else None, np.array(tlab, np.int32) if tips else None)
        gt_tips = f["points"][f["tip_idx"]] if len(f["tip_idx"]) else None
        write_ply(sub / "ground_truth" / f"gt_{seq}_D{day:02d}.ply", f["points"], f["labels"].astype(np.int32), f["is_train"].astype(np.int32),
                  gt_tips, f["labels"][f["tip_idx"]].astype(np.int32) if gt_tips is not None else None)

    size_mb = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    json.dump({"sequence_name": seq, "model_type": "monogrow", "variant": fc.variant, "observed_frames": obs_g, "source": fc.source,
               "aggregate_stats": {"training_time_seconds": fit_time, "total_model_size_mb": size_mb, "n_organs_fitted": len(fit_idx)}},
              open(seq_dir / "results.json", "w"), indent=1)
    if z is not None:
        np.savez(seq_dir / "codes.npz", z=z.cpu().numpy(), organ_id=[tracks[i].organ_id for i in fit_idx])
    log(f"  fitted in {fit_time:.1f}s, wrote {seq_dir}")


def main(argv):
    cfg = C.load(argv[0], argv[1:]); fc = cfg.fit
    torch.set_num_threads(fc.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = fc.out; exp_dir.mkdir(parents=True, exist_ok=True)
    logf = open(exp_dir / "fit_log.txt", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    json.dump({"config": {"dataset": {"dataset_path": str(cfg.data.tp3d_dir.parents[1]), "version": "v2", "label_field": "labels",
                                      "aligned_leaves": True}, "default_model": {"model_type": "monogrow"}},
               "monogrow": C.as_dict(cfg), "sequence_names": list(fc.sequences) or gf_split_plants(cfg.data, fc.split)},
              open(exp_dir / "experiment_config.json", "w"), indent=1)
    model, prior = load_prior_model(cfg, device)
    theta0 = {k: v.clone() for k, v in model.state_dict().items()}
    for seq in (fc.sequences or gf_split_plants(cfg.data, fc.split)):
        model.load_state_dict(theta0)   # stage-2 fine-tuning never leaks between plants
        fit_sequence(cfg, seq, model, prior, device, exp_dir, log)


if __name__ == "__main__":
    main(sys.argv[1:] or ["smoke"])
