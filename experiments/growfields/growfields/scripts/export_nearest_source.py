#!/usr/bin/env python3
"""Re-export a trained GrowFields run with a different rollout source. No retraining.

GrowFields predicts every frame of an organ by rolling the organ's FIRST observed frame
forward through all later frames ("first" source). This script reloads the saved model
of a finished run and writes a second export in GrowFields' exact layout where each
target frame is instead predicted from the organ's LATEST observed frame before it
("nearest" source, the same inference-time rule as MonoGrow's `fit.source=nearest`):

  * headline / interpolation: source = the observed frame right before the target
    (one-step predictions; for interpolation an odd frame from the even frame before it)
  * extrapolation (holdout last 3): source = the last observed frame, for all 3 targets

Source points = all points of the source frame (train + test), their is_train mask
inherited by the prediction (so test points are still scored); integration with the
run's own integrator settings (Euler sub-steps / odeint) and the trained field + organ
codes, per-organ normalisation and day-based t_norm exactly as at training; held-out
days use GrowFields' fractional rule (training sub-step, Euler remainder); leaves are
re-posed with the target day's GT transform; organs without a model (seen < 2 times,
or no observed frame before the target) are copied from ground truth into the
full-plant file, as GrowFields does.

Tip rule for "nearest" (MonoGrow's): the GT tip of the source frame if annotated there,
else the tip of the latest earlier observed annotated frame advected through the target,
else GrowFields' rule (first observed annotated frame between source and target).
`--source first` reproduces GrowFields' own export from the saved model (validation).

Output: a sibling experiment dir `<experiment_dir>_<source>/` (or --out_dir) holding
experiment_config.json and sequences/<seq>/{config.yaml,results.json,train_test_split.json,
full_plant/[<mode>/]{predicted,ground_truth}, organs/<organ>/[<mode>/]{predicted,ground_truth}}
so scripts/compute_metrics_fullplant.py / compute_metrics_per_organ.py (with
--interpolation / --extrapolation for held-out runs) score it unchanged.

Usage (CWD = the GrowFields work dir, i.e. the parent of this repo checkout, because the
configs use the relative dataset path):
    python growfields/scripts/export_nearest_source.py --experiment_dir runs/<cfg>/results/<exp> \
        [--sequences maize_control_plant2 ...] [--source nearest|first] \
        [--variant auto|headline|interpolation|extrapolation] [--out_dir DIR] [--device cpu]
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling script import
from generate_interpolation import (  # noqa: E402
    _load_continuous,
    _load_encoder_decoder,
    _load_shared_continuous,
)

from growfields.models.integration import (  # noqa: E402
    _FIXED_STEP_METHODS,
    _integrate_segment,
    _integrate_time_span,
    denormalise,
    make_t_norm_helpers,
)
from growfields.models.organ_model_manager import OrganModelManager  # noqa: E402
from growfields.pipeline.data import load_plant_dataset  # noqa: E402
from growfields.pipeline.splits import create_strided_split_file, resolve_point_splits_dir  # noqa: E402
from growfields.utils.io_utils import (  # noqa: E402
    estimate_normals,
    save_ply_with_organ_labels,
    write_json,
)

GLOBAL_MODEL_FILES = {
    "shared_continuous": "model_shared_continuous.pth",
    "encoder_decoder_continuous": "model_encoder_decoder.pth",
}


# ----------------------------------------------------------------------------- loading
def detect_variant(cli_args):
    if cli_args.get("train_holdout_last_n"):
        return "extrapolation"
    if (cli_args.get("train_timepoint_stride") or 1) > 1:
        return "interpolation"
    return "headline"


def find_split_files(exp_dir, seq_dir, seq_name, cli_args, stride, derive_dir=None):
    """(split file the training dataset used, original unstrided point split).

    `derive_dir`: where to derive a strided split when the run has none (only when the
    variant is overridden on the command line; never writes into `exp_dir`)."""
    original = os.path.join(
        resolve_point_splits_dir(cli_args.get("points_split_dir")), f"{seq_name}_split.json"
    )
    if not os.path.isfile(original):
        raise FileNotFoundError(f"original point split not found: {original}")
    train_split = original
    if stride > 1:
        cands = [
            os.path.join(exp_dir, "strided_splits", f"{seq_name}_stride{stride}_split.json"),
            os.path.join(seq_dir, "train_test_split.json"),  # prepare_results_dir's copy
        ]
        train_split = next((c for c in cands if os.path.isfile(c) and _split_stride(c) == stride), None)
        if train_split is None and derive_dir is not None:
            train_split = create_strided_split_file(original, stride, seq_name, derive_dir)
        if train_split is None:
            raise FileNotFoundError(f"no stride-{stride} split for {seq_name} under {exp_dir}")
    return train_split, original


def _split_stride(path):
    with open(path) as f:
        return json.load(f).get("timepoint_stride", 1) or 1


def load_manager(exp_cfg, seq_name, seq_dir, exp_dir, device, variant_override=None, derive_dir=None):
    """Rebuild the OrganModelManager as training built it, then load the saved weights."""
    cfg = exp_cfg["config"]
    cli = exp_cfg.get("cli_args", {})
    model_cfg = dict(cfg["default_model"])
    model_type = model_cfg["model_type"]
    stride = cli.get("train_timepoint_stride") or 1
    holdout = cli.get("train_holdout_last_n") or 0
    if variant_override == "headline":
        stride, holdout = 1, 0
    elif variant_override == "interpolation" and stride == 1:
        stride = 2
    elif variant_override == "extrapolation" and holdout == 0:
        holdout = 3

    train_split, original_split = find_split_files(
        exp_dir, seq_dir, seq_name, cli, stride, derive_dir if variant_override else None)
    dataset_cfg = OmegaConf.create(cfg["dataset"])
    if not dataset_cfg.get("aligned_leaves", True):
        raise NotImplementedError("this export assumes dataset.aligned_leaves: true")
    dataset = load_plant_dataset(
        dataset_cfg, seq_name, split_file=train_split, timepoint_stride=stride,
        normalize=False, device=device,
    )
    organ_cfgs = cfg.get("per_organ_training", {}).get("organ_configs", {}) or {}
    manager = OrganModelManager(
        dataset=dataset, default_config=model_cfg, organ_specific_configs=organ_cfgs,
        device=device, train_holdout_last_n=holdout or None,
        save_normals=cfg.get("output", {}).get("save_normals", True),
    )
    organs_dir = os.path.join(seq_dir, "organs")
    organ_subdirs = sorted(os.listdir(organs_dir)) if os.path.isdir(organs_dir) else []
    if model_type == "shared_continuous":
        _load_shared_continuous(manager, model_cfg, seq_dir, device)
    elif model_type == "continuous":
        _load_continuous(manager, model_cfg, organs_dir, organ_subdirs, device)
    elif model_type == "encoder_decoder_continuous":
        _load_encoder_decoder(manager, model_cfg, seq_dir, device)
    else:
        raise NotImplementedError(f"model_type {model_type!r} (pairwise has no velocity field)")

    # `use_global_tnorm` is only set on the manager while training; recompute it.
    if model_cfg.get("use_global_tnorm", False):
        ranges = [
            manager.organ_full_day_range[o] for o, s in manager.organ_sequences.items()
            if sum(c is not None for c in s["train_points"]) >= 2
        ]
        manager.global_day_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    return manager, original_split, stride, holdout


# ----------------------------------------------------------------------------- export
def organ_clouds(organ_seq):
    """GrowFields' valid clouds: [cat(train, test)] per frame with train points, + is_train masks."""
    clouds, masks, days_idx = [], [], []
    for t, (tr, te) in enumerate(zip(organ_seq["train_points"], organ_seq["test_points"])):
        if tr is None:
            continue
        if te is not None:
            clouds.append(torch.cat([tr, te], dim=0))
            masks.append(np.concatenate([np.ones(len(tr), np.int32), np.zeros(len(te), np.int32)]))
        else:
            clouds.append(tr)
            masks.append(np.ones(len(tr), np.int32))
        days_idx.append(t)
    return clouds, masks, days_idx


def model_has_organ(manager, organ_id):
    info = manager.organ_models.get(organ_id)
    if info is None or info.get("trained_model") is None:
        return False
    model = info["trained_model"]
    remap = getattr(model, "organ_id_remap", None)
    if remap is not None and organ_id not in remap:
        return False
    emb = getattr(model, "organ_embedding", None)
    if emb is not None and remap is None and organ_id >= emb.weight.shape[0]:
        return False
    return True


class OrganRollout:
    """Integrates clouds of one organ between its observed frames and to unseen days,
    with the run's integrator settings (mirrors _make_organ_advance / the fractional
    rule of save_interpolation_from_organ_models)."""

    def __init__(self, manager, organ_id, clouds):
        info = manager.organ_models[organ_id]
        self.cfg = cfg = info["config"]
        self.model = info["trained_model"]
        self.model_type = cfg["model_type"]
        self.days = manager.organ_train_days[organ_id]
        if self.model_type == "continuous" or manager.global_day_range is None:
            first_day, last_day = manager.organ_full_day_range[organ_id]
        else:
            first_day, last_day = manager.global_day_range
        self.first_day, self.day_range = first_day, last_day - first_day
        self.use_day_based = manager.use_day_based_tnorm
        self.get_t_norm, self.get_dt = make_t_norm_helpers(
            self.days, first_day, self.day_range, len(clouds), self.use_day_based)
        cond = {}
        if self.model_type == "shared_continuous":
            cond["organ_id"] = organ_id
        elif self.model_type == "encoder_decoder_continuous":
            with torch.no_grad():
                cond["latent_code"] = self.model.encode(clouds[0])
        self.cond = cond
        self.ode_kw = dict(
            use_odeint=cfg.get("use_odeint", False), ode_rtol=cfg.get("ode_rtol", 1e-4),
            ode_atol=cfg.get("ode_atol", 1e-5), ode_method=cfg.get("ode_method", "dopri5"),
            use_adjoint=False, n_substeps=cfg.get("train_integration_steps", 1),
            ode_n_steps=cfg.get("ode_n_steps", None), **cond,
        )
        self.span_kw = {k: v for k, v in self.ode_kw.items() if k not in ("n_substeps", "ode_n_steps")}

    def velocity(self, x, t):
        if "latent_code" in self.cond:
            return self.model(x, t, self.cond["latent_code"])
        if "organ_id" in self.cond:
            return self.model(x, t, self.cond["organ_id"])
        return self.model(x, t)

    @torch.no_grad()
    def between_frames(self, x, i_from, i_to):
        if i_to == i_from:
            return x.clone()
        moved, _ = _integrate_segment(
            self.model, x, i_from, i_to, self.get_t_norm, self.get_dt, **self.ode_kw)
        return moved

    @torch.no_grad()
    def to_unseen_day(self, x, i_prev, day):
        """From observed local frame i_prev to an unseen day (GrowFields' fractional rule)."""
        n = len(self.days)
        if self.use_day_based:
            t_prev = (self.days[i_prev] - self.first_day) / self.day_range
            t_target = (day - self.first_day) / self.day_range
        else:  # frame-index t_norm: one full step per missing frame is not defined; use GF's factor
            t_prev = float(i_prev) / (n - 1) if n > 1 else 0.0
            nxt = next((d for d in self.days if d > day), None)
            factor = 1.0 if nxt is None else (day - self.days[i_prev]) / (nxt - self.days[i_prev])
            t_target = t_prev + factor * (1.0 / (n - 1) if n > 1 else 0.0)
        use_odeint = self.ode_kw["use_odeint"]
        if use_odeint and self.ode_kw["ode_method"] not in _FIXED_STEP_METHODS:
            return _integrate_time_span(self.model, x, t_prev, t_target, **self.span_kw)
        if use_odeint:
            dt_sub = self.get_dt(i_prev) / max(self.ode_kw["ode_n_steps"] or 1, 1)
            n_floor = max(int((t_target - t_prev) / dt_sub), 1)
            return _integrate_time_span(
                self.model, x, t_prev, t_target, ode_n_steps=n_floor, **self.span_kw)
        dt_sub = self.get_dt(i_prev) / max(self.ode_kw["n_substeps"], 1)
        span = t_target - t_prev
        n_floor = int(span / dt_sub) if dt_sub > 0 else 0
        t_end = t_prev + n_floor * dt_sub
        x = x.clone()
        if n_floor > 0:
            x = _integrate_time_span(
                self.model, x, t_prev, t_end, num_substeps=n_floor, **self.span_kw)
        dt_rem = t_target - t_end
        if dt_rem > 1e-9:
            x = x + self.velocity(x, t_end) * dt_rem
        return x

    def roll(self, x, i_from, day, obs_day_to_idx):
        """Observed local frame i_from -> `day` (observed or unseen)."""
        if day in obs_day_to_idx:
            return self.between_frames(x, i_from, obs_day_to_idx[day])
        i_prev = max(i for i, d in enumerate(self.days) if d < day)
        return self.to_unseen_day(self.between_frames(x, i_from, i_prev), i_prev, day)


def pick_source(source, obs_days, day, gf_required_days=()):
    """Source observed day for target `day`, or None (GT fill).

    `gf_required_days`: for `first` on held-out days GrowFields also requires the organ at
    the plant-level training day before (and, interpolating, after) the target."""
    before = [d for d in obs_days if d < day]
    if not before:
        return None
    if source == "nearest":
        return before[-1]
    if source == "first":
        if any(d not in obs_days for d in gf_required_days):
            return None
        return obs_days[0]
    raise ValueError(source)


def pick_tip_day(source, obs_days, tips_by_day, s_day, target_day):
    """Observed, tip-annotated day the written tip is advected from (None: no tip)."""
    annotated = [d for d in obs_days if tips_by_day.get(d) is not None]
    if source == "first":
        return annotated[0] if annotated and annotated[0] < target_day else None
    at_or_before = [d for d in annotated if d <= s_day]
    if at_or_before:
        return at_or_before[-1]
    between = [d for d in annotated if s_day < d < target_day]
    return between[0] if between else None


def is_train_from_split(split_tp, n_points):
    m = np.full(n_points, -1, dtype=np.int32)
    tr, te = split_tp.get("train_indices", []), split_tp.get("test_indices", [])
    if tr:
        m[np.array(tr)] = 1
    if te:
        m[np.array(te)] = 0
    return m


def export_sequence(manager, seq_name, out_seq_dir, source, variant, original_split, label_field):
    from plant_shape_analysis import LeafSequencesDataset

    dataset = manager.dataset
    device = manager.device
    save_normals = manager.save_normals
    leaf_dataset = dataset.leaf_dataset
    plant_dataset = leaf_dataset.plant_dataset
    full_seq = plant_dataset.get_timeseries_by_sequence_name(seq_name)
    day_to_tp = {tp["day"]: tp for tp in full_seq["timepoints"]}
    all_days = sorted(day_to_tp)
    with open(original_split) as f:
        split_tps = json.load(f)["timepoints"]
    day_to_split = {tp["day"]: split_tps[i] for i, tp in enumerate(full_seq["timepoints"])
                    if i < len(split_tps)}
    # canonical-space leaf data (points, normals, leaf_tip) for every organ and day
    leaf_by_day = {}  # organ_id -> {day: ltp}
    for leaf_seq in leaf_dataset.get_timeseries_by_plant_sequence(seq_name):
        leaf_by_day[leaf_seq["leaf_id"]] = {ltp["day"]: ltp for ltp in leaf_seq["timepoints"]}
    inv_by_day = {}  # organ_id -> {day: inverse composed matrix}
    for organ_id, stages in manager.organ_transformations_full.items():
        composed = LeafSequencesDataset.compose_transformations(stages)
        inv_by_day[organ_id] = {
            d: LeafSequencesDataset.invert_transformation(c["composed_matrix"])
            for c, d in zip(composed, manager.organ_transformation_days.get(organ_id, []))
        }

    def gt_labels(tp):
        return np.asarray(tp.get(label_field, tp["labels"]))

    def organ_gt(organ_id, day):
        """Per-organ GT in the organ's model space (leaf canonical / stem plant space)."""
        tp = day_to_tp[day]
        mask = gt_labels(tp) == organ_id
        if not mask.any():
            return None
        is_tr = is_train_from_split(day_to_split[day], len(tp["points"]))[mask] if day in day_to_split else None
        if organ_id == 0:
            return dict(points=tp["points"][mask],
                        normals=tp["normals"][mask] if tp.get("normals") is not None else None,
                        tip=None, is_train=is_tr)
        ltp = leaf_by_day.get(organ_id, {}).get(day)
        if ltp is None:
            return None
        return dict(points=ltp["points"], normals=ltp.get("normals"), tip=ltp.get("leaf_tip"), is_train=is_tr)

    # observed (training) days, plant level, and the target days of this variant
    train_days = sorted({d for days in manager.organ_train_days.values() for d in days})
    if variant == "headline":
        target_days = all_days[1:]
        mode = None
    elif variant == "interpolation":
        target_days = [d for d in all_days if d not in train_days and train_days[0] < d < train_days[-1]]
        mode = "interpolation"
    else:
        target_days = [d for d in all_days if d not in train_days and d > train_days[-1]]
        mode = "extrapolation"
    print(f"  {seq_name}: days {all_days}, training days {train_days}, targets {target_days}, source={source}")

    # ---- per-organ predictions, model space: preds[organ][day] = (pred, tip, is_train)
    preds = {}
    for organ_id in sorted(manager.organ_sequences):
        organ_seq = manager.organ_sequences[organ_id]
        clouds, masks, _ = organ_clouds(organ_seq)
        obs_days = list(manager.organ_train_days[organ_id])
        assert len(obs_days) == len(clouds)
        if len(clouds) < 2 or not model_has_organ(manager, organ_id):
            continue
        obs_idx = {d: i for i, d in enumerate(obs_days)}
        tips_by_day = {d: leaf_by_day.get(organ_id, {}).get(d, {}).get("leaf_tip") for d in obs_days} \
            if organ_id > 0 else {}
        min_xyz, scale = manager.organ_normalizations[organ_id]
        min_b, scale_b = min_xyz.view(1, 1, 3), torch.tensor([scale.item()], device=device).view(1, 1, 1)
        roll = OrganRollout(manager, organ_id, clouds)
        preds[organ_id] = {}
        for day in target_days:
            if day not in day_to_tp or not (gt_labels(day_to_tp[day]) == organ_id).any():
                continue
            required = ()
            if variant != "headline":
                required = (max(d for d in train_days if d < day),)
                if variant == "interpolation":
                    required += (min(d for d in train_days if d > day),)
            s_day = pick_source(source, obs_days, day, gf_required_days=required)
            if s_day is None:
                if source == "first" and variant == "headline" and day == obs_days[0]:
                    # GrowFields quirk: an organ first seen mid-sequence gets a zero-step
                    # "rollout" (normalise/de-normalise round trip of its own cloud) in the
                    # full-plant file of that day, no per-organ file. Kept for exact parity.
                    pred = denormalise(clouds[0].unsqueeze(0), min_b, scale_b)[0].cpu().numpy()
                    preds[organ_id][day] = (pred, None, masks[0].copy(), None)
                continue
            i_s = obs_idx[s_day]
            x = roll.roll(clouds[i_s], i_s, day, obs_idx)
            pred = denormalise(x.unsqueeze(0), min_b, scale_b)[0].cpu().numpy()
            tip = None
            a_day = pick_tip_day(source, obs_days, tips_by_day, s_day, day) if organ_id > 0 else None
            if a_day is not None and day > a_day:
                tip0 = torch.as_tensor(np.asarray(tips_by_day[a_day]), dtype=torch.float32, device=device)
                tip0 = ((tip0 - min_xyz) / scale * 2 - 1).view(1, 3)
                tip_n = roll.roll(tip0, obs_idx[a_day], day, obs_idx)
                tip = denormalise(tip_n.unsqueeze(0), min_b, scale_b)[0].cpu().numpy()[0]
            preds[organ_id][day] = (pred, tip, masks[i_s].copy(), s_day)

    # ---- per-organ files
    for organ_id, by_day in preds.items():
        organ_type = manager.organ_sequences[organ_id]["type"]
        organ_dir = os.path.join(out_seq_dir, "organs", organ_type)
        os.makedirs(organ_dir, exist_ok=True)
        manager.save_organ_metrics(organ_id, organ_dir)
        sub = organ_dir if mode is None else os.path.join(organ_dir, mode)
        pred_dir, gt_dir = os.path.join(sub, "predicted"), os.path.join(sub, "ground_truth")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        oname = organ_type.replace("_", "")
        for day, (pred, tip, is_tr, s_day) in sorted(by_day.items()):
            if s_day is None:  # zero-step copy of the organ's first frame: full plant only
                continue
            gt = organ_gt(organ_id, day)
            fn = f"{seq_name}_D{day:02d}_{oname}.ply"
            normals = None
            if save_normals:
                ref = {} if gt is None or gt["normals"] is None else dict(ref_points=gt["points"], ref_normals=gt["normals"])
                normals = estimate_normals(pred, **ref)
            save_ply_with_organ_labels(
                pred, np.full(len(pred), organ_id, np.int32),
                None if tip is None else tip.reshape(1, 3),
                None if tip is None else np.array([organ_id], np.int32),
                os.path.join(pred_dir, f"pred_{fn}"), is_train=is_tr, normals=normals)
            if gt is None:
                continue
            save_ply_with_organ_labels(
                gt["points"], np.full(len(gt["points"]), organ_id, np.int32),
                None if gt["tip"] is None else np.asarray(gt["tip"]).reshape(1, 3),
                None if gt["tip"] is None else np.array([organ_id], np.int32),
                os.path.join(gt_dir, f"gt_{fn}"), is_train=gt["is_train"], normals=gt["normals"])

    # ---- full plant: re-posed predictions + GT fill, and plant-space GT
    fp = os.path.join(out_seq_dir, "full_plant") if mode is None else os.path.join(out_seq_dir, "full_plant", mode)
    pred_dir, gt_dir = os.path.join(fp, "predicted"), os.path.join(fp, "ground_truth")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    n_written = 0
    for day in target_days:
        tp = day_to_tp[day]
        labels = gt_labels(tp)
        full_is_train = is_train_from_split(day_to_split[day], len(tp["points"])) if day in day_to_split else None
        P, L, IT, tips, tip_labels, ref_normals = [], [], [], [], [], []
        for organ_id in sorted(int(o) for o in np.unique(labels)):
            hit = preds.get(organ_id, {}).get(day)
            if hit is not None:
                pred, tip, is_tr, _ = hit
                inv = inv_by_day.get(organ_id, {}).get(day) if organ_id > 0 else None
                if inv is not None:
                    pred = LeafSequencesDataset.apply_transformation(pred, inv)
                    if tip is not None:
                        tip = LeafSequencesDataset.apply_transformation(tip.reshape(1, 3), inv)[0]
                P.append(pred); L.append(np.full(len(pred), organ_id, np.int32)); IT.append(is_tr)
                if tip is not None:
                    tips.append(tip); tip_labels.append(organ_id)
            else:  # GrowFields' GT fill
                mask = labels == organ_id
                start = sum(len(p) for p in P)
                P.append(tp["points"][mask]); L.append(np.full(int(mask.sum()), organ_id, np.int32))
                if full_is_train is not None:
                    IT.append(full_is_train[mask])
                if tp.get("normals") is not None:
                    ref_normals.append((start, tp["normals"][mask]))
        if P:
            points, lab = np.concatenate(P), np.concatenate(L)
            is_tr = np.concatenate(IT) if len(IT) == len(P) else None
            normals = None
            if save_normals:
                ref = {} if tp.get("normals") is None else dict(
                    ref_points=tp["points"], ref_normals=tp["normals"], ref_labels=labels)
                normals = estimate_normals(points, lab, **ref)
                for start, nrm in ref_normals:
                    normals[start:start + len(nrm)] = nrm
            save_ply_with_organ_labels(
                points, lab, np.array(tips) if tips else None,
                np.array(tip_labels, np.int32) if tips else None,
                os.path.join(pred_dir, f"pred_{seq_name}_D{day:02d}.ply"), is_train=is_tr, normals=normals)
            n_written += 1
        tip_idx = np.asarray(tp.get("leaf_tip_idxs", np.array([])), dtype=int)
        save_ply_with_organ_labels(
            tp["points"], labels.astype(np.int32),
            tp["points"][tip_idx] if tip_idx.size else None,
            labels[tip_idx].astype(np.int32) if tip_idx.size else None,
            os.path.join(gt_dir, f"gt_{seq_name}_D{day:02d}.ply"), is_train=full_is_train, normals=tp.get("normals"))
    n_rows = sum(sum(s is not None for (_, _, _, s) in v.values()) for v in preds.values())
    print(f"  wrote {n_written} full-plant frames, {n_rows} organ frames -> {out_seq_dir}")
    return {"n_full_plant_frames": n_written, "n_organ_frames": n_rows,
            "source_days": {str(o): {str(d): s for d, (_, _, _, s) in v.items()} for o, v in preds.items()}}


def finished_sequences(exp_dir, wanted=None):
    seq_root = os.path.join(exp_dir, "sequences")
    out = []
    for name in sorted(os.listdir(seq_root)) if os.path.isdir(seq_root) else []:
        d = os.path.join(seq_root, name)
        if wanted and name not in wanted:
            continue
        runs = sorted(r for r in os.listdir(d) if r.startswith("run_")) if os.path.isdir(d) else []
        for run_dir in ([os.path.join(d, r) for r in runs] or [d]):
            has_model = os.path.isdir(os.path.join(run_dir, "models")) or any(
                os.path.isdir(os.path.join(run_dir, "organs", o, "models"))
                for o in (os.listdir(os.path.join(run_dir, "organs")) if os.path.isdir(os.path.join(run_dir, "organs")) else []))
            if has_model and os.path.isfile(os.path.join(run_dir, "results.json")):
                out.append((name, run_dir))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment_dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--source", default="nearest", choices=["nearest", "first"])
    ap.add_argument("--variant", default="auto", choices=["auto", "headline", "interpolation", "extrapolation"],
                    help="auto reads train_timepoint_stride / train_holdout_last_n from experiment_config.json")
    ap.add_argument("--out_dir", default=None, help="default: <experiment_dir>_<source>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    exp_dir = os.path.normpath(args.experiment_dir)
    out_dir = args.out_dir or f"{exp_dir}_{args.source}"
    with open(os.path.join(exp_dir, "experiment_config.json")) as f:
        exp_cfg = json.load(f)
    variant = detect_variant(exp_cfg.get("cli_args", {})) if args.variant == "auto" else args.variant
    label_field = exp_cfg["config"]["dataset"].get("label_field", "labels")
    os.makedirs(out_dir, exist_ok=True)
    out_cfg = dict(exp_cfg, rollout_source=args.source, export_variant=variant, source_experiment_dir=exp_dir)
    write_json(out_cfg, os.path.join(out_dir, "experiment_config.json"), clean=True)

    seqs = finished_sequences(exp_dir, args.sequences)
    if not seqs:
        print(f"no finished sequences (models/ + results.json) under {exp_dir}")
        return
    print(f"{len(seqs)} finished sequence run(s) in {exp_dir}; variant={variant}; out={out_dir}")
    for seq_name, run_dir in seqs:
        out_seq_dir = os.path.join(out_dir, os.path.relpath(run_dir, exp_dir))
        marker = os.path.join(out_seq_dir, "export_nearest_source.json")
        if os.path.isfile(marker) and not args.overwrite:
            print(f"  {seq_name}: already exported ({marker}), skipping")
            continue
        manager, original_split, stride, holdout = load_manager(
            exp_cfg, seq_name, run_dir, exp_dir, args.device,
            variant_override=None if args.variant == "auto" else variant, derive_dir=out_dir)
        os.makedirs(out_seq_dir, exist_ok=True)
        for fn in ("config.yaml", "train_test_split.json"):
            if os.path.isfile(os.path.join(run_dir, fn)):
                shutil.copy(os.path.join(run_dir, fn), os.path.join(out_seq_dir, fn))
        with open(os.path.join(run_dir, "results.json")) as f:
            results = json.load(f)
        results.update(rollout_source=args.source, export_variant=variant)
        write_json(results, os.path.join(out_seq_dir, "results.json"), clean=True)
        with torch.no_grad():
            info = export_sequence(manager, seq_name, out_seq_dir, args.source, variant, original_split, label_field)
        write_json(dict(info, source=args.source, variant=variant, stride=stride, holdout=holdout,
                        models_from=run_dir), marker, clean=True)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
