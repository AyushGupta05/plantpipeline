"""Organ tracks for MonoGrow.

A track is one organ (stem or leaf) of one plant sequence: its per-frame point sets in the canonical frame
(canon.py, identical to GrowFields/PSA for leaves), the plant->canonical rigid transform of every frame,
organ age (days since the organ's first observation), plant age (days since the plant's first scan) and species.

Two loaders share `build_tracks`:
  * `load_training_tracks` - data/unified (npz; no normals, so stage-3 normals are estimated with open3d);
    applies the leakage rule (TrackPlant3D train split only, Pheno4D M01/M02/M04/T01/T05 excluded).
  * `load_benchmark_sequence` - one TrackPlant3D v2 plant straight from the ETH PLYs (float64, normals, tips,
    GrowFields organ ids: 0 stem, k leaf) plus GrowFields' point split -> per-frame is_train masks.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from canon import apply_transform, canonicalise_leaf, canonicalise_stem, estimate_normals
from config import SPECIES_ID, DataConfig
from ply import read_ply


@dataclass
class OrganTrack:
    seq: str
    dataset: str
    species: str
    organ_id: int                       # GrowFields convention: 0 = stem, k >= 1 = leaf k
    frame_idx: list[int]                # global frame indices (into the plant's frame list) where the organ is present
    days: np.ndarray                    # (T,) absolute plant days of those frames
    points: list[np.ndarray]            # (T,) canonical points, mm, float32
    M: np.ndarray                       # (T,4,4) plant->canonical, row-vector convention
    tips: list                          # (T,) canonical tip (3,) or None
    plant_days: np.ndarray              # all days of the plant sequence
    is_train: list = field(default_factory=list)   # (T,) bool masks over `points` (benchmark sequences only)

    @property
    def is_stem(self) -> bool:
        return self.organ_id == 0

    @property
    def ages(self) -> np.ndarray:       # organ age in days
        return self.days - self.days[0]

    @property
    def plant_ages(self) -> np.ndarray:  # plant age in days
        return self.days - self.plant_days[0]

    def to_plant(self, canonical: np.ndarray, t: int) -> np.ndarray:
        """Re-pose canonical points with the GT transform of local frame t."""
        return apply_transform(canonical, np.linalg.inv(self.M[t]))


# --------------------------------------------------------------------------------------------- raw frames

def load_tp3d_frames(tp3d_dir: Path, seq: str) -> list[dict]:
    """TrackPlant3D v2 PLYs of one plant: [{day, points f64 (N,3), labels (0 stem), normals, tip_idx}] by day."""
    files = sorted(tp3d_dir.glob(f"*/*_{seq}_D*.ply"), key=lambda p: int(re.search(r"_D(\d+)\.ply$", p.name).group(1)))
    assert files, f"no PLYs for {seq} under {tp3d_dir}"
    frames = []
    for p in files:
        v = read_ply(p)
        frames.append({"day": int(re.search(r"_D(\d+)\.ply$", p.name).group(1)), "points": v["points"],
                       "labels": v["organ_label"].astype(int), "normals": v.get("normals"),
                       "tip_idx": np.where(v["is_leaf_tip"].astype(bool))[0]})
    return frames


def load_unified_frames(unified: Path, entry: dict) -> list[dict]:
    """data/unified npz frames of one sequence, labels remapped to GrowFields ids (stem 1 -> 0, leaf k -> k-1)."""
    frames = []
    for scan in entry["frames"]:
        d = np.load(unified / entry["dataset"] / entry["sequence_id"] / f"{scan}.npz")
        lab = d["labels"].astype(int)
        gf = np.where(lab >= 1, lab - 1, -1)  # unknown 0 -> -1 (ignored)
        frames.append({"day": float(d["day"]), "points": d["points"].astype(np.float64), "labels": gf,
                       "normals": None, "tip_idx": np.array([], dtype=int)})
    return frames


# --------------------------------------------------------------------------------------------- tracks

def build_tracks(frames: list[dict], seq: str, dataset: str, species: str, min_frames: int = 2, min_points: int = 3,
                 train_masks: list[np.ndarray] | None = None) -> list[OrganTrack]:
    plant_days = np.array([f["day"] for f in frames], dtype=np.float64)
    stems = {}
    for f in frames:
        s = f["points"][f["labels"] == 0]
        stems[f["day"]] = s if len(s) > 0 else f["points"]   # PSA raises here; we fall back to the whole plant
    organ_ids = sorted(set(int(o) for f in frames for o in np.unique(f["labels"]) if o >= 0))
    tracks = []
    for oid in organ_ids:
        organ_frames, idxs, masks = [], [], []
        for t, f in enumerate(frames):
            m = f["labels"] == oid
            if m.sum() < min_points:
                continue
            tip = None
            if oid > 0 and len(f["tip_idx"]):
                hit = f["tip_idx"][m[f["tip_idx"]]]
                if len(hit):
                    tip = f["points"][hit[0]].copy()
            normals = f["normals"][m] if f["normals"] is not None else None
            if oid > 0 and normals is None:
                normals = estimate_normals(f["points"][m])
            organ_frames.append({"day": f["day"], "points": f["points"][m], "normals": normals, "tip": tip})
            idxs.append(t); masks.append(m)
        if len(organ_frames) < min_frames:
            continue
        can, M = canonicalise_stem(organ_frames) if oid == 0 else canonicalise_leaf(organ_frames, stems)
        tracks.append(OrganTrack(
            seq=seq, dataset=dataset, species=species, organ_id=oid, frame_idx=idxs,
            days=np.array([f["day"] for f in organ_frames], dtype=np.float64),
            points=[c["points"].astype(np.float32) for c in can], M=M, tips=[c.get("tip") for c in can],
            plant_days=plant_days,
            is_train=[train_masks[t][m] for t, m in zip(idxs, masks)] if train_masks is not None else [],
        ))
    return tracks


def _cache_key(cfg: DataConfig) -> str:
    s = json.dumps({k: str(v) for k, v in vars(cfg).items() if k != "cache"}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def load_training_tracks(cfg: DataConfig, verbose: bool = True) -> list[OrganTrack]:
    """All organ tracks eligible for prior training under the leakage rule, cached as a pickle."""
    cfg.cache.mkdir(parents=True, exist_ok=True)
    path = cfg.cache / f"tracks_{_cache_key(cfg)}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    with open(cfg.unified / "sequences.json") as f:
        seqs = json.load(f)
    tracks = []
    for e in seqs:
        if e["dataset"] not in cfg.datasets or e["sequence_id"] in cfg.exclude_sequences:
            continue
        allowed = cfg.tp3d_splits if e["dataset"] == "trackplant3d" else cfg.other_splits
        if e["split"] not in allowed:
            continue
        frames = load_unified_frames(cfg.unified, e)
        t = build_tracks(frames, e["sequence_id"], e["dataset"], e["species"], cfg.min_frames, cfg.min_points)
        tracks += t
        if verbose:
            print(f"  {e['dataset']:12s} {e['sequence_id']:28s} {e['species']:10s} {len(frames):3d} frames  {len(t):3d} organ tracks")
    pickle.dump(tracks, open(path, "wb"))
    if verbose:
        print(f"{len(tracks)} organ tracks cached at {path}")
    return tracks


# --------------------------------------------------------------------------------------------- benchmark

def tp3d_species(seq: str) -> str:
    return {"tomato1": "tomato", "tomato2": "tomato"}.get(seq.split("_")[0], seq.split("_")[0])


def load_point_split(cfg: DataConfig, seq: str) -> dict:
    with open(cfg.point_splits / f"{seq}_split.json") as f:
        return json.load(f)


def load_benchmark_sequence(cfg: DataConfig, seq: str) -> tuple[list[dict], list[OrganTrack], dict]:
    """One GrowFields val/test plant: raw frames (with is_train), organ tracks (min 2 frames) and the split file."""
    frames = load_tp3d_frames(cfg.tp3d_dir, seq)
    split = load_point_split(cfg, seq)
    assert split["n_timepoints"] == len(frames)
    masks = []
    for f, tp in zip(frames, split["timepoints"]):
        assert tp["n_total_points"] == len(f["points"]), "point split does not match the loaded cloud"
        m = np.zeros(len(f["points"]), dtype=bool); m[np.array(tp["train_indices"])] = True
        f["is_train"] = m; masks.append(m)
    tracks = build_tracks(frames, seq, "trackplant3d", tp3d_species(seq), min_frames=2, min_points=cfg.min_points, train_masks=masks)
    return frames, tracks, split


def gf_split_plants(cfg: DataConfig, split: str) -> list[str]:
    with open(cfg.unified / "sequences.json") as f:
        seqs = json.load(f)
    return sorted(e["sequence_id"] for e in seqs if e["dataset"] == "trackplant3d" and e["split"] == split)


# --------------------------------------------------------------------------------------------- sampler

class TrackSampler:
    """Yields (organ, source frame s, target frame k > s, points, time grid) batches for the prior.

    Each item: dict(track=int, s, k, x0 (n,3) float32 mm, targets [(n,3)] for frames s+1..k, days (k-s+1,),
    age0 = organ age at s, plant_age0 = plant age at s, species = int). Points are subsampled with replacement."""

    def __init__(self, tracks: list[OrganTrack], p_consecutive: float, max_horizon_days: float, seed: int = 0,
                 observed: list[list[int]] | None = None, balance_species: bool = False):
        self.tracks = tracks
        self.p_consecutive, self.max_h = p_consecutive, max_horizon_days
        self.rng = np.random.default_rng(seed)
        counts = {s: sum(t.species == s for t in tracks) for s in set(t.species for t in tracks)}
        w = np.array([1.0 / counts[t.species] if balance_species else 1.0 for t in tracks])
        self.weights = w / w.sum()   # species-balanced sampling: every species gets equal expected share
        # local frame indices allowed as (s, k); default all frames of the track
        self.observed = observed or [list(range(len(t.days))) for t in tracks]

    def sample_pair(self, i: int) -> tuple[int, int]:
        obs = self.observed[i]; days = self.tracks[i].days
        if len(obs) == 2 or self.rng.random() < self.p_consecutive:
            j = self.rng.integers(0, len(obs) - 1)
            return obs[j], obs[j + 1]
        j = self.rng.integers(0, len(obs) - 1)
        later = [o for o in obs[j + 1:] if days[o] - days[obs[j]] <= self.max_h] or [obs[j + 1]]
        return obs[j], later[self.rng.integers(0, len(later))]

    def item(self, i: int, s: int, k: int, n: int, train_only: bool = False) -> dict:
        t = self.tracks[i]

        def pick(f):
            pts = t.points[f]
            if train_only and len(t.is_train):
                pts = pts[t.is_train[f]]
            return pts[self.rng.integers(0, len(pts), n)]
        return {"track": i, "s": s, "k": k, "x0": pick(s), "targets": [pick(f) for f in range(s + 1, k + 1)],
                "days": t.days[s:k + 1], "age0": float(t.ages[s]), "plant_age0": float(t.plant_ages[s]),
                "species": SPECIES_ID[t.species]}

    def batch(self, organs_per_step: int, points_per_step: int, train_only: bool = False) -> list[dict]:
        k = min(organs_per_step, len(self.tracks))
        n = max(points_per_step // k, 256)
        idx = self.rng.choice(len(self.tracks), size=k, replace=False, p=self.weights)
        return [self.item(int(i), *self.sample_pair(int(i)), n, train_only) for i in idx]


def to_tensor(x: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
