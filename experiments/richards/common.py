"""Shared plumbing for RichardsNet: thread limits, MonoGrow imports, organ size series, leaf ranks, MAP curve fit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np                                   # noqa: E402
from scipy.optimize import least_squares             # noqa: E402

HERE = Path(__file__).resolve().parent
MONOGROW = HERE.parent / "monogrow"
sys.path.insert(0, str(MONOGROW))                    # canon.py, data.py, ply.py, config.py, metrics.py of MonoGrow

from curves import CURVE, init_theta, length, width  # noqa: E402

GROUPS = ("maize", "tomato", "sorghum", "tobacco", "sugar_beet")


def group_of(track) -> str:
    return f"{track.species}/{'stem' if track.is_stem else 'leaf'}"


def series(track, local_frames) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ages since the first frame in `local_frames`, lengths, widths) in days / mm."""
    ages = track.days[local_frames] - track.days[local_frames[0]]
    return ages, np.array([length(track.points[t]) for t in local_frames]), np.array([width(track.points[t]) for t in local_frames])


def leaf_ranks(tracks, first_frames) -> dict[int, int]:
    """rank = order of first appearance among the plant's leaves (ties broken by organ id), starting at 1."""
    leaves = sorted((first_frames[i], t.organ_id) for i, t in enumerate(tracks) if not t.is_stem)
    return {oid: r + 1 for r, (_, oid) in enumerate(leaves)}


def fit_theta(family: str, ages: np.ndarray, L: np.ndarray, mu: np.ndarray, prec: np.ndarray, lam: float = 1.0,
              sigma_L: float = 5.0, log_space: bool = False, robust: bool = False) -> np.ndarray:
    """MAP estimate: min_theta sum ((L(a;theta) - L)/sigma_L)^2 + lam (theta-mu)^T prec (theta-mu).
    Levenberg-Marquardt-type trust-region least squares (scipy 'trf') on the stacked residual vector; the best of two
    starts (prior mean, data-driven init) is kept. One observation (or lam = inf) returns the prior mean."""
    if len(ages) < 2 or not np.isfinite(lam):
        return mu.copy()
    f = CURVE[family]
    R = np.linalg.cholesky(prec) * np.sqrt(lam)

    def resid(th):
        pred = f(th, ages)
        d = (np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(L, 1e-3))) / sigma_L if log_space else (pred - L) / sigma_L
        return np.concatenate([d, R.T @ (th - mu)])

    best = None
    for th0 in (mu, init_theta(family, ages, L)):
        try:
            r = least_squares(resid, th0, method="trf", loss="soft_l1" if robust else "linear", max_nfev=400)
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(r.x)) and (best is None or r.cost < best.cost):
            best = r
    return best.x if best is not None else mu.copy()
