"""Determinate-growth length curves, width allometry and closed-form organ decoding (DESIGN_MEMO section 5).

Every family maps an unconstrained 4-vector theta and organ age a (days since the organ's FIRST OBSERVED frame,
a >= 0) to a length L(a) in mm, so a Gaussian prior on theta is a proper hierarchical prior:
  richards  theta = (log A, log k, a_m, log nu)        L = A (1 + nu exp(-k (a - a_m)))^(-1/nu)
  beta      theta = (log L_m, log a_e, logit-ish(a_m / a_e), log a_0)   Yin et al. 2003 beta-sigmoid, evaluated at
            true age a + a_0 (a_0 = unknown age at first observation), constant L_m for a + a_0 >= a_e.
Width by allometry W = alpha L^beta; alpha cancels in the decoding, which only needs the length ratio.
Lengths are 99th-percentile canonical axial (Z) extent, widths 99th-percentile radial extent sqrt(x^2 + y^2).
"""

from __future__ import annotations

import numpy as np

FAMILIES = ("richards", "beta")
PARAM_NAMES = {"richards": ("logA", "logk", "a_m", "lognu"), "beta": ("logLm", "logae", "am_frac", "loga0")}


def length(points: np.ndarray) -> float:
    return float(np.percentile(points[:, 2], 99))


def width(points: np.ndarray) -> float:
    return float(np.percentile(np.hypot(points[:, 0], points[:, 1]), 99))


def richards(theta, a):
    A, k, am, nu = np.exp(theta[0]), np.exp(theta[1]), theta[2], np.exp(theta[3])
    e = np.clip(-k * (np.asarray(a, float) - am), -50.0, 50.0)
    return A * (1.0 + nu * np.exp(e)) ** (-1.0 / nu)


def beta_sigmoid(theta, a):
    Lm, ae, a0 = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[3])
    am = ae * (0.05 + 0.9 / (1.0 + np.exp(-np.clip(theta[2], -8, 8))))   # a_m strictly inside (0, a_e)
    x = np.clip(np.asarray(a, float) + a0, 0.0, ae)
    return Lm * (1.0 + (ae - x) / (ae - am)) * (x / ae) ** (ae / (ae - am))


CURVE = {"richards": richards, "beta": beta_sigmoid}


def init_theta(family: str, ages: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Data-driven starting point: asymptote 1.3 x max length, midpoint where L crosses half of it."""
    Lmax = max(float(np.max(L)), 1.0)
    span = max(float(ages[-1] - ages[0]), 1.0)
    half = np.interp(0.5 * Lmax, np.maximum.accumulate(L), ages) if L[0] < 0.5 * Lmax else ages[0] - 0.25 * span
    if family == "richards":
        return np.array([np.log(1.3 * Lmax), np.log(2.0 / span), half, 0.0])
    return np.array([np.log(1.3 * Lmax), np.log(2.0 * span + 2.0), 0.0, np.log(0.5 * span + 1.0)])


def decode(x0: np.ndarray, L_ratio: float, beta: float) -> np.ndarray:
    """x(a) = diag(W/W0, W/W0, L/L0) x0 with W/W0 = (L/L0)^beta (base at the origin stays fixed)."""
    s = np.array([L_ratio ** beta, L_ratio ** beta, L_ratio], dtype=x0.dtype)
    return x0 * s
