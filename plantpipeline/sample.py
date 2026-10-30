"""Downsample a cloud to a fixed number of points with farthest point sampling."""

from __future__ import annotations

import numpy as np
import fpsample


def fps(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Return indices of `n` farthest-point samples. If the cloud has <= n points, return all (with replacement to reach n)."""
    m = len(points)
    if m <= n:
        rng = np.random.default_rng(seed)
        extra = rng.integers(0, m, n - m) if m else np.zeros(n, dtype=int)
        return np.concatenate([np.arange(m), extra])
    h = int(np.clip(np.log2(m) - 3, 1, 9))  # kd-tree bucket depth; needs 2**h < m
    idx = fpsample.bucket_fps_kdline_sampling(points.astype(np.float32), n, h=h, start_idx=0)
    return np.asarray(idx, dtype=np.int64)
