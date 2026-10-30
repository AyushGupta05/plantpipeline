"""Separate plant from soil/pot.

Labelled scans: soil is label 0.
Unlabelled scans: every unlabelled day sits between labelled days of the same
plant, in the same fixed frame, and the soil surface barely moves. So we copy
labels from the nearest labelled neighbour(s) by 1-nearest-neighbour in 3D.
A point is soil only if every available neighbour says so (conservative).
"""

from __future__ import annotations

import numpy as np
from pykdtree.kdtree import KDTree  # ~100x faster than scipy.cKDTree on these dense scans

SOIL = 0


def soil_mask_from_labels(labels: np.ndarray) -> np.ndarray:
    return labels == SOIL


def transfer_labels(points: np.ndarray, neighbours: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:

    votes = []
    for npts, nlab in neighbours:
        tree = KDTree(npts)
        _, idx = tree.query(points, k=1)
        votes.append(nlab[idx])
    labels = votes[0].astype(np.int16)
    soil = np.logical_and.reduce([v == SOIL for v in votes])
    labels[~soil & (labels == SOIL)] = 1  # disagreement: treat as plant (stem) rather than soil
    return labels, soil


def plant_origin(points: np.ndarray, soil: np.ndarray) -> np.ndarray:

    plant = points[~soil]
    xy = plant[:, :2].mean(axis=0) if len(plant) else points[:, :2].mean(axis=0)
    z = np.median(points[soil, 2]) if soil.any() else points[:, 2].min()
    return np.array([xy[0], xy[1], z], dtype=np.float32)
