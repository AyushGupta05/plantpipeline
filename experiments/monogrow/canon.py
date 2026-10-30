"""Per-frame organ canonicalisation, a line-by-line numpy port of plant-shape-analysis v0.4.1
(`alignment/pca_alignment.py`, `LeafSequencesDataset._align_dataset`), which is what GrowFields uses.

Leaves: four stages applied in place and in sequence
  1. correct_pca_stem   PCA basis per frame, main axis signed base->tip using the day's stem (+ tip check, temporal vote)
  2. align_to_z         Rodrigues rotation about the centroid taking the main axis to +Z
  3. z_rotation_normals rotation about Z aligning the mean normal (over NN-matched points) with the previous frame
  4. align_base         translate the centroid of the lowest 1 % of points (along Z) to the origin
Every stage is p' = (p - c) @ R.T + c + t, composed into one 4x4 per frame for row vectors: p_can = [p, 1] @ M.T.
Stems: translation only (lowest-1 % centroid to the origin, same rule as stage 4), rotation identity.

`canonicalise_leaf` reproduces PSA to float precision given the same float64 inputs (verified in tests.py).
Without normals, stage 3 is the identity (PSA behaves the same when a frame has no normals).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


def pca_basis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt


def _stage_matrix(R: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
    T1 = np.eye(4); T1[:3, 3] = -c
    TR = np.eye(4); TR[:3, :3] = R
    T2 = np.eye(4); T2[:3, 3] = c + t
    return T2 @ TR @ T1


def apply_transform(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    return points @ M[:3, :3].T + M[:3, 3]


def _nn(a: np.ndarray, b: np.ndarray):
    """For each row of b, distance to and index of the nearest row of a."""
    if cKDTree is not None:
        return cKDTree(a).query(b, k=1)
    d = np.sqrt(((b[:, None, :] - a[None]) ** 2).sum(-1))
    idx = d.argmin(1)
    return d[np.arange(len(b)), idx], idx


def stage1_correct_pca_stem(frames: list[dict], stems: dict[float, np.ndarray]) -> list[dict]:
    out = []
    for f in frames:
        pts = f["points"]; center = pts.mean(axis=0); centered = pts - center
        axes = pca_basis(centered)
        axis = axes[0] / np.linalg.norm(axes[0])
        stem = stems[f["day"]]
        closest_stem = stem[np.argmin(np.linalg.norm(stem - center, axis=1))]
        base = pts[np.argmin(np.linalg.norm(pts - closest_stem, axis=1))]
        base_proj = np.dot(base - center, axis)
        growth = -axis if base_proj > 0 else axis
        basis = np.array([growth, axes[1] / np.linalg.norm(axes[1]), axes[2] / np.linalg.norm(axes[2])])
        if f.get("tip") is not None:
            tip_proj = np.dot(f["tip"] - center, growth)
            if not (np.dot(base - center, growth) < 0 and tip_proj > 0):
                basis[0] = -basis[0]
        out.append({"basis": basis, "center": center})
    no_tip = [i for i, f in enumerate(frames) if f.get("tip") is None]
    if 0 < len(no_tip) < len(frames):
        dirs = np.array([o["basis"][0] for o in out])
        with_tip = [i for i, f in enumerate(frames) if f.get("tip") is not None]
        ref = dirs[with_tip].mean(axis=0); ref /= np.linalg.norm(ref)
        for i in no_tip:
            if np.dot(out[i]["basis"][0], ref) < 0:
                out[i]["basis"][0] = -out[i]["basis"][0]
    return out


def stage2_align_to_z(frames: list[dict], s1: list[dict]) -> list[np.ndarray]:
    mats = []
    for f, t in zip(frames, s1):
        pts = f["points"]; center = pts.mean(axis=0); centered = pts - center
        a = t["basis"][0] / np.linalg.norm(t["basis"][0]); z = np.array([0.0, 0.0, 1.0])
        v = np.cross(a, z); c = np.dot(a, z)
        if abs(c - 1.0) < 1e-8:
            mats.append(_stage_matrix(np.eye(3), np.zeros(3), center)); continue
        if abs(c + 1.0) < 1e-8:
            R = np.diag([1.0, -1.0, -1.0])
        else:
            s = np.linalg.norm(v)
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + K + K @ K * ((1 - c) / s**2)
        f["points"] = centered @ R.T + center
        if f.get("normals") is not None:
            f["normals"] = f["normals"] @ R.T
        if f.get("tip") is not None:
            f["tip"] = (f["tip"] - center) @ R.T + center
        mats.append(_stage_matrix(R, np.zeros(3), center))
    return mats


def stage3_z_rotation_normals(frames: list[dict], percentile: int = 75) -> list[np.ndarray]:
    mats = [_stage_matrix(np.eye(3), np.zeros(3), frames[0]["points"].mean(axis=0))]
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        center = cur["points"].mean(axis=0)
        if prev.get("normals") is None or cur.get("normals") is None or len(prev["normals"]) == 0 or len(cur["normals"]) == 0:
            mats.append(_stage_matrix(np.eye(3), np.zeros(3), center)); continue
        dist, idx = _nn(prev["points"], cur["points"])
        ok = dist < np.percentile(dist, percentile)
        if not ok.any():
            mats.append(_stage_matrix(np.eye(3), np.zeros(3), center)); continue
        cn = cur["normals"][ok].mean(axis=0); cn /= np.linalg.norm(cn) + 1e-8
        pn = prev["normals"][idx[ok]].mean(axis=0); pn /= np.linalg.norm(pn) + 1e-8
        ang = np.arctan2(pn[1], pn[0]) - np.arctan2(cn[1], cn[0])
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
        cur["points"] = (cur["points"] - center) @ R.T + center
        cur["normals"] = cur["normals"] @ R.T
        if cur.get("tip") is not None:
            cur["tip"] = (cur["tip"] - center) @ R.T + center
        mats.append(_stage_matrix(R, np.zeros(3), center))
    return mats


def base_shift(points: np.ndarray, percentile: float = 1.0) -> np.ndarray:
    """-centroid of the points whose z is at or below the given percentile (PSA align_base_to_origin)."""
    z = points[:, 2]
    mask = z <= np.percentile(z, percentile)
    if mask.sum() < 1:
        return np.array([0.0, 0.0, -z.min()])
    return -points[mask].mean(axis=0)


def stage4_align_base(frames: list[dict]) -> list[np.ndarray]:
    mats = []
    for f in frames:
        shift = base_shift(f["points"])
        f["points"] = f["points"] + shift
        if f.get("tip") is not None:
            f["tip"] = f["tip"] + shift
        mats.append(_stage_matrix(np.eye(3), shift, np.zeros(3)))
    return mats


def canonicalise_leaf(frames: list[dict], stems: dict[float, np.ndarray]) -> tuple[list[dict], np.ndarray]:
    """frames: [{day, points (N,3) float64 plant space, normals (N,3)|None, tip (3,)|None}] sorted by day;
    stems: day -> stem points of that day. Returns (canonical frames, (T,4,4) plant->canonical matrices).
    A single-frame track is returned unchanged with identity matrices (PSA aligns only tracks with >= 2 frames)."""
    frames = [dict(f, points=np.asarray(f["points"], dtype=np.float64)) for f in frames]
    if len(frames) < 2:
        return frames, np.tile(np.eye(4), (len(frames), 1, 1))
    s1 = stage1_correct_pca_stem(frames, stems)
    m2 = stage2_align_to_z(frames, s1)
    m3 = stage3_z_rotation_normals(frames)
    m4 = stage4_align_base(frames)
    M = np.stack([c @ b @ a for a, b, c in zip(m2, m3, m4)])
    return frames, M


def canonicalise_stem(frames: list[dict]) -> tuple[list[dict], np.ndarray]:
    """Stem convention: translate the lowest-1 % centroid to the origin per frame, no rotation."""
    out, mats = [], []
    for f in frames:
        pts = np.asarray(f["points"], dtype=np.float64)
        shift = base_shift(pts)
        out.append(dict(f, points=pts + shift, tip=None))
        mats.append(_stage_matrix(np.eye(3), shift, np.zeros(3)))
    return out, np.stack(mats)


def estimate_normals(points: np.ndarray) -> np.ndarray | None:
    """Open3D normals oriented towards +Z (PSA `estimate_normals_for_point_cloud` without the tangent-plane pass,
    whose global sign flips between frames). Returns None when open3d is unavailable."""
    try:
        import open3d as o3d
    except ImportError:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    return np.asarray(pcd.normals)
