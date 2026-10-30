"""Evaluation metrics for predicted point clouds.

The `compute_*` wrappers are what the `scripts/compute_metrics_*.py` entry points
call: they take numpy or torch input, guard the degenerate cases, and return plain
floats or None instead of raising. They were duplicated byte-for-byte in both
scripts before living here.

`scene_flow_metrics` is third-party code; see NOTICE for its notice.
"""

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance

from growfields.pipeline.config import deterministic_enabled


# Adapted from DeformationPyramid (https://github.com/rabbityl/DeformationPyramid),
# MIT licensed, Copyright (c) 2023 Yang Li. These are the standard scene-flow
# measures: EPE, accuracy at a strict and a relaxed threshold, and an outlier rate.
def scene_flow_metrics(pred, labels, strict=0.025, relax=0.05):

    l2_norm = torch.sqrt(
        torch.sum((pred - labels) ** 2, 1)
    ).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 1)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: AccS
    error_lt_5 = torch.BoolTensor((l2_norm < strict))
    relative_err_lt_5 = torch.BoolTensor((relative_err < strict))
    AccS = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: AccR
    error_lt_10 = torch.BoolTensor((l2_norm < relax))
    relative_err_lt_10 = torch.BoolTensor((relative_err < relax))
    AccR = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    return EPE3D, AccS * 100, AccR * 100


def chamfer_loss(x, y):
    """Chamfer training loss, deterministic under `--deterministic`.

    The default path is pytorch3d's ``chamfer_distance``, whose KNN backward
    accumulates with atomics and so differs between identical calls. The
    deterministic path computes the same objective (squared distances, mean
    over points in each direction, summed, mean over the batch) from an
    explicit [B, N, M] distance tensor; its backward is a min-select and a
    broadcast sum, with no atomics. Memory is O(N*M), fine for training-sized
    samples, not for full evaluation clouds.
    """
    if not deterministic_enabled():
        return chamfer_distance(x, y)[0]
    d = (x[:, :, None, :] - y[:, None, :, :]).pow(2).sum(-1)
    return (d.min(2).values.mean(1) + d.min(1).values.mean(1)).mean()


def compute_chamfer_distance(pred_pts, gt_pts, pred_normals=None, gt_normals=None):
    """
    Compute Chamfer Distance between predicted and ground truth point clouds.

    Args:
        pred_pts: Predicted points
        gt_pts: Ground truth points
        pred_normals: Optional predicted normals
        gt_normals: Optional ground truth normals

    Returns:
        tuple: (cd, normal_consistency) where normal_consistency is None if normals not provided.
        Normal consistency is the mean |cos| between nearest-neighbour normals,
        averaged over both directions: in [0, 1], higher is better, sign-invariant.
    """
    if pred_pts is None or gt_pts is None:
        return None, None

    # Convert to tensors if needed
    if not isinstance(pred_pts, torch.Tensor):
        pred_pts = torch.from_numpy(pred_pts).float()
    if not isinstance(gt_pts, torch.Tensor):
        gt_pts = torch.from_numpy(gt_pts).float()

    # Ensure correct shape [1, N, 3]
    if pred_pts.dim() == 2:
        pred_pts = pred_pts.unsqueeze(0)
    if gt_pts.dim() == 2:
        gt_pts = gt_pts.unsqueeze(0)

    # Handle normals if provided
    normal_consistency = None
    if pred_normals is not None and gt_normals is not None:
        if not isinstance(pred_normals, torch.Tensor):
            pred_normals = torch.from_numpy(pred_normals).float()
        if not isinstance(gt_normals, torch.Tensor):
            gt_normals = torch.from_numpy(gt_normals).float()

        if pred_normals.dim() == 2:
            pred_normals = pred_normals.unsqueeze(0)
        if gt_normals.dim() == 2:
            gt_normals = gt_normals.unsqueeze(0)

        # pytorch3d returns the normal term as a loss, the sum over both directions
        # of mean(1 - |cos|), so it lies in [0, 2]; convert it to the consistency.
        cd, normal_loss = chamfer_distance(
            pred_pts, gt_pts, x_normals=pred_normals, y_normals=gt_normals
        )
        normal_consistency = 1.0 - float(normal_loss.item()) / 2
    else:
        # Compute Chamfer Distance without normals
        cd, _ = chamfer_distance(pred_pts, gt_pts)

    return float(cd.item()), normal_consistency


def compute_epe(pred_tips, gt_tips):
    """Compute End Point Error for leaf tips."""
    if pred_tips is None or gt_tips is None:
        return None
    if len(pred_tips) == 0 or len(gt_tips) == 0:
        return None

    # Convert to numpy if needed
    if isinstance(pred_tips, torch.Tensor):
        pred_tips = pred_tips.cpu().numpy()
    if isinstance(gt_tips, torch.Tensor):
        gt_tips = gt_tips.cpu().numpy()

    # Tips should always match by count - they're propagated from first appearance
    if len(pred_tips) != len(gt_tips):
        print(
            f"  ⚠️  Warning: Tip count mismatch - pred: {len(pred_tips)}, gt: {len(gt_tips)}"
        )
        return None

    # Direct comparison by tip index (tips are matched by identity/order)
    epe = np.linalg.norm(pred_tips - gt_tips, axis=-1).mean()

    return float(epe)


def compute_scene_flow_metrics(pred_pts, gt_pts, strict=0.025, relax=0.05):
    """
    Compute scene flow metrics (AccS, AccR).

    The metrics scripts pass leaf-tip positions in mm, not flow vectors, so the
    relative criterion divides by the tip's distance from the coordinate origin.
    The resulting `accs`/`accr` columns are not reported anywhere; EPE3D equals
    `compute_epe` up to float32 rounding.

    Args:
        pred_pts: Predicted points
        gt_pts: Ground truth points
        strict: Threshold for strict accuracy (default 0.025)
        relax: Threshold for relaxed accuracy (default 0.05)

    Returns:
        tuple: (AccS, AccR, EPE3D) or (None, None, None) if computation fails
    """
    if pred_pts is None or gt_pts is None:
        return None, None, None

    if len(pred_pts) != len(gt_pts):
        # Cannot compute scene flow metrics if point counts differ
        return None, None, None

    # Convert to tensors if needed
    if not isinstance(pred_pts, torch.Tensor):
        pred_pts = torch.from_numpy(pred_pts).float()
    if not isinstance(gt_pts, torch.Tensor):
        gt_pts = torch.from_numpy(gt_pts).float()

    try:
        epe3d, accs, accr = scene_flow_metrics(
            pred_pts, gt_pts, strict=strict, relax=relax
        )
        return accs, accr, epe3d
    except Exception as e:
        print(f"Warning: Failed to compute scene flow metrics: {e}")
        return None, None, None
