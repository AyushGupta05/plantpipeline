"""Perturbations applied to organ training data for ablation experiments.

None of these are part of normal training: they degrade the training set on
purpose, to measure how the method holds up when timepoints are missing
(`apply_holdout_last_n`), when leaves have gaps (`apply_train_holes`), or when
inputs are noisy (`save_debug_noise_plys` writes what the model actually saw).

They operate on the organ sequences directly rather than on the manager, so
what each one touches is visible in its signature.
"""

import os

import numpy as np
import torch

from growfields.utils.io_utils import save_ply_with_organ_labels

# A cloud smaller than this is left alone; removing from it would not leave
# enough surface to be a hole rather than a deletion.
_MIN_CLOUD_PTS = 200
# Never remove more than this fraction of one cloud, however many holes were asked for.
_MAX_REMOVAL_FRAC = 0.20


def apply_holdout_last_n(organ_sequences, n_timepoints, n):
    """Mask the last `n` timepoints out of every organ's training sequence.

    Used for extrapolation experiments: the model never sees these frames, and
    the metrics scripts later score predictions against them.
    """
    holdout_start = n_timepoints - n
    print(
        f"\n  Holding out last {n} timepoints "
        f"(indices {holdout_start}–{n_timepoints - 1}) from training"
    )
    for organ_seq in organ_sequences.values():
        for t in range(holdout_start, n_timepoints):
            organ_seq["train_points"][t] = None
            organ_seq["test_points"][t] = None
            organ_seq["presence_train"][t] = False
            organ_seq["presence_test"][t] = False


def apply_train_holes(
    organ_sequences,
    organ_normalizations,
    timepoint_days,
    n_holes,
    k_neighbors,
    debug_dir=None,
):
    """Cut clustered gaps into leaf training clouds, in place.

    For each leaf at each timepoint: pick `n_holes` random centres and drop the
    `k_neighbors` nearest points around each, so the gaps are contiguous patches
    rather than scattered dropout. The stem is left alone.
    """
    print(f"\n  Applying train holes: n_holes={n_holes}, k_neighbors={k_neighbors}")
    for organ_id, organ_seq in organ_sequences.items():
        if organ_id == 0:  # stem
            continue
        norm = organ_normalizations.get(organ_id)
        for t, cloud in enumerate(organ_seq["train_points"]):
            if cloud is None or cloud.shape[0] < _MIN_CLOUD_PTS:
                continue

            pts = cloud  # [N, 3], normalized
            max_remove = int(pts.shape[0] * _MAX_REMOVAL_FRAC)
            keep_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
            n_removed = 0

            for _ in range(n_holes):
                if n_removed >= max_remove:
                    break
                remaining = keep_mask.nonzero(as_tuple=False).squeeze(1)
                if remaining.numel() < 2:
                    break
                center_local = torch.randint(
                    0, remaining.numel(), (1,), device=pts.device
                ).item()
                center_idx = remaining[center_local]
                dists = torch.cdist(pts[center_idx].unsqueeze(0), pts[remaining])[0]
                k = min(k_neighbors, remaining.numel(), max_remove - n_removed)
                _, nn_local = dists.topk(k, largest=False)
                keep_mask[remaining[nn_local]] = False
                n_removed += k

            print(
                f"    organ={organ_id} t={t}: {cloud.shape[0]} → "
                f"{keep_mask.sum().item()} pts ({n_removed} removed)"
            )
            if debug_dir is not None and norm is not None:
                _save_debug_holes_ply(
                    organ_seq["type"], organ_id, t, pts, keep_mask, norm,
                    timepoint_days, debug_dir,
                )
            organ_seq["train_points"][t] = pts[keep_mask]


def _save_debug_holes_ply(
    organ_type, organ_id, t, pts_norm, keep_mask, norm, timepoint_days, debug_dir
):
    """Write the points that survived hole removal -- what the model will see."""
    min_xyz, scale = norm
    pts_mm = ((pts_norm[keep_mask].detach() + 1) / 2 * scale + min_xyz).cpu().numpy()
    labels = np.full(len(pts_mm), organ_id, dtype=np.int32)
    is_train = np.ones(len(pts_mm), dtype=np.int32)

    os.makedirs(debug_dir, exist_ok=True)
    day = timepoint_days[t] if t < len(timepoint_days) else t
    out_path = os.path.join(debug_dir, f"debug_{organ_type}_D{day:02d}.ply")
    save_ply_with_organ_labels(pts_mm, labels, is_train=is_train, output_path=out_path)
    print(f"    [debug holes] saved {organ_type} t={t} → {out_path}")


def save_debug_noise_plys(
    organ_seq,
    organ_id,
    normalization,
    noised_train_clouds,
    frame_days,
    timepoint_days,
    debug_dir,
):
    """Write the noised training clouds alongside the clean test points.

    Denormalised back to millimetres, with `is_train=1` on the noised training
    points and `0` on the clean test points, so the two are distinguishable when
    inspecting how much noise was actually applied.
    """
    if not debug_dir:
        return

    organ_type = organ_seq["type"]
    min_xyz, scale = normalization
    os.makedirs(debug_dir, exist_ok=True)

    day_to_tp = {day: tp for tp, day in enumerate(timepoint_days)}

    for cloud_norm, day in zip(noised_train_clouds, frame_days):
        pts_norm = cloud_norm.squeeze(0).detach()  # [N, 3]
        pts_np = ((pts_norm + 1) / 2 * scale + min_xyz).cpu().numpy()
        labels = np.full(len(pts_np), organ_id, dtype=np.int32)
        is_train_arr = np.ones(len(pts_np), dtype=np.int32)

        tp_idx = day_to_tp.get(day)
        if tp_idx is not None:
            test_pts = organ_seq["test_points"][tp_idx]
            if test_pts is not None:
                test_np = ((test_pts.detach() + 1) / 2 * scale + min_xyz).cpu().numpy()
                pts_np = np.vstack([pts_np, test_np])
                labels = np.concatenate(
                    [labels, np.full(len(test_np), organ_id, dtype=np.int32)]
                )
                is_train_arr = np.concatenate(
                    [is_train_arr, np.zeros(len(test_np), dtype=np.int32)]
                )

        out_path = os.path.join(debug_dir, f"debug_{organ_type}_D{day:02d}.ply")
        save_ply_with_organ_labels(
            pts_np, labels, is_train=is_train_arr, output_path=out_path
        )

    print(
        f"    [debug] saved noised PLYs for {organ_type} "
        f"({len(noised_train_clouds)} days) → {debug_dir}"
    )
