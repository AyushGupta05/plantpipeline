"""
I/O utilities for saving point clouds with attributes.
"""

import json
import os

import numpy as np
import open3d as o3d

from growfields.utils.transforms import interpolate_rigid_transform


def _clean_for_json(obj):
    # Helper to convert NaN to None for JSON serialization
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj


def write_json(obj, path, clean=False):
    """Write `obj` as indented JSON.

    Args:
        obj: The object to serialise.
        path: Destination file.
        clean: Run `_clean_for_json` first, replacing NaN with None. Needed for
            anything carrying computed metrics; plain config dumps do not.

    Returns:
        The path written.
    """
    if clean:
        obj = _clean_for_json(obj)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def save_ply_with_organ_labels(
    points,
    organ_labels,
    leaf_tip_positions=None,
    leaf_tip_labels=None,
    output_path=None,
    is_train=None,
    normals=None,
    colors=None,
):
    """
    Save point cloud with organ labels and multiple leaf tips using Open3D tensor API.

    Args:
        points: (N, 3) array of point coordinates
        organ_labels: (N,) array of organ labels (0=stem, 1=leaf_1, 2=leaf_2, etc.)
        leaf_tip_positions: (K, 3) array of leaf tip positions, or None
        leaf_tip_labels: (K,) array of organ labels for each tip, or None
        output_path: Path to save PLY file
        is_train: (N,) array where 1=train, 0=test, -1=unknown (optional)
        normals: (N, 3) array of surface normals (optional)
    """
    # Create a tensor-based point cloud
    pcd = o3d.t.geometry.PointCloud(
        o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    )

    # Add normals if provided
    if normals is not None:
        pcd.point["normals"] = o3d.core.Tensor(normals, dtype=o3d.core.Dtype.Float32)

    # Add per-point RGB colors (e.g. XYZ tracking colors)
    if colors is not None:
        pcd.point["red"] = o3d.core.Tensor(
            colors[:, 0:1].astype(np.uint8), dtype=o3d.core.Dtype.UInt8
        )
        pcd.point["green"] = o3d.core.Tensor(
            colors[:, 1:2].astype(np.uint8), dtype=o3d.core.Dtype.UInt8
        )
        pcd.point["blue"] = o3d.core.Tensor(
            colors[:, 2:3].astype(np.uint8), dtype=o3d.core.Dtype.UInt8
        )

    # Add organ labels
    if organ_labels is not None:
        pcd.point["organ_label"] = o3d.core.Tensor(
            organ_labels.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
        )

    # Add train/test split info
    if is_train is not None:
        pcd.point["is_train"] = o3d.core.Tensor(
            is_train.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
        )

    # Mark leaf tip points
    if leaf_tip_positions is not None and len(leaf_tip_positions) > 0:
        leaf_tip_mask = np.zeros(len(points), dtype=np.uint8)
        tip_organ_labels = np.zeros(len(points), dtype=np.int32)
        claimed = set()

        for i, tip_pos in enumerate(leaf_tip_positions):
            distances = np.linalg.norm(points - tip_pos.reshape(1, 3), axis=1)
            sorted_indices = np.argsort(distances)
            # Find nearest unclaimed point
            for tip_idx in sorted_indices:
                tip_idx = int(tip_idx)
                if tip_idx not in claimed:
                    break
            claimed.add(tip_idx)
            leaf_tip_mask[tip_idx] = 1
            if leaf_tip_labels is not None:
                tip_organ_labels[tip_idx] = leaf_tip_labels[i]

        pcd.point["is_leaf_tip"] = o3d.core.Tensor(
            leaf_tip_mask.reshape(-1, 1), dtype=o3d.core.Dtype.UInt8
        )

        if leaf_tip_labels is not None:
            pcd.point["tip_organ_label"] = o3d.core.Tensor(
                tip_organ_labels.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
            )

    # Save as PLY
    if output_path is not None:
        o3d.t.io.write_point_cloud(str(output_path), pcd)


def estimate_normals(points, labels=None, ref_points=None, ref_normals=None, ref_labels=None):
    """Consistently oriented normals for a predicted cloud, for visualisation.

    The models predict positions only. Normals are fitted to each organ's 30
    nearest neighbours, train and test points together, and each is flipped to
    agree with the nearest reference normal of the same organ -- ground truth
    for that frame, whose normals are consistently oriented. Organs without a
    reference fall back to Open3D's tangent-plane propagation, which is
    consistent within the organ but of arbitrary overall sign.

    No metric depends on this: NC compares normals up to sign.

    Args:
        points: (N, 3) array.
        labels: (N,) organ labels, or None to treat the cloud as one organ.
        ref_points, ref_normals: (M, 3) arrays in the same space as `points`, or None.
        ref_labels: (M,) organ labels of the reference, or None.

    Returns:
        (N, 3) float32 array.
    """
    from scipy.spatial import cKDTree

    if labels is None:
        labels = np.zeros(len(points), dtype=np.int32)
    has_ref = ref_points is not None and ref_normals is not None and len(ref_points) > 0
    if has_ref and ref_labels is None:
        ref_labels = np.zeros(len(ref_points), dtype=np.int32)

    normals = np.zeros((len(points), 3), dtype=np.float32)
    for organ_id in np.unique(labels):
        mask = labels == organ_id
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[mask].astype(np.float64)))
        pcd.estimate_normals()
        ref_mask = ref_labels == organ_id if has_ref else None
        if ref_mask is None or not ref_mask.any():
            if mask.sum() > 3:
                pcd.orient_normals_consistent_tangent_plane(30)
            normals[mask] = np.asarray(pcd.normals)
            continue
        organ_normals = np.asarray(pcd.normals)
        _, nearest = cKDTree(ref_points[ref_mask]).query(points[mask])
        agree = np.sum(organ_normals * ref_normals[ref_mask][nearest], axis=1)
        normals[mask] = np.where(agree[:, None] < 0, -organ_normals, organ_normals)
    return normals


def _save_predicted_clouds(
    advance,
    n_transitions,
    clouds,
    labels,
    min_xyz,
    scale,
    out_dir,
    cloud_paths=None,
    gt_clouds=None,
    gt_labels=None,
    gt_tips=None,
    gt_tip_labels=None,
    is_train=None,
    normals=None,
    save_normals=True,
):
    """Write predicted and ground-truth clouds for a whole-plant sequence.

    Pairwise and continuous models differ only in how a cloud moves from one
    frame to the next, so that step is passed in: `advance(points, i)` carries
    `points` from frame `i` to frame `i + 1`, and is applied to the cloud and to
    each tracked leaf tip alike.

    Args:
        advance: Callable moving points one frame forward.
        n_transitions: Number of frames to predict, i.e. how many times to call it.
        is_train: List of (N,) arrays where 1=train, 0=test for each timepoint.
                  If provided, this mask is saved with both predictions and GT.
        normals: Dataset normals per timepoint, shaped like `clouds`; they orient
            the predicted normals (see `estimate_normals`), and are written with
            `gt_clouds`.
        save_normals: Estimate normals for the predicted clouds.
    """
    import torch

    from growfields.models.integration import denormalise

    pred_dir = os.path.join(out_dir, "predicted")
    gt_dir = os.path.join(out_dir, "ground_truth")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Track tips by organ ID - tips are added as they appear and warped forward
    tips_by_organ = {}  # organ_id -> normalized_tip_tensor

    # Initialize with tips from frame 0 if they exist
    if gt_tips is not None and len(gt_tips) > 0:
        if gt_tips[0] is not None and gt_tip_labels[0] is not None:
            for tip_gt, label in zip(gt_tips[0], gt_tip_labels[0]):
                organ_id = label.item()
                tips_by_organ[organ_id] = (tip_gt - min_xyz[0, 0]) / scale[0, 0] * 2 - 1

    x = clouds[0][0].clone()
    x_labels = labels[0][0].clone()

    # Get is_train mask for frame 0 (predictions inherit this)
    is_train_mask_0 = None
    if is_train is not None and len(is_train) > 0:
        is_train_mask_0 = (
            is_train[0].cpu().numpy() if hasattr(is_train[0], "cpu") else is_train[0]
        )

    print(f"Saving predictions to {out_dir}")

    with torch.no_grad():
        for i in range(n_transitions):
            x = advance(x, i)
            pred = denormalise(x.unsqueeze(0), min_xyz, scale)[0].detach().cpu().numpy()
            pred_labels_np = x_labels.detach().cpu().numpy()

            # Warp all existing tips forward
            for organ_id in tips_by_organ:
                tips_by_organ[organ_id] = advance(tips_by_organ[organ_id], i)

            # Track which tips existed before this frame (these were just warped)
            organs_warped = set(tips_by_organ.keys())

            # Add newly appearing tips from GT at frame i+1 (but don't save until next frame)
            if gt_tips is not None and i + 1 < len(gt_tips):
                if gt_tips[i + 1] is not None and gt_tip_labels[i + 1] is not None:
                    for tip_gt, label in zip(gt_tips[i + 1], gt_tip_labels[i + 1]):
                        organ_id = label.item()
                        if organ_id not in tips_by_organ:
                            # New tip appearing - initialize from GT (will be warped in next iteration)
                            # `min_xyz` is (1, 1, 3) and `scale` is (1, 1) here, so
                            # [0, 0] takes the per-axis minimum and the scalar
                            # scale -- the right shapes for a single (3,) tip. This
                            # is the exact inverse of `denormalise`, which is
                            # `(x + 1) / 2 * scale + min_xyz`; the round trip is
                            # numerically exact.
                            tips_by_organ[organ_id] = (tip_gt - min_xyz[0, 0]) / scale[
                                0, 0
                            ] * 2 - 1

            # Prepare predicted tips - save ALL warped tips (not just ones in current GT)
            pred_tips_np = None
            pred_tip_labels_np = None
            if organs_warped:
                pred_tips_list = []
                pred_tip_labels_list = []
                for organ_id in sorted(organs_warped):  # Sort for consistent ordering
                    pred_tips_list.append(tips_by_organ[organ_id])
                    pred_tip_labels_list.append(organ_id)

                # Denormalize tips
                pred_tips_normalized_batch = torch.stack(pred_tips_list)
                pred_tips_denorm = denormalise(
                    pred_tips_normalized_batch.unsqueeze(0), min_xyz, scale
                )[0]
                pred_tips_np = pred_tips_denorm.detach().cpu().numpy()
                pred_tip_labels_np = np.array(pred_tip_labels_list, dtype=np.int32)

            # Get GT tips
            gt_tips_frame_np = None
            gt_tip_labels_frame_np = None
            if gt_tips is not None and i + 1 < len(gt_tips):
                if gt_tips[i + 1] is not None and len(gt_tips[i + 1]) > 0:
                    gt_tips_frame_np = gt_tips[i + 1].cpu().numpy()
                    gt_tip_labels_frame_np = gt_tip_labels[i + 1].cpu().numpy()

            base_name = (
                os.path.basename(cloud_paths[i + 1])
                if cloud_paths
                else f"frame_{i+1:03d}.ply"
            )

            pred_normals = None
            if save_normals:
                ref = {}
                if normals is not None and i + 1 < len(normals):
                    ref = dict(
                        ref_points=denormalise(clouds[i + 1], min_xyz, scale)[0].cpu().numpy(),
                        ref_normals=normals[i + 1][0].cpu().numpy(),
                        ref_labels=labels[i + 1][0].cpu().numpy(),
                    )
                pred_normals = estimate_normals(pred, pred_labels_np, **ref)

            save_ply_with_organ_labels(
                pred,
                pred_labels_np,
                pred_tips_np,
                pred_tip_labels_np,
                os.path.join(pred_dir, f"pred_{base_name}"),
                is_train=is_train_mask_0,
                normals=pred_normals,
            )

            if gt_clouds is not None and i + 1 < len(gt_clouds):
                gt = (
                    denormalise(gt_clouds[i + 1], min_xyz, scale)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                gt_labels_np = gt_labels[i + 1][0].detach().cpu().numpy()
                # Get is_train mask for this GT frame
                is_train_mask_gt = None
                if is_train is not None and i + 1 < len(is_train):
                    is_train_mask_gt = (
                        is_train[i + 1].cpu().numpy()
                        if hasattr(is_train[i + 1], "cpu")
                        else is_train[i + 1]
                    )

                gt_normals = None
                if normals is not None and i + 1 < len(normals):
                    gt_normals = normals[i + 1][0].cpu().numpy()

                save_ply_with_organ_labels(
                    gt,
                    gt_labels_np,
                    gt_tips_frame_np,
                    gt_tip_labels_frame_np,
                    os.path.join(gt_dir, f"gt_{base_name}"),
                    is_train=is_train_mask_gt,
                    normals=gt_normals,
                )

    print(f"✅ Saved {n_transitions} predicted point clouds")


def save_predictions_pairwise(
    models,
    clouds,
    labels,
    min_xyz,
    scale,
    out_dir,
    cloud_paths=None,
    gt_clouds=None,
    gt_labels=None,
    gt_tips=None,
    gt_tip_labels=None,
    is_train=None,
    normals=None,
    save_normals=True,
):
    """Save predictions from a chain of pairwise models, one per transition.

    Each model predicts a displacement, so a frame is advanced by adding its
    output to the current points.
    """

    def advance(points, i):
        return points + models[i](points)

    _save_predicted_clouds(
        advance, len(models), clouds, labels, min_xyz, scale, out_dir,
        cloud_paths=cloud_paths, gt_clouds=gt_clouds, gt_labels=gt_labels,
        gt_tips=gt_tips, gt_tip_labels=gt_tip_labels, is_train=is_train,
        normals=normals, save_normals=save_normals,
    )


def save_predictions_continuous(
    model,
    clouds,
    labels,
    min_xyz,
    scale,
    out_dir,
    frame_days,
    full_day_range,
    use_day_based_tnorm=True,
    cloud_paths=None,
    gt_clouds=None,
    gt_labels=None,
    gt_tips=None,
    gt_tip_labels=None,
    is_train=None,
    use_odeint=False,
    ode_method="dopri5",
    ode_rtol=1e-4,
    ode_atol=1e-5,
    train_integration_steps=1,
    ode_n_steps=None,
    normals=None,
    save_normals=True,
):
    """Save predictions from one time-conditioned model covering the sequence.

    The model gives velocity at (x, t), so a frame is advanced by integrating
    between the two timepoints' normalised times.

    Args:
        frame_days: List of day values, one per cloud (length = num_frames)
        full_day_range: Tuple (first_day, last_day) for the full timeseries
    """
    from growfields.models.integration import _integrate_segment, make_t_norm_helpers

    num_frames = len(clouds)
    first_day, last_day = full_day_range
    get_t_norm, get_dt = make_t_norm_helpers(
        frame_days, first_day, last_day - first_day, num_frames, use_day_based_tnorm
    )
    ode_kw = dict(
        use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
        ode_method=ode_method, use_adjoint=False,
        n_substeps=train_integration_steps, ode_n_steps=ode_n_steps,
    )

    def advance(points, i):
        moved, _ = _integrate_segment(
            model, points, i, i + 1, get_t_norm, get_dt, **ode_kw
        )
        return moved

    _save_predicted_clouds(
        advance, num_frames - 1, clouds, labels, min_xyz, scale, out_dir,
        cloud_paths=cloud_paths, gt_clouds=gt_clouds, gt_labels=gt_labels,
        gt_tips=gt_tips, gt_tip_labels=gt_tip_labels, is_train=is_train,
        normals=normals, save_normals=save_normals,
    )


def save_gt_from_dataset(dataset, out_dir, cloud_paths=None, skip_first_frame=True):
    """Save GT PLY files from the raw plant dataset (original order, raw coordinates).

    Reads GT directly from the underlying plant dataset, preserving original point
    order and coordinates. Builds is_train mask from the split file indices.

    Args:
        dataset: TrackPlant3DDataset instance (plant mode, with split_file set).
        out_dir: Output directory; GT files go into ``out_dir/ground_truth/``.
        cloud_paths: List of timepoint names (one per frame) for filenames.
        skip_first_frame: If True (default), skip frame 0 (predictions start at frame 1).
    """
    import json as _json

    gt_dir = os.path.join(out_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    # Get raw plant timeseries (original point order, plant-space coordinates)
    plant_seq = dataset.plant_dataset.get_timeseries_by_sequence_name(
        dataset.sequence_name
    )
    if plant_seq is None:
        print("  ⚠️  Could not find plant timeseries for GT saving")
        return

    # Build day -> plant timepoint lookup
    day_to_plant_tp = {tp["day"]: tp for tp in plant_seq["timepoints"]}

    # Load split data for index-based is_train mask
    split_data = None
    if (
        hasattr(dataset, "split_file")
        and dataset.split_file
        and os.path.isfile(dataset.split_file)
    ):
        with open(dataset.split_file) as f:
            split_data = _json.load(f)

    # Get the day list from the strided dataset (same frames used for training)
    days = [
        tp.get("day", i)
        for i, tp in enumerate(dataset.sequence_data["timepoints"])
    ]

    start = 1 if skip_first_frame else 0
    saved = 0

    for t in range(start, len(days)):
        day = days[t]
        plant_tp = day_to_plant_tp.get(day)
        if plant_tp is None:
            continue

        gt_points = plant_tp["points"]
        gt_labels = plant_tp["labels"].astype(np.int32)
        gt_normals = plant_tp.get("normals")

        # Build is_train from split file (index-based, preserves original order)
        gt_is_train = None
        if split_data and t < len(split_data["timepoints"]):
            split_info = split_data["timepoints"][t]
            gt_is_train = np.full(len(gt_points), -1, dtype=np.int32)
            train_indices = split_info.get("train_indices", [])
            test_indices = split_info.get("test_indices", [])
            if train_indices:
                gt_is_train[np.array(train_indices)] = 1
            if test_indices:
                gt_is_train[np.array(test_indices)] = 0

        # Get leaf tips from plant timepoint
        gt_tips_np = None
        gt_tip_labels_np = None
        leaf_tip_idxs = plant_tp.get("leaf_tip_idxs", np.array([]))
        if leaf_tip_idxs.size > 0:
            gt_tips_np = gt_points[leaf_tip_idxs]
            gt_tip_labels_np = gt_labels[leaf_tip_idxs]

        base_name = (
            os.path.basename(cloud_paths[t]) if cloud_paths else f"frame_{t:03d}.ply"
        )

        save_ply_with_organ_labels(
            gt_points,
            gt_labels,
            gt_tips_np,
            gt_tip_labels_np,
            os.path.join(gt_dir, f"gt_{base_name}"),
            is_train=gt_is_train,
            normals=gt_normals,
        )
        saved += 1

    print(f"✅ Saved {saved} GT point clouds from raw dataset")


#: Model types whose organ model predicts a velocity field, as opposed to the
#: pairwise chain of one displacement model per transition.
_CONTINUOUS_MODEL_TYPES = (
    "continuous",
    "shared_continuous",
    "encoder_decoder_continuous",
)


def _organ_conditioning(config, organ_info, organ_id, valid_clouds):
    """Return `(model, integration kwargs, velocity)` for a velocity-field organ.

    The three continuous variants differ only in what conditions the model: a
    shared SIREN takes an organ id, the encoder-decoder takes a latent code
    encoded from the organ's first cloud, and a plain continuous SIREN takes
    neither. Both forms are returned because the integrators take it as a
    keyword while a direct forward call takes it positionally.

    Returns None when the organ has no trained model.
    """
    import torch

    model = organ_info.get("trained_model")
    if model is None:
        return None

    model_type = config["model_type"]
    if model_type == "shared_continuous":
        extra = {"organ_id": organ_id}

        def velocity(points, t):
            return model(points, t, organ_id)

    elif model_type == "encoder_decoder_continuous":
        with torch.no_grad():
            latent_code = model.encode(valid_clouds[0])

        extra = {"latent_code": latent_code}

        def velocity(points, t):
            return model(points, t, latent_code)

    else:
        extra = {}

        def velocity(points, t):
            return model(points, t)

    return model, extra, velocity


def _make_organ_advance(
    config,
    organ_info,
    organ_id,
    valid_clouds,
    *,
    organ_train_days,
    organ_full_day_range,
    use_day_based_tnorm,
    ode_settings,
    use_pre_finetune_models=False,
    n_transitions_needed=0,
):
    """Return `advance(points, i)` for one organ, or None if it cannot predict.

    Every model type writes predictions the same way; they differ only in how a
    point cloud moves from one of the organ's timepoints to the next. Returning
    None means the organ has no usable model, which the callers treat as "skip
    this organ".

    Args:
        n_transitions_needed: How many steps the caller intends to take. Only
            the pairwise chain can come up short, having one model per
            transition.
    """
    import torch

    from growfields.models.integration import _integrate_segment, make_t_norm_helpers

    model_type = config["model_type"]

    if model_type == "pairwise":
        if use_pre_finetune_models and "pre_finetune_models" in organ_info:
            models = organ_info["pre_finetune_models"]
        else:
            models = organ_info.get("trained_models", [])
        if not models or n_transitions_needed > len(models):
            return None

        def advance(points, i):
            return points + models[i](points)

        return advance

    model = organ_info.get("trained_model")
    if model is None:
        return None

    first_day, last_day = organ_full_day_range[organ_id]
    get_t_norm, get_dt = make_t_norm_helpers(
        organ_train_days[organ_id],
        first_day,
        last_day - first_day,
        len(valid_clouds),
        use_day_based_tnorm,
    )

    ode_kw = dict(ode_settings, use_adjoint=False)
    if model_type == "shared_continuous":
        ode_kw["organ_id"] = organ_id
    elif model_type == "encoder_decoder_continuous":
        with torch.no_grad():
            ode_kw["latent_code"] = model.encode(valid_clouds[0])

    def advance(points, i):
        moved, _ = _integrate_segment(
            model, points, i, i + 1, get_t_norm, get_dt, **ode_kw
        )
        return moved

    return advance


def save_full_plant_from_organ_models(
    organ_models,
    organ_sequences,
    organ_normalizations,
    organ_leaf_tips,
    dataset,
    out_dir,
    sequence_name=None,
    timepoint_days=None,
    device="cuda",
    save_ground_truth=True,
    use_pre_finetune_models=False,
    organ_transformations=None,
    organ_train_days=None,
    organ_full_day_range=None,
    use_day_based_tnorm=True,
    use_odeint=False,
    ode_method="dopri5",
    ode_rtol=1e-4,
    ode_atol=1e-5,
    train_integration_steps=1,
    ode_n_steps=None,
    input_noise_std=0.0,
    label_field="labels",
    save_normals=True,
):
    """
    Save full plant predictions by concatenating predictions from all organ models.

    Args:
        organ_models: Dict of organ_id -> model_info from OrganModelManager
        organ_sequences: Dict of organ_id -> sequence_info from OrganModelManager
        organ_normalizations: Dict of organ_id -> (min_xyz, scale) from OrganModelManager
        organ_leaf_tips: Dict of organ_id -> list of tips from OrganModelManager
        dataset: TrackPlant3DDataset to get ground truth from
        out_dir: Output directory for full plant PLYs
        sequence_name: Sequence name for filenames (optional)
        timepoint_days: List of actual day indices (optional)
        device: torch device
        save_ground_truth: Whether to also save ground truth clouds
        use_pre_finetune_models: If True, use pre-finetune models for pairwise organs
        organ_transformations: Dict of organ_id -> transformation_stages for inverse transforms (optional)
        organ_train_days: Dict of organ_id -> list of training days when organ present
        organ_full_day_range: Dict of organ_id -> (first_day, last_day) from full unstrided data
        use_day_based_tnorm: If True, use day-based t_norm and velocity*dt. If False, legacy behavior.
        save_normals: Estimate normals for the predicted clouds, oriented against
            that day's ground truth (see `estimate_normals`).
    """
    import torch

    from growfields.models.integration import denormalise

    # Import LeafSequencesDataset utilities for inverse transformations
    try:
        from plant_shape_analysis import LeafSequencesDataset

        has_transform_utils = True
    except ImportError:
        has_transform_utils = False
        if organ_transformations:
            print(
                "Warning: plant_shape_analysis not available, cannot apply inverse transformations"
            )

    pred_dir = os.path.join(out_dir, "predicted")
    gt_dir = os.path.join(out_dir, "ground_truth")
    os.makedirs(pred_dir, exist_ok=True)
    if save_ground_truth:
        os.makedirs(gt_dir, exist_ok=True)

    # Determine number of frames from first organ
    first_organ_id = list(organ_models.keys())[0]
    first_organ_seq = organ_sequences[first_organ_id]

    # Combine train and test points for full plant prediction
    if "train_points" in first_organ_seq and "test_points" in first_organ_seq:
        valid_clouds_first = []
        for train_pts, test_pts in zip(
            first_organ_seq["train_points"], first_organ_seq["test_points"]
        ):
            if train_pts is not None and test_pts is not None:
                valid_clouds_first.append(torch.cat([train_pts, test_pts], dim=0))
            elif train_pts is not None:
                # test_pts is None (e.g. organ has no test points at this timepoint)
                valid_clouds_first.append(train_pts)
            else:
                valid_clouds_first.append(None)
        valid_clouds_first = [c for c in valid_clouds_first if c is not None]
    else:
        # Backward compatibility
        valid_clouds_first = [c for c in first_organ_seq["points"] if c is not None]

    num_frames = len(valid_clouds_first) - 1

    # Integration settings, identical for every organ.
    _ode_settings = dict(
        use_odeint=use_odeint,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        ode_method=ode_method,
        n_substeps=train_integration_steps,
        ode_n_steps=ode_n_steps,
    )

    # Get GT organ presence from dataset for each timepoint
    gt_organs_by_timepoint = {}
    for t in range(1, dataset.nt):  # Start from 1 since we predict from t=0
        all_labels_t = torch.cat(
            [dataset.train_labels[t], dataset.test_labels[t]], dim=0
        )
        gt_organs_by_timepoint[t] = set(all_labels_t.cpu().numpy())

    # Pre-compute inverse transformation matrices for all organs (if available)
    organ_inverse_matrices = {}
    if organ_transformations and has_transform_utils:
        for organ_id, transformation_stages in organ_transformations.items():
            # Compose all transformation stages for this organ
            composed = LeafSequencesDataset.compose_transformations(
                transformation_stages
            )
            # Pre-compute inverse matrices for each timepoint
            organ_inverse_matrices[organ_id] = [
                LeafSequencesDataset.invert_transformation(comp["composed_matrix"])
                for comp in composed
            ]
        print(
            f"  Pre-computed inverse transformations for {len(organ_inverse_matrices)} organs"
        )

    # Load plant-space GT data directly from the plant dataset (original order)
    # This avoids the train/test reordering and alignment round-trip issues
    # Also used for adding reference clouds of organs without trained models.
    day_to_plant_tp = {}
    plant_dataset = dataset.leaf_dataset.plant_dataset
    full_plant_seq = plant_dataset.get_timeseries_by_sequence_name(
        dataset.sequence_name
    )
    if full_plant_seq is not None:
        for tp in full_plant_seq["timepoints"]:
            day_to_plant_tp[tp["day"]] = tp

    # Load split data for index-based is_train mask
    split_data = None
    if (
        hasattr(dataset, "split_file")
        and dataset.split_file
        and os.path.isfile(dataset.split_file)
    ):
        import json as _json

        with open(dataset.split_file) as f:
            split_data = _json.load(f)

    # Generate predictions frame by frame
    for frame_idx in range(num_frames):
        # Target timepoint is frame_idx + 1 (since we start from t=0)
        target_timepoint = frame_idx + 1

        # Determine filename
        if sequence_name and timepoint_days and target_timepoint < len(timepoint_days):
            day = timepoint_days[target_timepoint]
            base_name = f"{sequence_name}_D{day:02d}.ply"
        else:
            base_name = f"frame_{target_timepoint:02d}.ply"

        all_pred_points = []
        all_pred_labels = []
        all_pred_tips = []
        all_pred_tip_labels = []
        all_pred_is_train = []  # Track train/test split

        # Get which organs should be present at target timepoint from GT
        organs_at_target = gt_organs_by_timepoint.get(
            target_timepoint, set(organ_models.keys())
        )

        for organ_id in sorted(organ_models.keys()):
            # Skip organ if it's not present at the target timepoint
            if organ_id not in organs_at_target:
                continue
            organ_info = organ_models[organ_id]
            config = organ_info["config"]
            organ_seq = organ_sequences[organ_id]

            # Get valid timepoint indices for this organ
            valid_timepoint_indices = [
                i for i, present in enumerate(organ_seq["presence_train"]) if present
            ]

            # Find organ's local index for the target timepoint
            if target_timepoint not in valid_timepoint_indices:
                continue
            organ_local_target_idx = valid_timepoint_indices.index(target_timepoint)

            # Get valid clouds (combine train and test for full plant prediction)
            # Also track is_train mask for initial frame (frame 0)
            organ_is_train_mask = None
            if "train_points" in organ_seq and "test_points" in organ_seq:
                valid_clouds = []
                for train_pts, test_pts in zip(
                    organ_seq["train_points"], organ_seq["test_points"]
                ):
                    if train_pts is not None and test_pts is not None:
                        valid_clouds.append(torch.cat([train_pts, test_pts], dim=0))
                    elif train_pts is not None:
                        # test_pts is None (e.g. organ has no test points at this timepoint)
                        valid_clouds.append(train_pts)
                    else:
                        valid_clouds.append(None)
                valid_clouds = [c for c in valid_clouds if c is not None]

                # Create is_train mask for first VALID frame (predictions inherit this)
                # First half are train (1), second half are test (0)
                # Use the first valid timepoint for this organ, not global timepoint 0
                if len(valid_clouds) > 0 and len(valid_timepoint_indices) > 0:
                    first_valid_t = valid_timepoint_indices[0]
                    train_pts_0 = organ_seq["train_points"][first_valid_t]
                    test_pts_0 = organ_seq["test_points"][first_valid_t]
                    if train_pts_0 is not None and test_pts_0 is not None:
                        n_train = len(train_pts_0)
                        n_test = len(test_pts_0)
                        organ_is_train_mask = np.concatenate(
                            [
                                np.ones(n_train, dtype=np.int32),
                                np.zeros(n_test, dtype=np.int32),
                            ]
                        )
                    elif train_pts_0 is not None:
                        # All organ points landed in train split at this timepoint
                        organ_is_train_mask = np.ones(len(train_pts_0), dtype=np.int32)
            else:
                # Backward compatibility
                valid_clouds = [c for c in organ_seq["points"] if c is not None]
            if len(valid_clouds) < 2 or organ_local_target_idx >= len(valid_clouds):
                continue

            # Get normalization
            min_xyz, scale = organ_normalizations[organ_id]
            min_xyz_batch = min_xyz.unsqueeze(0).unsqueeze(0)
            scale_batch = (
                torch.tensor([scale.item()], device=device).unsqueeze(0).unsqueeze(0)
            )

            # Get leaf tips
            gt_tips_organ = organ_leaf_tips.get(organ_id, None)

            # Initialize tip
            tip_normalized = None
            tip_start_idx = None
            if gt_tips_organ is not None:
                for idx, tip in enumerate(gt_tips_organ):
                    if tip is not None:
                        tip_normalized = (tip - min_xyz) / scale * 2 - 1
                        tip_normalized = tip_normalized.unsqueeze(0)
                        tip_start_idx = idx
                        break

            # Generate prediction for this organ at this frame
            advance = _make_organ_advance(
                config,
                organ_info,
                organ_id,
                valid_clouds,
                organ_train_days=organ_train_days,
                organ_full_day_range=organ_full_day_range,
                use_day_based_tnorm=use_day_based_tnorm,
                ode_settings=_ode_settings,
                use_pre_finetune_models=use_pre_finetune_models,
                n_transitions_needed=organ_local_target_idx,
            )
            if advance is None:
                continue

            x = valid_clouds[0].clone()
            with torch.no_grad():
                # Re-warping from frame 0 for every target frame repeats work, but
                # it keeps memory flat and this only runs while saving.
                for i in range(organ_local_target_idx):
                    current_global_t = valid_timepoint_indices[i + 1]
                    x = advance(x, i)
                    # A tip only moves once it has appeared.
                    if tip_normalized is not None and current_global_t > tip_start_idx:
                        tip_normalized = advance(tip_normalized, i)

                pred = (
                    denormalise(x.unsqueeze(0), min_xyz_batch, scale_batch)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                pred_tip = None
                if tip_normalized is not None and target_timepoint > tip_start_idx:
                    pred_tip = (
                        denormalise(
                            tip_normalized.unsqueeze(0), min_xyz_batch, scale_batch
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

            # Apply inverse transformation if this is an aligned leaf (not stem)
            if organ_id > 0 and organ_id in organ_inverse_matrices:
                # Get the pre-computed inverse matrix for this timepoint
                if organ_local_target_idx < len(organ_inverse_matrices[organ_id]):
                    inv_matrix = organ_inverse_matrices[organ_id][
                        organ_local_target_idx
                    ]

                    # Apply inverse transformation to predicted points
                    pred = LeafSequencesDataset.apply_transformation(pred, inv_matrix)

                    # Apply inverse transformation to predicted tip if available
                    if pred_tip is not None:
                        pred_tip_reshaped = pred_tip.reshape(1, -1)
                        pred_tip = LeafSequencesDataset.apply_transformation(
                            pred_tip_reshaped, inv_matrix
                        )[0]

            # Add to full plant prediction
            all_pred_points.append(pred)
            all_pred_labels.append(np.full(len(pred), organ_id, dtype=np.int32))
            if organ_is_train_mask is not None:
                all_pred_is_train.append(organ_is_train_mask)
            if pred_tip is not None:
                all_pred_tips.append(pred_tip)
                all_pred_tip_labels.append(organ_id)

        # Add reference clouds for organs present in GT but without a trained
        # model (e.g. organs that first appeared at the last timepoint).
        # Track (start_idx, normals) so they keep their dataset normals.
        ref_organ_normals = []  # list of (start_idx, normals_array)
        plant_tp = None
        if timepoint_days and target_timepoint < len(timepoint_days):
            day = timepoint_days[target_timepoint]
            plant_tp = day_to_plant_tp.get(day)
            if plant_tp is not None:
                predicted_organ_ids = (
                    set(int(l[0]) for l in all_pred_labels)
                    if all_pred_labels
                    else set()
                )
                n_pred_so_far = sum(len(p) for p in all_pred_points)
                _gt_labels_field = plant_tp.get(label_field, plant_tp["labels"])
                gt_organ_ids = set(int(o) for o in np.unique(_gt_labels_field))
                for missing_id in sorted(gt_organ_ids - predicted_organ_ids):
                    organ_mask = _gt_labels_field == missing_id
                    ref_points = plant_tp["points"][organ_mask]
                    if plant_tp.get("normals") is not None:
                        ref_organ_normals.append(
                            (n_pred_so_far, plant_tp["normals"][organ_mask])
                        )
                    n_pred_so_far += len(ref_points)
                    all_pred_points.append(ref_points)
                    all_pred_labels.append(
                        np.full(len(ref_points), missing_id, dtype=np.int32)
                    )
                    # Derive is_train from the split file
                    if split_data and target_timepoint < len(split_data["timepoints"]):
                        split_info = split_data["timepoints"][target_timepoint]
                        full_is_train = np.full(
                            len(plant_tp["points"]), -1, dtype=np.int32
                        )
                        train_indices = split_info.get("train_indices", [])
                        test_indices = split_info.get("test_indices", [])
                        if train_indices:
                            full_is_train[np.array(train_indices)] = 1
                        if test_indices:
                            full_is_train[np.array(test_indices)] = 0
                        all_pred_is_train.append(full_is_train[organ_mask])

        # Concatenate predictions from all organs
        if all_pred_points:
            full_pred_points = np.concatenate(all_pred_points, axis=0)
            full_pred_labels = np.concatenate(all_pred_labels, axis=0)
            full_pred_tips = np.array(all_pred_tips) if all_pred_tips else None
            full_pred_tip_labels = (
                np.array(all_pred_tip_labels, dtype=np.int32)
                if all_pred_tip_labels
                else None
            )
            full_pred_is_train = (
                np.concatenate(all_pred_is_train, axis=0) if all_pred_is_train else None
            )

            full_pred_normals = None
            if save_normals:
                ref = {}
                if plant_tp is not None and plant_tp.get("normals") is not None:
                    ref = dict(
                        ref_points=plant_tp["points"],
                        ref_normals=plant_tp["normals"],
                        ref_labels=np.asarray(plant_tp.get(label_field, plant_tp["labels"])),
                    )
                full_pred_normals = estimate_normals(full_pred_points, full_pred_labels, **ref)
                # Organs copied from ground truth keep its normals exactly.
                for start_idx, orig_normals in ref_organ_normals:
                    end_idx = start_idx + len(orig_normals)
                    full_pred_normals[start_idx:end_idx] = orig_normals

            save_ply_with_organ_labels(
                full_pred_points,
                full_pred_labels,
                full_pred_tips,
                full_pred_tip_labels,
                os.path.join(pred_dir, f"pred_{base_name}"),
                is_train=full_pred_is_train,
                normals=full_pred_normals,
            )

        # Get GT directly from plant dataset (original point order, plant space)
        # This avoids the train/test reordering and alignment round-trip issues
        # that occur when using cat(train_pts, test_pts) from the leaf-aligned dataset.
        if (
            save_ground_truth
            and timepoint_days
            and target_timepoint < len(timepoint_days)
        ):
            day = timepoint_days[target_timepoint]
            plant_tp = day_to_plant_tp.get(day)
            if plant_tp is not None:
                gt_points_np = plant_tp["points"]
                gt_labels_np = np.array(plant_tp.get(label_field, plant_tp["labels"]), dtype=np.int32)
                gt_normals = plant_tp.get("normals")

                # Build is_train from split file (index-based, preserves original order)
                gt_is_train = None
                if split_data and target_timepoint < len(split_data["timepoints"]):
                    split_info = split_data["timepoints"][target_timepoint]
                    gt_is_train = np.full(len(gt_points_np), -1, dtype=np.int32)
                    train_indices = split_info.get("train_indices", [])
                    test_indices = split_info.get("test_indices", [])
                    if train_indices:
                        gt_is_train[np.array(train_indices)] = 1
                    if test_indices:
                        gt_is_train[np.array(test_indices)] = 0

                # Add approximate noise to GT train points for visualization.
                # NOTE: this is a fresh noise draw, NOT the exact realizations seen during
                # training. Included only to visually inspect the noise level; do NOT use
                # these GT PLYs to compute metrics when input_noise_std > 0.
                if input_noise_std > 0 and gt_is_train is not None:
                    gt_points_np = gt_points_np.copy()
                    train_mask = gt_is_train == 1
                    for oid in np.unique(gt_labels_np[train_mask]):
                        organ_mask = train_mask & (gt_labels_np == oid)
                        if oid in organ_normalizations:
                            _, scale = organ_normalizations[oid]
                            # Factor /2: normalization maps to [-1,1] via (pts-min)/scale*2-1,
                            # so noise_norm * scale/2 = noise_mm.
                            noise_mm = input_noise_std * scale.item() / 2
                            gt_points_np[organ_mask] += (
                                np.random.randn(organ_mask.sum(), 3).astype(np.float32) * noise_mm
                            )

                # Get tips directly from plant timepoint (exact positions, no round-trip)
                gt_tips_np = None
                gt_tip_labels_np = None
                leaf_tip_idxs = plant_tp.get("leaf_tip_idxs", np.array([]))
                if leaf_tip_idxs.size > 0:
                    gt_tips_np = gt_points_np[leaf_tip_idxs]
                    gt_tip_labels_np = gt_labels_np[leaf_tip_idxs]

                save_ply_with_organ_labels(
                    gt_points_np,
                    gt_labels_np,
                    gt_tips_np,
                    gt_tip_labels_np,
                    os.path.join(gt_dir, f"gt_{base_name}"),
                    is_train=gt_is_train,
                    normals=gt_normals,
                )

    print(f"✅ Saved {num_frames} full plant frames")


def save_interpolation_from_organ_models(
    organ_models,
    organ_sequences,
    organ_normalizations,
    organ_leaf_tips,
    dataset,
    out_dir,
    sequence_name=None,
    timepoint_days=None,
    device="cuda",
    use_pre_finetune_models=False,
    organ_transformations_full=None,
    organ_transformation_days=None,
    organ_train_days=None,
    organ_full_day_range=None,
    use_day_based_tnorm=True,
    original_split_file=None,
    use_odeint=False,
    ode_method="dopri5",
    ode_rtol=1e-4,
    ode_atol=1e-5,
    train_integration_steps=1,
    ode_n_steps=None,
    use_pose_anchoring=True,
    extrapolation=False,
    label_field="labels",
    save_normals=True,
):
    """
    Save predictions and GT for unseen timepoints (interpolation or extrapolation).

    For interpolation experiments with stride N, this saves predictions for
    timepoints that were NOT seen during training (e.g., days 1, 3, 5, 7 for stride=2).

    When extrapolation=True, saves predictions for timepoints AFTER the last training
    day (e.g., the final 3 days held out from training).
    Unseen days are determined automatically from gaps in timepoint_days.

    Args:
        organ_models: Dict of organ_id -> model_info from OrganModelManager
        organ_sequences: Dict of organ_id -> sequence_info from OrganModelManager
        organ_normalizations: Dict of organ_id -> (min_xyz, scale) from OrganModelManager
        organ_leaf_tips: Dict of organ_id -> list of tips from OrganModelManager
        dataset: TrackPlant3DDataset (with stride applied)
        out_dir: Output directory for interpolated PLYs
        sequence_name: Sequence name for filenames (optional)
        timepoint_days: List of actual day indices from training (required)
        device: torch device
        use_pre_finetune_models: If True, use pre-finetune models for pairwise organs
        organ_transformations_full: Dict of organ_id -> full transformation stages (all days)
        organ_transformation_days: Dict of organ_id -> list of days for each transformation
        organ_train_days: Dict of organ_id -> list of training days when organ present
        organ_full_day_range: Dict of organ_id -> (first_day, last_day) from full unstrided data
        original_split_file: Path to original (unstrided) split JSON for is_train labels
        train_integration_steps: Sub-steps per frame interval for Euler integration.
            Used in the fractional part: floor(frac * N) full sub-steps + Euler remainder.
            Ignored when use_odeint=True (ode_n_steps controls fixed-step ODE stepping).
        save_normals: Estimate normals for the predicted clouds. They are oriented
            against that day's ground truth: only their sign comes from it, and no
            metric depends on that sign (see `estimate_normals`).
    """
    import torch

    from growfields.models.integration import (
        _integrate_segment,
        _integrate_time_span,
        denormalise,
        make_t_norm_helpers,
    )
    from growfields.models.integration import _FIXED_STEP_METHODS

    # Import LeafSequencesDataset utilities for inverse transformations
    try:
        from plant_shape_analysis import LeafSequencesDataset

        has_transform_utils = True
    except ImportError:
        has_transform_utils = False
        if organ_transformations_full:
            print(
                "Warning: plant_shape_analysis not available, cannot apply inverse transformations"
            )

    _odeint_fixed = use_odeint and ode_method in _FIXED_STEP_METHODS

    # Create directories for full plant predictions
    pred_dir = os.path.join(out_dir, "predicted")
    gt_dir = os.path.join(out_dir, "ground_truth")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Get base directory for organs (sibling to full_plant)
    # out_dir is typically .../full_plant/interpolation, so we go up to get the base
    base_dir = os.path.dirname(
        os.path.dirname(out_dir)
    )  # Go up from full_plant/interpolation
    organs_base_dir = os.path.join(base_dir, "organs")

    # Get the training days from timepoint_days
    if timepoint_days is None:
        print("  Warning: timepoint_days not provided, cannot determine unseen days")
        return

    if extrapolation and organ_train_days:
        # For extrapolation the dataset carries all timepoints, so timepoint_days includes
        # the held-out days. Use organ_train_days (already filtered by
        # models.organ_ablations.apply_holdout_last_n)
        # to get the true set of days the model was trained on.
        train_days = set()
        for days in organ_train_days.values():
            train_days.update(days)
    else:
        train_days = set(timepoint_days)

    # Load full GT data from the plant dataset (without stride)
    # We access this through the leaf_dataset which has all timepoints
    leaf_dataset = dataset.leaf_dataset
    plant_dataset = leaf_dataset.plant_dataset

    # Get the full plant sequence data
    full_plant_sequence = plant_dataset.get_timeseries_by_sequence_name(
        dataset.sequence_name
    )
    if full_plant_sequence is None:
        print(
            f"  Warning: Could not load full plant sequence for {dataset.sequence_name}"
        )
        return

    # Build a mapping from day to full plant timepoint data
    day_to_plant_tp = {}
    for tp in full_plant_sequence["timepoints"]:
        day_to_plant_tp[tp["day"]] = tp

    # Determine unseen days: days with GT data that were not used for training
    first_train_day = min(train_days)
    last_train_day = max(train_days)
    if extrapolation:
        # Days strictly after the last training day
        unseen_days = sorted(
            d for d in day_to_plant_tp if d not in train_days and d > last_train_day
        )
    else:
        # Days strictly within the training range (interpolation requires surrounding frames)
        unseen_days = sorted(
            d
            for d in day_to_plant_tp
            if d not in train_days and first_train_day < d < last_train_day
        )

    if not unseen_days:
        mode_label = "extrapolate" if extrapolation else "interpolate"
        print(f"  No unseen days to {mode_label}")
        return

    print(f"  Training days: {sorted(train_days)}")
    print(f"  Available GT days: {sorted(day_to_plant_tp.keys())}")
    print(f"  Unseen days for interpolation: {unseen_days}")

    # Get leaf sequences for this plant (with full transformations)
    leaf_sequences = leaf_dataset.get_timeseries_by_plant_sequence(
        dataset.sequence_name
    )
    day_to_leaf_data = {}  # day -> {leaf_id: leaf_tp_data}
    for leaf_seq in leaf_sequences:
        leaf_id = leaf_seq["leaf_id"]
        for ltp in leaf_seq["timepoints"]:
            day = ltp["day"]
            if day not in day_to_leaf_data:
                day_to_leaf_data[day] = {}
            day_to_leaf_data[day][leaf_id] = ltp

    # Pre-compute inverse transformation matrices for all organs and all days
    organ_inverse_matrices_full = {}
    if organ_transformations_full and has_transform_utils:
        for organ_id, transformation_stages in organ_transformations_full.items():
            composed = LeafSequencesDataset.compose_transformations(
                transformation_stages
            )
            organ_days = organ_transformation_days.get(organ_id, [])
            # Build day -> inverse matrix mapping
            organ_inverse_matrices_full[organ_id] = {}
            for comp, day in zip(composed, organ_days):
                organ_inverse_matrices_full[organ_id][day] = (
                    LeafSequencesDataset.invert_transformation(comp["composed_matrix"])
                )
        print(
            f"  Pre-computed inverse transformations for {len(organ_inverse_matrices_full)} organs"
        )

    # Build day -> split info mapping from original (unstrided) split file
    day_to_split_info = {}
    if original_split_file:
        import json as _json

        with open(original_split_file) as f:
            original_split_data = _json.load(f)
        # Map each timepoint's split to its day
        for i, tp in enumerate(full_plant_sequence["timepoints"]):
            if i < len(original_split_data["timepoints"]):
                day_to_split_info[tp["day"]] = original_split_data["timepoints"][i]
        print(f"  Loaded original split info for {len(day_to_split_info)} timepoints")

    saved_count = 0

    # Process each unseen day
    for unseen_day in unseen_days:
        # Skip if we don't have GT data for this day
        if unseen_day not in day_to_plant_tp:
            print(f"    Skipping day {unseen_day}: no GT data available")
            continue

        # Find the surrounding training days
        prev_train_day = max(d for d in train_days if d < unseen_day)
        if extrapolation:
            # No next training day exists; use last training day as anchor
            next_train_day = None
            interp_factor = None
        else:
            next_train_day = min(d for d in train_days if d > unseen_day)
            interp_factor = (unseen_day - prev_train_day) / (
                next_train_day - prev_train_day
            )

        # Find training timepoint indices
        prev_train_idx = timepoint_days.index(prev_train_day)
        next_train_idx = timepoint_days.index(next_train_day) if next_train_day is not None else None

        base_name = (
            f"{sequence_name}_D{unseen_day:02d}.ply"
            if sequence_name
            else f"frame_D{unseen_day:02d}.ply"
        )

        all_pred_points = []
        all_pred_labels = []
        all_pred_is_train = []
        all_pred_tips = []
        all_pred_tip_labels = []
        all_gt_tips = []
        all_gt_tip_labels = []

        # Per-organ storage for individual saving
        organ_predictions = (
            {}
        )  # organ_id -> {"points": ..., "normals": ..., "tip": ..., "is_train": ...}
        organ_ground_truths = (
            {}
        )  # organ_id -> {"points": ..., "normals": ..., "tip": ..., "is_train": ...}

        # Get GT plant data for this day
        plant_tp = day_to_plant_tp[unseen_day]

        # --- Full plant GT: use plant_tp["points"] directly ---
        # These come from plant_dataset which stores points in plant space already
        # (no inverse transforms needed, unlike dataset.train_pts/test_pts which
        # go through the leaf alignment pipeline)
        full_gt_points = plant_tp["points"]
        full_gt_labels = np.array(plant_tp.get(label_field, plant_tp["labels"]))
        full_gt_normals = plant_tp.get("normals")

        # Build is_train mask for GT from original split file
        gt_is_train = None
        if unseen_day in day_to_split_info:
            split_info = day_to_split_info[unseen_day]
            gt_is_train = np.full(len(full_gt_points), -1, dtype=np.int32)
            train_indices = split_info.get("train_indices", [])
            test_indices = split_info.get("test_indices", [])
            if train_indices:
                gt_is_train[np.array(train_indices)] = 1
            if test_indices:
                gt_is_train[np.array(test_indices)] = 0

        # Collect GT tips from leaf data
        for gt_organ_id, leaf_data in day_to_leaf_data.get(unseen_day, {}).items():
            gt_tip_organ = leaf_data.get("leaf_tip")
            if gt_tip_organ is not None:
                # Transform tip to plant space
                if gt_organ_id in organ_inverse_matrices_full:
                    if unseen_day in organ_inverse_matrices_full[gt_organ_id]:
                        inv_matrix = organ_inverse_matrices_full[gt_organ_id][
                            unseen_day
                        ]
                        gt_tip_organ = LeafSequencesDataset.apply_transformation(
                            np.array(gt_tip_organ).reshape(1, -1),
                            inv_matrix,
                        )[0]
                all_gt_tips.append(gt_tip_organ)
                all_gt_tip_labels.append(gt_organ_id)

        # --- Per-organ GT: populate organ_ground_truths in aligned space (no inverse transform) ---
        for gt_organ_id, leaf_data in day_to_leaf_data.get(unseen_day, {}).items():
            # Derive per-organ is_train from plant-level split
            organ_is_train_gt = None
            if gt_is_train is not None:
                organ_mask = full_gt_labels == gt_organ_id
                organ_is_train_gt = gt_is_train[organ_mask]

            organ_ground_truths[gt_organ_id] = {
                "points": leaf_data["points"],  # Already in aligned space
                "normals": leaf_data.get("normals"),
                "tip": leaf_data.get("leaf_tip"),
                "is_train": organ_is_train_gt,
            }

        # Also store stem GT for per-organ saving (stem is in plant space = aligned space)
        stem_mask = np.array(plant_tp.get(label_field, plant_tp["labels"])) == 0
        if np.any(stem_mask):
            stem_is_train = gt_is_train[stem_mask] if gt_is_train is not None else None
            organ_ground_truths[0] = {
                "points": plant_tp["points"][stem_mask],
                "normals": (
                    plant_tp["normals"][stem_mask]
                    if plant_tp.get("normals") is not None
                    else None
                ),
                "is_train": stem_is_train,
            }

        # Process each organ
        for organ_id in sorted(organ_models.keys()):
            organ_info = organ_models[organ_id]
            config = organ_info["config"]
            organ_seq = organ_sequences[organ_id]

            # Get valid timepoint indices for this organ (in training data)
            valid_timepoint_indices = [
                i for i, present in enumerate(organ_seq["presence_train"]) if present
            ]

            # Check if organ exists at the required training timepoints
            required_present = [prev_train_idx]
            if not extrapolation and next_train_idx is not None:
                required_present.append(next_train_idx)
            if any(idx not in valid_timepoint_indices for idx in required_present):
                # Can't interpolate, but if GT exists at this day, use it as
                # prediction in the full plant PLY (first appearance / no pair)
                if organ_id in organ_ground_truths:
                    gt_as_pred = organ_ground_truths[organ_id]
                    # GT-as-pred points need inverse transform to plant space
                    gt_pred_points = gt_as_pred["points"].copy()
                    if organ_id > 0 and organ_id in organ_inverse_matrices_full:
                        if unseen_day in organ_inverse_matrices_full[organ_id]:
                            inv_matrix = organ_inverse_matrices_full[organ_id][
                                unseen_day
                            ]
                            gt_pred_points = LeafSequencesDataset.apply_transformation(
                                gt_pred_points, inv_matrix
                            )
                    all_pred_points.append(gt_pred_points)
                    all_pred_labels.append(
                        np.full(len(gt_pred_points), organ_id, dtype=np.int32)
                    )
                    # Derive is_train from plant-level split for this organ
                    if gt_is_train is not None:
                        organ_mask_gt = full_gt_labels == organ_id
                        all_pred_is_train.append(gt_is_train[organ_mask_gt])
                    else:
                        all_pred_is_train.append(None)
                continue

            # Get organ's local indices for prev and next training timepoints
            organ_local_prev_idx = valid_timepoint_indices.index(prev_train_idx)
            organ_local_next_idx = (
                valid_timepoint_indices.index(next_train_idx)
                if next_train_idx is not None
                else None
            )

            # Get valid clouds for this organ
            if "train_points" in organ_seq and "test_points" in organ_seq:
                valid_clouds = []
                for train_pts, test_pts in zip(
                    organ_seq["train_points"], organ_seq["test_points"]
                ):
                    if train_pts is not None and test_pts is not None:
                        valid_clouds.append(torch.cat([train_pts, test_pts], dim=0))
                    elif train_pts is not None:
                        # test_pts is None (e.g. organ has no test points at this timepoint)
                        valid_clouds.append(train_pts)
                    else:
                        valid_clouds.append(None)
                valid_clouds = [c for c in valid_clouds if c is not None]
            else:
                valid_clouds = [c for c in organ_seq["points"] if c is not None]

            if len(valid_clouds) < 2:
                continue

            # Build is_train mask from first valid training frame
            organ_is_train_mask = None
            if "train_points" in organ_seq and "test_points" in organ_seq:
                if len(valid_clouds) > 0 and len(valid_timepoint_indices) > 0:
                    first_valid_t = valid_timepoint_indices[0]
                    train_pts_0 = organ_seq["train_points"][first_valid_t]
                    test_pts_0 = organ_seq["test_points"][first_valid_t]
                    if train_pts_0 is not None and test_pts_0 is not None:
                        n_train = len(train_pts_0)
                        n_test = len(test_pts_0)
                        organ_is_train_mask = np.concatenate(
                            [
                                np.ones(n_train, dtype=np.int32),
                                np.zeros(n_test, dtype=np.int32),
                            ]
                        )
                    elif train_pts_0 is not None:
                        # All organ points landed in train split at this timepoint
                        organ_is_train_mask = np.ones(len(train_pts_0), dtype=np.int32)

            # Get normalization
            min_xyz, scale = organ_normalizations[organ_id]
            min_xyz_batch = min_xyz.unsqueeze(0).unsqueeze(0)
            scale_batch = (
                torch.tensor([scale.item()], device=device).unsqueeze(0).unsqueeze(0)
            )

            # Get leaf tips for this organ and initialize normalized tip
            # Only consider tips at training timepoints where organ has data
            gt_tips_organ = organ_leaf_tips.get(organ_id, None)
            tip_normalized = None
            tip_start_idx = None
            if gt_tips_organ is not None:
                for idx, tip in enumerate(gt_tips_organ):
                    if tip is not None:
                        tip_normalized = (tip - min_xyz) / scale * 2 - 1
                        tip_normalized = tip_normalized.unsqueeze(0)
                        tip_start_idx = idx
                        break

            # Generate prediction based on model type
            pred_tip = None
            if config["model_type"] == "pairwise":
                # For pairwise: interpolate (or extrapolate) using surrounding training frames
                if use_pre_finetune_models and "pre_finetune_models" in organ_info:
                    models = organ_info["pre_finetune_models"]
                else:
                    models = organ_info.get("trained_models", [])

                if not models:
                    continue

                # Get prediction at prev training timepoint (with tip)
                x_prev = valid_clouds[0].clone()
                tip_prev = (
                    tip_normalized.clone()
                    if (tip_normalized is not None and prev_train_idx >= tip_start_idx)
                    else None
                )
                with torch.no_grad():
                    for i in range(organ_local_prev_idx):
                        x_prev = x_prev + models[i](x_prev)
                        if (
                            tip_prev is not None
                            and valid_timepoint_indices[i + 1] > tip_start_idx
                        ):
                            tip_prev = tip_prev + models[i](tip_prev)

                if extrapolation:
                    # Extrapolation: apply last model one additional step beyond last training frame
                    last_model = models[organ_local_prev_idx] if organ_local_prev_idx < len(models) else models[-1]
                    with torch.no_grad():
                        x_interp = x_prev + last_model(x_prev)
                        if tip_prev is not None:
                            tip_interp = tip_prev + last_model(tip_prev)
                            pred_tip = (
                                denormalise(
                                    tip_interp.unsqueeze(0), min_xyz_batch, scale_batch
                                )[0]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                else:
                    # Get prediction at next training timepoint (with tip)
                    x_next = valid_clouds[0].clone()
                    tip_next = (
                        tip_normalized.clone()
                        if (tip_normalized is not None and prev_train_idx >= tip_start_idx)
                        else None
                    )
                    with torch.no_grad():
                        for i in range(organ_local_next_idx):
                            x_next = x_next + models[i](x_next)
                            if (
                                tip_next is not None
                                and valid_timepoint_indices[i + 1] > tip_start_idx
                            ):
                                tip_next = tip_next + models[i](tip_next)

                    # Linear interpolation for points
                    x_interp = x_prev + interp_factor * (x_next - x_prev)

                    # Linear interpolation for tip
                    if tip_prev is not None and tip_next is not None:
                        tip_interp = tip_prev + interp_factor * (tip_next - tip_prev)
                        pred_tip = (
                            denormalise(
                                tip_interp.unsqueeze(0), min_xyz_batch, scale_batch
                            )[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                pred = (
                    denormalise(x_interp.unsqueeze(0), min_xyz_batch, scale_batch)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

            elif config["model_type"] in _CONTINUOUS_MODEL_TYPES:
                conditioning = _organ_conditioning(
                    config, organ_info, organ_id, valid_clouds
                )
                if conditioning is None:
                    continue
                model, _cond_kw, velocity = conditioning

                # t_norm: day-based or frame-index-based
                o_days = organ_train_days[organ_id]
                first_day, last_day = organ_full_day_range[organ_id]
                day_range = last_day - first_day
                num_frames_organ = len(valid_clouds)
                get_t_norm, get_dt = make_t_norm_helpers(
                    o_days, first_day, day_range, num_frames_organ, use_day_based_tnorm)

                _ode_kw = dict(
                    use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
                    ode_method=ode_method, use_adjoint=False,
                    n_substeps=train_integration_steps, ode_n_steps=ode_n_steps,
                    **_cond_kw,
                )
                _span_kw = dict(
                    use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
                    ode_method=ode_method, use_adjoint=False,
                    **_cond_kw,
                )

                x = valid_clouds[0].clone()
                tip = (
                    tip_normalized.clone()
                    if (tip_normalized is not None and prev_train_idx >= tip_start_idx)
                    else None
                )
                with torch.no_grad():
                    # Full steps to reach prev training timepoint
                    for i in range(organ_local_prev_idx):
                        x, _ = _integrate_segment(
                            model, x, i, i + 1, get_t_norm, get_dt, **_ode_kw)
                        if (
                            tip is not None
                            and valid_timepoint_indices[i + 1] > tip_start_idx
                        ):
                            tip, _ = _integrate_segment(
                                model, tip, i, i + 1, get_t_norm, get_dt, **_ode_kw)

                    # Fractional integration using training sub-step size
                    if use_day_based_tnorm:
                        t_prev_norm = (o_days[organ_local_prev_idx] - first_day) / day_range
                        t_target_norm = (unseen_day - first_day) / day_range
                    else:
                        t_prev_norm = (
                            float(organ_local_prev_idx) / (num_frames_organ - 1)
                            if num_frames_organ > 1
                            else 0.0
                        )
                        # interp_factor is None for extrapolation; extrapolate by one full step
                        _factor = interp_factor if interp_factor is not None else 1.0
                        t_target_norm = t_prev_norm + _factor * (
                            1.0 / (num_frames_organ - 1) if num_frames_organ > 1 else 0.0
                        )

                    warp_tip = (
                        tip is not None
                        and tip_start_idx is not None
                        and organ_local_prev_idx >= tip_start_idx
                    )

                    if use_odeint and not _odeint_fixed:
                        # Adaptive: single call
                        x = _integrate_time_span(
                            model, x, t_prev_norm, t_target_norm, **_span_kw)
                        if warp_tip:
                            tip = _integrate_time_span(
                                model, tip, t_prev_norm, t_target_norm, **_span_kw)
                    elif use_odeint:
                        # Fixed-step: maintain training step size;
                        # dense output handles fractional remainder
                        dt_sub = get_dt(organ_local_prev_idx) / max(ode_n_steps or 1, 1)
                        n_floor = max(int((t_target_norm - t_prev_norm) / dt_sub), 1)
                        x = _integrate_time_span(
                            model, x, t_prev_norm, t_target_norm,
                            ode_n_steps=n_floor, **_span_kw)
                        if warp_tip:
                            tip = _integrate_time_span(
                                model, tip, t_prev_norm, t_target_norm,
                                ode_n_steps=n_floor, **_span_kw)
                    else:
                        # Euler: n_floor sub-steps + Euler remainder
                        dt_sub = get_dt(organ_local_prev_idx) / max(train_integration_steps, 1)
                        span = t_target_norm - t_prev_norm
                        n_floor = int(span / dt_sub) if dt_sub > 0 else 0
                        t_step_end = t_prev_norm + n_floor * dt_sub
                        if n_floor > 0:
                            x = _integrate_time_span(
                                model, x, t_prev_norm, t_step_end,
                                num_substeps=n_floor, **_span_kw)
                            if warp_tip:
                                tip = _integrate_time_span(
                                    model, tip, t_prev_norm, t_step_end,
                                    num_substeps=n_floor, **_span_kw)
                        dt_rem = t_target_norm - t_step_end
                        if dt_rem > 1e-9:
                            x = x + velocity(x, t_step_end) * dt_rem
                            if warp_tip:
                                tip = tip + velocity(tip, t_step_end) * dt_rem

                if tip is not None:
                    pred_tip = (
                        denormalise(tip.unsqueeze(0), min_xyz_batch, scale_batch)[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                pred = (
                    denormalise(x.unsqueeze(0), min_xyz_batch, scale_batch)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                continue

            # Store per-organ prediction in aligned space (before inverse transform)
            organ_predictions[organ_id] = {
                "points": pred.copy(),
                "normals": None,  # Will be estimated at per-organ save time
                "tip": pred_tip.copy() if pred_tip is not None else None,
                "is_train": organ_is_train_mask,
            }

            # Apply inverse transformation for full plant (plant space).
            # use_pose_anchoring=True: use exact per-day matrix for this unseen day.
            # use_pose_anchoring=False: SLERP between the two surrounding training-day
            #   matrices (no eval-day transform data used).
            if organ_id > 0 and organ_id in organ_inverse_matrices_full:
                inv_matrix = None
                day_matrices = organ_inverse_matrices_full[organ_id]
                if use_pose_anchoring:
                    if unseen_day in day_matrices:
                        inv_matrix = day_matrices[unseen_day]
                else:
                    train_days = sorted(organ_train_days.get(organ_id, []))
                    lo = max((d for d in train_days if d <= unseen_day), default=None)
                    hi = min((d for d in train_days if d >= unseen_day), default=None)
                    if (lo is not None and hi is not None and lo != hi
                            and lo in day_matrices and hi in day_matrices):
                        f = (unseen_day - lo) / (hi - lo)
                        inv_matrix = interpolate_rigid_transform(day_matrices[lo], day_matrices[hi], f)
                    elif lo is not None and lo in day_matrices:
                        inv_matrix = day_matrices[lo]
                    elif hi is not None and hi in day_matrices:
                        inv_matrix = day_matrices[hi]
                if inv_matrix is not None:
                    pred = LeafSequencesDataset.apply_transformation(pred, inv_matrix)
                    if pred_tip is not None:
                        pred_tip = LeafSequencesDataset.apply_transformation(
                            pred_tip.reshape(1, -1), inv_matrix
                        )[0]

            all_pred_points.append(pred)
            all_pred_labels.append(np.full(len(pred), organ_id, dtype=np.int32))
            all_pred_is_train.append(organ_is_train_mask)
            if pred_tip is not None:
                all_pred_tips.append(pred_tip)
                all_pred_tip_labels.append(organ_id)

        # Save full plant predictions with normals, tips, and is_train
        if all_pred_points:
            full_pred_points = np.concatenate(all_pred_points, axis=0)
            full_pred_labels = np.concatenate(all_pred_labels, axis=0)
            full_pred_tips = np.array(all_pred_tips) if all_pred_tips else None
            full_pred_tip_labels = (
                np.array(all_pred_tip_labels, dtype=np.int32)
                if all_pred_tip_labels
                else None
            )

            # Concatenate is_train masks (filter out None entries)
            valid_is_train = [m for m in all_pred_is_train if m is not None]
            full_pred_is_train = (
                np.concatenate(valid_is_train, axis=0)
                if valid_is_train and len(valid_is_train) == len(all_pred_points)
                else None
            )

            full_pred_normals = None
            if save_normals:
                ref = {}
                if full_gt_normals is not None:
                    ref = dict(
                        ref_points=full_gt_points,
                        ref_normals=full_gt_normals,
                        ref_labels=full_gt_labels,
                    )
                full_pred_normals = estimate_normals(full_pred_points, full_pred_labels, **ref)

            save_ply_with_organ_labels(
                full_pred_points,
                full_pred_labels,
                full_pred_tips,
                full_pred_tip_labels,
                os.path.join(pred_dir, f"pred_{base_name}"),
                is_train=full_pred_is_train,
                normals=full_pred_normals,
            )

        # Save full plant GT with normals, tips, and is_train
        if len(full_gt_points) > 0:
            full_gt_tips_arr = np.array(all_gt_tips) if all_gt_tips else None
            full_gt_tip_labels = (
                np.array(all_gt_tip_labels, dtype=np.int32)
                if all_gt_tip_labels
                else None
            )

            save_ply_with_organ_labels(
                full_gt_points,
                full_gt_labels,
                full_gt_tips_arr,
                full_gt_tip_labels,
                os.path.join(gt_dir, f"gt_{base_name}"),
                is_train=gt_is_train,
                normals=full_gt_normals,
            )

        # Save per-organ predictions and GT inside organs/{organ_type}/{mode}/
        mode_dir = "extrapolation" if extrapolation else "interpolation"
        for organ_id, pred_data in organ_predictions.items():
            organ_type = organ_sequences[organ_id]["type"]
            organ_pred_dir = os.path.join(
                organs_base_dir, organ_type, mode_dir, "predicted"
            )
            os.makedirs(organ_pred_dir, exist_ok=True)

            # Prepare tip for this organ
            organ_tip = pred_data.get("tip")
            organ_tip_arr = np.array([organ_tip]) if organ_tip is not None else None
            organ_tip_label = (
                np.array([organ_id], dtype=np.int32) if organ_tip is not None else None
            )

            # The organ's ground truth is in the same (aligned) space.
            pred_is_train = pred_data.get("is_train")
            organ_pred_normals = None
            if save_normals:
                gt_data = organ_ground_truths.get(organ_id, {})
                ref = {}
                if gt_data.get("normals") is not None:
                    ref = dict(ref_points=gt_data["points"], ref_normals=gt_data["normals"])
                organ_pred_normals = estimate_normals(pred_data["points"], **ref)

            # Build organ-specific filename: e.g. pred_maize_control_plant2_D02_leaf1.ply
            organ_name = organ_type.replace("_", "")
            if sequence_name:
                organ_base_name = f"{sequence_name}_D{unseen_day:02d}_{organ_name}.ply"
            else:
                organ_base_name = f"frame_D{unseen_day:02d}_{organ_name}.ply"

            # Save organ prediction (use single organ label)
            organ_labels = np.full(len(pred_data["points"]), organ_id, dtype=np.int32)
            save_ply_with_organ_labels(
                pred_data["points"],
                organ_labels,
                organ_tip_arr,
                organ_tip_label,
                os.path.join(organ_pred_dir, f"pred_{organ_base_name}"),
                is_train=pred_is_train,
                normals=organ_pred_normals,
            )

        for organ_id, gt_data in organ_ground_truths.items():
            if organ_id not in organ_sequences or organ_id not in organ_predictions:
                continue
            organ_type = organ_sequences[organ_id]["type"]
            organ_gt_dir = os.path.join(
                organs_base_dir, organ_type, mode_dir, "ground_truth"
            )
            os.makedirs(organ_gt_dir, exist_ok=True)

            # Prepare tip for this organ
            organ_tip = gt_data.get("tip")
            organ_tip_arr = np.array([organ_tip]) if organ_tip is not None else None
            organ_tip_label = (
                np.array([organ_id], dtype=np.int32) if organ_tip is not None else None
            )

            # Build organ-specific filename
            organ_name = organ_type.replace("_", "")
            if sequence_name:
                organ_base_name = f"{sequence_name}_D{unseen_day:02d}_{organ_name}.ply"
            else:
                organ_base_name = f"frame_D{unseen_day:02d}_{organ_name}.ply"

            # Save organ GT (use single organ label)
            organ_labels = np.full(len(gt_data["points"]), organ_id, dtype=np.int32)
            save_ply_with_organ_labels(
                gt_data["points"],
                organ_labels,
                organ_tip_arr,
                organ_tip_label,
                os.path.join(organ_gt_dir, f"gt_{organ_base_name}"),
                is_train=gt_data.get("is_train"),
                normals=gt_data["normals"],
            )

        saved_count += 1

    print(f"✅ Saved {saved_count} interpolated frames for unseen days: {unseen_days}")


def load_ply(path):
    """Read a prediction or ground-truth PLY into a dict of numpy arrays.

    Returns `points` always, plus whichever of `organ_label`, `is_train`,
    `is_leaf_tip`, `tip_organ_label` and `normals` the file carries -- so absent
    fields are missing keys, and `.get(...)` yields None for them. This is the
    reader for the files `save_ply_with_labels` writes; it was duplicated as two
    positional-tuple variants in the metrics scripts and a third dict variant in
    the tests, which is how one of them came to read `tip_organ_label` and the
    other did not.
    """
    pcd = o3d.t.io.read_point_cloud(str(path))
    result = {"points": pcd.point.positions.numpy()}

    for attr in ("organ_label", "is_train", "is_leaf_tip", "tip_organ_label"):
        if attr in pcd.point:
            result[attr] = pcd.point[attr].numpy().flatten()

    if "normals" in pcd.point:
        result["normals"] = pcd.point["normals"].numpy()

    return result
