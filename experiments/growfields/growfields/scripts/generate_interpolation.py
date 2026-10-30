#!/usr/bin/env python3
"""
Generate dense temporal interpolation from trained per-organ models.

Loads trained models from a batch results directory and generates full-plant
PLY files at intermediate timepoints between training frames.

Usage:
    # Shared continuous model
    python scripts/generate_interpolation.py \
        --results_dir results/fullplant_per_organ_batch/20260303_004424_fullplant_batch_per_organ_shared_continuous_test_aligned \
        --n_interp 5

    # Encoder-decoder model
    python scripts/generate_interpolation.py \
        --results_dir results/fullplant_per_organ_batch/20260302_040408_fullplant_batch_per_organ_encoder_decoder_test_aligned \
        --n_interp 5

    # Specific sequences only
    python scripts/generate_interpolation.py \
        --results_dir results/fullplant_per_organ_batch/20260303_004424_... \
        --sequences maize_control_plant2 tobacco_control_plant1 \
        --n_interp 10
"""

import argparse
import json
import os
import re

import numpy as np
import torch

from growfields.datasets.trackplant3d import TrackPlant3DDataset
from growfields.models.encoder_decoder import EncoderDecoderModel
from growfields.models.organ_model_manager import OrganModelManager
from growfields.models.integration import (
    _integrate_segment,
    _integrate_time_span,
    denormalise,
    make_t_norm_helpers,
)
from growfields.models.siren import ContinuousSiren, PairwiseSiren, SharedContinuousSiren
from growfields.models.integration import _FIXED_STEP_METHODS
from growfields.utils.io_utils import estimate_normals, save_ply_with_organ_labels
from growfields.utils.transforms import interpolate_rigid_transform
from growfields.pipeline.splits import resolve_point_splits_dir


def load_manager_with_model(
    experiment_config, sequence_name, sequence_dir, device="cuda",
    run_dir=None, split_dir=None,
):
    """Recreate OrganModelManager and load trained model weights for a sequence.

    Args:
        experiment_config: Loaded experiment_config.json dict.
        sequence_name: Name of the sequence (e.g. "maize_control_plant2").
        sequence_dir: Path to the per-run sequence directory
            (e.g. run_dir/sequences/seq/run_N or run_dir/sequences/seq).
        device: Torch device string.
        run_dir: Root experiment directory (where experiment_config.json lives).
            Used to locate strided_splits/ reliably, regardless of single/multi-run layout.
            If None, falls back to the old heuristic (one level above sequence_dir).
        split_dir: Optional explicit directory containing <seq>_split.json files.

    Returns:
        manager: OrganModelManager with model weights loaded
    """
    cfg = experiment_config["config"]
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["default_model"]
    model_type = model_cfg["model_type"]

    # Determine training stride
    cli_args = experiment_config.get("cli_args", {})
    training_stride = cli_args.get("train_timepoint_stride", dataset_cfg.get("timepoint_stride", 1))
    if training_stride is None:  # explicit null in config means no striding (stride 1)
        training_stride = 1

    # Resolve the experiment root directory for finding split files.
    # run_dir is preferred; fall back to two-levels-up heuristic for backward compat.
    if run_dir is not None:
        exp_root = run_dir
    else:
        # Old heuristic: sequence_dir/../.. (works for single-run layout only)
        exp_root = os.path.normpath(os.path.join(os.path.dirname(sequence_dir), ".."))

    # Find split file: strided split (contains correct point + temporal splits for training)
    split_file = None
    if training_stride > 1:
        candidate = os.path.join(
            exp_root, "strided_splits",
            f"{sequence_name}_stride{training_stride}_split.json",
        )
        if os.path.isfile(candidate):
            split_file = candidate
            print(f"  Using strided split: {candidate}")

    # Fall back to original point-level split file
    if split_file is None:
        # 1. Explicit split_dir argument
        if split_dir:
            candidate = os.path.join(split_dir, f"{sequence_name}_split.json")
            if os.path.isfile(candidate):
                split_file = candidate
                print(f"  Using point split (split_dir): {candidate}")
        # 2. points_split_dir from experiment cli_args
        if split_file is None:
            points_split_dir = resolve_point_splits_dir(
                cli_args.get("points_split_dir")
            )
            candidate = os.path.join(points_split_dir, f"{sequence_name}_split.json")
            if os.path.isfile(candidate):
                split_file = candidate
                print(f"  Using point split (cli_args): {candidate}")

    if split_file is None:
        print(f"  ⚠️  No split file found for {sequence_name} — using sequential train_ratio split")
    dataset = TrackPlant3DDataset(
        dataset_path=dataset_cfg["dataset_path"],
        sequence_name=sequence_name,
        version=dataset_cfg.get("version", "v2"),
        mode="plant",
        train_ratio=dataset_cfg.get("train_ratio", 1.0),
        split_file=split_file,
        aligned_leaves=dataset_cfg.get("aligned_leaves", True),
        normalize=False,
        device=device,
        timepoint_stride=training_stride,
    )

    # Create manager (this extracts organs, normalizes, etc.)
    organ_configs = {}
    if "per_organ_training" in cfg and "organ_configs" in cfg["per_organ_training"]:
        organ_configs = cfg["per_organ_training"]["organ_configs"]

    manager = OrganModelManager(
        dataset=dataset,
        default_config=model_cfg,
        organ_specific_configs=organ_configs,
        device=device,
    )

    organs_dir = os.path.join(sequence_dir, "organs")
    organ_subdirs = sorted(os.listdir(organs_dir))

    if model_type == "pairwise":
        _load_pairwise(manager, model_cfg, organs_dir, organ_subdirs, device)
    elif model_type == "continuous":
        _load_continuous(manager, model_cfg, organs_dir, organ_subdirs, device)
    elif model_type == "shared_continuous":
        _load_shared_continuous(manager, model_cfg, sequence_dir, device)
    elif model_type == "encoder_decoder_continuous":
        _load_encoder_decoder(manager, model_cfg, sequence_dir, device)
    else:
        raise ValueError(f"Unsupported model_type for loading: {model_type}")

    return manager

def _load_pairwise(manager, model_cfg, organs_dir, organ_subdirs, device):
    """Load pairwise model weights for each organ."""
    config = model_cfg.copy()

    for organ_id in sorted(manager.organ_sequences.keys()):
        organ_type = "stem" if organ_id == 0 else f"leaf_{organ_id}"
        organ_subdir = None
        for d in organ_subdirs:
            if d == organ_type:
                organ_subdir = d
                break
        if organ_subdir is None:
            continue

        models_dir = os.path.join(organs_dir, organ_subdir, "models")
        if not os.path.isdir(models_dir):
            continue

        # Find all pairwise model files and sort them by numeric start day
        # (alphabetical sort is wrong when day numbers mix single and double digits)
        def _sort_key(fn):
            m = re.match(r"model_(\d+)_to_(\d+)\.pth", fn)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        
        model_files = sorted(
            (f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".pth")),
            key=_sort_key,
        )
        if not model_files:
            continue

        trained_models = []
        for mf in model_files:
            model = PairwiseSiren(
                h_f=model_cfg.get("hidden_features", 256),
                h_layers=model_cfg.get("hidden_layers", 3),
                omega_0=model_cfg.get("omega_0", 30.0),
            ).to(device)
            state_dict = torch.load(os.path.join(models_dir, mf), map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            trained_models.append(model)

        manager.organ_models[organ_id] = {
            "model": None,
            "config": config,
            "type": "stem" if organ_id == 0 else "leaf",
            "trained_models": trained_models,
        }

    print(f"  Loaded pairwise models for {len(manager.organ_models)} organs")

def _load_continuous(manager, model_cfg, organs_dir, organ_subdirs, device):
    """Load per-organ continuous model weights."""
    config = model_cfg.copy()

    for organ_id in sorted(manager.organ_sequences.keys()):
        organ_type = "stem" if organ_id == 0 else f"leaf_{organ_id}"
        organ_subdir = None
        for d in organ_subdirs:
            if d == organ_type:
                organ_subdir = d
                break
        if organ_subdir is None:
            continue

        model_path = os.path.join(organs_dir, organ_subdir, "models", "model_continuous.pth")
        if not os.path.isfile(model_path):
            continue

        model = ContinuousSiren(
            h_f=model_cfg.get("hidden_features", 256),
            h_layers=model_cfg.get("hidden_layers", 5),
            first_omega_0=model_cfg.get("first_omega_0", 30.0),
            hidden_omega_0=model_cfg.get("hidden_omega_0", 30.0),
        ).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        manager.organ_models[organ_id] = {
            "model": None,
            "config": config,
            "type": "stem" if organ_id == 0 else "leaf",
            "trained_model": model,
        }

    print(f"  Loaded continuous models for {len(manager.organ_models)} organs")

def _find_global_model(run_dir, filename):
    """Path of a shared or encoder-decoder checkpoint for one run.

    Saved once under ``<run_dir>/models/``; runs from before that layout carry
    an identical copy under every ``organs/*/models/``, so take the first.
    """
    candidates = [os.path.join(run_dir, "models")]
    organs_dir = os.path.join(run_dir, "organs")
    if os.path.isdir(organs_dir):
        candidates += [
            os.path.join(organs_dir, d, "models") for d in sorted(os.listdir(organs_dir))
        ]
    for models_dir in candidates:
        path = os.path.join(models_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No {filename} in {run_dir}/models/ or {organs_dir}/*/models/"
    )

def _load_shared_continuous(manager, model_cfg, run_dir, device):
    """Load shared continuous model weights."""
    model_path = _find_global_model(run_dir, "model_shared_continuous.pth")
    remap_path = os.path.join(os.path.dirname(model_path), "organ_id_remap.json")
    if not os.path.isfile(remap_path):
        remap_path = None

    # Build organ_id_remap
    organ_ids = sorted(manager.organ_sequences.keys())
    dense_ids = list(range(len(organ_ids)))
    if organ_ids != dense_ids:
        remap = {orig: dense for dense, orig in enumerate(organ_ids)}
    else:
        remap = None

    # If saved remap exists, use it
    if remap_path:
        with open(remap_path) as f:
            saved_remap = json.load(f)
        remap = {int(k): v for k, v in saved_remap.items()}

    # Infer num_organs from checkpoint to handle organs present in data but not trained
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    num_organs = state_dict["organ_embedding.weight"].shape[0]
    if model_cfg.get("velocity_param", "free") == "monotone":
        from growfields.models.siren import MonotoneSharedContinuousSiren
        model_cls = MonotoneSharedContinuousSiren  # organ_base buffer comes from the state dict
    else:
        model_cls = SharedContinuousSiren
    model = model_cls(
        num_organs=num_organs,
        embed_dim=model_cfg.get("embed_dim", 16),
        h_f=model_cfg.get("hidden_features", 256),
        h_layers=model_cfg.get("hidden_layers", 5),
        first_omega_0=model_cfg.get("first_omega_0", 30.0),
        hidden_omega_0=model_cfg.get("hidden_omega_0", 30.0),
        conditioning=model_cfg.get("conditioning", "concat"),
        embed_init=model_cfg.get("embed_init", "normal"),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    if remap is not None:
        model.organ_id_remap = remap

    manager.shared_model = model

    # Populate organ_models entries
    config = model_cfg.copy()
    for organ_id in manager.organ_sequences.keys():
        organ_type = "stem" if organ_id == 0 else "leaf"
        manager.organ_models[organ_id] = {
            "model": None,
            "config": config,
            "type": organ_type,
            "trained_model": model,
        }

    print(f"  Loaded shared_continuous model from {model_path}")
    print(f"  {num_organs} organs, remap={remap}")

def _load_encoder_decoder(manager, model_cfg, run_dir, device):
    """Load encoder-decoder model weights."""
    model_path = _find_global_model(run_dir, "model_encoder_decoder.pth")

    model = EncoderDecoderModel(
        latent_dim=model_cfg.get("latent_dim", 128),
        encoder_hidden_dim=model_cfg.get("encoder_hidden_dim", 128),
        decoder_dims=model_cfg.get("decoder_dims", [256, 256, 256, 256, 256]),
        decoder_skip_in=model_cfg.get("decoder_skip_in", [3]),
        zero_init_last=model_cfg.get("zero_init_last", False),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    manager.encoder_decoder_model = model

    config = model_cfg.copy()
    for organ_id in manager.organ_sequences.keys():
        organ_type = "stem" if organ_id == 0 else "leaf"
        manager.organ_models[organ_id] = {
            "model": None,
            "config": config,
            "type": organ_type,
            "trained_model": model,
        }

    print(f"  Loaded encoder_decoder model from {model_path}")

def generate_dense_interpolation(
    manager,
    out_dir,
    n_interp=5,
    device="cuda",
    tracking_cmap="turbo",
    use_pose_anchoring=True,
    save_canonical=False,
    save_normals=True,
    split_dir=None,
):
    """Generate full-plant PLYs at dense intermediate timepoints.

    For each pair of consecutive training timepoints, generates n_interp
    intermediate frames by integrating the velocity field.

    Args:
        manager: OrganModelManager with loaded models
        out_dir: Output directory for PLY files
        n_interp: Number of intermediate frames between each pair of training frames
        device: torch device
        save_normals: Estimate normals for each frame. A training-day frame is
            oriented against that day's ground truth, every other frame against
            the frame before it, so no evaluation-day data is used.
        split_dir: Directory holding the unstrided `<seq>_split.json`, for the
            ground-truth frames' is_train (default: the splits shipped with the repo).
    """
    try:
        from plant_shape_analysis import LeafSequencesDataset
        has_transform_utils = True
    except ImportError:
        has_transform_utils = False

    pred_dir = os.path.join(out_dir, "predicted")
    os.makedirs(pred_dir, exist_ok=True)
    # Optional: also save the pre-alignment (organ-canonical) assembly, i.e. each
    # organ's denormalised prediction BEFORE the per-day rigid alignment matrix is
    # applied. Lets the velocity analysis separate the model's intrinsic growth
    # velocity from the time-varying pose-anchoring motion.
    canonical_dir = os.path.join(out_dir, "predicted_canonical")
    if save_canonical:
        os.makedirs(canonical_dir, exist_ok=True)

    organ_models = manager.organ_models
    organ_sequences = manager.organ_sequences
    organ_normalizations = manager.organ_normalizations
    organ_train_days = manager.organ_train_days
    organ_full_day_range = manager.organ_full_day_range if hasattr(manager, 'organ_full_day_range') else {}
    organ_leaf_tips = manager.organ_leaf_tips
    timepoint_days = manager.timepoint_days
    use_day_based_tnorm = manager.use_day_based_tnorm
    model_type = manager.default_config["model_type"]
    train_integration_steps = manager.default_config.get("train_integration_steps", 1)
    use_odeint  = manager.default_config.get("use_odeint", False)
    ode_method  = manager.default_config.get("ode_method", "dopri5")
    ode_rtol    = manager.default_config.get("ode_rtol", 1e-4)
    ode_atol    = manager.default_config.get("ode_atol", 1e-5)
    ode_n_steps = manager.default_config.get("ode_n_steps", None)
    _odeint_fixed = use_odeint and ode_method in _FIXED_STEP_METHODS
    print(f"Generating interpolation with model_type={model_type}, train_integration_steps={train_integration_steps}, "
          f"use_odeint={use_odeint}, ode_method={ode_method}")

    # Use global day range for shared model if available
    full_day_range = dict(organ_full_day_range)
    if manager.global_day_range is not None:
        for oid in full_day_range:
            full_day_range[oid] = manager.global_day_range

    # Pre-compute inverse transformations (training-day list, used as fallback)
    organ_inverse_matrices = {}
    if manager.organ_transformations and has_transform_utils:
        for organ_id, transformation_stages in manager.organ_transformations.items():
            composed = LeafSequencesDataset.compose_transformations(transformation_stages)
            organ_inverse_matrices[organ_id] = [
                LeafSequencesDataset.invert_transformation(comp["composed_matrix"])
                for comp in composed
            ]

    # Pre-compute inverse transformations for all available days (day-keyed dict).
    # Used in preference to the training-day list so that days with a ground-truth
    # transform get the exact measured matrix rather than a linear interpolation.
    organ_inverse_matrices_full = {}
    if manager.organ_transformations_full and has_transform_utils:
        for organ_id, transformation_stages in manager.organ_transformations_full.items():
            composed = LeafSequencesDataset.compose_transformations(transformation_stages)
            organ_days = manager.organ_transformation_days.get(organ_id, [])
            organ_inverse_matrices_full[organ_id] = {
                day: LeafSequencesDataset.invert_transformation(comp["composed_matrix"])
                for comp, day in zip(composed, organ_days)
            }

    # Get number of training timepoints
    first_organ_id = list(organ_models.keys())[0]
    first_organ_seq = organ_sequences[first_organ_id]
    if "train_points" in first_organ_seq and "test_points" in first_organ_seq:
        valid_clouds_first = [
            torch.cat([tr, te], dim=0)
            if tr is not None and te is not None else None
            for tr, te in zip(first_organ_seq["train_points"], first_organ_seq["test_points"])
        ]
        valid_clouds_first = [c for c in valid_clouds_first if c is not None]
    else:
        valid_clouds_first = [c for c in first_organ_seq["points"] if c is not None]

    num_training_frames = len(valid_clouds_first)
    sequence_name = manager.dataset.sequence_name if hasattr(manager.dataset, 'sequence_name') else "plant"

    print(f"  Sequence: {sequence_name}")
    print(f"  Training frames: {num_training_frames}, days: {timepoint_days}")
    print(f"  Generating {n_interp} intermediate frames between each pair")
    print(f"  Total output frames: {(num_training_frames - 1) * (n_interp + 1) + 1}")

    # Build interpolation schedule: list of (target_day_float, pair_idx, frac)
    # For each pair of training timepoints, generate n_interp intermediate + the endpoint
    interp_schedule = []
    for pair_idx in range(num_training_frames - 1):
        day_start = timepoint_days[pair_idx]
        day_end = timepoint_days[pair_idx + 1]
        # Include start frame (only for first pair to avoid duplicates)
        if pair_idx == 0:
            interp_schedule.append((day_start, pair_idx, 0.0))
        for sub in range(1, n_interp + 1):
            frac = sub / (n_interp + 1)
            day_float = day_start + frac * (day_end - day_start)
            interp_schedule.append((day_float, pair_idx, frac))
        # Include endpoint
        interp_schedule.append((day_end, pair_idx, 1.0))

    # XYZ tracking colors: global bbox from first assembled frame,
    # each point colored once at first appearance, carried forward unchanged.
    xyz_bbox_min = None
    xyz_bbox_max = None
    organ_xyz_colors = {}

    # Normals reference: ground truth on training days, else the previous frame.
    day_to_plant_tp = {}
    plant_seq = manager.dataset.plant_dataset.get_timeseries_by_sequence_name(sequence_name)
    if plant_seq is not None:
        day_to_plant_tp = {tp["day"]: tp for tp in plant_seq["timepoints"]}
    normals_ref = {}

    saved_count = 0
    for frame_global_idx, (target_day, pair_idx, frac) in enumerate(interp_schedule):
        all_pred_points = []
        all_pred_points_canonical = []   # pre-alignment (organ-canonical) points
        all_pred_labels = []
        all_pred_is_train = []
        all_pred_tips = []
        all_pred_tip_labels = []
        all_pred_colors = []

        for organ_id in sorted(organ_models.keys()):
            organ_info = organ_models[organ_id]
            config = organ_info["config"]
            organ_seq = organ_sequences[organ_id]

            valid_timepoint_indices = [
                i for i, present in enumerate(organ_seq["presence_train"]) if present
            ]

            # Check organ is present at both endpoints of this pair
            if pair_idx not in valid_timepoint_indices or pair_idx + 1 not in valid_timepoint_indices:
                # If it's an exact training frame and organ is present, use GT-as-pred fallback
                if frac == 0.0 and pair_idx in valid_timepoint_indices:
                    pass  # will be handled below
                elif frac == 1.0 and (pair_idx + 1) in valid_timepoint_indices:
                    pass
                else:
                    continue

            # Get valid clouds and build is_train mask
            organ_is_train_mask = None
            if "train_points" in organ_seq and "test_points" in organ_seq:
                valid_clouds = [
                    torch.cat([tr, te], dim=0)
                    if tr is not None and te is not None else None
                    for tr, te in zip(organ_seq["train_points"], organ_seq["test_points"])
                ]
                valid_clouds = [c for c in valid_clouds if c is not None]

                # Create is_train mask from first valid frame (predictions inherit this)
                if len(valid_clouds) > 0 and len(valid_timepoint_indices) > 0:
                    first_valid_t = valid_timepoint_indices[0]
                    train_pts_0 = organ_seq["train_points"][first_valid_t]
                    test_pts_0 = organ_seq["test_points"][first_valid_t]
                    if train_pts_0 is not None and test_pts_0 is not None:
                        n_train = len(train_pts_0)
                        n_test = len(test_pts_0)
                        organ_is_train_mask = np.concatenate([
                            np.ones(n_train, dtype=np.int32),
                            np.zeros(n_test, dtype=np.int32),
                        ])
            else:
                valid_clouds = [c for c in organ_seq["points"] if c is not None]

            if len(valid_clouds) < 2:
                continue

            organ_local_pair_start = valid_timepoint_indices.index(pair_idx) if pair_idx in valid_timepoint_indices else None
            if organ_local_pair_start is None:
                continue

            min_xyz, scale = organ_normalizations[organ_id]
            min_xyz_batch = min_xyz.unsqueeze(0).unsqueeze(0)
            scale_batch = torch.tensor([scale.item()], device=device).unsqueeze(0).unsqueeze(0)

            # Initialize leaf tip tracking
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

            o_days = organ_train_days[organ_id]
            first_day, last_day = full_day_range[organ_id]
            day_range = last_day - first_day
            num_frames_organ = len(valid_clouds)
            get_t_norm, get_dt = make_t_norm_helpers(
                o_days, first_day, day_range, num_frames_organ, use_day_based_tnorm)

            x = valid_clouds[0].clone()

            with torch.no_grad():
                if config["model_type"] == "pairwise":
                    models = organ_info.get("trained_models", [])
                    if not models or organ_local_pair_start >= len(models):
                        continue

                    # Apply models sequentially to reach pair_idx
                    for i in range(organ_local_pair_start):
                        current_global_t = valid_timepoint_indices[i + 1]
                        x = x + models[i](x)
                        if tip_normalized is not None and current_global_t > tip_start_idx:
                            tip_normalized = tip_normalized + models[i](tip_normalized)

                    # Fractional part: linearly interpolate the displacement
                    if frac > 0.0:
                        displacement = models[organ_local_pair_start](x)
                        x = x + frac * displacement
                        if tip_normalized is not None:
                            current_global_t = valid_timepoint_indices[organ_local_pair_start + 1]
                            if current_global_t > tip_start_idx:
                                tip_displacement = models[organ_local_pair_start](tip_normalized)
                                tip_normalized = tip_normalized + frac * tip_displacement

                elif config["model_type"] in ("shared_continuous", "continuous"):
                    model = organ_info.get("trained_model")
                    if model is None:
                        continue

                    is_shared = config["model_type"] == "shared_continuous"
                    _ode_kw = dict(
                        use_odeint=use_odeint, ode_method=ode_method,
                        ode_rtol=ode_rtol, ode_atol=ode_atol,
                        use_adjoint=False, n_substeps=train_integration_steps,
                        ode_n_steps=ode_n_steps,
                    )
                    if is_shared:
                        _ode_kw["organ_id"] = organ_id

                    # Integrate full steps to reach pair_idx
                    for i in range(organ_local_pair_start):
                        current_global_t = valid_timepoint_indices[i + 1]
                        x, _ = _integrate_segment(
                            model, x, i, i + 1, get_t_norm, get_dt, **_ode_kw)
                        if tip_normalized is not None and current_global_t > tip_start_idx:
                            tip_normalized, _ = _integrate_segment(
                                model, tip_normalized, i, i + 1, get_t_norm, get_dt, **_ode_kw)

                    # Now integrate the fractional part
                    if frac > 0.0:
                        if frac == 1.0:
                            x, _ = _integrate_segment(
                                model, x, organ_local_pair_start, organ_local_pair_start + 1,
                                get_t_norm, get_dt, **_ode_kw)
                            if tip_normalized is not None:
                                current_global_t = valid_timepoint_indices[organ_local_pair_start + 1]
                                if current_global_t > tip_start_idx:
                                    tip_normalized, _ = _integrate_segment(
                                        model, tip_normalized, organ_local_pair_start, organ_local_pair_start + 1,
                                        get_t_norm, get_dt, **_ode_kw)
                        else:
                            t_start_norm = get_t_norm(organ_local_pair_start)
                            if use_day_based_tnorm:
                                t_target_norm = (target_day - first_day) / day_range
                            else:
                                t_next_norm = get_t_norm(organ_local_pair_start + 1)
                                t_target_norm = t_start_norm + frac * (t_next_norm - t_start_norm)
                            _span_kw = dict(
                                use_odeint=use_odeint, ode_method=ode_method,
                                ode_rtol=ode_rtol, ode_atol=ode_atol,
                            )
                            if is_shared:
                                _span_kw["organ_id"] = organ_id

                            warp_tip = (
                                tip_normalized is not None
                                and tip_start_idx is not None
                                and valid_timepoint_indices[organ_local_pair_start] >= tip_start_idx
                            )

                            if use_odeint and not _odeint_fixed:
                                # Adaptive: single call
                                x = _integrate_time_span(
                                    model, x, t_start_norm, t_target_norm, **_span_kw)
                                if warp_tip:
                                    tip_normalized = _integrate_time_span(
                                        model, tip_normalized, t_start_norm, t_target_norm, **_span_kw)
                            elif use_odeint:
                                # Fixed-step: maintain training step size;
                                # dense output handles fractional remainder
                                dt_sub = get_dt(organ_local_pair_start) / max(ode_n_steps or 1, 1)
                                n_floor = max(int((t_target_norm - t_start_norm) / dt_sub), 1)
                                x = _integrate_time_span(
                                    model, x, t_start_norm, t_target_norm,
                                    ode_n_steps=n_floor, **_span_kw)
                                if warp_tip:
                                    tip_normalized = _integrate_time_span(
                                        model, tip_normalized, t_start_norm, t_target_norm,
                                        ode_n_steps=n_floor, **_span_kw)
                            else:
                                # Euler: n_floor sub-steps + Euler remainder
                                dt_sub = get_dt(organ_local_pair_start) / max(train_integration_steps, 1)
                                span = t_target_norm - t_start_norm
                                n_floor = int(span / dt_sub) if dt_sub > 0 else 0
                                t_step_end = t_start_norm + n_floor * dt_sub
                                if n_floor > 0:
                                    x = _integrate_time_span(
                                        model, x, t_start_norm, t_step_end,
                                        num_substeps=n_floor, **_span_kw)
                                    if warp_tip:
                                        tip_normalized = _integrate_time_span(
                                            model, tip_normalized, t_start_norm, t_step_end,
                                            num_substeps=n_floor, **_span_kw)
                                dt_rem = t_target_norm - t_step_end
                                if dt_rem > 1e-9:
                                    v = model(x, t_step_end, organ_id) if is_shared else model(x, t_step_end)
                                    x = x + v * dt_rem
                                    if warp_tip:
                                        v_tip = model(tip_normalized, t_step_end, organ_id) if is_shared else model(tip_normalized, t_step_end)
                                        tip_normalized = tip_normalized + v_tip * dt_rem

                elif config["model_type"] == "encoder_decoder_continuous":
                    model = organ_info.get("trained_model")
                    if model is None:
                        continue

                    latent_code = model.encode(valid_clouds[0])

                    _ode_kw = dict(
                        use_odeint=use_odeint, ode_method=ode_method,
                        ode_rtol=ode_rtol, ode_atol=ode_atol,
                        use_adjoint=False, n_substeps=train_integration_steps,
                        ode_n_steps=ode_n_steps, latent_code=latent_code,
                    )
                    _span_kw = dict(
                        use_odeint=use_odeint, ode_method=ode_method,
                        ode_rtol=ode_rtol, ode_atol=ode_atol,
                        latent_code=latent_code,
                    )

                    # Integrate full steps to reach pair_idx
                    for i in range(organ_local_pair_start):
                        current_global_t = valid_timepoint_indices[i + 1]
                        x, _ = _integrate_segment(model, x, i, i + 1, get_t_norm, get_dt, **_ode_kw)
                        if tip_normalized is not None and current_global_t > tip_start_idx:
                            tip_normalized, _ = _integrate_segment(
                                model, tip_normalized, i, i + 1, get_t_norm, get_dt, **_ode_kw)

                    # Fractional part
                    if frac > 0.0:
                        if frac == 1.0:
                            x, _ = _integrate_segment(
                                model, x, organ_local_pair_start, organ_local_pair_start + 1,
                                get_t_norm, get_dt, **_ode_kw)
                            if tip_normalized is not None:
                                current_global_t = valid_timepoint_indices[organ_local_pair_start + 1]
                                if current_global_t > tip_start_idx:
                                    tip_normalized, _ = _integrate_segment(
                                        model, tip_normalized, organ_local_pair_start,
                                        organ_local_pair_start + 1, get_t_norm, get_dt, **_ode_kw)
                        else:
                            t_start_norm = get_t_norm(organ_local_pair_start)
                            if use_day_based_tnorm:
                                t_target_norm = (target_day - first_day) / day_range
                            else:
                                t_next_norm = get_t_norm(organ_local_pair_start + 1)
                                t_target_norm = t_start_norm + frac * (t_next_norm - t_start_norm)
                            warp_tip = (
                                tip_normalized is not None
                                and valid_timepoint_indices[organ_local_pair_start] >= tip_start_idx
                            )

                            if use_odeint and not _odeint_fixed:
                                # Adaptive: single call
                                x = _integrate_time_span(
                                    model, x, t_start_norm, t_target_norm, **_span_kw)
                                if warp_tip:
                                    tip_normalized = _integrate_time_span(
                                        model, tip_normalized, t_start_norm, t_target_norm, **_span_kw)
                            elif use_odeint:
                                # Fixed-step: maintain training step size;
                                # dense output handles fractional remainder
                                dt_sub = get_dt(organ_local_pair_start) / max(ode_n_steps or 1, 1)
                                n_floor = max(int((t_target_norm - t_start_norm) / dt_sub), 1)
                                x = _integrate_time_span(
                                    model, x, t_start_norm, t_target_norm,
                                    ode_n_steps=n_floor, **_span_kw)
                                if warp_tip:
                                    tip_normalized = _integrate_time_span(
                                        model, tip_normalized, t_start_norm, t_target_norm,
                                        ode_n_steps=n_floor, **_span_kw)
                            else:
                                # Euler: n_floor sub-steps + Euler remainder
                                dt_sub = get_dt(organ_local_pair_start) / max(train_integration_steps, 1)
                                span = t_target_norm - t_start_norm
                                n_floor = int(span / dt_sub) if dt_sub > 0 else 0
                                t_step_end = t_start_norm + n_floor * dt_sub
                                if n_floor > 0:
                                    x = _integrate_time_span(
                                        model, x, t_start_norm, t_step_end,
                                        num_substeps=n_floor, **_span_kw)
                                    if warp_tip:
                                        tip_normalized = _integrate_time_span(
                                            model, tip_normalized, t_start_norm, t_step_end,
                                            num_substeps=n_floor, **_span_kw)
                                dt_rem = t_target_norm - t_step_end
                                if dt_rem > 1e-9:
                                    x = x + model(x, t_step_end, latent_code) * dt_rem
                                    if warp_tip:
                                        tip_normalized = tip_normalized + model(
                                            tip_normalized, t_step_end, latent_code) * dt_rem
                else:
                    continue

                pred = (
                    denormalise(x.unsqueeze(0), min_xyz_batch, scale_batch)[0]
                    .detach().cpu().numpy()
                )

                # Denormalize tip if it has been warped at least once
                pred_tip = None
                if tip_normalized is not None and tip_start_idx is not None:
                    # Tip is valid if we've integrated past its first appearance
                    global_t_at_pair_start = valid_timepoint_indices[organ_local_pair_start]
                    tip_active = global_t_at_pair_start > tip_start_idx or (
                        global_t_at_pair_start >= tip_start_idx and frac > 0.0
                    )
                    if tip_active:
                        pred_tip = (
                            denormalise(tip_normalized.unsqueeze(0), min_xyz_batch, scale_batch)[0]
                            .detach().cpu().numpy()
                        )

            # Canonical (pre-alignment) copy: denormalised prediction in the organ's
            # own frame, before any rigid alignment matrix is applied below.
            pred_canonical = pred.copy()

            # Apply inverse transformation for aligned leaves.
            # use_pose_anchoring=True: prefer exact per-day matrices from the full dataset;
            #   for sub-integer virtual days, piecewise-interpolate between the nearest
            #   available measured-day matrices (not just the training-frame endpoints).
            # use_pose_anchoring=False: always interpolate between the two surrounding
            #   training-frame matrices only (no eval-day transform data used).
            if organ_id > 0 and (
                organ_id in organ_inverse_matrices_full or organ_id in organ_inverse_matrices
            ):
                inv_matrix = None
                if use_pose_anchoring and organ_id in organ_inverse_matrices_full:
                    target_day_int = int(target_day) if target_day == int(target_day) else None
                    if (target_day_int is not None
                            and target_day_int in organ_inverse_matrices_full[organ_id]):
                        inv_matrix = organ_inverse_matrices_full[organ_id][target_day_int]
                    else:
                        # Piecewise: find nearest bracketing days in the full dict
                        sorted_days = sorted(organ_inverse_matrices_full[organ_id].keys())
                        lo = max((d for d in sorted_days if d <= target_day), default=None)
                        hi = min((d for d in sorted_days if d >= target_day), default=None)
                        if lo is not None and hi is not None and lo != hi:
                            f = (target_day - lo) / (hi - lo)
                            inv_matrix = interpolate_rigid_transform(
                                organ_inverse_matrices_full[organ_id][lo],
                                organ_inverse_matrices_full[organ_id][hi],
                                f,
                            )
                        elif lo is not None:
                            inv_matrix = organ_inverse_matrices_full[organ_id][lo]
                        elif hi is not None:
                            inv_matrix = organ_inverse_matrices_full[organ_id][hi]
                elif organ_id in organ_inverse_matrices:
                    # Pose anchoring OFF: SLERP between training-frame matrices only
                    n = len(organ_inverse_matrices[organ_id])
                    if frac == 0.0 and organ_local_pair_start < n:
                        inv_matrix = organ_inverse_matrices[organ_id][organ_local_pair_start]
                    elif frac == 1.0 and organ_local_pair_start + 1 < n:
                        inv_matrix = organ_inverse_matrices[organ_id][organ_local_pair_start + 1]
                    elif organ_local_pair_start < n and organ_local_pair_start + 1 < n:
                        inv_a = organ_inverse_matrices[organ_id][organ_local_pair_start]
                        inv_b = organ_inverse_matrices[organ_id][organ_local_pair_start + 1]
                        inv_matrix = interpolate_rigid_transform(inv_a, inv_b, frac)
                if inv_matrix is not None:
                    pred = LeafSequencesDataset.apply_transformation(pred, inv_matrix)
                    if pred_tip is not None:
                        pred_tip = LeafSequencesDataset.apply_transformation(pred_tip.reshape(1, -1), inv_matrix)[0]

            all_pred_points.append(pred)
            all_pred_points_canonical.append(pred_canonical)
            all_pred_labels.append(np.full(len(pred), organ_id, dtype=np.int32))
            if organ_is_train_mask is not None:
                all_pred_is_train.append(organ_is_train_mask)
            if pred_tip is not None:
                all_pred_tips.append(pred_tip)
                all_pred_tip_labels.append(organ_id)

            # XYZ tracking colors: store pred for deferred color assignment
            all_pred_colors.append((organ_id, len(all_pred_points) - 1))

        if not all_pred_points:
            continue

        full_pred_points = np.concatenate(all_pred_points, axis=0)
        full_pred_labels = np.concatenate(all_pred_labels, axis=0)
        full_pred_tips = np.array(all_pred_tips) if all_pred_tips else None
        full_pred_tip_labels = (
            np.array(all_pred_tip_labels, dtype=np.int32) if all_pred_tip_labels else None
        )
        full_pred_is_train = (
            np.concatenate(all_pred_is_train, axis=0) if all_pred_is_train else None
        )

        # XYZ tracking colors: global per-axis normalization from first frame,
        # each point colored once at first appearance, carried forward.
        if xyz_bbox_min is None and len(full_pred_points) > 0:
            xyz_bbox_min = full_pred_points.min(axis=0).astype(np.float64)
            xyz_bbox_max = full_pred_points.max(axis=0).astype(np.float64)

        full_pred_colors = None
        if xyz_bbox_min is not None and all_pred_colors:
            color_arrays = []
            denom = xyz_bbox_max - xyz_bbox_min
            denom = np.where(denom < 1e-8, 1.0, denom)
            for organ_id, pred_idx in all_pred_colors:
                if organ_id not in organ_xyz_colors:
                    xyz = all_pred_points[pred_idx].astype(np.float64)
                    norm_xyz = (xyz - xyz_bbox_min) / denom  # per-axis [0,1]
                    if tracking_cmap == "rgb":
                        # Triangle wave at higher frequency for richer variation
                        freq = 3.0
                        color = np.abs((norm_xyz * freq) % 2 - 1)
                        organ_xyz_colors[organ_id] = (color * 255).clip(0, 255).astype(np.uint8)
                    else:
                        # Turbo on each axis independently, then blend
                        # Different frequencies per axis + phase offsets to avoid
                        # degenerate dark/bright bands
                        r = np.abs((norm_xyz[:, 0] * 3.0) % 2 - 1)
                        g = np.abs((norm_xyz[:, 1] * 4.0 + 0.33) % 2 - 1)
                        b = np.abs((norm_xyz[:, 2] * 5.0 + 0.67) % 2 - 1)
                        color = np.stack([r, g, b], axis=1)
                        organ_xyz_colors[organ_id] = (color * 255).clip(0, 255).astype(np.uint8)
                color_arrays.append(organ_xyz_colors[organ_id])
            full_pred_colors = np.concatenate(color_arrays, axis=0)

        full_pred_normals = None
        if save_normals:
            plant_tp = day_to_plant_tp.get(target_day) if frac in (0.0, 1.0) else None
            if plant_tp is not None and plant_tp.get("normals") is not None:
                normals_ref = dict(
                    ref_points=plant_tp["points"],
                    ref_normals=plant_tp["normals"],
                    ref_labels=np.asarray(plant_tp.get(manager.label_field, plant_tp["labels"])),
                )
            full_pred_normals = estimate_normals(full_pred_points, full_pred_labels, **normals_ref)
            normals_ref = dict(
                ref_points=full_pred_points,
                ref_normals=full_pred_normals,
                ref_labels=full_pred_labels,
            )

        # Save PLY (binary format via Open3D, matching full-plant output)
        day_str = f"{target_day:.2f}"
        filename = f"{sequence_name}_D{day_str}_F{frame_global_idx:04d}.ply"
        ply_path = os.path.join(pred_dir, filename)

        save_ply_with_organ_labels(
            full_pred_points,
            full_pred_labels,
            leaf_tip_positions=full_pred_tips,
            leaf_tip_labels=full_pred_tip_labels,
            output_path=ply_path,
            is_train=full_pred_is_train,
            normals=full_pred_normals,
            colors=full_pred_colors,
        )

        # Canonical (pre-alignment) assembly: points + organ labels only.
        if save_canonical and all_pred_points_canonical:
            save_ply_with_organ_labels(
                np.concatenate(all_pred_points_canonical, axis=0),
                full_pred_labels,
                output_path=os.path.join(canonical_dir, filename),
                is_train=full_pred_is_train,
            )

        saved_count += 1

    print(f"  Saved {saved_count} interpolated full-plant PLYs to {pred_dir}")

    # Save ground truth frames at matching positions in the interpolation schedule
    _save_gt_frames(
        manager, out_dir, sequence_name, interp_schedule, timepoint_days, split_dir,
    )

def _save_gt_frames(manager, out_dir, sequence_name, interp_schedule, timepoint_days, split_dir=None):
    """Save GT point clouds for all available days in the sequence."""
    gt_dir = os.path.join(out_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    dataset = manager.dataset
    plant_dataset = dataset.plant_dataset
    full_plant_seq = plant_dataset.get_timeseries_by_sequence_name(
        dataset.sequence_name
    )
    if full_plant_seq is None:
        print("  Warning: could not access plant dataset for GT, skipping GT saving")
        return

    # Load the full (unstrided) split file for is_train masks
    split_data = None
    full_split = os.path.join(
        resolve_point_splits_dir(split_dir), f"{dataset.sequence_name}_split.json"
    )
    if os.path.isfile(full_split):
        with open(full_split) as f:
            split_data = json.load(f)
    else:
        print(f"  Warning: no split file at {full_split}; GT frames get no is_train")

    # Map each day to its interpolation schedule frame index
    schedule_day_to_frame = {}
    for frame_idx, (target_day, _, _) in enumerate(interp_schedule):
        # First occurrence wins (avoids duplicates at endpoints)
        if target_day not in schedule_day_to_frame:
            schedule_day_to_frame[target_day] = frame_idx

    training_day_set = set(timepoint_days)
    gt_count = 0

    for tp_idx, plant_tp in enumerate(full_plant_seq["timepoints"]):
        day = plant_tp["day"]
        gt_points = plant_tp["points"]
        gt_labels = plant_tp["labels"].astype(np.int32)
        gt_normals = plant_tp.get("normals")

        # Build is_train from split file
        gt_is_train = None
        if split_data and tp_idx < len(split_data["timepoints"]):
            split_info = split_data["timepoints"][tp_idx]
            gt_is_train = np.full(len(gt_points), -1, dtype=np.int32)
            train_indices = split_info.get("train_indices", [])
            test_indices = split_info.get("test_indices", [])
            if train_indices:
                gt_is_train[np.array(train_indices)] = 1
            if test_indices:
                gt_is_train[np.array(test_indices)] = 0

        # Get leaf tips
        gt_tips = None
        gt_tip_labels = None
        leaf_tip_idxs = plant_tp.get("leaf_tip_idxs", np.array([]))
        if leaf_tip_idxs.size > 0:
            gt_tips = gt_points[leaf_tip_idxs]
            gt_tip_labels = gt_labels[leaf_tip_idxs]

        # Use interpolation schedule frame index if available, else closest match
        frame_idx = schedule_day_to_frame.get(day)
        if frame_idx is None:
            frame_idx = min(schedule_day_to_frame.values(),
                           key=lambda fi: abs(interp_schedule[fi][0] - day))

        day_str = f"{day:.2f}"
        split_tag = "train" if day in training_day_set else "eval"
        filename = f"{sequence_name}_D{day_str}_F{frame_idx:04d}_{split_tag}.ply"
        ply_path = os.path.join(gt_dir, filename)

        save_ply_with_organ_labels(
            gt_points,
            gt_labels,
            leaf_tip_positions=gt_tips,
            leaf_tip_labels=gt_tip_labels,
            output_path=ply_path,
            is_train=gt_is_train,
            normals=gt_normals,
        )
        gt_count += 1

    n_train = sum(1 for d in training_day_set if d in {tp["day"] for tp in full_plant_seq["timepoints"]})
    n_eval = gt_count - n_train
    print(f"  Saved {gt_count} ground truth PLYs to {gt_dir} ({n_train} train, {n_eval} eval)")

def _find_experiment_root(start_dir):
    """Walk up from start_dir to find the directory containing experiment_config.json."""
    d = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(d, "experiment_config.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent

def _collect_run_dirs(results_dir, sequence_dir):
    """Return list of (run_dir, run_tag) for a sequence.

    Handles three layouts:
    - results_dir itself has organs/ (user pointed directly at a run dir): single entry, no tag
    - sequence_dir has organs/ directly (classic single-run layout): single entry, no tag
    - sequence_dir has run_N/ subdirs each containing organs/ (multi-run layout): one entry per run
    """
    if os.path.isdir(os.path.join(results_dir, "organs")):
        base = os.path.basename(results_dir)
        run_tag = base if re.match(r"^run_\d+$", base) else None
        return [(results_dir, run_tag)]

    if os.path.isdir(os.path.join(sequence_dir, "organs")):
        return [(sequence_dir, None)]

    run_subdirs = sorted(
        d for d in os.listdir(sequence_dir)
        if d.startswith("run_")
        and os.path.isdir(os.path.join(sequence_dir, d, "organs"))
    )
    if run_subdirs:
        return [(os.path.join(sequence_dir, r), r) for r in run_subdirs]

    return []

def main():
    parser = argparse.ArgumentParser(description="Generate dense interpolation from trained models")
    parser.add_argument("--results_dir", required=True,
                        help="Path to experiment root, sequence dir, or run dir. "
                             "experiment_config.json is found by searching upward if needed.")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequences to process (default: all)")
    parser.add_argument("--run", default=None,
                        help="Process only this run number (e.g. 1 for run_1). Default: all runs.")
    parser.add_argument("--n_interp", type=int, default=5,
                        help="Number of intermediate frames between training frames")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <results_dir>/interpolation)")
    parser.add_argument("--device", default="cuda",
                        help="Device to use")
    parser.add_argument("--split_dir", default="configs/datasets/TrackPlant3D/point_splits",
                        help="Directory with point split files")
    parser.add_argument("--tracking_cmap", default="turbo",
                        choices=["turbo", "rgb"],
                        help="Tracking color mode: 'turbo' (perceptual colormap) "
                             "or 'rgb' (R=X, G=Y, B=Z)")
    parser.add_argument(
        "--use_pose_anchoring", action=argparse.BooleanOptionalAction, default=True,
        help="Use exact per-day transformation matrices for aligned leaves (default: on). "
             "--no-use_pose_anchoring: interpolate between training-frame matrices only, "
             "so no evaluation-day transform data is used."
    )
    parser.add_argument(
        "--save-canonical", dest="save_canonical", action="store_true",
        help="Also write predicted_canonical/ : the pre-alignment (organ-canonical) "
             "assembly, for separating intrinsic model velocity from pose-anchoring motion."
    )
    parser.add_argument(
        "--save_normals", action=argparse.BooleanOptionalAction, default=True,
        help="Estimate normals for the predicted frames (default: on). The models "
             "predict positions only; --no-save_normals skips the estimation."
    )
    args = parser.parse_args()

    # Normalize --run: accept "1" or "run_1"
    if args.run is not None and not args.run.startswith("run_"):
        args.run = f"run_{args.run}"

    results_dir = os.path.abspath(args.results_dir)

    # Find experiment root by searching upward for experiment_config.json
    exp_root = _find_experiment_root(results_dir)
    if exp_root is None:
        raise FileNotFoundError(
            f"Could not find experiment_config.json in or above {results_dir}"
        )

    config_path = os.path.join(exp_root, "experiment_config.json")
    with open(config_path) as f:
        experiment_config = json.load(f)

    model_type = experiment_config["config"]["default_model"]["model_type"]
    sequences = args.sequences or experiment_config["sequence_names"]
    output_base = args.output_dir or os.path.join(exp_root, "interpolation")

    print(f"Experiment root: {exp_root}")
    print(f"Model type: {model_type}")
    print(f"Sequences: {sequences}")
    print(f"Intermediate frames: {args.n_interp}")
    print(f"Output: {output_base}")
    print()

    for seq_name in sequences:
        print(f"{'=' * 60}")
        print(f"Processing: {seq_name}")
        print(f"{'=' * 60}")

        sequence_dir = os.path.join(exp_root, "sequences", seq_name)
        if not os.path.isdir(sequence_dir) and not os.path.isdir(os.path.join(results_dir, "organs")):
            print(f"  Skipping {seq_name}: directory not found at {sequence_dir}")
            continue

        run_dirs = _collect_run_dirs(results_dir, sequence_dir)
        if not run_dirs:
            print(f"  Skipping {seq_name}: no organs/ or run_N/ subdirs found")
            continue

        for run_dir, run_tag in run_dirs:
            if args.run and run_tag != args.run:
                continue
            if run_tag:
                print(f"  --- Run: {run_tag} ---")

            try:
                manager = load_manager_with_model(
                    experiment_config, seq_name, run_dir,
                    device=args.device, split_dir=args.split_dir,
                    run_dir=exp_root,
                )

                if run_tag:
                    seq_out_dir = os.path.join(output_base, seq_name, run_tag)
                else:
                    seq_out_dir = os.path.join(output_base, seq_name)

                generate_dense_interpolation(
                    manager,
                    out_dir=seq_out_dir,
                    n_interp=args.n_interp,
                    device=args.device,
                    tracking_cmap=args.tracking_cmap,
                    use_pose_anchoring=args.use_pose_anchoring,
                    save_canonical=args.save_canonical,
                    save_normals=args.save_normals,
                    split_dir=args.split_dir,
                )
            except Exception as e:
                label = f"{seq_name}/{run_tag}" if run_tag else seq_name
                print(f"  ERROR processing {label}: {e}")
                import traceback
                traceback.print_exc()

        print()

    print("Done!")

if __name__ == "__main__":
    main()
