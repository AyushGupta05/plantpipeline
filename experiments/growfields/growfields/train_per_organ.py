#!/usr/bin/env python3
"""
Train per-organ motion fields: one field per plant organ (stem + leaves).

Reads the `default_model` and `per_organ_training` sections of a config and
drives `OrganModelManager` directly -- it does not shell out to
`train_fullplant.py`, which is the separate single-model entry point reading
`cfg.model`. One or many sequences may be given; results are aggregated across
them for comparison.

Usage:
    # Step 1: Generate train/test splits for all sequences (do this ONCE)
    for seq in maize_control_plant1 maize_control_plant2 maize_control_plant3; do
        python scripts/create_train_test_splits.py \\
            --sequence $seq \\
            --train_ratio 0.5 \\
            --seed 42
    done

    # Step 2: Run batch experiments with pre-computed point splits
    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits\\
        --sequences maize_control_plant1 maize_control_plant2 maize_control_plant3

    # Run on predefined val/test sequence splits (recommended)
    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --split val \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits \\

    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --split test \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits

    # Run on sequences matching filters
    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --crop maize --treatment control --max_sequences 5 \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits

    # Run with aligned leaves (two equivalent ways)
    # Option 1: Using --aligned_leaves flag
    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --split val \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits \\
        --aligned_leaves

    # Option 2: Using Hydra-style override
    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --split val \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits \\
        dataset.aligned_leaves=true

    # Override any config parameter (e.g., change model parameters)
    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --split val \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits \\
        default_model.n_steps=5000 \\
        default_model.finetune_steps=2000

    # Run multiple times with different model initialization seeds (same train/test point split)
    python train_per_organ.py \\
        --config configs/per_organ/continuous.yaml \\
        --split test \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits \\
        --n_runs 3 \\
        --seed 42

    # Interpolation experiments: train on every other timepoint
    # Step 1: Train models (saves models, predictions only on training timepoints)
    python train_per_organ.py \\
        --config configs/per_organ/pairwise.yaml \\
        --split val \\
        --points_split_dir configs/datasets/TrackPlant3D/point_splits \\
        --train_timepoint_stride 2

    # Step 2: Evaluate on ALL timepoints (including unseen) using saved models
    python scripts/compute_metrics_per_organ.py \\
        --experiment_dir results/fullplant_per_organ_batch/<experiment_name>

    # Step 3: Generate LaTeX table filtering to unseen timepoints only
    python scripts/create_latex_table_per_organ.py \\
        --csv results/fullplant_per_organ_batch/<experiment_name>/all_metrics.csv \\
        --full_table \\
        --interpolation \\
        --min_timepoints_per_organ 3

Note:
    - The --seed parameter controls model initialization (PyTorch, NumPy seeds), NOT the train/test split
    - Train/test point splits are pre-computed and loaded from --points_split_dir
    - This ensures all runs use identical train/test splits for fair comparison
    - --train_timepoint_stride: Trains on every Nth timepoint, saves models only
    - For full evaluation on ALL timepoints, use scripts/compute_metrics_per_organ.py after training
    - --interpolation flag in create_latex_row filters metrics to unseen timepoints only
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

try:
    import wandb
except ImportError:
    wandb = None

from growfields.pipeline.config import (
    DEVICE,
    deterministic_enabled,
    enable_deterministic,
    merge_cli_overrides,
    set_seed,
)
from growfields.pipeline.data import load_plant_dataset
from growfields.pipeline.results import (
    prepare_results_dir,
    slurm_job_suffix,
    write_experiment_config,
)
from growfields.pipeline.splits import (
    create_holdout_last_n_split_file,
    create_strided_split_file,
    load_sequence_split,
    point_split_path,
    require_point_splits,
)
from growfields.pipeline.wandb_logging import finish_run, start_sequence_run, wandb_settings
from growfields.datasets.trackplant3d import list_available_sequences
from growfields.models.organ_model_manager import OrganModelManager
from growfields.utils.io_utils import write_json

def run_single_experiment(
    sequence_name, cfg, args, experiment_dir, run_idx=None, use_wandb=False
):
    """Run experiment on a single plant sequence.

    Args:
        sequence_name: Name of sequence to process
        cfg: OmegaConf configuration object
        args: Argument namespace
        experiment_dir: Base experiment directory
        run_idx: Run number (1-indexed), None if not using multiple runs
        use_wandb: If True, log metrics to wandb

    Returns:
        Dict with results including metrics, organ info, and success status
    """
    run_str = f" [Run {run_idx}/{args.n_runs}]" if run_idx is not None else ""
    print("\n" + "=" * 80)
    print(f"Processing: {sequence_name}{run_str}")
    print("=" * 80)

    start_time = time.time()

    seq_dir = os.path.join(experiment_dir, "sequences", sequence_name)
    run_dir = seq_dir if run_idx is None else os.path.join(seq_dir, f"run_{run_idx}")

    try:
        # Override sequence name in config
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_copy.dataset.sequence = sequence_name

        # Determine if per-organ training is enabled
        per_organ_enabled = cfg_copy.get("per_organ_training", {}).get("enabled", False)

        if not per_organ_enabled:
            print("⚠️  Warning: per_organ_training not enabled in config")
            print("This entry point trains per-organ models; see train_fullplant.py.")
            return {
                "sequence_name": sequence_name,
                "success": False,
                "error": "per_organ_training not enabled in config",
            }

        # Determine normalization strategy
        normalize = False if per_organ_enabled else cfg_copy.dataset.normalize

        # Load dataset with train/test point split
        split_file = cfg_copy.dataset.get("split_file", None)

        # If points_split_dir is provided via CLI, construct split file path
        if args.points_split_dir and not split_file:
            # main() already checked that this file exists (require_point_splits).
            split_file = point_split_path(args.points_split_dir, sequence_name)
            print(f"  Using point split file: {split_file}")
        elif split_file:
            # Use pre-computed split file from config
            print(f"  Using point split file: {split_file}")

        # Load dataset with optional timepoint stride for training
        timepoint_stride_for_training = (
            args.train_timepoint_stride if args.train_timepoint_stride else 1
        )

        # If using stride for training, create a filtered split file for the training phase
        train_split_file = split_file
        if args.train_timepoint_stride and split_file and os.path.exists(split_file):
            train_split_file = create_strided_split_file(
                split_file, args.train_timepoint_stride, sequence_name, experiment_dir
            )
            print(f"  Created strided split file for training: {train_split_file}")
        elif args.train_holdout_last_n and split_file and os.path.exists(split_file):
            # Create for reference/documentation, but dataset always uses the full split file.
            # Holdout filtering itself happens in models.organ_ablations.
            # apply_holdout_last_n, which OrganModelManager calls on init.
            holdout_split_file = create_holdout_last_n_split_file(
                split_file, args.train_holdout_last_n, sequence_name, experiment_dir
            )
            print(f"  Created holdout-last-{args.train_holdout_last_n} split file (reference): {holdout_split_file}")

        dataset = load_plant_dataset(
            cfg_copy.dataset,
            sequence_name,
            split_file=train_split_file,
            timepoint_stride=timepoint_stride_for_training,
            normalize=normalize,
            device=DEVICE,
        )

        info = dataset.get_sequence_info()
        if args.train_holdout_last_n:
            print(
                f"Loaded {info['n_timepoints']} training timepoints (holdout_last_n={args.train_holdout_last_n}) "
                f"with {info['n_leaves']} leaves"
            )
        elif args.train_timepoint_stride:
            print(
                f"Loaded {info['n_timepoints']} training timepoints (stride={args.train_timepoint_stride}) "
                f"with {info['n_leaves']} leaves"
            )
        else:
            print(
                f"Loaded {info['n_timepoints']} timepoints with {info['n_leaves']} leaves"
            )

        # Create OrganModelManager
        print("\n" + "=" * 80)
        print("🌿 Per-Organ Training Mode")
        print("=" * 80)

        debug_seq_dir = os.path.join(experiment_dir, "sequences", sequence_name)

        # Holes are cut during OrganModelManager.__init__, so this has to reach it
        # through the config rather than being set on the manager afterwards.
        if args.train_holes_n and args.train_holes_n > 0:
            cfg_copy.default_model.debug_holes_dir = os.path.join(
                debug_seq_dir, "debug_holes"
            )

        manager = OrganModelManager(
            dataset=dataset,
            default_config=OmegaConf.to_container(cfg_copy.default_model, resolve=True),
            organ_specific_configs=(
                OmegaConf.to_container(
                    cfg_copy.per_organ_training.organ_configs, resolve=True
                )
                if "organ_configs" in cfg_copy.per_organ_training
                else {}
            ),
            device=DEVICE,
            train_holdout_last_n=args.train_holdout_last_n,
            save_normals=cfg_copy.output.get("save_normals", True),
        )

        manager.use_wandb = use_wandb

        # Noise is applied per training step, after init, so this one is an attribute.
        input_noise_std = cfg_copy.default_model.get("input_noise_std", 0.0)
        if input_noise_std and input_noise_std > 0:
            manager.debug_noise_dir = os.path.join(debug_seq_dir, "debug_noise")

        # Display organ info
        organ_info = manager.get_organ_info()
        print(f"\nOrgan summary:")
        for organ_id, info_dict in organ_info["organs"].items():
            print(
                f"  Organ {organ_id} ({info_dict['type']}): "
                f"{info_dict['n_timepoints_present']}/{info_dict['n_total_timepoints']} timepoints, "
                f"model_type={info_dict.get('model_type', 'N/A')}"
            )

        # Train all organ models
        training_start = time.time()
        manager.train_all_organs()
        training_duration = time.time() - training_start

        # For interpolation experiments, just save models
        # Use scripts/compute_metrics_per_organ.py to evaluate on full dataset
        if args.train_timepoint_stride:
            print("\n" + "=" * 80)
            print("⚠️  Interpolation Experiment Mode")
            print("=" * 80)
            print(
                f"  Trained on stride={args.train_timepoint_stride} ({info['n_timepoints']} timepoints)"
            )
            print(f"  Models will be saved")
            print(f"  To evaluate on ALL timepoints (including unseen), run:")
            print(
                f"    python scripts/compute_metrics_per_organ.py --experiment_dir <path_to_this_experiment>"
            )

        # Save results. prepare_results_dir also drops config.yaml and a copy of
        # the split file in, so a batch run directory records as much about how
        # it was produced as a single run does.
        prepare_results_dir(run_dir, cfg_copy, train_split_file, verbose=False)

        # Save per-organ predictions, models, and metrics
        if cfg_copy.output.save_predictions or cfg_copy.output.save_models:
            manager.save_all_organs(
                run_dir,
                save_predictions=cfg_copy.output.save_predictions,
                save_models=cfg_copy.output.save_models,
                save_metrics=True,
                save_full_plant=cfg_copy.output.get("save_full_plant", False),
            )

            # Save predictions for unseen timepoints (interpolation or extrapolation)
            if cfg_copy.output.get("save_full_plant", False):
                if args.train_holdout_last_n and args.train_holdout_last_n > 0:
                    print(f"\n📊 Saving extrapolation predictions for held-out final timepoints...")
                    extrap_dir = os.path.join(run_dir, "full_plant", "extrapolation")
                    os.makedirs(extrap_dir, exist_ok=True)
                    manager.save_extrapolation_predictions(
                        out_dir=extrap_dir,
                        use_pre_finetune_models=False,
                        original_split_file=split_file,
                    )
                elif args.train_timepoint_stride and args.train_timepoint_stride > 1:
                    print(f"\n📊 Saving interpolation predictions for unseen timepoints...")
                    interp_dir = os.path.join(run_dir, "full_plant", "interpolation")
                    os.makedirs(interp_dir, exist_ok=True)
                    manager.save_interpolation_predictions(
                        out_dir=interp_dir,
                        use_pre_finetune_models=False,
                        original_split_file=split_file,
                    )

        # Storage: the one number that cannot be recovered from the predictions.
        # Per-organ models live under organs/*/models/, a shared or encoder-decoder
        # model once under models/.
        total_model_size_mb = 0
        if cfg_copy.output.save_models:
            for sub in ("organs", "models"):
                for root, _, files in os.walk(os.path.join(run_dir, sub)):
                    for file in files:
                        if file.endswith(".pth"):
                            total_model_size_mb += os.path.getsize(
                                os.path.join(root, file)
                            ) / (1024 * 1024)

        print(f"\n⏱️  Training time: {training_duration/60:.2f} minutes")
        if cfg_copy.output.save_models:
            print(f"💾 Total model storage: {total_model_size_mb:.2f} MB")

        if use_wandb and wandb is not None and wandb.run is not None:
            wandb.log({"final/training_time_min": float(training_duration / 60)})

        elapsed_time = time.time() - start_time

        # Quality metrics are deliberately absent: they are recomputed from the
        # saved predictions by scripts/compute_metrics_per_organ.py and
        # scripts/compute_metrics_fullplant.py, which produce the published
        # numbers. Those scripts read exactly the timing, storage and run index
        # below out of this file; everything else here is provenance.
        result = {
            "sequence_name": sequence_name,
            "n_timepoints": info["n_timepoints"],
            "n_leaves": info["n_leaves"],
            "n_organs": organ_info["n_organs"],
            "train_timepoint_stride": args.train_timepoint_stride,
            "n_training_timepoints": info["n_timepoints"],
            "training_time_seconds": float(training_duration),
            "training_time_minutes": float(training_duration / 60),
            "total_time_seconds": float(elapsed_time),
            "total_model_size_mb": (
                float(total_model_size_mb) if cfg_copy.output.save_models else 0.0
            ),
            "success": True,
        }

        if run_idx is not None:
            result["run_idx"] = run_idx

        results_dict = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "sequence_name": sequence_name,
            "model_type": "per_organ",
            "aligned_leaves": cfg_copy.dataset.aligned_leaves,
            "train_timepoint_stride": args.train_timepoint_stride,
            "n_training_timepoints": info["n_timepoints"],
            "dataset_info": info,
            "aggregate_stats": result,
            "organ_info": organ_info,
        }

        write_json(results_dict, os.path.join(run_dir, "results.json"), clean=True)

        print(f"✅ Success: {sequence_name}")
        return result

    except Exception as e:
        print(f"❌ Failed: {sequence_name}")
        print(f"   Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {
            "sequence_name": sequence_name,
            "success": False,
            "error": str(e),
        }


def completed_run_dirs(experiment_dir, sequence_names, n_runs):
    """Run dirs of the requested sequences that already hold a finished run."""
    run_dirs = []
    for name in sequence_names:
        seq_dir = os.path.join(experiment_dir, "sequences", name)
        if n_runs > 1:
            run_dirs += [os.path.join(seq_dir, f"run_{i}") for i in range(1, n_runs + 1)]
        else:
            run_dirs.append(seq_dir)
    return [d for d in run_dirs if os.path.exists(os.path.join(d, "results.json"))]

def main():
    parser = argparse.ArgumentParser(
        description="Train one motion field per plant organ, over one or more sequences"
    )

    # Config and dataset selection
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/per_organ/continuous.yaml)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override dataset path from config",
    )
    parser.add_argument(
        "--points_split_dir",
        type=str,
        default="configs/datasets/TrackPlant3D/point_splits",
        help="Directory containing pre-computed train/test point split files (default: configs/datasets/TrackPlant3D/point_splits/)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        help="Use predefined data split (val or test). Loads from configs/datasets/TrackPlant3D/data_splits.json",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        help="Specific sequences to process (if not provided, uses --split or filters)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        help="Filter by crop (e.g., 'maize', 'tomato', 'sorghum')",
    )
    parser.add_argument(
        "--treatment",
        type=str,
        help="Filter by treatment (e.g., 'control', 'heat')",
    )
    parser.add_argument(
        "--min_timepoints",
        type=int,
        default=None,
        help="Minimum number of timepoints (overrides config)",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        help="Maximum number of sequences to process",
    )

    # Experiment options
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: results/fullplant_per_organ_batch)",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help=(
            "Use an existing experiment directory instead of creating a new timestamped one, "
            "to add sequences that are missing from it. Every listed sequence is trained from "
            "scratch for all --n_runs; there is no mid-training resume. Refuses to start if "
            "any of those runs already completed, unless --overwrite is given. "
            "Example: results/fullplant_per_organ_batch/20260410_123456_fullplant_batch_shared_continuous_test_aligned"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "With --experiment_dir, retrain listed runs that already completed. Without it "
            "the script refuses to start, since every listed run is trained from scratch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for model initialization (controls PyTorch/NumPy randomness, NOT train/test split)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Bitwise-repeatable training for a given seed: an atomic-free Chamfer loss "
        "plus torch.use_deterministic_algorithms; about 30%% slower and uses more GPU "
        "memory. Same as "
        "GROWFIELDS_DETERMINISTIC=1. The published tables were trained without it.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of runs per sequence with different seeds (default: 1)",
    )
    parser.add_argument(
        "--aligned_leaves",
        action="store_true",
        help="Canonicalise each leaf: train in canonical leaf space instead of plant "
        "space. This is the paper's canonicalisation; omitting it is the ablation.",
    )
    parser.add_argument(
        "--train_timepoint_stride",
        type=int,
        default=None,
        help="Stride for selecting training timepoints (e.g., 2 = train on every other timepoint). "
        "Models will still be evaluated on ALL timepoints for interpolation experiments.",
    )
    parser.add_argument(
        "--train_holdout_last_n",
        type=int,
        default=None,
        help="Hold out the last N timepoints from training for extrapolation experiments. "
        "After training, predictions are saved for all timepoints (including held-out); "
        "use compute_metrics_per_organ.py --interpolation to evaluate on the held-out ones.",
    )
    parser.add_argument(
        "--input_noise_std",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise added to training input point clouds "
        "(in normalized [-1,1] space). 0 = no noise (default). "
        "Convert to mm: sigma_mm = input_noise_std * per_organ_scale.",
    )
    parser.add_argument(
        "--train_holes_n",
        type=int,
        default=0,
        help="Number of spatial hole centres removed per leaf organ per timepoint (0 = off). "
        "Each centre removes --train_holes_k nearest neighbours from the training cloud.",
    )
    parser.add_argument(
        "--train_holes_k",
        type=int,
        default=10,
        help="Number of nearest neighbours removed per hole centre (default 10). "
        "Total removal is capped at 30%% of the cloud; clouds < 50 pts are skipped.",
    )
    args, unknown = parser.parse_known_args()

    if args.deterministic or deterministic_enabled():
        enable_deterministic()
        args.deterministic = True

    # Treat unknown args as config overrides (Hydra-style: key=value)
    config_overrides = unknown

    # Load configuration
    print("=" * 80)
    print("Loading configuration...")
    print("=" * 80)
    cfg = OmegaConf.load(args.config)
    print(f"Config: {args.config}")

    # Apply CLI overrides from argparse
    if args.dataset_path:
        cfg.dataset.dataset_path = args.dataset_path
    if args.min_timepoints:
        cfg.dataset.min_timepoints = args.min_timepoints
    if args.aligned_leaves:
        cfg.dataset.aligned_leaves = True
    if args.input_noise_std > 0:
        cfg.default_model.input_noise_std = args.input_noise_std
        print(f"Overriding: input_noise_std = {args.input_noise_std}")
        print("Overriding: aligned_leaves = True")
    if args.train_holes_n > 0:
        cfg.default_model.train_holes_n = args.train_holes_n
        cfg.default_model.train_holes_k = args.train_holes_k
        print(f"Overriding: train_holes_n = {args.train_holes_n}, train_holes_k = {args.train_holes_k}")

    # Override output directory for batch runs (unless user specifies otherwise)
    if args.output_dir:
        cfg.output.output_dir = args.output_dir
    else:
        # Default: use batch-specific output directory
        cfg.output.output_dir = "results/fullplant_per_organ_batch"

    print(f"Output directory: {cfg.output.output_dir}")

    # Set random seed if specified
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Get list of sequences (and extract any config overrides from --sequences argument)
    if args.split:
        try:
            plants, split_data = load_sequence_split(args.split)
        except (FileNotFoundError, KeyError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        sequences = [{"name": plant} for plant in plants]

        print(f"Using predefined '{args.split}' split:")
        print(f"  {split_data['n_plants']} plants")
        print(f"  Treatments: {', '.join(split_data['treatments'])}")

    elif args.sequences:
        # Filter out config overrides (key=value format) that may have been captured by nargs="+"
        actual_sequences = [seq for seq in args.sequences if "=" not in seq]
        sequences = [{"name": seq} for seq in actual_sequences]

        # Add filtered overrides back to config_overrides
        override_args = [seq for seq in args.sequences if "=" in seq]
        if override_args:
            config_overrides.extend(override_args)
    else:
        sequences = list_available_sequences(
            cfg.dataset.dataset_path,
            version=cfg.dataset.version,
            mode="plant",
            min_timepoints=cfg.dataset.min_timepoints,
            crop_filter=args.crop,
            exclude_sequences=cfg.dataset.get("exclude_sequences", None),
        )

        # Additional filtering by treatment
        if args.treatment:
            sequences = [s for s in sequences if args.treatment in s["name"]]

    if args.max_sequences and not args.split:
        sequences = sequences[: args.max_sequences]

    # Apply Hydra-style config overrides (key=value format)
    # This is done AFTER sequence extraction to catch any overrides that were
    # mistakenly captured by --sequences nargs="+"
    cfg = merge_cli_overrides(cfg, config_overrides)
    if config_overrides:
        print(f"Applied config overrides: {config_overrides}")

    print("=" * 80)
    print("Batch Full Plant (Per-Organ) Experiments")
    print("=" * 80)
    print(f"Found {len(sequences)} sequences to process:")
    for seq in sequences:
        print(f"  - {seq['name']}")

    if args.points_split_dir:
        try:
            require_point_splits(args.points_split_dir, [s["name"] for s in sequences])
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            sys.exit(1)

    # Check that per-organ training is enabled
    if not cfg.get("per_organ_training", {}).get("enabled", False):
        print("\n⚠️  ERROR: per_organ_training.enabled is not True in config!")
        print("This entry point requires per-organ training to be enabled.")
        print("Please use a config file with per_organ_training.enabled: true")
        sys.exit(1)

    # Create experiment directory
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
        experiment_name = Path(experiment_dir).name
        print(f"Adding to existing experiment directory: {experiment_dir}")
        done = completed_run_dirs(experiment_dir, [s["name"] for s in sequences], args.n_runs)
        if done and not args.overwrite:
            print(f"\n❌ {len(done)} of the requested runs already completed:")
            for d in done:
                print(f"   {d}")
            print(
                "Every listed sequence is retrained from scratch for all --n_runs, so these "
                "would be overwritten. List only the missing sequences, or pass --overwrite."
            )
            sys.exit(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(args.config).stem
        experiment_name = f"{timestamp}_fullplant_batch_{config_name}"
        if args.split:
            experiment_name += f"_{args.split}"
        if cfg.dataset.aligned_leaves:
            experiment_name += "_aligned"
        if args.train_timepoint_stride:
            experiment_name += f"_stride{args.train_timepoint_stride}"
        if args.train_holdout_last_n:
            experiment_name += f"_holdlast{args.train_holdout_last_n}"
        if args.train_holes_n:
            experiment_name += f"_holes_n{args.train_holes_n}_k{args.train_holes_k}"
        if args.crop:
            experiment_name += f"_{args.crop}"
        if args.treatment:
            experiment_name += f"_{args.treatment}"
        experiment_name += slurm_job_suffix()
        experiment_dir = os.path.join(cfg.output.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    write_experiment_config(
        experiment_dir,
        cfg,
        args,
        [s["name"] for s in sequences],
        config_file=args.config,
    )

    wandb_cfg, use_wandb = wandb_settings(cfg)

    # Run experiments
    results = []
    for i, seq in enumerate(sequences):
        sequence_name = seq["name"]
        print(f"\n[{i+1}/{len(sequences)}] {sequence_name}")

        # Init wandb run per sequence (per run with --n_runs > 1, below)
        seq_dir = os.path.join(experiment_dir, "sequences", sequence_name)
        os.makedirs(seq_dir, exist_ok=True)
        if use_wandb and args.n_runs == 1:
            start_sequence_run(cfg, wandb_cfg, sequence_name, experiment_name, seq_dir)

        if args.n_runs > 1:
            # Run multiple times with different seeds
            for run_idx in range(1, args.n_runs + 1):
                # Set seed for this run (base seed + run index)
                run_seed = (args.seed if args.seed is not None else 42) + run_idx
                set_seed(run_seed)
                if use_wandb:
                    start_sequence_run(
                        cfg, wandb_cfg, sequence_name, experiment_name,
                        os.path.join(seq_dir, f"run_{run_idx}"), run_idx=run_idx,
                    )

                result = run_single_experiment(
                    sequence_name,
                    cfg,
                    args,
                    experiment_dir,
                    run_idx=run_idx,
                    use_wandb=use_wandb,
                )
                # Keep failed runs too, so they count against the exit code.
                result.update(run_idx=run_idx, seed=run_seed)
                results.append(result)
                if not result.get("success", False):
                    print(f"❌ {sequence_name} run {run_idx} (seed {run_seed}) failed: "
                          f"{result.get('error')}")
            # Runs are reported individually; averaging across seeds is done by
            # the metrics scripts, which group the CSV by (sequence, run).
        else:
            # Single run
            result = run_single_experiment(
                sequence_name,
                cfg,
                args,
                experiment_dir,
                run_idx=None,
                use_wandb=use_wandb,
            )
            results.append(result)

        # Finish wandb run for this sequence
        if use_wandb:
            finish_run()

    n_ok = sum(1 for r in results if r.get("success", False))
    print("\n" + "=" * 80)
    print(f"✅ Batch experiments complete! {n_ok}/{len(results)} succeeded")
    print(f"Results saved to: {experiment_dir}")
    print("=" * 80)

    # Exit non-zero when anything failed: per-sequence errors are caught and
    # recorded above, so without this a batch where every run failed still
    # reports success to the shell, and a scheduler sees a healthy job.
    if n_ok < len(results):
        sys.exit(1)

if __name__ == "__main__":
    main()
