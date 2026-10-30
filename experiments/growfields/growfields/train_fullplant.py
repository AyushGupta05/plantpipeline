#!/usr/bin/env python3
"""
Train motion fields on FULL PLANT sequences from TrackPlant3D using OmegaConf.

Supports both single-sequence and batch mode with YAML config files and CLI overrides.

Usage:
    # Single sequence (backward compatible)
    python train_fullplant.py \
        --config configs/full_plant/pairwise.yaml \
        dataset.sequence=maize_control_plant2

    # Batch: all test sequences (replaces scripts/run_fullplant_single_model_batch.sh)
    python train_fullplant.py \
        --config configs/full_plant/pairwise.yaml \
        --split test \
        --points_split_dir configs/datasets/TrackPlant3D/point_splits \
        model.n_steps=100 model.finetune_steps=100

    # Batch: specific sequences
    python train_fullplant.py \
        --config configs/full_plant/pairwise.yaml \
        --sequences maize_control_plant2 maize_control_plant3 \
        --points_split_dir configs/datasets/TrackPlant3D/point_splits

    # Use config file with CLI overrides
    python train_fullplant.py \
        --config configs/full_plant/continuous.yaml \
        dataset.sequence=maize_control_plant2 \
        model.n_steps=100000

    # List sequences
    python train_fullplant.py --list_sequences
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

from growfields.datasets.trackplant3d import list_available_sequences
from growfields.pipeline.config import (
    DEVICE,
    deterministic_enabled,
    enable_deterministic,
    load_config,
    set_seed,
)
from growfields.pipeline.data import (
    combine_train_test_clouds,
    get_day_range,
    get_normalization_params,
    load_plant_dataset,
)
from growfields.pipeline.results import (
    prepare_results_dir,
    slurm_job_suffix,
    write_experiment_config,
)
from growfields.pipeline.splits import (
    create_strided_split_file,
    load_sequence_split,
    point_split_path,
    require_point_splits,
)
from growfields.pipeline.wandb_logging import finish_run, start_sequence_run, wandb_settings
from growfields.pipeline.training import SequenceArrays, train_full_plant
from growfields.utils.io_utils import write_json

def run_single_sequence(cfg, results_dir, use_wandb=False, timepoint_stride=1, original_split_file=None):
    """Run training on a single sequence.

    Args:
        cfg: OmegaConf config with dataset.sequence already set.
        results_dir: Directory to save results (created if needed).
        use_wandb: If True, log metrics to wandb.
        timepoint_stride: Stride for selecting training timepoints (1 = all).
        original_split_file: Path to the original (unstrided) split file.
            Used for interpolation GT is_train masks. If None, inferred from cfg.

    Returns:
        dict with training results and metrics.
    """

    sequence_name = cfg.dataset.sequence

    training_mode = f"{cfg.model.model_type} model"

    print("=" * 80)
    print(f"Training {training_mode} on full plant: {sequence_name}")
    print(f"Leaf alignment: {cfg.dataset.aligned_leaves}")
    print("=" * 80)

    normalize = cfg.dataset.normalize

    # Handle split_file parameter
    split_file = cfg.dataset.get("split_file", None)
    if split_file and not os.path.isabs(split_file):
        # Make relative paths relative to project root
        split_file = os.path.join(os.path.dirname(__file__), split_file)

    # Create strided split file if stride > 1 and a split file exists.
    # In batch mode the strided split may already be set by the caller;
    # detect this via the "timepoint_stride" key inside the JSON.
    if original_split_file is None:
        original_split_file = split_file  # Default: same as split_file
    if timepoint_stride > 1 and split_file and os.path.isfile(split_file):
        with open(split_file) as _f:
            _split_check = json.load(_f)
        if "timepoint_stride" not in _split_check:
            # Not yet strided — create strided version (single-sequence mode)
            original_split_file = split_file
            split_file = create_strided_split_file(
                split_file, timepoint_stride, sequence_name, results_dir
            )
            print(f"Using strided split file (stride={timepoint_stride}): {split_file}")

    # Load dataset
    dataset = load_plant_dataset(
        cfg.dataset,
        cfg.dataset.sequence,
        split_file=split_file,
        timepoint_stride=timepoint_stride,
        normalize=normalize,
        device=DEVICE,
    )

    info = dataset.get_sequence_info()
    print("\nDataset info:")
    for key, val in info.items():
        print(f"  {key}: {val}")

    # Full plant is predicted from all points; the split is kept for metrics.
    clouds, labels, normals, is_train_masks = combine_train_test_clouds(dataset)

    gt_tips = dataset.get_leaf_tips(normalized=False, as_tensors=True)
    gt_tip_labels = dataset.get_leaf_tip_labels(as_tensors=True)
    cloud_paths = dataset.get_timepoint_names()

    days, full_day_range = get_day_range(dataset)

    print(f"\nLoaded {len(clouds)} clouds with {info['n_leaves']} leaves")
    print(f"Days: {days}, full_day_range: {full_day_range}")
    n_tips_total = sum(info["leaf_tips_per_timepoint"])
    print(f"Total leaf tips across all timepoints: {n_tips_total}")

    min_xyz, scale = get_normalization_params(dataset, DEVICE)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prepare_results_dir(results_dir, cfg, split_file)

    # Train model (single model mode)
    print("\n" + "=" * 80)
    print(f"Training {cfg.model.model_type} model...")
    print("=" * 80)

    # Track training time
    training_start_time = time.time()

    train_full_plant(
        cfg,
        SequenceArrays(
            clouds=clouds,
            labels=labels,
            normals=normals,
            is_train_masks=is_train_masks,
            min_xyz=min_xyz,
            scale=scale,
            days=days,
            full_day_range=full_day_range,
            gt_tips=gt_tips,
            gt_tip_labels=gt_tip_labels,
            cloud_paths=cloud_paths,
            dataset=dataset,
        ),
        results_dir,
        device=DEVICE,
        use_wandb=use_wandb,
    )
    # Record training end time (including finetuning if enabled)
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time

    # Compute model size
    total_model_size_mb = 0.0
    if cfg.output.save_models:
        model_dir = os.path.join(results_dir, "models")
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".pth"):
                        filepath = os.path.join(root, file)
                        total_model_size_mb += os.path.getsize(filepath) / (1024 * 1024)

    print(f"\n⏱️  Training time: {training_duration/60:.2f} minutes")
    if cfg.output.save_models and total_model_size_mb > 0:
        print(f"💾 Total model storage: {total_model_size_mb:.2f} MB")

    if use_wandb and wandb is not None and wandb.run is not None:
        wandb.log({"final/training_time_min": float(training_duration / 60)})

    # Quality metrics are deliberately absent: they are recomputed from the saved
    # predictions by scripts/compute_metrics_fullplant.py, which produces the
    # published numbers. That script reads exactly the timing and storage fields
    # below out of this file; everything else here is provenance.
    results = {
        "timestamp": timestamp,
        "sequence_name": sequence_name,
        "model_type": cfg.model.model_type,
        "aligned_leaves": cfg.dataset.aligned_leaves,
        "dataset_info": info,
        "days": days,
        "training_time_seconds": float(training_duration),
        "training_time_minutes": float(training_duration / 60),
        "total_model_size_mb": (
            float(total_model_size_mb) if cfg.output.save_models else 0.0
        ),
    }

    if cfg.model.model_type == "pairwise":
        results["finetune"] = cfg.model.finetune_enabled
        results["finetune_steps"] = (
            cfg.model.finetune_steps if cfg.model.finetune_enabled else None
        )
        results["finetune_lr"] = (
            cfg.model.finetune_lr if cfg.model.finetune_enabled else None
        )
        results["w_global"] = cfg.model.w_global if cfg.model.finetune_enabled else None
        results["w_local"] = cfg.model.w_local if cfg.model.finetune_enabled else None

    write_json(results, os.path.join(results_dir, "results.json"), clean=True)

    print(f"\n✅ Results saved to: {results_dir}")
    print("=" * 80)

    return results

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train motion fields on full plant sequences (OmegaConf version)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/full_plant/pairwise.yaml)",
    )
    parser.add_argument(
        "--list_sequences",
        action="store_true",
        help="List available plant sequences and exit",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        help="Run on predefined split from configs/datasets/TrackPlant3D/data_splits.json",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Explicit list of sequences to process",
    )
    parser.add_argument(
        "--points_split_dir",
        type=str,
        default=None,
        help="Directory with pre-computed point split files (e.g. configs/datasets/TrackPlant3D/point_splits)",
    )
    parser.add_argument(
        "--train_timepoint_stride",
        type=int,
        default=None,
        help="Stride for selecting training timepoints (e.g., 2 = train on every other timepoint)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for model initialization (default: None = no seed set)",
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

    # Parse known args to separate config file from OmegaConf overrides
    args, remaining = parser.parse_known_args()

    if args.deterministic or deterministic_enabled():
        enable_deterministic()
        args.deterministic = True

    # Filter OmegaConf overrides that may have been captured by --sequences nargs="+"
    if args.sequences:
        actual_sequences = [s for s in args.sequences if "=" not in s]
        accidental_overrides = [s for s in args.sequences if "=" in s]
        args.sequences = actual_sequences if actual_sequences else None
        remaining = list(remaining) + accidental_overrides

    # Load configuration
    cfg = load_config(args.config, remaining)

    # Override list_sequences if specified via CLI flag
    if args.list_sequences:
        cfg.list_sequences = True

    # List sequences if requested
    if cfg.list_sequences:
        print("=" * 80)
        print("Available plant sequences:")
        print("=" * 80)
        sequences = list_available_sequences(
            cfg.dataset.dataset_path,
            version=cfg.dataset.version,
            mode="plant",
            min_timepoints=cfg.dataset.min_timepoints,
            exclude_sequences=cfg.dataset.get("exclude_sequences", None),
        )
        for seq in sequences[:20]:
            print(f"  {seq['name']:40s} - {seq['n_timepoints']} timepoints")
        print(f"\n... and {len(sequences) - 20} more sequences\n")
        return

    print(f"Using device: {DEVICE}")

    wandb_cfg, use_wandb = wandb_settings(cfg)

    timepoint_stride = args.train_timepoint_stride or cfg.dataset.get("stride", 1)

    batch_mode = args.split is not None or args.sequences is not None

    # For single-sequence mode, wandb init is deferred until results_dir is created (below)
    # For batch mode, init per-sequence in the loop

    if batch_mode:
        # ===== Batch mode =====

        # Load sequence list
        if args.split:
            try:
                sequences, _ = load_sequence_split(args.split)
            except (FileNotFoundError, KeyError) as e:
                print(f"Error: {e}")
                sys.exit(1)
            split_label = args.split
        else:
            sequences = args.sequences
            split_label = "custom"

        if args.points_split_dir:
            try:
                require_point_splits(args.points_split_dir, sequences)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

        model_type_str = cfg.model.model_type

        # Create batch directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"{timestamp}_{model_type_str}_{split_label}"
        if timepoint_stride > 1:
            batch_name += f"_stride{timepoint_stride}"
        batch_name += slurm_job_suffix()
        batch_dir = os.path.join(cfg.output.output_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("Batch Full Plant Experiments")
        print("=" * 80)
        print(f"Model type: {model_type_str}")
        print(f"Split: {split_label}")
        print(f"Batch directory: {batch_dir}")
        print(f"Number of sequences: {len(sequences)}")
        if args.points_split_dir:
            print(f"Point splits directory: {args.points_split_dir}")
        print("\nSequences to process:")
        for seq in sequences:
            print(f"  - {seq}")
        print("=" * 80)

        # Save experiment_config.json at batch root
        write_experiment_config(
            batch_dir, cfg, args, sequences, config_file=args.config
        )

        # Process sequences
        all_results = []
        successful = 0
        failed = 0
        failed_sequences = []

        for i, seq_name in enumerate(sequences):
            print(f"\n{'=' * 80}")
            print(f"[{i+1}/{len(sequences)}] Processing: {seq_name}")
            print(f"{'=' * 80}")

            # Set sequence in config
            cfg.dataset.sequence = seq_name

            # Auto-resolve split file
            seq_original_split_file = None
            if args.points_split_dir:
                # main() already checked that this file exists (require_point_splits).
                split_path = point_split_path(args.points_split_dir, seq_name)
                seq_original_split_file = split_path
                cfg.dataset.split_file = split_path
                print(f"Using point split: {split_path}")
                # Create strided split at batch root level (for test discovery)
                if timepoint_stride > 1:
                    strided_split = create_strided_split_file(
                        split_path, timepoint_stride, seq_name, batch_dir
                    )
                    cfg.dataset.split_file = strided_split
                    print(f"Using strided split (stride={timepoint_stride}): {strided_split}")
            else:
                cfg.dataset.split_file = None

            results_dir = os.path.join(batch_dir, "sequences", seq_name)

            if use_wandb and args.n_runs == 1:
                start_sequence_run(cfg, wandb_cfg, seq_name, batch_name, results_dir)

            try:
                if args.n_runs > 1:
                    run_results = []
                    for run_idx in range(1, args.n_runs + 1):
                        run_seed = (args.seed if args.seed is not None else 42) + run_idx
                        set_seed(run_seed)
                        run_dir = os.path.join(results_dir, f"run_{run_idx}")
                        print(f"\n[Run {run_idx}/{args.n_runs}] seed={run_seed}")
                        if use_wandb:
                            start_sequence_run(
                                cfg, wandb_cfg, seq_name, batch_name, run_dir, run_idx=run_idx
                            )
                        run_results.append(
                            run_single_sequence(
                                cfg,
                                run_dir,
                                use_wandb=use_wandb,
                                timepoint_stride=timepoint_stride,
                                original_split_file=seq_original_split_file,
                            )
                        )
                    all_results.append({
                        "sequence": seq_name,
                        "status": "success",
                        "n_runs": args.n_runs,
                        "training_time_minutes": float(
                            sum(r["training_time_minutes"] for r in run_results)
                            / len(run_results)
                        ),
                    })
                else:
                    if args.seed is not None:
                        set_seed(args.seed)
                    result = run_single_sequence(cfg, results_dir, use_wandb=use_wandb, timepoint_stride=timepoint_stride, original_split_file=seq_original_split_file)
                    all_results.append(
                        {
                            "sequence": seq_name,
                            "status": "success",
                            "training_time_minutes": result["training_time_minutes"],
                        }
                    )
                successful += 1
                print(f"✅ Success: {seq_name}")
            except Exception as e:
                traceback.print_exc()
                all_results.append(
                    {
                        "sequence": seq_name,
                        "status": "failed",
                        "error": str(e),
                    }
                )
                failed += 1
                failed_sequences.append(seq_name)
                print(f"❌ Failed: {seq_name}: {e}")

        # Print batch summary
        print(f"\n{'=' * 80}")
        print("Batch Processing Complete")
        print(f"{'=' * 80}")
        print(f"Total sequences: {len(sequences)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if failed_sequences:
            print(f"\nFailed sequences:")
            for seq in failed_sequences:
                print(f"  - {seq}")

        # Save batch summary. Quality metrics live in the CSVs the metrics
        # scripts build from the saved predictions, not here.
        batch_summary = {
            "timestamp": timestamp,
            "split": args.split,
            "model_type": model_type_str,
            "n_sequences": len(sequences),
            "successful": successful,
            "failed": failed,
            "failed_sequences": failed_sequences,
            "per_sequence_results": all_results,
        }

        write_json(
            batch_summary,
            os.path.join(batch_dir, "batch_summary.json"),
            clean=True,
        )

        print(f"\nResults saved to: {batch_dir}")
        print("=" * 80)

        # Exit non-zero when anything failed, for the same reason as in
        # train_per_organ.py: failures are caught per sequence, so the process
        # would otherwise exit 0 having produced nothing.
        if failed:
            sys.exit(1)

        if failed > 0:
            sys.exit(1)

    else:
        # ===== Single-sequence mode (backward compatible) =====

        if cfg.dataset.sequence is None:
            print(
                "Please specify dataset.sequence=<name>, "
                "or use --split/--sequences for batch mode"
            )
            return

        # Create timestamped results directory (original behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        leaf_align_str = "leaf_aligned" if cfg.dataset.aligned_leaves else "plant_space"

        model_type_str = cfg.model.model_type
        if cfg.model.model_type == "pairwise":
            finetune_str = "_finetune" if cfg.model.finetune_enabled else ""
        else:
            finetune_str = ""

        folder_name = f"{timestamp}_{model_type_str}_fullplant_{cfg.dataset.sequence}_{leaf_align_str}{finetune_str}"
        results_dir = os.path.join(cfg.output.output_dir, folder_name)
        os.makedirs(results_dir, exist_ok=True)

        # Save experiment_config.json for test discovery. The tests find the
        # split file via cli_args.points_split_dir, which is not set when a
        # split_file is configured directly, so derive it here.
        if args.points_split_dir is None:
            split_file = cfg.dataset.get("split_file", None)
            if split_file:
                if not os.path.isabs(split_file):
                    split_file = os.path.join(os.path.dirname(__file__), split_file)
                args.points_split_dir = os.path.abspath(os.path.dirname(split_file))

        write_experiment_config(
            results_dir,
            cfg,
            args,
            [cfg.dataset.sequence],
            config_file=args.config,
        )

        # Init wandb for single-sequence mode (now that results_dir exists)
        if use_wandb and args.n_runs == 1:
            start_sequence_run(
                cfg, wandb_cfg, cfg.dataset.sequence, folder_name, results_dir
            )

        if args.n_runs > 1:
            for run_idx in range(1, args.n_runs + 1):
                run_seed = (args.seed if args.seed is not None else 42) + run_idx
                set_seed(run_seed)
                run_dir = os.path.join(results_dir, f"run_{run_idx}")
                print(f"\n[Run {run_idx}/{args.n_runs}] seed={run_seed}")
                if use_wandb:
                    start_sequence_run(
                        cfg, wandb_cfg, cfg.dataset.sequence, folder_name, run_dir,
                        run_idx=run_idx,
                    )
                run_single_sequence(cfg, run_dir, use_wandb=use_wandb, timepoint_stride=timepoint_stride)
        else:
            if args.seed is not None:
                set_seed(args.seed)
            run_single_sequence(cfg, results_dir, use_wandb=use_wandb, timepoint_stride=timepoint_stride)

    if use_wandb:
        finish_run()

if __name__ == "__main__":
    main()
