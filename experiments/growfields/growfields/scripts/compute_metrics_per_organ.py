#!/usr/bin/env python3
"""
Compute per-organ metrics from a run's saved prediction PLYs.

This is the per-organ evaluation entry point, and the source of the paper's
per-organ numbers. Training deliberately does not write quality metrics (see
`OrganModelManager.save_organ_metrics`): they are always derived from the saved
PLYs here, so there is one implementation and one answer. The full-plant
counterpart is `compute_metrics_fullplant.py`.

Normally invoked through `compute_metrics_and_latex_table_per_organ.sh`, which
runs this and then `create_latex_table_per_organ.py`.

Usage:
    # One experiment directory
    python scripts/compute_metrics_per_organ.py \\
        --experiment_dir results/fullplant_per_organ/20260410_123456_fullplant_batch_shared_continuous_test_aligned

    # Every experiment under a parent directory
    python scripts/compute_metrics_per_organ.py --results_dir results/fullplant_per_organ

    # Overwrite metrics files that already exist
    python scripts/compute_metrics_per_organ.py --results_dir results/fullplant_per_organ --update
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from growfields.pipeline.results import extract_hyperparameters, load_results_json
from growfields.utils.io_utils import load_ply
from growfields.utils.metrics import (
    compute_chamfer_distance,
    compute_epe,
    compute_scene_flow_metrics,
)


def load_point_cloud_from_ply(ply_path):
    """Load point cloud from PLY file."""
    if not os.path.exists(ply_path):
        return None, None, None, None

    try:
        ply = load_ply(ply_path)
        return (
            ply["points"],
            ply.get("is_train"),
            ply.get("is_leaf_tip"),
            ply.get("normals"),
        )
    except Exception as e:
        print(f"Warning: Failed to load {ply_path}: {e}")
        return None, None, None, None

def recompute_organ_metrics(
    organ_dir,
    organ_id,
    organ_type,
    sequence_name="",
    run_idx=None,
    hparams=None,
    run_dir=None,
    interpolation=False,
    extrapolation=False,
    dataset_plant=None,
):
    """
    Recompute metrics for a single organ from saved predictions.

    Args:
        organ_dir: Directory containing predictions for this organ
        organ_id: Organ ID
        organ_type: Organ type (e.g., 'stem', 'leaf_1')
        sequence_name: Name of the sequence
        run_idx: Run index (if multiple runs)
        hparams: Dict of hyperparameters to include in output
        run_dir: Directory holding this run's results.json (timing and storage)
        interpolation: Process interpolation PLY files instead of normal predictions
        extrapolation: Process extrapolation PLY files (held-out final timepoints)
        dataset_plant: Pre-loaded TrackPlant3DDataset (plant mode) with sequence already set

    Returns:
        list: List of dicts, one per timepoint with all metrics
    """
    if extrapolation:
        predicted_dir = os.path.join(organ_dir, "extrapolation", "predicted")
        gt_dir = os.path.join(organ_dir, "extrapolation", "ground_truth")
        if not (os.path.exists(predicted_dir) and os.path.exists(gt_dir)):
            return []
    elif interpolation:
        predicted_dir = os.path.join(organ_dir, "interpolation", "predicted")
        gt_dir = os.path.join(organ_dir, "interpolation", "ground_truth")
        if not (os.path.exists(predicted_dir) and os.path.exists(gt_dir)):
            return []
    else:
        # Check for different directory structures:
        # 1. organs/*/predicted/ and organs/*/ground_truth/ (non-finetuned)
        # 2. organs/*/post_finetune/predicted/ and organs/*/post_finetune/ground_truth/ (finetuned)
        # 3. organs/*/pre_finetune/predicted/ and organs/*/pre_finetune/ground_truth/ (pre-finetuning)

        predicted_dir = os.path.join(organ_dir, "predicted")
        gt_dir = os.path.join(organ_dir, "ground_truth")

        # If not found, check for post_finetune (preferred for phase 2)
        if not (os.path.exists(predicted_dir) and os.path.exists(gt_dir)):
            post_ft_predicted = os.path.join(organ_dir, "post_finetune", "predicted")
            post_ft_gt = os.path.join(organ_dir, "post_finetune", "ground_truth")

            if os.path.exists(post_ft_predicted) and os.path.exists(post_ft_gt):
                predicted_dir = post_ft_predicted
                gt_dir = post_ft_gt
            else:
                # Try pre_finetune as fallback
                pre_ft_predicted = os.path.join(organ_dir, "pre_finetune", "predicted")
                pre_ft_gt = os.path.join(organ_dir, "pre_finetune", "ground_truth")

                if os.path.exists(pre_ft_predicted) and os.path.exists(pre_ft_gt):
                    predicted_dir = pre_ft_predicted
                    gt_dir = pre_ft_gt
                else:
                    return []

    # Load prediction files
    pred_files = sorted(
        [
            f
            for f in os.listdir(predicted_dir)
            if f.endswith(".ply") and "_tips" not in f
        ]
    )

    if not pred_files:
        return []

    # Extract day numbers from filenames
    # Format: pred_{sequence}_D{day}_{organ}.ply (normal) or pred_{sequence}_D{day}.ply (interpolation)
    day_pattern = re.compile(r"_D(\d+)(?:_|\.)")

    timepoints = []
    for f in pred_files:
        match = day_pattern.search(f)
        if match:
            timepoints.append(int(match.group(1)))

    if not timepoints:
        return []

    timepoints = sorted(set(timepoints))
    results = []

    # Load runtime and storage metrics once per organ (they are per-sequence)
    results_metrics = load_results_json(run_dir)

    # Load GT leaf tips directly from dataset to determine true first appearance
    # (PLY files start from timestep 1, so we need to check timestep 0 in the dataset)
    tip_first_appearance_global_frame = None
    global_frame_to_day = {}
    if organ_id > 0 and dataset_plant is not None:  # Only leaves have tips
        gt_tips = dataset_plant.get_leaf_tips(normalized=False, as_tensors=True)
        gt_tip_labels = dataset_plant.get_leaf_tip_labels(as_tensors=True)

        # Find first appearance of this organ's tip in GLOBAL frames
        for t, (tips, labels) in enumerate(zip(gt_tips, gt_tip_labels)):
            if tips is not None:
                for label in labels:
                    if label.item() == organ_id:
                        tip_first_appearance_global_frame = t
                        break
                if tip_first_appearance_global_frame is not None:
                    break

        # Load plant-mode dataset to get ALL timepoint names (not just for this organ)
        # We need the GLOBAL frame→day mapping
        plant_timepoint_names = dataset_plant.get_timepoint_names()

        # Build GLOBAL frame to day mapping from plant dataset
        for i, name in enumerate(plant_timepoint_names):
            # Extract day number from name (e.g., "D02" -> 2)
            match = re.search(r"D(\d+)", name)
            if match:
                global_frame_to_day[i] = int(match.group(1))

    for day in timepoints:
        # Find matching files for this day
        pred_file = next(
            (f for f in pred_files if re.search(rf"_D{day:02d}(?:_|\.)", f)), None
        )

        if pred_file is None:
            continue

        # Construct GT filename (same pattern but in ground_truth dir)
        gt_file = pred_file.replace("pred_", "gt_")

        pred_path = os.path.join(predicted_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        pred_pts, pred_is_train, pred_is_leaf_tip, pred_normals = (
            load_point_cloud_from_ply(pred_path)
        )
        gt_pts, gt_is_train, gt_is_leaf_tip, gt_normals = load_point_cloud_from_ply(
            gt_path
        )

        if pred_pts is None or gt_pts is None:
            continue

        # Compute CD metrics (with normals if available)
        cd_train, cd_test, nc_train, nc_test, n_train, n_test = compute_cd_metrics(
            pred_pts, gt_pts, pred_is_train, gt_is_train, pred_normals, gt_normals
        )

        # Compute EPE and scene flow metrics (AccS, AccR) for leaf tips only
        # Skip a tip on the frame it first appears: there is no previous frame to
        # have predicted it from, so it has no prediction to score.
        epe, accs, accr, n_tips = None, None, None, None
        if organ_id > 0 and pred_is_leaf_tip is not None and gt_is_leaf_tip is not None:
            # Map current day to GLOBAL frame index
            day_to_global_frame = {v: k for k, v in global_frame_to_day.items()}
            current_global_frame = day_to_global_frame.get(day)

            # Skip if this is the first appearance (comparing GLOBAL frames)
            skip_first_appearance = (
                current_global_frame is not None
                and tip_first_appearance_global_frame is not None
                and current_global_frame == tip_first_appearance_global_frame
            )

            if not skip_first_appearance:
                # Extract tip points using is_leaf_tip mask
                pred_tip_mask = pred_is_leaf_tip.astype(bool)
                gt_tip_mask = gt_is_leaf_tip.astype(bool)

                if pred_tip_mask.any() and gt_tip_mask.any():
                    pred_tips = pred_pts[pred_tip_mask]
                    gt_tips = gt_pts[gt_tip_mask]
                    epe = compute_epe(pred_tips, gt_tips)
                    n_tips = len(gt_tips)

                    # Compute AccS and AccR for leaf tips
                    accs, accr, _ = compute_scene_flow_metrics(pred_tips, gt_tips)

        # Build organ sequence name: e.g., "A1_leaf1", "A1_stem"
        organ_prefix = organ_type.split("_")[0]
        organ_seq_name = (
            sequence_name
            + "_"
            + organ_prefix
            + (str(organ_id) if organ_id != 0 else "")
        )

        row = {
            "sequence": sequence_name,
            "sequence_name": organ_seq_name,
            "organ_id": organ_id,
            "organ_type": organ_type,
            "run": run_idx if run_idx is not None else 1,
            "timepoint": day,
            "cd_train": cd_train,
            "cd_test": cd_test,
            "nc_train": nc_train,
            "nc_test": nc_test,
            "n_points_total": len(gt_pts),
            "n_points_train": n_train,
            "n_points_test": n_test,
            "epe": epe,
            "accs": accs,
            "accr": accr,
            "n_tips": n_tips,
        }

        # Add runtime and storage metrics
        row.update(results_metrics)

        # Add hyperparameters if provided
        if hparams:
            row.update(hparams)

        results.append(row)

    return results

def compute_cd_metrics(
    pred_pts, gt_pts, pred_is_train, gt_is_train, pred_normals=None, gt_normals=None
):
    """Helper to compute CD metrics (train, test) with optional normals."""
    # Compute CD for train/test splits if available
    cd_train = None
    cd_test = None
    nc_train = None
    nc_test = None
    n_train = None
    n_test = None

    if pred_is_train is not None and gt_is_train is not None:
        # Train points
        pred_train_mask = pred_is_train.astype(bool)
        gt_train_mask = gt_is_train.astype(bool)

        if pred_train_mask.any() and gt_train_mask.any():
            pred_train = pred_pts[pred_train_mask]
            gt_train = gt_pts[gt_train_mask]
            pred_train_normals = (
                pred_normals[pred_train_mask] if pred_normals is not None else None
            )
            gt_train_normals = (
                gt_normals[gt_train_mask] if gt_normals is not None else None
            )
            cd_train, nc_train = compute_chamfer_distance(
                pred_train, gt_train, pred_train_normals, gt_train_normals
            )
            n_train = len(gt_train)

        # Test points
        pred_test_mask = ~pred_train_mask
        gt_test_mask = ~gt_train_mask

        if pred_test_mask.any() and gt_test_mask.any():
            pred_test = pred_pts[pred_test_mask]
            gt_test = gt_pts[gt_test_mask]
            pred_test_normals = (
                pred_normals[pred_test_mask] if pred_normals is not None else None
            )
            gt_test_normals = (
                gt_normals[gt_test_mask] if gt_normals is not None else None
            )
            cd_test, nc_test = compute_chamfer_distance(
                pred_test, gt_test, pred_test_normals, gt_test_normals
            )
            n_test = len(gt_test)

    return cd_train, cd_test, nc_train, nc_test, n_train, n_test

def recompute_experiment_metrics(
    experiment_dir, verbose=True, ablation=False, interpolation=False, extrapolation=False
):
    """
    Recompute metrics for all organs in an experiment.

    Args:
        experiment_dir: Path to experiment directory
        verbose: Print progress
        ablation: Include hyperparameters from experiment_config.json
        interpolation: Process interpolation PLY files instead of normal predictions
        extrapolation: Process extrapolation PLY files (held-out final timepoints)

    Returns:
        tuple: (list of metric rows, path to sequences parent directory)
    """
    experiment_dir = Path(experiment_dir)
    all_rows = []

    # Extract hyperparameters only for ablation experiments
    hparams = extract_hyperparameters(experiment_dir) if ablation else {}

    # Find sequences directory - may be directly under experiment_dir or one level deeper
    sequences_dir = experiment_dir / "sequences"
    sequences_parent = experiment_dir
    if not sequences_dir.exists():
        # Check if there's a single timestamped subdirectory
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            sequences_dir = subdirs[0] / "sequences"
            sequences_parent = subdirs[0]
            if not sequences_dir.exists():
                if verbose:
                    print(
                        f"No sequences directory found in {experiment_dir} or {subdirs[0]}"
                    )
                return all_rows, None
        else:
            if verbose:
                print(f"No sequences directory found in {experiment_dir}")
            return all_rows, None

    sequences = [d for d in sequences_dir.iterdir() if d.is_dir()]
    if not sequences:
        if verbose:
            print(f"No sequences found in {sequences_dir}")
        return all_rows, sequences_parent

    # Load dataset once for leaf tip first-appearance checks
    from growfields.datasets.trackplant3d import TrackPlant3DDataset

    dataset_plant = TrackPlant3DDataset(
        dataset_path="data/TrackPlant3D/versions",
        version="v2",
        mode="plant",
        normalize=True,
    )

    for seq_dir in sequences:
        sequence_name = seq_dir.name

        if verbose:
            print(f"\nProcessing sequence: {sequence_name}")

        dataset_plant.set_sequence(sequence_name, verbose=False)

        # Check for run directories or direct organs directory
        run_dirs = sorted(
            [d for d in seq_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        )

        if not run_dirs:
            # Single run or no run structure
            run_dirs = [(seq_dir, None)]
        else:
            # Extract run indices
            run_dirs = [(d, int(d.name.replace("run_", ""))) for d in run_dirs]

        for run_dir, run_idx in run_dirs:
            organs_dir = run_dir / "organs"
            if not organs_dir.exists():
                continue

            if verbose and run_idx is not None:
                print(f"  Run: {run_idx}")

            # Find all organ directories
            organ_dirs = [d for d in organs_dir.iterdir() if d.is_dir()]

            for organ_dir in organ_dirs:
                # Extract organ info from directory name
                organ_name = organ_dir.name

                # Try to determine organ ID from name or existing metrics
                organ_id = None
                organ_type = organ_name

                metrics_file = organ_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            existing_metrics = json.load(f)
                            organ_id = existing_metrics.get("organ_id")
                            organ_type = existing_metrics.get("organ_type", organ_name)
                    except:
                        pass

                # Extract organ_id from name if possible
                if organ_id is None:
                    if organ_name == "stem":
                        organ_id = 0
                    elif organ_name.startswith("leaf_"):
                        organ_id = int(organ_name.split("_")[1])
                    else:
                        organ_id = -1  # Unknown

                # Recompute metrics
                if verbose:
                    print(f"    {organ_name}: recomputing metrics...")

                rows = recompute_organ_metrics(
                    organ_dir,
                    organ_id,
                    organ_type,
                    sequence_name,
                    run_idx,
                    hparams,
                    run_dir,
                    interpolation=interpolation,
                    extrapolation=extrapolation,
                    dataset_plant=dataset_plant,
                )

                if not rows:
                    if verbose:
                        print(f"    {organ_name}: no predictions found, skipping")
                    continue

                all_rows.extend(rows)

                if verbose:
                    # Use cd_test if available, otherwise cd_train
                    cd_values = [r.get("cd_test") or r.get("cd_train") for r in rows]
                    cd_values = [cd for cd in cd_values if cd is not None]
                    if cd_values:
                        mean_cd = np.mean(cd_values)
                        print(
                            f"    {organ_name}: ✓ {len(rows)} timepoints, mean_cd = {mean_cd:.6f}"
                        )
                    else:
                        print(
                            f"    {organ_name}: ✓ {len(rows)} timepoints (no CD computed)"
                        )

    return all_rows, sequences_parent

def main():
    parser = argparse.ArgumentParser(
        description="Recompute metrics from saved predictions and save to CSV"
    )
    parser.add_argument(
        "--experiment_dir", type=str, help="Path to single experiment directory"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to directory containing multiple experiments (will process all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: saves to results_dir/all_metrics.csv or experiment_dir/metrics.csv)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Include hyperparameters from experiment_config.json (for ablation experiments)",
    )
    parser.add_argument(
        "--interpolation",
        action="store_true",
        help="Compute metrics from interpolation PLY files instead of normal predictions",
    )
    parser.add_argument(
        "--extrapolation",
        action="store_true",
        help="Compute metrics from extrapolation PLY files (held-out final timepoints)",
    )

    args = parser.parse_args()

    if not args.experiment_dir and not args.results_dir:
        parser.error("Must specify either --experiment_dir or --results_dir")

    verbose = not args.quiet
    all_rows = []
    output_dir = None

    if args.experiment_dir:
        # Process single experiment
        if verbose:
            print(f"Processing experiment: {args.experiment_dir}")

        rows, output_dir = recompute_experiment_metrics(
            args.experiment_dir,
            verbose,
            ablation=args.ablation,
            interpolation=args.interpolation,
            extrapolation=args.extrapolation,
        )

        if not rows:
            print(f"\n❌ No metrics found in {args.experiment_dir}")
            return 1

        all_rows.extend(rows)

        if verbose:
            print(f"\n✅ Extracted {len(rows)} timepoint metrics")

    elif args.results_dir:
        # Process all experiments in directory
        results_dir = Path(args.results_dir)

        if not results_dir.exists():
            print(f"Error: Directory not found: {results_dir}")
            return 1

        # Find all experiment directories
        experiment_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

        if not experiment_dirs:
            print(f"No experiment directories found in {results_dir}")
            return 1

        print(f"Found {len(experiment_dirs)} experiments to process\n")

        for exp_dir in tqdm(
            experiment_dirs, desc="Processing experiments", disable=verbose
        ):
            if verbose:
                print(f"\nProcessing: {exp_dir.name}")

            rows, _ = recompute_experiment_metrics(
                exp_dir,
                verbose=verbose,
                ablation=args.ablation,
                interpolation=args.interpolation,
                extrapolation=args.extrapolation,
            )
            all_rows.extend(rows)

            if verbose and rows:
                print(f"  → {len(rows)} timepoint metrics")

        # For results_dir, save at the results_dir level
        output_dir = results_dir

    # Determine output path
    if args.extrapolation:
        csv_name = "all_metrics_extrapolation.csv"
    elif args.interpolation:
        csv_name = "all_metrics_interpolation.csv"
    else:
        csv_name = "all_metrics.csv"
    if args.output is None:
        if output_dir:
            args.output = os.path.join(output_dir, csv_name)
        else:
            args.output = csv_name  # Fallback to current directory

    # Convert to DataFrame
    if not all_rows:
        print("\n❌ No metrics found in any experiments")
        return 1

    df = pd.DataFrame(all_rows)

    # Reorder columns for better readability
    base_cols = [
        "sequence",
        "sequence_name",
        "organ_id",
        "organ_type",
        "run",
        "timepoint",
    ]

    # Hyperparameter columns (only for ablation experiments)
    hparam_cols = (
        [
            "model_type",
            "hidden_features",
            "hidden_layers",
            "lr",
            "w_global",
            "w_local",
            "finetune_lr",
            "finetune_enabled",
            # Ablation parameters
            "aligned_leaves",
        ]
        if args.ablation
        else []
    )

    # Metric columns
    metric_cols = [
        "cd_train",
        "cd_test",
        "nc_train",
        "nc_test",
        "n_points_total",
        "n_points_train",
        "n_points_test",
        "epe",
        "accs",
        "accr",
        "n_tips",
        "training_time_seconds",
        "total_model_size_mb",
    ]

    # Build column order: base + hparams + metrics
    col_order = base_cols + [c for c in hparam_cols if c in df.columns] + metric_cols

    # Only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Save to CSV
    df.to_csv(args.output, index=False)

    print(f"\n✅ Saved {len(df)} rows to {args.output}")
    print(f"\nSummary:")
    print(f"  Sequences: {df['sequence'].nunique()}")
    print(f"  Unique organ types: {df['organ_type'].nunique()}")
    print(f"  Runs: {df['run'].nunique()}")
    # Calculate actual organ instances (unique sequence-organ-run combinations)
    n_organ_instances = df.groupby(["sequence", "organ_id", "run"]).ngroups
    print(f"  Total organ instances: {n_organ_instances}")
    print(f"  Average timepoints per organ: {len(df) / n_organ_instances:.1f}")

    # Print some basic statistics
    print(f"\nMetrics (test set):")
    if df["cd_test"].notna().any():
        print(
            f"  Mean CD (test): {df['cd_test'].mean():.6f} ± {df['cd_test'].std():.6f}"
        )
    if df["epe"].notna().any():
        print(f"  Mean EPE: {df['epe'].mean():.4f} ± {df['epe'].std():.4f}")
    if df["accs"].notna().any():
        print(f"  Mean AccS: {df['accs'].mean():.2f}% ± {df['accs'].std():.2f}%")
    if df["accr"].notna().any():
        print(f"  Mean AccR: {df['accr'].mean():.2f}% ± {df['accr'].std():.2f}%")

    return 0

if __name__ == "__main__":
    exit(main())
