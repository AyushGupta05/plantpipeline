#!/usr/bin/env python3
"""
Compute full-plant metrics for a single experiment.

This script:
- Computes metrics at the full plant level (not per-organ)
- Extracts runtime and storage information from results.json
- Works on a single experiment directory
- Can optionally process just a single run for faster computation

Usage:
    # Compute metrics for all runs in an experiment
    python scripts/compute_metrics_fullplant.py --experiment_dir results/fullplant_per_organ/20260410_123456_fullplant_batch_shared_continuous_test_aligned

    # Compute metrics for only run 1 (much faster - ~5x speedup)
    python scripts/compute_metrics_fullplant.py --experiment_dir results/fullplant_per_organ/20260410_123456_fullplant_batch_shared_continuous_test_aligned --run 1

    # Specify custom output file
    python scripts/compute_metrics_fullplant.py --experiment_dir results/fullplant_per_organ/20260410_123456_fullplant_batch_shared_continuous_test_aligned --output my_metrics.csv
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

def compute_metrics_fullplant_for_run(
    run_dir, sequence_name, run_idx, hparams, verbose=True, finetune_stage="post", dataset_plant=None, interpolation=False, extrapolation=False
):
    """
    Compute full-plant metrics for a single run.

    Args:
        run_dir: Path to run directory
        sequence_name: Name of the sequence
        run_idx: Run index
        hparams: Hyperparameters dict
        verbose: Print progress
        finetune_stage: Which finetune stage to use: 'pre' or 'post'
        dataset_plant: Pre-loaded TrackPlant3DDataset (must already have set_sequence called).
            Created by compute_experiment_fullplant_metrics using the experiment's version/label_field.

    Returns:
        list: List of metric dicts, one per timepoint
    """
    predicted_dir = None
    gt_dir = None

    if extrapolation:
        full_plant_dir = run_dir / "full_plant"
        predicted_dir = full_plant_dir / "extrapolation" / "predicted"
        gt_dir = full_plant_dir / "extrapolation" / "ground_truth"
    elif interpolation:
        # Interpolation mode: look in full_plant/interpolation/
        full_plant_dir = run_dir / "full_plant"
        predicted_dir = full_plant_dir / "interpolation" / "predicted"
        gt_dir = full_plant_dir / "interpolation" / "ground_truth"
    else:
        # Check for different directory structures:
        # 1. full_plant/predicted/ and full_plant/ground_truth/ (continuous per-organ)
        # 2. full_plant/pre_finetune/predicted/ (pairwise per-organ pre-finetune)
        # 3. full_plant/post_finetune/predicted/ (pairwise per-organ post-finetune)
        # 4. pre_finetune/predicted/ (pairwise single-model pre-finetune)
        # 5. post_finetune/predicted/ (pairwise single-model post-finetune)

        # Try full_plant directory first (per-organ models)
        full_plant_dir = run_dir / "full_plant"
        if full_plant_dir.exists():
            # Try direct predicted/ground_truth (continuous)
            if (full_plant_dir / "predicted").exists() and (
                full_plant_dir / "ground_truth"
            ).exists():
                predicted_dir = full_plant_dir / "predicted"
                gt_dir = full_plant_dir / "ground_truth"
            # Try finetune stage based on parameter
            elif (
                finetune_stage == "post"
                and (full_plant_dir / "post_finetune" / "predicted").exists()
            ):
                predicted_dir = full_plant_dir / "post_finetune" / "predicted"
                gt_dir = full_plant_dir / "post_finetune" / "ground_truth"
            elif (
                finetune_stage == "pre"
                and (full_plant_dir / "pre_finetune" / "predicted").exists()
            ):
                predicted_dir = full_plant_dir / "pre_finetune" / "predicted"
                gt_dir = full_plant_dir / "pre_finetune" / "ground_truth"
            # Fallback to whichever exists
            elif (full_plant_dir / "post_finetune" / "predicted").exists():
                predicted_dir = full_plant_dir / "post_finetune" / "predicted"
                gt_dir = full_plant_dir / "post_finetune" / "ground_truth"
            elif (full_plant_dir / "pre_finetune" / "predicted").exists():
                predicted_dir = full_plant_dir / "pre_finetune" / "predicted"
                gt_dir = full_plant_dir / "pre_finetune" / "ground_truth"

        # Try direct pre/post_finetune directories (single-model pairwise)
        if predicted_dir is None:
            if (
                finetune_stage == "post"
                and (run_dir / "post_finetune" / "predicted").exists()
            ):
                predicted_dir = run_dir / "post_finetune" / "predicted"
                gt_dir = run_dir / "post_finetune" / "ground_truth"
            elif (
                finetune_stage == "pre"
                and (run_dir / "pre_finetune" / "predicted").exists()
            ):
                predicted_dir = run_dir / "pre_finetune" / "predicted"
                gt_dir = run_dir / "pre_finetune" / "ground_truth"
            # Fallback to whichever exists
            elif (run_dir / "post_finetune" / "predicted").exists():
                predicted_dir = run_dir / "post_finetune" / "predicted"
                gt_dir = run_dir / "post_finetune" / "ground_truth"
            elif (run_dir / "pre_finetune" / "predicted").exists():
                predicted_dir = run_dir / "pre_finetune" / "predicted"
                gt_dir = run_dir / "pre_finetune" / "ground_truth"

    if (
        predicted_dir is None
        or gt_dir is None
        or not predicted_dir.exists()
        or not gt_dir.exists()
    ):
        if verbose:
            print(f"    No valid predicted/ground_truth directories found in {run_dir}")
        return []

    # Load prediction files
    pred_files = sorted([f for f in os.listdir(predicted_dir) if f.endswith(".ply")])

    if not pred_files:
        if verbose:
            print(f"    No prediction files found in {predicted_dir}")
        return []

    # Extract day numbers from filenames
    # Format: pred_{sequence}_D{day}.ply or pred_{sequence}_day{day}.ply
    day_pattern = re.compile(r"_[Dd](?:ay)?(\d+)\.ply")

    timepoints = []
    for f in pred_files:
        match = day_pattern.search(f)
        if match:
            timepoints.append(int(match.group(1)))

    if not timepoints:
        if verbose:
            print(f"    Could not extract timepoints from filenames")
        return []

    timepoints = sorted(set(timepoints))
    results = []

    # Load runtime and storage metrics
    results_metrics = load_results_json(run_dir)

    # Load GT leaf tips directly from dataset to determine true first appearance
    # (PLY files start from timestep 1, so we need to check timestep 0 in the dataset)
    organ_first_appearance_frame = {}  # organ_id -> first_frame_index
    frame_to_day = {}  # frame_index -> day_number

    gt_tips = dataset_plant.get_leaf_tips(normalized=False, as_tensors=True)
    gt_tip_labels = dataset_plant.get_leaf_tip_labels(as_tensors=True)

    # Build GLOBAL frame to day mapping from plant dataset timepoint names
    plant_timepoint_names = dataset_plant.get_timepoint_names()
    for i, name in enumerate(plant_timepoint_names):
        # Extract day number from name (e.g., "D02" -> 2)
        match = re.search(r"D(\d+)", name)
        if match:
            frame_to_day[i] = int(match.group(1))

    # Track first appearance by frame index
    for t, (tips, labels) in enumerate(zip(gt_tips, gt_tip_labels)):
        if tips is not None:
            for label in labels:
                organ_id = label.item()
                if organ_id not in organ_first_appearance_frame:
                    organ_first_appearance_frame[organ_id] = t

    if verbose:
        print(
            f"  First appearance per organ (frame index): {organ_first_appearance_frame}"
        )
        print(f"  Frame to day mapping: {frame_to_day}")

    for day in timepoints:
        # Find matching files for this day
        # Try standard format _D{day:02d}.ply first, fall back to legacy _day{day:03d}.ply
        pred_file = next((f for f in pred_files if f"_D{day:02d}.ply" in f), None)
        if pred_file is None:
            # Fall back to legacy format
            pred_file = next((f for f in pred_files if f"_day{day:03d}.ply" in f), None)

        if pred_file is None:
            continue

        # Construct GT filename
        gt_file = pred_file.replace("pred_", "gt_")

        pred_path = predicted_dir / pred_file
        gt_path = gt_dir / gt_file

        if not gt_path.exists():
            if verbose:
                print(f"    Warning: GT file not found: {gt_path}")
            continue

        # Load point clouds
        pred_ply = load_ply(pred_path)
        pred_pts = pred_ply["points"]
        pred_is_train = pred_ply.get("is_train")
        pred_is_leaf_tip = pred_ply.get("is_leaf_tip")
        pred_normals = pred_ply.get("normals")
        pred_tip_organ_label = pred_ply.get("tip_organ_label")

        gt_ply = load_ply(gt_path)
        gt_pts = gt_ply["points"]
        gt_is_train = gt_ply.get("is_train")
        gt_is_leaf_tip = gt_ply.get("is_leaf_tip")
        gt_normals = gt_ply.get("normals")
        gt_tip_organ_label = gt_ply.get("tip_organ_label")

        if pred_pts is None or gt_pts is None:
            if verbose:
                print(f"    Warning: Failed to load PLY files for day {day}")
            continue

        # Compute Chamfer Distance (full plant)
        cd, nc = compute_chamfer_distance(pred_pts, gt_pts, pred_normals, gt_normals)

        # Split by train/test if available
        cd_train, cd_test, nc_train, nc_test = None, None, None, None
        n_train, n_test = None, None

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

        # Compute EPE and AccS/AccR for leaf tips
        epe, accs, accr, n_tips = None, None, None, None
        if pred_is_leaf_tip is not None and gt_is_leaf_tip is not None:
            pred_tip_mask = pred_is_leaf_tip.astype(bool)
            gt_tip_mask = gt_is_leaf_tip.astype(bool)

            if pred_tip_mask.any() and gt_tip_mask.any():
                pred_tips_all = pred_pts[pred_tip_mask]
                gt_tips_all = gt_pts[gt_tip_mask]

                # Extract organ labels for tips
                pred_tip_labels = (
                    pred_tip_organ_label[pred_tip_mask]
                    if pred_tip_organ_label is not None
                    else None
                )
                gt_tip_labels = (
                    gt_tip_organ_label[gt_tip_mask]
                    if gt_tip_organ_label is not None
                    else None
                )

                # Match tips by organ_id - only compute metrics for organs present in both
                if pred_tip_labels is not None and gt_tip_labels is not None:
                    # Find common organ IDs
                    pred_organ_ids = set(pred_tip_labels)
                    gt_organ_ids = set(gt_tip_labels)
                    common_organ_ids = pred_organ_ids & gt_organ_ids

                    if common_organ_ids:
                        # Filter tips to only include common organs
                        # Skip a tip on the frame it first appears: there is no
                        # previous frame to have predicted it from, so no prediction
                        # exists to score.

                        # Map current day back to frame index
                        day_to_frame = {v: k for k, v in frame_to_day.items()}
                        current_frame = day_to_frame.get(day)

                        pred_tips_matched = []
                        gt_tips_matched = []
                        for organ_id in sorted(common_organ_ids):
                            # Skip if this is the first appearance
                            if (
                                current_frame is not None
                                and organ_id in organ_first_appearance_frame
                            ):
                                if (
                                    current_frame
                                    == organ_first_appearance_frame[organ_id]
                                ):
                                    continue

                            # Find pred tip with this organ_id
                            pred_idx = np.where(pred_tip_labels == organ_id)[0]
                            gt_idx = np.where(gt_tip_labels == organ_id)[0]

                            if len(pred_idx) > 0 and len(gt_idx) > 0:
                                pred_tips_matched.append(pred_tips_all[pred_idx[0]])
                                gt_tips_matched.append(gt_tips_all[gt_idx[0]])

                        if pred_tips_matched:
                            pred_tips = np.array(pred_tips_matched)
                            gt_tips = np.array(gt_tips_matched)

                            # Debug: log if there were unmatched tips
                            n_pred_total = len(pred_tips_all)
                            n_gt_total = len(gt_tips_all)
                            n_matched = len(pred_tips)
                            if n_pred_total != n_matched or n_gt_total != n_matched:
                                print(
                                    f"  Day {day}: Matched {n_matched} tips (pred had {n_pred_total}, gt had {n_gt_total})"
                                )

                            epe = compute_epe(pred_tips, gt_tips)
                            n_tips = len(gt_tips)

                            # Compute AccS and AccR for leaf tips
                            accs, accr, _ = compute_scene_flow_metrics(
                                pred_tips, gt_tips
                            )
                    else:
                        print(
                            f"  Day {day}: No common organ IDs - pred: {pred_organ_ids}, gt: {gt_organ_ids}"
                        )
                        n_tips = 0  # Tips exist but no matches
                else:
                    # Fallback: no organ labels, use direct matching (old behavior)
                    print(
                        f"  Day {day}: Warning - no tip organ labels, skipping metrics"
                    )
                    if len(pred_tips_all) != len(gt_tips_all):
                        print(
                            f"  Day {day}: Tip count mismatch - pred: {len(pred_tips_all)}, gt: {len(gt_tips_all)}"
                        )

        row = {
            "sequence": sequence_name,
            "run": run_idx,
            "timepoint": day,
            "cd": cd,
            "cd_train": cd_train,
            "cd_test": cd_test,
            "nc": nc,
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

        # Add hyperparameters
        if hparams:
            row.update(hparams)

        results.append(row)

    return results

def _has_direct_predictions(experiment_dir, finetune_stage="post"):
    """
    Check if experiment_dir has direct pre/post_finetune prediction directories.

    This handles the case where there's no sequences/ directory structure,
    typically in debugging runs or single-sequence experiments.

    Args:
        experiment_dir: Path to experiment directory
        finetune_stage: Which finetune stage to check for

    Returns:
        bool: True if direct prediction directories exist
    """
    experiment_dir = Path(experiment_dir)

    # Check for direct pre_finetune or post_finetune directories
    if finetune_stage == "pre":
        return (experiment_dir / "pre_finetune" / "predicted").exists()
    elif finetune_stage == "post":
        return (experiment_dir / "post_finetune" / "predicted").exists()

    return False

def _process_run_and_report(
    run_dir, sequence_name, run_idx, hparams, verbose, finetune_stage, dataset_plant=None, interpolation=False, extrapolation=False
):
    """
    Process a single run and print progress report.

    Args:
        run_dir: Path to run directory
        sequence_name: Name of the sequence
        run_idx: Run index
        hparams: Hyperparameters dict
        verbose: Print progress
        finetune_stage: Which finetune stage to use
        dataset_plant: Pre-loaded TrackPlant3DDataset with sequence already set
        interpolation: Compute metrics from interpolation PLY files

    Returns:
        list: List of metric dicts
    """
    if verbose:
        print(f"  Run {run_idx}...")

    rows = compute_metrics_fullplant_for_run(
        run_dir,
        sequence_name,
        run_idx,
        hparams,
        verbose=False,
        finetune_stage=finetune_stage,
        dataset_plant=dataset_plant,
        interpolation=interpolation,
        extrapolation=extrapolation,
    )

    if rows:
        if verbose:
            mean_cd = np.mean(
                [
                    r["cd_test"] or r["cd"]
                    for r in rows
                    if r.get("cd_test") or r.get("cd")
                ]
            )
            print(f"    ✓ {len(rows)} timepoints, mean_cd = {mean_cd:.6f}")
    else:
        if verbose:
            print(f"    ✗ No metrics computed")

    return rows

def _process_single_sequence_run(
    experiment_dir, hparams, verbose, run_filter, finetune_stage, dataset_plant=None, interpolation=False, extrapolation=False
):
    """
    Process a single sequence/run without the standard sequences/ directory structure.

    This handles debugging runs where predictions are directly in:
    experiment_dir/pre_finetune/predicted/ or experiment_dir/post_finetune/predicted/

    Args:
        experiment_dir: Path to experiment directory
        hparams: Hyperparameters dict
        verbose: Print progress
        run_filter: Optional run number to process
        finetune_stage: Which finetune stage to use
        dataset_plant: Pre-loaded TrackPlant3DDataset (will set_sequence internally)
        interpolation: Compute metrics from interpolation PLY files

    Returns:
        list: List of metric dicts
    """
    experiment_dir = Path(experiment_dir)

    # Extract sequence name from directory name
    sequence_name = experiment_dir.name

    # For debugging runs, we typically have a single run (run 1)
    run_idx = 1

    # Skip if run_filter is set and doesn't match
    if run_filter is not None and run_idx != run_filter:
        return []

    if verbose:
        print(f"\nProcessing sequence: {sequence_name}")

    if dataset_plant is not None:
        dataset_plant.set_sequence(sequence_name, verbose=False)

    rows = _process_run_and_report(
        experiment_dir, sequence_name, run_idx, hparams, verbose, finetune_stage, dataset_plant=dataset_plant, interpolation=interpolation, extrapolation=extrapolation
    )
    return rows if rows else []

def compute_experiment_fullplant_metrics(
    experiment_dir, verbose=True, run_filter=None, finetune_stage="post", ablation=False, interpolation=False, extrapolation=False
):
    """
    Compute full-plant metrics for all sequences and runs in an experiment.

    Args:
        experiment_dir: Path to experiment directory
        verbose: Print progress
        run_filter: Optional run number to process (None = process all runs)
        finetune_stage: Which finetune stage to use: 'pre' or 'post'
        ablation: Include hyperparameters from experiment_config.json
        interpolation: Compute metrics from interpolation PLY files

    Returns:
        list: List of metric dicts
    """
    experiment_dir = Path(experiment_dir)
    all_rows = []

    # Extract hyperparameters only for ablation experiments
    hparams = extract_hyperparameters(experiment_dir) if ablation else {}

    # Load dataset once for leaf tip first-appearance checks.
    # Read version and label_field from the experiment config so that runs
    # trained on auto-seg labels (predicted_labels) use the correct organ IDs.
    from growfields.datasets.trackplant3d import TrackPlant3DDataset

    _exp_config_path = experiment_dir / "experiment_config.json"
    if not _exp_config_path.exists():
        _subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if len(_subdirs) == 1:
            _exp_config_path = _subdirs[0] / "experiment_config.json"
    if not _exp_config_path.exists():
        raise FileNotFoundError(
            f"No experiment_config.json found under {experiment_dir}. It records "
            "which dataset the run was trained on; refusing to guess, since a wrong "
            "version or path would silently produce metrics against the wrong data."
        )
    with open(_exp_config_path) as _f:
        _exp_cfg = json.load(_f)
    _ds_cfg = _exp_cfg.get("config", {}).get("dataset", {})

    # Dataset identity comes from the run's own config -- never defaulted.
    _dataset_path = _ds_cfg["dataset_path"]
    _dataset_version = _ds_cfg["version"]
    # label_field is an optional schema field; "labels" is the correct default.
    _label_field = _ds_cfg.get("label_field", "labels")

    dataset_plant = TrackPlant3DDataset(
        dataset_path=_dataset_path,
        version=_dataset_version,
        mode="plant",
        normalize=True,
        label_field=_label_field,
    )

    # Find sequences directory
    sequences_dir = experiment_dir / "sequences"
    if not sequences_dir.exists():
        # Check if there's a single timestamped subdirectory
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            sequences_dir = subdirs[0] / "sequences"
            if not sequences_dir.exists():
                # Check if this is a single sequence/run setup with direct pre/post_finetune directories
                if _has_direct_predictions(experiment_dir, finetune_stage):
                    if verbose:
                        print(
                            f"Processing single sequence/run without sequences directory structure"
                        )
                    return _process_single_sequence_run(
                        experiment_dir, hparams, verbose, run_filter, finetune_stage, dataset_plant=dataset_plant, interpolation=interpolation, extrapolation=extrapolation
                    )
                if verbose:
                    print(f"No sequences directory found in {experiment_dir}")
                return all_rows
        else:
            # Check if this is a single sequence/run setup with direct pre/post_finetune directories
            if _has_direct_predictions(experiment_dir, finetune_stage):
                if verbose:
                    print(
                        f"Processing single sequence/run without sequences directory structure"
                    )
                return _process_single_sequence_run(
                    experiment_dir, hparams, verbose, run_filter, finetune_stage, dataset_plant=dataset_plant, interpolation=interpolation, extrapolation=extrapolation
                )
            if verbose:
                print(f"No sequences directory found in {experiment_dir}")
            return all_rows

    sequences = [d for d in sequences_dir.iterdir() if d.is_dir()]
    if not sequences:
        if verbose:
            print(f"No sequences found in {sequences_dir}")
        return all_rows

    for seq_dir in tqdm(sequences, desc="Processing sequences", disable=not verbose):
        sequence_name = seq_dir.name

        if verbose:
            print(f"\nProcessing sequence: {sequence_name}")

        dataset_plant.set_sequence(sequence_name, verbose=False)

        # Check for run directories
        run_dirs = [
            d for d in seq_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
        ]

        if not run_dirs:
            # Single run or no run structure
            # Check if there's a timestamped subdirectory (single-model batch structure)
            timestamped_dirs = [
                d
                for d in seq_dir.iterdir()
                if d.is_dir() and not d.name.startswith("run_")
            ]
            if len(timestamped_dirs) == 1:
                # Use the timestamped subdirectory as the run directory
                run_dirs = [(timestamped_dirs[0], 1)]
            else:
                # Use seq_dir itself
                run_dirs = [(seq_dir, 1)]
        else:
            # Extract run indices
            run_dirs = [(d, int(d.name.replace("run_", ""))) for d in run_dirs]

        for run_dir, run_idx in sorted(run_dirs, key=lambda x: x[1]):
            # Skip this run if run_filter is set and doesn't match
            if run_filter is not None and run_idx != run_filter:
                continue

            rows = _process_run_and_report(
                run_dir, sequence_name, run_idx, hparams, verbose, finetune_stage, dataset_plant=dataset_plant, interpolation=interpolation, extrapolation=extrapolation
            )
            if rows:
                all_rows.extend(rows)

    return all_rows

def main():
    parser = argparse.ArgumentParser(
        description="Compute full-plant metrics for a single experiment"
    )
    parser.add_argument(
        "--experiment_dir", type=str, required=True, help="Path to experiment directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: fullplant_metrics.csv in experiment_dir)",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Process only a specific run number (default: process all runs)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--ablation", action="store_true",
        help="Include hyperparameters from experiment_config.json (for ablation experiments)")
    parser.add_argument(
        "--finetune_stage",
        type=str,
        choices=["pre", "post", "both"],
        default="post",
        help="Which finetune stage to process: 'pre', 'post', or 'both' (default: post)",
    )
    parser.add_argument(
        "--interpolation",
        action="store_true",
        help="Compute metrics from interpolation PLY files instead of normal predictions",
    )
    parser.add_argument(
        "--extrapolation",
        action="store_true",
        help="Compute metrics from extrapolation PLY files (full_plant/extrapolation/)",
    )

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print(f"Processing experiment: {args.experiment_dir}")
        if args.extrapolation:
            print(f"Mode: EXTRAPOLATION (held-out last frames)")
        elif args.interpolation:
            print(f"Mode: INTERPOLATION (unseen frames)")
        if args.run is not None:
            print(f"Processing only run {args.run}")
        else:
            print(f"Processing all runs")
        print(f"Finetune stage: {args.finetune_stage}\n")

    # Handle "both" finetune stages
    all_rows = []
    if args.finetune_stage == "both":
        for stage in ["pre", "post"]:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing {stage}_finetune stage")
                print(f"{'='*60}")
            rows = compute_experiment_fullplant_metrics(
                args.experiment_dir,
                verbose=verbose,
                run_filter=args.run,
                finetune_stage=stage,
                ablation=args.ablation,
                interpolation=args.interpolation,
                extrapolation=args.extrapolation,
            )
            # Add finetune_stage column
            for row in rows:
                row["finetune_stage"] = stage
            all_rows.extend(rows)
    else:
        all_rows = compute_experiment_fullplant_metrics(
            args.experiment_dir,
            verbose=verbose,
            run_filter=args.run,
            finetune_stage=args.finetune_stage,
            ablation=args.ablation,
            interpolation=args.interpolation,
            extrapolation=args.extrapolation,
        )
        # Add finetune_stage column
        for row in all_rows:
            row["finetune_stage"] = args.finetune_stage

    if not all_rows:
        print(f"\n❌ No metrics found in {args.experiment_dir}")
        return 1

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    # Reorder columns for better readability
    base_cols = ["sequence", "run", "timepoint", "finetune_stage"]

    # Metric columns
    metric_cols = [
        "cd",
        "cd_train",
        "cd_test",
        "nc",
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

    # Hyperparameter columns (only for ablation experiments)
    hparam_cols = [
        "model_type",
        "hidden_features",
        "hidden_layers",
        "lr",
        "w_global",
        "w_local",
        "finetune_lr",
        "finetune_enabled",
        "aligned_leaves",
    ] if args.ablation else []

    # Build column order
    col_order = base_cols + metric_cols + [c for c in hparam_cols if c in df.columns]

    # Only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Determine output path
    if args.output is None:
        if args.extrapolation:
            filename = "fullplant_metrics_extrapolation.csv"
        elif args.interpolation:
            filename = "fullplant_metrics_interpolation.csv"
        else:
            filename = "fullplant_metrics.csv"
        args.output = os.path.join(args.experiment_dir, filename)

    # Save to CSV
    df.to_csv(args.output, index=False)

    print(f"\n✅ Saved {len(df)} rows to {args.output}")
    print(f"\nSummary:")
    print(f"  Sequences: {df['sequence'].nunique()}")
    print(f"  Runs: {df['run'].nunique()}")
    print(f"  Total timepoints: {len(df)}")

    # Print basic statistics
    print(f"\nMetrics (full plant):")
    if df["cd"].notna().any():
        print(f"  Mean CD (all): {df['cd'].mean():.6f} ± {df['cd'].std():.6f}")
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

    # Runtime and storage
    if "training_time_seconds" in df.columns and df["training_time_seconds"].notna().any():
        print(f"\nRuntime and Storage:")
        print(
            f"  Mean training time: {df['training_time_seconds'].mean():.1f}s ({df['training_time_seconds'].mean()/60:.1f} min)"
        )
        if "total_model_size_mb" in df.columns and df["total_model_size_mb"].notna().any():
            print(f"  Mean model size: {df['total_model_size_mb'].mean():.2f} MB")

    return 0

if __name__ == "__main__":
    exit(main())
