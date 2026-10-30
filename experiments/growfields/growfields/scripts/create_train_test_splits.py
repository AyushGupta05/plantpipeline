#!/usr/bin/env python3
"""
Create fixed train/test splits for plant sequences.

Usage:
    # Single sequence
    python scripts/create_train_test_splits.py \
        --sequence maize_control_plant1 \
        --train_ratio 0.5 \
        --seed 42

    # Multiple sequences from config
    python scripts/create_train_test_splits.py \
        --split val \
        --train_ratio 0.5 \
        --seed 42

    # All test sequences
    python scripts/create_train_test_splits.py \
        --split test \
        --train_ratio 0.5 \
        --seed 42
"""

import argparse
import os

import numpy as np
import torch

from growfields.datasets.trackplant3d import TrackPlant3DDataset
from growfields.pipeline.splits import load_sequence_split, new_split, write_split

def create_split_for_sequence(
    sequence_name,
    dataset_path,
    version="v2",
    train_ratio=0.5,
    seed=42,
    output_dir="configs/datasets/TrackPlant3D/point_splits",
):
    """
    Create and save train/test split for a single sequence.

    Args:
        sequence_name: Name of the sequence (e.g., "maize_control_plant1")
        dataset_path: Path to TrackPlant3D dataset
        version: Dataset version
        train_ratio: Ratio of training points (default: 0.5 = 50/50)
        seed: Random seed for split
        output_dir: Directory to save split JSON files

    Returns:
        Path to saved split file
    """
    print(f"\n{'='*80}")
    print(f"Creating split for: {sequence_name}")
    print(f"{'='*80}")

    # Load dataset (just to get the points, no split yet)
    print(f"  Loading dataset...")
    dataset = TrackPlant3DDataset(
        dataset_path=dataset_path,
        sequence_name=sequence_name,
        version=version,
        mode="plant",
        train_ratio=1.0,  # Load all points first
        aligned_leaves=False,
        normalize=False,
        device=torch.device("cpu"),  # Use CPU for preprocessing
    )

    # Initialize RNG
    rng = np.random.RandomState(seed)

    # Create split for each timepoint
    split_data = new_split(
        sequence_name=sequence_name,
        seed=seed,
        train_ratio=train_ratio,
        version=version,
        n_timepoints=dataset.nt,
    )

    # Get all points (train_pts only since we loaded with train_ratio=1.0)
    all_pts = [pts.cpu().numpy() for pts in dataset.train_pts]
    all_labels = [labels.cpu().numpy() for labels in dataset.train_labels]

    # Get leaf tips if available
    has_tips = (
        hasattr(dataset, "leaf_tips_original")
        and dataset.leaf_tips_original is not None
    )

    print(f"  Processing {dataset.nt} timepoints...")

    for t in range(dataset.nt):
        pts = all_pts[t]
        labels = all_labels[t]
        n_pts = len(pts)
        n_train = int(n_pts * train_ratio)

        # Handle edge cases
        if train_ratio >= 1.0:
            train_indices = np.arange(n_pts)
            test_indices = np.array([], dtype=int)
        elif train_ratio <= 0.0:
            train_indices = np.array([], dtype=int)
            test_indices = np.arange(n_pts)
        else:
            # Random permutation
            indices = rng.permutation(n_pts)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            # CRITICAL: Ensure leaf tips are in test set
            tips_moved = 0
            if has_tips and t < len(dataset.leaf_tips_original):
                tips_t = dataset.leaf_tips_original[t]
                if tips_t is not None:
                    # Handle both single tip and multiple tips
                    if isinstance(tips_t, list):
                        tip_coords_list = tips_t
                    elif tips_t.ndim == 1:  # Single tip [3]
                        tip_coords_list = [tips_t]
                    else:  # Multiple tips [K, 3]
                        tip_coords_list = [tips_t[i] for i in range(len(tips_t))]

                    # Move any tips in train to test
                    train_set = set(train_indices.tolist())
                    test_set = set(test_indices.tolist())

                    for tip_coords in tip_coords_list:
                        distances = np.linalg.norm(pts - tip_coords, axis=1)
                        tip_idx = np.argmin(distances)

                        if tip_idx in train_set:
                            train_set.remove(tip_idx)
                            test_set.add(tip_idx)
                            tips_moved += 1

                    train_indices = np.array(sorted(train_set))
                    test_indices = np.array(sorted(test_set))

            if tips_moved > 0:
                print(f"    Timepoint {t}: Moved {tips_moved} leaf tip(s) to test set")

        # Count points per organ
        organ_counts_train = {}
        organ_counts_test = {}
        for organ_id in np.unique(labels):
            organ_counts_train[int(organ_id)] = int(
                np.sum(labels[train_indices] == organ_id)
            )
            organ_counts_test[int(organ_id)] = int(
                np.sum(labels[test_indices] == organ_id)
            )

        split_data["timepoints"].append(
            {
                "timepoint_idx": t,
                "n_total_points": int(n_pts),
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "organ_counts_train": organ_counts_train,
                "organ_counts_test": organ_counts_test,
            }
        )

        actual_ratio = len(train_indices) / n_pts if n_pts > 0 else 0
        print(
            f"    Timepoint {t}: {len(train_indices)} train, {len(test_indices)} test "
            f"(ratio: {actual_ratio:.3f})"
        )

    # Save to JSON
    output_file = write_split(
        split_data, os.path.join(output_dir, f"{sequence_name}_split.json")
    )

    print(f"\n✅ Split saved to: {output_file}")

    # Summary statistics
    total_train = sum(tp["n_train"] for tp in split_data["timepoints"])
    total_test = sum(tp["n_test"] for tp in split_data["timepoints"])
    avg_ratio = (
        total_train / (total_train + total_test)
        if (total_train + total_test) > 0
        else 0
    )
    print(
        f"   Total: {total_train} train, {total_test} test points (avg ratio: {avg_ratio:.3f})"
    )

    return output_file

def main():
    parser = argparse.ArgumentParser(
        description="Create fixed train/test splits for plant sequences"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Single sequence name (e.g., 'maize_control_plant1')",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        help="Create splits for all sequences in val or test split from config",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/TrackPlant3D/versions",
        help="Path to TrackPlant3D dataset",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Ratio of training points (default: 0.5 = 50/50 split)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for split (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs/datasets/TrackPlant3D/point_splits",
        help="Directory to save split JSON files",
    )

    args = parser.parse_args()

    if args.sequence is None and args.split is None:
        parser.error("Must specify either --sequence or --split")

    sequences = []

    if args.sequence:
        sequences = [args.sequence]
    elif args.split:
        try:
            sequences, _ = load_sequence_split(args.split)
        except (FileNotFoundError, KeyError) as exc:
            print(f"Error: {exc}")
            return
        print(f"Found {len(sequences)} sequences in '{args.split}' split")

    # Create splits for all sequences
    print(f"\n{'='*80}")
    print(f"Creating splits with:")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")
    print(f"{'='*80}")

    created_files = []
    failed_sequences = []

    for seq in sequences:
        try:
            output_file = create_split_for_sequence(
                sequence_name=seq,
                dataset_path=args.dataset_path,
                train_ratio=args.train_ratio,
                seed=args.seed,
                output_dir=args.output_dir,
            )
            created_files.append(output_file)
        except Exception as e:
            print(f"\n❌ Error creating split for {seq}: {e}")
            import traceback

            traceback.print_exc()
            failed_sequences.append(seq)
            continue

    print(f"\n{'='*80}")
    print(f"✅ Successfully created {len(created_files)}/{len(sequences)} split files")
    if failed_sequences:
        print(f"❌ Failed: {failed_sequences}")
    print(f"   Output directory: {args.output_dir}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
