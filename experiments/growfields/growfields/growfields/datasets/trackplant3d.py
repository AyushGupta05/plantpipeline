"""
Unified TrackPlant3D Dataset for both individual leaves and full plants.

This module provides a single dataset class that can operate in two modes:
- mode='leaf': Individual leaf sequences (aligned)
- mode='plant': Full plant sequences with organ labels

No resampling is performed - point clouds are returned at their native resolution.
Point sampling is done during training (see models/siren_dynamics.py).
"""

import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from growfields.pipeline.splits import validate_split

try:
    from plant_shape_analysis import LeafSequencesDataset, PlantSequencesDataset
except ImportError:
    raise ImportError(
        "plant-shape-analysis is required to load the TrackPlant3D dataset. "
        "Install it with:\n"
        "  pip install -r requirements.txt\n"
        "which pins the exact commit this release was tested against "
        "(https://github.com/Joaquin-Gajardo/plant-shape-analysis)."
    )


class TrackPlant3DDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        sequence_name=None,
        version="v2",
        mode="leaf",  # 'leaf' or 'plant'
        train_ratio=0.25,
        split_file=None,  # Path to a pre-computed train/test point split JSON
        aligned_leaves=None,  # canonicalise each leaf; None = on in leaf mode, off in plant
        label_field="labels",  # Which timepoint key to use as organ labels ("labels" for GT, "predicted_labels" for auto-seg)
        min_timepoints=1,
        max_timepoints=None,
        timepoint_stride=1,
        timepoint_indices=None,
        auto_download=True,
        normalize=True,
        exclude_sequences=None,
        device=(
            torch.device("cuda", index=0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        ),
    ):
        """
        Unified TrackPlant3D Dataset for both leaf and full plant sequences.

        Args:
            dataset_path: Path to TrackPlant3D dataset
            sequence_name: Name of sequence to load:
                - For mode='leaf': 'maize_control_plant1_leaf2'
                - For mode='plant': 'maize_control_plant1'
                If None, dataset is loaded but no sequence is selected.
            version: Dataset version (default: 'v2' - pre-aligned with corrected normals)
            mode: Dataset mode:
                - 'leaf': Individual leaf sequences (default)
                - 'plant': Full plant sequences with organ labels
            train_ratio: Ratio of points for training vs testing (default: 0.25 = 25% train, 75% test)
            split_file: Path to pre-computed split JSON file (optional). If provided, loads
                fixed train/test split instead of computing randomly.
            aligned_leaves: If True, canonicalise each leaf by PCA alignment, so organs
                are learnt in canonical leaf space instead of plant space. This is the
                paper's canonicalisation and `aligned_leaves=false` is its ablation.
                None (default) canonicalises in mode='leaf' and not in mode='plant',
                matching the defaults of the two flags this replaces.
            min_timepoints: Minimum number of timepoints for sequences
            max_timepoints: Maximum number of timepoints to use (default: None, use all)
            timepoint_stride: Stride for selecting timepoints (default: 1, use every frame)
            timepoint_indices: List of specific timepoint indices (overrides stride/max)
            auto_download: If True, automatically download dataset if not found
            normalize: If True, normalize point clouds globally across all timepoints
                - Normalization maps all points to [-1, 1] range
                - Use normalize=True for single-model training (leaf or plant)
                - Use normalize=False for per-organ training (OrganModelManager handles it)
            exclude_sequences: Optional list of plant sequence names to exclude entirely
                (e.g. corrupted or otherwise unusable sequences) from the loaded dataset.
            device: Torch device for tensors

        Usage Patterns:
            1. Single leaf training:
                TrackPlant3DDataset(mode='leaf', normalize=True)

            2. Single model on full plant:
                TrackPlant3DDataset(mode='plant', normalize=True)

            3. Per-organ models (Phase 2):
                TrackPlant3DDataset(mode='plant', normalize=False)
                # OrganModelManager will handle per-organ normalization

        Note:
            - Version v2 is pre-aligned and has corrected normals
            - No resampling is done - returns full point clouds
            - Point sampling happens during training (see models/siren_dynamics.py)
            - For plant mode, organ labels are available in train_labels/test_labels
            - Access underlying datasets via self.leaf_dataset and self.plant_dataset
        """
        self.dataset_path = Path(dataset_path)
        self.sequence_name = sequence_name
        self.version = version
        self.mode = mode
        self.train_ratio = train_ratio
        self.split_file = split_file
        self.label_field = label_field
        self._warn_if_label_field_contradicts_version(version, label_field)
        # One switch for both modes. LeafSequencesDataset computes the per-leaf transforms
        # that plant mode consumes via _process_aligned_leaves, so it gets the same value.
        if aligned_leaves is None:
            aligned_leaves = mode == "leaf"
        self.aligned_leaves = aligned_leaves
        self.max_timepoints = max_timepoints
        self.timepoint_stride = timepoint_stride
        self.timepoint_indices = timepoint_indices
        self.normalize = normalize
        self.exclude_sequences = exclude_sequences
        self.device = device

        # Validate mode
        if mode not in ["leaf", "plant"]:
            raise ValueError(f"mode must be 'leaf' or 'plant', got '{mode}'")

        # Load LeafSequencesDataset (this also loads PlantSequencesDataset internally)
        print(
            f"Loading LeafSequencesDataset from {dataset_path} (version: {version})..."
        )
        # For plant mode override min_timepoints to include all leaves
        leaf_min_timepoints = 1 if mode == "plant" else min_timepoints

        self.leaf_dataset = LeafSequencesDataset(
            dataset_path=str(dataset_path),
            version=version,
            min_timepoints=leaf_min_timepoints,
            use_ply=True,
            apply_alignment=self.aligned_leaves,
            plant_alignment_method=None,  # v2 is pre-aligned
            estimate_normals=False,  # v2 has normals
            auto_download=auto_download,
            exclude_sequences=exclude_sequences,
        )

        # Access plant dataset (already loaded inside leaf_dataset)
        self.plant_dataset = self.leaf_dataset.plant_dataset

        if mode == "leaf":
            print(
                f"✓ Loaded dataset with {len(self.leaf_dataset.leaf_timeseries)} leaf sequences"
            )
        else:  # plant
            print(
                f"✓ Loaded dataset with {len(self.plant_dataset.plant_timeseries)} plant sequences"
            )

        # Initialize sequence data
        self.sequence_data = None
        self.train_pts = None
        self.test_pts = None
        self.train_normals = None
        self.test_normals = None
        self.train_labels = None  # Only for plant mode
        self.test_labels = None  # Only for plant mode
        self.leaf_transformations = (
            {}
        )  # Stores transformation matrices for aligned leaves (organ_id -> transformations)
        self.leaf_transformations_full = (
            {}
        )  # Full transformations for all days (for interpolation evaluation)
        self.leaf_transformation_days = (
            {}
        )  # Maps organ_id -> list of days corresponding to each transformation

        # If sequence_name is provided, select it
        if sequence_name is not None:
            success = self.set_sequence(sequence_name, verbose=True)
            if not success:
                available_sequences = self.get_available_sequences()
                raise ValueError(
                    f"Sequence '{sequence_name}' not found in dataset.\n"
                    f"Available sequences: {[s['name'] for s in available_sequences[:10]]}...\n"
                    f"Total: {len(available_sequences)} sequences"
                )

    def _process_sequence(self):
        """Process sequence data into train/test tensors."""
        if self.mode == "leaf":
            self._process_leaf_sequence()
        else:
            self._process_plant_sequence()

    def _process_leaf_sequence(self):
        """Process individual leaf sequence."""
        timepoints = self.sequence_data["timepoints"]

        # Select timepoints
        timepoints = self._select_timepoints(timepoints)
        self.nt = len(timepoints)
        print(f"  Final number of timepoints: {self.nt}")

        # Update sequence_data to reflect filtered timepoints (important for day extraction)
        self.sequence_data["timepoints"] = timepoints

        # Collect all point clouds
        all_pts = []
        all_normals = []

        for tp in timepoints:
            points = tp["points"]
            normals = tp.get("normals")

            all_pts.append(points)
            if normals is not None:
                all_normals.append(normals)
            else:
                # Fallback: random normals
                normals_random = np.random.randn(len(points), 3)
                normals_random = normals_random / (
                    np.linalg.norm(normals_random, axis=1, keepdims=True) + 1e-8
                )
                all_normals.append(normals_random)

        # Extract leaf tips (before normalization)
        self.leaf_tips_original = []
        for tp in timepoints:
            if tp.get("leaf_tip") is not None:
                self.leaf_tips_original.append(
                    np.array(tp["leaf_tip"], dtype=np.float32)
                )
            else:
                self.leaf_tips_original.append(None)

        # Global normalization if requested
        if self.normalize:
            all_pts, self.global_min, self.global_scale = self._normalize_global(
                all_pts
            )
            self.leaf_tips_normalized = self._normalize_leaf_tips(
                self.leaf_tips_original, self.global_min, self.global_scale
            )
        else:
            self.leaf_tips_normalized = self.leaf_tips_original

        # Convert to tensors and split into train/test
        self._create_train_test_split(all_pts, all_normals)

    def _process_plant_sequence(self):
        """Process full plant sequence with organ labels."""
        timepoints = self.sequence_data["timepoints"]

        # Select timepoints
        timepoints = self._select_timepoints(timepoints)
        self.nt = len(timepoints)
        print(f"  Final number of timepoints: {self.nt}")

        # Update sequence_data to reflect filtered timepoints (important for day extraction)
        self.sequence_data["timepoints"] = timepoints

        # Handle aligned_leaves mode
        if self.aligned_leaves:
            all_pts, all_normals, all_labels = self._process_aligned_leaves(timepoints)
            # Extract leaf tips from aligned leaf sequences (in aligned space)
            self._extract_aligned_leaf_tips(timepoints)
        else:
            all_pts, all_normals, all_labels = self._process_plant_space(timepoints)
            # Extract leaf tips from plant space timepoints
            self._extract_plant_leaf_tips(timepoints)

        # Global normalization if requested
        if self.normalize:
            all_pts, self.global_min, self.global_scale = self._normalize_global(
                all_pts
            )
            self._normalize_plant_leaf_tips()
        else:
            self.leaf_tips_normalized = self.leaf_tips_original
            self.leaf_tip_labels = self.leaf_tip_labels_original

        # Convert to tensors and split into train/test
        self._create_train_test_split(all_pts, all_normals, all_labels)

    def _process_plant_space(self, timepoints):
        """Process plant sequence in plant space (no leaf alignment)."""
        print("  Processing in plant space (no leaf alignment)")

        all_pts = []
        all_normals = []
        all_labels = []

        for tp in timepoints:
            points = tp["points"]
            labels = tp[self.label_field]
            normals = tp.get("normals")

            all_pts.append(points)
            all_labels.append(labels)

            if normals is not None:
                all_normals.append(normals)
            else:
                # Fallback: random normals
                normals_random = np.random.randn(len(points), 3)
                normals_random = normals_random / (
                    np.linalg.norm(normals_random, axis=1, keepdims=True) + 1e-8
                )
                all_normals.append(normals_random)

        return all_pts, all_normals, all_labels

    @staticmethod
    def _warn_if_label_field_contradicts_version(version, label_field):
        """Warn when `label_field` disagrees with the labels the dataset version uses.

        Organ labels reach the pipeline twice: `label_field` sets the training points
        and tips, while `LeafSequencesDataset` takes its own from the version for leaf
        tracks and alignment transforms and accepts no override -- so the two can
        disagree silently, as `v2-autoseg` did before psa 0.3.0. A warning rather than
        an error because the mix can be deliberate: `v2-autoseg` holds both label sets.
        """
        natural = PlantSequencesDataset.DATASET_CONFIGS.get(version, {}).get(
            "label_field"
        )
        # Unknown version: let PlantSequencesDataset raise its own error instead.
        if natural is None or label_field == natural:
            return
        warnings.warn(
            f"label_field={label_field!r} disagrees with version {version!r}, whose "
            f"labels are {natural!r}: leaf tracking and alignment will use {natural!r} "
            f"while training points use {label_field!r}. Pass label_field={natural!r} "
            f"to match.",
            stacklevel=3,
        )

    def _process_aligned_leaves(self, plant_timepoints):
        """Process plant with per-leaf alignment and store transformation matrices."""
        print("  Processing with leaf-level alignment")

        # Get all leaf sequences for this plant
        leaf_sequences = self.leaf_dataset.get_timeseries_by_plant_sequence(
            self.sequence_name
        )

        if len(leaf_sequences) == 0:
            print("  Warning: No leaf sequences found, falling back to plant space")
            return self._process_plant_space(plant_timepoints)

        print(f"  Found {len(leaf_sequences)} leaf sequences")

        all_pts = []
        all_normals = []
        all_labels = []

        # Get the set of days in the (possibly filtered) plant timepoints
        plant_days = {tp["day"] for tp in plant_timepoints}

        # Store transformations for each leaf (if available)
        # Filter transformations to only include days present in plant_timepoints
        for leaf_seq in leaf_sequences:
            leaf_id = leaf_seq["leaf_id"]

            # Get transformation stages directly from the leaf sequence
            if (
                "transformation_stages" in leaf_seq
                and leaf_seq["transformation_stages"]
            ):
                # Build mapping from leaf timepoint index to day
                leaf_days = [ltp["day"] for ltp in leaf_seq["timepoints"]]

                # Store full transformations and day mapping (for interpolation evaluation)
                self.leaf_transformations_full[leaf_id] = leaf_seq["transformation_stages"]
                self.leaf_transformation_days[leaf_id] = leaf_days

                # Find which leaf timepoint indices correspond to days in plant_timepoints
                valid_indices = [i for i, day in enumerate(leaf_days) if day in plant_days]

                # Filter each transformation stage to only include valid indices
                filtered_stages = []
                for stage in leaf_seq["transformation_stages"]:
                    stage_name, stage_data = stage
                    filtered_data = [stage_data[i] for i in valid_indices]
                    filtered_stages.append((stage_name, filtered_data))

                self.leaf_transformations[leaf_id] = filtered_stages
                print(f"    Stored transformations for leaf {leaf_id}")

        for plant_tp in plant_timepoints:
            day = plant_tp["day"]

            # Start from original plant order (so split file indices stay valid)
            tp_points = plant_tp["points"].copy()
            tp_normals = (
                plant_tp["normals"].copy()
                if plant_tp.get("normals") is not None
                else np.zeros_like(plant_tp["points"])
            )
            tp_labels = plant_tp[self.label_field].copy()

            # Replace leaf points with aligned versions (in-place at original positions)
            for leaf_seq in leaf_sequences:
                leaf_id = leaf_seq["leaf_id"]

                leaf_tp = None
                for ltp in leaf_seq["timepoints"]:
                    if ltp["day"] == day:
                        leaf_tp = ltp
                        break

                if leaf_tp is not None:
                    leaf_mask = tp_labels == leaf_id
                    n_plant = int(np.sum(leaf_mask))
                    n_leaf = len(leaf_tp["points"])
                    if n_plant != n_leaf:
                        print(
                            f"  Warning: organ {leaf_id} day {day}: "
                            f"plant has {n_plant} pts, leaf dataset has {n_leaf}"
                        )
                        continue
                    tp_points[leaf_mask] = leaf_tp["points"]
                    if leaf_tp.get("normals") is not None:
                        tp_normals[leaf_mask] = leaf_tp["normals"]

            all_pts.append(tp_points)
            all_normals.append(tp_normals)
            all_labels.append(tp_labels)

        return all_pts, all_normals, all_labels

    def _select_timepoints(self, timepoints):
        """Select timepoints based on indices, stride, or max_timepoints."""
        if self.timepoint_indices is not None:
            timepoints = [timepoints[i] for i in self.timepoint_indices]
            print(f"  Using specific timepoint indices: {self.timepoint_indices}")
        elif self.timepoint_stride > 1:
            timepoints = timepoints[:: self.timepoint_stride]
            print(f"  Using every {self.timepoint_stride} timepoint(s)")
            if self.max_timepoints is not None:
                timepoints = timepoints[: self.max_timepoints]
        elif self.max_timepoints is not None:
            timepoints = timepoints[: self.max_timepoints]
            print(f"  Limiting to first {self.max_timepoints} timepoints")

        return timepoints

    def _normalize_global(self, all_pts):
        """Apply global normalization across all timepoints."""
        all_pts_concat = np.concatenate(all_pts, axis=0)
        min_xyz = np.min(all_pts_concat, axis=0)
        max_xyz = np.max(all_pts_concat, axis=0)
        scale = np.max(max_xyz - min_xyz)
        all_pts_normalized = [(pts - min_xyz) / scale * 2 - 1 for pts in all_pts]
        return all_pts_normalized, min_xyz, scale

    def _normalize_leaf_tips(self, tips, min_xyz, scale):
        """Normalize leaf tips using global normalization parameters."""
        normalized = []
        for tip in tips:
            if tip is not None:
                normalized.append((tip - min_xyz) / scale * 2 - 1)
            else:
                normalized.append(None)
        return normalized

    def _extract_aligned_leaf_tips(self, plant_timepoints):
        """Extract leaf tips from aligned leaf sequences (in aligned space)."""
        self.leaf_tips_original = []
        self.leaf_tip_labels_original = []
        self.leaf_tip_counts = []

        # Get aligned leaf sequences
        leaf_sequences = self.leaf_dataset.get_timeseries_by_plant_sequence(
            self.sequence_name
        )

        for plant_tp in plant_timepoints:
            day = plant_tp["day"]

            # Collect tips for this timepoint from aligned leaves
            tp_tips = []
            tp_tip_labels = []

            # Iterate through leaf sequences to find matching timepoints
            for leaf_seq in leaf_sequences:
                leaf_id = leaf_seq["leaf_id"]

                # Find matching timepoint in leaf sequence
                for ltp in leaf_seq["timepoints"]:
                    if ltp["day"] == day:
                        # Check if this timepoint has a leaf tip
                        if "leaf_tip" in ltp and ltp["leaf_tip"] is not None:
                            tp_tips.append(ltp["leaf_tip"])
                            tp_tip_labels.append(leaf_id)
                        break

            # Store tips for this timepoint
            if len(tp_tips) > 0:
                self.leaf_tips_original.append(np.array(tp_tips, dtype=np.float32))
                self.leaf_tip_labels_original.append(
                    np.array(tp_tip_labels, dtype=np.int32)
                )
                self.leaf_tip_counts.append(len(tp_tips))
            else:
                self.leaf_tips_original.append(None)
                self.leaf_tip_labels_original.append(None)
                self.leaf_tip_counts.append(0)

    def _extract_plant_leaf_tips(self, timepoints):
        """Extract leaf tips from plant timepoints with organ labels."""
        self.leaf_tips_original = []
        self.leaf_tip_labels_original = []
        self.leaf_tip_counts = []

        for tp in timepoints:
            leaf_tip_idxs = tp.get("leaf_tip_idxs", np.array([]))
            if leaf_tip_idxs.size > 0:
                tips = tp["points"][leaf_tip_idxs]
                tip_labels = tp.get(self.label_field, tp["labels"])[leaf_tip_idxs]
                self.leaf_tips_original.append(tips)
                self.leaf_tip_labels_original.append(tip_labels)
                self.leaf_tip_counts.append(len(tips))
            else:
                self.leaf_tips_original.append(None)
                self.leaf_tip_labels_original.append(None)
                self.leaf_tip_counts.append(0)

    def _normalize_plant_leaf_tips(self):
        """Normalize plant leaf tips using global normalization."""
        self.leaf_tips_normalized = []
        self.leaf_tip_labels = []
        for tips, labels in zip(self.leaf_tips_original, self.leaf_tip_labels_original):
            if tips is not None:
                tips_norm = (tips - self.global_min) / self.global_scale * 2 - 1
                self.leaf_tips_normalized.append(tips_norm)
                self.leaf_tip_labels.append(labels)
            else:
                self.leaf_tips_normalized.append(None)
                self.leaf_tip_labels.append(None)

    def _create_train_test_split(self, all_pts, all_normals, all_labels=None):
        """Create train/test split, either from file or by computing."""

        # Check if we should load pre-computed split
        if self.split_file is not None:
            self._load_precomputed_split(all_pts, all_normals, all_labels)
            return

        # Otherwise, compute split on the fly (sequential split, backward compatibility)
        # Convert to tensors
        self.train_pts = []
        self.test_pts = []
        self.train_normals = []
        self.test_normals = []
        self.train_indices = []
        self.test_indices = []

        if all_labels is not None:
            self.train_labels = []
            self.test_labels = []

        for pts, norms in zip(all_pts, all_normals):
            n_train = int(len(pts) * self.train_ratio)

            pts_tensor = torch.tensor(pts, dtype=torch.float32).to(self.device)
            norms_tensor = torch.tensor(norms, dtype=torch.float32).to(self.device)

            # Sequential split
            train_indices = list(range(n_train))
            test_indices = list(range(n_train, len(pts)))

            self.train_indices.append(train_indices)
            self.test_indices.append(test_indices)

            self.train_pts.append(pts_tensor[:n_train, :])
            self.test_pts.append(pts_tensor[n_train:, :])
            self.train_normals.append(norms_tensor[:n_train, :])
            self.test_normals.append(norms_tensor[n_train:, :])

        if all_labels is not None:
            for labels in all_labels:
                n_train = int(len(labels) * self.train_ratio)
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
                self.train_labels.append(labels_tensor[:n_train])
                self.test_labels.append(labels_tensor[n_train:])

    def _load_precomputed_split(self, all_pts, all_normals, all_labels=None):
        """Load pre-computed train/test split from JSON file."""
        import json

        print(f"  Loading pre-computed split from: {self.split_file}")

        with open(self.split_file) as f:
            split_data = json.load(f)

        # Validate split matches current sequence
        if split_data["sequence_name"] != self.sequence_name:
            raise ValueError(
                f"Split file is for sequence '{split_data['sequence_name']}', "
                f"but loading sequence '{self.sequence_name}'"
            )

        validate_split(split_data, [len(pts) for pts in all_pts])

        # Store metadata
        self.random_seed = split_data["random_seed"]
        self.split_train_ratio = split_data["train_ratio"]

        # Initialize lists
        self.train_pts = []
        self.test_pts = []
        self.train_normals = []
        self.test_normals = []
        self.train_indices = []
        self.test_indices = []

        if all_labels is not None:
            self.train_labels = []
            self.test_labels = []

        # Load indices and create splits
        for t, tp_data in enumerate(split_data["timepoints"]):
            train_indices = tp_data["train_indices"]
            test_indices = tp_data["test_indices"]

            # Store indices
            self.train_indices.append(train_indices)
            self.test_indices.append(test_indices)

            # Create tensors with split
            pts = all_pts[t]
            norms = all_normals[t]

            pts_tensor = torch.tensor(pts, dtype=torch.float32).to(self.device)
            norms_tensor = torch.tensor(norms, dtype=torch.float32).to(self.device)

            self.train_pts.append(pts_tensor[train_indices])
            self.test_pts.append(pts_tensor[test_indices])
            self.train_normals.append(norms_tensor[train_indices])
            self.test_normals.append(norms_tensor[test_indices])

            if all_labels is not None:
                labels = all_labels[t]
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
                self.train_labels.append(labels_tensor[train_indices])
                self.test_labels.append(labels_tensor[test_indices])

        print(
            f"  ✅ Loaded split: seed={self.random_seed}, train_ratio={self.split_train_ratio}"
        )
        print(
            f"     Avg train/test: {np.mean([len(idx) for idx in self.train_indices]):.0f} / "
            f"{np.mean([len(idx) for idx in self.test_indices]):.0f} points"
        )

    def get_sequence_info(self):
        """Get information about the loaded sequence."""
        n_train = len(self.train_pts[0]) if len(self.train_pts) > 0 else 0
        n_test = len(self.test_pts[0]) if len(self.test_pts) > 0 else 0

        info = {
            "sequence_name": self.sequence_name,
            "mode": self.mode,
            "n_timepoints": self.nt,
            "n_train": n_train,
            "n_test": n_test,
        }

        if self.mode == "leaf":
            info["plant_sequence_name"] = self.sequence_data["plant_sequence_name"]
            info["leaf_id"] = self.sequence_data["leaf_id"]
        else:  # plant
            # Get unique organ labels
            unique_labels = set()
            for labels in self.train_labels + self.test_labels:
                unique_labels.update(labels.cpu().numpy().tolist())
            n_organs = len(unique_labels)
            n_leaves = n_organs - 1 if 0 in unique_labels else n_organs

            info["n_organs"] = n_organs
            info["n_leaves"] = n_leaves
            info["aligned_leaves"] = self.aligned_leaves
            info["leaf_tips_per_timepoint"] = self.leaf_tip_counts

        return info

    def get_leaf_tips(self, normalized=True, as_tensors=True):
        """Get leaf tip positions for the sequence."""
        tips = self.leaf_tips_normalized if normalized else self.leaf_tips_original

        if as_tensors:
            result = [
                (
                    torch.tensor(tip, dtype=torch.float32).to(self.device)
                    if tip is not None
                    else None
                )
                for tip in tips
            ]
        else:
            result = tips

        return result

    def get_leaf_tip_labels(self, as_tensors=True):
        """
        Get organ labels for each leaf tip (plant mode only).

        Returns:
            List of label arrays/tensors (one per timepoint)
        """
        if self.mode != "plant":
            return [None] * self.nt

        if not hasattr(self, "leaf_tip_labels"):
            return [None] * self.nt

        if as_tensors:
            return [
                (
                    torch.tensor(labels, dtype=torch.long).to(self.device)
                    if labels is not None
                    else None
                )
                for labels in self.leaf_tip_labels
            ]
        else:
            return self.leaf_tip_labels

    def get_organ_ids(self):
        """Get unique organ IDs present in the sequence (plant mode only)."""
        if self.mode != "plant":
            return None

        unique_labels = set()
        for labels in self.train_labels + self.test_labels:
            unique_labels.update(labels.cpu().numpy().tolist())
        return sorted(unique_labels)

    def get_organ_presence_map(self):
        """
        Get dict mapping organ_id -> list of timepoint indices where it appears.
        Plant mode only.
        """
        if self.mode != "plant":
            return None

        presence = defaultdict(list)

        for t in range(len(self.train_labels)):
            # Combine train and test labels for each timepoint
            combined_labels = torch.cat([self.train_labels[t], self.test_labels[t]])
            for organ_id in torch.unique(combined_labels):
                organ_id = int(organ_id)
                if t not in presence[organ_id]:
                    presence[organ_id].append(t)

        return dict(presence)

    def get_timepoint_names(self):
        """Get list of names for each timepoint (e.g., for saving predictions)."""
        names = []
        for i, tp in enumerate(self.sequence_data["timepoints"]):
            if self.mode == "leaf":
                if "file_path" in tp and tp["file_path"] is not None:
                    base_name = os.path.basename(tp["file_path"])
                    base_name = base_name.replace(".ply", f"_leaf{tp['leaf_id']}.ply")
                    names.append(base_name)
                else:
                    names.append(f"{self.sequence_name}_t{i:03d}.ply")
            else:  # plant
                if "day" in tp and tp["day"] is not None:
                    names.append(f"{self.sequence_name}_D{tp['day']:02d}.ply")
                else:
                    names.append(f"{self.sequence_name}_t{i:03d}.ply")
        return names

    def set_sequence(self, sequence_name, verbose=True):
        """
        Switch to a different sequence without reloading the entire dataset.

        Args:
            sequence_name: Name of the sequence to load
            verbose: Print processing info (default: True)

        Returns:
            True if successful, False if sequence not found
        """
        # Find the sequence based on mode
        sequence_data = None

        if self.mode == "leaf":
            for leaf_ts in self.leaf_dataset.leaf_timeseries:
                if leaf_ts["sequence_name"] == sequence_name:
                    sequence_data = leaf_ts
                    break
        else:  # plant
            for plant_ts in self.plant_dataset.plant_timeseries:
                if plant_ts["sequence_name"] == sequence_name:
                    sequence_data = plant_ts
                    break

        if sequence_data is None:
            if verbose:
                print(f"❌ Sequence '{sequence_name}' not found in dataset")
            return False

        # Update sequence
        self.sequence_name = sequence_name
        # Make a shallow copy to avoid mutating the original dataset when filtering timepoints
        self.sequence_data = {
            k: (list(v) if k == "timepoints" else v)
            for k, v in sequence_data.items()
        }

        if verbose:
            print(f"✓ Switched to {self.mode} sequence: {sequence_name}")
            print(f"  Timepoints: {len(self.sequence_data['timepoints'])}")

        # Reprocess with new sequence
        self._process_sequence()

        if verbose:
            print(f"✓ Processed sequence:")
            print(f"  Train points: {len(self.train_pts)} timepoints")
            if len(self.train_pts) > 0:
                print(f"    Example shape: {self.train_pts[0].shape}")
        return True

    def get_available_sequences(self):
        """
        Get list of all available sequences in the loaded dataset.

        Returns:
            List of dicts with sequence info
        """
        sequences = []

        if self.mode == "leaf":
            for leaf_ts in self.leaf_dataset.leaf_timeseries:
                sequences.append(
                    {
                        "name": leaf_ts["sequence_name"],
                        "plant": leaf_ts["plant_sequence_name"],
                        "leaf_id": leaf_ts["leaf_id"],
                        "n_timepoints": len(leaf_ts["timepoints"]),
                    }
                )
        else:  # plant
            for plant_ts in self.plant_dataset.plant_timeseries:
                sequences.append(
                    {
                        "name": plant_ts["sequence_name"],
                        "n_timepoints": len(plant_ts["timepoints"]),
                    }
                )

        return sequences


def list_available_sequences(
    dataset_path,
    version="v2",
    min_timepoints=3,
    mode="leaf",
    crop_filter=None,
    exclude_sequences=None,
):
    """
    List all available sequences in the TrackPlant3D dataset.

    Args:
        dataset_path: Path to TrackPlant3D dataset
        version: Dataset version
        min_timepoints: Minimum number of timepoints
        mode: 'leaf' or 'plant'
        crop_filter: Optional crop name to filter (e.g., 'maize', 'tomato')
        exclude_sequences: Optional list of plant sequence names to exclude entirely

    Returns:
        List of sequence info dicts
    """
    print(f"Loading TrackPlant3D dataset to list {mode} sequences...")

    # Create dataset without selecting a sequence
    dataset = TrackPlant3DDataset(
        dataset_path=dataset_path,
        sequence_name=None,
        version=version,
        mode=mode,
        min_timepoints=min_timepoints,
        auto_download=True,
        exclude_sequences=exclude_sequences,
    )

    sequences = dataset.get_available_sequences()

    # Apply crop filter
    if crop_filter is not None:
        sequences = [s for s in sequences if s["name"].startswith(crop_filter)]

    return sequences


if __name__ == "__main__":
    # Example usage
    dataset_path = "data/TrackPlant3D/versions"

    print("=" * 80)
    print("Testing LEAF mode:")
    print("=" * 80)

    sequences = list_available_sequences(
        dataset_path, version="v2", min_timepoints=5, mode="leaf", crop_filter="maize"
    )

    print(f"\nFound {len(sequences)} maize leaf sequences")
    for seq in sequences[:3]:
        print(f"  {seq['name']:50s} - {seq['n_timepoints']} timepoints")

    if len(sequences) > 0:
        print(f"\nLoading first sequence: {sequences[0]['name']}")
        dataset = TrackPlant3DDataset(
            dataset_path=dataset_path,
            sequence_name=sequences[0]["name"],
            version="v2",
            mode="leaf",
            train_ratio=0.25,
        )

        info = dataset.get_sequence_info()
        print("\nSequence info:")
        for key, val in info.items():
            print(f"  {key}: {val}")

    print("\n" + "=" * 80)
    print("Testing PLANT mode:")
    print("=" * 80)

    plant_sequences = list_available_sequences(
        dataset_path, version="v2", min_timepoints=5, mode="plant", crop_filter="maize"
    )

    print(f"\nFound {len(plant_sequences)} maize plant sequences")
    for seq in plant_sequences[:3]:
        print(f"  {seq['name']:40s} - {seq['n_timepoints']} timepoints")

    if len(plant_sequences) > 0:
        print(f"\nLoading first plant: {plant_sequences[0]['name']}")
        plant_dataset = TrackPlant3DDataset(
            dataset_path=dataset_path,
            sequence_name=plant_sequences[0]["name"],
            version="v2",
            mode="plant",
            train_ratio=0.25,
        )

        plant_info = plant_dataset.get_sequence_info()
        print("\nPlant sequence info:")
        for key, val in plant_info.items():
            print(f"  {key}: {val}")

        print(f"\nOrgan IDs: {plant_dataset.get_organ_ids()}")
        print(f"Organ presence: {plant_dataset.get_organ_presence_map()}")
