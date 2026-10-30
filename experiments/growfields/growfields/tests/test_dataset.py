"""Dataset-level checks on the plant sequence (independent of PLY files)."""

import numpy as np
import pytest
from helpers import day_from_filename


class TestDataset:
    """Tests for the dataset GT."""

    def _check_leaf_tips_are_test_points(
        self,
        gt_files,
        full_plant_sequence,
        day_set,
        day_to_split_info,
    ):
        """Shared logic: dataset leaf tip indices must be test points."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}

        checked = 0
        for gt_path in gt_files:
            day = day_from_filename(gt_path)
            if day not in day_set or day not in day_to_tp:
                continue

            tp = day_to_tp[day]
            leaf_tip_idxs = tp.get("leaf_tip_idxs", np.array([]))
            if leaf_tip_idxs.size == 0:
                continue

            if day not in day_to_split_info:
                continue

            test_indices = set(day_to_split_info[day]["test_indices"])
            for idx in leaf_tip_idxs:
                assert (
                    int(idx) in test_indices
                ), f"Day {day}: leaf tip index {idx} is not a test point"
            checked += 1

        return checked

    def test_training_leaf_tips_are_test_points(
        self,
        training_gt_files,
        full_plant_sequence,
        training_day_set,
        day_to_split_info,
    ):
        """Training: dataset leaf tip indices must be test points."""
        checked = self._check_leaf_tips_are_test_points(
            training_gt_files,
            full_plant_sequence,
            training_day_set,
            day_to_split_info,
        )
        if checked == 0:
            pytest.skip(
                "No training GT files with leaf tip data to check against split"
            )

    def test_interpolation_leaf_tips_are_test_points(
        self,
        interpolation_gt_files,
        full_plant_sequence,
        interpolation_days,
        day_to_split_info,
    ):
        """Interpolation: dataset leaf tip indices must be test points."""
        checked = self._check_leaf_tips_are_test_points(
            interpolation_gt_files,
            full_plant_sequence,
            set(interpolation_days),
            day_to_split_info,
        )
        if checked == 0:
            pytest.skip(
                "No interpolation GT files with leaf tip data to check against split"
            )

    def test_at_most_one_leaf_tip_per_organ(self, full_plant_sequence):
        """Each organ should have at most one leaf tip per timepoint."""
        for tp in full_plant_sequence["timepoints"]:
            leaf_tip_idxs = tp.get("leaf_tip_idxs", np.array([]))
            if leaf_tip_idxs.size == 0:
                continue

            tip_labels = tp["labels"][leaf_tip_idxs]
            unique, counts = np.unique(tip_labels, return_counts=True)
            duplicates = unique[counts > 1]
            assert (
                len(duplicates) == 0
            ), f"Day {tp['day']}: organs {duplicates.tolist()} have multiple leaf tips"
