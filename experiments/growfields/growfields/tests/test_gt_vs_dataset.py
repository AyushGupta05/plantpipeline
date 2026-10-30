"""Verify GT PLY files match the plant dataset and leaf dataset."""

import os

import numpy as np
import pytest
from growfields.utils.io_utils import load_ply
from helpers import (
    build_day_leaf_tip_map,
    day_from_filename,
    find_ply,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_gt_points_match_dataset(gt_files, day_to_tp, day_set, label):
    """GT PLY points must match dataset points for days in day_set."""
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        assert day in day_set, f"Day {day} in {label} GT but not in {label} set"
        assert day in day_to_tp, f"Day {day} not in plant dataset"

        ply = load_ply(gt_path)
        dataset_points = day_to_tp[day]["points"]

        assert len(ply["points"]) == len(dataset_points), (
            f"Day {day}: PLY has {len(ply['points'])} pts, "
            f"dataset has {len(dataset_points)}"
        )
        np.testing.assert_allclose(
            ply["points"],
            dataset_points,
            atol=1e-5,
            err_msg=f"Day {day}: GT points differ from plant dataset",
        )


def _check_gt_labels_match_dataset(gt_files, day_to_tp):
    """GT PLY organ labels must match dataset labels."""
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        if day not in day_to_tp:
            continue

        ply = load_ply(gt_path)
        dataset_labels = day_to_tp[day]["labels"]

        np.testing.assert_array_equal(
            ply["organ_label"],
            dataset_labels.astype(np.int32),
            err_msg=f"Day {day}: GT labels differ from plant dataset",
        )


def _check_gt_leaf_tips_match_dataset(gt_files, day_to_tp, day_set):
    """GT PLY leaf tips must match dataset leaf tips for days in day_set.

    Returns the number of files checked.
    """
    checked = 0
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        if day not in day_set or day not in day_to_tp:
            continue

        ply = load_ply(gt_path)
        tp = day_to_tp[day]
        leaf_tip_idxs = tp.get("leaf_tip_idxs", np.array([]))

        if leaf_tip_idxs.size == 0:
            if "is_leaf_tip" in ply:
                assert (
                    np.sum(ply["is_leaf_tip"] == 1) == 0
                ), f"Day {day}: PLY has leaf tips but dataset does not"
            continue

        assert "is_leaf_tip" in ply, f"Day {day}: missing is_leaf_tip field"
        assert "tip_organ_label" in ply, f"Day {day}: missing tip_organ_label field"

        tip_mask = ply["is_leaf_tip"] == 1
        ply_tip_points = ply["points"][tip_mask]
        ply_tip_labels = ply["tip_organ_label"][tip_mask]

        dataset_tip_points = tp["points"][leaf_tip_idxs]
        dataset_tip_labels = tp["labels"][leaf_tip_idxs].astype(np.int32)

        assert len(ply_tip_points) == len(dataset_tip_points), (
            f"Day {day}: PLY has {len(ply_tip_points)} leaf tips, "
            f"dataset has {len(dataset_tip_points)}"
        )

        ds_order = np.argsort(dataset_tip_labels)
        ply_order = np.argsort(ply_tip_labels)

        np.testing.assert_array_equal(
            ply_tip_labels[ply_order],
            dataset_tip_labels[ds_order],
            err_msg=f"Day {day}: leaf tip organ labels differ from dataset",
        )

        np.testing.assert_allclose(
            ply_tip_points[ply_order],
            dataset_tip_points[ds_order],
            atol=1e-5,
            err_msg=f"Day {day}: leaf tip positions differ from dataset",
        )
        checked += 1
    return checked


def _check_gt_point_count_matches_split(gt_files, day_to_split_info):
    """GT PLY total point count must match split file n_total_points."""
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        if day not in day_to_split_info:
            continue

        ply = load_ply(gt_path)
        expected_total = day_to_split_info[day]["n_total_points"]

        assert len(ply["points"]) == expected_total, (
            f"Day {day}: PLY has {len(ply['points'])} pts, "
            f"split expects {expected_total}"
        )


# ---------------------------------------------------------------------------
# Full-plant GT vs dataset
# ---------------------------------------------------------------------------


class TestFullPlantGTMatchesDataset:
    """GT points/labels/leaf-tips in PLY should match plant dataset."""

    def test_interpolation_gt_points_match_dataset(
        self, interpolation_gt_files, full_plant_sequence, interpolation_days,
    ):
        """Interpolation GT PLY points must match dataset points."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        _check_gt_points_match_dataset(
            interpolation_gt_files, day_to_tp, set(interpolation_days), "interpolation",
        )

    def test_training_gt_points_match_dataset(
        self, training_gt_files, full_plant_sequence, training_day_set,
    ):
        """Training GT PLY points must match dataset points."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        _check_gt_points_match_dataset(
            training_gt_files, day_to_tp, training_day_set, "training",
        )

    def test_interpolation_gt_labels_match_dataset(
        self, interpolation_gt_files, full_plant_sequence,
    ):
        """Interpolation GT PLY organ labels must match dataset labels."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        _check_gt_labels_match_dataset(interpolation_gt_files, day_to_tp)

    def test_training_gt_labels_match_dataset(
        self, training_gt_files, full_plant_sequence,
    ):
        """Training GT PLY organ labels must match dataset labels."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        _check_gt_labels_match_dataset(training_gt_files, day_to_tp)

    def test_interpolation_gt_leaf_tips_match_dataset(
        self, interpolation_gt_files, full_plant_sequence, interpolation_days,
    ):
        """Interpolation GT PLY leaf tips must match dataset leaf tips."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        checked = _check_gt_leaf_tips_match_dataset(
            interpolation_gt_files, day_to_tp, set(interpolation_days),
        )
        if checked == 0:
            pytest.skip("No GT files with leaf tip data to check")

    def test_training_gt_leaf_tips_match_dataset(
        self, training_gt_files, full_plant_sequence, training_day_set,
    ):
        """Training GT PLY leaf tips must match dataset leaf tips."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        checked = _check_gt_leaf_tips_match_dataset(
            training_gt_files, day_to_tp, training_day_set,
        )
        if checked == 0:
            pytest.skip("No GT files with leaf tip data to check")

    def test_interpolation_gt_point_count_matches_split(
        self, interpolation_gt_files, day_to_split_info,
    ):
        """Interpolation GT PLY point count must match split file."""
        _check_gt_point_count_matches_split(interpolation_gt_files, day_to_split_info)

    def test_training_gt_point_count_matches_split(
        self, training_gt_files, day_to_split_info,
    ):
        """Training GT PLY point count must match split file."""
        _check_gt_point_count_matches_split(training_gt_files, day_to_split_info)


# ---------------------------------------------------------------------------
# Per-organ GT vs dataset
# ---------------------------------------------------------------------------


class TestPerOrganGTMatchesDataset:
    """Per-organ GT PLYs should match leaf dataset (aligned space + leaf tips)."""

    def _check_per_organ_gt_matches_leaf_dataset(
        self, organs_base_dir, is_aligned, dataset_path, sequence_name, days,
        is_interpolation,
    ):
        """Shared logic for per-organ GT vs leaf dataset checks.

        Only meaningful for an aligned experiment: the comparison is against the
        leaf dataset, which is always loaded with `apply_alignment=True`, so for an
        unaligned run this would compare canonical leaf space against plant space.
        The sibling `_check_per_organ_gt_leaf_tips` already skips on the same
        grounds.
        """
        from plant_shape_analysis import LeafSequencesDataset

        if not is_aligned:
            pytest.skip("Not an aligned experiment")
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        leaf_ds = LeafSequencesDataset(dataset_path, version="v2", apply_alignment=True)
        leaf_sequences = leaf_ds.get_timeseries_by_plant_sequence(sequence_name)

        # Build day -> {leaf_id: points}
        day_leaf_points = {}
        for leaf_seq in leaf_sequences:
            leaf_id = leaf_seq["leaf_id"]
            for ltp in leaf_seq["timepoints"]:
                day = ltp["day"]
                if day not in day_leaf_points:
                    day_leaf_points[day] = {}
                day_leaf_points[day][leaf_id] = ltp["points"]

        checked = 0
        for day in days:
            if day not in day_leaf_points:
                continue

            if is_interpolation:
                organ_gt_files = find_ply(
                    organs_base_dir,
                    f"**/interpolation/ground_truth/gt_*D{day:02d}*.ply",
                )
            else:
                organ_gt_files = find_ply(
                    organs_base_dir,
                    f"**/ground_truth/gt_*D{day:02d}*.ply",
                )
                organ_gt_files = [
                    f for f in organ_gt_files if "interpolation" not in f
                ]

            for gt_path in organ_gt_files:
                ply = load_ply(gt_path)
                organ_id = int(np.unique(ply["organ_label"])[0])
                if organ_id == 0:
                    continue  # stem has no alignment

                if organ_id not in day_leaf_points.get(day, {}):
                    continue

                leaf_points = day_leaf_points[day][organ_id]
                np.testing.assert_allclose(
                    ply["points"],
                    leaf_points,
                    atol=1e-5,
                    err_msg=(
                        f"Day {day} organ {organ_id}: "
                        "per-organ GT not in aligned space"
                    ),
                )
                checked += 1

        return checked

    def test_interpolation_per_organ_gt_matches_leaf_dataset(
        self, organs_base_dir, is_aligned, dataset_path, sequence_name,
        interpolation_days,
    ):
        """Interpolation per-organ GT points should match leaf dataset points (aligned space)."""
        checked = self._check_per_organ_gt_matches_leaf_dataset(
            organs_base_dir, is_aligned, dataset_path, sequence_name,
            interpolation_days, is_interpolation=True,
        )
        if checked == 0:
            pytest.skip("No per-organ interpolation GT files matched leaf dataset entries")

    def test_training_per_organ_gt_matches_leaf_dataset(
        self, organs_base_dir, is_aligned, dataset_path, sequence_name,
        training_day_set,
    ):
        """Training per-organ GT points should match leaf dataset points (aligned space)."""
        checked = self._check_per_organ_gt_matches_leaf_dataset(
            organs_base_dir, is_aligned, dataset_path, sequence_name,
            sorted(training_day_set), is_interpolation=False,
        )
        if checked == 0:
            pytest.skip("No per-organ training GT files matched leaf dataset entries")

    def _check_per_organ_gt_leaf_tips(
        self, organs_base_dir, is_aligned, leaf_sequences, days, is_interpolation,
    ):
        """Shared logic for per-organ GT leaf tip checks."""
        if not is_aligned:
            pytest.skip("Not an aligned experiment")
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        day_leaf_tips = build_day_leaf_tip_map(leaf_sequences)

        checked = 0
        for day in sorted(days):
            if day not in day_leaf_tips:
                continue

            if is_interpolation:
                organ_gt_files = find_ply(
                    organs_base_dir,
                    f"**/interpolation/ground_truth/gt_*D{day:02d}*.ply",
                )
            else:
                organ_gt_files = find_ply(
                    organs_base_dir,
                    f"**/ground_truth/gt_*D{day:02d}*.ply",
                )
                organ_gt_files = [
                    f for f in organ_gt_files if "interpolation" not in f
                ]

            for gt_path in organ_gt_files:
                ply = load_ply(gt_path)
                organ_id = int(np.unique(ply["organ_label"])[0])
                if organ_id == 0:
                    continue  # stem has no leaf tip

                expected_tip = day_leaf_tips.get(day, {}).get(organ_id)
                if expected_tip is None:
                    if "is_leaf_tip" in ply:
                        assert np.sum(ply["is_leaf_tip"] == 1) == 0, (
                            f"Day {day} organ {organ_id}: PLY has leaf tip "
                            "but dataset does not"
                        )
                    continue

                assert (
                    "is_leaf_tip" in ply
                ), f"Day {day} organ {organ_id}: missing is_leaf_tip field"
                tip_mask = ply["is_leaf_tip"] == 1
                ply_tip_points = ply["points"][tip_mask]
                assert len(ply_tip_points) == 1, (
                    f"Day {day} organ {organ_id}: expected 1 leaf tip, "
                    f"got {len(ply_tip_points)}"
                )

                np.testing.assert_allclose(
                    ply_tip_points[0],
                    expected_tip,
                    atol=1e-5,
                    err_msg=(
                        f"Day {day} organ {organ_id}: per-organ GT leaf tip "
                        "differs from leaf dataset"
                    ),
                )

                if "tip_organ_label" in ply:
                    ply_tip_label = ply["tip_organ_label"][tip_mask][0]
                    assert ply_tip_label == organ_id, (
                        f"Day {day} organ {organ_id}: tip_organ_label "
                        f"is {ply_tip_label}, expected {organ_id}"
                    )

                checked += 1
        return checked

    def test_interpolation_per_organ_gt_leaf_tips(
        self, organs_base_dir, is_aligned, leaf_sequences, interpolation_days,
    ):
        """Interpolation per-organ GT leaf tips must match leaf dataset tips."""
        checked = self._check_per_organ_gt_leaf_tips(
            organs_base_dir, is_aligned, leaf_sequences, interpolation_days,
            is_interpolation=True,
        )
        if checked == 0:
            pytest.skip("No per-organ interpolation GT files with leaf tip data")

    def test_training_per_organ_gt_leaf_tips(
        self, organs_base_dir, is_aligned, leaf_sequences, training_day_set,
    ):
        """Training per-organ GT leaf tips must match leaf dataset tips."""
        checked = self._check_per_organ_gt_leaf_tips(
            organs_base_dir, is_aligned, leaf_sequences, training_day_set,
            is_interpolation=False,
        )
        if checked == 0:
            pytest.skip("No per-organ training GT files with leaf tip data")


# ---------------------------------------------------------------------------
# GT contains all organs
# ---------------------------------------------------------------------------


class TestGTContainsAllOrgans:
    """Full-plant GT PLYs must contain all organs from the dataset."""

    @staticmethod
    def _expected_organs(full_plant_sequence, day):
        """Set of organ IDs present in the dataset at a given day."""
        for tp in full_plant_sequence["timepoints"]:
            if tp["day"] == day:
                return set(int(o) for o in np.unique(tp["labels"]))
        return None

    def test_gt_contains_all_organs(
        self, full_plant_sequence, training_gt_dir, interpolation_gt_dir,
    ):
        """Full-plant GT PLYs must contain all organ labels from the dataset."""
        all_gt_files = find_ply(training_gt_dir, "gt_*.ply") + find_ply(
            interpolation_gt_dir, "gt_*.ply"
        )
        if not all_gt_files:
            pytest.skip("No full-plant GT files found")

        checked = 0
        for gt_path in all_gt_files:
            day = day_from_filename(gt_path)
            expected = self._expected_organs(full_plant_sequence, day)
            if expected is None:
                continue

            ply = load_ply(gt_path)
            actual = set(int(o) for o in np.unique(ply["organ_label"]))
            assert actual == expected, (
                f"Day {day}: full-plant GT has organs {sorted(actual)}, "
                f"dataset has {sorted(expected)}"
            )
            checked += 1

        if checked == 0:
            pytest.skip("No GT files matched dataset days")
