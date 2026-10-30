"""Verify GT PLY is_train labels and organ counts match the split file."""

import os

import numpy as np
import pytest
from growfields.utils.io_utils import load_ply
from helpers import day_from_filename, find_ply


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_gt_has_is_train(gt_files):
    """All GT PLYs must have an is_train field."""
    for gt_path in gt_files:
        ply = load_ply(gt_path)
        assert (
            "is_train" in ply
        ), f"Day {day_from_filename(gt_path)}: missing is_train field"


def _check_gt_is_train_counts(gt_files, day_to_split_info):
    """GT is_train train/test counts must match the split file."""
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        if day not in day_to_split_info:
            continue

        ply = load_ply(gt_path)
        split_info = day_to_split_info[day]

        assert "is_train" in ply, f"Day {day}: missing is_train"
        is_train = ply["is_train"]

        n_train_ply = int(np.sum(is_train == 1))
        n_test_ply = int(np.sum(is_train == 0))

        assert n_train_ply == split_info["n_train"], (
            f"Day {day}: PLY has {n_train_ply} train pts, "
            f"split expects {split_info['n_train']}"
        )
        assert n_test_ply == split_info["n_test"], (
            f"Day {day}: PLY has {n_test_ply} test pts, "
            f"split expects {split_info['n_test']}"
        )


def _check_gt_is_train_indices(gt_files, day_to_split_info):
    """GT is_train must flag the exact same point indices as the split file."""
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        if day not in day_to_split_info:
            continue

        ply = load_ply(gt_path)
        split_info = day_to_split_info[day]
        is_train = ply["is_train"]

        expected = np.full(len(is_train), -1, dtype=np.int32)
        train_idx = np.array(split_info["train_indices"])
        test_idx = np.array(split_info["test_indices"])
        expected[train_idx] = 1
        expected[test_idx] = 0

        np.testing.assert_array_equal(
            is_train,
            expected,
            err_msg=f"Day {day}: is_train indices don't match split file",
        )


def _check_gt_train_test_points(gt_files, day_to_tp, day_to_split_info):
    """End-to-end: dataset_points[train/test_indices] == ply_points[is_train==1/0]."""
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        if day not in day_to_split_info or day not in day_to_tp:
            continue

        ply = load_ply(gt_path)
        if "is_train" not in ply:
            continue

        split_info = day_to_split_info[day]
        dataset_points = day_to_tp[day]["points"]
        train_indices = np.array(split_info["train_indices"])
        test_indices = np.array(split_info["test_indices"])

        expected_train = dataset_points[train_indices]
        ply_train = ply["points"][ply["is_train"] == 1]
        np.testing.assert_allclose(
            ply_train,
            expected_train,
            atol=1e-5,
            err_msg=f"Day {day}: PLY train points != dataset[train_indices]",
        )

        expected_test = dataset_points[test_indices]
        ply_test = ply["points"][ply["is_train"] == 0]
        np.testing.assert_allclose(
            ply_test,
            expected_test,
            atol=1e-5,
            err_msg=f"Day {day}: PLY test points != dataset[test_indices]",
        )


def _check_is_train_consistency_plant_organ(gt_files, organs_base_dir, is_interpolation):
    """Per-organ GT is_train must equal full-plant GT is_train[label==organ].

    Returns the number of organ files checked.
    """
    if not os.path.isdir(organs_base_dir):
        pytest.skip("No organs directory")

    checked = 0
    for gt_path in gt_files:
        day = day_from_filename(gt_path)
        plant_ply = load_ply(gt_path)

        if "is_train" not in plant_ply:
            continue

        plant_labels = plant_ply["organ_label"]
        plant_is_train = plant_ply["is_train"]

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
            organ_gt_files = [f for f in organ_gt_files if "interpolation" not in f]

        for organ_gt_path in organ_gt_files:
            organ_ply = load_ply(organ_gt_path)
            organ_labels_unique = np.unique(organ_ply["organ_label"])
            if len(organ_labels_unique) != 1:
                continue
            organ_id = organ_labels_unique[0]

            if "is_train" not in organ_ply:
                pytest.fail(
                    f"Day {day} organ {organ_id}: per-organ GT missing is_train"
                )

            plant_organ_is_train = plant_is_train[plant_labels == organ_id]

            assert len(organ_ply["is_train"]) == len(plant_organ_is_train), (
                f"Day {day} organ {organ_id}: count mismatch "
                f"({len(organ_ply['is_train'])} vs {len(plant_organ_is_train)})"
            )
            np.testing.assert_array_equal(
                organ_ply["is_train"],
                plant_organ_is_train,
                err_msg=(
                    f"Day {day} organ {organ_id}: "
                    "per-organ is_train != full plant is_train[label==organ]"
                ),
            )
            checked += 1
    return checked


# ---------------------------------------------------------------------------
# GT is_train matches split file
# ---------------------------------------------------------------------------


class TestGTIsTrainMatchesSplit:
    """Verify GT is_train labels match the original split file exactly."""

    def test_interpolation_gt_has_is_train_field(self, interpolation_gt_files):
        """All interpolation GT PLYs must have an is_train field."""
        _check_gt_has_is_train(interpolation_gt_files)

    def test_training_gt_has_is_train_field(self, training_gt_files):
        """All training GT PLYs must have an is_train field."""
        _check_gt_has_is_train(training_gt_files)

    def test_interpolation_gt_is_train_counts_match_split(
        self, interpolation_gt_files, day_to_split_info,
    ):
        """Interpolation GT is_train counts must match the split file."""
        _check_gt_is_train_counts(interpolation_gt_files, day_to_split_info)

    def test_training_gt_is_train_counts_match_split(
        self, training_gt_files, day_to_split_info,
    ):
        """Training GT is_train counts must match the split file."""
        _check_gt_is_train_counts(training_gt_files, day_to_split_info)

    def test_interpolation_gt_is_train_indices_match_split(
        self, interpolation_gt_files, day_to_split_info,
    ):
        """Interpolation GT is_train must flag exact same indices as split file."""
        _check_gt_is_train_indices(interpolation_gt_files, day_to_split_info)

    def test_training_gt_is_train_indices_match_split(
        self, training_gt_files, day_to_split_info,
    ):
        """Training GT is_train must flag exact same indices as split file."""
        _check_gt_is_train_indices(training_gt_files, day_to_split_info)

    def test_interpolation_gt_train_test_points_match_dataset_split(
        self, interpolation_gt_files, full_plant_sequence, day_to_split_info,
    ):
        """End-to-end: interpolation PLY train/test subsets match dataset + split."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        _check_gt_train_test_points(
            interpolation_gt_files, day_to_tp, day_to_split_info,
        )

    def test_training_gt_train_test_points_match_dataset_split(
        self, training_gt_files, full_plant_sequence, day_to_split_info,
    ):
        """End-to-end: training PLY train/test subsets match dataset + split."""
        day_to_tp = {tp["day"]: tp for tp in full_plant_sequence["timepoints"]}
        _check_gt_train_test_points(
            training_gt_files, day_to_tp, day_to_split_info,
        )


# ---------------------------------------------------------------------------
# is_train consistency between full plant and per-organ
# ---------------------------------------------------------------------------


class TestGTIsTrainConsistencyPlantOrgan:
    """Per-organ GT is_train should equal full plant is_train filtered by label."""

    def test_interpolation_is_train_consistency(
        self, interpolation_gt_files, organs_base_dir,
    ):
        """Interpolation per-organ GT is_train must match full-plant."""
        checked = _check_is_train_consistency_plant_organ(
            interpolation_gt_files, organs_base_dir, is_interpolation=True,
        )
        if checked == 0:
            pytest.skip("No per-organ interpolation GT files with is_train found")

    def test_training_is_train_consistency(
        self, training_gt_files, organs_base_dir,
    ):
        """Training per-organ GT is_train must match full-plant."""
        checked = _check_is_train_consistency_plant_organ(
            training_gt_files, organs_base_dir, is_interpolation=False,
        )
        if checked == 0:
            pytest.skip("No per-organ training GT files with is_train found")


# ---------------------------------------------------------------------------
# GT organ counts match split
# ---------------------------------------------------------------------------


class TestGTOrganCountsMatchSplit:
    """Per-organ train/test counts in GT PLYs must match split file organ_counts."""

    def _check_organ_counts(self, gt_files, day_to_split_info, label):
        """Shared logic for interpolation and training GT files."""
        checked = 0
        for gt_path in gt_files:
            day = day_from_filename(gt_path)
            if day not in day_to_split_info:
                continue

            split_info = day_to_split_info[day]
            organ_counts_train = split_info.get("organ_counts_train")
            organ_counts_test = split_info.get("organ_counts_test")
            if organ_counts_train is None and organ_counts_test is None:
                continue

            ply = load_ply(gt_path)
            if "is_train" not in ply or "organ_label" not in ply:
                continue

            labels = ply["organ_label"]
            is_train = ply["is_train"]

            if organ_counts_train is not None:
                for organ_str, expected_count in organ_counts_train.items():
                    organ_id = int(organ_str)
                    actual = int(np.sum((labels == organ_id) & (is_train == 1)))
                    assert actual == expected_count, (
                        f"{label} day {day} organ {organ_id}: "
                        f"PLY has {actual} train pts, split expects {expected_count}"
                    )

            if organ_counts_test is not None:
                for organ_str, expected_count in organ_counts_test.items():
                    organ_id = int(organ_str)
                    actual = int(np.sum((labels == organ_id) & (is_train == 0)))
                    assert actual == expected_count, (
                        f"{label} day {day} organ {organ_id}: "
                        f"PLY has {actual} test pts, split expects {expected_count}"
                    )

            checked += 1
        return checked

    def test_interpolation_gt_organ_counts(
        self, interpolation_gt_files, day_to_split_info,
    ):
        """Interpolation GT per-organ train/test counts must match split file."""
        checked = self._check_organ_counts(
            interpolation_gt_files, day_to_split_info, "Interpolation",
        )
        if checked == 0:
            pytest.skip("No interpolation GT files with organ counts in split")

    def test_training_gt_organ_counts(
        self, training_gt_files, day_to_split_info,
    ):
        """Training GT per-organ train/test counts must match split file."""
        checked = self._check_organ_counts(
            training_gt_files, day_to_split_info, "Training",
        )
        if checked == 0:
            pytest.skip("No training GT files with organ counts in split")

    def _check_per_organ_gt_counts(
        self, organs_base_dir, day_to_split_info, days, is_interpolation,
    ):
        """Shared logic for per-organ GT train/test counts vs split file."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        checked = 0
        for day in days:
            if day not in day_to_split_info:
                continue

            split_info = day_to_split_info[day]
            organ_counts_train = split_info.get("organ_counts_train", {})
            organ_counts_test = split_info.get("organ_counts_test", {})

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
                organ_labels_unique = np.unique(ply["organ_label"])
                if len(organ_labels_unique) != 1:
                    continue
                organ_id = int(organ_labels_unique[0])
                organ_str = str(organ_id)

                if "is_train" not in ply:
                    continue

                n_train = int(np.sum(ply["is_train"] == 1))
                n_test = int(np.sum(ply["is_train"] == 0))

                if organ_str in organ_counts_train:
                    assert n_train == organ_counts_train[organ_str], (
                        f"Day {day} organ {organ_id}: per-organ PLY has "
                        f"{n_train} train pts, split expects "
                        f"{organ_counts_train[organ_str]}"
                    )
                if organ_str in organ_counts_test:
                    assert n_test == organ_counts_test[organ_str], (
                        f"Day {day} organ {organ_id}: per-organ PLY has "
                        f"{n_test} test pts, split expects "
                        f"{organ_counts_test[organ_str]}"
                    )
                checked += 1

        return checked

    def test_interpolation_per_organ_gt_counts_match_split(
        self, organs_base_dir, day_to_split_info, interpolation_days,
    ):
        """Interpolation per-organ GT PLY train/test counts must match split organ_counts."""
        checked = self._check_per_organ_gt_counts(
            organs_base_dir, day_to_split_info, interpolation_days,
            is_interpolation=True,
        )
        if checked == 0:
            pytest.skip("No per-organ interpolation GT files with organ counts in split")

    def test_training_per_organ_gt_counts_match_split(
        self, organs_base_dir, day_to_split_info, training_day_set,
    ):
        """Training per-organ GT PLY train/test counts must match split organ_counts."""
        checked = self._check_per_organ_gt_counts(
            organs_base_dir, day_to_split_info, sorted(training_day_set),
            is_interpolation=False,
        )
        if checked == 0:
            pytest.skip("No per-organ training GT files with organ counts in split")
