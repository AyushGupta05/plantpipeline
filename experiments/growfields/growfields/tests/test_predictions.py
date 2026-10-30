"""Verify prediction PLY files (is_train, points, leaf tips)."""

import glob
import os

import numpy as np
import pytest
from growfields.utils.io_utils import load_ply
from helpers import (
    build_day_leaf_tip_map,
    day_from_filename,
    find_ply,
    find_training_organ_files,
)


# ---------------------------------------------------------------------------
# Prediction is_train field
# ---------------------------------------------------------------------------


class TestPredIsTrainField:
    """Prediction is_train should reflect train/test split consistently."""

    def test_interpolation_pred_has_is_train_field(self, interpolation_pred_files):
        """All interpolation prediction PLYs must have an is_train field."""
        for pred_path in interpolation_pred_files:
            ply = load_ply(pred_path)
            assert (
                "is_train" in ply
            ), f"Day {day_from_filename(pred_path)}: pred missing is_train"

    def test_training_pred_has_is_train_field(self, training_pred_files):
        """All training prediction PLYs must have an is_train field."""
        for pred_path in training_pred_files:
            ply = load_ply(pred_path)
            assert (
                "is_train" in ply
            ), f"Day {day_from_filename(pred_path)}: pred missing is_train"

    def test_interpolation_pred_is_train_nonzero(self, interpolation_pred_files):
        """Interpolation prediction is_train should have both train and test points."""
        for pred_path in interpolation_pred_files:
            ply = load_ply(pred_path)
            if "is_train" not in ply:
                continue
            is_train = ply["is_train"]
            n_train = np.sum(is_train == 1)
            n_test = np.sum(is_train == 0)
            assert (
                n_train > 0
            ), f"Day {day_from_filename(pred_path)}: pred has 0 train points"
            assert (
                n_test > 0
            ), f"Day {day_from_filename(pred_path)}: pred has 0 test points"

    def test_training_pred_is_train_nonzero(self, training_pred_files):
        """Training prediction is_train should have both train and test points."""
        for pred_path in training_pred_files:
            ply = load_ply(pred_path)
            if "is_train" not in ply:
                continue
            is_train = ply["is_train"]
            n_train = np.sum(is_train == 1)
            n_test = np.sum(is_train == 0)
            assert (
                n_train > 0
            ), f"Day {day_from_filename(pred_path)}: pred has 0 train points"
            assert (
                n_test > 0
            ), f"Day {day_from_filename(pred_path)}: pred has 0 test points"

    def test_per_organ_pred_is_train_stable(
        self, organs_base_dir, full_plant_sequence, day_to_split_info,
        training_day_set,
    ):
        """Per-organ prediction is_train counts should be the same across all
        days (training + interpolation) and match the organ's first appearance
        in the dataset split."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        first_counts = TestPredIsTrainMatchesFirstAppearance._first_appearance_counts(
            organs_base_dir, full_plant_sequence, day_to_split_info, training_day_set,
        )

        checked = 0
        for organ_dir in sorted(glob.glob(os.path.join(organs_base_dir, "*"))):
            if not os.path.isdir(organ_dir):
                continue
            organ_type = os.path.basename(organ_dir)

            # Collect ALL prediction files (training + interpolation)
            training_preds = find_training_organ_files(organ_dir, "predicted", "pred")
            interp_preds = find_ply(organ_dir, "interpolation/predicted/pred_*.ply")
            all_preds = training_preds + interp_preds

            if not all_preds:
                continue

            # Determine reference: from dataset split if available, else earliest pred
            if organ_type in first_counts:
                ref_n_train, ref_n_test, ref_day = first_counts[organ_type]
                ref_source = f"first appearance day {ref_day} (dataset split)"
            else:
                earliest = None
                for pred_path in all_preds:
                    ply = load_ply(pred_path)
                    if "is_train" not in ply:
                        continue
                    day = day_from_filename(pred_path)
                    if earliest is None or day < earliest[0]:
                        earliest = (
                            day,
                            int(np.sum(ply["is_train"] == 1)),
                            int(np.sum(ply["is_train"] == 0)),
                        )
                if earliest is None:
                    continue
                ref_day, ref_n_train, ref_n_test = earliest
                ref_source = f"earliest pred day {ref_day}"

            for pred_path in all_preds:
                ply = load_ply(pred_path)
                if "is_train" not in ply:
                    continue
                day = day_from_filename(pred_path)
                n_train = int(np.sum(ply["is_train"] == 1))
                n_test = int(np.sum(ply["is_train"] == 0))
                assert n_train == ref_n_train and n_test == ref_n_test, (
                    f"Organ {organ_type} day {day}: "
                    f"{n_train}/{n_test} (train/test), "
                    f"expected {ref_n_train}/{ref_n_test} from {ref_source}"
                )
                checked += 1

        if checked == 0:
            pytest.skip("No per-organ prediction files with is_train found")


# ---------------------------------------------------------------------------
# Prediction is_train consistency between full plant and per-organ
# ---------------------------------------------------------------------------


class TestPredIsTrainConsistencyPlantOrgan:
    """Per-organ prediction is_train should equal full plant prediction
    is_train filtered by label."""

    def test_interpolation_pred_is_train_consistency(
        self, interpolation_pred_files, organs_base_dir,
    ):
        """Interpolation per-organ pred is_train must equal full-plant pred
        is_train[label==organ]."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        checked = 0
        for pred_path in interpolation_pred_files:
            day = day_from_filename(pred_path)
            plant_ply = load_ply(pred_path)
            if "is_train" not in plant_ply:
                continue

            plant_labels = plant_ply["organ_label"]
            plant_is_train = plant_ply["is_train"]

            organ_pred_files = find_ply(
                organs_base_dir,
                f"**/interpolation/predicted/pred_*D{day:02d}*.ply",
            )
            for organ_pred_path in organ_pred_files:
                organ_ply = load_ply(organ_pred_path)
                organ_labels_unique = np.unique(organ_ply["organ_label"])
                if len(organ_labels_unique) != 1:
                    continue
                organ_id = organ_labels_unique[0]

                if "is_train" not in organ_ply:
                    pytest.fail(
                        f"Day {day} organ {organ_id}: per-organ pred missing is_train"
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
                        f"Day {day} organ {organ_id}: per-organ pred is_train "
                        "!= full plant pred is_train[label==organ]"
                    ),
                )
                checked += 1

        if checked == 0:
            pytest.skip("No per-organ interpolation pred files with is_train found")

    def test_training_pred_is_train_consistency(
        self, training_pred_files, organs_base_dir,
    ):
        """Training per-organ pred is_train must equal full-plant pred
        is_train[label==organ]."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        checked = 0
        for pred_path in training_pred_files:
            day = day_from_filename(pred_path)
            plant_ply = load_ply(pred_path)
            if "is_train" not in plant_ply:
                continue

            plant_labels = plant_ply["organ_label"]
            plant_is_train = plant_ply["is_train"]

            organ_pred_files = find_ply(
                organs_base_dir,
                f"**/predicted/pred_*D{day:02d}*.ply",
            )
            organ_pred_files = [
                f for f in organ_pred_files if "interpolation" not in f
            ]
            for organ_pred_path in organ_pred_files:
                organ_ply = load_ply(organ_pred_path)
                organ_labels_unique = np.unique(organ_ply["organ_label"])
                if len(organ_labels_unique) != 1:
                    continue
                organ_id = organ_labels_unique[0]

                if "is_train" not in organ_ply:
                    pytest.fail(
                        f"Day {day} organ {organ_id}: per-organ pred missing is_train"
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
                        f"Day {day} organ {organ_id}: per-organ pred is_train "
                        "!= full plant pred is_train[label==organ]"
                    ),
                )
                checked += 1

        if checked == 0:
            pytest.skip("No per-organ training pred files with is_train found")


# ---------------------------------------------------------------------------
# Prediction points & leaf tips consistency between full plant and per-organ
# ---------------------------------------------------------------------------


class TestPredPointsConsistencyPlantOrgan:
    """Per-organ prediction points (and leaf tips) should match full-plant
    prediction points for that organ, after inverse-transforming aligned leaves."""

    @staticmethod
    def _build_inverse_transforms(leaf_sequences):
        """Build organ_id -> {day: inv_matrix} from leaf sequence transforms."""
        from plant_shape_analysis import LeafSequencesDataset

        result = {}
        for leaf_seq in leaf_sequences:
            organ_id = leaf_seq["leaf_id"]
            if "transformation_stages" not in leaf_seq:
                continue
            stages = leaf_seq["transformation_stages"]
            if not stages:
                continue
            composed = LeafSequencesDataset.compose_transformations(stages)
            leaf_days = [ltp["day"] for ltp in leaf_seq["timepoints"]]
            inv_by_day = {}
            for i, day in enumerate(leaf_days):
                inv_by_day[day] = LeafSequencesDataset.invert_transformation(
                    composed[i]["composed_matrix"]
                )
            result[organ_id] = inv_by_day
        return result

    def test_pred_points_match_plant_organ(
        self,
        is_aligned,
        leaf_sequences,
        organs_base_dir,
        training_pred_dir,
        interpolation_pred_dir,
    ):
        """Per-organ prediction points (inverse-transformed if aligned) must
        match full-plant prediction points for that organ."""
        from plant_shape_analysis import LeafSequencesDataset

        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        inv_transforms = (
            self._build_inverse_transforms(leaf_sequences) if is_aligned else {}
        )

        all_pred_files = find_ply(training_pred_dir, "pred_*.ply") + find_ply(
            interpolation_pred_dir, "pred_*.ply"
        )
        if not all_pred_files:
            pytest.skip("No full-plant prediction files found")

        checked_points = 0
        for pred_path in all_pred_files:
            plant_ply = load_ply(pred_path)
            day = day_from_filename(pred_path)
            plant_labels = plant_ply["organ_label"]

            parts = os.path.dirname(pred_path).replace("\\", "/").split("/")
            try:
                fp_idx = parts.index("full_plant")
                sub_path = "/".join(parts[fp_idx + 1 :])
            except ValueError:
                sub_path = "predicted"
            organ_pred_files = find_ply(
                organs_base_dir, f"*/{sub_path}/pred_*D{day:02d}*.ply"
            )
            for organ_pred_path in organ_pred_files:
                organ_ply = load_ply(organ_pred_path)
                organ_ids = np.unique(organ_ply["organ_label"])
                if len(organ_ids) != 1:
                    continue
                organ_id = int(organ_ids[0])

                organ_points = organ_ply["points"].copy()

                if is_aligned and organ_id > 0 and organ_id in inv_transforms:
                    inv = inv_transforms[organ_id].get(day)
                    if inv is not None:
                        organ_points = LeafSequencesDataset.apply_transformation(
                            organ_points, inv
                        )

                plant_organ_points = plant_ply["points"][plant_labels == organ_id]
                assert len(organ_points) == len(plant_organ_points), (
                    f"Day {day} organ {organ_id}: point count mismatch "
                    f"({len(organ_points)} vs {len(plant_organ_points)})"
                )
                np.testing.assert_allclose(
                    organ_points,
                    plant_organ_points,
                    atol=1e-4,
                    err_msg=(
                        f"Day {day} organ {organ_id}: per-organ pred points "
                        "don't match full-plant pred points"
                    ),
                )
                checked_points += 1

                organ_has_tip = "is_leaf_tip" in organ_ply and np.any(
                    organ_ply["is_leaf_tip"] == 1
                )
                plant_has_tip_for_organ = False
                if "is_leaf_tip" in plant_ply and "tip_organ_label" in plant_ply:
                    tip_mask = (plant_ply["is_leaf_tip"] == 1) & (
                        plant_ply["tip_organ_label"] == organ_id
                    )
                    plant_has_tip_for_organ = np.any(tip_mask)

                if organ_has_tip and plant_has_tip_for_organ:
                    organ_tip = organ_ply["points"][organ_ply["is_leaf_tip"] == 1]
                    if is_aligned and organ_id > 0 and organ_id in inv_transforms:
                        inv = inv_transforms[organ_id].get(day)
                        if inv is not None:
                            organ_tip = LeafSequencesDataset.apply_transformation(
                                organ_tip, inv
                            )
                    plant_tip = plant_ply["points"][tip_mask]
                    np.testing.assert_allclose(
                        organ_tip,
                        plant_tip,
                        atol=1e-5,
                        err_msg=(
                            f"Day {day} organ {organ_id}: per-organ pred leaf "
                            "tip doesn't match full-plant pred leaf tip"
                        ),
                    )
                elif organ_has_tip != plant_has_tip_for_organ:
                    pytest.fail(
                        f"Day {day} organ {organ_id}: leaf tip mismatch — "
                        f"per-organ has tip: {organ_has_tip}, "
                        f"full-plant has tip: {plant_has_tip_for_organ}"
                    )

        if checked_points == 0:
            pytest.skip("No per-organ prediction files to compare points")


# ---------------------------------------------------------------------------
# Prediction contains all organs
# ---------------------------------------------------------------------------


class TestPredContainsAllOrgans:
    """Full-plant prediction PLYs must contain all organs from the dataset.

    Only for per-organ runs, whose full-plant prediction is assembled from one
    model per organ, each started at the organ's first appearance. A single-model
    run (DPF full-plant) warps earlier frames forward, so a leaf that emerges
    later cannot appear in its prediction; it has no `organs/` directory.
    """

    @staticmethod
    def _expected_organs(full_plant_sequence, day):
        """Set of organ IDs present in the dataset at a given day."""
        for tp in full_plant_sequence["timepoints"]:
            if tp["day"] == day:
                return set(int(o) for o in np.unique(tp["labels"]))
        return None

    def test_pred_contains_all_organs(
        self, experiment_dir, full_plant_sequence, training_pred_dir, interpolation_pred_dir,
    ):
        """Full-plant predictions must contain all organ labels from the dataset."""
        if not os.path.isdir(os.path.join(experiment_dir, "organs")):
            pytest.skip("Single-model run: organs that emerge after the first frame cannot be predicted")
        all_pred_files = find_ply(training_pred_dir, "pred_*.ply") + find_ply(
            interpolation_pred_dir, "pred_*.ply"
        )
        if not all_pred_files:
            pytest.skip("No full-plant prediction files found")

        checked = 0
        for pred_path in all_pred_files:
            day = day_from_filename(pred_path)
            expected = self._expected_organs(full_plant_sequence, day)
            if expected is None:
                continue

            ply = load_ply(pred_path)
            actual = set(int(o) for o in np.unique(ply["organ_label"]))
            assert actual == expected, (
                f"Day {day}: full-plant prediction has organs {sorted(actual)}, "
                f"dataset has {sorted(expected)}"
            )
            checked += 1

        if checked == 0:
            pytest.skip("No prediction files matched dataset days")


# ---------------------------------------------------------------------------
# Prediction train/test counts match first appearance in dataset
# ---------------------------------------------------------------------------


class TestPredIsTrainMatchesFirstAppearance:
    """Prediction PLYs should have the same per-organ train/test counts as
    the organ's first training day from the dataset split."""

    @staticmethod
    def _first_appearance_counts(
        organs_base_dir, full_plant_sequence, day_to_split_info, training_day_set,
    ):
        """Get expected train/test counts per organ type from the split file
        at each organ's first training day.

        Returns dict: organ_type -> (n_train, n_test, first_day)
        """
        day_to_labels = {
            tp["day"]: tp["labels"] for tp in full_plant_sequence["timepoints"]
        }

        organ_type_to_ids = {}
        for organ_dir in sorted(glob.glob(os.path.join(organs_base_dir, "*"))):
            if not os.path.isdir(organ_dir):
                continue
            organ_type = os.path.basename(organ_dir)
            any_plys = (
                find_training_organ_files(organ_dir, "ground_truth", "gt")
                + find_training_organ_files(organ_dir, "predicted", "pred")
                + find_ply(organ_dir, "interpolation/predicted/pred_*.ply")
                + find_ply(organ_dir, "interpolation/ground_truth/gt_*.ply")
            )
            for p in any_plys:
                ply = load_ply(p)
                ids = np.unique(ply["organ_label"])
                if len(ids) == 1:
                    organ_type_to_ids[organ_type] = int(ids[0])
                    break

        result = {}
        sorted_train_days = sorted(training_day_set)

        for organ_type, organ_id in organ_type_to_ids.items():
            organ_str = str(organ_id)
            for day in sorted_train_days:
                if day not in day_to_labels or day not in day_to_split_info:
                    continue
                labels = day_to_labels[day]
                if organ_id not in labels:
                    continue

                split_info = day_to_split_info[day]
                counts_train = split_info.get("organ_counts_train", {})
                counts_test = split_info.get("organ_counts_test", {})

                n_train = counts_train.get(organ_str, 0)
                n_test = counts_test.get(organ_str, 0)
                result[organ_type] = (n_train, n_test, day)
                break

        return result

    def test_training_pred_organ_counts_match_first_appearance(
        self, organs_base_dir, full_plant_sequence, day_to_split_info,
        training_day_set,
    ):
        """Training per-organ prediction train/test counts must match
        the organ's first training day from the dataset split."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        first_counts = self._first_appearance_counts(
            organs_base_dir, full_plant_sequence, day_to_split_info, training_day_set,
        )
        if not first_counts:
            pytest.skip("Could not determine first-appearance counts from dataset")

        checked = 0
        for organ_dir in sorted(glob.glob(os.path.join(organs_base_dir, "*"))):
            organ_type = os.path.basename(organ_dir)
            if organ_type not in first_counts:
                continue

            expected_train, expected_test, first_day = first_counts[organ_type]

            pred_files = find_training_organ_files(organ_dir, "predicted", "pred")

            for pred_path in pred_files:
                ply = load_ply(pred_path)
                if "is_train" not in ply:
                    continue
                day = day_from_filename(pred_path)
                n_train = int(np.sum(ply["is_train"] == 1))
                n_test = int(np.sum(ply["is_train"] == 0))

                assert n_train == expected_train, (
                    f"Organ {organ_type} day {day}: pred has {n_train} train pts, "
                    f"expected {expected_train} (from first appearance day {first_day})"
                )
                assert n_test == expected_test, (
                    f"Organ {organ_type} day {day}: pred has {n_test} test pts, "
                    f"expected {expected_test} (from first appearance day {first_day})"
                )
                checked += 1

        if checked == 0:
            pytest.skip("No per-organ training prediction files with is_train")

    def test_interpolation_pred_organ_counts_match_first_appearance(
        self, organs_base_dir, full_plant_sequence, day_to_split_info,
        training_day_set,
    ):
        """Interpolation per-organ prediction train/test counts must match
        the organ's first training day from the dataset split."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        first_counts = self._first_appearance_counts(
            organs_base_dir, full_plant_sequence, day_to_split_info, training_day_set,
        )
        if not first_counts:
            pytest.skip("Could not determine first-appearance counts from dataset")

        checked = 0
        for organ_dir in sorted(glob.glob(os.path.join(organs_base_dir, "*"))):
            organ_type = os.path.basename(organ_dir)
            if organ_type not in first_counts:
                continue

            expected_train, expected_test, first_day = first_counts[organ_type]

            pred_files = find_ply(organ_dir, "interpolation/predicted/pred_*.ply")
            for pred_path in pred_files:
                ply = load_ply(pred_path)
                if "is_train" not in ply:
                    continue
                day = day_from_filename(pred_path)
                n_train = int(np.sum(ply["is_train"] == 1))
                n_test = int(np.sum(ply["is_train"] == 0))

                assert n_train == expected_train, (
                    f"Organ {organ_type} day {day}: interp pred has "
                    f"{n_train} train pts, expected {expected_train} "
                    f"(from first appearance day {first_day})"
                )
                assert n_test == expected_test, (
                    f"Organ {organ_type} day {day}: interp pred has "
                    f"{n_test} test pts, expected {expected_test} "
                    f"(from first appearance day {first_day})"
                )
                checked += 1

        if checked == 0:
            pytest.skip("No per-organ interpolation prediction files with is_train")


# ---------------------------------------------------------------------------
# Prediction train/test counts consistent across training and interpolation
# ---------------------------------------------------------------------------


class TestPredIsTrainConsistentAcrossDays:
    """Per-organ prediction train/test counts should be identical between
    training and interpolation days, since all predictions warp the same
    initial cloud."""

    def test_pred_is_train_consistent_across_training_interpolation(
        self, organs_base_dir,
    ):
        """Per-organ prediction is_train counts must be the same for training
        and interpolation days."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        checked = 0
        for organ_dir in sorted(glob.glob(os.path.join(organs_base_dir, "*"))):
            if not os.path.isdir(organ_dir):
                continue
            organ_type = os.path.basename(organ_dir)

            training_pred_files = find_training_organ_files(
                organ_dir, "predicted", "pred",
            )

            interp_pred_files = find_ply(
                organ_dir, "interpolation/predicted/pred_*.ply"
            )

            if not training_pred_files or not interp_pred_files:
                continue

            # Get counts from first training prediction
            training_counts = None
            for pred_path in training_pred_files:
                ply = load_ply(pred_path)
                if "is_train" not in ply:
                    continue
                training_day = day_from_filename(pred_path)
                training_counts = (
                    int(np.sum(ply["is_train"] == 1)),
                    int(np.sum(ply["is_train"] == 0)),
                )
                break

            if training_counts is None:
                continue

            # Compare against all interpolation predictions
            for pred_path in interp_pred_files:
                ply = load_ply(pred_path)
                if "is_train" not in ply:
                    continue
                day = day_from_filename(pred_path)
                n_train = int(np.sum(ply["is_train"] == 1))
                n_test = int(np.sum(ply["is_train"] == 0))

                assert n_train == training_counts[0], (
                    f"Organ {organ_type}: interpolation day {day} has "
                    f"{n_train} train pts, but training day {training_day} "
                    f"has {training_counts[0]}"
                )
                assert n_test == training_counts[1], (
                    f"Organ {organ_type}: interpolation day {day} has "
                    f"{n_test} test pts, but training day {training_day} "
                    f"has {training_counts[1]}"
                )
                checked += 1

        if checked == 0:
            pytest.skip(
                "No per-organ prediction files to compare across "
                "training/interpolation"
            )


# ---------------------------------------------------------------------------
# Prediction leaf tips present after first GT occurrence
# ---------------------------------------------------------------------------


class TestPredLeafTipPresenceAfterFirstGT:
    """Per-organ prediction PLYs should have a leaf tip if and only if the
    organ's GT already had a leaf tip at or before that day."""

    @staticmethod
    def _collect_pred_files(organ_dir):
        """Collect all prediction files (training + interpolation) for an organ."""
        training_pred = find_training_organ_files(organ_dir, "predicted", "pred")
        interp_pred = find_ply(organ_dir, "interpolation/predicted/pred_*.ply")
        return training_pred + interp_pred

    def test_pred_has_leaf_tip_after_first_gt(
        self, organs_base_dir, leaf_sequences, training_day_set,
    ):
        """Per-organ predictions at days > first training leaf tip day must
        have a leaf tip; predictions at or before that day must not."""
        if not os.path.isdir(organs_base_dir):
            pytest.skip("No organs directory")

        day_leaf_tips = build_day_leaf_tip_map(leaf_sequences)
        first_tip_day_per_organ = {}
        for day in sorted(day_leaf_tips):
            if day not in training_day_set:
                continue
            for leaf_id in day_leaf_tips[day]:
                if leaf_id not in first_tip_day_per_organ:
                    first_tip_day_per_organ[leaf_id] = day

        checked = 0
        for organ_dir in sorted(glob.glob(os.path.join(organs_base_dir, "*"))):
            if not os.path.isdir(organ_dir):
                continue
            organ_type = os.path.basename(organ_dir)

            pred_files = self._collect_pred_files(organ_dir)
            if not pred_files:
                continue

            stem = os.path.splitext(os.path.basename(pred_files[0]))[0]
            organ_suffix = stem.rsplit("_", 1)[-1]
            if organ_suffix == "stem":
                organ_id = 0
            else:
                organ_id = int("".join(c for c in organ_suffix if c.isdigit()))

            first_tip_day = first_tip_day_per_organ.get(organ_id)

            for pred_path in pred_files:
                ply = load_ply(pred_path)
                day = day_from_filename(pred_path)
                has_tip = "is_leaf_tip" in ply and np.any(ply["is_leaf_tip"] == 1)

                if first_tip_day is not None and day > first_tip_day:
                    assert has_tip, (
                        f"Organ {organ_type} day {day}: prediction missing "
                        f"leaf tip (first dataset leaf tip at day "
                        f"{first_tip_day})"
                    )
                else:
                    assert not has_tip, (
                        f"Organ {organ_type} day {day}: prediction has leaf "
                        f"tip but dataset leaf tip not yet present"
                        + (
                            f" (first at day {first_tip_day})"
                            if first_tip_day is not None
                            else " (no dataset leaf tip found)"
                        )
                    )
                checked += 1

        if checked == 0:
            pytest.skip("No per-organ prediction files to check for leaf tips")

    def test_fullplant_pred_has_leaf_tip_after_first_gt(
        self,
        full_plant_sequence,
        training_day_set,
        training_pred_dir,
        interpolation_pred_dir,
    ):
        """Full-plant predictions must have a leaf tip for each organ
        present in the prediction, provided the dataset has a leaf tip for
        that organ by that day. At an organ's first training day (reference
        cloud), the leaf tip is intentionally excluded."""
        first_tip_day_per_organ = {}
        for tp in full_plant_sequence["timepoints"]:
            day = tp["day"]
            if day not in training_day_set:
                continue
            leaf_tip_idxs = tp.get("leaf_tip_idxs", np.array([]))
            if leaf_tip_idxs.size == 0:
                continue
            tip_labels = tp["labels"][leaf_tip_idxs]
            for organ_id in np.unique(tip_labels):
                organ_id = int(organ_id)
                if organ_id not in first_tip_day_per_organ:
                    first_tip_day_per_organ[organ_id] = day

        if not first_tip_day_per_organ:
            pytest.skip("No leaf tip data in dataset")

        first_train_day_per_organ = {}
        for tp in full_plant_sequence["timepoints"]:
            if tp["day"] not in training_day_set:
                continue
            day = tp["day"]
            for organ_id in np.unique(tp["labels"]):
                organ_id = int(organ_id)
                if organ_id not in first_train_day_per_organ:
                    first_train_day_per_organ[organ_id] = day

        all_pred_files = find_ply(training_pred_dir, "pred_*.ply") + find_ply(
            interpolation_pred_dir, "pred_*.ply"
        )
        if not all_pred_files:
            pytest.skip("No full-plant prediction files found")

        checked = 0
        for pred_path in all_pred_files:
            ply = load_ply(pred_path)
            day = day_from_filename(pred_path)

            organs_in_pred = set(int(o) for o in np.unique(ply["organ_label"]))

            tip_organs_in_ply = set()
            if "is_leaf_tip" in ply and "tip_organ_label" in ply:
                tip_mask = ply["is_leaf_tip"] == 1
                if np.any(tip_mask):
                    tip_organs_in_ply = set(
                        int(o) for o in np.unique(ply["tip_organ_label"][tip_mask])
                    )

            for organ_id, first_tip_day in first_tip_day_per_organ.items():
                if organ_id not in organs_in_pred:
                    continue

                ref_day = first_train_day_per_organ.get(organ_id)
                if ref_day is not None and day == ref_day:
                    assert organ_id not in tip_organs_in_ply, (
                        f"Day {day}: full-plant prediction has leaf tip "
                        f"for organ {organ_id} at its reference day"
                    )
                elif day > first_tip_day:
                    assert organ_id in tip_organs_in_ply, (
                        f"Day {day}: full-plant prediction missing leaf tip "
                        f"for organ {organ_id} (first dataset tip at day "
                        f"{first_tip_day}, reference day {ref_day})"
                    )
                else:
                    assert organ_id not in tip_organs_in_ply, (
                        f"Day {day}: full-plant prediction has leaf tip "
                        f"for organ {organ_id} but dataset tip not yet present "
                        f"(first at day {first_tip_day})"
                    )
                checked += 1

        if checked == 0:
            pytest.skip("No full-plant prediction files to check for leaf tips")
