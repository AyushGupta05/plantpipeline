# Tests

Two kinds:

- **PLY verification** (all files but one): post-hoc tests that load the saved PLY files of a completed experiment and check them against the dataset and split files. They need `--experiment-dir` and skip without it.
- **Standalone** (`test_standalone.py`): self-contained checks of the training loss, pose interpolation and alignment invariance, run with `pytest tests/test_standalone.py`.

## Quick start

```bash
# After running an experiment with stride:
python train_per_organ.py \
    --config configs/per_organ/continuous.yaml \
    --split test \
    --points_split_dir configs/datasets/TrackPlant3D/point_splits \
    --train_timepoint_stride 2

# Run tests against one sequence's output:
pytest tests/ -v \
    --experiment-dir results/fullplant_.../sequences/maize_control_plant1

# Run against ALL sequences in an experiment:
pytest tests/ -v \
    --experiment-dir results/fullplant_.../sequences
```

The `--experiment-dir` should point to a **sequence-level** directory (the one containing `full_plant/` and `organs/`), or a `sequences/` folder containing multiple sequence subdirectories.

Works for both strided and unstrided experiments: training-day tests always run; interpolation-day tests skip gracefully when there is no interpolation directory.

## CLI options

| Option | Required | Description |
|--------|----------|-------------|
| `--experiment-dir` | No | Path to sequence-level experiment directory. Without it, every test except `test_standalone.py` skips |
| `--split-file` | No | Original (unstrided) split JSON. Auto-discovered from `experiment_config.json` if omitted |
| `--dataset-path` | No | TrackPlant3D dataset root. Auto-discovered from `experiment_config.json` if omitted |

## File structure

```
tests/
├── README.md                          # This file
├── helpers.py                         # Reusable PLY I/O + shared helpers
├── conftest.py                        # Shared pytest fixtures and CLI option registration
├── test_dataset.py                    # Dataset-level checks (1 class, 3 tests)
├── test_gt_vs_dataset.py              # GT PLY ↔ dataset verification (3 classes, 13 tests)
├── test_gt_split.py                   # GT PLY ↔ split file verification (3 classes, 14 tests)
├── test_predictions.py                # Prediction PLY checks (7 classes, 14 tests)
├── test_interpolation_consistency.py  # Cross-pipeline consistency (1 class, 2 tests)
└── test_standalone.py                 # No experiment needed: loss, poses, alignment (3 classes, 10 tests)
```

## Test inventory

### `test_dataset.py`

Dataset-level checks on the plant sequence (independent of PLY files).

#### `TestDataset`

| Test | What it checks |
|------|----------------|
| `test_training_leaf_tips_are_test_points` | Training: leaf tip indices are in the test split |
| `test_interpolation_leaf_tips_are_test_points` | Interpolation: leaf tip indices are in the test split |
| `test_at_most_one_leaf_tip_per_organ` | Each organ has at most one leaf tip per timepoint |

### `test_gt_vs_dataset.py`

Verifies GT PLY files match the plant dataset and leaf dataset.

#### `TestFullPlantGTMatchesDataset`

| Test | What it checks |
|------|----------------|
| `test_interpolation_gt_points_match_dataset` | Interpolation GT PLY points == dataset points |
| `test_training_gt_points_match_dataset` | Training GT PLY points == dataset points |
| `test_interpolation_gt_labels_match_dataset` | Interpolation GT PLY labels == dataset labels |
| `test_training_gt_labels_match_dataset` | Training GT PLY labels == dataset labels |
| `test_interpolation_gt_leaf_tips_match_dataset` | Interpolation GT leaf tips match dataset `leaf_tip_idxs` |
| `test_training_gt_leaf_tips_match_dataset` | Training GT leaf tips match dataset `leaf_tip_idxs` |
| `test_interpolation_gt_point_count_matches_split` | Interpolation GT point count == split `n_total_points` |
| `test_training_gt_point_count_matches_split` | Training GT point count == split `n_total_points` |

#### `TestPerOrganGTMatchesDataset`

| Test | What it checks |
|------|----------------|
| `test_interpolation_per_organ_gt_matches_leaf_dataset` | Interpolation per-organ GT points == leaf dataset points (aligned space) |
| `test_training_per_organ_gt_matches_leaf_dataset` | Training per-organ GT points == leaf dataset points (aligned space) |
| `test_interpolation_per_organ_gt_leaf_tips` | Interpolation per-organ GT leaf tips match leaf dataset |
| `test_training_per_organ_gt_leaf_tips` | Training per-organ GT leaf tips match leaf dataset |

#### `TestGTContainsAllOrgans`

| Test | What it checks |
|------|----------------|
| `test_gt_contains_all_organs` | Full-plant GT organ labels == dataset organ labels per day |

### `test_gt_split.py`

Verifies GT PLY `is_train` labels and organ counts match the split file.

#### `TestGTIsTrainMatchesSplit`

| Test | What it checks |
|------|----------------|
| `test_interpolation_gt_has_is_train_field` | `is_train` exists in all interpolation GT PLYs |
| `test_training_gt_has_is_train_field` | `is_train` exists in all training GT PLYs |
| `test_interpolation_gt_is_train_counts_match_split` | Interpolation GT train/test counts match split |
| `test_training_gt_is_train_counts_match_split` | Training GT train/test counts match split |
| `test_interpolation_gt_is_train_indices_match_split` | Interpolation GT `is_train` flags exact split indices |
| `test_training_gt_is_train_indices_match_split` | Training GT `is_train` flags exact split indices |
| `test_interpolation_gt_train_test_points_match_dataset_split` | End-to-end: interpolation PLY subsets match dataset + split |
| `test_training_gt_train_test_points_match_dataset_split` | End-to-end: training PLY subsets match dataset + split |

#### `TestGTIsTrainConsistencyPlantOrgan`

| Test | What it checks |
|------|----------------|
| `test_interpolation_is_train_consistency` | Interpolation per-organ GT `is_train` == full-plant `is_train[label==id]` |
| `test_training_is_train_consistency` | Training per-organ GT `is_train` == full-plant `is_train[label==id]` |

#### `TestGTOrganCountsMatchSplit`

| Test | What it checks |
|------|----------------|
| `test_interpolation_gt_organ_counts` | Interpolation GT per-organ train/test counts match split |
| `test_training_gt_organ_counts` | Training GT per-organ train/test counts match split |
| `test_interpolation_per_organ_gt_counts_match_split` | Interpolation per-organ GT PLY train/test counts match split |
| `test_training_per_organ_gt_counts_match_split` | Training per-organ GT PLY train/test counts match split |

### `test_interpolation_consistency.py`

Verifies that `save_interpolation_from_organ_models` (Pipeline 1, training-time) and `generate_interpolation.py --n_interp 1` (Pipeline 2, post-training) produce identical full-plant point clouds at unseen days. Requires a stride > 1 experiment and Pipeline 2 to have been run with `--n_interp 1` so intermediate frames fall exactly on the unseen days.

Both tests compare only organs present in both outputs; organs that first appear at an unseen day are expected to be absent from Pipeline 2 and are silently skipped.

#### `TestInterpolationConsistency`

| Test | What it checks |
|------|----------------|
| `test_predicted_full_plant_points_match_between_pipelines` | Per-organ predicted points at each common unseen day are identical between pipelines (atol=1e-4) |
| `test_gt_full_plant_points_match_between_pipelines` | Per-organ GT points at each common unseen day are identical between pipelines (atol=1e-4) |

**How to run:**

```bash
# First generate Pipeline 2 output with n_interp=1
python scripts/generate_interpolation.py \
    --results_dir results/fullplant_.../  \
    --n_interp 1

# Then run (pass sequences/ subdir so fixtures resolve correctly)
pytest tests/test_interpolation_consistency.py -v \
    --experiment-dir results/fullplant_.../sequences
```

### `test_predictions.py`

Verifies prediction PLY files (is_train, points, leaf tips).

#### `TestPredIsTrainField`

| Test | What it checks |
|------|----------------|
| `test_interpolation_pred_has_is_train_field` | `is_train` exists in all interpolation pred PLYs |
| `test_training_pred_has_is_train_field` | `is_train` exists in all training pred PLYs |
| `test_interpolation_pred_is_train_nonzero` | Interpolation pred: both train and test subsets are non-empty |
| `test_training_pred_is_train_nonzero` | Training pred: both train and test subsets are non-empty |
| `test_per_organ_pred_is_train_stable` | Per-organ pred train/test counts identical across all days, matching first appearance |

#### `TestPredIsTrainConsistencyPlantOrgan`

| Test | What it checks |
|------|----------------|
| `test_interpolation_pred_is_train_consistency` | Interpolation per-organ pred `is_train` == full-plant pred `is_train[label==id]` |
| `test_training_pred_is_train_consistency` | Training per-organ pred `is_train` == full-plant pred `is_train[label==id]` |

#### `TestPredPointsConsistencyPlantOrgan`

| Test | What it checks |
|------|----------------|
| `test_pred_points_match_plant_organ` | Per-organ pred points (inverse-transformed if aligned) == full-plant pred points. Also checks leaf tip positions. |

#### `TestPredContainsAllOrgans`

| Test | What it checks |
|------|----------------|
| `test_pred_contains_all_organs` | Full-plant prediction organ labels == dataset organ labels per day |

#### `TestPredIsTrainMatchesFirstAppearance`

| Test | What it checks |
|------|----------------|
| `test_training_pred_organ_counts_match_first_appearance` | Training per-organ pred counts match first-appearance split counts |
| `test_interpolation_pred_organ_counts_match_first_appearance` | Interpolation per-organ pred counts match first-appearance split counts |

#### `TestPredIsTrainConsistentAcrossDays`

| Test | What it checks |
|------|----------------|
| `test_pred_is_train_consistent_across_training_interpolation` | Per-organ pred is_train counts identical between training and interpolation days |

#### `TestPredLeafTipPresenceAfterFirstGT`

| Test | What it checks |
|------|----------------|
| `test_pred_has_leaf_tip_after_first_gt` | Per-organ pred has leaf tip at days > first GT leaf tip day; no tip before |
| `test_fullplant_pred_has_leaf_tip_after_first_gt` | Full-plant pred has leaf tip per organ at days > first dataset leaf tip day (skipping reference day) |

## Code changes (fixed bugs exposed by these tests)

### `scripts/generate_interpolation.py` — inverse transform fix

`generate_interpolation.py` was applying a linearly interpolated transformation matrix (`(1-frac)*M_prev + frac*M_next`) to bring leaves from aligned space back to plant space for interpolated frames, while `save_interpolation_from_organ_models` uses the exact measured transform for each calendar day. These differ whenever the actual per-day transforms are not linear in time.

**Fix:** build a day-keyed dict `organ_inverse_matrices_full` from `manager.organ_transformations_full` / `manager.organ_transformation_days` (mirroring what `io_utils.py` does). At the transform application site, use the exact per-day matrix when available; fall back to matrix interpolation only for fractional target days with no ground-truth transform (e.g. `n_interp > 1` producing non-integer intermediate days).


### `datasets/trackplant3d.py` — `_process_aligned_leaves` fix

The concatenation approach (`cat(stem, leaf1, leaf2, ...)`) reordered points relative to the original PLY, breaking split file index application. Replaced with in-place replacement: starts from a copy of the original plant points/normals/labels, then overwrites leaf positions using boolean masks (`labels == leaf_id`). This preserves the original index order so split file indices remain valid.

### `utils/io_utils.py` — `save_full_plant_from_organ_models` fix

Organs that appear too late to have a trained per-organ model (e.g., first seen at the last timepoint) were missing from full-plant prediction PLYs. Added a fallback after the organ-model loop: for each organ present in the GT but absent from the assembled prediction, the GT reference cloud (with `is_train` from the split file) is included as the prediction.

## Fixtures (conftest.py)

Session-scoped fixtures that auto-discover paths from `experiment_config.json`:

| Fixture | Description |
|---------|-------------|
| `experiment_dir` | Absolute path from `--experiment-dir` |
| `experiment_config` | Parsed `experiment_config.json` |
| `split_file_path` | Original split JSON path (auto-discovered or `--split-file`) |
| `original_split_data` | Parsed split JSON |
| `dataset_path` | Dataset root (auto-discovered or `--dataset-path`) |
| `sequence_name` | From directory name |
| `plant_dataset` | `PlantSequencesDataset` instance |
| `full_plant_sequence` | All timepoints for this sequence |
| `leaf_dataset` | `LeafSequencesDataset` instance (aligned) |
| `leaf_sequences` | Leaf sequences for current plant (aligned space) |
| `day_to_split_info` | `{day: {train_indices, test_indices, ...}}` |
| `all_days` | Sorted list of all day numbers |
| `training_day_set` | Set of training day numbers |
| `interpolation_days` | Sorted list of interpolation day numbers |
| `interpolation_gt_dir` / `interpolation_gt_files` | Interpolation GT directory / PLY paths (files fixture skips if empty) |
| `interpolation_pred_dir` / `interpolation_pred_files` | Interpolation pred directory / PLY paths (files fixture skips if empty) |
| `training_gt_dir` / `training_gt_files` | Training GT directory / PLY paths (supports `post_finetune/` layout) |
| `training_pred_dir` / `training_pred_files` | Training pred directory / PLY paths (supports `post_finetune/` layout) |
| `organs_base_dir` | Path to `organs/` directory |
| `is_aligned` | True if experiment uses aligned leaf data |
| `generate_interp_pred_dir` / `generate_interp_pred_files` | `generate_interpolation.py` predicted output dir / PLY paths (`None`/`[]` if absent) |
| `generate_interp_gt_dir` / `generate_interp_gt_files` | `generate_interpolation.py` GT output dir / PLY paths (`None`/`[]` if absent) |

## Helpers (`helpers.py`)

| Helper | Description |
|--------|-------------|
| `load_ply(path)` | Load PLY file → dict of numpy arrays |
| `find_ply(base_dir, pattern)` | Glob for PLY files under a directory |
| `day_from_filename(path)` | Extract day number from `..._D05.ply` |
| `day_from_filename_float(path)` | Extract integer day from `generate_interpolation.py` filenames (`..._D3.00_F0003.ply`, `..._D3.00_F0003_eval.ply`); returns `None` for non-integer days |
| `find_training_organ_files(organ_dir, kind, prefix)` | Find training organ files (direct or finetuned layout) |
| `build_day_leaf_tip_map(leaf_sequences)` | Build `{day: {leaf_id: tip_position}}` from leaf sequences |

## Adding new tests

### Adding tests to an existing file

Add a new method to an existing test class, or create a new class. Use fixtures from `conftest.py` for shared data. Use helpers from `helpers.py` for PLY I/O:

```python
from helpers import day_from_filename, find_ply, load_ply

class TestMyNewCheck:
    def test_something(self, interpolation_gt_files, day_to_split_info):
        for gt_path in interpolation_gt_files:
            day = day_from_filename(gt_path)
            ply = load_ply(gt_path)
            # ... assertions ...
```

### Adding a new test file

Create `tests/test_<topic>.py`. Fixtures from `conftest.py` are available automatically. Import helpers explicitly:

```python
from helpers import load_ply, find_ply, day_from_filename
```
