"""
Shared fixtures for PLY verification tests.

Usage:
    # Run against a specific sequence directory
    pytest tests/ --experiment-dir /path/to/experiment/sequences/maize_control_plant1

    # Run against all sequences in a sequences/ folder
    pytest tests/ --experiment-dir /path/to/experiment/sequences

    # Run with a specific split file (if not discoverable from experiment config)
    pytest tests/ --experiment-dir /path/to/seq_dir --split-file /path/to/split.json

    # Verbose output for debugging
    pytest tests/ -v --experiment-dir /path/to/seq_dir
"""

import glob
import json
import os
import re

import pytest
from helpers import find_ply

from growfields.pipeline.results import read_model_config
from growfields.pipeline.splits import resolve_point_splits_dir

# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--experiment-dir",
        default=None,
        help="Path to a sequence-level directory (contains full_plant/ and organs/) "
        "or a sequences/ folder containing multiple sequence subdirectories. "
        "Tests that inspect saved PLYs skip without it.",
    )
    parser.addoption(
        "--split-file",
        default=None,
        help="Path to the original (unstrided) split JSON. "
        "If omitted, discovered from experiment_config.json.",
    )
    parser.addoption(
        "--dataset-path",
        default=None,
        help="Path to TrackPlant3D dataset root. "
        "If omitted, discovered from experiment_config.json.",
    )


# ---------------------------------------------------------------------------
# Multi-sequence support
# ---------------------------------------------------------------------------

def _parse_experiment_dir(experiment_dir):
    """Return (seq_name, run_tag) for a given experiment_dir.

    Handles two layouts:
    - .../sequences/seq_name        → (seq_name, None)
    - .../sequences/seq_name/run_N  → (seq_name, "run_N")
    """
    base = os.path.basename(experiment_dir)
    if re.match(r"^run_\d+$", base):
        return os.path.basename(os.path.dirname(experiment_dir)), base
    return base, None


# Directories that never contain an experiment; skipped when searching.
_NON_EXPERIMENT_DIRS = {"full_plant", "organs", "interpolation", "models", "summary", "wandb"}


def _is_experiment_dir(path):
    """An experiment dir is one holding a full_plant/ subdirectory."""
    return os.path.isdir(os.path.join(path, "full_plant"))


def _resolve_experiment_dirs(path, max_depth=3):
    """Collect the experiment dirs at or below path.

    Handles every layout the batch runner produces:
      .../seq_name                  → that sequence
      .../seq_name/run_N            → each run of one sequence
      .../sequences                 → every run of every sequence in a batch

    Falls back to [path] when nothing is found, so a bad path surfaces as a
    normal assertion in the experiment_dir fixture rather than an empty run.
    """
    path = os.path.abspath(path)
    if _is_experiment_dir(path):
        return [path]

    found, frontier = [], [(path, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for child in sorted(glob.glob(os.path.join(current, "*"))):
            name = os.path.basename(child)
            if not os.path.isdir(child) or name in _NON_EXPERIMENT_DIRS:
                continue
            if _is_experiment_dir(child):
                found.append(child)
            else:
                frontier.append((child, depth + 1))

    return found if found else [path]


def _experiment_id(experiment_dir):
    """Readable pytest id: "seq_name" or "seq_name/run_N"."""
    seq_name, run_tag = _parse_experiment_dir(experiment_dir)
    return f"{seq_name}/{run_tag}" if run_tag else seq_name


def pytest_generate_tests(metafunc):
    if "experiment_dir" in metafunc.fixturenames:
        path = metafunc.config.getoption("--experiment-dir")
        if path is None:
            metafunc.parametrize(
                "experiment_dir",
                [pytest.param(None, marks=pytest.mark.skip(reason="needs --experiment-dir"))],
                scope="session", indirect=True,
            )
            return
        dirs = _resolve_experiment_dirs(path)
        ids = [_experiment_id(d) for d in dirs]
        metafunc.parametrize(
            "experiment_dir", dirs, ids=ids, scope="session", indirect=True
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def experiment_dir(request):
    """Absolute path to the sequence-level experiment directory."""
    path = request.param
    assert os.path.isdir(path), f"Experiment dir not found: {path}"
    return path


@pytest.fixture(scope="session")
def experiment_config(experiment_dir):
    """Load experiment_config.json (inside, one, two, or three levels up from sequence dir)."""
    _seq_name, run_tag = _parse_experiment_dir(experiment_dir)
    parents = [
        experiment_dir,
        os.path.dirname(experiment_dir),
        os.path.dirname(os.path.dirname(experiment_dir)),
    ]
    if run_tag:
        parents.append(os.path.dirname(os.path.dirname(os.path.dirname(experiment_dir))))
    for parent in parents:
        cfg_path = os.path.join(parent, "experiment_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                return json.load(f)
    return None


@pytest.fixture(scope="session")
def split_file_path(request, experiment_config, experiment_dir, sequence_name):
    """Resolve the original (unstrided) split file path."""
    explicit = request.config.getoption("--split-file")
    if explicit:
        return explicit

    if experiment_config is not None:
        cli_args = experiment_config.get("cli_args", {})
        points_split_dir = resolve_point_splits_dir(cli_args.get("points_split_dir"))
        seq_name = sequence_name
        if points_split_dir:
            candidate = os.path.join(points_split_dir, f"{seq_name}_split.json")
            if os.path.isfile(candidate):
                return candidate

        config = experiment_config.get("config", {})
        sf = config.get("dataset", {}).get("split_file")
        if sf and os.path.isfile(sf):
            return sf

    pytest.skip("Could not discover split file; pass --split-file explicitly")


@pytest.fixture(scope="session")
def original_split_data(split_file_path):
    """The full original (unstrided) split JSON as a dict."""
    with open(split_file_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def dataset_path(request, experiment_config):
    """Resolve the TrackPlant3D dataset path."""
    explicit = request.config.getoption("--dataset-path")
    if explicit:
        return explicit
    if experiment_config is not None:
        config = experiment_config.get("config", {})
        dp = config.get("dataset", {}).get("dataset_path")
        if dp and os.path.isdir(dp):
            return dp
    pytest.skip("Could not discover dataset path; pass --dataset-path explicitly")


@pytest.fixture(scope="session")
def sequence_name(experiment_dir):
    """Sequence name (derived from directory name, run_N dirs use parent name)."""
    seq_name, _run_tag = _parse_experiment_dir(experiment_dir)
    return seq_name


@pytest.fixture(scope="session")
def plant_dataset(dataset_path):
    """Load the PlantSequencesDataset (raw plant data, no alignment)."""
    from plant_shape_analysis import PlantSequencesDataset

    return PlantSequencesDataset(dataset_path, version="v2")


@pytest.fixture(scope="session")
def full_plant_sequence(plant_dataset, sequence_name):
    """Full plant sequence data (all timepoints, plant space)."""
    seq = plant_dataset.get_timeseries_by_sequence_name(sequence_name)
    assert seq is not None, f"Sequence {sequence_name} not found in dataset"
    return seq


@pytest.fixture(scope="session")
def day_to_split_info(full_plant_sequence, original_split_data):
    """Mapping from day number to split info dict."""
    mapping = {}
    for i, tp in enumerate(full_plant_sequence["timepoints"]):
        if i < len(original_split_data["timepoints"]):
            mapping[tp["day"]] = original_split_data["timepoints"][i]
    return mapping


@pytest.fixture(scope="session")
def all_days(full_plant_sequence):
    """Sorted list of all day numbers in the sequence."""
    return sorted(tp["day"] for tp in full_plant_sequence["timepoints"])


@pytest.fixture(scope="session")
def training_days(experiment_dir):
    """Training timepoint indices and stride, discovered from strided split file."""
    seq_name, run_tag = _parse_experiment_dir(experiment_dir)
    if run_tag:
        experiment_root = os.path.dirname(os.path.dirname(os.path.dirname(experiment_dir)))
    else:
        experiment_root = os.path.dirname(os.path.dirname(experiment_dir))
    strided_dir = os.path.join(experiment_root, "strided_splits")
    strided_files = glob.glob(
        os.path.join(strided_dir, f"{seq_name}_stride*_split.json")
    )
    if strided_files:
        with open(strided_files[0]) as f:
            strided = json.load(f)
        stride = strided.get("timepoint_stride", 1)
        original_n = strided.get("original_n_timepoints", strided["n_timepoints"])
        training_tp_indices = list(range(0, original_n, stride))
        return training_tp_indices, stride
    return None, 1


@pytest.fixture(scope="session")
def training_day_set(training_days, all_days):
    """Set of actual training day numbers."""
    training_tp_indices, _stride = training_days
    if training_tp_indices is not None:
        return set(all_days[i] for i in training_tp_indices if i < len(all_days))
    return set(all_days)


@pytest.fixture(scope="session")
def interpolation_days(all_days, training_day_set):
    """Sorted list of interpolation (unseen) day numbers."""
    return sorted(d for d in all_days if d not in training_day_set)


@pytest.fixture(scope="session")
def interpolation_gt_dir(experiment_dir):
    """Path to interpolation GT directory."""
    return os.path.join(experiment_dir, "full_plant", "interpolation", "ground_truth")


@pytest.fixture(scope="session")
def interpolation_pred_dir(experiment_dir):
    """Path to interpolation prediction directory."""
    return os.path.join(experiment_dir, "full_plant", "interpolation", "predicted")


@pytest.fixture(scope="session")
def interpolation_gt_files(interpolation_gt_dir):
    """List of interpolation GT PLY paths, or skip if none found."""
    files = find_ply(interpolation_gt_dir, "gt_*.ply")
    if not files:
        pytest.skip("No interpolation GT files found")
    return files


@pytest.fixture(scope="session")
def interpolation_pred_files(interpolation_pred_dir):
    """List of interpolation prediction PLY paths, or skip if none found."""
    files = find_ply(interpolation_pred_dir, "pred_*.ply")
    if not files:
        pytest.skip("No interpolation prediction files found")
    return files


@pytest.fixture(scope="session")
def generate_interp_pred_dir(experiment_dir, sequence_name):
    """Path to generate_interpolation.py output directory.

    Single-run layout: <exp_root>/interpolation/<seq_name>/predicted/
    Multi-run layout:  <exp_root>/interpolation/<seq_name>/<run_tag>/predicted/
    Returns None if the directory does not exist (test will skip).
    """
    seq_name, run_tag = _parse_experiment_dir(experiment_dir)
    if run_tag:
        exp_root = os.path.dirname(os.path.dirname(os.path.dirname(experiment_dir)))
        d = os.path.join(exp_root, "interpolation", seq_name, run_tag, "predicted")
    else:
        exp_root = os.path.dirname(os.path.dirname(experiment_dir))
        d = os.path.join(exp_root, "interpolation", seq_name, "predicted")
    return d if os.path.isdir(d) else None


@pytest.fixture(scope="session")
def generate_interp_pred_files(generate_interp_pred_dir):
    """List of PLY paths from generate_interpolation.py output, or [] if not found."""
    if generate_interp_pred_dir is None:
        return []
    return find_ply(generate_interp_pred_dir, "*.ply")


@pytest.fixture(scope="session")
def generate_interp_gt_dir(experiment_dir, sequence_name):
    """Path to generate_interpolation.py GT directory.

    Single-run layout: <exp_root>/interpolation/<seq_name>/ground_truth/
    Multi-run layout:  <exp_root>/interpolation/<seq_name>/<run_tag>/ground_truth/
    Returns None if the directory does not exist (test will skip).
    """
    seq_name, run_tag = _parse_experiment_dir(experiment_dir)
    if run_tag:
        exp_root = os.path.dirname(os.path.dirname(os.path.dirname(experiment_dir)))
        d = os.path.join(exp_root, "interpolation", seq_name, run_tag, "ground_truth")
    else:
        exp_root = os.path.dirname(os.path.dirname(experiment_dir))
        d = os.path.join(exp_root, "interpolation", seq_name, "ground_truth")
    return d if os.path.isdir(d) else None


@pytest.fixture(scope="session")
def generate_interp_gt_files(generate_interp_gt_dir):
    """List of GT PLY paths from generate_interpolation.py output, or [] if not found."""
    if generate_interp_gt_dir is None:
        return []
    return find_ply(generate_interp_gt_dir, "*.ply")


@pytest.fixture(scope="session")
def training_gt_dir(experiment_dir):
    """Path to training GT directory.

    Supports two layouts:
    - Continuous: full_plant/ground_truth/
    - Pairwise with finetuning: full_plant/post_finetune/ground_truth/
    """
    fp = os.path.join(experiment_dir, "full_plant")
    # Direct layout (continuous models)
    direct = os.path.join(fp, "ground_truth")
    if os.path.isdir(direct):
        return direct
    # Finetuned layout — prefer post_finetune (final model), fall back to pre_finetune
    for variant in ("post_finetune", "pre_finetune"):
        candidate = os.path.join(fp, variant, "ground_truth")
        if os.path.isdir(candidate):
            return candidate
    return direct  # return default path; training_gt_files will skip if empty


@pytest.fixture(scope="session")
def training_gt_files(training_gt_dir):
    """List of training GT PLY paths, or skip if none found."""
    files = find_ply(training_gt_dir, "gt_*.ply")
    if not files:
        pytest.skip("No training GT files found")
    return files


@pytest.fixture(scope="session")
def training_pred_dir(experiment_dir):
    """Path to training prediction directory.

    Supports two layouts:
    - Continuous: full_plant/predicted/
    - Pairwise with finetuning: full_plant/post_finetune/predicted/
    """
    fp = os.path.join(experiment_dir, "full_plant")
    direct = os.path.join(fp, "predicted")
    if os.path.isdir(direct):
        return direct
    for variant in ("post_finetune", "pre_finetune"):
        candidate = os.path.join(fp, variant, "predicted")
        if os.path.isdir(candidate):
            return candidate
    return direct


@pytest.fixture(scope="session")
def training_pred_files(training_pred_dir):
    """List of training prediction PLY paths, or skip if none found."""
    files = find_ply(training_pred_dir, "pred_*.ply")
    if not files:
        pytest.skip("No training prediction files found")
    return files


@pytest.fixture(scope="session")
def compare_atol(experiment_config):
    """Absolute tolerance for pipeline comparison tests.

    dopri5 (adaptive ODE) accumulates non-determinism across GPU runs → use 1e-2.
    All other methods (fixed-step, or no odeint) → use tight 1e-4.
    """
    default_atol = 1e-4
    if experiment_config is None:
        return default_atol
    model_cfg = read_model_config(experiment_config)
    if model_cfg.get("ode_method") == "dopri5":
        return 1e-2
    return default_atol


@pytest.fixture(scope="session")
def organs_base_dir(experiment_dir):
    """Path to organs/ directory."""
    return os.path.join(experiment_dir, "organs")


@pytest.fixture(scope="session")
def is_aligned(experiment_config, experiment_dir):
    """True if the experiment used leaf-level canonical alignment.
    """
    if experiment_config is not None:
        dataset_cfg = experiment_config.get("config", {}).get("dataset", {})
        recorded = dataset_cfg.get("aligned_leaves")
        if recorded is not None:
            return bool(recorded)
    return "aligned" in experiment_dir


@pytest.fixture(scope="session")
def leaf_dataset(dataset_path):
    """Load the LeafSequencesDataset (aligned leaf data)."""
    from plant_shape_analysis import LeafSequencesDataset

    return LeafSequencesDataset(
        dataset_path, version="v2", min_timepoints=1, apply_alignment=True
    )


@pytest.fixture(scope="session")
def leaf_sequences(leaf_dataset, sequence_name):
    """Leaf sequences for the current plant (aligned space)."""
    return leaf_dataset.get_timeseries_by_plant_sequence(sequence_name)
