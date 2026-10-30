"""Train/test split files: schema, reading, writing, and derivation.

A split file records which points of which timepoints are training data for one
sequence. `scripts/create_train_test_splits.py` produces them; the training
entry points derive filtered variants (strided, holdout) at run time. All three
go through the helpers here so the schema is owned in one place.
"""

import json
import os

# Metadata carried unchanged from a base split onto any split derived from it.
SPLIT_METADATA_KEYS = (
    "sequence_name",
    "random_seed",
    "train_ratio",
    "created_date",
    "dataset_version",
)


def read_split(split_file):
    """Load a split file."""
    with open(split_file) as f:
        return json.load(f)


def point_split_path(points_split_dir, sequence_name):
    """Path of a sequence's split file inside `--points_split_dir`."""
    return os.path.join(points_split_dir, f"{sequence_name}_split.json")


def require_point_splits(points_split_dir, sequence_names):
    """Raise if any requested sequence has no split file.

    Without one the dataset silently falls back to a sequential split, so a typo in
    `--points_split_dir` would train on different points than the published runs.
    """
    missing = [
        point_split_path(points_split_dir, name)
        for name in sequence_names
        if not os.path.isfile(point_split_path(points_split_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} point split file(s) not found, e.g. {missing[0]}. "
            "Check --points_split_dir, or create them with "
            "scripts/create_train_test_splits.py."
        )


def validate_split(split_data, n_points_per_timepoint):
    """Raise ValueError unless a split's indices fit the loaded sequence.

    A split from another dataset version or a hand-edited file would otherwise
    index the wrong points, or silently drop and duplicate some.

    Args:
        split_data: A loaded split file.
        n_points_per_timepoint: Point count of each loaded timepoint, in order.
    """
    timepoints = split_data["timepoints"]
    if not (split_data["n_timepoints"] == len(timepoints) == len(n_points_per_timepoint)):
        raise ValueError(
            f"Split has n_timepoints={split_data['n_timepoints']} and "
            f"{len(timepoints)} entries, but the sequence has "
            f"{len(n_points_per_timepoint)} timepoints"
        )
    for t, (tp, n_points) in enumerate(zip(timepoints, n_points_per_timepoint)):
        train, test = tp["train_indices"], tp["test_indices"]
        where = f"Split timepoint {t}"
        if tp.get("n_total_points", n_points) != n_points:
            raise ValueError(
                f"{where} was made for {tp['n_total_points']} points, "
                f"but the loaded cloud has {n_points}"
            )
        all_idx = train + test
        if any(i < 0 or i >= n_points for i in all_idx):
            raise ValueError(f"{where} has indices outside [0, {n_points})")
        if len(set(train)) != len(train) or len(set(test)) != len(test):
            raise ValueError(f"{where} repeats indices within train or test")
        if set(train) & set(test):
            raise ValueError(f"{where} puts some points in both train and test")


def write_split(split_data, split_path):
    """Write a split file, creating its directory. Returns the path."""
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    return split_path


def new_split(sequence_name, seed, train_ratio, version, n_timepoints):
    """Build the metadata header for a freshly created split.

    The caller appends per-timepoint entries to ``split_data["timepoints"]``.
    """
    from datetime import datetime

    return {
        "sequence_name": sequence_name,
        "random_seed": seed,
        "train_ratio": train_ratio,
        "created_date": datetime.now().isoformat(),
        "dataset_version": version,
        "n_timepoints": n_timepoints,
        "timepoints": [],
    }


def derive_split(base, timepoints, **extra):
    """Carry a base split's metadata onto a filtered timepoint list.

    Args:
        base: The split being derived from, as returned by `read_split`.
        timepoints: The retained subset of ``base["timepoints"]``.
        extra: Marker fields describing the derivation, e.g.
            ``timepoint_stride=2`` or ``holdout_last_n=2``.

    Raises:
        KeyError: if ``base`` is missing metadata. Every split written by
            `new_split` carries it; a file without it is malformed and should
            not be silently assigned a default dataset version.
    """
    derived = {k: base[k] for k in SPLIT_METADATA_KEYS}
    derived["n_timepoints"] = len(timepoints)
    derived["original_n_timepoints"] = base["n_timepoints"]
    derived.update(extra)
    derived["timepoints"] = timepoints
    return derived


def create_strided_split_file(split_file, stride, sequence_name, experiment_dir):
    """Derive a split keeping every `stride`-th timepoint. Returns its path."""
    base = read_split(split_file)
    timepoints = [tp for i, tp in enumerate(base["timepoints"]) if i % stride == 0]
    return write_split(
        derive_split(base, timepoints, timepoint_stride=stride),
        os.path.join(
            experiment_dir, "strided_splits", f"{sequence_name}_stride{stride}_split.json"
        ),
    )


def create_holdout_last_n_split_file(split_file, n, sequence_name, experiment_dir):
    """Derive a split holding out the last `n` timepoints. Returns its path."""
    base = read_split(split_file)
    timepoints = base["timepoints"][:-n]
    return write_split(
        derive_split(base, timepoints, holdout_last_n=n),
        os.path.join(
            experiment_dir,
            "extrapolation_splits",
            f"{sequence_name}_holdout_last{n}_split.json",
        ),
    )


#: Which plants belong to each named split, relative to the repo root.
SEQUENCE_SPLITS_FILE = os.path.join("configs", "datasets", "TrackPlant3D", "data_splits.json")

#: Where the per-point train/test splits live, relative to the repo root.
POINT_SPLITS_DIR = os.path.join("configs", "datasets", "TrackPlant3D", "point_splits")


def resolve_point_splits_dir(recorded_dir=None):
    """Return the directory to read per-point train/test splits from.

    The `points_split_dir` recorded in an `experiment_config.json` is
    provenance, not a location -- it need not exist on the machine reading the
    results back -- so fall back to the splits shipped with the repo. Callers
    degrade silently rather than failing when they find no split file.
    """
    if recorded_dir and os.path.isdir(recorded_dir):
        return recorded_dir
    return POINT_SPLITS_DIR


def load_sequence_split(split_name, repo_root=None):
    """Return (plant names, split metadata) for a named sequence split.

    This is the plant-level split -- which plants are val and which are test --
    not the point-level train/test split the rest of this module deals with.

    Raises:
        FileNotFoundError: if `data_splits.json` is missing.
        KeyError: if `split_name` is not one of the splits it defines.
    """
    if repo_root is None:
        # this file is <repo>/growfields/pipeline/splits.py, so up three
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    path = os.path.join(repo_root, SEQUENCE_SPLITS_FILE)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Split configuration not found at {path}")

    with open(path) as f:
        split_config = json.load(f)

    splits = split_config["splits"]
    if split_name not in splits:
        raise KeyError(
            f"Split '{split_name}' not found in {path}; have {sorted(splits)}"
        )

    split_data = splits[split_name]
    return list(split_data["plants"]), split_data
