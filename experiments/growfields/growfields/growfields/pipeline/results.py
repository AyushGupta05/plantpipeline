"""Results directory layout.

Creating a results directory and recording what produced it. JSON writing lives
in `utils.io_utils.write_json`, which `models/` and `scripts/` also use.
"""

import json
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from growfields.pipeline.config import deterministic_enabled
from growfields.utils.io_utils import write_json


def prepare_results_dir(results_dir, cfg, split_file=None, verbose=True):
    """Create the results directory and record what produced it.

    Writes the resolved config as `config.yaml` and copies the split file in as
    `train_test_split.json`, so a results directory is self-describing.

    Returns:
        The results directory path.
    """
    os.makedirs(results_dir, exist_ok=True)

    config_save_path = os.path.join(results_dir, "config.yaml")
    OmegaConf.save(cfg, config_save_path)
    if verbose:
        print(f"\nSaved config to: {config_save_path}")

    if split_file and os.path.exists(split_file):
        split_copy_path = os.path.join(results_dir, "train_test_split.json")
        shutil.copy(split_file, split_copy_path)
        if verbose:
            print(f"Copied split file to: {split_copy_path}")

    return results_dir


#: Keys a config may use for its model section. The per-organ runner nests its
#: model settings under `default_model`; the single-model entry point uses
#: `model`. Readers should not care which.
MODEL_SECTION_KEYS = ("default_model", "model")


def slurm_job_suffix():
    """`_job<SLURM_JOB_ID>` inside a Slurm job, else "".

    Experiment dirs are named to the second, so jobs that start together (an array,
    or a batch of submissions) would otherwise land in the same directory. Every
    array task has its own SLURM_JOB_ID.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    return f"_job{job_id}" if job_id else ""


def write_experiment_config(out_dir, cfg, args, sequence_names, config_file=None):
    """Record what produced an experiment directory, in one shape for both runners.

    The whole resolved config goes in, not a hand-picked subset. The metrics
    scripts and the test suite read `config.dataset.*` and the model section
    back out of this file, so picking fields by hand silently dropped every
    hyperparameter column for whichever runner forgot one.

    Returns:
        Path to the written file.
    """
    experiment_config = {
        "config_file": config_file,
        "cli_args": vars(args) if not isinstance(args, dict) else dict(args),
        # Effective, not just the CLI flag: GROWFIELDS_DETERMINISTIC=1 also turns it on.
        "deterministic": deterministic_enabled(),
        "n_sequences": len(sequence_names),
        "sequence_names": list(sequence_names),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    path = os.path.join(out_dir, "experiment_config.json")
    write_json(experiment_config, path, clean=True)
    return path


def read_model_config(experiment_config):
    """Return the model section of an `experiment_config.json`, whichever key it used.

    See `MODEL_SECTION_KEYS`. Returns an empty dict when there is no config at
    all, so callers can `.get()` straight off the result.
    """
    if not experiment_config:
        return {}
    config = experiment_config.get("config", {}) or {}
    for key in MODEL_SECTION_KEYS:
        section = config.get(key)
        if section:
            return section
    return {}


def extract_hyperparameters(experiment_dir):
    """Extract hyperparameters from experiment_config.json.

    `experiment_dir` may be a str or a Path; it is used as a Path either way.
    """
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        # Check in timestamped subdirectory
        subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            config_path = subdirs[0] / "experiment_config.json"

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Extract relevant hyperparameters from default_model
        default_model = read_model_config(config)
        dataset = config.get("config", {}).get("dataset", {})

        hparams = {
            # Architecture hyperparameters
            "hidden_features": default_model.get("hidden_features"),
            "hidden_layers": default_model.get("hidden_layers"),
            "lr": default_model.get("lr"),
            "w_global": default_model.get("w_global"),
            "w_local": default_model.get("w_local"),
            "finetune_lr": default_model.get("finetune_lr"),
            "finetune_enabled": default_model.get("finetune_enabled"),
            "model_type": default_model.get("model_type"),
            # Ablation parameters - alignment
            "aligned_leaves": dataset.get("aligned_leaves"),
        }

        # Remove None values
        hparams = {k: v for k, v in hparams.items() if v is not None}

        return hparams
    except Exception as e:
        print(f"Warning: Failed to extract hyperparameters from {config_path}: {e}")
        return {}


def load_results_json(run_dir):
    """
    Load runtime and storage metrics from a run's results.json.

    Args:
        run_dir: Directory holding results.json -- a sequence directory for a
            single run, or its run_N subdirectory when there are several.

    Returns:
        dict: 'training_time_seconds' and 'total_model_size_mb', or empty dict.
    """
    if run_dir is None:
        return {}

    results_json_path = os.path.join(str(run_dir), "results.json")

    if not os.path.exists(results_json_path):
        return {}

    try:
        with open(results_json_path) as f:
            results = json.load(f)

        # Per-organ runs nest these under aggregate_stats; the single-model entry
        # point writes them at the root. Same order as compute_metrics_fullplant.
        aggregate_stats = results.get("aggregate_stats", {})
        return {
            "training_time_seconds": aggregate_stats.get("training_time_seconds")
            or results.get("training_time_seconds"),
            "total_model_size_mb": aggregate_stats.get("total_model_size_mb")
            or results.get("total_model_size_mb"),
        }
    except Exception as e:
        print(f"Warning: Failed to load {results_json_path}: {e}")
        return {}
