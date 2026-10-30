"""Shared configuration, seeding, and split-file helpers.

Extracted from the two entry points, which each carried byte-identical
copies of `set_seed`. Split-file helpers live in `pipeline.splits`.
"""

import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

DEVICE = (
    torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
)


DETERMINISTIC_ENV = "GROWFIELDS_DETERMINISTIC"


def deterministic_enabled():
    """True when training uses the deterministic Chamfer loss; see `enable_deterministic`."""
    return os.environ.get(DETERMINISTIC_ENV) == "1"


def enable_deterministic():
    """Make training bitwise repeatable for a given seed, GPU model and library stack.

    Swaps pytorch3d's Chamfer loss, whose KNN backward accumulates with CUDA
    atomics, for the atomic-free `metrics.chamfer_loss`, and turns on
    `torch.use_deterministic_algorithms`. About 30% slower, and the explicit
    distance tensor raises peak GPU memory (1.7x for GrowFields' paper config). The switch lives in
    the environment so `chamfer_loss` sees it wherever it is called from.
    """
    os.environ[DETERMINISTIC_ENV] = "1"
    # cuBLAS reads this when it creates its handle, at the first matmul, so
    # setting it here is early enough as long as nothing has run on the GPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic_enabled():
        enable_deterministic()


def load_config(config_path: str = None, cli_args: list = None):
    """
    Load configuration from YAML file and merge with CLI overrides.

    Args:
        config_path: Path to YAML config file (optional)
        cli_args: List of CLI arguments for overrides (optional)

    Returns:
        OmegaConf DictConfig object
    """
    # Determine which base config to use
    if config_path:
        # User provided a config file
        base_cfg = OmegaConf.load(config_path)
    else:
        # No config file - need to determine model type from CLI args
        model_type = "pairwise"  # default
        if cli_args:
            for arg in cli_args:
                if arg.startswith("model.model_type="):
                    model_type = arg.split("=")[1]
                    break

        # Load appropriate default
        if model_type == "continuous":
            base_cfg = OmegaConf.load("configs/full_plant/continuous.yaml")
        else:
            base_cfg = OmegaConf.load("configs/full_plant/pairwise.yaml")

    return merge_cli_overrides(base_cfg, cli_args)


def merge_cli_overrides(cfg, cli_args):
    """Merge Hydra-style `key=value` overrides into a config.

    The entry points take two kinds of argument: real flags (`--seed 0`) and
    config overrides (`model.n_steps=5000`). argparse only knows the flags, so
    they parse with `parse_known_args` and treat whatever it did not recognise
    as overrides. The catch is that a *misspelled flag* is indistinguishable
    from an override -- it lands in the same leftover pile, where

        OmegaConf.from_dotlist(["--gpu", "0"]) == {"--gpu": None, "0": None}

    merges a key named `--gpu` into the config, nothing reads it, and the run
    exits 0 having done nothing the user asked for. Hence: every leftover must
    contain `=`.

    NOTE: A mistyped *key* still gets through -- `dataset.sequenc=maize` leaves
    `dataset.sequence` at its default and trains on the wrong thing silently.
    Rejecting keys absent from the config would catch it, but the configs are
    deliberately sparse: `dataset.n_points` is a documented override that
    appears in none of them, and each model family declares only its own keys.
    Validating against them would reject working commands, so it is not done.
    """
    if not cli_args:
        return cfg
    stray = [a for a in cli_args if "=" not in a]
    if stray:
        raise SystemExit(
            f"Not config overrides: {' '.join(stray)}\n"
            "Overrides are key=value, e.g. model.n_steps=5000. If you meant a "
            "flag, check its spelling against --help."
        )
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_args))
