"""Weights & Biases run handling, shared by both entry points.

Both runners log one W&B run per sequence -- per run, with `--n_runs > 1`, since
each run restarts its step count and W&B drops steps that go backwards -- so the
settings lookup and the init call live here rather than being written out twice.
"""

import os

from omegaconf import OmegaConf

try:
    import wandb
except ImportError:
    wandb = None


def wandb_settings(cfg):
    """Return (settings dict, enabled) for a config's `wandb` section.

    Enabled means both that the config asks for it and that the package is
    importable -- a missing `wandb` install disables logging rather than
    failing the run.
    """
    settings = (
        OmegaConf.to_container(cfg.get("wandb", {}), resolve=True)
        if "wandb" in cfg
        else {}
    )
    return settings, bool(settings.get("enabled", False)) and wandb is not None


def start_sequence_run(cfg, settings, sequence_name, experiment_name, out_dir, run_idx=None):
    """Start a W&B run for one sequence, closing any run still open.

    The whole resolved config is logged, plus the names needed to find the run
    again: which sequence it is, which experiment it belongs to, and with
    `run_idx`, which of its runs. Runs of one sequence share its W&B group.
    """
    if wandb is None:
        return None

    if wandb.run is not None:
        wandb.finish()

    config = OmegaConf.to_container(cfg, resolve=True)
    config["sequence_name"] = sequence_name
    config["experiment_name"] = experiment_name
    name = f"{experiment_name}_{sequence_name}"
    if run_idx is not None:
        config["run_idx"] = run_idx
        name += f"_run{run_idx}"
    os.makedirs(out_dir, exist_ok=True)

    return wandb.init(
        project=settings.get("project", "growfields"),
        entity=settings.get("entity", None),
        config=config,
        tags=list(settings.get("tags", [])),
        notes=settings.get("notes", None),
        group=settings.get("group", sequence_name),
        name=name,
        dir=out_dir,
        reinit=True,
    )


def finish_run():
    """Close the active W&B run, if there is one."""
    if wandb is not None and wandb.run is not None:
        wandb.finish()
