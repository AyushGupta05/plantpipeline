"""Full-plant training: one model (or one chain of models) for the whole plant.

The per-organ counterpart lives in `models.organ_model_manager`; this module is
what `train_fullplant.py` drives. Each trainer owns its
whole stage -- train, write predictions, write weights -- so the entry point
only has to pick one out of `FULL_PLANT_TRAINERS`.

No metrics are computed here. Chamfer and EPE are recomputed from the saved
predictions by `scripts/compute_metrics_fullplant.py`, which is what produces
the published numbers; computing them here as well would mean two answers to
the same question with no way to tell which one a table came from.
"""

import os
from dataclasses import dataclass
from typing import Any

import torch

from growfields.models.siren_dynamics import hybrid_finetune, train_chained, train_continuous
from growfields.utils.io_utils import (
    save_gt_from_dataset,
    save_predictions_continuous,
    save_predictions_pairwise,
)


@dataclass
class SequenceArrays:
    """One sequence, as the parallel per-timepoint lists the trainers consume.

    Built by the entry point from `pipeline.data`; grouped here so a trainer
    signature stays readable instead of taking eleven positional tensors.
    """

    clouds: list
    labels: list
    normals: list
    is_train_masks: list
    min_xyz: Any
    scale: Any
    days: list
    full_day_range: tuple
    gt_tips: list
    gt_tip_labels: list
    cloud_paths: list
    dataset: Any


def train_pairwise_full_plant(cfg, data, results_dir, *, device, use_wandb):
    """Train a chain of frame-to-frame models, optionally jointly fine-tuned.

    Fine-tuning rewrites the models in place, so the pre-finetune predictions
    are written before it runs. Both stages end up on disk, and the metrics
    script reports whichever the caller asks for.
    """
    models = train_chained(
        data.clouds,
        n_steps=cfg.model.n_steps,
        lr=cfg.model.lr,
        n_samples=cfg.model.n_samples,
        h_f=cfg.model.hidden_features,
        h_layers=cfg.model.hidden_layers,
        omega_0=cfg.model.omega_0,
        device=device,
        verbose=cfg.output.verbose,
        use_wandb=use_wandb,
    )

    def save_predictions(stage):
        if not cfg.output.save_predictions:
            return
        print("\n" + "=" * 80)
        print(f"Saving {stage.replace('_', '-')} predictions...")
        print("=" * 80)
        out_dir = os.path.join(results_dir, "full_plant", stage)
        save_predictions_pairwise(
            models,
            data.clouds,
            data.labels,
            data.min_xyz,
            data.scale,
            out_dir,
            cloud_paths=data.cloud_paths,
            gt_tips=data.gt_tips,
            gt_tip_labels=data.gt_tip_labels,
            is_train=data.is_train_masks,
            normals=data.normals,
            save_normals=cfg.output.get("save_normals", True),
        )
        save_gt_from_dataset(data.dataset, out_dir, cloud_paths=data.cloud_paths)

    save_predictions("pre_finetune")

    if cfg.model.finetune_enabled:
        print("\n" + "=" * 80)
        print("Joint fine-tuning...")
        print("=" * 80)
        hybrid_finetune(
            models,
            data.clouds,
            data.min_xyz,
            data.scale,
            n_steps=cfg.model.finetune_steps,
            lr=cfg.model.finetune_lr,
            w_global=cfg.model.w_global,
            w_local=cfg.model.w_local,
            n_samples=cfg.model.n_samples,
            device=device,
            verbose=cfg.output.verbose,
            use_wandb=use_wandb,
        )
        save_predictions("post_finetune")

    _save_models(cfg, results_dir, models)


def train_continuous_full_plant(cfg, data, results_dir, *, device, use_wandb):
    """Train one time-conditioned model covering the whole sequence."""
    use_day_based_tnorm = cfg.model.get("use_day_based_tnorm", True)
    ode_kwargs = dict(
        use_odeint=cfg.model.get("use_odeint", False),
        ode_method=cfg.model.get("ode_method", "dopri5"),
        ode_rtol=cfg.model.get("ode_rtol", 1e-4),
        ode_atol=cfg.model.get("ode_atol", 1e-5),
        ode_n_steps=cfg.model.get("ode_n_steps", None),
    )

    model, _loss_logs = train_continuous(
        data.clouds,
        data.min_xyz,
        data.scale,
        frame_days=data.days,
        full_day_range=data.full_day_range,
        use_day_based_tnorm=use_day_based_tnorm,
        n_steps=cfg.model.n_steps,
        lr=cfg.model.lr,
        n_samples=cfg.model.n_samples,
        h_f=cfg.model.hidden_features,
        h_layers=cfg.model.hidden_layers,
        first_omega_0=cfg.model.first_omega_0,
        hidden_omega_0=cfg.model.hidden_omega_0,
        device=device,
        verbose=cfg.output.verbose,
        log_interval=cfg.model.log_interval,
        use_wandb=use_wandb,
        use_adjoint=cfg.model.get("use_adjoint", False),
        use_multistep_loss=cfg.model.get("use_multistep_loss", False),
        direction=cfg.model.get("direction", "forward"),
        **ode_kwargs,
    )

    if cfg.output.save_predictions:
        print("\n" + "=" * 80)
        print("Saving predictions...")
        print("=" * 80)
        out_dir = os.path.join(results_dir, "full_plant")
        save_predictions_continuous(
            model,
            data.clouds,
            data.labels,
            data.min_xyz,
            data.scale,
            out_dir,
            frame_days=data.days,
            full_day_range=data.full_day_range,
            use_day_based_tnorm=use_day_based_tnorm,
            cloud_paths=data.cloud_paths,
            gt_tips=data.gt_tips,
            gt_tip_labels=data.gt_tip_labels,
            is_train=data.is_train_masks,
            normals=data.normals,
            save_normals=cfg.output.get("save_normals", True),
            **ode_kwargs,
        )
        save_gt_from_dataset(data.dataset, out_dir, cloud_paths=data.cloud_paths)

    _save_models(cfg, results_dir, [model])


def _save_models(cfg, results_dir, models):
    """Write model weights.

    A pairwise run has one model per transition, a continuous run has one for the
    whole sequence; the filenames say which, so `generate_interpolation.py` can
    reload either from the directory alone.
    """
    if not cfg.output.save_models:
        return

    print("\n" + "=" * 80)
    print("Saving models...")
    print("=" * 80)
    model_dir = os.path.join(results_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    if cfg.model.model_type == "pairwise":
        for i, model in enumerate(models):
            torch.save(
                model.state_dict(), os.path.join(model_dir, f"model_{i}_to_{i + 1}.pth")
            )
        print(f"✅ Saved {len(models)} pairwise models to {model_dir}")
    else:
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(models[0].state_dict(), model_path)
        print(f"✅ Saved continuous model to {model_path}")


#: Full-plant trainers by ``model.model_type``.
FULL_PLANT_TRAINERS = {
    "pairwise": train_pairwise_full_plant,
    "continuous": train_continuous_full_plant,
}


def train_full_plant(cfg, data, results_dir, *, device, use_wandb):
    """Run the trainer registered for `cfg.model.model_type`."""
    model_type = cfg.model.model_type
    if model_type not in FULL_PLANT_TRAINERS:
        raise ValueError(f"Unknown model type: {model_type}")
    FULL_PLANT_TRAINERS[model_type](
        cfg, data, results_dir, device=device, use_wandb=use_wandb
    )
