"""
OrganModelManager for per-organ training in plant growth modeling.

This module implements the core functionality for training separate models
per organ (stem, individual leaves) with organ-specific hyperparameters and
model types.

Key Features:
- Extract and normalize organ-specific point sequences
- Create and train separate models per organ
- Support different model types (pairwise vs continuous) per organ
- Combine predictions from organ models back to plant space
"""

import copy
import json
import os

import numpy as np
import torch

from growfields.models.encoder_decoder import train_encoder_decoder_continuous
from growfields.models.organ_ablations import (
    apply_holdout_last_n,
    apply_train_holes,
    save_debug_noise_plys,
)
from growfields.models.integration import _integrate_segment, denormalise, make_t_norm_helpers
from growfields.models.siren import ContinuousSiren, PairwiseSiren
from growfields.models.siren_dynamics import (
    hybrid_finetune,
    train_continuous,
    train_pairwise,
    train_shared_continuous,
)
from growfields.utils.io_utils import (
    estimate_normals,
    write_json,
    save_full_plant_from_organ_models,
    save_ply_with_organ_labels,
)


def _new_pairwise_siren(config, device):
    return PairwiseSiren(
        h_f=config["hidden_features"],
        h_layers=config["hidden_layers"],
        omega_0=config["omega_0"],
    ).to(device)


def _new_continuous_siren(config, device):
    return ContinuousSiren(
        h_f=config["hidden_features"],
        h_layers=config["hidden_layers"],
        first_omega_0=config["first_omega_0"],
        hidden_omega_0=config["hidden_omega_0"],
    ).to(device)


#: How to build the model for one organ, keyed by ``model_type``.
#:
#: The two global variants train a single model over all organs at once, so they
#: have nothing to build per organ -- their model is created inside the trainer
#: and stored on the manager. Registering them as ``None`` keeps every supported
#: model type in one place rather than splitting the check across call sites.
ORGAN_MODEL_FACTORIES = {
    "pairwise": _new_pairwise_siren,
    "continuous": _new_continuous_siren,
    "shared_continuous": lambda config, device: None,
    "encoder_decoder_continuous": lambda config, device: None,
}


def _shared_continuous_kwargs(config):
    """Trainer arguments specific to the organ-conditioned shared SIREN."""
    return dict(
        embed_dim=config.get("embed_dim", 16),
        conditioning=config.get("conditioning", "concat"),
        embed_init=config.get("embed_init", "normal"),
        n_steps=config.get("n_steps", 50000),
        lr=config.get("lr", 1e-5),
        lr_embed=config.get("lr_embed", None),
        n_samples=config.get("n_samples", 2048),
        organs_per_step=config.get("organs_per_step", 1),
        h_f=config.get("hidden_features", 256),
        h_layers=config.get("hidden_layers", 5),
        first_omega_0=config.get("first_omega_0", 30.0),
        hidden_omega_0=config.get("hidden_omega_0", 30.0),
        embed_reg_lambda=config.get("embed_reg_lambda", 1e-4),
        use_multistep_loss=config.get("use_multistep_loss", False),
        # ODE integration
        use_odeint=config.get("use_odeint", False),
        ode_method=config.get("ode_method", "dopri5"),
        ode_rtol=config.get("ode_rtol", 1e-4),
        ode_atol=config.get("ode_atol", 1e-5),
        use_adjoint=config.get("use_adjoint", True),
        direction=config.get("direction", "forward"),
        train_integration_steps=config.get("train_integration_steps", 1),
        ode_n_steps=config.get("ode_n_steps", None),
        # monotone velocity factorisation (plantpipeline addition; "free" = original model)
        velocity_param=config.get("velocity_param", "free"),
        lambda_w=config.get("lambda_w", 0.0),
        lambda_g=config.get("lambda_g", 0.0),
        base_eps=config.get("base_eps", 1e-3),
        g_bias_init=config.get("g_bias_init", -2.0),
    )


def _encoder_decoder_kwargs(config):
    """Trainer arguments specific to the encoder-decoder model."""
    return dict(
        latent_dim=config.get("latent_dim", 128),
        encoder_hidden_dim=config.get("encoder_hidden_dim", 128),
        decoder_dims=config.get("decoder_dims", [256, 256, 256, 256, 256]),
        decoder_skip_in=config.get("decoder_skip_in", [3]),
        zero_init_last=config.get("zero_init_last", False),
        n_steps=config.get("n_steps", 50000),
        lr=config.get("lr", 1e-5),
        lr_encoder=config.get("lr_encoder", None),
        decoder_weight_decay=config.get("decoder_weight_decay", 0.0),
        n_samples=config.get("n_samples", 2048),
        n_enc_samples=config.get("n_enc_samples", 1024),
        organs_per_step=config.get("organs_per_step", 1),
        use_multistep_loss=config.get("use_multistep_loss", False),
        use_latent_reg=config.get("use_latent_reg", False),
        latent_reg_lambda=config.get("latent_reg_lambda", 1e-4),
        train_integration_steps=config.get("train_integration_steps", 1),
        # ODE integration
        use_odeint=config.get("use_odeint", False),
        ode_rtol=config.get("ode_rtol", 1e-4),
        ode_atol=config.get("ode_atol", 1e-5),
        ode_method=config.get("ode_method", "dopri5"),
        ode_n_steps=config.get("ode_n_steps", None),
        use_adjoint=config.get("use_adjoint", False),
    )


#: Model types trained once over all organs rather than once per organ.
#:
#: ``attr`` is where the single trained model is stored on the manager; the
#: evaluation and saving code reads it from there.
GLOBAL_MODEL_TRAINERS = {
    "shared_continuous": {
        "label": "shared organ-conditioned",
        "attr": "shared_model",
        "train_fn": train_shared_continuous,
        "kwargs_fn": _shared_continuous_kwargs,
    },
    "encoder_decoder_continuous": {
        "label": "encoder-decoder organ",
        "attr": "encoder_decoder_model",
        "train_fn": train_encoder_decoder_continuous,
        "kwargs_fn": _encoder_decoder_kwargs,
    },
}




class OrganModelManager:
    """
    Manages per-organ models with dynamic organ appearance/disappearance.

    Responsibilities:
    - Extract organ-specific point sequences from plant data
    - Normalize each organ sequence independently
    - Create and train models for each organ
    - Combine predictions back to plant space
    - Handle newly appearing organs
    """

    def __init__(
        self,
        dataset,
        default_config: dict,
        organ_specific_configs: dict = None,
        device="cuda",
        train_holdout_last_n: int = None,
        save_normals: bool = True,
    ):
        """
        Args:
            dataset: TrackPlant3DDataset with mode='plant', normalize=False
            default_config: Default model config used for all organs unless overridden
            organ_specific_configs: Optional dict mapping organ_id/type -> config
                Example:
                {
                    'stem': {...},  # Override for stem (organ_id=0)
                    'leaf_1': {...},  # Override for specific leaf
                }
                Note: All organs use default_config unless explicitly overridden
            device: torch device
            save_normals: Estimate normals for every predicted cloud saved
                (`estimate_normals`); the models predict positions only.
        """
        self.dataset = dataset
        self.save_normals = save_normals
        self.default_config = default_config
        self.organ_configs = organ_specific_configs or {}
        self.organ_models = {}  # organ_id -> model info
        self.organ_sequences = {}  # organ_id -> extracted sequences
        self.organ_normalizations = {}  # organ_id -> (min, scale)
        self.organ_leaf_tips = {}  # organ_id -> list of tip positions (original space)
        self.organ_transformations = {}  # organ_id -> transformation matrices (for aligned leaves)
        self.organ_transformations_full = {}  # Full transformations for all days (for interpolation)
        self.organ_transformation_days = {}  # Maps organ_id -> list of days for each transformation
        self.organ_train_days = {}      # organ_id -> [day0, day2, ...] training days when organ present
        self.organ_full_day_range = {}  # organ_id -> (first_day, last_day) from full unstrided data
        self.shared_model = None  # For shared_continuous model type
        self.encoder_decoder_model = None  # For encoder_decoder_continuous model type
        self.global_day_range = None  # Set when use_global_tnorm=True for shared model
        self.use_wandb = False
        self.use_day_based_tnorm = default_config.get("use_day_based_tnorm", True)
        self.device = device
        self.debug_noise_dir = None  # Set externally to enable per-organ debug PLY saves
        self.debug_holes_dir = default_config.get("debug_holes_dir", None)
        self.label_field = getattr(dataset, 'label_field', 'labels')

        # Extract timepoint day indices from dataset
        self.timepoint_days = self._extract_timepoint_days()

        # Store transformation matrices if available (for aligned leaves)
        if hasattr(dataset, 'leaf_transformations') and dataset.leaf_transformations:
            self.organ_transformations = dataset.leaf_transformations
            print(f"  Loaded transformations for {len(self.organ_transformations)} organs")

        # Store full transformations for interpolation evaluation
        if hasattr(dataset, 'leaf_transformations_full') and dataset.leaf_transformations_full:
            self.organ_transformations_full = dataset.leaf_transformations_full
            self.organ_transformation_days = dataset.leaf_transformation_days

        # Validate dataset
        if self.dataset.mode != "plant":
            raise ValueError(
                "OrganModelManager requires dataset with mode='plant', "
                f"got mode='{self.dataset.mode}'"
            )

        if self.dataset.normalize:
            print(
                "WARNING: Dataset has normalize=True. For per-organ training, "
                "it's recommended to use normalize=False to avoid double normalization."
            )

        # Extract organ sequences and normalize each independently
        self._extract_and_normalize_organs()

        # Mask out held-out timepoints before computing day info
        if train_holdout_last_n:
            apply_holdout_last_n(
                self.organ_sequences, self.dataset.nt, train_holdout_last_n
            )

        # Apply spatial holes to training clouds (robustness experiment)
        train_holes_n = default_config.get("train_holes_n", 0)
        if train_holes_n:
            apply_train_holes(
                self.organ_sequences,
                self.organ_normalizations,
                self.timepoint_days,
                train_holes_n,
                default_config.get("train_holes_k", 10),
                debug_dir=self.debug_holes_dir,
            )

        # Compute per-organ day info for day-based t_norm
        self._compute_organ_day_info()

        # Extract leaf tips per organ
        self._extract_organ_leaf_tips()

    def _extract_timepoint_days(self):
        """
        Extract actual timepoint day indices from dataset.

        Returns:
            List of day indices (integers) or None if not available
        """
        if (
            not hasattr(self.dataset, "sequence_data")
            or self.dataset.sequence_data is None
        ):
            return None

        timepoints = self.dataset.sequence_data.get("timepoints", [])
        if not timepoints:
            return None

        days = []
        for tp in timepoints:
            if "day" in tp and tp["day"] is not None:
                days.append(int(tp["day"]))
            else:
                # If no day info, use sequential indices
                days.append(len(days))

        # Verify days match the number of timepoints
        if len(days) != self.dataset.nt:
            raise ValueError(
                f"Mismatch between extracted days ({len(days)}) and dataset timepoints "
                f"({self.dataset.nt}). This likely indicates that sequence_data['timepoints'] "
                f"was not updated after filtering. Check TrackPlant3DDataset._select_timepoints()."
            )

        return days

    def _compute_organ_day_info(self):
        """
        Compute per-organ training days and full day range for day-based t_norm.

        organ_train_days: days when each organ is present in training data.
        organ_full_day_range: (first_day, last_day) from full unstrided data,
            used to normalize time as t_norm = (day - first_day) / (last_day - first_day).

        Full day range is determined from the full (unstrided) plant timeseries by
        checking which timepoints contain each organ's label. This is uniform for
        all organs and doesn't depend on alignment transformations existing.
        """
        # Get full plant timeseries (unstrided) to determine organ presence across all days
        plant_dataset = self.dataset.plant_dataset
        full_plant_sequence = plant_dataset.get_timeseries_by_sequence_name(
            self.dataset.sequence_name
        )
        full_timepoints = full_plant_sequence["timepoints"]

        for organ_id, organ_seq in self.organ_sequences.items():
            # Training days: days when organ is present in training split
            days = [self.timepoint_days[t] for t, present
                    in enumerate(organ_seq["presence_train"]) if present]
            self.organ_train_days[organ_id] = days

            # Full day range: check all timepoints in full unstrided data for organ presence
            organ_present_days = [
                tp["day"] for tp in full_timepoints
                if organ_id in tp.get(self.label_field, tp.get("labels", []))
            ]
            self.organ_full_day_range[organ_id] = (min(organ_present_days), max(organ_present_days))

            print(f"  Organ {organ_id}: train_days={self.organ_train_days[organ_id]}, "
                  f"full_day_range={self.organ_full_day_range[organ_id]}")

    def _apply_normalization(self, pts_list, min_xyz, scale):
        """
        Apply normalization transform to point list.

        Args:
            pts_list: List of [N_t, 3] tensors or None
            min_xyz: Normalization min [3]
            scale: Normalization scale (scalar)

        Returns:
            List of normalized tensors
        """
        if min_xyz is None or scale is None:
            return pts_list

        normalized = []
        for pts in pts_list:
            if pts is not None:
                pts_norm = (pts - min_xyz) / scale * 2 - 1
                normalized.append(pts_norm)
            else:
                normalized.append(None)
        return normalized

    def _normalize_organ_sequence_traintest(self, train_pts_list, test_pts_list):
        """
        Normalize organ sequence using ONLY train points.

        Computes normalization parameters from train set, applies to both train and test.
        Logs warning if test points exceed expected range.

        Args:
            train_pts_list: List of [N_t, 3] tensors or None
            test_pts_list: List of [M_t, 3] tensors or None

        Returns:
            train_normalized, test_normalized, min_xyz, scale
        """
        # Compute normalization from train only
        valid_train = [pts for pts in train_pts_list if pts is not None]
        if len(valid_train) == 0:
            return train_pts_list, test_pts_list, None, None

        all_train_concat = torch.cat(valid_train, dim=0)
        min_xyz = all_train_concat.min(dim=0)[0]
        max_xyz = all_train_concat.max(dim=0)[0]
        scale = (max_xyz - min_xyz).max()

        # Apply to both train and test
        train_norm = self._apply_normalization(train_pts_list, min_xyz, scale)
        test_norm = self._apply_normalization(test_pts_list, min_xyz, scale)

        # Check if test is out of range
        valid_test = [pts for pts in test_norm if pts is not None]
        if len(valid_test) > 0:
            test_concat = torch.cat(valid_test, dim=0)
            test_max = test_concat.abs().max().item()
            if test_max > 1.2:
                print(f"  ⚠️  Warning: test points outside expected range: {test_max:.2f}")

        return train_norm, test_norm, min_xyz, scale

    def _extract_and_normalize_organs(self):
        """
        Extract organ-specific point sequences and normalize each independently.

        For each organ:
        1. Extract train and test points separately with organ_id label across all timepoints
        2. Compute organ-specific normalization using ONLY train points (min, scale)
        3. Normalize both train and test sequences to [-1, 1]
        4. Store normalized sequences and normalization params
        """
        print("\n" + "=" * 80)
        print("Extracting and normalizing organ sequences...")
        print("=" * 80)

        # Get organ IDs present in the dataset
        organ_ids = self.dataset.get_organ_ids()
        print(f"Found {len(organ_ids)} organs: {organ_ids}")

        for organ_id in organ_ids:
            # Determine organ type
            organ_type = "stem" if organ_id == 0 else f"leaf_{organ_id}"
            print(f"\nProcessing organ {organ_id} ({organ_type})...")

            # Extract organ sequence across all timepoints - KEEP TRAIN/TEST SEPARATE
            train_organ_pts = []
            test_organ_pts = []
            train_organ_normals = []
            test_organ_normals = []

            for t in range(self.dataset.nt):
                # Extract train points for this organ
                train_mask = self.dataset.train_labels[t] == organ_id
                if train_mask.any():
                    train_organ_pts.append(self.dataset.train_pts[t][train_mask])
                    train_organ_normals.append(self.dataset.train_normals[t][train_mask])
                else:
                    train_organ_pts.append(None)
                    train_organ_normals.append(None)

                # Extract test points for this organ
                test_mask = self.dataset.test_labels[t] == organ_id
                if test_mask.any():
                    test_organ_pts.append(self.dataset.test_pts[t][test_mask])
                    test_organ_normals.append(self.dataset.test_normals[t][test_mask])
                else:
                    test_organ_pts.append(None)
                    test_organ_normals.append(None)

            # Count presence in train and test
            n_train_present = sum([pts is not None for pts in train_organ_pts])
            n_test_present = sum([pts is not None for pts in test_organ_pts])
            print(f"  Present in train: {n_train_present}/{self.dataset.nt} timepoints")
            print(f"  Present in test: {n_test_present}/{self.dataset.nt} timepoints")

            # Normalize using ONLY train points
            train_normalized, test_normalized, min_xyz, scale = \
                self._normalize_organ_sequence_traintest(train_organ_pts, test_organ_pts)

            if min_xyz is not None:
                print(
                    f"  Normalization (from train): min={min_xyz.cpu().numpy()}, scale={scale.item():.4f}"
                )

            # Store both train and test
            self.organ_sequences[organ_id] = {
                "train_points": train_normalized,
                "test_points": test_normalized,
                "train_normals": train_organ_normals,
                "test_normals": test_organ_normals,
                "presence_train": [pts is not None for pts in train_normalized],
                "presence_test": [pts is not None for pts in test_normalized],
                "type": organ_type,
            }
            self.organ_normalizations[organ_id] = (min_xyz, scale)

        print("\n✅ Organ extraction and normalization complete!")

    def _extract_organ_leaf_tips(self):
        """
        Extract leaf tip positions per organ from the dataset.

        For each organ (that is a leaf), stores list of tip positions in original space.
        """
        if (
            not hasattr(self.dataset, "leaf_tips_original")
            or self.dataset.leaf_tips_original is None
        ):
            print("  No leaf tips available in dataset")
            return

        gt_tips = self.dataset.leaf_tips_original
        gt_tip_labels = (
            self.dataset.leaf_tip_labels_original
            if hasattr(self.dataset, "leaf_tip_labels_original")
            else None
        )

        if gt_tip_labels is None:
            print("  No leaf tip labels available")
            return

        # Initialize tip storage for each organ
        for organ_id in self.organ_sequences.keys():
            self.organ_leaf_tips[organ_id] = [None] * self.dataset.nt

        # Extract tips per organ
        for t in range(self.dataset.nt):
            if gt_tips[t] is not None and gt_tip_labels[t] is not None:
                tips_t = (
                    gt_tips[t]
                    if isinstance(gt_tips[t], np.ndarray)
                    else gt_tips[t].cpu().numpy()
                )
                labels_t = (
                    gt_tip_labels[t]
                    if isinstance(gt_tip_labels[t], np.ndarray)
                    else gt_tip_labels[t].cpu().numpy()
                )

                for tip, label in zip(tips_t, labels_t):
                    organ_id = int(label)
                    if organ_id in self.organ_leaf_tips:
                        self.organ_leaf_tips[organ_id][t] = torch.tensor(
                            tip, device=self.device, dtype=torch.float32
                        )

    def get_or_create_model(self, organ_id: int):
        """Get existing model or create new one."""
        if organ_id not in self.organ_models:
            # Determine organ type
            organ_type = "stem" if organ_id == 0 else "leaf"

            # Determine config for this organ
            config = self._get_organ_config(organ_id)

            model_type = config["model_type"]
            if model_type not in ORGAN_MODEL_FACTORIES:
                raise ValueError(f"Unknown model type: {model_type}")
            model = ORGAN_MODEL_FACTORIES[model_type](config, self.device)

            self.organ_models[organ_id] = {
                "model": model,
                "config": config,
                "type": organ_type,
            }

        return self.organ_models[organ_id]

    def _get_organ_config(self, organ_id: int):
        """Determine config for a specific organ."""
        # Priority:
        # 1. Organ-specific config (e.g., 'stem', 'leaf_1', 'leaf_2')
        # 2. Global default config (applies to all organs)

        if organ_id == 0:  # Stem
            key = "stem"
        else:
            key = f"leaf_{organ_id}"

        if key in self.organ_configs:
            # Merge organ-specific config with defaults (to fill in missing keys)
            config = {**self.default_config, **self.organ_configs[key]}
            return config
        else:
            return self.default_config.copy()




    def _finetune_organ_model(self, organ_id: int):
        """
        Fine-tune a pairwise organ model (only applicable to pairwise models).

        Args:
            organ_id: ID of the organ to fine-tune
        """
        organ_info = self.organ_models[organ_id]
        config = organ_info["config"]

        if config["model_type"] != "pairwise":
            print(f"    ⚠️  Fine-tuning only supported for pairwise models, skipping")
            return

        if not config.get("finetune_enabled", False):
            return

        organ_seq = self.organ_sequences[organ_id]
        valid_clouds = [c.unsqueeze(0) for c in organ_seq["train_points"] if c is not None]
        input_noise_std = self.default_config.get("input_noise_std", 0.0)
        if input_noise_std > 0:
            valid_clouds = [c + torch.randn_like(c) * input_noise_std for c in valid_clouds]

        if len(valid_clouds) < 2:
            return

        models = organ_info.get("trained_models", [])
        if not models:
            return

        # Get normalization
        min_xyz, scale = self.organ_normalizations[organ_id]
        min_xyz_batch = min_xyz.unsqueeze(0).unsqueeze(0)
        scale_batch = (
            torch.tensor([scale.item()], device=self.device).unsqueeze(0).unsqueeze(0)
        )

        print(f"\n    🔧 Fine-tuning pairwise models...")
        hybrid_finetune(
            models,
            valid_clouds,
            min_xyz_batch,
            scale_batch,
            n_steps=config.get("finetune_steps", 1500),
            lr=config.get("finetune_lr", 2e-5),
            lr_x0=config.get("finetune_lr_x0", 5e-6),
            w_global=config.get("w_global", 3.0),
            w_local=config.get("w_local", 1.0),
            n_samples=config.get("n_samples", 2048),
            device=self.device,
            verbose=True,
            optimise_x0=config.get("optimise_x0", False),
            log_interval=config.get("log_interval", 100),
            use_wandb=self.use_wandb,
            wandb_prefix=f"organ_{organ_id}",
        )

    def train_all_organs(self):
        """
        Train all organ models on their normalized sequences.

        For each organ:
        1. Get or create model
        2. Extract organ's normalized sequence
        3. Train model on organ sequence
        4. (Optional) Evaluate pre-finetune performance
        5. (Optional) Apply fine-tuning for pairwise models
        6. (Optional) Evaluate post-finetune performance
        7. Store trained model

        """
        # The global model types train once over all organs, not per organ.
        model_type = self.default_config.get("model_type")
        if model_type in GLOBAL_MODEL_TRAINERS:
            self._train_global_model(model_type)
            return

        print("\n" + "=" * 80)
        print("Training all organ models...")
        print("=" * 80)

        organ_ids = list(self.organ_sequences.keys())

        for organ_id in organ_ids:
            organ_seq = self.organ_sequences[organ_id]
            print(f"\n{'=' * 80}")
            print(f"Training organ {organ_id} ({organ_seq['type']})...")
            print(f"{'=' * 80}")

            organ_info = self.get_or_create_model(organ_id)
            config = organ_info["config"]

            # Get organ's normalized TRAIN sequence (train on train points only!)
            organ_clouds = organ_seq["train_points"]

            # Filter to only timepoints where organ is present in train
            valid_clouds = [c.unsqueeze(0) for c in organ_clouds if c is not None]
            input_noise_std = self.default_config.get("input_noise_std", 0.0)
            if input_noise_std > 0:
                valid_clouds = [c + torch.randn_like(c) * input_noise_std for c in valid_clouds]

            if len(valid_clouds) < 2:
                print(f"  ⚠️  Skipping organ {organ_id}: insufficient timepoints")
                continue

            print(f"  Model type: {config['model_type']}")
            print(f"  Training on {len(valid_clouds)} timepoints")

            if config["model_type"] == "pairwise":
                self._train_organ_pairwise(organ_id, valid_clouds)
            else:
                self._train_organ_continuous(organ_id, valid_clouds)

        print("\n" + "=" * 80)
        print("✅ All organ models trained!")
        print("=" * 80)

    def _train_organ_pairwise(self, organ_id, valid_clouds):
        """Train one frame-to-frame model per transition, then optionally fine-tune.

        Fine-tuning rewrites the models in place, so the pre-finetune copies are
        kept and both stages are recorded -- `save_organ_predictions` can then
        write predictions from either.
        """
        organ_info = self.organ_models[organ_id]
        config = organ_info["config"]

        models = []
        for i in range(1, len(valid_clouds)):
            print(f"\n  🌱 Training f_{i-1}->{i}")
            models.append(
                train_pairwise(
                    valid_clouds[i - 1][0],
                    valid_clouds[i][0],
                    n_steps=config["n_steps"],
                    lr=config["lr"],
                    n_samples=config["n_samples"],
                    h_f=config["hidden_features"],
                    h_layers=config["hidden_layers"],
                    omega_0=config["omega_0"],
                    device=self.device,
                    verbose=True,
                    use_wandb=self.use_wandb,
                    wandb_prefix=f"organ_{organ_id}",
                )
            )

        organ_info["trained_models"] = models
        print(f"  ✅ Trained {len(models)} pairwise models")

        if not config.get("finetune_enabled", False):
            return

        organ_info["pre_finetune_models"] = copy.deepcopy(models)
        print(f"\n    💾 Saved copy of pre-finetune models")

        self._finetune_organ_model(organ_id)

    def _train_organ_continuous(self, organ_id, valid_clouds):
        """Train one time-conditioned model covering this organ's whole sequence."""
        organ_info = self.organ_models[organ_id]
        config = organ_info["config"]

        min_xyz, scale = self.organ_normalizations[organ_id]
        min_xyz_batch = min_xyz.unsqueeze(0).unsqueeze(0)
        scale_batch = (
            torch.tensor([scale.item()], device=self.device).unsqueeze(0).unsqueeze(0)
        )

        model, loss_log = train_continuous(
            valid_clouds,
            min_xyz_batch,
            scale_batch,
            frame_days=self.organ_train_days[organ_id],
            full_day_range=self.organ_full_day_range[organ_id],
            use_day_based_tnorm=self.use_day_based_tnorm,
            n_steps=config["n_steps"],
            lr=config["lr"],
            n_samples=config["n_samples"],
            h_f=config["hidden_features"],
            h_layers=config["hidden_layers"],
            first_omega_0=config["first_omega_0"],
            hidden_omega_0=config["hidden_omega_0"],
            device=self.device,
            verbose=True,
            log_interval=config.get("log_interval", 200),
            use_wandb=self.use_wandb,
            wandb_prefix=f"organ_{organ_id}",
            use_multistep_loss=config.get("use_multistep_loss", False),
            # ODE integration
            use_odeint=config.get("use_odeint", False),
            ode_method=config.get("ode_method", "dopri5"),
            ode_rtol=config.get("ode_rtol", 1e-4),
            ode_atol=config.get("ode_atol", 1e-5),
            use_adjoint=config.get("use_adjoint", True),
            direction=config.get("direction", "forward"),
            train_integration_steps=config.get("train_integration_steps", 1),
            ode_n_steps=config.get("ode_n_steps", None),
        )
        organ_info["trained_model"] = model
        organ_info["loss_log"] = loss_log
        print(f"  ✅ Continuous model trained")


    def _collect_organ_data(self):
        """Gather the per-organ training clouds the global trainers consume.

        Organs present in fewer than two timepoints are skipped: a single frame
        carries no displacement to learn from. They still get an entry in
        `organ_models` later so the saving code can fall back to ground truth.
        """
        config = self.default_config
        input_noise_std = config.get("input_noise_std", 0.0)
        organ_data = {}

        for organ_id, organ_seq in self.organ_sequences.items():
            valid_clouds = [c.unsqueeze(0) for c in organ_seq["train_points"] if c is not None]
            if len(valid_clouds) < 2:
                print(f"  Skipping organ {organ_id}: insufficient timepoints ({len(valid_clouds)})")
                continue
            if input_noise_std > 0:
                valid_clouds = [c + torch.randn_like(c) * input_noise_std for c in valid_clouds]
                save_debug_noise_plys(
                    organ_seq,
                    organ_id,
                    self.organ_normalizations[organ_id],
                    valid_clouds,
                    self.organ_train_days[organ_id],
                    self.timepoint_days,
                    self.debug_noise_dir,
                )

            min_xyz, scale = self.organ_normalizations[organ_id]
            organ_data[organ_id] = {
                "clouds": valid_clouds,
                "min_xyz": min_xyz.unsqueeze(0).unsqueeze(0),
                "scale": (
                    torch.tensor([scale.item()], device=self.device).unsqueeze(0).unsqueeze(0)
                ),
                "frame_days": self.organ_train_days[organ_id],
                "full_day_range": self.organ_full_day_range[organ_id],
            }

        return organ_data

    def _apply_global_tnorm(self, organ_data):
        """Optionally put every organ on the plant's day range instead of its own.

        By default each organ's t_norm spans its own lifetime, so t=0.5 means a
        different day for a late-emerging leaf than for the stem. Enabling
        `use_global_tnorm` makes t comparable across organs, which matters for
        the models that condition one network on all of them.
        """
        if not self.default_config.get("use_global_tnorm", False):
            return

        ranges = [self.organ_full_day_range[organ_id] for organ_id in organ_data]
        self.global_day_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))
        print(f"  Using global t_norm: day range {self.global_day_range} for all organs")
        for organ in organ_data.values():
            organ["full_day_range"] = self.global_day_range

    def _register_global_model_organs(self, model, loss_logs, trained_organ_ids):
        """Point every organ's `organ_models` entry at the single global model.

        Organs skipped by `_collect_organ_data` still get an entry, with no
        trained model, so the save functions can apply their ground-truth
        fallbacks instead of raising on a missing key.
        """
        config = self.default_config

        for organ_id in trained_organ_ids:
            self.organ_models[organ_id] = {
                "model": None,  # the one global model lives on the manager
                "config": config.copy(),
                "type": "stem" if organ_id == 0 else "leaf",
                "trained_model": model,
                "loss_log": loss_logs,
            }

        for organ_id in self.organ_sequences:
            if organ_id not in self.organ_models:
                self.organ_models[organ_id] = {
                    "model": None,
                    "config": config.copy(),
                    "type": "stem" if organ_id == 0 else "leaf",
                    "trained_model": None,
                }

    def _train_global_model(self, model_type):
        """Train the one model that covers every organ.

        The shared organ-conditioned SIREN and the encoder-decoder differ only in
        which trainer they call, which hyperparameters that trainer takes, and
        where the result is stored -- see `GLOBAL_MODEL_TRAINERS`. Everything
        around that is the same, so it lives here once.
        """
        spec = GLOBAL_MODEL_TRAINERS[model_type]
        label = spec["label"]

        print("\n" + "=" * 80)
        print(f"Training {label} model...")
        print("=" * 80)

        organ_data = self._collect_organ_data()
        if not organ_data:
            print(f"  No organs with sufficient data for {label} training!")
            return

        self._apply_global_tnorm(organ_data)
        print(f"  Training on {len(organ_data)} organs: {sorted(organ_data.keys())}")

        model, loss_logs = spec["train_fn"](
            organ_data,
            device=self.device,
            verbose=True,
            use_wandb=self.use_wandb,
            log_interval=self.default_config.get("log_interval", 200),
            **spec["kwargs_fn"](self.default_config),
        )
        setattr(self, spec["attr"], model)

        self._register_global_model_organs(model, loss_logs, list(organ_data))

        print("\n" + "=" * 80)
        print(f"  {label.capitalize()} model trained!")
        print("=" * 80)

    def get_organ_info(self):
        """Get summary information about organs and their models."""
        info = {
            "n_organs": len(self.organ_sequences),
            "organs": {},
        }

        for organ_id, seq in self.organ_sequences.items():
            organ_info = {
                "type": seq["type"],
                "n_timepoints_present": sum(seq["presence_train"]),
                "n_total_timepoints": len(seq["presence_train"]),
            }

            if organ_id in self.organ_models:
                model_info = self.organ_models[organ_id]
                organ_info["model_type"] = model_info["config"]["model_type"]
                organ_info["trained"] = (
                    "trained_models" in model_info or "trained_model" in model_info
                )

            info["organs"][organ_id] = organ_info

        return info

    def save_organ_models(self, organ_id: int, out_dir: str):
        """
        Save trained model weights for a single organ.

        Args:
            organ_id: ID of the organ
            out_dir: Output directory for this organ
        """
        organ_info = self.organ_models.get(organ_id)
        if organ_info is None:
            return

        config = organ_info["config"]
        models_dir = os.path.join(out_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Get organ's valid timepoint indices (where organ is present)
        organ_seq = self.organ_sequences[organ_id]
        valid_timepoint_indices = [
            i for i, present in enumerate(organ_seq["presence_train"]) if present
        ]

        if config["model_type"] == "pairwise":
            models = organ_info.get("trained_models", [])
            for i, model in enumerate(models):
                # Model i goes from valid_timepoint_indices[i] to valid_timepoint_indices[i+1]
                t_from_idx = valid_timepoint_indices[i]
                t_to_idx = valid_timepoint_indices[i + 1]

                # Use actual day numbers if available
                if self.timepoint_days is not None:
                    day_from = self.timepoint_days[t_from_idx]
                    day_to = self.timepoint_days[t_to_idx]
                    filename = f"model_{day_from}_to_{day_to}.pth"
                else:
                    filename = f"model_{t_from_idx}_to_{t_to_idx}.pth"

                torch.save(model.state_dict(), os.path.join(models_dir, filename))

        elif config["model_type"] == "continuous":
            model = organ_info.get("trained_model")
            if model is not None:
                torch.save(
                    model.state_dict(), os.path.join(models_dir, "model_continuous.pth")
                )

    def save_global_model(self, base_out_dir: str):
        """
        Save the one model of a `GLOBAL_MODEL_TRAINERS` type, once per run.

        It goes to ``<base_out_dir>/models/``, beside ``organs/``. Runs before
        this layout stored an identical copy under every ``organs/*/models/``;
        `scripts/generate_interpolation.py` still reads those. Files are always
        overwritten: a run retrained into an existing dir must not keep the old
        weights.

        Args:
            base_out_dir: The run directory (the one holding ``organs/``)
        """
        model_type = self.default_config.get("model_type")
        model = getattr(self, GLOBAL_MODEL_TRAINERS[model_type]["attr"])
        if model is None:
            return

        models_dir = os.path.join(base_out_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        if model_type == "shared_continuous":
            torch.save(
                model.state_dict(), os.path.join(models_dir, "model_shared_continuous.pth")
            )
            emb = model.get_all_embeddings().cpu().numpy()
            np.save(os.path.join(models_dir, "organ_embeddings.npy"), emb)
            remap_path = os.path.join(models_dir, "organ_id_remap.json")
            if model.organ_id_remap is not None:
                with open(remap_path, "w") as f:
                    json.dump(
                        {str(k): v for k, v in model.organ_id_remap.items()}, f, indent=2
                    )
            elif os.path.exists(remap_path):
                # The loader trusts this file over the organ IDs it sees.
                os.remove(remap_path)

        elif model_type == "encoder_decoder_continuous":
            torch.save(
                model.state_dict(), os.path.join(models_dir, "model_encoder_decoder.pth")
            )

    def save_organ_predictions(
        self, organ_id: int, out_dir: str, save_ground_truth=True, use_pre_finetune_models=False
    ):
        """
        Save predicted point clouds for a single organ with leaf tips marked.

        Args:
            organ_id: ID of the organ to save
            out_dir: Output directory for this organ
            save_ground_truth: Whether to also save ground truth clouds
            use_pre_finetune_models: If True, use pre-finetune models (for pairwise only)
        """
        organ_info = self.organ_models.get(organ_id)
        if organ_info is None:
            print(f"  ⚠️  No model found for organ {organ_id}")
            return

        organ_seq = self.organ_sequences[organ_id]
        organ_type = organ_seq["type"]

        # Combine train and test clouds, tracking which points are train/test
        valid_clouds = []
        valid_is_train_masks = []  # Track which points are train (1) vs test (0)
        valid_normals = []  # Track GT normals (train + test concatenated)

        for train_pts, test_pts, train_norms, test_norms in zip(
            organ_seq["train_points"], organ_seq["test_points"],
            organ_seq["train_normals"], organ_seq["test_normals"]
        ):
            if train_pts is not None and test_pts is not None:
                # Concatenate train and test
                combined = torch.cat([train_pts, test_pts], dim=0)
                valid_clouds.append(combined)

                # Create is_train mask: 1 for train points, 0 for test points
                is_train_mask = torch.cat([
                    torch.ones(len(train_pts), dtype=torch.int32, device=self.device),
                    torch.zeros(len(test_pts), dtype=torch.int32, device=self.device)
                ])
                valid_is_train_masks.append(is_train_mask)

                # Concatenate GT normals
                if train_norms is not None and test_norms is not None:
                    combined_normals = torch.cat([train_norms, test_norms], dim=0)
                    valid_normals.append(combined_normals)
                else:
                    valid_normals.append(None)
            elif train_pts is not None:
                # Only train points present
                valid_clouds.append(train_pts)
                valid_is_train_masks.append(
                    torch.ones(len(train_pts), dtype=torch.int32, device=self.device)
                )
                valid_normals.append(train_norms)
            elif test_pts is not None:
                # Only test points present
                valid_clouds.append(test_pts)
                valid_is_train_masks.append(
                    torch.zeros(len(test_pts), dtype=torch.int32, device=self.device)
                )
                valid_normals.append(test_norms)
            else:
                # Neither present - skip
                pass

        if len(valid_clouds) < 2:
            return

        # Get valid timepoint indices (where organ is present in either train or test)
        valid_timepoint_indices = [
            i for i, (train_present, test_present) in enumerate(
                zip(organ_seq["presence_train"], organ_seq["presence_test"])
            ) if train_present or test_present
        ]

        # Get sequence name for filenames
        sequence_name = (
            self.dataset.sequence_name
            if hasattr(self.dataset, "sequence_name")
            else None
        )

        # Create directories
        pred_dir = os.path.join(out_dir, "predicted")
        gt_dir = os.path.join(out_dir, "ground_truth")
        os.makedirs(pred_dir, exist_ok=True)
        if save_ground_truth:
            os.makedirs(gt_dir, exist_ok=True)

        # Get normalization
        min_xyz, scale = self.organ_normalizations[organ_id]
        min_xyz_batch = min_xyz.unsqueeze(0).unsqueeze(0)
        scale_batch = (
            torch.tensor([scale.item()], device=self.device).unsqueeze(0).unsqueeze(0)
        )

        # Get leaf tips for this organ
        gt_tips_organ = self.organ_leaf_tips.get(organ_id, None)

        # Seed the organ's leaf tip from the first GT frame that has one, in the
        # organ's normalized space, so it can be warped forward with the point
        # cloud below. This is what ends up written as `is_leaf_tip`, so it is what
        # the per-organ EPE is computed from.
        # The expression is identical to `_apply_normalization`, which normalizes
        # the organ's points with the same `min_xyz` [3] and scalar `scale` from
        # `organ_normalizations[organ_id]`, so the tip lands in the same space as
        # the points the model operates on.
        tip_normalized = None
        tip_start_idx = None
        if gt_tips_organ is not None:
            for idx, tip in enumerate(gt_tips_organ):
                if tip is not None:
                    # Normalize tip
                    tip_normalized = (tip - min_xyz) / scale * 2 - 1
                    tip_normalized = tip_normalized.unsqueeze(0)
                    tip_start_idx = idx
                    break

        # --- Load leaf dataset for GT (avoids alignment round-trip) ---
        day_to_leaf_data = {}
        day_to_plant_tp = {}
        split_data = None
        if save_ground_truth:
            leaf_dataset = self.dataset.leaf_dataset
            leaf_sequences = leaf_dataset.get_timeseries_by_plant_sequence(
                self.dataset.sequence_name
            )
            for leaf_seq in leaf_sequences:
                if leaf_seq["leaf_id"] == organ_id:
                    for ltp in leaf_seq["timepoints"]:
                        day_to_leaf_data[ltp["day"]] = ltp
                    break

            # Build day -> plant timepoint mapping (for is_train derivation)
            plant_dataset_obj = leaf_dataset.plant_dataset
            full_plant_seq = plant_dataset_obj.get_timeseries_by_sequence_name(
                self.dataset.sequence_name
            )
            if full_plant_seq is not None:
                for tp in full_plant_seq["timepoints"]:
                    day_to_plant_tp[tp["day"]] = tp

                # For stem (organ_id == 0), use plant dataset directly
                if organ_id == 0:
                    for tp in full_plant_seq["timepoints"]:
                        stem_mask = np.array(tp.get(self.label_field, tp["labels"])) == 0
                        if np.any(stem_mask):
                            day_to_leaf_data[tp["day"]] = {
                                "points": tp["points"][stem_mask],
                                "normals": (
                                    tp["normals"][stem_mask]
                                    if tp.get("normals") is not None
                                    else None
                                ),
                            }

            # Load strided split file for is_train (indexed by strided timepoint)
            if (
                hasattr(self.dataset, "split_file")
                and self.dataset.split_file
                and os.path.isfile(self.dataset.split_file)
            ):
                with open(self.dataset.split_file) as f:
                    split_data = json.load(f)

        # Generate predictions
        advance, n_transitions = self._make_organ_stepper(
            organ_id, organ_info, valid_clouds, use_pre_finetune_models
        )
        if advance is None:
            return

        x = valid_clouds[0].clone()
        # Every prediction inherits is_train from frame 0: these models move the
        # initial points rather than producing new ones, so the split carries over.
        is_train_mask_pred = valid_is_train_masks[0].cpu().numpy()

        with torch.no_grad():
            for i in range(n_transitions):
                target_t_idx = valid_timepoint_indices[i + 1]

                x = advance(x, i)
                pred = (
                    denormalise(x.unsqueeze(0), min_xyz_batch, scale_batch)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Oriented against this organ's dataset cloud for the same frame,
                # which lives in the same space as the prediction.
                pred_normals = None
                if self.save_normals:
                    ref = {}
                    if valid_normals[i + 1] is not None:
                        ref = dict(
                            ref_points=denormalise(
                                valid_clouds[i + 1].unsqueeze(0), min_xyz_batch, scale_batch
                            )[0].cpu().numpy(),
                            ref_normals=valid_normals[i + 1].cpu().numpy(),
                        )
                    pred_normals = estimate_normals(pred, **ref)

                # Tips are only written once they have actually been warped, i.e.
                # after the timepoint they first appear at.
                pred_tip = None
                if tip_normalized is not None and target_t_idx > tip_start_idx:
                    tip_normalized = advance(tip_normalized, i)
                    pred_tip = (
                        denormalise(
                            tip_normalized.unsqueeze(0), min_xyz_batch, scale_batch
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                organ_labels_pred = np.full(len(pred), organ_id, dtype=np.int32)

                leaf_tips_list = None
                leaf_tip_labels_list = None
                if pred_tip is not None:
                    leaf_tips_list = pred_tip.reshape(1, 3)
                    leaf_tip_labels_list = np.array([organ_id], dtype=np.int32)

                day = None
                if (
                    sequence_name
                    and self.timepoint_days
                    and target_t_idx < len(self.timepoint_days)
                ):
                    day = self.timepoint_days[target_t_idx]
                    organ_name = organ_type.replace("_", "")
                    filename = f"pred_{sequence_name}_D{day:02d}_{organ_name}.ply"
                else:
                    filename = f"pred_frame_{i+1:02d}.ply"

                save_ply_with_organ_labels(
                    pred,
                    organ_labels_pred,
                    leaf_tips_list,
                    leaf_tip_labels_list,
                    os.path.join(pred_dir, filename),
                    is_train=is_train_mask_pred,
                    normals=pred_normals,
                )

                if not save_ground_truth:
                    continue

                # Ground truth comes from the leaf dataset directly, which avoids
                # a round-trip through the alignment transform.
                gt_day = (
                    self.timepoint_days[target_t_idx]
                    if self.timepoint_days and target_t_idx < len(self.timepoint_days)
                    else None
                )
                leaf_data = day_to_leaf_data.get(gt_day) if gt_day is not None else None
                if leaf_data is None:
                    print(
                        f"WARNING: No leaf dataset entry for organ {organ_id} "
                        f"day {gt_day}, skipping GT PLY save"
                    )
                    continue

                gt = leaf_data["points"]
                gt_normals = leaf_data.get("normals")

                # is_train comes from the split file, which indexes the whole plant,
                # so it has to be filtered down to this organ's points.
                is_train_mask_gt = None
                if (
                    split_data is not None
                    and target_t_idx < len(split_data["timepoints"])
                    and gt_day in day_to_plant_tp
                ):
                    split_info = split_data["timepoints"][target_t_idx]
                    plant_tp = day_to_plant_tp[gt_day]
                    full_is_train = np.full(len(plant_tp["points"]), -1, dtype=np.int32)
                    train_indices = split_info.get("train_indices", [])
                    test_indices = split_info.get("test_indices", [])
                    if train_indices:
                        full_is_train[np.array(train_indices)] = 1
                    if test_indices:
                        full_is_train[np.array(test_indices)] = 0
                    organ_mask = (
                        np.array(plant_tp.get(self.label_field, plant_tp["labels"]))
                        == organ_id
                    )
                    is_train_mask_gt = full_is_train[organ_mask]

                gt_tips_list = None
                gt_tip_labels_list = None
                gt_tip_pos = leaf_data.get("leaf_tip")
                if gt_tip_pos is not None:
                    gt_tips_list = np.array(gt_tip_pos).reshape(1, 3)
                    gt_tip_labels_list = np.array([organ_id], dtype=np.int32)

                organ_labels_gt = np.full(len(gt), organ_id, dtype=np.int32)

                if day is not None:
                    organ_name = organ_type.replace("_", "")
                    gt_filename = f"gt_{sequence_name}_D{day:02d}_{organ_name}.ply"
                else:
                    gt_filename = f"gt_frame_{i+1:02d}.ply"

                save_ply_with_organ_labels(
                    gt,
                    organ_labels_gt,
                    gt_tips_list,
                    gt_tip_labels_list,
                    os.path.join(gt_dir, gt_filename),
                    is_train=is_train_mask_gt,
                    normals=gt_normals,
                )

    def _make_organ_stepper(
        self, organ_id, organ_info, valid_clouds, use_pre_finetune_models=False
    ):
        """Return `(advance, n_transitions)` for one organ's trained model.

        All four model types write predictions the same way; the only thing that
        differs is how a point cloud moves from one timepoint to the next.
        `advance(points, i)` carries `points` from frame `i` to frame `i + 1`,
        and is used for both the organ cloud and its leaf tip.

        Returns `(None, 0)` when the organ has no trained model, which is how a
        skipped organ presents.
        """
        config = organ_info["config"]
        model_type = config["model_type"]

        if model_type == "pairwise":
            if use_pre_finetune_models and "pre_finetune_models" in organ_info:
                models = organ_info["pre_finetune_models"]
            else:
                models = organ_info.get("trained_models", [])
            if not models:
                return None, 0

            # One model per transition, each predicting a displacement.
            def advance(points, i):
                return points + models[i](points)

            return advance, len(models)

        if model_type == "continuous":
            model = organ_info.get("trained_model")
        elif model_type == "shared_continuous":
            model = self.shared_model
        else:
            model = self.encoder_decoder_model
        if model is None:
            return None, 0

        num_frames = len(valid_clouds)

        # The global-model types put every organ on the plant's day range when
        # `use_global_tnorm` was set; a per-organ model only ever has its own.
        if model_type == "continuous" or self.global_day_range is None:
            first_day, last_day = self.organ_full_day_range[organ_id]
        else:
            first_day, last_day = self.global_day_range

        get_t_norm, get_dt = make_t_norm_helpers(
            self.organ_train_days[organ_id],
            first_day,
            last_day - first_day,
            num_frames,
            self.use_day_based_tnorm,
        )

        ode_kw = dict(
            use_odeint=config.get("use_odeint", False),
            ode_rtol=config.get("ode_rtol", 1e-4),
            ode_atol=config.get("ode_atol", 1e-5),
            ode_method=config.get("ode_method", "dopri5"),
            use_adjoint=False,
            n_substeps=config.get("train_integration_steps", 1),
            ode_n_steps=config.get("ode_n_steps", None),
        )
        if model_type == "shared_continuous":
            ode_kw["organ_id"] = organ_id
        elif model_type == "encoder_decoder_continuous":
            with torch.no_grad():
                ode_kw["latent_code"] = model.encode(valid_clouds[0])

        def advance(points, i):
            moved, _ = _integrate_segment(
                model, points, i, i + 1, get_t_norm, get_dt, **ode_kw
            )
            return moved

        return advance, num_frames - 1

    def save_full_plant_predictions(self, out_dir: str, save_ground_truth=True, use_pre_finetune_models=False):
        """
        Save full plant predictions by concatenating all organ model predictions.

        Args:
            out_dir: Output directory for full plant PLYs
            save_ground_truth: Whether to also save ground truth clouds
            use_pre_finetune_models: If True, use pre-finetune models for pairwise organs
        """
        # Use global day range for shared model if available
        full_day_range = dict(self.organ_full_day_range)
        if self.global_day_range is not None:
            for oid in full_day_range:
                full_day_range[oid] = self.global_day_range

        save_full_plant_from_organ_models(
            self.organ_models,
            self.organ_sequences,
            self.organ_normalizations,
            self.organ_leaf_tips,
            self.dataset,
            out_dir,
            sequence_name=(
                self.dataset.sequence_name
                if hasattr(self.dataset, "sequence_name")
                else None
            ),
            timepoint_days=self.timepoint_days,
            device=self.device,
            save_ground_truth=save_ground_truth,
            use_pre_finetune_models=use_pre_finetune_models,
            organ_transformations=self.organ_transformations,
            organ_train_days=self.organ_train_days,
            organ_full_day_range=full_day_range,
            use_day_based_tnorm=self.use_day_based_tnorm,
            use_odeint=self.default_config.get("use_odeint", False),
            ode_method=self.default_config.get("ode_method", "dopri5"),
            ode_rtol=self.default_config.get("ode_rtol", 1e-4),
            ode_atol=self.default_config.get("ode_atol", 1e-5),
            train_integration_steps=self.default_config.get("train_integration_steps", 1),
            ode_n_steps=self.default_config.get("ode_n_steps", None),
            input_noise_std=self.default_config.get("input_noise_std", 0.0),
            label_field=self.label_field,
            save_normals=self.save_normals,
        )

    def save_interpolation_predictions(
        self,
        out_dir: str,
        use_pre_finetune_models: bool = False,
        original_split_file: str = None,
        use_pose_anchoring: bool = True,
    ):
        """
        Save predictions and GT for interpolated (unseen) timepoints.

        For interpolation experiments with stride N, this saves predictions for
        timepoints that were NOT seen during training. Unseen days are determined
        automatically from gaps in the training timepoint_days.

        Args:
            out_dir: Output directory for interpolated PLYs
            use_pre_finetune_models: If True, use pre-finetune models for pairwise organs
        """
        from growfields.utils.io_utils import save_interpolation_from_organ_models

        # Use global day range for shared model if available
        full_day_range = dict(self.organ_full_day_range)
        if self.global_day_range is not None:
            for oid in full_day_range:
                full_day_range[oid] = self.global_day_range

        save_interpolation_from_organ_models(
            organ_models=self.organ_models,
            organ_sequences=self.organ_sequences,
            organ_normalizations=self.organ_normalizations,
            organ_leaf_tips=self.organ_leaf_tips,
            dataset=self.dataset,
            out_dir=out_dir,
            sequence_name=(
                self.dataset.sequence_name
                if hasattr(self.dataset, "sequence_name")
                else None
            ),
            timepoint_days=self.timepoint_days,
            device=self.device,
            use_pre_finetune_models=use_pre_finetune_models,
            organ_transformations_full=self.organ_transformations_full,
            organ_transformation_days=self.organ_transformation_days,
            organ_train_days=self.organ_train_days,
            organ_full_day_range=full_day_range,
            use_day_based_tnorm=self.use_day_based_tnorm,
            original_split_file=original_split_file,
            use_pose_anchoring=use_pose_anchoring,
            use_odeint=self.default_config.get("use_odeint", False),
            ode_method=self.default_config.get("ode_method", "dopri5"),
            ode_rtol=self.default_config.get("ode_rtol", 1e-4),
            ode_atol=self.default_config.get("ode_atol", 1e-5),
            train_integration_steps=self.default_config.get("train_integration_steps", 1),
            ode_n_steps=self.default_config.get("ode_n_steps", None),
            label_field=self.label_field,
            save_normals=self.save_normals,
        )

    def save_extrapolation_predictions(
        self,
        out_dir: str,
        use_pre_finetune_models: bool = False,
        original_split_file: str = None,
        use_pose_anchoring: bool = True,
    ):
        """
        Save predictions and GT for held-out final timepoints (extrapolation).

        Analogous to save_interpolation_predictions but targets days AFTER the
        last training day rather than days between training days.
        """
        from growfields.utils.io_utils import save_interpolation_from_organ_models

        full_day_range = dict(self.organ_full_day_range)
        if self.global_day_range is not None:
            for oid in full_day_range:
                full_day_range[oid] = self.global_day_range

        save_interpolation_from_organ_models(
            organ_models=self.organ_models,
            organ_sequences=self.organ_sequences,
            organ_normalizations=self.organ_normalizations,
            organ_leaf_tips=self.organ_leaf_tips,
            dataset=self.dataset,
            out_dir=out_dir,
            sequence_name=(
                self.dataset.sequence_name
                if hasattr(self.dataset, "sequence_name")
                else None
            ),
            timepoint_days=self.timepoint_days,
            device=self.device,
            use_pre_finetune_models=use_pre_finetune_models,
            organ_transformations_full=self.organ_transformations_full,
            organ_transformation_days=self.organ_transformation_days,
            organ_train_days=self.organ_train_days,
            organ_full_day_range=full_day_range,
            use_day_based_tnorm=self.use_day_based_tnorm,
            original_split_file=original_split_file,
            use_pose_anchoring=use_pose_anchoring,
            use_odeint=self.default_config.get("use_odeint", False),
            ode_method=self.default_config.get("ode_method", "dopri5"),
            ode_rtol=self.default_config.get("ode_rtol", 1e-4),
            ode_atol=self.default_config.get("ode_atol", 1e-5),
            train_integration_steps=self.default_config.get("train_integration_steps", 1),
            ode_n_steps=self.default_config.get("ode_n_steps", None),
            extrapolation=True,
            label_field=self.label_field,
            save_normals=self.save_normals,
        )

    def save_organ_metrics(self, organ_id: int, out_dir: str):
        """Write the organ's identity to `metrics.json`.

        Quality metrics are not written here. They are recomputed from the saved
        predictions by `scripts/compute_metrics_per_organ.py`, which is where the
        published numbers come from; computing them twice invites two answers.
        What stays is what makes the directory self-describing, and what that
        script reads back.
        """
        organ_info = self.organ_models.get(organ_id)
        if organ_info is None:
            return

        organ_seq = self.organ_sequences[organ_id]
        write_json(
            {
                "organ_id": organ_id,
                "organ_type": organ_seq["type"],
                "model_type": organ_info["config"]["model_type"],
                "n_timepoints_present": sum(organ_seq["presence_train"]),
            },
            os.path.join(out_dir, "metrics.json"),
        )


    def save_all_organs(
        self,
        base_out_dir: str,
        save_predictions=True,
        save_models=True,
        save_metrics=True,
        save_full_plant=True,
    ):
        """
        Save predictions, models, and metrics for all organs, plus full plant predictions.

        Args:
            base_out_dir: Base output directory
            save_predictions: Whether to save prediction PLY files per organ
            save_models: Whether to save model weights
            save_metrics: Whether to save metrics JSON files
            save_full_plant: Whether to save full plant predictions (concatenated from all organs)
        """
        print("\n" + "=" * 80)
        print("Saving per-organ results...")
        print("=" * 80)

        # A global model is one model for every organ, so it is saved once.
        global_model = self.default_config.get("model_type") in GLOBAL_MODEL_TRAINERS
        if save_models and global_model:
            print("\n  Saving model...")
            self.save_global_model(base_out_dir)

        for organ_id in sorted(self.organ_models.keys()):
            organ_seq = self.organ_sequences[organ_id]
            organ_type = organ_seq["type"]
            organ_info = self.organ_models[organ_id]
            config = organ_info["config"]

            organ_dir = os.path.join(base_out_dir, "organs", organ_type)
            os.makedirs(organ_dir, exist_ok=True)

            print(f"\n  Saving organ {organ_id} ({organ_type})...")

            # Save models
            if save_models and not global_model:
                print(f"    Saving models...")
                self.save_organ_models(organ_id, organ_dir)

            # Save predictions
            if save_predictions:
                # For pairwise models with finetuning, save both pre and post
                if config["model_type"] == "pairwise" and config.get(
                    "finetune_enabled", False
                ):
                    # Check for new train/test split keys or old combined keys
                    if "pre_finetune_test_mean_cd" in organ_info or "pre_finetune_mean_cd" in organ_info:
                        print(f"    Saving pre-finetune predictions...")
                        pre_dir = os.path.join(organ_dir, "pre_finetune")
                        self.save_organ_predictions(organ_id, pre_dir, use_pre_finetune_models=True)

                    if "post_finetune_test_mean_cd" in organ_info or "post_finetune_mean_cd" in organ_info:
                        print(f"    Saving post-finetune predictions...")
                        post_dir = os.path.join(organ_dir, "post_finetune")
                        self.save_organ_predictions(organ_id, post_dir, use_pre_finetune_models=False)
                else:
                    # For continuous models or pairwise without finetuning
                    print(f"    Saving predictions...")
                    self.save_organ_predictions(organ_id, organ_dir)

            # Save metrics
            if save_metrics:
                print(f"    Saving metrics...")
                self.save_organ_metrics(organ_id, organ_dir)

        # Save full plant predictions (concatenated from all organs)
        if save_full_plant and save_predictions:
            print("\n" + "=" * 80)
            print("Saving full plant predictions...")
            print("=" * 80)

            # Check if any organ has finetuning enabled
            has_finetuning = any(
                organ_info["config"]["model_type"] == "pairwise"
                and organ_info["config"].get("finetune_enabled", False)
                and "pre_finetune_models" in organ_info
                for organ_info in self.organ_models.values()
            )

            if has_finetuning:
                # Save both pre and post finetune full plant predictions
                print("  Detected pairwise models with finetuning - saving pre and post predictions")

                pre_full_plant_dir = os.path.join(base_out_dir, "full_plant", "pre_finetune")
                os.makedirs(pre_full_plant_dir, exist_ok=True)
                print("  Saving pre-finetune full plant predictions...")
                self.save_full_plant_predictions(pre_full_plant_dir, save_ground_truth=True, use_pre_finetune_models=True)

                post_full_plant_dir = os.path.join(base_out_dir, "full_plant", "post_finetune")
                os.makedirs(post_full_plant_dir, exist_ok=True)
                print("  Saving post-finetune full plant predictions...")
                self.save_full_plant_predictions(post_full_plant_dir, save_ground_truth=True, use_pre_finetune_models=False)
            else:
                # No finetuning - save single set of predictions
                full_plant_dir = os.path.join(base_out_dir, "full_plant")
                os.makedirs(full_plant_dir, exist_ok=True)
                self.save_full_plant_predictions(full_plant_dir, save_ground_truth=True)

        print("\n✅ All organs saved!")
        print("=" * 80)


