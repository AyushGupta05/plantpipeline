"""
SIREN-based deformation models for plant dynamics.

This module contains the SIREN (Sinusoidal Representation Networks) architecture
for learning deformation fields between point cloud timepoints.
"""

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from torch import nn

from growfields.models.integration import (
    _integrate_segment,
    denormalise,
    make_t_norm_helpers,
)
from growfields.models.siren import (
    ContinuousSiren,
    MonotoneSharedContinuousSiren,
    PairwiseSiren,
    SharedContinuousSiren,
)
from growfields.utils.metrics import chamfer_loss as chamfer_loss_fn


try:
    import wandb as _wandb
except ImportError:
    _wandb = None

def _wandb_log(data, step=None):
    """Log to wandb if available and a run is active."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.run.log(data, step=step)


# ---------------------------------------------------------------------------
# ODE integration helpers
# ---------------------------------------------------------------------------


def train_pairwise(
    xsrc,
    xtrg,
    n_steps=4000,
    lr=1e-4,
    n_samples=2048,
    h_f=128,
    h_layers=3,
    omega_0=30,
    device="cuda",
    verbose=True,
    use_wandb=False,
    wandb_prefix=None,
    log_interval=200,
):
    """
    Train a pairwise SIREN model to deform source to target point cloud.

    Args:
        xsrc: Source point cloud [N, 3]
        xtrg: Target point cloud [M, 3]
        n_steps: Number of training iterations
        lr: Learning rate
        n_samples: Number of points to sample per iteration
        h_f: Hidden features dimension
        h_layers: Number of hidden layers
        omega_0: SIREN frequency parameter
        device: torch device
        verbose: Print training progress

    Returns:
        Trained PairwiseSiren model in eval mode
    """
    model = PairwiseSiren(h_f=h_f, h_layers=h_layers, omega_0=omega_0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

    for i in range(n_steps):
        idx_s = torch.randint(0, xsrc.shape[0], (n_samples,), device=device)
        idx_t = torch.randint(0, xtrg.shape[0], (n_samples,), device=device)
        pred = xsrc[idx_s] + model(xsrc[idx_s])
        loss = chamfer_loss_fn(pred[None], xtrg[idx_t][None])

        opt.zero_grad()
        loss.backward()

        # --- Diagnostics (computed before opt.step) ---
        diag = {}
        if i % log_interval == 0:
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float('inf'))
            diag["model_grad_norm"] = model_grad_norm.item()

            with torch.no_grad():
                model_w_norm = sum(
                    p.norm().item() ** 2 for p in model.parameters()
                ) ** 0.5
            diag["model_weight_norm"] = model_w_norm

            with torch.no_grad():
                disp = model(xsrc[idx_s])
                disp_norms = disp.norm(dim=1)
                diag["disp_mean"] = disp_norms.mean().item()
                diag["disp_std"] = disp_norms.std().item()
                diag["disp_max"] = disp_norms.max().item()

        opt.step()

        if i % log_interval == 0:
            sched.step(loss.item())
            if verbose:
                print(
                    f"  Step {i:04d} | Chamfer: {loss.item():.6f} | "
                    f"grad={diag.get('model_grad_norm', 0):.4e} "
                    f"w={diag.get('model_weight_norm', 0):.2f} | "
                    f"disp={diag.get('disp_mean', 0):.4e} +/- {diag.get('disp_std', 0):.4e} "
                    f"(max {diag.get('disp_max', 0):.4e})"
                )
            if use_wandb:
                pfx = f"{wandb_prefix}_" if wandb_prefix else ""
                wandb_dict = {f"pairwise/{pfx}chamfer": loss.item()}
                for k, v in diag.items():
                    wandb_dict[f"diag/{pfx}{k}"] = v
                _wandb_log(wandb_dict)

    return model.eval()


def train_chained(
    clouds,
    n_steps=4000,
    lr=1e-4,
    n_samples=2048,
    h_f=128,
    h_layers=3,
    omega_0=30,
    device="cuda",
    verbose=True,
    use_wandb=False,
):
    """
    Train a sequence of pairwise SIREN models for temporal deformation.

    Args:
        clouds: List of point clouds, each with shape [B, N, 3]
        n_steps: Number of training iterations per pair
        lr: Learning rate
        n_samples: Number of points to sample per iteration
        h_f: Hidden features dimension
        h_layers: Number of hidden layers
        omega_0: SIREN frequency parameter
        device: torch device
        verbose: Print training progress

    Returns:
        List of trained PairwiseSiren models
    """
    models = []
    for i in range(1, len(clouds)):
        if verbose:
            print(f"\n🌱 Training f_{i-1}->{i}")
        m = train_pairwise(
            clouds[i - 1][0],
            clouds[i][0],
            n_steps,
            lr,
            n_samples,
            h_f,
            h_layers,
            omega_0,
            device,
            verbose,
            use_wandb=use_wandb,
        )
        models.append(m)

    if verbose:
        print(f"✅ Trained {len(models)} models.")
    return models


def train_continuous(
    clouds,
    min_xyz,
    scale,
    frame_days,
    full_day_range,
    use_day_based_tnorm=True,
    n_steps=50000,
    lr=1e-5,
    n_samples=2048,
    h_f=256,
    h_layers=5,
    first_omega_0=30.0,
    hidden_omega_0=30.0,
    device="cuda",
    verbose=True,
    log_interval=200,
    use_wandb=False,
    wandb_prefix=None,
    use_multistep_loss=False,
    # --- ODE integration (optional) ---
    use_odeint=False,
    ode_method="dopri5",
    ode_rtol=1e-4,
    ode_atol=1e-5,
    use_adjoint=True,
    direction="forward",
    train_integration_steps=1,
    ode_n_steps=None,
):
    """
    Train a single continuous time-aware SIREN model with optional biological priors.

    This model learns a deformation field across all timepoints.  Integration
    can use either Euler steps or an adaptive ODE solver (via *torchdiffeq*).
    Training uses a mixed sampling strategy: 50% local (consecutive frames) and
    50% global (arbitrary frame pairs).

    Biological priors (optional):
    - Spatial smoothness: Neighboring points have similar velocities
    - Directional persistence: Overall growth direction is maintained
    - Energy regularization: Dampens rapid acceleration
    - Apical bias: Encourages upward growth (+z direction)
    - Growth noise: Adds jitter to time for better generalization
    - Temporal gradient regularization: Penalizes large changes in velocity w.r.t. time
    - Divergence regularization: Penalizes negative divergence (contraction) for growing tissue

    Args:
        clouds: List of normalized point clouds [B, N, 3]
        min_xyz: Normalization min coordinates for evaluation
        scale: Normalization scale for evaluation
        frame_days: List of day values, one per cloud (length = num_frames)
        full_day_range: Tuple (first_day, last_day) for organ's full timeseries
        use_day_based_tnorm: If True, use day-based t_norm and velocity*dt.
            If False, use frame-index-based t_norm and dt=1 (legacy behavior).
        n_steps: Number of training iterations
        lr: Learning rate
        n_samples: Number of points to sample per iteration
        h_f: Hidden features dimension
        h_layers: Number of hidden layers
        first_omega_0: SIREN frequency for first layer
        hidden_omega_0: SIREN frequency for hidden layers
        device: torch device
        verbose: Print training progress
        log_interval: Steps between evaluation/logging
        use_odeint: Use torchdiffeq adaptive solver instead of Euler
        ode_method: ODE solver method ("dopri5", "rk4", "euler", ...)
        ode_rtol: Relative tolerance for adaptive ODE solver
        ode_atol: Absolute tolerance for adaptive ODE solver
        use_adjoint: Use adjoint method for memory-efficient backprop
        direction: "forward" or "backward" integration direction

    Returns:
        model: Trained ContinuousSiren model in eval mode
        loss_logs: Dictionary with loss history for each component
    """
    assert direction in ("forward", "backward"), \
        f"direction must be 'forward' or 'backward', got '{direction}'"
    backward = direction == "backward"

    if verbose:
        integrator = ode_method if use_odeint else "euler"
        print(f"\n🌱 Training Continuous Time-Aware SIREN ({direction}, {integrator})")

    num_frames = len(clouds)

    # t_norm helper: day-based or frame-index-based
    first_day, last_day = full_day_range
    day_range = last_day - first_day
    get_t_norm, get_dt = make_t_norm_helpers(
        frame_days, first_day, day_range, num_frames, use_day_based_tnorm)

    model = ContinuousSiren(
        h_f=h_f,
        h_layers=h_layers,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    loss_logs = {
        "total": [],
        "chamfer": [],
        "smooth": [],
        "dir": [],
        "energy": [],
        "apical": [],
        "temporal": [],
        "divergence": [],
    }

    for step in range(n_steps):
        model.train()

        # Sample strategy: 50% local (t -> t+1), 50% global (t_i -> t_j)
        is_local_step = np.random.rand() < 0.5

        # For sequences with only 2 frames, force local step (only one possible pair: 0->1)
        if is_local_step or num_frames <= 2:
            t_start = np.random.randint(0, num_frames - 1)
            t_end = t_start + 1
        else:
            t_start = np.random.randint(0, num_frames - 2)
            t_end = np.random.randint(t_start + 1, num_frames)

        if not backward:
            x_start_full, x_gt_full = clouds[t_start][0], clouds[t_end][0]
        else:
            x_start_full, x_gt_full = clouds[t_end][0], clouds[t_start][0]
        idx_s = torch.randint(0, x_start_full.shape[0], (n_samples,), device=device)
        idx_t = torch.randint(0, x_gt_full.shape[0], (n_samples,), device=device)
        x_start, x_gt = x_start_full[idx_s], x_gt_full[idx_t]

        # Integrate from t_start to t_end
        n_steps_traj = t_end - t_start
        if use_multistep_loss:
            # Compute Chamfer at every intermediate frame
            x_pred = x_start.clone()
            velocities = []
            chamfer_loss = torch.tensor(0.0, device=device)
            for t in range(t_start, t_end):
                x_pred, vels_t = _integrate_segment(
                    model, x_pred, t, t + 1, get_t_norm, get_dt,
                    use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
                    ode_method=ode_method, use_adjoint=use_adjoint,
                    backward=backward,
                    n_substeps=train_integration_steps, ode_n_steps=ode_n_steps,
                )
                velocities.extend(vels_t)

                gt_cloud_t = clouds[t + 1][0]
                idx_gt = torch.randint(0, gt_cloud_t.shape[0], (n_samples,), device=device)
                x_gt_t = gt_cloud_t[idx_gt]
                cd_t = chamfer_loss_fn(x_pred[None], x_gt_t[None])
                chamfer_loss = chamfer_loss + cd_t / n_steps_traj
        else:
            x_pred, velocities = _integrate_segment(
                model, x_start, t_start, t_end, get_t_norm, get_dt,
                use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
                ode_method=ode_method, use_adjoint=use_adjoint,
                backward=backward,
                n_substeps=train_integration_steps, ode_n_steps=ode_n_steps,
            )

        # Compute loss components
        if not use_multistep_loss:
            chamfer_loss = chamfer_loss_fn(x_pred[None], x_gt[None])
        loss = chamfer_loss

        opt.zero_grad()
        loss.backward()

        # --- Diagnostics (computed before opt.step) ---
        diag = {}
        if step % log_interval == 0:
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float('inf'))
            diag["model_grad_norm"] = model_grad_norm.item()

            with torch.no_grad():
                model_w_norm = sum(
                    p.norm().item() ** 2 for p in model.parameters()
                ) ** 0.5
            diag["model_weight_norm"] = model_w_norm

            with torch.no_grad():
                vel_norms = velocities[-1].norm(dim=1)
                diag["vel_mean"] = vel_norms.mean().item()
                diag["vel_std"] = vel_norms.std().item()
                diag["vel_max"] = vel_norms.max().item()

        opt.step()

        # Logging
        loss_logs["total"].append(loss.item())
        loss_logs["chamfer"].append(chamfer_loss.item())

        # Intermediate evaluation
        if step % log_interval == 0:
            sched.step(loss.item())

            if use_wandb:
                pfx = f"{wandb_prefix}_" if wandb_prefix else ""
                wandb_dict = {
                    f"train/{pfx}total": loss.item(),
                    f"train/{pfx}chamfer": chamfer_loss.item(),
                }
                for k, v in diag.items():
                    wandb_dict[f"diag/{pfx}{k}"] = v
                _wandb_log(wandb_dict)

            if verbose:
                model.eval()
                with torch.no_grad():
                    cds = []
                    eval_vel_norms = []
                    x = clouds[0][0]
                    for i in range(num_frames - 1):
                        x, vels = _integrate_segment(
                            model, x, i, i + 1, get_t_norm, get_dt,
                            use_odeint=use_odeint, ode_rtol=ode_rtol,
                            ode_atol=ode_atol, ode_method=ode_method,
                            use_adjoint=False, n_substeps=train_integration_steps,
                            ode_n_steps=ode_n_steps,
                        )
                        if vels:
                            eval_vel_norms.append(vels[-1].norm(dim=1).mean().item())

                        cd, _ = chamfer_distance(
                            denormalise(x, min_xyz, scale),
                            denormalise(clouds[i + 1][0], min_xyz, scale),
                        )
                        cds.append(cd.item())

                    mean_cd = np.mean(cds)

                if use_wandb:
                    pfx = f"{wandb_prefix}_" if wandb_prefix else "mean_"
                    eval_log = {f"eval/{pfx}cd": mean_cd}
                    if eval_vel_norms:
                        pfx2 = f"{wandb_prefix}_" if wandb_prefix else ""
                        eval_log[f"diag/{pfx2}eval_vel_mean"] = np.mean(eval_vel_norms)
                        eval_log[f"diag/{pfx2}eval_vel_std"] = np.std(eval_vel_norms)
                    _wandb_log(eval_log)

                print(
                    f"Step {step:05d} | Total={loss.item():.6f} | CD={chamfer_loss.item():.6f} "
                    f"| MeanEvalCD={mean_cd:.6f}"
                )
                print(
                    f"  DIAG: grad={diag.get('model_grad_norm', 0):.4e} | "
                    f"w={diag.get('model_weight_norm', 0):.2f} | "
                    f"vel={diag.get('vel_mean', 0):.4e} +/- {diag.get('vel_std', 0):.4e} "
                    f"(max {diag.get('vel_max', 0):.4e})"
                )

    if verbose:
        print("✅ Training complete.")

    return model.eval(), loss_logs


def train_shared_continuous(
    organ_data,
    num_organs=None,
    embed_dim=16,
    conditioning="concat",
    embed_init="normal",
    n_steps=50000,
    lr=1e-5,
    lr_embed=None,
    n_samples=2048,
    organs_per_step=1,
    h_f=256,
    h_layers=5,
    first_omega_0=30.0,
    hidden_omega_0=30.0,
    device="cuda",
    verbose=True,
    embed_reg_lambda=1e-4,
    log_interval=200,
    use_wandb=False,
    use_multistep_loss=False,
    # --- ODE integration (optional) ---
    use_odeint=False,
    ode_method="dopri5",
    ode_rtol=1e-4,
    ode_atol=1e-5,
    use_adjoint=True,
    direction="forward",
    train_integration_steps=1,
    ode_n_steps=None,
    # --- monotone velocity factorisation (plantpipeline addition; off by default) ---
    velocity_param="free",
    lambda_w=0.0,
    lambda_g=0.0,
    base_eps=1e-3,
    g_bias_init=-2.0,
):
    """
    Train a shared organ-conditioned continuous SIREN model.

    `velocity_param="monotone"` swaps the model for `MonotoneSharedContinuousSiren`
    (v = softplus(g) r_hat + P_perp w about a per-organ base point) and adds
    lambda_g * E[softplus(g)] + lambda_w * E||P_perp w||^2 to the loss, averaged over
    every velocity evaluation of the step. Everything else is unchanged.

    One SIREN conditioned on learnable organ embeddings learns velocity fields
    for all organs simultaneously, enabling cross-organ transfer.  Integration
    can use either Euler steps or an adaptive ODE solver.

    Args:
        organ_data: dict mapping organ_id -> {
            'clouds': list of [1, N, 3] tensors,
            'min_xyz': [1, 1, 3] tensor,
            'scale': [1, 1, 1] tensor,
            'frame_days': list of day ints,
            'full_day_range': (first_day, last_day),
        }
        num_organs: Number of organs (auto-detected if None)
        embed_dim: Organ embedding dimension
        n_steps: Training iterations
        lr: Learning rate
        n_samples: Points sampled per iteration
        h_f: SIREN hidden features
        h_layers: SIREN hidden layers
        first_omega_0: SIREN first layer frequency
        hidden_omega_0: SIREN hidden layer frequency
        device: torch device
        verbose: Print training progress
        embed_reg_lambda: Weight for embedding L2 regularization
        log_interval: Steps between logging

    Returns:
        model: Trained SharedContinuousSiren in eval mode
        loss_logs: Dict with loss history
    """
    valid_organ_ids = sorted(organ_data.keys())

    # Build remap if organ IDs are not contiguous 0-indexed
    dense_ids = list(range(len(valid_organ_ids)))
    needs_remap = (valid_organ_ids != dense_ids)
    if needs_remap:
        remap = {orig: dense for dense, orig in enumerate(valid_organ_ids)}
        if verbose:
            print(f"  Remapping organ IDs: {remap}")
    else:
        remap = None

    actual_num_organs = len(valid_organ_ids)
    if num_organs is None:
        num_organs = actual_num_organs

    if verbose:
        print(f"\n🌱 Training Shared Organ-Conditioned SIREN")
        print(f"   {actual_num_organs} organs, embed_dim={embed_dim}")

    # Build per-organ t_norm helpers
    organ_tnorm_helpers = {}
    for oid in valid_organ_ids:
        od = organ_data[oid]
        first_day, last_day = od['full_day_range']
        day_range = last_day - first_day
        frame_days = od['frame_days']
        num_frames = len(od['clouds'])

        get_t_norm, get_dt = make_t_norm_helpers(
            frame_days, first_day, day_range, num_frames, use_day_based_tnorm=True)
        organ_tnorm_helpers[oid] = {
            'get_t_norm': get_t_norm,
            'get_dt': get_dt,
            'num_frames': num_frames,
        }

    monotone = velocity_param == "monotone"
    if monotone:
        model = MonotoneSharedContinuousSiren(
            num_organs=num_organs,
            embed_dim=embed_dim,
            h_f=h_f,
            h_layers=h_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            conditioning=conditioning,
            embed_init=embed_init,
            base_eps=base_eps,
            g_bias_init=g_bias_init,
        ).to(device)
        # Per-organ base point in the organ's normalised coordinates. Leaves are in
        # the PSA canonical frame (base at the origin); the stem (organ 0) stays in
        # plant space, so its base is the soil point under the first training frame
        # (median x, y; min z). Assumes `dataset.aligned_leaves: true`.
        with torch.no_grad():
            for dense, oid in enumerate(valid_organ_ids):
                od = organ_data[oid]
                if oid == 0:
                    first = od['clouds'][0][0]
                    base = torch.stack([first[:, 0].median(), first[:, 1].median(), first[:, 2].min()])
                else:
                    base = (-od['min_xyz'][0, 0]) / od['scale'][0, 0, 0] * 2 - 1
                model.organ_base[dense] = base.to(device)
        model.collect_reg = (lambda_w > 0) or (lambda_g > 0)
        if verbose:
            print(f"   monotone velocity: lambda_w={lambda_w}, lambda_g={lambda_g}, "
                  f"base_eps={base_eps}, g_bias_init={g_bias_init}")
            for dense, oid in enumerate(valid_organ_ids):
                print(f"     organ {oid}: base (normalised coords) = "
                      f"{model.organ_base[dense].cpu().numpy().round(3)}")
    else:
        model = SharedContinuousSiren(
            num_organs=num_organs,
            embed_dim=embed_dim,
            h_f=h_f,
            h_layers=h_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            conditioning=conditioning,
            embed_init=embed_init,
        ).to(device)

    if remap is not None:
        model.organ_id_remap = remap

    # Separate learning rate for embeddings (and FiLM generator if used)
    embed_lr = lr_embed if lr_embed is not None else lr
    if conditioning == "none":
        # No conditioning: the organ embedding is unused, so keep it out of the
        # optimizer (its only gradient would come from embed_reg on dead weights).
        cond_params = []
    else:
        cond_params = list(model.organ_embedding.parameters())
        if conditioning == "film":
            cond_params += list(model.film_gen.parameters())
    cond_param_ids = {id(p) for p in cond_params}
    backbone_params = [p for p in model.parameters() if id(p) not in cond_param_ids]
    opt = torch.optim.Adam([
        {"params": backbone_params, "lr": lr},
        {"params": cond_params, "lr": embed_lr},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    loss_logs = {
        "total": [], "chamfer": [], "smooth": [], "dir": [], "energy": [],
        "apical": [], "temporal": [], "divergence": [], "embed_reg": [],
        "monotone_reg": [],
    }

    # Multi-organ sampling: sample K organs per step, split points equally
    n_total_organs = len(valid_organ_ids)
    k_organs = min(organs_per_step, n_total_organs)

    for step in range(n_steps):
        model.train()

        # Sample K organs (without replacement) for this step
        step_organ_ids = [int(x) for x in np.random.choice(
            valid_organ_ids, size=k_organs, replace=False,
        )]
        samples_per_organ = max(n_samples // k_organs, 256)

        # Accumulate losses across sampled organs
        chamfer_loss = torch.tensor(0.0, device=device)

        for oid in step_organ_ids:
            od = organ_data[oid]
            clouds = od['clouds']
            helpers = organ_tnorm_helpers[oid]
            num_frames = helpers['num_frames']
            get_t_norm = helpers['get_t_norm']
            get_dt = helpers['get_dt']

            # Sample time pair: 50% local, 50% global
            is_local = np.random.rand() < 0.5
            if is_local or num_frames <= 2:
                t_start = np.random.randint(0, num_frames - 1)
                t_end = t_start + 1
            else:
                t_start = np.random.randint(0, num_frames - 2)
                t_end = np.random.randint(t_start + 1, num_frames)

            backward = direction == "backward"
            if not backward:
                x_start_full, x_gt_full = clouds[t_start][0], clouds[t_end][0]
            else:
                x_start_full, x_gt_full = clouds[t_end][0], clouds[t_start][0]
            idx_s = torch.randint(0, x_start_full.shape[0], (samples_per_organ,), device=device)
            idx_t = torch.randint(0, x_gt_full.shape[0], (samples_per_organ,), device=device)
            x_start, x_gt = x_start_full[idx_s], x_gt_full[idx_t]

            # Integrate
            n_steps_traj = t_end - t_start
            if use_multistep_loss:
                # Compute Chamfer at every intermediate frame
                x_pred = x_start.clone()
                velocities = []
                for t in range(t_start, t_end):
                    x_pred, vels_t = _integrate_segment(
                        model, x_pred, t, t + 1, get_t_norm, get_dt,
                        use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
                        ode_method=ode_method, use_adjoint=use_adjoint,
                        backward=backward,
                        organ_id=oid, n_substeps=train_integration_steps,
                        ode_n_steps=ode_n_steps,
                    )
                    velocities.extend(vels_t)

                    gt_cloud_t = clouds[t + 1][0]
                    idx_gt = torch.randint(0, gt_cloud_t.shape[0], (samples_per_organ,), device=device)
                    x_gt_t = gt_cloud_t[idx_gt]
                    cd_t = chamfer_loss_fn(x_pred[None], x_gt_t[None])
                    chamfer_loss = chamfer_loss + cd_t / n_steps_traj
            else:
                x_pred, velocities = _integrate_segment(
                    model, x_start, t_start, t_end, get_t_norm, get_dt,
                    use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
                    ode_method=ode_method, use_adjoint=use_adjoint,
                    backward=backward,
                    organ_id=oid, n_substeps=train_integration_steps,
                    ode_n_steps=ode_n_steps,
                )

            if not use_multistep_loss:
                cd = chamfer_loss_fn(x_pred[None], x_gt[None])
                chamfer_loss = chamfer_loss + cd

        # Average across sampled organs
        chamfer_loss = chamfer_loss / k_organs

        # Embedding L2 regularization (no-op when conditioning is disabled)
        if conditioning == "none":
            embed_loss = torch.zeros((), device=device)
        else:
            embed_loss = model.organ_embedding.weight.pow(2).mean() * embed_reg_lambda

        # Monotone-model regularisers (zero unless velocity_param=monotone and a lambda > 0)
        mono_loss = torch.zeros((), device=device)
        if monotone and model.collect_reg:
            terms = model.pop_reg_terms()
            if terms:
                g_mean = torch.stack([g for g, _ in terms]).mean()
                w_mean = torch.stack([w for _, w in terms]).mean()
                mono_loss = lambda_g * g_mean + lambda_w * w_mean

        loss = chamfer_loss + embed_loss + mono_loss

        opt.zero_grad()
        loss.backward()

        # --- Diagnostics (computed before opt.step) ---
        diag = {}
        if step % log_interval == 0:
            backbone_grad_norm = torch.nn.utils.clip_grad_norm_(
                backbone_params, float('inf'))
            diag["backbone_grad_norm"] = backbone_grad_norm.item()
            if cond_params:
                embed_grad_norm = torch.nn.utils.clip_grad_norm_(
                    cond_params, float('inf'))
                diag["embed_grad_norm"] = embed_grad_norm.item()

            with torch.no_grad():
                backbone_w_norm = sum(
                    p.norm().item() ** 2 for p in backbone_params
                ) ** 0.5
                embed_w_norm = sum(
                    p.norm().item() ** 2 for p in cond_params
                ) ** 0.5
            diag["backbone_weight_norm"] = backbone_w_norm
            if cond_params:
                diag["embed_weight_norm"] = embed_w_norm

            with torch.no_grad():
                vel_norms = velocities[-1].norm(dim=1)
                diag["vel_mean"] = vel_norms.mean().item()
                diag["vel_std"] = vel_norms.std().item()
                diag["vel_max"] = vel_norms.max().item()

            if conditioning != "none":
                with torch.no_grad():
                    emb_all = model.get_all_embeddings()
                    emb_norms = emb_all.norm(dim=1)
                    diag["embed_mean_norm"] = emb_norms.mean().item()
                    diag["embed_std_norm"] = emb_norms.std().item()
                    if emb_all.shape[0] > 1:
                        import torch.nn.functional as F
                        emb_normed = F.normalize(emb_all, dim=1)
                        cos_mat = emb_normed @ emb_normed.T
                        mask_diag = ~torch.eye(emb_all.shape[0], dtype=torch.bool, device=device)
                        diag["embed_mean_cos_sim"] = cos_mat[mask_diag].mean().item()
                    else:
                        diag["embed_mean_cos_sim"] = 0.0

        opt.step()

        # Logging
        loss_logs["total"].append(loss.item())
        loss_logs["chamfer"].append(chamfer_loss.item())
        loss_logs["embed_reg"].append(embed_loss.item())
        loss_logs["monotone_reg"].append(mono_loss.item())

        if step % log_interval == 0:
            sched.step(loss.item())

            if use_wandb:
                wandb_dict = {
                    "train/total": loss.item(),
                    "train/chamfer": chamfer_loss.item(),
                    "train/embed_reg": embed_loss.item(),
                }
                for k, v in diag.items():
                    wandb_dict[f"diag/{k}"] = v
                _wandb_log(wandb_dict, step=step)

            if verbose:
                model.eval()
                with torch.no_grad():
                    # Quick eval: mean CD across all organs
                    eval_cds = []
                    per_organ_eval_cds = {}
                    per_organ_vel_stats = {}
                    for eval_oid in valid_organ_ids:
                        eval_od = organ_data[eval_oid]
                        eval_clouds = eval_od['clouds']
                        eval_helpers = organ_tnorm_helpers[eval_oid]

                        organ_cds = []
                        organ_vel_norms = []
                        x_eval = eval_clouds[0][0]
                        for ei in range(eval_helpers['num_frames'] - 1):
                            x_eval, vels = _integrate_segment(
                                model, x_eval, ei, ei + 1,
                                eval_helpers['get_t_norm'], eval_helpers['get_dt'],
                                use_odeint=use_odeint, ode_rtol=ode_rtol,
                                ode_atol=ode_atol, ode_method=ode_method,
                                use_adjoint=False,
                                organ_id=eval_oid,
                                n_substeps=train_integration_steps,
                                ode_n_steps=ode_n_steps,
                            )
                            if vels:
                                organ_vel_norms.append(vels[-1].norm(dim=1).mean().item())
                            min_xyz = eval_od['min_xyz']
                            scale_val = eval_od['scale']
                            cd, _ = chamfer_distance(
                                denormalise(x_eval, min_xyz, scale_val),
                                denormalise(eval_clouds[ei + 1][0], min_xyz, scale_val),
                            )
                            eval_cds.append(cd.item())
                            organ_cds.append(cd.item())
                        if organ_cds:
                            per_organ_eval_cds[eval_oid] = np.mean(organ_cds)
                        if organ_vel_norms:
                            per_organ_vel_stats[eval_oid] = {
                                "mean": np.mean(organ_vel_norms),
                                "std": np.std(organ_vel_norms),
                                "max": np.max(organ_vel_norms),
                            }
                    mean_eval_cd = np.mean(eval_cds) if eval_cds else 0.0

                if use_wandb:
                    eval_log = {"eval/mean_cd": mean_eval_cd}
                    for eval_oid, organ_cd in per_organ_eval_cds.items():
                        eval_log[f"eval/organ_{eval_oid}_cd"] = organ_cd
                    for eval_oid, vs in per_organ_vel_stats.items():
                        eval_log[f"diag/organ_{eval_oid}_vel_mean"] = vs["mean"]
                        eval_log[f"diag/organ_{eval_oid}_vel_std"] = vs["std"]
                    _wandb_log(eval_log, step=step)

                print(
                    f"Step {step:05d} | Total={loss.item():.6f} | CD={chamfer_loss.item():.6f} "
                    f"| EmbReg={embed_loss.item():.4e} "
                    f"| MeanEvalCD={mean_eval_cd:.6f}"
                )
                print(
                    f"  DIAG: bb_grad={diag.get('backbone_grad_norm', 0):.4e} "
                    f"emb_grad={diag.get('embed_grad_norm', 0):.4e} | "
                    f"bb_w={diag.get('backbone_weight_norm', 0):.2f} "
                    f"emb_w={diag.get('embed_weight_norm', 0):.2f} | "
                    f"vel={diag.get('vel_mean', 0):.4e} +/- {diag.get('vel_std', 0):.4e} "
                    f"(max {diag.get('vel_max', 0):.4e}) | "
                    f"emb_norm={diag.get('embed_mean_norm', 0):.3f} +/- {diag.get('embed_std_norm', 0):.3f} "
                    f"cos_sim={diag.get('embed_mean_cos_sim', 0):.3f}"
                )
                for eval_oid in valid_organ_ids:
                    vs = per_organ_vel_stats.get(eval_oid, {})
                    ocd = per_organ_eval_cds.get(eval_oid, 0.0)
                    print(
                        f"    organ {eval_oid}: CD={ocd:.4f} | "
                        f"vel={vs.get('mean', 0):.4e} +/- {vs.get('std', 0):.4e} "
                        f"(max {vs.get('max', 0):.4e})"
                    )

    if verbose:
        print("✅ Shared model training complete.")

    return model.eval(), loss_logs


def _chained_chamfer_distances(models, clouds, min_xyz, scale):
    """Chamfer distance at each frame of a pairwise chain, rolled out from frame 0.

    Only the progress log in `hybrid_finetune` uses this; the paper's metrics come
    from `scripts/compute_metrics_{fullplant,per_organ}.py`, which score the saved
    predictions after training.

    Args:
        models: Trained PairwiseSiren models, one per transition
        clouds: Normalized point clouds [B, N, 3], one per frame
        min_xyz: Normalization min coordinates
        scale: Normalization scale

    Returns:
        List of Chamfer distances in original units, one per transition
    """
    cds = []
    x = clouds[0][0]
    with torch.no_grad():
        for i, m in enumerate(models):
            x = x + m(x)
            cd, _ = chamfer_distance(
                denormalise(x, min_xyz, scale), denormalise(clouds[i + 1][0], min_xyz, scale)
            )
            cds.append(cd.item())
    return cds


def hybrid_finetune(
    models,
    clouds,
    min_xyz,
    scale,
    n_steps=1500,
    lr=2e-5,
    lr_x0=5e-6,
    w_global=3.0,
    w_local=1.0,
    n_samples=2048,
    freeze_early=0,
    device="cuda",
    verbose=True,
    # --- Optimise initial cloud ---
    optimise_x0=False,
    log_interval=100,
    use_wandb=False,
    wandb_prefix=None,
):

    """
    Fine-tune all models jointly with global and local consistency losses and optional biological priors.

    This function supports biological priors similar to the continuous model:
    - Spatial smoothness: Neighboring points have similar velocities
    - Directional persistence: Overall growth direction is maintained
    - Energy regularization: Dampens rapid acceleration
    - Apical bias: Encourages upward growth (+z direction)
    - Growth noise: Adds jitter to predictions for better generalization

    Note: Temporal gradient regularization is not available for pairwise models
    as it requires continuous time input (only available in train_continuous).

    Args:
        models: List of trained PairwiseSiren models
        clouds: List of normalized point clouds [B, N, 3]
        min_xyz: Normalization min coordinates
        scale: Normalization scale
        n_steps: Number of fine-tuning iterations
        lr: Learning rate for models
        lr_x0: Learning rate for initial cloud (if optimise_x0=True)
        w_global: Weight for global consistency loss
        w_local: Weight for local pairwise loss
        n_samples: Number of points to sample per iteration
        freeze_early: Number of early models to freeze
        device: torch device
        verbose: Print progress
        optimise_x0: If True, optimize the initial cloud x0 alongside models
    """

    # Parameter groups
    model_params = [p for m in models for p in m.parameters()]
    if optimise_x0:
        x0_base = clouds[0][0].detach().clone()
        x0_delta = nn.Parameter(torch.zeros_like(x0_base))
        param_groups = [
            {'params': model_params, 'lr': lr},
            {'params': [x0_delta], 'lr': lr_x0}
        ]
    else:
        x0_base = clouds[0][0]
        x0_delta = None
        param_groups = [{'params': model_params, 'lr': lr}]

    opt = torch.optim.Adam(param_groups)

    for step in range(n_steps):
        # Pick a random target time
        t_idx = np.random.randint(1, len(clouds))
        x_gt = clouds[t_idx][0]
        x0_current = x0_base + x0_delta if optimise_x0 else x0_base

        idx0 = torch.randint(0, x0_current.shape[0], (n_samples,), device=device)
        idxt = torch.randint(0, x_gt.shape[0], (n_samples,), device=device)

        # Forward chain
        x_pred = x0_current[idx0].clone()
        velocities = []
        for j, m in enumerate(models[:t_idx]):
            if j < freeze_early:
                with torch.no_grad():
                    v = m(x_pred)
                    x_pred = x_pred + v
            else:
                v = m(x_pred)
                velocities.append(v)
                x_pred = x_pred + v

        # --- Global reconstruction loss drives both model + x0 ---
        loss_global = chamfer_loss_fn(x_pred[None], x_gt[idxt][None])

        # --- Local consistency ---
        loss_local = torch.tensor(0.0, device=device)
        for k in range(1, len(models)):
            xs, xt = clouds[k - 1][0], clouds[k][0]
            is_ = torch.randint(0, xs.shape[0], (n_samples,), device=device)
            it_ = torch.randint(0, xt.shape[0], (n_samples,), device=device)
            loss_local += chamfer_loss_fn(
                (xs[is_] + models[k - 1](xs[is_]))[None], xt[it_][None]
            )
        loss_local /= max(1, len(models))


        # --- Total loss ---
        loss = (
            w_global * loss_global
            + w_local * loss_local
        )

        opt.zero_grad()
        loss.backward()

        # --- Diagnostics (computed before opt.step) ---
        diag = {}
        if step % log_interval == 0:
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                model_params, float('inf'))
            diag["model_grad_norm"] = model_grad_norm.item()

            with torch.no_grad():
                model_w_norm = sum(
                    p.norm().item() ** 2 for p in model_params
                ) ** 0.5
            diag["model_weight_norm"] = model_w_norm

            if velocities:
                with torch.no_grad():
                    vel_norms = velocities[-1].norm(dim=1)
                    diag["vel_mean"] = vel_norms.mean().item()
                    diag["vel_std"] = vel_norms.std().item()
                    diag["vel_max"] = vel_norms.max().item()

            if optimise_x0:
                x0_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [x0_delta], float('inf'))
                diag["x0_grad_norm"] = x0_grad_norm.item()

        opt.step()

        if step % log_interval == 0 and verbose:
            cd_mean = np.mean(_chained_chamfer_distances(models, clouds, min_xyz, scale))
            msg = (
                f"[{step:04d}] Mean CD={cd_mean:.4f} | "
                f"Glob={loss_global.item():.4f} Loc={loss_local.item():.4f} | "
            )
            if optimise_x0:
                msg += f" | x0d2={x0_delta.pow(2).mean().item():.3e}"
            msg += (
                f" | grad={diag.get('model_grad_norm', 0):.4e} "
                f"w={diag.get('model_weight_norm', 0):.2f}"
            )
            if 'vel_mean' in diag:
                msg += (
                    f" | vel={diag['vel_mean']:.4e} +/- {diag['vel_std']:.4e} "
                    f"(max {diag['vel_max']:.4e})"
                )
            print(msg)

            if use_wandb:
                pfx = f"{wandb_prefix}_" if wandb_prefix else ""
                wandb_dict = {
                    f"finetune/{pfx}mean_cd": cd_mean,
                    f"finetune/{pfx}global_loss": loss_global.item(),
                    f"finetune/{pfx}local_loss": loss_local.item(),
                }
                for k, v in diag.items():
                    wandb_dict[f"diag/{pfx}{k}"] = v
                _wandb_log(wandb_dict)

    if verbose:
        print("✅ Fine-tuning complete.")

    if optimise_x0:
        with torch.no_grad():
            clouds[0][0].data.copy_((x0_base + x0_delta).data)
