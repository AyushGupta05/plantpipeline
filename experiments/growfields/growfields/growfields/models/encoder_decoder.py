"""
Encoder-Decoder model for organ growth prediction.

PointNet encoder (ResnetPointnet) produces latent codes from organ point cloud
geometry, paired with a ReLU MLP decoder (ImplicitNet) that predicts velocity
fields. Inspired by COAP (Compositional Articulated Occupancy of People).

Architecture:
  Encoder: point cloud [B, N, 3] -> latent code [B, latent_dim]
  Decoder: (x, y, z, t_norm, latent) -> velocity (vx, vy, vz)
"""

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance

from growfields.utils.metrics import chamfer_loss as chamfer_loss_fn
from torch import nn

from growfields.models.integration import (
    _integrate_segment,
    denormalise,
)
from growfields.models.siren_dynamics import _wandb_log


# ---------------------------------------------------------------------------
# ResNet FC block. Adapted from COAP (https://github.com/markomih/COAP),
# MIT licensed, Copyright (c) 2022 Marko Mihajlovic; see NOTICE.
# ---------------------------------------------------------------------------
class ResnetBlockFC(nn.Module):
    """Fully-connected ResNet block with optional shortcut projection."""

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


# ---------------------------------------------------------------------------
# ResnetPointnet encoder. Adapted from COAP; see NOTICE.
# ---------------------------------------------------------------------------
class ResnetPointnet(nn.Module):
    """PointNet encoder with ResNet blocks.

    Input:  [B, T, 3] point cloud
    Output: [B, out_dim] latent code
    """

    def __init__(self, out_dim=128, hidden_dim=128):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.fc_pos = nn.Linear(3, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, out_dim)
        self.actvn = nn.ReLU()

    def pool(self, x, dim=-2, keepdim=False):
        return x.max(dim=dim, keepdim=keepdim)[0]

    def forward(self, p):
        """
        Args:
            p: [B, T, 3] point cloud

        Returns:
            c: [B, out_dim] latent code
        """
        net = self.fc_pos(p)          # [B, T, 2*hidden_dim]
        net = self.block_0(net)       # [B, T, hidden_dim]

        pooled = self.pool(net, dim=-2, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=-1)  # [B, T, 2*hidden_dim]
        net = self.block_1(net)

        pooled = self.pool(net, dim=-2, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=-1)
        net = self.block_2(net)

        pooled = self.pool(net, dim=-2, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=-1)
        net = self.block_3(net)

        pooled = self.pool(net, dim=-2, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=-1)
        net = self.block_4(net)

        c = self.pool(net, dim=-2)    # [B, hidden_dim]
        c = self.fc_c(self.actvn(c))  # [B, out_dim]
        return c


# ---------------------------------------------------------------------------
# ImplicitNet decoder. Adapted from COAP's version, which itself credits IGR
# (Gropp et al.); see NOTICE.
# ---------------------------------------------------------------------------
class ImplicitNet(nn.Module):
    """ReLU MLP with skip connections for velocity prediction.

    Adapted from IGR/COAP ImplicitNet. Uses ReLU activation (beta=0)
    and no geometric initialization (we predict velocity, not SDF).
    """

    def __init__(
        self,
        d_in,
        d_out=3,
        dims=None,
        skip_in=None,
        geometric_init=False,
        beta=0,
        zero_init_last=False,
    ):
        super().__init__()

        if dims is None:
            dims = [256, 256, 256, 256, 256]
        if skip_in is None:
            skip_in = [3]

        self.d_in = d_in
        self.d_out = d_out
        self.skip_in = skip_in

        dims = [d_in] + dims + [d_out]
        self.num_layers = len(dims)

        self.layers = nn.ModuleList()
        for l in range(self.num_layers - 1):
            din = dims[l]
            if l in skip_in:
                din += d_in
            dout = dims[l + 1]
            self.layers.append(nn.Linear(din, dout))

        # Activation
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        # Initialization
        if geometric_init:
            for l in range(self.num_layers - 1):
                lin = self.layers[l]
                if l == self.num_layers - 2:
                    # Last layer: geometric init for SDF
                    nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.00001)
                    nn.init.constant_(lin.bias, -1.0)
                elif l == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dout))
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                elif l in skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dout))
                    nn.init.constant_(lin.weight[:, -(d_in):], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dout))

        # Zero-init last layer so model starts by predicting zero velocity
        if zero_init_last:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        """
        Args:
            x: [N, d_in] input features

        Returns:
            [N, d_out] output
        """
        x_in = x
        for l in range(self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, x_in], dim=-1)
            x = self.layers[l](x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x


# ---------------------------------------------------------------------------
# EncoderDecoderModel wrapper
# ---------------------------------------------------------------------------
class EncoderDecoderModel(nn.Module):
    """PointNet encoder + MLP decoder for organ velocity prediction.

    Encoder produces a shape-aware latent code from the organ's point cloud.
    Decoder takes (x, y, z, t_norm, latent) and predicts velocity (vx, vy, vz).
    """

    def __init__(
        self,
        latent_dim=128,
        encoder_hidden_dim=128,
        decoder_dims=None,
        decoder_skip_in=None,
        zero_init_last=False,
    ):
        super().__init__()
        if decoder_dims is None:
            decoder_dims = [256, 256, 256, 256, 256]
        if decoder_skip_in is None:
            decoder_skip_in = [3]

        self.latent_dim = latent_dim
        self.encoder = ResnetPointnet(out_dim=latent_dim, hidden_dim=encoder_hidden_dim)
        self.decoder = ImplicitNet(
            d_in=4 + latent_dim,  # (x, y, z, t_norm) + latent
            d_out=3,              # (vx, vy, vz)
            dims=decoder_dims,
            skip_in=decoder_skip_in,
            geometric_init=False,
            beta=0,               # ReLU
            zero_init_last=zero_init_last,
        )

    def encode(self, point_cloud):
        """Encode a point cloud into a latent code.

        Args:
            point_cloud: [N, 3] or [B, N, 3] point cloud

        Returns:
            latent: [1, latent_dim] or [B, latent_dim]
        """
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)  # [1, N, 3]
        return self.encoder(point_cloud)

    def forward(self, x, t_norm, latent_code):
        """Predict velocity field.

        Args:
            x: [N, 3] query points
            t_norm: scalar or tensor, normalized time
            latent_code: [latent_dim] or [1, latent_dim] or [N, latent_dim]

        Returns:
            velocity: [N, 3]
        """
        N = x.shape[0]

        # Handle t_norm broadcasting (same pattern as SharedContinuousSiren)
        if isinstance(t_norm, (int, float)):
            t_norm = torch.tensor([t_norm], dtype=x.dtype, device=x.device)
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0)
        t_norm = t_norm.expand(N, 1) if t_norm.shape[0] == 1 else t_norm.unsqueeze(-1)

        # Handle latent_code broadcasting
        if latent_code.dim() == 1:
            latent_code = latent_code.unsqueeze(0)  # [1, latent_dim]
        if latent_code.shape[0] == 1:
            latent_code = latent_code.expand(N, -1)  # [N, latent_dim]

        # Concatenate: [x, y, z, t_norm, latent] -> [N, 4 + latent_dim]
        inp = torch.cat([x, t_norm, latent_code], dim=-1)
        return self.decoder(inp)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def train_encoder_decoder_continuous(
    organ_data,
    latent_dim=128,
    encoder_hidden_dim=128,
    decoder_dims=None,
    decoder_skip_in=None,
    zero_init_last=False,
    n_steps=50000,
    lr=1e-5,
    lr_encoder=None,
    decoder_weight_decay=0.0,
    n_samples=2048,
    n_enc_samples=1024,
    organs_per_step=1,
    use_multistep_loss=False,
    device="cuda",
    verbose=True,
    # --- Regularization (optional) ---
    use_latent_reg=False,
    latent_reg_lambda=1e-4,
    log_interval=200,
    use_wandb=False,
    train_integration_steps=1,
    use_odeint=False,
    ode_rtol=1e-4,
    ode_atol=1e-5,
    ode_method="dopri5",
    ode_n_steps=None,
    use_adjoint=False,
):
    """
    Train an encoder-decoder model for organ velocity prediction.

    PointNet encoder produces latent codes from organ point clouds, MLP decoder
    predicts velocity fields conditioned on those latent codes.

    Args:
        organ_data: dict mapping organ_id -> {
            'clouds': list of [1, N, 3] tensors,
            'min_xyz': [1, 1, 3] tensor,
            'scale': [1, 1, 1] tensor,
            'frame_days': list of day ints,
            'full_day_range': (first_day, last_day),
        }
        latent_dim: Encoder output / latent code dimension
        encoder_hidden_dim: ResnetPointnet hidden dimension
        decoder_dims: ImplicitNet hidden layer dimensions
        decoder_skip_in: ImplicitNet skip connection layers
        n_steps: Training iterations
        lr: Decoder learning rate
        lr_encoder: Encoder learning rate (defaults to lr if None)
        n_samples: Points sampled per iteration for Euler integration
        n_enc_samples: Points sampled for encoding
        organs_per_step: Number of organs sampled per training step
        use_multistep_loss: If True, compute chamfer at every intermediate Euler step
            (not just the endpoint). Improves trajectory generalization. Default True.
        device: torch device
        verbose: Print training progress
        latent_reg_lambda: L2 regularization on encoder output (analogous to embed_reg_lambda)
        log_interval: Steps between logging
        use_wandb: Enable wandb logging

    Returns:
        model: Trained EncoderDecoderModel in eval mode
        loss_logs: Dict with loss history
    """
    if decoder_dims is None:
        decoder_dims = [256, 256, 256, 256, 256]
    if decoder_skip_in is None:
        decoder_skip_in = [3]

    valid_organ_ids = sorted(organ_data.keys())
    actual_num_organs = len(valid_organ_ids)

    if verbose:
        print(f"\n  Training Encoder-Decoder Model")
        print(f"   {actual_num_organs} organs, latent_dim={latent_dim}")

    # Build per-organ t_norm helpers
    organ_tnorm_helpers = {}
    for oid in valid_organ_ids:
        od = organ_data[oid]
        first_day, last_day = od['full_day_range']
        day_range = last_day - first_day
        frame_days = od['frame_days']

        def _make_helpers(fd, fday, dr):
            def get_t_norm(frame_idx):
                return (fd[frame_idx] - fday) / dr
            def get_dt(frame_idx):
                return get_t_norm(frame_idx + 1) - get_t_norm(frame_idx)
            return get_t_norm, get_dt
        get_t_norm, get_dt = _make_helpers(frame_days, first_day, day_range)
        organ_tnorm_helpers[oid] = {
            'get_t_norm': get_t_norm,
            'get_dt': get_dt,
            'num_frames': len(od['clouds']),
        }

    model = EncoderDecoderModel(
        latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_dims=decoder_dims,
        decoder_skip_in=decoder_skip_in,
        zero_init_last=zero_init_last,
    ).to(device)

    # Separate learning rates for encoder and decoder
    enc_lr = lr_encoder if lr_encoder is not None else lr
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    opt = torch.optim.Adam([
        {"params": decoder_params, "lr": lr, "weight_decay": decoder_weight_decay},
        {"params": encoder_params, "lr": enc_lr},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    loss_logs = {
        "total": [], "chamfer": [], "smooth": [], "dir": [],
        "divergence": [], "latent_reg": [],
    }

    # Multi-organ sampling
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
        latent_reg_loss = torch.tensor(0.0, device=device)

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

            x_start_full = clouds[t_start][0]
            idx_s = torch.randint(0, x_start_full.shape[0], (samples_per_organ,), device=device)
            x_start = x_start_full[idx_s]

            # Encode the starting cloud (sample points for encoding)
            enc_idx = torch.randint(0, x_start_full.shape[0], (n_enc_samples,), device=device)
            latent_code = model.encode(x_start_full[enc_idx])  # [1, latent_dim]

            # Latent L2 regularization (analogous to embed_reg_lambda in shared model)
            if use_latent_reg:
                latent_reg_loss = latent_reg_loss + latent_code.pow(2).mean() * latent_reg_lambda

            # Euler integrate (with optional sub-stepping)
            x_pred = x_start.clone()
            velocities = []
            n_steps_traj = t_end - t_start
            _ode_kw = dict(
                use_odeint=use_odeint, ode_rtol=ode_rtol, ode_atol=ode_atol,
                ode_method=ode_method, use_adjoint=use_adjoint, n_substeps=train_integration_steps,
                latent_code=latent_code,
                ode_n_steps=ode_n_steps,
            )
            for t in range(t_start, t_end):
                x_pred, step_vels = _integrate_segment(
                    model, x_pred, t, t + 1, get_t_norm, get_dt, **_ode_kw)
                velocities.append(step_vels[-1])

                if use_multistep_loss:
                    # Chamfer at every intermediate step (not just endpoint)
                    gt_cloud_t = clouds[t + 1][0]
                    idx_gt = torch.randint(0, gt_cloud_t.shape[0], (samples_per_organ,), device=device)
                    x_gt_t = gt_cloud_t[idx_gt]
                    cd_t = chamfer_loss_fn(x_pred[None], x_gt_t[None])
                    chamfer_loss = chamfer_loss + cd_t / n_steps_traj

            if not use_multistep_loss:
                # Endpoint-only chamfer (original behavior)
                x_gt_full = clouds[t_end][0]
                idx_t = torch.randint(0, x_gt_full.shape[0], (samples_per_organ,), device=device)
                x_gt = x_gt_full[idx_t]
                cd = chamfer_loss_fn(x_pred[None], x_gt[None])
                chamfer_loss = chamfer_loss + cd

        # Average across sampled organs
        chamfer_loss = chamfer_loss / k_organs
        latent_reg_loss = latent_reg_loss / k_organs

        loss = chamfer_loss + latent_reg_loss

        opt.zero_grad()
        loss.backward()

        # --- Diagnostics: gradient norms (before opt.step clips nothing) ---
        diag = {}
        if step % log_interval == 0:
            enc_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.encoder.parameters(), float('inf'))
            dec_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.decoder.parameters(), float('inf'))
            diag["enc_grad_norm"] = enc_grad_norm.item()
            diag["dec_grad_norm"] = dec_grad_norm.item()

            # Weight norms
            with torch.no_grad():
                enc_w_norm = sum(
                    p.norm().item() ** 2 for p in model.encoder.parameters()
                ) ** 0.5
                dec_w_norm = sum(
                    p.norm().item() ** 2 for p in model.decoder.parameters()
                ) ** 0.5
            diag["enc_weight_norm"] = enc_w_norm
            diag["dec_weight_norm"] = dec_w_norm

            # Velocity magnitude from last training forward
            with torch.no_grad():
                vel_norms = velocities[-1].norm(dim=1)
                diag["vel_mean"] = vel_norms.mean().item()
                diag["vel_std"] = vel_norms.std().item()
                diag["vel_max"] = vel_norms.max().item()

            # Latent code diagnostics (from last organ in this step)
            with torch.no_grad():
                diag["latent_norm"] = latent_code.norm().item()
                # Encode all organs' first clouds to check diversity
                all_latents = []
                for oid_diag in valid_organ_ids:
                    lc = model.encode(organ_data[oid_diag]['clouds'][0][0])
                    all_latents.append(lc)
                all_latents_t = torch.cat(all_latents, dim=0)  # [num_organs, latent_dim]
                latent_norms = all_latents_t.norm(dim=1)
                diag["latent_mean_norm"] = latent_norms.mean().item()
                diag["latent_std_norm"] = latent_norms.std().item()
                if all_latents_t.shape[0] > 1:
                    import torch.nn.functional as F
                    lat_normed = F.normalize(all_latents_t, dim=1)
                    cos_mat = lat_normed @ lat_normed.T
                    mask = ~torch.eye(cos_mat.shape[0], dtype=torch.bool, device=device)
                    diag["latent_mean_cos_sim"] = cos_mat[mask].mean().item()
                else:
                    diag["latent_mean_cos_sim"] = 0.0

        opt.step()

        # Logging
        loss_logs["total"].append(loss.item())
        loss_logs["chamfer"].append(chamfer_loss.item())
        loss_logs["latent_reg"].append(latent_reg_loss.item())

        if step % log_interval == 0:
            sched.step(loss.item())

            if use_wandb:
                wandb_dict = {
                    "train/total": loss.item(),
                    "train/chamfer": chamfer_loss.item(),
                    "train/latent_reg": latent_reg_loss.item(),
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

                        # Encode first cloud
                        eval_latent = model.encode(eval_clouds[0][0])

                        x_eval = eval_clouds[0][0]
                        for ei in range(eval_helpers['num_frames'] - 1):
                            t_n = eval_helpers['get_t_norm'](ei)
                            dt_n = eval_helpers['get_dt'](ei)
                            v = model(x_eval, t_n, eval_latent)
                            organ_vel_norms.append(v.norm(dim=1).mean().item())
                            x_eval = x_eval + v * dt_n
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
                    f"| LatReg={latent_reg_loss.item():.4e} "
                    f"| MeanEvalCD={mean_eval_cd:.6f}"
                )
                print(
                    f"  DIAG: enc_grad={diag.get('enc_grad_norm', 0):.4e} "
                    f"dec_grad={diag.get('dec_grad_norm', 0):.4e} | "
                    f"enc_w={diag.get('enc_weight_norm', 0):.2f} "
                    f"dec_w={diag.get('dec_weight_norm', 0):.2f} | "
                    f"latent_norm={diag.get('latent_mean_norm', 0):.3f} +/- {diag.get('latent_std_norm', 0):.3f} "
                    f"cos_sim={diag.get('latent_mean_cos_sim', 0):.3f}"
                )
                # Per-organ velocity and CD breakdown
                for eval_oid in valid_organ_ids:
                    vs = per_organ_vel_stats.get(eval_oid, {})
                    ocd = per_organ_eval_cds.get(eval_oid, 0.0)
                    print(
                        f"    organ {eval_oid}: CD={ocd:.4f} | "
                        f"vel={vs.get('mean', 0):.4e} +/- {vs.get('std', 0):.4e} "
                        f"(max {vs.get('max', 0):.4e})"
                    )

    if verbose:
        print("  Encoder-decoder model training complete.")

    return model.eval(), loss_logs

