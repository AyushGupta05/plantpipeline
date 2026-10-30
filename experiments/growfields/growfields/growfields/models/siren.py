"""SIREN architectures.

The velocity-field networks themselves: a sine-activated MLP and the four
wrappers that adapt it to a pairwise, continuous, or organ-conditioned model.
Training loops live in `models/siren_dynamics.py`.

`SineLayer` and `Siren` are adapted from SIREN (Sitzmann et al.); see NOTICE.
"""

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """Single SIREN layer with sinusoidal activation."""

    def __init__(self, in_f, out_f, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features, 1 / self.linear.in_features
                )
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """SIREN network for implicit neural representations."""

    def __init__(self, in_f, h_f, h_layers, out_f, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        layers = [SineLayer(in_f, h_f, is_first=True, omega_0=first_omega_0)]
        layers += [SineLayer(h_f, h_f, omega_0=hidden_omega_0) for _ in range(h_layers)]

        final = nn.Linear(h_f, out_f)
        with torch.no_grad():
            bound = np.sqrt(6 / h_f) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FiLMSiren(nn.Module):
    """SIREN with FiLM conditioning — hidden layers accept external scale and shift."""

    def __init__(self, in_f, h_f, h_layers, out_f, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.first_layer = SineLayer(in_f, h_f, is_first=True, omega_0=first_omega_0)
        self.hidden_layers = nn.ModuleList(
            [SineLayer(h_f, h_f, omega_0=hidden_omega_0) for _ in range(h_layers)]
        )
        self.final = nn.Linear(h_f, out_f)
        with torch.no_grad():
            bound = np.sqrt(6 / h_f) / hidden_omega_0
            self.final.weight.uniform_(-bound, bound)

    def forward(self, x, gammas, betas):
        """
        Args:
            x: input tensor [N, in_f]
            gammas: list of scale tensors, one per hidden layer [N, h_f]
            betas: list of shift tensors, one per hidden layer [N, h_f]
        """
        h = self.first_layer(x)
        for i, layer in enumerate(self.hidden_layers):
            h = gammas[i] * h + betas[i]
            h = layer(h)
        return self.final(h)


class PairwiseSiren(nn.Module):
    """SIREN model for learning deformation between two point clouds."""

    def __init__(self, h_f=128, h_layers=3, omega_0=30):
        super().__init__()
        self.model = Siren(3, h_f, h_layers, 3, omega_0, omega_0)

    def forward(self, x):
        return self.model(x)


class ContinuousSiren(nn.Module):
    """Time-aware SIREN that outputs velocity field for continuous deformation."""

    def __init__(self, h_f=256, h_layers=5, first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        # Input: (x, y, z, t_norm) → Output: (vx, vy, vz)
        self.model = Siren(4, h_f, h_layers, 3, first_omega_0, hidden_omega_0)

    def forward(self, x, t_norm):
        """
        Args:
            x: Point cloud [N, 3]
            t_norm: Normalized time scalar or tensor [N, 1]

        Returns:
            velocity: [N, 3]
        """
        if isinstance(t_norm, float):
            t_norm = torch.full((x.shape[0], 1), t_norm, device=x.device, dtype=x.dtype)
        elif t_norm.dim() == 0:
            t_norm = t_norm.view(1, 1).expand(x.shape[0], 1)

        model_input = torch.cat([x, t_norm], dim=1)
        return self.model(model_input)


class SharedContinuousSiren(nn.Module):
    """Organ-conditioned shared SIREN for cross-organ velocity field learning.

    A single SIREN model conditioned on learnable organ embeddings, enabling
    cross-organ transfer and an interpretable growth phenotype space.

    Supports three conditioning modes:
        - "concat": embedding concatenated to input (simple, default)
        - "none": no organ conditioning — backbone takes only (x,y,z,t). Ablation
          for the relevance of the latent code: identical shared-backbone training
          loop, but the model cannot distinguish organs (xyzt input only).
        - "film": FiLM conditioning — embedding produces per-layer scale/shift,
          modulating every hidden layer for stronger organ-dependent behavior

    Input: (x, y, z, t_norm) [+ organ embedding if concat] -> (vx, vy, vz)
    """

    def __init__(self, num_organs, embed_dim=16, h_f=256, h_layers=5,
                 first_omega_0=30.0, hidden_omega_0=30.0, conditioning="concat",
                 embed_init="normal"):
        super().__init__()
        self.num_organs = num_organs
        self.embed_dim = embed_dim
        self.conditioning = conditioning
        self.h_layers = h_layers
        self.organ_embedding = nn.Embedding(num_organs, embed_dim)
        if embed_init == "onehot" and embed_dim >= num_organs:
            with torch.no_grad():
                self.organ_embedding.weight.zero_()
                self.organ_embedding.weight[:num_organs, :num_organs] = torch.eye(num_organs)
        else:
            nn.init.normal_(self.organ_embedding.weight, mean=0.0, std=0.1)
        self.organ_id_remap = None  # Set externally if organ IDs have gaps

        if conditioning == "none":
            # No conditioning — backbone takes only (x,y,z,t). The organ embedding
            # is still created so trainer/logging references remain valid, but it is
            # never used in forward(), so it cannot influence the output.
            self.model = Siren(4, h_f, h_layers, 3, first_omega_0, hidden_omega_0)
        elif conditioning == "film":
            # SIREN backbone takes only (x,y,z,t) — embedding modulates hidden layers
            self.model = FiLMSiren(4, h_f, h_layers, 3, first_omega_0, hidden_omega_0)
            # Mapping from embedding to per-layer gamma (scale) and beta (shift)
            self.film_gen = nn.Linear(embed_dim, h_layers * h_f * 2)
            # Init: gamma~1, beta~0 so FiLM starts as identity
            with torch.no_grad():
                self.film_gen.weight.zero_()
                # bias: first half (gammas) = 1, second half (betas) = 0
                bias = torch.zeros(h_layers * h_f * 2)
                bias[:h_layers * h_f] = 1.0
                self.film_gen.bias.copy_(bias)
            self.h_f = h_f
        else:
            # in_f = 4 (x,y,z,t) + embed_dim
            self.model = Siren(4 + embed_dim, h_f, h_layers, 3, first_omega_0, hidden_omega_0)

    def forward(self, x, t_norm, organ_id):
        """
        Args:
            x: Point cloud [N, 3]
            t_norm: Normalized time scalar, 0-d tensor, or [N, 1] tensor
            organ_id: Integer organ ID or tensor of organ IDs

        Returns:
            velocity: [N, 3]
        """
        N = x.shape[0]
        if isinstance(t_norm, float):
            t_norm = torch.full((N, 1), t_norm, device=x.device, dtype=x.dtype)
        elif t_norm.dim() == 0:
            t_norm = t_norm.view(1, 1).expand(N, 1)

        if self.conditioning == "none":
            # Ablation: no organ identity — output depends only on (x,y,z,t).
            return self.model(torch.cat([x, t_norm], dim=1))

        # Remap organ_id if needed
        if self.organ_id_remap is not None and isinstance(organ_id, int):
            organ_id = self.organ_id_remap[organ_id]

        if isinstance(organ_id, int):
            oid = torch.tensor([organ_id], device=x.device, dtype=torch.long)
        else:
            oid = organ_id
        z = self.organ_embedding(oid).expand(N, -1)  # [N, embed_dim]

        if self.conditioning == "film":
            xt = torch.cat([x, t_norm], dim=1)  # [N, 4]
            film_params = self.film_gen(z)  # [N, h_layers * h_f * 2]
            gammas, betas = film_params.split(self.h_layers * self.h_f, dim=1)
            gammas = gammas.reshape(N, self.h_layers, self.h_f)
            betas = betas.reshape(N, self.h_layers, self.h_f)
            # Split into per-layer lists
            gammas = [gammas[:, i, :] for i in range(self.h_layers)]
            betas = [betas[:, i, :] for i in range(self.h_layers)]
            return self.model(xt, gammas, betas)
        else:
            return self.model(torch.cat([x, t_norm, z], dim=1))

    def get_all_embeddings(self):
        """Return detached copy of all organ embeddings."""
        return self.organ_embedding.weight.detach()


class MonotoneSharedContinuousSiren(SharedContinuousSiren):
    """Organ-conditioned SIREN with a monotone (outward-only) velocity factorisation.

    Added for the plantpipeline "MonoGrow" experiment (DESIGN_MEMO.md, Candidate A).
    Same trunk, conditioning, embeddings and remap as `SharedContinuousSiren`; the
    only differences are a 4-d head instead of 3-d and how the head is turned into
    a velocity. With r = x - b (b = per-organ base point in the model's normalised
    coordinates), r_hat = r / ||r|| and P_perp = I - r_hat r_hat^T:

        v(x, t, z) = softplus(g(x, t, z)) * r_hat + P_perp(x) * w(x, t, z)

    so d||r||/dt = softplus(g) >= 0: every point's distance from the organ base is
    non-decreasing along the flow (monotone growth), while P_perp w carries
    bending/twisting. Where ||r|| < eps, r_hat is set to 0 (v = w, no radial term).

    `organ_base` is a buffer [num_organs, 3] indexed by *dense* organ index (after
    `organ_id_remap`), set by the trainer from the organ data and saved with the
    state dict. Leaves (canonical frame, base at the origin) get the normalised
    image of the origin; the stem (organ 0, plant space) gets its soil point.

    Optional regularisers are read by the trainer from `pop_reg_terms()` when
    `collect_reg` is True: per forward call it records (mean softplus(g),
    mean ||P_perp w||^2).
    """

    def __init__(self, num_organs, embed_dim=16, h_f=256, h_layers=5,
                 first_omega_0=30.0, hidden_omega_0=30.0, conditioning="concat",
                 embed_init="normal", base_eps=1e-3, g_bias_init=-2.0):
        super().__init__(num_organs, embed_dim, h_f, h_layers, first_omega_0,
                         hidden_omega_0, conditioning, embed_init)
        # Rebuild the trunk with a 4-d head: [g, w_x, w_y, w_z].
        if conditioning == "none":
            self.model = Siren(4, h_f, h_layers, 4, first_omega_0, hidden_omega_0)
        elif conditioning == "film":
            self.model = FiLMSiren(4, h_f, h_layers, 4, first_omega_0, hidden_omega_0)
        else:
            self.model = Siren(4 + embed_dim, h_f, h_layers, 4, first_omega_0, hidden_omega_0)
        final = self.model.final if conditioning == "film" else self.model.net[-1]
        with torch.no_grad():
            final.bias[0] = g_bias_init  # softplus(-2) ~ 0.13: start with little growth
        self.base_eps = base_eps
        self.register_buffer("organ_base", torch.zeros(num_organs, 3))
        self.collect_reg = False
        self._reg_terms = []

    def _dense_id(self, organ_id):
        if not isinstance(organ_id, int):
            organ_id = int(organ_id.reshape(-1)[0].item())
        if self.organ_id_remap is not None:
            organ_id = self.organ_id_remap.get(organ_id, organ_id)
        return organ_id

    def _head(self, x, t_norm, organ_id):
        """Raw 4-d head output; mirrors `SharedContinuousSiren.forward` input assembly."""
        N = x.shape[0]
        if isinstance(t_norm, float):
            t_norm = torch.full((N, 1), t_norm, device=x.device, dtype=x.dtype)
        elif t_norm.dim() == 0:
            t_norm = t_norm.view(1, 1).expand(N, 1)
        if self.conditioning == "none":
            return self.model(torch.cat([x, t_norm], dim=1))
        oid = torch.tensor([self._dense_id(organ_id)], device=x.device, dtype=torch.long)
        z = self.organ_embedding(oid).expand(N, -1)
        if self.conditioning == "film":
            xt = torch.cat([x, t_norm], dim=1)
            film_params = self.film_gen(z)
            gammas, betas = film_params.split(self.h_layers * self.h_f, dim=1)
            gammas = gammas.reshape(N, self.h_layers, self.h_f)
            betas = betas.reshape(N, self.h_layers, self.h_f)
            gammas = [gammas[:, i, :] for i in range(self.h_layers)]
            betas = [betas[:, i, :] for i in range(self.h_layers)]
            return self.model(xt, gammas, betas)
        return self.model(torch.cat([x, t_norm, z], dim=1))

    def forward(self, x, t_norm, organ_id):
        out = self._head(x, t_norm, organ_id)
        growth = torch.nn.functional.softplus(out[:, :1])  # [N, 1] >= 0
        w = out[:, 1:]  # [N, 3]
        r = x - self.organ_base[self._dense_id(organ_id)]
        norm = r.norm(dim=1, keepdim=True)
        r_hat = torch.where(norm > self.base_eps, r / norm.clamp_min(self.base_eps),
                            torch.zeros_like(r))
        w_perp = w - (w * r_hat).sum(dim=1, keepdim=True) * r_hat
        if self.collect_reg and self.training:
            self._reg_terms.append((growth.mean(), w_perp.pow(2).sum(dim=1).mean()))
        return growth * r_hat + w_perp

    def pop_reg_terms(self):
        """Return and clear the (mean softplus(g), mean ||P_perp w||^2) pairs recorded since last call."""
        terms, self._reg_terms = self._reg_terms, []
        return terms
