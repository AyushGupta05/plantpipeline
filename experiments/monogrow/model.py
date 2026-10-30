"""MonoGrow velocity field.

  v(x, a, c) = softplus(g(x,a,c)) * x_hat + P_perp(x) w(x,a,c),   x_hat = x/||x||,  P_perp = I - x_hat x_hat^T
  c = [z (32-d organ code), species embedding (8-d), plant age / p_max]
so d||x||/dt = softplus(g) >= 0: every point's distance to the organ base is non-decreasing along the flow.
`monotone=False` gives GrowFields' plain 3-d head (v = w) for the ablation.

Trunk: SIREN (Sitzmann et al.) with GrowFields' layer convention and initialisation; two heads from one final
linear layer (1 + 3 outputs). Spatial units are canonical mm / scale_mm, time in days.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig


class SineLayer(nn.Module):
    def __init__(self, i, o, omega, first=False):
        super().__init__()
        self.omega, self.linear = omega, nn.Linear(i, o)
        with torch.no_grad():
            b = 1 / i if first else math.sqrt(6 / i) / omega
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class MonoGrow(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_f = 3 + 1 + cfg.z_dim + cfg.species_dim + 1
        layers = [SineLayer(in_f, cfg.hidden, cfg.omega, first=True)]
        layers += [SineLayer(cfg.hidden, cfg.hidden, cfg.omega) for _ in range(cfg.hidden_layers)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden, 4)   # g (1) + w (3)
        with torch.no_grad():
            b = math.sqrt(6 / cfg.hidden) / cfg.omega
            self.head.weight.uniform_(-b, b)
        self.species = nn.Embedding(cfg.n_species, cfg.species_dim)
        nn.init.normal_(self.species.weight, std=0.1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def velocity(self, x: torch.Tensor, age: float, plant_age: float, z: torch.Tensor, species: int):
        """x (N,3) in scaled units -> v (N,3) scaled units / day, plus (softplus(g), P_perp w) for regularisers."""
        n = x.shape[0]
        c = torch.cat([z, self.species.weight[species]])
        inp = torch.cat([x, x.new_full((n, 1), age / self.cfg.a_max), c.expand(n, -1),
                         x.new_full((n, 1), plant_age / self.cfg.p_max)], dim=1)
        out = self.head(self.trunk(inp))
        g, w = out[:, :1], out[:, 1:]
        if not self.cfg.monotone:
            return w, torch.zeros_like(g), w
        xhat = x / (x.norm(dim=1, keepdim=True) + self.cfg.eps)
        w_perp = w - xhat * (xhat * w).sum(1, keepdim=True)
        sg = F.softplus(g)
        return sg * xhat + w_perp, sg, w_perp

    def time_grid(self, days: np.ndarray) -> np.ndarray:
        """Integration times from days[0] through every target day: GF parity = the target days only,
        default = every dt_days from the start plus the target days themselves (all days here are integers)."""
        days = np.asarray(days, dtype=np.float64)
        if self.cfg.step_per_frame:
            return days
        grid = np.arange(days[0], days[-1] + 1e-9, self.cfg.dt_days)
        return np.unique(np.concatenate([grid, days]))

    def rollout(self, x0_mm: torch.Tensor, days, age0: float, plant_age0: float, z: torch.Tensor, species: int,
                collect_reg: bool = False):
        """Advect x0 (points at days[0], canonical mm) to each of days[1:]. Returns (list of (N,3) mm predictions
        at days[1:], dict of mean regulariser terms). Forward Euler on `time_grid(days)`."""
        grid = self.time_grid(days)
        targets = set(float(d) for d in np.asarray(days)[1:])
        x = x0_mm / self.cfg.scale_mm
        preds, reg_g, reg_w = [], [], []
        for t0, t1 in zip(grid[:-1], grid[1:]):
            v, sg, wp = self.velocity(x, age0 + (t0 - grid[0]), plant_age0 + (t0 - grid[0]), z, species)
            x = x + v * float(t1 - t0)
            if collect_reg:
                reg_g.append(sg.mean()); reg_w.append(wp.pow(2).sum(1).mean())
            if float(t1) in targets:
                preds.append(x * self.cfg.scale_mm)
        reg = {"g": torch.stack(reg_g).mean() if reg_g else x.new_zeros(()),
               "w": torch.stack(reg_w).mean() if reg_w else x.new_zeros(())}
        return preds, reg


def chamfer(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """GrowFields / pytorch3d default: mean_i min_j ||a_i-b_j||^2 + mean_j min_i ||a_i-b_j||^2 (squared, summed)."""
    def nn_sq(p, q):
        out = []
        for s in range(0, p.shape[0], chunk):
            out.append((torch.cdist(p[s:s + chunk], q) ** 2).min(dim=1).values)
        return torch.cat(out)
    return nn_sq(a, b).mean() + nn_sq(b, a).mean()


def multistep_chamfer(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
    """Mean over frames s+1..k of CD(pred, gt) (GrowFields divides the multi-step loss by the number of frames)."""
    return torch.stack([chamfer(p, t) for p, t in zip(preds, targets)]).mean()
