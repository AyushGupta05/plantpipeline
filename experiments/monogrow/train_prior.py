"""Train the MonoGrow prior: one velocity field theta shared by every eligible training organ track, with a
per-organ auto-decoded code z_i (DeepSDF style), then fit q(z | species) = N(mu_s, Sigma_s).

  python train_prior.py smoke                      # CPU, minutes
  python train_prior.py full train.out=runs/full   # one GPU
  any `section.key=value` from config.py can be appended

Writes <out>/log.csv, <out>/ckpt.pt (theta, all z, track metadata, config) and <out>/prior.pt.
"""

from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
import torch

import config as C
from data import TrackSampler, load_training_tracks, to_tensor
from model import MonoGrow, multistep_chamfer


def fit_species_prior(z: np.ndarray, species: list[str], ridge: float, shrink: float) -> dict:
    """Gaussian per species; covariance shrunk towards the pooled covariance with `shrink` pseudo-organs."""
    mu_all, cov_all = z.mean(0), np.cov(z.T) + ridge * np.eye(z.shape[1])
    mu, cov, n = [], [], []
    for s in C.SPECIES:
        zs = z[[i for i, sp in enumerate(species) if sp == s]]
        n.append(len(zs))
        if len(zs) == 0:
            mu.append(mu_all); cov.append(cov_all); continue
        c_s = np.cov(zs.T) if len(zs) > 1 else np.zeros_like(cov_all)
        mu.append(zs.mean(0))
        cov.append((len(zs) * c_s + shrink * cov_all) / (len(zs) + shrink) + ridge * np.eye(z.shape[1]))
    mu, cov = np.stack(mu), np.stack(cov)
    return {"species": C.SPECIES, "mu": torch.tensor(mu, dtype=torch.float32), "cov": torch.tensor(cov, dtype=torch.float32),
            "cov_diag": torch.tensor(np.stack([np.diag(c) for c in cov]), dtype=torch.float32), "n": n}


def main(argv):
    cfg = C.load(argv[0], argv[1:])
    tc, mc = cfg.train, cfg.model
    torch.set_num_threads(tc.threads); torch.manual_seed(tc.seed); np.random.seed(tc.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc.out.mkdir(parents=True, exist_ok=True)
    json.dump(C.as_dict(cfg), open(tc.out / "config.json", "w"), indent=1)

    tracks = load_training_tracks(cfg.data)
    if tc.max_tracks and tc.max_tracks < len(tracks):
        rng = np.random.default_rng(tc.seed)
        tracks = [tracks[i] for i in sorted(rng.choice(len(tracks), tc.max_tracks, replace=False))]
    print(f"{len(tracks)} organ tracks, {sum(len(t.days) for t in tracks)} organ frames, device={device}")

    model = MonoGrow(mc).to(device)
    z = torch.nn.Parameter(torch.randn(len(tracks), mc.z_dim, device=device) * tc.z_init_std)
    print(f"theta: {model.n_params():,} parameters; z: {len(tracks)} x {mc.z_dim}")
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": tc.lr_theta}, {"params": [z], "lr": tc.lr_z}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.steps) if tc.lr_decay == "cosine" else None
    sampler = TrackSampler(tracks, tc.p_consecutive, tc.max_horizon_days, tc.seed, balance_species=tc.balance_species)

    log = csv.writer(open(tc.out / "log.csv", "w", newline=""))
    log.writerow(["step", "loss", "cd_mm2", "reg_g", "reg_w", "z_norm", "seconds"])
    t0 = time.time()
    for step in range(1, tc.steps + 1):
        batch = sampler.batch(tc.organs_per_step, tc.points_per_step)
        cd_sum = reg_g = reg_w = 0.0
        for it in batch:
            preds, reg = model.rollout(to_tensor(it["x0"], device), it["days"], it["age0"], it["plant_age0"],
                                       z[it["track"]], it["species"], collect_reg=True)
            cd = multistep_chamfer([p / mc.scale_mm for p in preds], [to_tensor(t, device) / mc.scale_mm for t in it["targets"]])
            cd_sum = cd_sum + cd; reg_g = reg_g + reg["g"]; reg_w = reg_w + reg["w"]
        k = len(batch)
        cd_sum, reg_g, reg_w = cd_sum / k, reg_g / k, reg_w / k
        loss = cd_sum + tc.lambda_z * z.pow(2).sum(1).mean() + tc.lambda_w * reg_w + tc.lambda_g * reg_g
        if not torch.isfinite(loss):
            print(f"step={step} non-finite loss, skipped", flush=True); opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [z], tc.grad_clip)
        opt.step()
        if sched is not None: sched.step()

        if step % tc.log_every == 0 or step == 1:
            row = [step, f"{loss.item():.6f}", f"{cd_sum.item() * mc.scale_mm ** 2:.4f}", f"{float(reg_g.detach()):.4f}",
                   f"{float(reg_w.detach()):.4f}", f"{z.norm(dim=1).mean().item():.3f}", f"{time.time() - t0:.1f}"]
            log.writerow(row); print(" ".join(f"{a}={b}" for a, b in zip(["step", "loss", "cd_mm2", "g", "w", "|z|", "s"], row)), flush=True)
        if step % tc.ckpt_every == 0 or step == tc.steps:
            save(cfg, model, z, tracks)
        if step % tc.keep_every == 0:   # tagged snapshots survive the rolling ckpt.pt (late-training drift seen at 1e-4 and 2e-5)
            import shutil; shutil.copy(cfg.train.out / "ckpt.pt", cfg.train.out / f"ckpt_{step}.pt"); shutil.copy(cfg.train.out / "prior.pt", cfg.train.out / f"prior_{step}.pt")
    print(f"done: {tc.steps} steps in {time.time() - t0:.0f}s ({(time.time() - t0) / tc.steps:.3f} s/step)")


def save(cfg, model, z, tracks):
    meta = [{"seq": t.seq, "organ_id": t.organ_id, "species": t.species, "dataset": t.dataset, "n_frames": len(t.days)} for t in tracks]
    torch.save({"model": model.state_dict(), "z": z.detach().cpu(), "tracks": meta, "config": C.as_dict(cfg)}, cfg.train.out / "ckpt.pt")
    prior = fit_species_prior(z.detach().cpu().numpy(), [m["species"] for m in meta], cfg.train.prior_ridge, cfg.train.prior_shrink)
    torch.save(prior, cfg.train.out / "prior.pt")


if __name__ == "__main__":
    main(sys.argv[1:] or ["smoke"])
