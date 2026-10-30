"""Species-level hierarchical prior for RichardsNet, estimated from the TRAINING organ tracks (MonoGrow's
load_training_tracks: TrackPlant3D train split + Pheno4D minus the leaked plants + Sugar4D).

  python prior.py [--out prior.npz] [--min_frames 4] [--iters 2]

Per curve family and per group species/{leaf,stem}:
  * Gaussian N(mu, Sigma) over theta (4 unconstrained curve parameters, curves.py), by empirical Bayes: every track
    with >= min_frames frames is fitted by robust least squares, pooled (3-MAD clipping, shrinkage of mu/Sigma towards
    the pooled organ-type Gaussian with 5 pseudo-organs), then `iters` rounds of refit-with-prior + re-pool.
  * rank slope: ridge regression of theta on (leaf rank - mean rank), rank = order of first appearance in the plant.
  * allometry: beta from within-organ regression of log W on log L, between-organ sd of per-organ slopes, residual sd.
  * phyllochron: mean / sd of the days between successive leaf emergences (leaves first seen after the plant's first scan).
Writes prior.npz (one pickled dict) and prior_table.md.
"""

from __future__ import annotations

import argparse
import collections
import pickle

import numpy as np

from common import GROUPS, MONOGROW, fit_theta, group_of, leaf_ranks, series  # noqa: F401  (sets threads / sys.path)
from config import DataConfig
from curves import FAMILIES, PARAM_NAMES
from data import load_training_tracks

PSEUDO = 5.0          # shrinkage pseudo-count towards the pooled organ-type Gaussian
RIDGE = np.diag([0.05, 0.05, 1.0, 0.05]) ** 2   # floor on Sigma (log A, log k, a_m [days], log nu  /  beta analogues)


def robust_gauss(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(T, 0); mad = 1.4826 * np.median(np.abs(T - med), 0) + 1e-6
    keep = np.all(np.abs(T - med) <= 3 * mad, 1)
    T = T[keep] if keep.sum() >= 3 else T
    return T.mean(0), np.atleast_2d(np.cov(T.T)) + RIDGE if len(T) > 1 else RIDGE


def pool(thetas: dict[str, list[np.ndarray]]) -> dict[str, dict]:
    """Per-group Gaussian, shrunk towards the organ-type pool (all species) with PSEUDO pseudo-organs."""
    out = {}
    for otype in ("leaf", "stem"):
        allT = np.array([th for g, ts in thetas.items() if g.endswith(otype) for th in ts])
        mu0, S0 = robust_gauss(allT)
        for g, ts in thetas.items():
            if not g.endswith(otype):
                continue
            T = np.array(ts); n = len(T)
            mu, S = robust_gauss(T) if n >= 3 else (mu0, S0)
            w = n / (n + PSEUDO)
            out[g] = {"mu": w * mu + (1 - w) * mu0, "cov": w * S + (1 - w) * S0, "n": n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MONOGROW.parent / "richards" / "prior.npz"))
    ap.add_argument("--min_frames", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2)
    a = ap.parse_args()

    tracks = load_training_tracks(DataConfig(), verbose=False)
    by_plant = collections.defaultdict(list)
    for t in tracks:
        by_plant[(t.dataset, t.seq)].append(t)
    rank = {}
    phyllo = collections.defaultdict(list)
    for (ds, seq), ts in by_plant.items():
        r = leaf_ranks(ts, [t.frame_idx[0] for t in ts])
        for t in ts:
            rank[id(t)] = r.get(t.organ_id)
        emerged = sorted(t.days[0] for t in ts if not t.is_stem and t.frame_idx[0] > 0)
        phyllo[ts[0].species] += list(np.diff(emerged))

    prior = {"families": list(FAMILIES), "groups": {}, "phyllochron": {}, "n_tracks_used": 0}
    use = [t for t in tracks if len(t.days) >= a.min_frames]
    prior["n_tracks_used"] = len(use)
    data = [(group_of(t), rank[id(t)], *series(t, list(range(len(t.days))))) for t in use]
    print(f"{len(tracks)} training tracks, {len(use)} with >= {a.min_frames} frames")

    for fam in FAMILIES:
        flat = {g: np.zeros(4) for g in {d[0] for d in data}}
        wide = np.eye(4) * 1e4                                # stage 0: essentially unregularised robust fit
        gauss = None
        for it in range(a.iters + 1):
            thetas = collections.defaultdict(list)
            for g, _, ages, L, _ in data:
                mu, cov = (flat[g], wide) if gauss is None else (gauss[g]["mu"], gauss[g]["cov"])
                thetas[g].append(fit_theta(fam, ages, L, mu, np.linalg.inv(cov), lam=1.0, robust=(gauss is None)))
            gauss = pool(thetas)
            print(f"  {fam} EB iteration {it}: " + ", ".join(f"{g}: n={v['n']}" for g, v in sorted(gauss.items())))
        # rank slope (leaves): ridge regression of theta on centred rank, per group (thetas[g] is in `data` order)
        for g, v in gauss.items():
            v["rank_slope"] = np.zeros(4); v["rank_mean"] = 1.0
            if not g.endswith("leaf"):
                continue
            ranks = [d[1] for d in data if d[0] == g]
            keep = [i for i, r in enumerate(ranks) if r is not None]
            if len(keep) < 10:
                continue
            T = np.array(thetas[g])[keep]; R = np.array([ranks[i] for i in keep], float)
            rc = R - R.mean()
            v["rank_slope"] = (rc @ (T - T.mean(0))) / (rc @ rc + 10.0)
            v["rank_mean"] = float(R.mean())
        prior["groups"][fam] = gauss

    # allometry per group (family independent)
    allo = {}
    for g in {d[0] for d in data}:
        lL, lW, slopes, res = [], [], [], []
        for gg, _, ages, L, W in data:
            if gg != g:
                continue
            x, y = np.log(np.maximum(L, 1e-3)), np.log(np.maximum(W, 1e-3))
            lL.append(x - x.mean()); lW.append(y - y.mean())
            if len(x) >= 4 and np.var(x) > 1e-4:
                slopes.append(np.polyfit(x, y, 1)[0])
        x, y = np.concatenate(lL), np.concatenate(lW)
        beta = float((x @ y) / (x @ x + 1e-9))
        allo[g] = {"beta": beta, "sigma_beta": float(np.std(slopes)) if len(slopes) > 2 else 0.5,
                   "sigma_eps": float(np.std(y - beta * x)), "n": len(lL)}
    prior["allometry"] = allo
    for s, d in phyllo.items():
        d = np.array(d, float)
        prior["phyllochron"][s] = {"mean": float(d.mean()) if len(d) else float("nan"), "sd": float(d.std()) if len(d) > 1 else float("nan"), "n": int(len(d))}

    np.savez(a.out, prior=np.array(pickle.dumps(prior)))
    lines = ["# RichardsNet prior (fitted parameters per species)", "",
             f"{len(tracks)} training organ tracks, {len(use)} with >= {a.min_frames} frames used for the curve fits.", ""]
    for fam in FAMILIES:
        lines += [f"## {fam}: mean of theta = ({', '.join(PARAM_NAMES[fam])}) and sd per dimension", "",
                  "| group | n | " + " | ".join(PARAM_NAMES[fam]) + " | natural-scale mean | rank slope |", "|---|---|" + "---|" * (4 + 2)]
        for g, v in sorted(prior["groups"][fam].items()):
            sd = np.sqrt(np.diag(v["cov"])); nat = natural(fam, v["mu"])
            lines.append(f"| {g} | {v['n']} | " + " | ".join(f"{m:.2f} +- {s:.2f}" for m, s in zip(v["mu"], sd)) + f" | {nat} | " +
                         ", ".join(f"{x:+.3f}" for x in v["rank_slope"]) + " |")
        lines.append("")
    lines += ["## Width allometry W = alpha L^beta (within-organ regression)", "", "| group | n organs | beta | sd(beta) across organs | residual sd (log) |", "|---|---|---|---|---|"]
    for g, v in sorted(allo.items()):
        lines.append(f"| {g} | {v['n']} | {v['beta']:.3f} | {v['sigma_beta']:.3f} | {v['sigma_eps']:.3f} |")
    lines += ["", "## Phyllochron (days between successive leaf emergences, leaves first seen after the first scan)", "", "| species | n intervals | mean | sd |", "|---|---|---|---|"]
    for s, v in sorted(prior["phyllochron"].items()):
        lines.append(f"| {s} | {v['n']} | {v['mean']:.2f} | {v['sd']:.2f} |")
    open(str(a.out).replace(".npz", "_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def natural(fam: str, mu: np.ndarray) -> str:
    if fam == "richards":
        return f"A={np.exp(mu[0]):.0f} mm, k={np.exp(mu[1]):.3f}/d, a_m={mu[2]:.1f} d, nu={np.exp(mu[3]):.2f}"
    ae = np.exp(mu[1]); am = ae * (0.05 + 0.9 / (1 + np.exp(-mu[2])))
    return f"L_m={np.exp(mu[0]):.0f} mm, a_e={ae:.1f} d, a_m={am:.1f} d, a_0={np.exp(mu[3]):.1f} d"


def load_prior(path) -> dict:
    return pickle.loads(np.load(path, allow_pickle=True)["prior"].item())


if __name__ == "__main__":
    main()
