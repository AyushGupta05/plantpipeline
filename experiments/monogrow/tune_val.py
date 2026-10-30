"""Validation-split tuning of the test-time fitting hyper-parameters (fit.split=val: the 4 GrowFields val plants).

  python tune_val.py --ckpt_dir runs/full_cpu --variant headline [--grid small|full] [--resume] [--override fit.k=v ...]

For every grid point: snapshot <ckpt_dir>/{ckpt,prior}.pt (the trainer rewrites them every ckpt_every steps),
run fit_test.py on the val plants into <out>/fits/<config>/, score with metrics.py (full-plant / per-organ CD and
tip EPE on observed frames and, for interpolation / extrapolation, on the held-out frames) and growth_metrics.py
(violation rate, negative-growth mass, length MAE incl. +1/+2/+3-day horizons), and append one row to
<out>/results.csv (keyed by config id + checkpoint step, so re-running after the prior improved adds new rows and
never mixes checkpoints). <out>/summary.md ranks the rows of the newest checkpoint step. Default out: runs/tune/<variant>.
`--resume` skips configs that already have a row for the current checkpoint step. Runs with 2 torch threads.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

import config as C
from metrics import aggregate, evaluate
from growth_metrics import organ_rows, summarise

HERE = Path(__file__).resolve().parent
PY = sys.executable

GRID_FULL = dict(n_z=[300, 1000], lr_z=[3e-3, 1e-2], lambda_prior=[0.0, 1e-3, 1e-1],
                 stage2=[(0, 0.0), (500, 1e-6), (500, 1e-5), (2000, 1e-6), (2000, 1e-5)])
# 6-config subset: z-only around the defaults (one axis at a time) + one cheap stage-2 run
GRID_SMALL = [dict(n_z=300, lr_z=1e-2, lambda_prior=1e-3, n_theta=0, lr_theta=0.0),
              dict(n_z=300, lr_z=3e-3, lambda_prior=1e-3, n_theta=0, lr_theta=0.0),
              dict(n_z=300, lr_z=1e-2, lambda_prior=0.0, n_theta=0, lr_theta=0.0),
              dict(n_z=300, lr_z=1e-2, lambda_prior=1e-1, n_theta=0, lr_theta=0.0),
              dict(n_z=1000, lr_z=1e-2, lambda_prior=1e-3, n_theta=0, lr_theta=0.0),
              dict(n_z=300, lr_z=1e-2, lambda_prior=1e-3, n_theta=500, lr_theta=1e-5)]
PARAMS = ["n_z", "lr_z", "lambda_prior", "n_theta", "lr_theta"]


def grid(name: str) -> list[dict]:
    if name == "small":
        return GRID_SMALL
    g = GRID_FULL
    return [dict(n_z=nz, lr_z=lr, lambda_prior=lp, n_theta=nt, lr_theta=lt if nt else 0.0)
            for nz, lr, lp, (nt, lt) in itertools.product(g["n_z"], g["lr_z"], g["lambda_prior"], g["stage2"])]


def config_id(c: dict) -> str:
    s = f"nz{c['n_z']}_lr{c['lr_z']:g}_lp{c['lambda_prior']:g}_nt{c['n_theta']}"
    return s + (f"_lt{c['lr_theta']:g}" if c["n_theta"] else "")


def checkpoint_step(ckpt_dir: Path, override: int | None = None) -> int:
    """Training step of <ckpt_dir>/ckpt.pt: train_prior.py saves it every ckpt_every steps (and at the end) but does
    not store the step, so take the newest logged step (log.csv, or the `step=N` stdout log in runs/logs/<name>.log,
    which is flushed while log.csv is buffered) rounded down to a checkpoint boundary. --ckpt_step overrides."""
    if override is not None:
        return override
    tc = json.load(open(ckpt_dir / "config.json"))["train"]
    steps = [int(r["step"]) for r in csv.DictReader(open(ckpt_dir / "log.csv"))] if (ckpt_dir / "log.csv").exists() else []
    stdout_log = ckpt_dir.parent / "logs" / f"{ckpt_dir.name}.log"
    if stdout_log.exists():
        steps += [int(m.group(1)) for m in re.finditer(r"^step=(\d+)", stdout_log.read_text(), re.M)]
    last = max(steps, default=0)
    return tc["steps"] if last >= tc["steps"] else last - last % tc["ckpt_every"]


def snapshot_checkpoint(ckpt_dir: Path, out: Path, step_override: int | None = None) -> tuple[Path, int]:
    """Copy ckpt.pt + prior.pt to <out>/ckpt_step<N>/ (retrying while the trainer is mid-write)."""
    for attempt in range(10):
        step = checkpoint_step(ckpt_dir, step_override)
        snap = out / f"ckpt_step{step}"
        if (snap / "ckpt.pt").exists() and (snap / "prior.pt").exists():
            return snap, step
        snap.mkdir(parents=True, exist_ok=True)
        try:
            for f in ("ckpt.pt", "prior.pt"):
                shutil.copy2(ckpt_dir / f, snap / (f + ".tmp"))
                torch.load(snap / (f + ".tmp"), map_location="cpu", weights_only=False)   # complete file?
                os.replace(snap / (f + ".tmp"), snap / f)
            if checkpoint_step(ckpt_dir, step_override) == step:
                return snap, step
        except Exception as e:   # partially written checkpoint: wait for the trainer to finish
            print(f"  checkpoint not readable yet ({e}); retrying in 30 s", flush=True)
            time.sleep(30)
    raise RuntimeError(f"could not snapshot a consistent checkpoint from {ckpt_dir}")


def run_fit(c: dict, fit_dir: Path, snap: Path, variant: str, overrides: list[str], threads: int) -> float:
    if fit_dir.exists():
        shutil.rmtree(fit_dir)
    cmd = [PY, str(HERE / "fit_test.py"), "full", f"fit.variant={variant}", "fit.split=val", f"fit.out={fit_dir}",
           f"fit.ckpt={snap / 'ckpt.pt'}", f"fit.prior={snap / 'prior.pt'}", f"fit.threads={threads}",
           f"fit.n_z={c['n_z']}", f"fit.lr_z={c['lr_z']}", f"fit.lambda_prior={c['lambda_prior']}",
           f"fit.n_theta={c['n_theta']}", f"fit.lr_theta={c['lr_theta'] or 1e-6}", *overrides]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    t0 = time.time()
    fit_dir.mkdir(parents=True, exist_ok=True)
    with open(fit_dir.parent / f"{fit_dir.name}.log", "w") as logf:
        subprocess.run(cmd, check=True, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=HERE)
    return time.time() - t0


def score(fit_dir: Path, variant: str, dc: C.DataConfig) -> dict:
    row = {}
    modes = ["normal"] + ([] if variant == "headline" else [variant])
    for mode in modes:
        full_rows, org_rows = evaluate(fit_dir, dc.tp3d_dir, mode)
        agg = aggregate(full_rows, org_rows, mode)
        pre = "" if mode == "normal" else "heldout_"
        row[pre + "full_cd"], row[pre + "full_epe"] = agg["full"]["Mean"]
        row[pre + "organ_cd"], row[pre + "organ_epe"] = agg["organ"]["Mean"]
        row[pre + "n_tips"] = agg["n_tips_full"]
        json.dump(agg, open(fit_dir / f"monogrow_summary{'' if mode == 'normal' else '_' + mode}.json", "w"), indent=1)
    g = summarise(organ_rows(fit_dir, dc, ["normal", "interpolation", "extrapolation"]))
    row.update(viol_rate=g["violation_rate_pred"], viol_rate_gt=g["violation_rate_gt"], neg_mass_mm=g["neg_growth_mass_mm_pred"],
               len_mae_obs=g["length_mae_mm_observed"], len_mae_heldout=g["length_mae_mm_heldout"],
               len_mae_h1=g["length_mae_mm_horizon+1"], len_mae_h2=g["length_mae_mm_horizon+2"], len_mae_h3=g["length_mae_mm_horizon+3"])
    json.dump(g, open(fit_dir / "growth_summary.json", "w"), indent=1)
    row["fit_time_s"] = sum(json.load(open(p))["aggregate_stats"]["training_time_seconds"] for p in fit_dir.glob("sequences/*/results.json"))
    return row


COLUMNS = ["config", "ckpt_step", *PARAMS, "full_cd", "full_epe", "organ_cd", "organ_epe", "n_tips",
           "heldout_full_cd", "heldout_full_epe", "heldout_organ_cd", "heldout_organ_epe", "heldout_n_tips",
           "viol_rate", "viol_rate_gt", "neg_mass_mm", "len_mae_obs", "len_mae_heldout", "len_mae_h1", "len_mae_h2", "len_mae_h3",
           "fit_time_s", "wall_s", "finished"]


def read_results(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def write_results(path: Path, rows: list[dict]):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def write_summary(path: Path, rows: list[dict], variant: str, out: Path):
    def f(r, k, nd=2):
        try:
            return f"{float(r[k]):.{nd}f}" if r.get(k) not in (None, "", "nan") else "-"
        except ValueError:
            return "-"
    key = "full_cd" if variant == "headline" else "heldout_full_cd"
    rows = [r for r in rows if r.get(key) not in (None, "", "nan")]
    steps = sorted({int(r["ckpt_step"]) for r in rows})
    lines = [f"# Tuning on the 4 GrowFields val plants, variant `{variant}`", "",
             f"Ranked by `{key}` (mm^2). Checkpoint steps seen: {steps}. Fits in `{out / 'fits'}`.", ""]
    hdr = ["rank", "config", "ckpt step", "n_z", "lr_z", "lambda_prior", "n_theta", "lr_theta", "full CD", "full EPE", "organ CD", "organ EPE"]
    if variant != "headline":
        hdr += ["held-out full CD", "held-out full EPE", "held-out organ CD", "held-out organ EPE"]
    hdr += ["viol. rate", "neg mass mm", "len MAE obs", "len MAE held-out"]
    if variant == "extrapolation":
        hdr += ["len MAE +1", "len MAE +2", "len MAE +3"]
    hdr += ["fit s", "wall s"]
    for step in reversed(steps):
        sub = sorted([r for r in rows if int(r["ckpt_step"]) == step], key=lambda r: float(r[key]))
        lines += [f"## checkpoint step {step}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for i, r in enumerate(sub, 1):
            cells = [str(i), r["config"], *(str(r[k]) for k in ("ckpt_step", *PARAMS)),
                     f(r, "full_cd"), f(r, "full_epe"), f(r, "organ_cd"), f(r, "organ_epe")]
            if variant != "headline":
                cells += [f(r, "heldout_full_cd"), f(r, "heldout_full_epe"), f(r, "heldout_organ_cd"), f(r, "heldout_organ_epe")]
            cells += [f(r, "viol_rate", 3), f(r, "neg_mass_mm", 1), f(r, "len_mae_obs", 1), f(r, "len_mae_heldout", 1)]
            if variant == "extrapolation":
                cells += [f(r, "len_mae_h1", 1), f(r, "len_mae_h2", 1), f(r, "len_mae_h3", 1)]
            cells += [f(r, "fit_time_s", 0), f(r, "wall_s", 0)]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", type=Path, required=True, help="train_prior.py output dir (ckpt.pt, prior.pt, log.csv, config.json)")
    ap.add_argument("--variant", default="headline", choices=["headline", "interpolation", "extrapolation"])
    ap.add_argument("--grid", default="small", choices=["small", "full"])
    ap.add_argument("--resume", action="store_true", help="skip configs already scored for the current checkpoint step")
    ap.add_argument("--out", type=Path, default=None, help="default runs/tune/<variant>")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ckpt_step", type=int, default=None, help="training step of ckpt.pt if it cannot be read from the logs")
    ap.add_argument("--wait_minutes", type=float, default=0, help="poll every minute for ckpt.pt + prior.pt to appear, up to this long")
    ap.add_argument("--override", nargs="*", default=[], help="extra fit_test overrides, e.g. fit.points_per_organ=512 fit.sequences=maize_control_plant1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = (a.out or HERE / "runs" / "tune" / a.variant).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_wait = time.time()
    while not all((a.ckpt_dir / f).exists() for f in ("ckpt.pt", "prior.pt")):
        if time.time() - t_wait > a.wait_minutes * 60:
            sys.exit(f"no ckpt.pt + prior.pt in {a.ckpt_dir} after {a.wait_minutes} min")
        print(f"waiting for {a.ckpt_dir}/ckpt.pt ...", flush=True); time.sleep(60)
    snap, step = snapshot_checkpoint(a.ckpt_dir.resolve(), out, a.ckpt_step)
    if step == 0:
        print("warning: checkpoint step unknown (no logged steps yet); rows will carry ckpt_step=0", flush=True)
    configs = grid(a.grid)
    results = read_results(out / "results.csv")
    done = {(r["config"], int(r["ckpt_step"])) for r in results if r.get("finished") == "1"}
    print(f"checkpoint step {step} ({snap}); {len(configs)} configs, {sum((config_id(c), step) in done for c in configs)} already done", flush=True)
    dc = C.DataConfig()
    for i, c in enumerate(configs, 1):
        cid = config_id(c)
        if a.resume and (cid, step) in done:
            print(f"[{i}/{len(configs)}] {cid}: done for step {step}, skipping", flush=True); continue
        print(f"[{i}/{len(configs)}] {cid} ...", flush=True)
        fit_dir = out / "fits" / cid
        t0 = time.time()
        wall = run_fit(c, fit_dir, snap, a.variant, a.override, a.threads)
        row = {"config": cid, "ckpt_step": step, **c, **score(fit_dir, a.variant, dc), "finished": 1}
        row["wall_s"] = time.time() - t0
        results = [r for r in results if not (r["config"] == cid and int(r["ckpt_step"]) == step)] + [row]
        write_results(out / "results.csv", results)
        write_summary(out / "summary.md", results, a.variant, out)
        key = "full_cd" if a.variant == "headline" else "heldout_full_cd"
        print(f"    {key} {row[key]:.2f}  full EPE {row['full_epe']:.2f}  organ CD {row['organ_cd']:.2f}  viol {row['viol_rate']:.3f}  "
              f"len MAE obs {row['len_mae_obs']:.1f}  fit {wall:.0f}s", flush=True)
    write_summary(out / "summary.md", results, a.variant, out)
    print(f"wrote {out / 'results.csv'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()
