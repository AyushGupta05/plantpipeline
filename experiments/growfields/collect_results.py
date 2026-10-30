#!/usr/bin/env python
"""Collect results for one config: runs/<config>/results.md.

Runs GrowFields' own metric scripts on the experiment directory (metrics are only ever
computed from the exported PLYs), their LaTeX table scripts (--per_sequence, the paper's
aggregation), and growth_metrics.py, then writes a Markdown summary with CD (mm^2) / EPE (mm)
per sequence and Mean, plus monotonicity-violation rate and organ-length MAE.

Aggregation mirrors REPRODUCIBILITY.md: full plant = mean over frames per (plant, run),
mean over runs, unweighted mean over plants; per organ = frames -> organs (equal weight)
-> runs -> plants, organs with < 2 scored frames dropped.

Usage (CWD must be experiments/growfields: compute_metrics_per_organ.py reads
data/TrackPlant3D/versions relative to CWD):
    .venv/bin/python collect_results.py <config> [--experiment_dir <exp>] [--skip_metrics]
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GF = HERE / "growfields"
PY = sys.executable
ABBR = {
    "maize_control_plant2": "MaC2", "maize_control_plant3": "MaC3",
    "sorghum_control_plant2": "SoC2", "sorghum_highlight_plant2": "SoH2",
    "tobacco_control_plant1": "TbC1", "tobacco_shade_plant3": "TbS3",
    "tomato1_heat_plant3": "Tm1H3", "tomato1_shade_plant1": "Tm1S1",
}
PAPER = {  # Table 1 / Table 2 "Ours (full)" CD / EPE
    "full": {"MaC2": (2.51, 2.38), "MaC3": (1.43, 2.01), "SoC2": (0.69, 1.34), "SoH2": (0.74, 0.97),
             "TbC1": (0.59, 1.66), "TbS3": (0.45, 1.30), "Tm1H3": (0.12, 0.80), "Tm1S1": (0.32, 1.34),
             "Mean": (0.86, 1.47)},
    "organ": {"MaC2": (2.26, 2.59), "MaC3": (1.48, 2.46), "SoC2": (0.93, 1.51), "SoH2": (0.64, 0.85),
              "TbC1": (0.49, 2.04), "TbS3": (0.45, 1.20), "Tm1H3": (0.20, 0.98), "Tm1S1": (0.46, 1.33),
              "Mean": (0.87, 1.62)},
}


def run(cmd, cwd=HERE):
    print("+", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r.stdout


def fullplant_table(csv):
    df = pd.read_csv(csv)
    per = df.groupby(["sequence", "run"]).agg(cd=("cd_test", "mean"), epe=("epe", "mean"),
                                              n_tips=("n_tips", "sum"), t=("training_time_seconds", "first"))
    seq = per.groupby("sequence").agg(cd=("cd", "mean"), epe=("epe", "mean"), cd_std=("cd", "std"),
                                      epe_std=("epe", "std"), n_tips=("n_tips", "first"), t=("t", "mean"), runs=("cd", "size"))
    return seq


def perorgan_table(csv, min_tp=2):
    df = pd.read_csv(csv)
    cnt = df.groupby(["sequence", "run", "organ_id"])["timepoint"].transform("size")
    df = df[cnt >= min_tp]
    org = df.groupby(["sequence", "run", "organ_id"]).agg(cd=("cd_test", "mean"), epe=("epe", "mean"), n_tips=("n_tips", "sum"))
    per = org.groupby(["sequence", "run"]).agg(cd=("cd", "mean"), epe=("epe", "mean"), n_tips=("n_tips", "sum"), n_organs=("cd", "size"))
    seq = per.groupby("sequence").agg(cd=("cd", "mean"), epe=("epe", "mean"), n_tips=("n_tips", "first"), n_organs=("n_organs", "first"))
    return seq


def md_table(seq, kind, growth=None):
    rows = ["| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips |"
            + (" viol. rate (GT floor) | neg. mass mm | length MAE mm |" if growth is not None else ""),
            "|---|---|---|---|---|---|" + ("---|---|---|" if growth is not None else "")]
    for name, r in seq.iterrows():
        ab = ABBR.get(name, name)
        p = PAPER[kind].get(ab, (np.nan, np.nan))
        extra = ""
        if growth is not None and name in growth.index:
            g = growth.loc[name]
            extra = f" {g.viol_rate:.3f} ({g.viol_rate_gt:.3f}) | {g.neg_mass:.2f} | {g.len_mae:.2f} |"
        scored = f"{int(r.get('n_organs', 0))} organs" if "n_organs" in r else f"{int(r.get('runs', 1))} run(s)"
        rows.append(f"| {ab} | {scored} | {r.cd:.2f} | {r.epe:.2f} | {p[0]:.2f} / {p[1]:.2f} | {int(r.n_tips)} |{extra}")
    m = seq.mean(numeric_only=True)
    p = PAPER[kind]["Mean"]
    extra = ""
    if growth is not None and "Mean" in growth.index:
        g = growth.loc["Mean"]
        extra = f" {g.viol_rate:.3f} ({g.viol_rate_gt:.3f}) | {g.neg_mass:.2f} | {g.len_mae:.2f} |"
    rows.append(f"| **Mean ({len(seq)} plants)** | | **{m.cd:.2f}** | **{m.epe:.2f}** | {p[0]:.2f} / {p[1]:.2f} | {int(seq.n_tips.sum())} |{extra}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--experiment_dir", default=None)
    ap.add_argument("--skip_metrics", action="store_true", help="reuse existing CSVs")
    args = ap.parse_args()
    run_root = HERE / "runs" / args.config
    exp = Path(args.experiment_dir) if args.experiment_dir else sorted((run_root / "results").glob("*fullplant_batch*"))[-1]
    if not args.skip_metrics or not (exp / "all_metrics.csv").exists():
        run([PY, GF / "scripts/compute_metrics_per_organ.py", "--experiment_dir", exp, "--quiet"])
        run([PY, GF / "scripts/compute_metrics_fullplant.py", "--experiment_dir", exp, "--quiet"])
    latex_full = run([PY, GF / "scripts/create_latex_table_fullplant.py", "--csv", exp / "fullplant_metrics.csv",
                      "--full_table", "--per_sequence", "--label", args.config])
    latex_organ = run([PY, GF / "scripts/create_latex_table_per_organ.py", "--csv", exp / "all_metrics.csv",
                       "--full_table", "--per_sequence", "--min_timepoints_per_organ", "2", "--label", args.config])
    growth_out = run([PY, HERE / "growth_metrics.py", "--experiment_dir", exp])
    growth = pd.read_csv(exp / "growth_metrics_summary.csv", index_col=0)

    full = fullplant_table(exp / "fullplant_metrics.csv")
    organ = perorgan_table(exp / "all_metrics.csv")
    seqs = sorted((exp / "sequences").iterdir())
    logs = sorted(run_root.glob("*.log"))
    md = [f"# {args.config}: results", f"Generated {datetime.now():%Y-%m-%d %H:%M}. Experiment dir: `{exp}`.",
          f"Sequences present: {len(seqs)} of 8 test plants. Integrity: full-plant n_tips sum = {int(full.n_tips.sum())} "
          f"(192 expected for 8 plants, 1 run), per-organ = {int(organ.n_tips.sum())} (189 expected).", "",
          "## Full-plant (Table 1 aggregation, cd_test over all frames t>=1)", md_table(full, "full"), "",
          "Training time per plant (s, first run): " + ", ".join(f"{ABBR.get(n, n)} {r.t:.0f}" for n, r in full.iterrows()), "",
          "## Per-organ (Table 2 aggregation, organs with >= 2 scored frames) + growth metrics",
          "Growth metrics from `growth_metrics.py`: organ length L = 99th-percentile axial (z) extent per exported frame in the "
          "canonical organ frame; violation = L(t+1) < L(t) - 2 mm over consecutive exported frames; GT floor in parentheses; "
          "length MAE = mean |L_pred - L_gt| (mm).",
          md_table(organ, "organ", growth), "",
          "## GrowFields' own LaTeX table rows (`create_latex_table_*.py --per_sequence`; full plant, then per organ)", "```",
          *[l for l in latex_full.splitlines() if l.startswith(args.config)],
          *[l for l in latex_organ.splitlines() if l.startswith(args.config) or "Removed" in l or "timepoints (avg" in l], "```", "",
          "## growth_metrics.py output", "```", growth_out.strip(), "```", "",
          "## Logs", *[f"- `{l.relative_to(HERE)}`" for l in logs]]
    (run_root / "results.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {run_root / 'results.md'}")


if __name__ == "__main__":
    main()
