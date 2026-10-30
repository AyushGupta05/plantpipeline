"""Markdown table of every run under runs/: CD / EPE (full plant and per organ) on observed and held-out frames,
monotonicity violation rate, negative growth mass and length MAE from monogrow's metric outputs.
  python collect.py [--split test|val] [--pattern substring]"""
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--pattern", default="")
a = ap.parse_args()
hdr = "| run | frames | full CD | full EPE | organ CD | organ EPE | n_tips | viol. rate (GT) | neg. mass mm (GT) | length MAE mm obs / held-out | MAE +1/+2/+3 d | fit s |"
print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
for d in sorted(Path(__file__).parent.glob(f"runs/{a.split}_*")):
    if a.pattern not in d.name or not (d / "monogrow_summary.json").exists():
        continue
    g = json.load(open(d / "growth_summary.json"))
    fit_s = sum(json.load(open(r))["aggregate_stats"]["training_time_seconds"] for r in d.glob("sequences/*/results.json"))
    variant = d.name.split("_")[1]
    for mode in ("", f"_{variant}" if variant != "headline" else None):
        if mode is None or not (d / f"monogrow_summary{mode}.json").exists():
            continue
        m = json.load(open(d / f"monogrow_summary{mode}.json"))
        f, o = m["full"]["Mean"], m["organ"]["Mean"]
        growth = (f"{g['violation_rate_pred']:.3f} ({g['violation_rate_gt']:.3f}) | {g['neg_growth_mass_mm_pred']:.1f} ({g['neg_growth_mass_mm_gt']:.1f}) | "
                  f"{g['length_mae_mm_observed']:.1f} / {g['length_mae_mm_heldout']:.1f} | " + "/".join(f"{g[f'length_mae_mm_horizon+{h}']:.0f}" for h in (1, 2, 3)))
        print(f"| {d.name[len(a.split) + 1:]} | {'observed' if not mode else 'held-out'} | {f[0]:.2f} | {f[1]:.2f} | {o[0]:.2f} | {o[1]:.2f} | {m['n_tips_full']}/{m['n_tips_organ']} | {growth} | {fit_s:.2f} |")
