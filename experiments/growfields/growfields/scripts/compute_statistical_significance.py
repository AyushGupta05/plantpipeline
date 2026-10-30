#!/usr/bin/env python3
"""
Wilcoxon signed-rank test comparing GrowFields against the COAP and DPF baselines on the
per-organ metrics of Table 2, which is the significance claim the paper reports.

Takes one `all_metrics.csv` per method and aggregates them exactly as
`create_latex_table_per_organ.py` does, so the organ sequences tested are the same 45 the
table averages. Reports Shapiro-Wilk on the paired differences (non-normality is what
justifies Wilcoxon over a paired t-test) and the one-sided p-value for each comparison.

Usage:
    python scripts/compute_statistical_significance.py \
      --ours <ours_experiment>/all_metrics.csv \
      --coap <coap_experiment>/all_metrics.csv \
      --dpf  <dpf_experiment>/all_metrics.csv
"""

import argparse

import pandas as pd
from scipy.stats import shapiro, wilcoxon

# Organs are kept when they have at least this many evaluated timepoints, matching
# `--min_timepoints_per_organ 2` in the table scripts (the first frame has no prediction).
MIN_TIMEPOINTS = 2


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """Aggregate per-organ metrics, mirroring create_latex_row_per_organ.py.

    Each experiment has multiple runs (different random seeds). To stay consistent
    with the paper's LaTeX table pipeline, the aggregation order is:
      1. Filter organs by min timepoints per (sequence, organ_id, run).
      2. Average over timepoints for each (sequence, organ_id, run).
      3. Average over runs for each (sequence, organ_id).
    (For our balanced panels — every timepoint has all runs — step 2/3 order is
    mathematically interchangeable, but we follow the paper's order explicitly.)
    """
    df = pd.read_csv(csv_path)

    # Drop rows where all test metrics are NaN (first timepoint of each organ has no results)
    df = df.dropna(subset=["cd_test", "epe"], how="all")

    # Step 1: filter to organs with >= MIN_TIMEPOINTS timepoints *per run*
    # (matching filter_organs_by_min_timepoints / --min_timepoints_per_organ).
    counts = df.groupby(["sequence", "organ_id", "run"]).size()
    valid = counts[counts >= MIN_TIMEPOINTS].reset_index()[
        ["sequence", "organ_id", "run"]
    ]
    df = df.merge(valid, on=["sequence", "organ_id", "run"], how="inner")

    # Step 2: mean over timepoints for each (sequence, organ_id, run)
    per_run = (
        df.groupby(["sequence", "organ_id", "run"])[["cd_test", "epe"]]
        .mean()
        .reset_index()
    )

    # Step 3: mean over runs for each (sequence, organ_id)
    agg = (
        per_run.groupby(["sequence", "organ_id"])[["cd_test", "epe"]]
        .mean()
        .reset_index()
    )
    return agg


def format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def run_test(name: str, ours_vals: pd.Series, baseline_vals: pd.Series) -> float:
    # Drop organs where either method has no data for this metric
    mask = ours_vals.notna() & baseline_vals.notna()
    o = ours_vals[mask]
    b = baseline_vals[mask]
    n = mask.sum()

    # Shapiro-Wilk on differences to confirm non-normality (justifies Wilcoxon over paired t-test)
    diffs = o.values - b.values
    sw_stat, sw_p = shapiro(diffs)
    normality = "normal" if sw_p >= 0.05 else "non-normal"
    wilcoxon_ok = "→ Wilcoxon justified" if sw_p < 0.05 else "→ paired t-test also valid"
    print(f"  {name} (N={n}): Shapiro-Wilk on differences: W={sw_stat:.3f}, p={sw_p:.3f} [{normality}] {wilcoxon_ok}")

    stat, p = wilcoxon(o, b, alternative="less")
    sig = "✓ significant" if p < 0.05 else "✗ NOT significant"
    print(f"  {name} (N={n}): Wilcoxon stat={stat:.1f}, {format_p(p)}  [{sig}]")
    print(f"    Ours     mean={o.mean():.4f}  std={o.std():.4f}")
    print(f"    Baseline mean={b.mean():.4f}  std={b.std():.4f}")
    return p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank test on per-organ metrics (Table 2)"
    )
    parser.add_argument(
        "--ours", required=True, help="all_metrics.csv for GrowFields"
    )
    parser.add_argument(
        "--coap", required=True, help="all_metrics.csv for the COAP baseline"
    )
    parser.add_argument(
        "--dpf", required=True, help="all_metrics.csv for the DPF per-organ baseline"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ours = load_and_aggregate(args.ours)
    coap = load_and_aggregate(args.coap)
    dpf  = load_and_aggregate(args.dpf)

    # Find intersection of organ sequences across all three methods
    keys = ["sequence", "organ_id"]
    ours_keys = set(map(tuple, ours[keys].values.tolist()))
    coap_keys = set(map(tuple, coap[keys].values.tolist()))
    dpf_keys  = set(map(tuple, dpf[keys].values.tolist()))
    common = ours_keys & coap_keys & dpf_keys

    print(f"Organ sequences per method — Ours: {len(ours_keys)}, COAP: {len(coap_keys)}, DPF: {len(dpf_keys)}")
    print(f"Matched (intersection): N = {len(common)}\n")

    # Align on common organs
    def align(df):
        return (
            df.set_index(keys)
            .loc[list(common)]
            .sort_index()
        )

    ours_a = align(ours)
    coap_a = align(coap)
    dpf_a  = align(dpf)

    print("=== Wilcoxon signed-rank test (one-sided: Ours < Baseline) ===\n")

    print("Ours vs COAP:")
    p_cd_coap  = run_test("CD ", ours_a["cd_test"], coap_a["cd_test"])
    p_epe_coap = run_test("EPE", ours_a["epe"],     coap_a["epe"])

    print("\nOurs vs DPF:")
    p_cd_dpf  = run_test("CD ", ours_a["cd_test"], dpf_a["cd_test"])
    p_epe_dpf = run_test("EPE", ours_a["epe"],     dpf_a["epe"])

    print("\n=== Summary ===")
    print(f"  Ours vs COAP CD:  {format_p(p_cd_coap)}")
    print(f"  Ours vs COAP EPE: {format_p(p_epe_coap)}")
    print(f"  Ours vs DPF  CD:  {format_p(p_cd_dpf)}")
    print(f"  Ours vs DPF  EPE: {format_p(p_epe_dpf)}")


if __name__ == "__main__":
    main()
