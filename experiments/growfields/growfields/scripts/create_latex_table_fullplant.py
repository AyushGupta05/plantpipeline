#!/usr/bin/env python3
"""
Create a full-plant LaTeX table from fullplant_metrics.csv.

Reads the CSV written by `compute_metrics_fullplant.py` and emits a complete table
-- header, one aggregated row per method, and footer. `--header_only` emits just
the header. The per-organ counterpart is `create_latex_table_per_organ.py`;
both are normally invoked through the `compute_metrics_and_latex_table_*.sh`
wrappers rather than directly.

Default behavior: Aggregates by sequence (sequence-level means) which is appropriate for papers.

Usage:
    # One method's row (default: aggregate by sequence)
    python scripts/create_latex_table_fullplant.py --csv path/to/fullplant_metrics.csv --label "GrowFields (Ours)"

    # Generate just the table header
    python scripts/create_latex_table_fullplant.py --header_only

    # Generate complete table with header, row, and footer
    python scripts/create_latex_table_fullplant.py --csv path/to/fullplant_metrics.csv --label "My Method" --full_table

    # Use timepoint-level aggregation instead (smaller error bars, not recommended)
    python scripts/create_latex_table_fullplant.py --csv path/to/fullplant_metrics.csv --label "My Method" --by_timepoint

    # Use std instead of SEM
    python scripts/create_latex_table_fullplant.py --csv path/to/fullplant_metrics.csv --label "My Method" --format mean_std

    # Save to file
    python scripts/create_latex_table_fullplant.py --csv path/to/fullplant_metrics.csv --label "My Method" --output table.tex
"""

import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd


def cd_test_column(df):
    """Return the CD column the tables report, or raise if it has no values.

    Only `cd_test` is held out; the all-points `cd` would silently inflate a table
    entry when a run was trained without a point split.
    """
    if "cd_test" in df.columns and df["cd_test"].notna().any():
        return "cd_test"
    raise ValueError(
        "No cd_test values in the metrics CSV: the run had no held-out test points. "
        "Train with --points_split_dir to get a table-comparable CD."
    )


def aggregate_metrics(df, metrics=None):
    """
    Aggregate metrics across all sequences and runs.

    Args:
        df: DataFrame with fullplant metrics
        metrics: List of metric names to aggregate (None = use defaults)

    Returns:
        Dictionary with aggregated statistics
    """
    if metrics is None:
        metrics = [
            "cd_test",
            "epe",
            "accs",
            "accr",
            "training_time_seconds",
            "total_model_size_mb",
        ]

    stats = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        values = df[metric].dropna()

        if len(values) == 0:
            continue

        # Compute statistics
        stats[f"{metric}_mean"] = values.mean()
        stats[f"{metric}_std"] = values.std()
        stats[f"{metric}_sem"] = values.sem()  # Standard error of mean
        stats[f"{metric}_median"] = values.median()
        stats[f"{metric}_min"] = values.min()
        stats[f"{metric}_max"] = values.max()
        stats[f"{metric}_n"] = len(values)

    # Add sequence-level statistics
    stats["n_sequences"] = df["sequence"].nunique()
    stats["n_runs"] = df["run"].nunique()
    stats["n_timepoints_total"] = len(df)

    return stats


def aggregate_by_sequence(df, metrics=None):
    """
    Aggregate metrics by sequence (across runs), then compute overall stats.

    This gives sequence-level means ± SEM across sequences.

    Args:
        df: DataFrame with fullplant metrics
        metrics: List of metric names to aggregate

    Returns:
        Dictionary with aggregated statistics
    """
    if metrics is None:
        # CD must be cd_test; NC falls back to all points when nc_test is absent
        metrics = []
        metrics.append(cd_test_column(df))

        if "nc_test" in df.columns and df["nc_test"].notna().any():
            metrics.append("nc_test")
        elif "nc" in df.columns and df["nc"].notna().any():
            metrics.append("nc")

        metrics.extend(["epe", "accs", "accr"])

    # First aggregate by sequence and run
    seq_stats = df.groupby(["sequence", "run"])[metrics].mean().reset_index()

    # Then aggregate across runs for each sequence
    seq_means = seq_stats.groupby("sequence")[metrics].mean()

    stats = {}
    for metric in metrics:
        if metric not in seq_means.columns:
            continue

        values = seq_means[metric].dropna()

        if len(values) == 0:
            continue

        stats[f"{metric}_mean"] = values.mean()
        stats[f"{metric}_std"] = values.std()
        stats[f"{metric}_sem"] = values.sem()
        stats[f"{metric}_n"] = len(values)

    # Handle training time and model size separately
    # (these are constant per sequence/run, so we just take the first value per sequence/run)
    if "training_time_seconds" in df.columns:
        # Get unique value per sequence/run, then average across sequences
        time_per_seq_run = (
            df.groupby(["sequence", "run"])["training_time_seconds"]
            .first()
            .reset_index()
        )
        time_per_seq = time_per_seq_run.groupby("sequence")[
            "training_time_seconds"
        ].mean()
        stats["training_time_seconds_mean"] = time_per_seq.mean()
        stats["training_time_seconds_std"] = time_per_seq.std()
        stats["training_time_seconds_sem"] = time_per_seq.sem()

    if "total_model_size_mb" in df.columns:
        # Get unique value per sequence/run, then average across sequences
        size_per_seq_run = (
            df.groupby(["sequence", "run"])["total_model_size_mb"].first().reset_index()
        )
        size_per_seq = size_per_seq_run.groupby("sequence")[
            "total_model_size_mb"
        ].mean()
        stats["total_model_size_mb_mean"] = size_per_seq.mean()
        stats["total_model_size_mb_std"] = size_per_seq.std()
        stats["total_model_size_mb_sem"] = size_per_seq.sem()

    return stats


def compute_per_sequence_metrics(df):
    """
    Compute per-sequence metrics: mCD, mEPE, lEPE.

    For each (sequence, run):
      - mCD: mean CD across timepoints (cd_test if available, else cd)
      - mEPE: mean EPE across timepoints
      - lEPE: EPE at the last (max) timepoint

    Then averages across runs per sequence to get mean ± std.

    Returns:
        (per_seq_stats, has_multiple_runs) where per_seq_stats is an OrderedDict
        keyed by sequence name, each value is a dict with keys:
        mcd_mean, mcd_std, mepe_mean, mepe_std, lepe_mean, lepe_std
    """
    # Determine CD column
    cd_col = cd_test_column(df)

    has_multiple_runs = df["run"].nunique() > 1

    # Compute per (sequence, run) metrics
    records = []
    for (seq, run), group in df.groupby(["sequence", "run"]):
        rec = {"sequence": seq, "run": run}

        # mCD: mean CD across timepoints
        if cd_col and cd_col in group.columns:
            rec["mcd"] = group[cd_col].dropna().mean()
        else:
            rec["mcd"] = np.nan

        # mEPE: mean EPE across timepoints
        if "epe" in group.columns:
            rec["mepe"] = group["epe"].dropna().mean()
        else:
            rec["mepe"] = np.nan

        # lEPE: EPE at the last timepoint
        if "epe" in group.columns:
            last_tp = group["timepoint"].max()
            last_row = group[group["timepoint"] == last_tp]
            rec["lepe"] = (
                last_row["epe"].dropna().values[0]
                if len(last_row["epe"].dropna()) > 0
                else np.nan
            )
        else:
            rec["lepe"] = np.nan

        records.append(rec)

    run_df = pd.DataFrame(records)

    # Aggregate across runs per sequence
    per_seq_stats = OrderedDict()
    sequences = sorted(run_df["sequence"].unique())

    for seq in sequences:
        seq_data = run_df[run_df["sequence"] == seq]
        stats = {}
        for metric in ["mcd", "mepe", "lepe"]:
            vals = seq_data[metric].dropna()
            stats[f"{metric}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            stats[f"{metric}_std"] = vals.std() if len(vals) > 1 else 0.0
        per_seq_stats[seq] = stats

    return per_seq_stats, has_multiple_runs


def compute_mean_column_run_std(df):
    """
    Standard deviation of the Mean column across runs.

    Averages the sequences within each run first, so the result is the run-to-run
    repeatability of the headline number. This is deliberately not the spread between
    plants, which is what `aggregate_by_sequence` reports and which is far larger.

    Returns:
        Dict with keys mcd_std and mepe_std; a value is None when there is only one run.
    """
    cd_col = cd_test_column(df)

    out = {}
    for key, col in (("mcd", cd_col), ("mepe", "epe")):
        if col is None or col not in df.columns:
            out[f"{key}_std"] = None
            continue
        per_run = df.groupby(["sequence", "run"])[col].mean().groupby("run").mean()
        per_run = per_run.dropna()
        out[f"{key}_std"] = per_run.std() if len(per_run) > 1 else None
    return out


def abbreviate_sequence(seq_name):
    """
    Generate a short abbreviation from a sequence name.

    Pattern: {species}_{condition}_plant{N} -> XxY#
    Species: Ma=maize, So=sorghum, Tb=tobacco, Tm=tomato1/tomato2
    Condition: C=control, H=heat/highlight, S=shade
    """
    species_map = {
        "maize": "Ma",
        "sorghum": "So",
        "tobacco": "Tb",
        "tomato1": "Tm1",
        "tomato2": "Tm2",
    }
    condition_map = {
        "control": "C",
        "heat": "H",
        "highlight": "H",
        "shade": "S",
    }
    parts = seq_name.split("_")
    # Find species (first part, may include number like tomato1)
    species = parts[0]
    # Find condition and plant number
    condition = parts[1]
    plant_num = parts[2].replace("plant", "")

    sp = species_map.get(species, species[:2].capitalize())
    cond = condition_map.get(condition, condition[0].upper())
    return f"{sp}{cond}{plant_num}"


def generate_per_sequence_header(sequences, include_lepe=False):
    """Generate LaTeX table header for per-sequence table."""
    n_seq = len(sequences)
    n_metrics = 3 if include_lepe else 2

    metric_labels = [r"CD $\downarrow$", r"EPE $\downarrow$"]
    if include_lepe:
        metric_labels.append(r"$\text{EPE}_T$ $\downarrow$")

    # Build abbreviation mapping
    abbrevs = {seq: abbreviate_sequence(seq) for seq in sequences}

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")

    # Caption with metric definitions and sequence abbreviation legend
    caption = r"\caption{Full plant reconstruction results. "
    caption += r"We report mean temporal metrics per sequence. "
    caption += r"CD: Chamfer Distance ($\text{mm}^2$), EPE: End-Point Error on leaf tips ($\text{mm}$)"
    if include_lepe:
        caption += r", $\text{EPE}_T$: last-timepoint EPE ($\text{mm}$)"
    # caption += ". Sequences: "
    # legend_parts = []
    # for seq in sequences:
    #     seq_escaped = seq.replace("_", r"\_")
    #     legend_parts.append(
    #         r"\texttt{" + abbrevs[seq] + r"}: \texttt{" + seq_escaped + "}"
    #     )
    # caption += ", ".join(legend_parts)
    caption += ".}"
    lines.append(caption)
    lines.append(r"\label{tab:per_sequence_metrics}")

    # Column spec: l for method, then n_metrics c's per group
    col_group = "c" * n_metrics
    col_spec = "l" + (" " + col_group) * (n_seq + 1)
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # First header row: multicolumn for each sequence abbreviation + Avg
    header_parts = [""]
    mc = str(n_metrics)
    for seq in sequences:
        header_parts.append(r"\multicolumn{" + mc + r"}{c}{" + abbrevs[seq] + "}")
    header_parts.append(r"\multicolumn{" + mc + r"}{c}{\textit{Mean}}")
    lines.append(" & ".join(header_parts) + r" \\")

    # Cmidrules under each sequence group
    cmidrules = []
    for i in range(n_seq + 1):  # sequences + avg
        start = 2 + i * n_metrics
        end = start + n_metrics - 1
        cmidrules.append(r"\cmidrule(lr){" + str(start) + "-" + str(end) + "}")
    lines.append(" ".join(cmidrules))

    # Second header row: metric sub-columns repeated
    sub_parts = ["Method"]
    for _ in range(n_seq + 1):
        sub_parts.extend(metric_labels)
    lines.append(" & ".join(sub_parts) + r" \\")
    lines.append(r"\midrule")

    return "\n".join(lines)


def generate_per_sequence_footer():
    """Generate LaTeX table footer for per-sequence table."""
    lines = []
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # close \resizebox
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def format_per_sequence_row(
    label, per_seq_stats, has_multiple_runs, include_lepe=False, mean_run_std=None
):
    """
    Format a per-sequence LaTeX table row.

    Args:
        label: Method name
        per_seq_stats: OrderedDict from compute_per_sequence_metrics
        has_multiple_runs: Whether to show ± std
        include_lepe: Whether to include the EPE_T column
    """
    # Always the standard deviation over runs: --format is not plumbed in here, since a
    # per-cell SEM over a handful of runs is not a figure worth reporting.
    parts = [label]
    metrics = ["mcd", "mepe"]
    if include_lepe:
        metrics.append("lepe")

    # Per-sequence columns
    for seq in per_seq_stats:
        stats = per_seq_stats[seq]
        for m in metrics:
            val = stats[f"{m}_mean"]
            if np.isnan(val):
                parts.append("--")
            elif has_multiple_runs and stats[f"{m}_std"] > 0:
                parts.append(f"{val:.2f}\\std{{{stats[f'{m}_std']:.2f}}}")
            else:
                parts.append(f"{val:.2f}")

    # Avg column: average of per-sequence means
    for m in metrics:
        seq_means = [
            per_seq_stats[s][f"{m}_mean"]
            for s in per_seq_stats
            if not np.isnan(per_seq_stats[s][f"{m}_mean"])
        ]
        if len(seq_means) == 0:
            parts.append("--")
            continue
        val = np.mean(seq_means)
        # The mean stays the average of the per-sequence means; only the +- is new.
        run_std = (mean_run_std or {}).get(f"{m}_std")
        if has_multiple_runs and run_std:
            parts.append(f"{val:.2f}\\std{{{run_std:.2f}}}")
        else:
            parts.append(f"{val:.2f}")

    return " & ".join(parts) + " \\\\"


def format_latex_row(label, stats, format_spec="mean_sem"):
    """
    Format statistics as a LaTeX table row.

    Args:
        label: Label for the row (method name)
        stats: Dictionary of aggregated statistics
        format_spec: How to format values:
            - 'mean_sem': mean ± SEM
            - 'mean_std': mean ± std
            - 'mean_only': just mean

    Returns:
        String with LaTeX table row
    """
    parts = [label]

    # CD (test) - fall back to cd if cd_test not available
    cd_key = "cd_test" if "cd_test_mean" in stats else "cd"
    if f"{cd_key}_mean" in stats:
        cd_mean = stats[f"{cd_key}_mean"]
        if format_spec == "mean_sem" and f"{cd_key}_sem" in stats:
            cd_err = stats[f"{cd_key}_sem"]
            parts.append(f"{cd_mean:.4f}\\std{{{cd_err:.4f}}}")
        elif format_spec == "mean_std" and f"{cd_key}_std" in stats:
            cd_std = stats[f"{cd_key}_std"]
            parts.append(f"{cd_mean:.4f}\\std{{{cd_std:.4f}}}")
        else:
            parts.append(f"{cd_mean:.4f}")
    else:
        parts.append("--")

    # NC (test) - fall back to nc if nc_test not available
    nc_key = "nc_test" if "nc_test_mean" in stats else "nc"
    if f"{nc_key}_mean" in stats:
        nc_mean = stats[f"{nc_key}_mean"]
        if format_spec == "mean_sem" and f"{nc_key}_sem" in stats:
            nc_err = stats[f"{nc_key}_sem"]
            parts.append(f"{nc_mean:.4f}\\std{{{nc_err:.4f}}}")
        elif format_spec == "mean_std" and f"{nc_key}_std" in stats:
            nc_std = stats[f"{nc_key}_std"]
            parts.append(f"{nc_mean:.4f}\\std{{{nc_std:.4f}}}")
        else:
            parts.append(f"{nc_mean:.4f}")
    else:
        parts.append("--")

    # EPE
    if "epe_mean" in stats:
        epe_mean = stats["epe_mean"]
        if format_spec == "mean_sem" and "epe_sem" in stats:
            epe_err = stats["epe_sem"]
            parts.append(f"{epe_mean:.4f}\\std{{{epe_err:.4f}}}")
        elif format_spec == "mean_std" and "epe_std" in stats:
            epe_std = stats["epe_std"]
            parts.append(f"{epe_mean:.4f}\\std{{{epe_std:.4f}}}")
        else:
            parts.append(f"{epe_mean:.4f}")
    else:
        parts.append("--")

    # AccS
    if "accs_mean" in stats:
        accs_mean = stats["accs_mean"]
        if format_spec == "mean_sem" and "accs_sem" in stats:
            accs_err = stats["accs_sem"]
            parts.append(f"{accs_mean:.2f}\\std{{{accs_err:.2f}}}")
        elif format_spec == "mean_std" and "accs_std" in stats:
            accs_std = stats["accs_std"]
            parts.append(f"{accs_mean:.2f}\\std{{{accs_std:.2f}}}")
        else:
            parts.append(f"{accs_mean:.2f}")
    else:
        parts.append("--")

    # AccR
    if "accr_mean" in stats:
        accr_mean = stats["accr_mean"]
        if format_spec == "mean_sem" and "accr_sem" in stats:
            accr_err = stats["accr_sem"]
            parts.append(f"{accr_mean:.2f}\\std{{{accr_err:.2f}}}")
        elif format_spec == "mean_std" and "accr_std" in stats:
            accr_std = stats["accr_std"]
            parts.append(f"{accr_mean:.2f}\\std{{{accr_std:.2f}}}")
        else:
            parts.append(f"{accr_mean:.2f}")
    else:
        parts.append("--")

    # Training time
    if "training_time_seconds_mean" in stats:
        time_min = stats["training_time_seconds_mean"] / 60
        parts.append(f"{time_min:.1f}")
    else:
        parts.append("--")

    # Model size
    if "total_model_size_mb_mean" in stats:
        size_mb = stats["total_model_size_mb_mean"]
        parts.append(f"{size_mb:.2f}")
    else:
        parts.append("--")

    return " & ".join(parts) + " \\\\"


def generate_latex_header():
    """Generate LaTeX table header."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Full plant reconstruction metrics. We report the mean over sequences of the mean temporal metrics, with standard error of the mean (SEM). CD: Chamfer Distance, NC: Normal Consistency, EPE: End Point Error, AccS/AccR: Strict/Relaxed Accuracy.}"
    )
    lines.append(r"\label{tab:fullplant_metrics}")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & CD $\downarrow$ & NC $\uparrow$ & EPE $\downarrow$ & AccS $\uparrow$ & AccR $\uparrow$ & Time (min) & Size (MB) \\"
    )
    lines.append(r"\midrule")
    return "\n".join(lines)


def generate_latex_footer():
    """Generate LaTeX table footer."""
    lines = []
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def print_summary_stats(stats):
    """Print summary statistics in a readable format."""
    print("\n" + "=" * 80)
    print("AGGREGATED STATISTICS")
    print("=" * 80)

    if "n_sequences" in stats:
        print(f"\nDataset info:")
        print(f"  Sequences: {stats.get('n_sequences', 'N/A')}")
        print(f"  Runs: {stats.get('n_runs', 'N/A')}")
        print(f"  Total timepoints: {stats.get('n_timepoints_total', 'N/A')}")

    print(f"\nMetrics:")

    # CD - check for cd_test first, then fall back to cd
    cd_key = "cd_test" if "cd_test_mean" in stats else "cd"
    if f"{cd_key}_mean" in stats:
        label = "CD (test)" if cd_key == "cd_test" else "CD"
        print(
            f"  {label}:  {stats[f'{cd_key}_mean']:.6f} ± {stats.get(f'{cd_key}_sem', stats.get(f'{cd_key}_std', 0)):.6f}"
        )

    # NC - check for nc_test first, then fall back to nc
    nc_key = "nc_test" if "nc_test_mean" in stats else "nc"
    if f"{nc_key}_mean" in stats:
        label = "NC (test)" if nc_key == "nc_test" else "NC"
        print(
            f"  {label}:  {stats[f'{nc_key}_mean']:.6f} ± {stats.get(f'{nc_key}_sem', stats.get(f'{nc_key}_std', 0)):.6f}"
        )

    if "epe_mean" in stats:
        print(
            f"  EPE:        {stats['epe_mean']:.4f} ± {stats.get('epe_sem', stats.get('epe_std', 0)):.4f}"
        )

    if "accs_mean" in stats:
        print(
            f"  AccS:       {stats['accs_mean']:.2f}% ± {stats.get('accs_sem', stats.get('accs_std', 0)):.2f}%"
        )

    if "accr_mean" in stats:
        print(
            f"  AccR:       {stats['accr_mean']:.2f}% ± {stats.get('accr_sem', stats.get('accr_std', 0)):.2f}%"
        )

    if "training_time_seconds_mean" in stats:
        print(f"  Training:   {stats['training_time_seconds_mean']/60:.1f} min")

    if "total_model_size_mb_mean" in stats:
        print(f"  Model size: {stats['total_model_size_mb_mean']:.2f} MB")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create LaTeX table row from fullplant_metrics.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to fullplant_metrics.csv file (not required with --header_only)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Method",
        help="Label for the LaTeX row (method name)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mean_sem", "mean_std", "mean_only"],
        default="mean_sem",
        # Only the aggregate table reads this; --per_sequence always prints the
        # standard deviation over runs, whatever is passed here.
        help="Format for the aggregate table: mean_sem (default), mean_std, or "
        "mean_only. Ignored with --per_sequence, which always reports std over runs",
    )
    parser.add_argument(
        "--by_sequence",
        action="store_true",
        default=True,
        help="Aggregate by sequence first (gives sequence-level statistics) [default: True]",
    )
    parser.add_argument(
        "--by_timepoint",
        action="store_true",
        help="Aggregate across all timepoints instead of by sequence",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for LaTeX row (default: print to stdout)",
    )
    parser.add_argument(
        "--header_only",
        action="store_true",
        help="Print only the LaTeX table header (no data row)",
    )
    parser.add_argument(
        "--full_table",
        action="store_true",
        help="Print complete LaTeX table structure (header + row + footer)",
    )
    parser.add_argument(
        "--finetune_stage",
        type=str,
        choices=["pre", "post"],
        default=None,
        help="Filter by finetune stage (pre or post). Only applies if finetune_stage column exists.",
    )
    parser.add_argument(
        "--per_sequence",
        action="store_true",
        help="Generate per-sequence table with CD_bar, EPE_bar columns per sequence + Avg",
    )
    parser.add_argument(
        "--no_std",
        action="store_true",
        help="Read only in --per_sequence mode, dropping the ± even when multiple runs "
        "are available (the mean still averages over runs)",
    )
    parser.add_argument(
        "--skip_sequences",
        type=str,
        default=None,
        help="Comma-separated sequence names to exclude from the table (e.g. 'sorghum_highlight_plant2').",
    )
    parser.add_argument(
        "--include_lepe",
        action="store_true",
        help="Include EPE_T (last-timepoint EPE) column in per-sequence table",
    )

    args = parser.parse_args()

    # Handle header_only mode (no CSV needed)
    if args.header_only:
        header = generate_latex_header()
        print("\n" + "=" * 80)
        print("LATEX TABLE HEADER")
        print("=" * 80)
        print("\n" + header + "\n")
        if args.output:
            with open(args.output, "w") as f:
                f.write(header + "\n")
            print(f"Header saved to: {args.output}")
        return

    # Validate that CSV is provided
    if args.csv is None:
        parser.error("--csv is required unless using --header_only")

    # Load CSV
    print(f"Loading metrics from: {args.csv}")
    df = pd.read_csv(args.csv)

    print(f"Loaded {len(df)} rows")
    print(f"  Sequences: {df['sequence'].nunique()}")
    print(f"  Runs: {df['run'].nunique()}")

    # Exclude requested sequences (e.g. ones not run for this experiment)
    if args.skip_sequences:
        skip = [s.strip() for s in args.skip_sequences.split(",") if s.strip()]
        df = df[~df["sequence"].isin(skip)]
        print(f"  After skip_sequences filter (excluded {skip}): {len(df)} rows")

    # Filter by finetune stage if specified
    if args.finetune_stage is not None:
        if "finetune_stage" in df.columns:
            df = df[df["finetune_stage"] == args.finetune_stage]
            print(
                f"  Filtered to finetune_stage='{args.finetune_stage}': {len(df)} rows"
            )
        else:
            print(
                f"  Warning: finetune_stage column not found, ignoring --finetune_stage filter"
            )

    # Per-sequence table mode
    if args.per_sequence:
        include_lepe = args.include_lepe
        print(
            "\nComputing per-sequence metrics (CD_bar, EPE_bar"
            + (", EPE_T" if include_lepe else "")
            + ")..."
        )
        per_seq_stats, has_multiple_runs = compute_per_sequence_metrics(df)
        mean_run_std = compute_mean_column_run_std(df)
        if args.no_std:
            has_multiple_runs = False

        # Print summary
        print(f"\n{'='*80}")
        print("PER-SEQUENCE METRICS")
        print(f"{'='*80}")
        for seq, stats in per_seq_stats.items():
            print(f"\n  {seq}:")
            print(
                f"    CD_mean:  {stats['mcd_mean']:.3f}"
                + (f" ± {stats['mcd_std']:.3f}" if has_multiple_runs else "")
            )
            print(
                f"    EPE_mean: {stats['mepe_mean']:.3f}"
                + (f" ± {stats['mepe_std']:.3f}" if has_multiple_runs else "")
            )
            if include_lepe:
                print(
                    f"    EPE_T:    {stats['lepe_mean']:.3f}"
                    + (f" ± {stats['lepe_std']:.3f}" if has_multiple_runs else "")
                )
        print()

        sequences = list(per_seq_stats.keys())
        latex_row = format_per_sequence_row(
            args.label,
            per_seq_stats,
            has_multiple_runs,
            include_lepe,
            mean_run_std=mean_run_std,
        )

        if args.full_table:
            header = generate_per_sequence_header(sequences, include_lepe)
            footer = generate_per_sequence_footer()
            full_table = f"{header}\n{latex_row}\n{footer}"

            print("=" * 80)
            print("COMPLETE LATEX TABLE (per-sequence)")
            print("=" * 80)
            print("\n" + full_table + "\n")

            if args.output:
                with open(args.output, "w") as f:
                    f.write(full_table + "\n")
                print(f"Table saved to: {args.output}")
        else:
            print("=" * 80)
            print("LATEX TABLE ROW (per-sequence)")
            print("=" * 80)
            print("\nCopy this row into your LaTeX table:")
            print()
            print(latex_row)
            print()

            if args.output:
                with open(args.output, "w") as f:
                    f.write(latex_row + "\n")
                print(f"LaTeX row saved to: {args.output}")

            print("\nTo see the complete table, run with --full_table")
        return

    # Determine aggregation method (--by_timepoint overrides default --by_sequence)
    use_sequence_aggregation = args.by_sequence and not args.by_timepoint

    # Aggregate metrics
    if use_sequence_aggregation:
        print("\nAggregating by sequence (computing sequence-level means)...")
        stats = aggregate_by_sequence(df)
    else:
        print("\nAggregating across all timepoints...")
        stats = aggregate_metrics(df)

    # Print summary
    print_summary_stats(stats)

    # Generate LaTeX row
    latex_row = format_latex_row(args.label, stats, format_spec=args.format)

    # Output based on mode
    if args.full_table:
        # Full table with header, row, and footer
        header = generate_latex_header()
        footer = generate_latex_footer()
        full_table = f"{header}\n{latex_row}\n{footer}"

        print("=" * 80)
        print("COMPLETE LATEX TABLE")
        print("=" * 80)
        print("\n" + full_table + "\n")

        if args.output:
            with open(args.output, "w") as f:
                f.write(full_table + "\n")
            print(f"Table saved to: {args.output}")
    else:
        # Just the row (default)
        print("=" * 80)
        print("LATEX TABLE ROW")
        print("=" * 80)
        print("\nCopy this row into your LaTeX table:")
        print()
        print(latex_row)
        print()

        if args.output:
            with open(args.output, "w") as f:
                f.write(latex_row + "\n")
            print(f"LaTeX row saved to: {args.output}")

        print(
            "\nColumn order: Label & CD (test) & NC (test) & EPE & AccS & AccR & Time (min) & Size (MB)"
        )
        print("\nTo see the table header, run with --header_only")
        print("To see the complete table, run with --full_table")


if __name__ == "__main__":
    main()
