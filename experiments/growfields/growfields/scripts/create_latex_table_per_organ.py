#!/usr/bin/env python3
"""
Create a per-organ LaTeX table from all_metrics.csv.

Reads the CSV written by `compute_metrics_per_organ.py` and emits a complete
table -- header, one aggregated row per method, and footer. `--header_only` emits
just the header. The full-plant counterpart is `create_latex_table_fullplant.py`;
both are normally invoked through the `compute_metrics_and_latex_table_*.sh`
wrappers rather than directly
at the per-organ level.

Default behavior: Aggregates by sequence (sequence-level means) which is appropriate for papers.

Usage:
    # One method's row
    python scripts/create_latex_table_per_organ.py \
        --csv results/fullplant_per_organ/20260410_123456_fullplant_batch_shared_continuous_test_aligned/all_metrics.csv \
        --label "GrowFields (Ours)"

    # Generate just the table header
    python scripts/create_latex_table_per_organ.py --header_only

    # Generate complete table with header, row, and footer
    python scripts/create_latex_table_per_organ.py \
        --csv all_metrics.csv \
        --label "My Method" \
        --full_table

    # Show breakdown by organ category (leaf vs stem)
    python scripts/create_latex_table_per_organ.py \
        --csv all_metrics.csv \
        --label "My Method" \
        --by_organ_category

    # For interpolation experiments: use all_metrics_interpolation.csv from compute_metrics_per_organ.py --interpolation
    python scripts/create_latex_table_per_organ.py \
        --csv results/interpolation_exp/all_metrics_interpolation.csv \
        --label "Temporal Interpolation" \
        --full_table

    # Filter to organs with at least 3 timepoints (exclude organs with too few observations)
    python scripts/create_latex_table_per_organ.py \
        --csv all_metrics.csv \
        --label "My Method" \
        --full_table \
        --min_timepoints_per_organ 3

    # Process only a single run (faster, useful for quick iteration)
    python scripts/create_latex_table_per_organ.py \
        --csv all_metrics.csv \
        --label "My Method" \
        --run 1

    # Export detailed per-organ metrics to CSV
    python scripts/create_latex_table_per_organ.py \
        --csv all_metrics.csv \
        --export_csv output/tables/per_organ/results.csv

Notes:
    - Default (aggregate) table: organ-weighted.
      1. Average over timepoints for each (sequence, organ, run)
      2. Average over runs for each (sequence, organ)
      3. Mean ± SEM across all organ sequences (n = 45 on the test split), so plants
         with more organs carry more weight
    - With --per_sequence: plant-weighted, which is what the paper reports.
      Organs are averaged within a plant first (equal weight per plant), and the ±
      is the standard deviation across runs. The two give different means --
      0.80 organ-weighted vs 0.87 plant-weighted on the main test run.
    - CSV export computation:
      1. Average over timepoints for each (sequence, organ, run)
      2. Average over runs → shows mean ± std for each (sequence, organ_id) pair
    - CSV export with --run: shows raw metrics per (sequence, organ_id), std columns are NaN
"""

import argparse
import warnings
from collections import OrderedDict
from pathlib import Path

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


def filter_organs_by_min_timepoints(df, min_timepoints):
    """
    Filter out organs that don't have enough timepoints.

    Args:
        df: DataFrame to filter
        min_timepoints: Minimum number of timepoints required per (sequence, organ_id, run)

    Returns:
        Tuple of (filtered dataframe, removed_organs_summary)
    """
    if min_timepoints is None or min_timepoints <= 0:
        return df, None

    # Count timepoints per (sequence, organ_id, run)
    timepoint_counts = (
        df.groupby(["sequence", "organ_id", "run"])
        .size()
        .reset_index(name="n_timepoints")
    )

    # Find organs that meet the threshold
    valid_organs = timepoint_counts[timepoint_counts["n_timepoints"] >= min_timepoints]
    removed_organs = timepoint_counts[timepoint_counts["n_timepoints"] < min_timepoints]

    # Get summary of removed organs (by sequence and organ_id)
    removed_summary = None
    if len(removed_organs) > 0:
        removed_summary = (
            removed_organs.groupby(["sequence", "organ_id"])
            .agg({"n_timepoints": ["min", "max", "mean"], "run": "count"})
            .reset_index()
        )
        removed_summary.columns = [
            "sequence",
            "organ_id",
            "min_timepoints",
            "max_timepoints",
            "avg_timepoints",
            "n_runs",
        ]

    # Merge to filter
    filtered = df.merge(
        valid_organs[["sequence", "organ_id", "run"]],
        on=["sequence", "organ_id", "run"],
        how="inner",
    )

    return filtered, removed_summary

def aggregate_metrics(df, metrics=None):
    """
    Aggregate metrics across all sequences and runs.

    Args:
        df: DataFrame with per-organ metrics
        metrics: List of metric names to aggregate (None = use defaults)

    Returns:
        Dictionary with aggregated statistics
    """
    if metrics is None:
        metrics = ["cd_test", "nc_test", "epe", "accs", "accr"]

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
        df: DataFrame with per-organ metrics
        metrics: List of metric names to aggregate

    Returns:
        Dictionary with aggregated statistics
    """
    if metrics is None:
        metrics = ["cd_test", "nc_test", "epe", "accs", "accr"]

    # First aggregate by sequence and run (across organs and timepoints)
    seq_stats = df.groupby(["sequence", "organ_type", "run"])[metrics].agg(
        ["mean", "count"]
    )

    # Check that each (sequence, organ_type) saw the same number of timepoints in every
    # run. `valid` is a DataFrame, so test it with .to_numpy().all() -- bare all() would
    # iterate the column labels and always be True.
    run_counts = seq_stats.xs("count", axis=1, level=1)
    valid = run_counts.groupby(["sequence", "organ_type"]).nunique().eq(1)
    if not valid.to_numpy().all():
        warnings.warn(
            "Inconsistent number of timepoints per (sequence, organ_type) across runs: "
            "the sequence-level means below average over different numbers of frames.",
            stacklevel=2,
        )

    # Then aggregate across runs for each sequence
    run_means = seq_stats.xs("mean", axis=1, level=1)
    seq_means = (
        run_means.groupby(["sequence", "organ_type"])[metrics].mean().reset_index()
    )

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
    Compute per-sequence metrics from per-organ data: mCD, mEPE, lEPE.

    Aggregation:
      1. For each (sequence, organ_id, run): mean CD/EPE across timepoints, last-timepoint EPE
      2. For each (sequence, run): average across organs (NaN-safe for EPE)
      3. For each sequence: average across runs -> mean ± std

    Returns:
        (per_seq_stats, has_multiple_runs) where per_seq_stats is an OrderedDict
        keyed by sequence name, each value is a dict with keys:
        mcd_mean, mcd_std, mepe_mean, mepe_std, lepe_mean, lepe_std
    """
    # Determine CD column
    cd_col = cd_test_column(df)

    has_multiple_runs = df["run"].nunique() > 1

    # Step 1: For each (sequence, organ_id, run) -> mean metrics across timepoints
    organ_records = []
    for (seq, organ_id, run), group in df.groupby(["sequence", "organ_id", "run"]):
        rec = {"sequence": seq, "organ_id": organ_id, "run": run}

        # mCD: mean CD across timepoints for this organ
        if cd_col and cd_col in group.columns:
            rec["mcd"] = group[cd_col].dropna().mean()
        else:
            rec["mcd"] = np.nan

        # mEPE: mean EPE across timepoints for this organ
        if "epe" in group.columns:
            rec["mepe"] = group["epe"].dropna().mean()
        else:
            rec["mepe"] = np.nan

        # lEPE: EPE at the last timepoint for this organ
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

        organ_records.append(rec)

    organ_df = pd.DataFrame(organ_records)

    # Step 2: For each (sequence, run) -> average across organs (NaN-safe)
    seq_run_records = []
    for (seq, run), group in organ_df.groupby(["sequence", "run"]):
        rec = {"sequence": seq, "run": run}
        for metric in ["mcd", "mepe", "lepe"]:
            vals = group[metric].dropna()
            rec[metric] = vals.mean() if len(vals) > 0 else np.nan
        seq_run_records.append(rec)

    seq_run_df = pd.DataFrame(seq_run_records)

    # Step 3: For each sequence -> average across runs
    per_seq_stats = OrderedDict()
    sequences = sorted(seq_run_df["sequence"].unique())

    for seq in sequences:
        seq_data = seq_run_df[seq_run_df["sequence"] == seq]
        stats = {}
        for metric in ["mcd", "mepe", "lepe"]:
            vals = seq_data[metric].dropna()
            stats[f"{metric}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            stats[f"{metric}_std"] = vals.std() if len(vals) > 1 else 0.0
        per_seq_stats[seq] = stats

    return per_seq_stats, has_multiple_runs

def compute_mean_column_run_std(df):
    """
    Standard deviation of the Mean column across runs, for the per-organ table.

    Organs are averaged within a plant and plants within a run, so the result is the
    run-to-run repeatability of the headline number. Pass the same filtered frame that
    `compute_per_sequence_metrics` received, or the minimum-timepoint filter is missed.

    Returns:
        Dict with keys mcd_std and mepe_std; a value is None when there is only one run.
    """
    cd_col = cd_test_column(df)

    out = {}
    for key, col in (("mcd", cd_col), ("mepe", "epe")):
        if col is None or col not in df.columns:
            out[f"{key}_std"] = None
            continue
        per_run = (
            df.groupby(["sequence", "organ_id", "run"])[col]
            .mean()
            .groupby(["sequence", "run"])
            .mean()
            .groupby("run")
            .mean()
            .dropna()
        )
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
    species = parts[0]
    condition = parts[1]
    plant_num = parts[2].replace("plant", "")

    sp = species_map.get(species, species[:2].capitalize())
    cond = condition_map.get(condition, condition[0].upper())
    return f"{sp}{cond}{plant_num}"

def generate_per_sequence_header(sequences, include_lepe=False):
    """Generate LaTeX table header for per-sequence per-organ table."""
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

    # Caption with metric definitions
    caption = r"\caption{Per-organ reconstruction results. "
    caption += r"We report mean temporal metrics per sequence, averaged across organs. "
    caption += r"CD: Chamfer Distance ($\text{mm}^2$), EPE: End-Point Error on leaf tips ($\text{mm}$)"
    if include_lepe:
        caption += r", $\text{EPE}_T$: last-timepoint EPE ($\text{mm}$)"
    caption += ".}"
    lines.append(caption)
    lines.append(r"\label{tab:per_organ_per_sequence_metrics}")

    # Column spec: l for method, then n_metrics c's per group
    col_group = "c" * n_metrics
    col_spec = "l" + (" " + col_group) * (n_seq + 1)
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # First header row: multicolumn for each sequence abbreviation + Mean
    header_parts = [""]
    mc = str(n_metrics)
    for seq in sequences:
        header_parts.append(r"\multicolumn{" + mc + r"}{c}{" + abbrevs[seq] + "}")
    header_parts.append(r"\multicolumn{" + mc + r"}{c}{\textit{Mean}}")
    lines.append(" & ".join(header_parts) + r" \\")

    # Cmidrules under each sequence group
    cmidrules = []
    for i in range(n_seq + 1):  # sequences + Mean
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
    """Generate LaTeX table footer for per-sequence per-organ table."""
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
    Format a per-sequence LaTeX table row for per-organ metrics.

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

    # Mean column: average of per-sequence means, +- its spread across runs
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

    # CD (test)
    if "cd_test_mean" in stats:
        cd_mean = stats["cd_test_mean"]
        if format_spec == "mean_sem" and "cd_test_sem" in stats:
            cd_err = stats["cd_test_sem"]
            parts.append(f"{cd_mean:.4f}\\std{{{cd_err:.4f}}}")
        elif format_spec == "mean_std" and "cd_test_std" in stats:
            cd_std = stats["cd_test_std"]
            parts.append(f"{cd_mean:.4f}\\std{{{cd_std:.4f}}}")
        else:
            parts.append(f"{cd_mean:.4f}")
    else:
        parts.append("--")

    # NC (test)
    if "nc_test_mean" in stats:
        nc_mean = stats["nc_test_mean"]
        if format_spec == "mean_sem" and "nc_test_sem" in stats:
            nc_err = stats["nc_test_sem"]
            parts.append(f"{nc_mean:.4f}\\std{{{nc_err:.4f}}}")
        elif format_spec == "mean_std" and "nc_test_std" in stats:
            nc_std = stats["nc_test_std"]
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
        r"\caption{Per-organ reconstruction metrics. We report the mean over sequences of the mean temporal metrics, with standard error of the mean (SEM). CD: Chamfer Distance, NC: Normal Consistency, EPE: End Point Error, AccS/AccR: Strict/Relaxed Accuracy.}"
    )
    lines.append(r"\label{tab:per_organ_metrics}")
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

def export_detailed_csv(df, output_path, metrics=None):
    """
    Export detailed organ-level statistics to CSV.

    For each (sequence, organ_id):
    - If multiple runs: show mean ± std across runs
    - If single run: show raw metrics (std columns will be NaN)

    Args:
        df: Filtered DataFrame with per-organ metrics
        output_path: Path to save CSV file
        metrics: List of metrics to include
    """
    if metrics is None:
        metrics = ["cd_test", "nc_test", "epe", "accs", "accr"]

    # Aggregate by (sequence, organ_id, run) to get mean across timepoints
    organ_stats = (
        df.groupby(["sequence", "organ_id", "organ_type", "run"])[metrics]
        .mean()
        .reset_index()
    )

    # Aggregate across runs for each (sequence, organ_id)
    organ_summary = (
        organ_stats.groupby(["sequence", "organ_id", "organ_type"])
        .agg({metric: ["mean", "std"] for metric in metrics})
        .reset_index()
    )

    # Flatten column names properly
    organ_summary.columns = ["sequence", "organ_id", "organ_type"] + [
        f"{metric}_{stat}" for metric in metrics for stat in ["mean", "std"]
    ]

    # Count runs per (sequence, organ_id)
    n_runs = organ_stats.groupby(["sequence", "organ_id"])["run"].count().reset_index()
    n_runs.columns = ["sequence", "organ_id", "n_runs"]
    organ_summary = organ_summary.merge(n_runs, on=["sequence", "organ_id"])

    # Count timepoints per (sequence, organ_id)
    timepoint_counts = (
        df.groupby(["sequence", "organ_id"]).size().reset_index(name="n_measurements")
    )
    organ_summary = organ_summary.merge(timepoint_counts, on=["sequence", "organ_id"])

    # Reorder columns to have n_runs and n_measurements at the end
    cols = (
        ["sequence", "organ_id", "organ_type"]
        + [f"{metric}_{stat}" for metric in metrics for stat in ["mean", "std"]]
        + ["n_runs", "n_measurements"]
    )
    organ_summary = organ_summary[cols]

    # Sort by sequence and organ_id
    organ_summary = organ_summary.sort_values(["sequence", "organ_id"])

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    organ_summary.to_csv(output_path, index=False)

    return organ_summary

def print_summary_stats(stats, category=None, n_organs=None):
    """Print summary statistics in a readable format."""
    if category:
        print("\n" + "=" * 80)
        print(f"AGGREGATED STATISTICS - {category.upper()}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("AGGREGATED STATISTICS")
        print("=" * 80)

    if "n_sequences" in stats:
        print(f"\nDataset info:")
        print(f"  Sequences: {stats.get('n_sequences', 'N/A')}")
        print(f"  Runs: {stats.get('n_runs', 'N/A')}")
        if n_organs is not None:
            print(f"  Unique organs: {n_organs}")
        print(f"  Total measurements: {stats.get('n_timepoints_total', 'N/A')}")

    print(f"\nMetrics:")

    if "cd_test_mean" in stats:
        print(
            f"  CD (test):  {stats['cd_test_mean']:.6f} ± {stats.get('cd_test_sem', stats.get('cd_test_std', 0)):.6f}"
        )

    if "nc_test_mean" in stats:
        print(
            f"  NC (test):  {stats['nc_test_mean']:.6f} ± {stats.get('nc_test_sem', stats.get('nc_test_std', 0)):.6f}"
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
        description="Create LaTeX table row from all_metrics.csv (per-organ)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to all_metrics.csv file (not required with --header_only)",
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
        "--by_organ_category",
        action="store_true",
        help="Show breakdown by organ category (leaf vs stem)",
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
        "--interpolation",
        action="store_true",
        help="[DEPRECATED] No longer needed. Use compute_metrics_per_organ.py --interpolation to produce all_metrics_interpolation.csv, then pass that CSV here directly.",
    )
    parser.add_argument(
        "--min_timepoints_per_organ",
        type=int,
        default=None,
        help="Minimum number of timepoints required per organ (filters out organs with fewer timepoints)",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default=None,
        help="Export aggregated metrics to CSV file (e.g., output/tables/per_organ/results.csv)",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Process only a specific run number (default: process all runs)",
    )
    parser.add_argument(
        "--per_sequence",
        action="store_true",
        help="Generate per-sequence table (columns per plant sequence, rows per method)",
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
        help="Include last-timepoint EPE (EPE_T) column in per-sequence table",
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

    # Warn if deprecated --interpolation flag is used
    if args.interpolation:
        print(
            "\n⚠️  --interpolation is deprecated and has no effect. "
            "Use compute_metrics_per_organ.py --interpolation to produce all_metrics_interpolation.csv, "
            "then pass that CSV here with --csv."
        )

    # Load CSV
    print(f"Loading per-organ metrics from: {args.csv}")
    df = pd.read_csv(args.csv)

    print(f"Loaded {len(df)} rows (organ × timepoint × run × sequence)")
    print(f"  Sequences: {df['sequence'].nunique()}")
    print(f"  Runs: {df['run'].nunique()}")

    # Exclude requested sequences (e.g. ones not run for this experiment)
    if args.skip_sequences:
        skip = [s.strip() for s in args.skip_sequences.split(",") if s.strip()]
        df = df[~df["sequence"].isin(skip)]
        print(f"  After skip_sequences filter (excluded {skip}): {len(df)} rows")
    print(
        f"  Unique (sequence, organ) pairs: {df.groupby(['sequence', 'organ_id']).ngroups}"
    )
    print(f"  Unique organ types: {df['organ_id'].nunique()}")

    # Filter by run if specified
    if args.run is not None:
        print(f"\n🔍 Filtering to run {args.run} only")
        df = df[df["run"] == args.run].copy()
        print(f"  Filtered to {len(df)} rows")
        if len(df) == 0:
            print(f"\n❌ No data found for run {args.run}!")
            return


    # Apply minimum timepoints filter if requested
    if args.min_timepoints_per_organ:
        print(
            f"\n🔍 Filtering organs by minimum timepoints (>= {args.min_timepoints_per_organ})"
        )
        print(f"  Before filtering: {len(df)} rows")
        print(
            f"    Unique (sequence, organ) pairs: {df.groupby(['sequence', 'organ_id']).ngroups}"
        )
        df, removed_summary = filter_organs_by_min_timepoints(
            df, args.min_timepoints_per_organ
        )
        print(f"  After filtering: {len(df)} rows")
        print(f"    Sequences: {df['sequence'].nunique()}")
        print(f"    Runs: {df['run'].nunique()}")
        print(
            f"    Unique (sequence, organ) pairs: {df.groupby(['sequence', 'organ_id']).ngroups}"
        )

        if removed_summary is not None and len(removed_summary) > 0:
            print(
                f"\n  Removed {len(removed_summary)} organ sequences with insufficient timepoints:"
            )
            for _, row in removed_summary.iterrows():
                print(
                    f"    {row['sequence']}, organ_id={row['organ_id']}: {row['min_timepoints']:.0f}-{row['max_timepoints']:.0f} timepoints (avg: {row['avg_timepoints']:.1f}, {row['n_runs']:.0f} runs)"
                )

        if len(df) == 0:
            print("\n❌ No organs with enough timepoints found!")
            return

    # Per-sequence table mode
    if args.per_sequence:
        print("\nGenerating per-sequence table (per-organ aggregation)...")
        per_seq_stats, has_multiple_runs = compute_per_sequence_metrics(df)
        mean_run_std = compute_mean_column_run_std(df)
        if args.no_std:
            has_multiple_runs = False
        sequences = list(per_seq_stats.keys())
        print(f"  Sequences: {sequences}")
        print(f"  Multiple runs: {has_multiple_runs}")

        header = generate_per_sequence_header(
            sequences, include_lepe=args.include_lepe
        )
        row = format_per_sequence_row(
            args.label,
            per_seq_stats,
            has_multiple_runs,
            include_lepe=args.include_lepe,
            mean_run_std=mean_run_std,
        )
        footer = generate_per_sequence_footer()

        if args.full_table:
            full_table = f"{header}\n{row}\n{footer}"
            print("\n" + "=" * 80)
            print("COMPLETE LATEX TABLE (PER-SEQUENCE, PER-ORGAN)")
            print("=" * 80)
            print("\n" + full_table + "\n")

            if args.output:
                with open(args.output, "w") as f:
                    f.write(full_table + "\n")
                print(f"Table saved to: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("LATEX TABLE ROW (PER-SEQUENCE, PER-ORGAN)")
            print("=" * 80)
            print("\nCopy this row into your LaTeX table:")
            print()
            print(row)
            print()

            if args.output:
                with open(args.output, "w") as f:
                    f.write(row + "\n")
                print(f"LaTeX row saved to: {args.output}")
        return

    # Determine aggregation method
    use_sequence_aggregation = args.by_sequence and not args.by_timepoint

    # Aggregate all organs together
    if use_sequence_aggregation:
        print("\nAggregating by sequence (computing sequence-level means)...")
        stats = aggregate_by_sequence(df)
    else:
        print("\nAggregating across all timepoints...")
        stats = aggregate_metrics(df)

    # Count unique organs
    n_organs = df.groupby(["sequence", "organ_id"]).ngroups

    # Print summary
    print_summary_stats(stats, n_organs=n_organs)

    # Export detailed CSV if requested
    if args.export_csv:
        print(f"\n📊 Exporting detailed organ-level metrics to CSV...")
        csv_summary = export_detailed_csv(df, args.export_csv)
        print(
            f"  Exported {len(csv_summary)} (sequence, organ_id) pairs to: {args.export_csv}"
        )
        if args.run is not None:
            print(
                f"  Each row shows metrics for run {args.run} (std columns will be NaN)"
            )
        else:
            print(
                f"  Each row shows mean ± std across runs for each (sequence, organ_id)"
            )
        print(f"  n_runs: number of model runs")
        print(f"  n_measurements: total number of (timepoint × run) observations")

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
