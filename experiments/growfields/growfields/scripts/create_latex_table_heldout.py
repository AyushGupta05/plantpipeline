#!/usr/bin/env python3
"""
Create a LaTeX table pairing seen (training) frames against held-out frames.

Held-out means either **interpolated** -- days between training days -- or
**extrapolated** -- days after the last training day; both are supported, which is
why the name says held-out rather than naming one of them. Unlike
`create_latex_table_fullplant.py` and `create_latex_table_per_organ.py`, this one
serves both evaluation axes through `--metric_type`, so it carries no axis suffix.

Supports two layouts:
  1. Aggregated (default): 5 metrics (CD, NC, EPE, AccS, AccR) in Training/Held-out
     column groups
  2. Per-sequence (--per_sequence): CD and EPE per plant sequence + Mean, with
     Training/Held-out sub-rows

Supports two metric types:
  - per_organ (default): Uses all_metrics.csv with hyperparameter and
    min_timepoints filtering
  - fullplant: Uses fullplant_metrics.csv (no filtering)

Normally invoked through the `compute_metrics_and_latex_table_*.sh` wrappers, which
pass `--metrics_interp_times` pointing at either the interpolation or the
extrapolation CSV.

Usage:
    # Aggregated per-organ mode (backward compatible)
    python scripts/create_latex_table_heldout.py \
        --metrics_training_times results/.../all_metrics.csv \
        --metrics_interp_times results/.../all_metrics_interpolation.csv \
        --experiment_dir results/.../experiment_dir \
        --min_timepoints_per_organ 2 \
        --label "Ours" --full_table

    # Per-sequence, full plant
    python scripts/create_latex_table_heldout.py \
        --metrics_training_times results/.../fullplant_metrics.csv \
        --metrics_interp_times results/.../fullplant_metrics_interpolation.csv \
        --metric_type fullplant --per_sequence --label "\\textbf{Ours}" --full_table

    # Per-sequence, per-organ
    python scripts/create_latex_table_heldout.py \
        --metrics_training_times results/.../all_metrics.csv \
        --metrics_interp_times results/.../all_metrics_interpolation.csv \
        --experiment_dir results/.../experiment_dir \
        --metric_type per_organ --per_sequence --label "\\textbf{Ours}" --full_table \
        --min_timepoints_per_organ 2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from create_latex_table_fullplant import abbreviate_sequence
from create_latex_table_fullplant import aggregate_by_sequence as aggregate_by_sequence_fullplant
from create_latex_table_fullplant import (
    compute_mean_column_run_std as compute_mean_column_run_std_fullplant,
    compute_per_sequence_metrics as compute_per_sequence_metrics_fullplant,
)
from create_latex_table_per_organ import aggregate_by_sequence as aggregate_by_sequence_per_organ
from create_latex_table_per_organ import (
    compute_mean_column_run_std as compute_mean_column_run_std_per_organ,
    compute_per_sequence_metrics as compute_per_sequence_metrics_per_organ,
    filter_organs_by_min_timepoints,
)

METRICS = ["cd_test", "nc_test", "epe", "accs", "accr"]
METRIC_LABELS = {
    "cd_test": r"CD $\downarrow$",
    "nc_test": r"NC $\uparrow$",
    "epe": r"EPE $\downarrow$",
    "accs": r"AccS $\uparrow$",
    "accr": r"AccR $\uparrow$",
}


def process_csv(
    csv_path,
    metric_type,
    min_timepoints_per_organ=None,
    skip_sequences=None,
):
    """Load CSV, apply filters, and return aggregated stats."""
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(
        f"  {len(df)} rows, {df['sequence'].nunique()} sequences, {df['run'].nunique()} runs"
    )

    if skip_sequences:
        df = df[~df["sequence"].isin(skip_sequences)]
        print(f"  After skip_sequences filter (excluded {skip_sequences}): {len(df)} rows")

    if metric_type == "per_organ":
        if min_timepoints_per_organ:
            df, removed = filter_organs_by_min_timepoints(df, min_timepoints_per_organ)
            n_removed = len(removed) if removed is not None else 0
            print(
                f"  After min_timepoints filter (>={min_timepoints_per_organ}): {len(df)} rows ({n_removed} organs removed)"
            )

    if len(df) == 0:
        print("  WARNING: No data after filtering!")
        return {}

    if metric_type == "fullplant":
        stats = aggregate_by_sequence_fullplant(df)
    else:
        stats = aggregate_by_sequence_per_organ(df)
    return stats


def process_csv_per_sequence(
    csv_path,
    metric_type,
    min_timepoints_per_organ=None,
    skip_sequences=None,
):
    """Load CSV, apply filters, and return per-sequence stats."""
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(
        f"  {len(df)} rows, {df['sequence'].nunique()} sequences, {df['run'].nunique()} runs"
    )

    if skip_sequences:
        df = df[~df["sequence"].isin(skip_sequences)]
        print(f"  After skip_sequences filter (excluded {skip_sequences}): {len(df)} rows")

    if metric_type == "per_organ":
        if min_timepoints_per_organ:
            df, removed = filter_organs_by_min_timepoints(df, min_timepoints_per_organ)
            n_removed = len(removed) if removed is not None else 0
            print(
                f"  After min_timepoints filter (>={min_timepoints_per_organ}): {len(df)} rows ({n_removed} organs removed)"
            )

    if len(df) == 0:
        print("  WARNING: No data after filtering!")
        return {}, False, None

    if metric_type == "fullplant":
        per_seq_stats, has_multiple_runs = compute_per_sequence_metrics_fullplant(df)
        mean_run_std = compute_mean_column_run_std_fullplant(df)
    else:
        per_seq_stats, has_multiple_runs = compute_per_sequence_metrics_per_organ(df)
        mean_run_std = compute_mean_column_run_std_per_organ(df)

    return per_seq_stats, has_multiple_runs, mean_run_std


def format_metric_value(stats, metric, format_spec="mean_sem"):
    """Format a single metric value as LaTeX string."""
    mean_key = f"{metric}_mean"
    sem_key = f"{metric}_sem"
    std_key = f"{metric}_std"

    if mean_key not in stats:
        return "--"

    mean_val = stats[mean_key]

    # Use more decimals for CD/NC/EPE, fewer for AccS/AccR
    if metric in ("accs", "accr"):
        fmt_mean, fmt_err = ".2f", ".2f"
    else:
        fmt_mean, fmt_err = ".3f", ".3f"

    if format_spec == "mean_sem" and sem_key in stats:
        err_val = stats[sem_key]
        return f"{mean_val:{fmt_mean}}\\std{{{err_val:{fmt_err}}}}"
    elif format_spec == "mean_std" and std_key in stats:
        err_val = stats[std_key]
        return f"{mean_val:{fmt_mean}}\\std{{{err_val:{fmt_err}}}}"
    else:
        return f"{mean_val:{fmt_mean}}"


# --- Aggregated (non-per-sequence) table functions ---


def generate_combined_table(label, seen_stats, unseen_stats, format_spec="mean_sem", extrapolation=False):
    """Generate full LaTeX table with seen and unseen metrics side by side."""
    n_metrics = len(METRICS)
    unseen_word = "extrapolation" if extrapolation else "interpolation"
    unseen_label = "Extrapolation times" if extrapolation else "Interpolation times"

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Per-organ reconstruction metrics on training and " + unseen_word + r" frames. "
        r"Mean $\pm$ SEM across sequences. "
        r"CD: Chamfer Distance, NC: Normal Consistency, EPE: End Point Error, AccS/AccR: Strict/Relaxed Accuracy.}"
    )
    lines.append(r"\label{tab:interp_metrics}")

    # Column spec: label + 5 metrics + 5 metrics
    col_spec = "l" + "c" * n_metrics + "c" * n_metrics
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Top header row: group names spanning columns
    lines.append(
        f"& \\multicolumn{{{n_metrics}}}{{c}}{{Training times}}"
        f" & \\multicolumn{{{n_metrics}}}{{c}}{{{unseen_label}}} \\\\"
    )
    # cmidrules under each group
    seen_start, seen_end = 2, 1 + n_metrics
    unseen_start, unseen_end = 2 + n_metrics, 1 + 2 * n_metrics
    lines.append(
        f"\\cmidrule(lr){{{seen_start}-{seen_end}}} "
        f"\\cmidrule(lr){{{unseen_start}-{unseen_end}}}"
    )

    # Sub-header row: metric names
    metric_headers = [METRIC_LABELS[m] for m in METRICS]
    sub_header = (
        "Method & "
        + " & ".join(metric_headers)
        + " & "
        + " & ".join(metric_headers)
        + r" \\"
    )
    lines.append(sub_header)
    lines.append(r"\midrule")

    # Data row
    seen_values = [format_metric_value(seen_stats, m, format_spec) for m in METRICS]
    unseen_values = [format_metric_value(unseen_stats, m, format_spec) for m in METRICS]
    row = (
        label
        + " & "
        + " & ".join(seen_values)
        + " & "
        + " & ".join(unseen_values)
        + r" \\"
    )
    lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# --- Per-sequence combined table functions ---


def generate_per_sequence_combined_header(sequences, metric_type, extrapolation=False):
    """Generate LaTeX header for per-sequence combined training/interpolation table."""
    n_seq = len(sequences)
    n_metrics = 2  # CD, EPE per sequence

    metric_labels = [r"CD $\downarrow$", r"EPE $\downarrow$"]
    abbrevs = {seq: abbreviate_sequence(seq) for seq in sequences}

    unseen_word = "extrapolation" if extrapolation else "interpolation"
    unseen_phrase = "(held-out last timepoints)" if extrapolation else "(every other timestep)"

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")

    if metric_type == "fullplant":
        caption = r"\caption{Full plant " + unseen_word + r" results " + unseen_phrase + r". "
        caption += r"We report mean temporal metrics per sequence. "
        caption += r"CD: Chamfer Distance ($\text{mm}^2$), EPE: End-Point Error on leaf tips ($\text{mm}$).}"
        label = r"\label{tab:interp_metrics}"
    else:
        caption = r"\caption{Per-organ " + unseen_word + r" results " + unseen_phrase + r". "
        caption += r"We report mean temporal metrics per sequence, averaged across organs. "
        caption += r"CD: Chamfer Distance ($\text{mm}^2$), EPE: End-Point Error on leaf tips ($\text{mm}$).}"
        label = r"\label{tab:interp_metrics_perorgan}"

    lines.append(caption)
    lines.append(label)

    # Column spec: l for method, then n_metrics c's per group (sequences + Mean)
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


def generate_per_sequence_combined_footer():
    """Generate LaTeX table footer for per-sequence combined table."""
    lines = []
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # close \resizebox
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def format_per_sequence_combined_rows(
    label,
    train_stats,
    interp_stats,
    has_multiple_runs,
    extrapolation=False,
    train_mean_run_std=None,
    interp_mean_run_std=None,
):
    """
    Generate 3 LaTeX lines for a method in the per-sequence combined table:
    1. Method header row (bold name, empty cells)
    2. Training sub-row
    3. Interpolation/Extrapolation sub-row

    Args:
        label: Method name (e.g. r"\\textbf{Ours}")
        train_stats: OrderedDict from compute_per_sequence_metrics for training
        interp_stats: OrderedDict from compute_per_sequence_metrics for interpolation/extrapolation
        has_multiple_runs: Whether to show ± std
        extrapolation: If True, label sub-row as Extrapolation instead of Interpolation
    """
    # Use the union of sequences from both, sorted
    all_sequences = sorted(set(list(train_stats.keys()) + list(interp_stats.keys())))
    metrics = ["mcd", "mepe"]

    # Total data columns = len(all_sequences) * 2 metrics + 2 metrics (Mean)
    n_data_cols = (len(all_sequences) + 1) * len(metrics)

    # Line 1: Method header row
    header_row = label + " &" * n_data_cols + " \\\\"

    # Helper to format a sub-row
    def _format_sub_row(sub_label, per_seq_stats, mean_run_std=None):
        parts = [sub_label]
        for seq in all_sequences:
            if seq in per_seq_stats:
                stats = per_seq_stats[seq]
                for m in metrics:
                    val = stats[f"{m}_mean"]
                    if np.isnan(val):
                        parts.append("--")
                    elif has_multiple_runs and stats[f"{m}_std"] > 0:
                        parts.append(
                            f"{val:.2f}\\std{{{stats[f'{m}_std']:.2f}}}"
                        )
                    else:
                        parts.append(f"{val:.2f}")
            else:
                parts.extend(["--"] * len(metrics))

        # Mean column: average of per-sequence means
        for m in metrics:
            seq_means = [
                per_seq_stats[s][f"{m}_mean"]
                for s in all_sequences
                if s in per_seq_stats and not np.isnan(per_seq_stats[s][f"{m}_mean"])
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

    # Line 2: Training sub-row
    train_row = _format_sub_row(
        r"\quad \textit{Training}", train_stats, train_mean_run_std
    )

    # Line 3: Interpolation/Extrapolation sub-row
    unseen_sub_label = r"\quad \textit{Extrapolation}" if extrapolation else r"\quad \textit{Interpolation}"
    interp_row = _format_sub_row(unseen_sub_label, interp_stats, interp_mean_run_std)

    return "\n".join([header_row, train_row, interp_row])


def main():
    parser = argparse.ArgumentParser(
        description="Create combined LaTeX table with seen and unseen frame metrics"
    )
    parser.add_argument(
        "--metrics_training_times",
        type=str,
        default=None,
        help="Path to metrics CSV for seen/training frames. If not provided, auto-resolved from --experiment_dir.",
    )
    parser.add_argument(
        "--metrics_interp_times",
        type=str,
        default=None,
        help="Path to metrics CSV for unseen/interpolation frames. If not provided, auto-resolved from --experiment_dir.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Experiment directory. Used for hparam filtering (per_organ) and to auto-resolve CSV paths.",
    )
    parser.add_argument(
        "--label", type=str, default="Ours", help="Row label (method name)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mean_sem", "mean_std", "mean_only"],
        default="mean_sem",
        help="Format: mean_sem (default), mean_std, or mean_only",
    )
    parser.add_argument(
        "--min_timepoints_per_organ",
        type=int,
        default=None,
        help="Min timepoints per organ to include (per_organ only)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output .tex file (default: stdout)"
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        choices=["fullplant", "per_organ"],
        default="per_organ",
        help="Metric type: fullplant (fullplant_metrics.csv) or per_organ (all_metrics.csv, default)",
    )
    parser.add_argument(
        "--per_sequence",
        action="store_true",
        help="Per-sequence column layout (CD, EPE per sequence + Mean) with Training/Interpolation sub-rows",
    )
    parser.add_argument(
        "--no_std",
        action="store_true",
        help="Do not report ± std even when multiple runs are available (still averages over runs)",
    )
    parser.add_argument(
        "--full_table",
        action="store_true",
        help="Generate complete table (header + rows + footer). Without this, only rows are printed.",
    )
    parser.add_argument(
        "--extrapolation",
        action="store_true",
        help="Label unseen frames as Extrapolation instead of Interpolation in captions and sub-rows.",
    )
    parser.add_argument(
        "--skip_sequences",
        type=str,
        default=None,
        help="Comma-separated sequence names to exclude from the table (e.g. 'sorghum_highlight_plant2').",
    )

    args = parser.parse_args()

    skip_sequences = (
        [s.strip() for s in args.skip_sequences.split(",") if s.strip()]
        if args.skip_sequences
        else None
    )

    # Auto-resolve CSV paths from experiment_dir if not explicitly provided
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        if args.metric_type == "fullplant":
            default_train = exp_dir / "fullplant_metrics.csv"
            default_interp = exp_dir / "fullplant_metrics_interpolation.csv"
        else:
            default_train = exp_dir / "all_metrics.csv"
            default_interp = exp_dir / "all_metrics_interpolation.csv"

        if args.metrics_training_times is None:
            if default_train.exists():
                args.metrics_training_times = str(default_train)
                print(f"Auto-resolved training CSV: {args.metrics_training_times}")
            else:
                parser.error(
                    f"Could not find {default_train}. Provide --metrics_training_times explicitly."
                )
        if args.metrics_interp_times is None:
            if default_interp.exists():
                args.metrics_interp_times = str(default_interp)
                print(f"Auto-resolved interpolation CSV: {args.metrics_interp_times}")
            else:
                parser.error(
                    f"Could not find {default_interp}. Provide --metrics_interp_times explicitly."
                )

    if args.metrics_training_times is None or args.metrics_interp_times is None:
        parser.error(
            "Either provide --experiment_dir (to auto-resolve CSVs) or both "
            "--metrics_training_times and --metrics_interp_times."
        )

    # --- Per-sequence mode ---
    if args.per_sequence:
        print("=" * 60)
        print(f"Per-sequence combined table (metric_type={args.metric_type})")
        print("=" * 60)

        print("\n--- Processing TRAINING frames ---")
        train_stats, train_multi, train_mean_run_std = process_csv_per_sequence(
            args.metrics_training_times,
            args.metric_type,
            args.min_timepoints_per_organ,
            skip_sequences,
        )

        unseen_label = "EXTRAPOLATION" if args.extrapolation else "INTERPOLATION"
        print(f"\n--- Processing {unseen_label} frames ---")
        interp_stats, interp_multi, interp_mean_run_std = process_csv_per_sequence(
            args.metrics_interp_times,
            args.metric_type,
            args.min_timepoints_per_organ,
            skip_sequences,
        )

        has_multiple_runs = train_multi or interp_multi
        if args.no_std:
            has_multiple_runs = False

        # Print summary
        all_sequences = sorted(set(list(train_stats.keys()) + list(interp_stats.keys())))
        print(f"\n  Sequences: {all_sequences}")
        print(f"  Multiple runs: {has_multiple_runs}")

        unseen_display = "Extrapolation" if args.extrapolation else "Interpolation"
        for group_name, stats in [("Training", train_stats), (unseen_display, interp_stats)]:
            print(f"\n  {group_name}:")
            for seq, s in stats.items():
                print(
                    f"    {seq}: CD={s['mcd_mean']:.3f}, EPE={s['mepe_mean']:.3f}"
                )

        # Generate rows
        rows = format_per_sequence_combined_rows(
            args.label,
            train_stats,
            interp_stats,
            has_multiple_runs,
            extrapolation=args.extrapolation,
            train_mean_run_std=train_mean_run_std,
            interp_mean_run_std=interp_mean_run_std,
        )

        if args.full_table:
            header = generate_per_sequence_combined_header(all_sequences, args.metric_type, extrapolation=args.extrapolation)
            footer = generate_per_sequence_combined_footer()
            table = f"{header}\n{rows}\n{footer}"

            print("\n" + "=" * 60)
            print("LATEX TABLE (per-sequence combined)")
            print("=" * 60)
            print("\n" + table + "\n")

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(table + "\n")
                print(f"Saved to: {args.output}")
        else:
            print("\n" + "=" * 60)
            print("LATEX ROWS (per-sequence combined)")
            print("=" * 60)
            print("\n" + rows + "\n")

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(rows + "\n")
                print(f"Saved to: {args.output}")

        return

    # --- Aggregated mode (default, backward compatible) ---
    print("=" * 60)
    print("Processing SEEN frames")
    print("=" * 60)
    seen_stats = process_csv(
        args.metrics_training_times,
        args.metric_type,
        args.min_timepoints_per_organ,
        skip_sequences,
    )

    print("\n" + "=" * 60)
    print("Processing UNSEEN frames")
    print("=" * 60)
    unseen_stats = process_csv(
        args.metrics_interp_times,
        args.metric_type,
        args.min_timepoints_per_organ,
        skip_sequences,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for group_name, stats in [("Seen", seen_stats), ("Unseen", unseen_stats)]:
        print(f"\n  {group_name}:")
        for m in METRICS:
            key = f"{m}_mean"
            if key in stats:
                fmt = ".2f" if m in ("accs", "accr") else ".4f"
                print(f"    {m}: {stats[key]:{fmt}}")

    if args.full_table:
        # Generate full table
        table = generate_combined_table(
            args.label, seen_stats, unseen_stats, args.format, extrapolation=args.extrapolation
        )

        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print("\n" + table + "\n")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(table + "\n")
            print(f"Saved to: {args.output}")
    else:
        # Generate just the data row
        seen_values = [
            format_metric_value(seen_stats, m, args.format) for m in METRICS
        ]
        unseen_values = [
            format_metric_value(unseen_stats, m, args.format) for m in METRICS
        ]
        row = (
            args.label
            + " & "
            + " & ".join(seen_values)
            + " & "
            + " & ".join(unseen_values)
            + r" \\"
        )

        print("\n" + "=" * 60)
        print("LATEX ROW")
        print("=" * 60)
        print("\n" + row + "\n")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(row + "\n")
            print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
