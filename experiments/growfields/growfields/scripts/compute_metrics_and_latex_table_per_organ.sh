set -eou pipefail

echo $1

exp_dir="$1"

# Consume --interpolation / --extrapolation from args; pass the rest to the python scripts
INTERPOLATION=false
EXTRAPOLATION=false
OTHER_ARGS=()
for arg in "${@:2}"; do
    if [[ "$arg" == "--interpolation" ]]; then
        INTERPOLATION=true
    elif [[ "$arg" == "--extrapolation" ]]; then
        EXTRAPOLATION=true
    else
        OTHER_ARGS+=("$arg")
    fi
done

csv_training="${exp_dir}/all_metrics.csv"
csv_interp="${exp_dir}/all_metrics_interpolation.csv"
csv_extrap="${exp_dir}/all_metrics_extrapolation.csv"

# Compute training metrics if needed
if [ -f "$csv_training" ]; then
    echo "CSV file exists: $csv_training"
else
    echo "Computing training metrics..."
    python scripts/compute_metrics_per_organ.py --experiment_dir "$exp_dir"
fi

if [ "$EXTRAPOLATION" = true ]; then
    # Compute extrapolation metrics if needed
    if [ -f "$csv_extrap" ]; then
        echo "CSV file exists: $csv_extrap"
    else
        echo "Computing extrapolation metrics..."
        python scripts/compute_metrics_per_organ.py --experiment_dir "$exp_dir" --extrapolation
    fi

    # Generate combined training + extrapolation LaTeX table
    python scripts/create_latex_table_heldout.py \
        --metrics_training_times "$csv_training" \
        --metrics_interp_times "$csv_extrap" \
        --experiment_dir "$exp_dir" \
        --metric_type per_organ \
        --per_sequence \
        --full_table \
        --min_timepoints_per_organ 2 \
        --extrapolation \
        "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}"

elif [ "$INTERPOLATION" = true ]; then
    # Compute interpolation metrics if needed
    if [ -f "$csv_interp" ]; then
        echo "CSV file exists: $csv_interp"
    else
        echo "Computing interpolation metrics..."
        python scripts/compute_metrics_per_organ.py --experiment_dir "$exp_dir" --interpolation
    fi

    # Generate combined training + interpolation LaTeX table
    python scripts/create_latex_table_heldout.py \
        --metrics_training_times "$csv_training" \
        --metrics_interp_times "$csv_interp" \
        --experiment_dir "$exp_dir" \
        --metric_type per_organ \
        --per_sequence \
        --full_table \
        --min_timepoints_per_organ 2 \
        "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}"
else
    # Generate training-only per-organ LaTeX table
    python scripts/create_latex_table_per_organ.py \
        --csv "$csv_training" \
        --full_table \
        --per_sequence \
        --min_timepoints_per_organ 2 \
        "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}"
fi
