set -eou pipefail

echo $1

exp_dir="$1"

INTERPOLATION=false
EXTRAPOLATION=false
TABLE_ARGS=()  # args meant only for the LaTeX scripts, not the metrics scripts
OTHER_ARGS=()
CAPTURE_NEXT=false
for arg in "${@:2}"; do
    if [[ "$CAPTURE_NEXT" == true ]]; then
        TABLE_ARGS+=("$arg")
        CAPTURE_NEXT=false
    elif [[ "$arg" == "--interpolation" ]]; then
        INTERPOLATION=true
    elif [[ "$arg" == "--extrapolation" ]]; then
        EXTRAPOLATION=true
    elif [[ "$arg" == "--no_std" ]]; then
        TABLE_ARGS+=("$arg")
    elif [[ "$arg" == "--skip_sequences" ]]; then
        # Flag takes a value as the next token; capture both
        TABLE_ARGS+=("$arg")
        CAPTURE_NEXT=true
    elif [[ "$arg" == --skip_sequences=* ]]; then
        TABLE_ARGS+=("$arg")
    else
        OTHER_ARGS+=("$arg")
    fi
done

csv_training="${exp_dir}/fullplant_metrics.csv"

if [ "$EXTRAPOLATION" = true ]; then
    csv_output="${exp_dir}/fullplant_metrics_extrapolation.csv"
elif [ "$INTERPOLATION" = true ]; then
    csv_output="${exp_dir}/fullplant_metrics_interpolation.csv"
else
    csv_output="${exp_dir}/fullplant_metrics.csv"
fi

# For combined-table modes, ensure the training CSV also exists
if [ "$EXTRAPOLATION" = true ] || [ "$INTERPOLATION" = true ]; then
    if [ -f "$csv_training" ]; then
        echo "CSV file exists: $csv_training"
    else
        echo "Computing training metrics..."
        python scripts/compute_metrics_fullplant.py --experiment_dir "$exp_dir" "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}"
    fi
fi

# Compute unseen-frame metrics if needed
echo "Metrics computed and saved to $csv_output"
if [ -f "$csv_output" ]; then
    echo "CSV file exists: $csv_output"
else
    RECOMPUTE_FLAGS=""
    [ "$INTERPOLATION" = true ] && RECOMPUTE_FLAGS="--interpolation"
    [ "$EXTRAPOLATION" = true ] && RECOMPUTE_FLAGS="--extrapolation"
    python scripts/compute_metrics_fullplant.py --experiment_dir "$exp_dir" $RECOMPUTE_FLAGS "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}"
fi

# LaTeX table generation
if [ "$EXTRAPOLATION" = true ]; then
    python scripts/create_latex_table_heldout.py \
        --metrics_training_times "$csv_training" \
        --metrics_interp_times "$csv_output" \
        --metric_type fullplant \
        --per_sequence \
        --full_table \
        --extrapolation \
        "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}" "${TABLE_ARGS[@]+"${TABLE_ARGS[@]}"}"
elif [ "$INTERPOLATION" = true ]; then
    python scripts/create_latex_table_heldout.py \
        --metrics_training_times "$csv_training" \
        --metrics_interp_times "$csv_output" \
        --metric_type fullplant \
        --per_sequence \
        --full_table \
        "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}" "${TABLE_ARGS[@]+"${TABLE_ARGS[@]}"}"
else
    python scripts/create_latex_table_fullplant.py --csv $csv_output --full_table --per_sequence "${TABLE_ARGS[@]+"${TABLE_ARGS[@]}"}"
fi
