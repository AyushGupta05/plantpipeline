#!/bin/bash
# Batch create train/test splits for multiple sequences
# Usage: bash scripts/create_splits_batch.sh [seed] [train_ratio]

SEED=${1:-42}
TRAIN_RATIO=${2:-0.5}
OUTPUT_DIR="configs/datasets/TrackPlant3D/point_splits"

echo "Creating train/test splits with seed=$SEED and train_ratio=$TRAIN_RATIO"
echo "Output directory: $OUTPUT_DIR"
echo ""

# List of sequences (add/remove as needed)
SEQUENCES=(
    "maize_control_plant1"
    "maize_control_plant2"
    "maize_control_plant3"
    "sorghum_control_plant1"
    "sorghum_control_plant2"
    "sorghum_highlight_plant2"
    "tobacco_control_plant1"
    "tobacco_control_plant2"
    "tobacco_shade_plant3"
    "tomato1_heat_plant3"
    "tomato1_shade_plant1"
    "tomato2_control_plant3"
)

# Create splits for each sequence
for seq in "${SEQUENCES[@]}"; do
    echo "Processing $seq..."
    python scripts/create_train_test_splits.py \
        --sequence "$seq" \
        --train_ratio "$TRAIN_RATIO" \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ Created split for $seq"
    else
        echo "✗ Failed to create split for $seq"
    fi
    echo ""
done

echo "Done! Splits saved to $OUTPUT_DIR/"
echo ""
echo "To run batch experiments with these splits:"
echo "  python train_per_organ.py \\"
echo "    --config configs/per_organ/shared_continuous.yaml \\"
echo "    --points_split_dir $OUTPUT_DIR \\"
echo "    --split test"
