#!/bin/bash
#SBATCH -J growfields
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8G
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=3
#SBATCH --output=sbatch_log/%j.out
#SBATCH --error=sbatch_log/%j.out

# Usage:
#   sbatch run_euler.sh [model_type] [options]
#
# Arguments:
#   model_type: "continuous", "shared", "pairwise", or "encoder_decoder" (default: pairwise)
#
# Options:
#   --split val|test           : Use predefined data split
#   --sequences SEQ1 SEQ2 ...  : Specific sequences to process
#   --crop maize|tomato|...    : Filter by crop type
#   --treatment control|heat   : Filter by treatment
#   --points_split_dir DIR     : Directory with train/test point splits (default: configs/datasets/TrackPlant3D/point_splits)
#   --n_runs N                 : Number of runs per sequence (default: 1)
#   --seed SEED                : Random seed for reproducibility
#   --aligned_leaves           : Use aligned leaves (train in aligned leaf space)
#   --experiment_dir DIR       : Resume into an existing output folder (skips creating a new timestamped dir)
#
#   Config overrides (Hydra-style):
#     default_model.KEY=VALUE  : Override any model parameter
#
#   Examples of config overrides:
#     default_model.hidden_features=256
#     default_model.n_steps=5000
#     default_model.lr=1e-5
#     default_model.finetune_enabled=false
#     output.save_models=false
#
# Examples:
#   # Run pairwise per-organ on test split
#   sbatch run_euler.sh pairwise --split test --points_split_dir configs/datasets/TrackPlant3D/point_splits
#
#   # Run continuous per-organ with custom architecture
#   sbatch run_euler.sh continuous --split test --points_split_dir configs/datasets/TrackPlant3D/point_splits \
#       default_model.hidden_features=384 default_model.n_steps=50000
#
#   # Run shared organ-conditioned model with FiLM and 3 organs per step
#   sbatch run_euler.sh shared --split test --points_split_dir configs/datasets/TrackPlant3D/point_splits \
#       default_model.conditioning=film default_model.organs_per_step=3
#
#   # Run encoder-decoder per-organ model
#   sbatch run_euler.sh encoder_decoder --split test --points_split_dir configs/datasets/TrackPlant3D/point_splits \
#       default_model.latent_dim=128 default_model.n_steps=50000
#
#   # Grid search: disable finetuning and test architecture
#   sbatch run_euler.sh pairwise --split test --points_split_dir configs/datasets/TrackPlant3D/point_splits \
#       default_model.finetune_enabled=false \
#       default_model.hidden_features=128 \
#       default_model.hidden_layers=3 \
#       default_model.lr=1e-4
#

# Parse model type (default: pairwise)
MODEL_TYPE="${1:-pairwise}"
shift

# Validate model type
VALID_TYPES=("continuous" "shared" "pairwise" "encoder_decoder")
VALID=false
for t in "${VALID_TYPES[@]}"; do
    if [[ "$MODEL_TYPE" == "$t" ]]; then VALID=true; break; fi
done
if [[ "$VALID" == false ]]; then
    echo "Error: Invalid model type '$MODEL_TYPE'. Must be one of: ${VALID_TYPES[*]}"
    exit 1
fi

# Select config file and training script based on model type
TRAIN_SCRIPT="train_per_organ.py"
if [[ "$MODEL_TYPE" == "continuous" ]]; then
    CONFIG_FILE="configs/per_organ/continuous.yaml"
elif [[ "$MODEL_TYPE" == "shared" ]]; then
    CONFIG_FILE="configs/per_organ/shared_continuous.yaml"
elif [[ "$MODEL_TYPE" == "encoder_decoder" ]]; then
    CONFIG_FILE="configs/per_organ/encoder_decoder.yaml"
else
    CONFIG_FILE="configs/per_organ/pairwise.yaml"
fi

# Environment setup
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
echo "Loaded modules:"
module list
echo "Conda env: $CONDA_DEFAULT_ENV"
    
echo "=========================================="
echo "GrowFields Per-Organ Training - Euler Cluster"
echo "=========================================="
echo "Model type: $MODEL_TYPE"
echo "Training script: $TRAIN_SCRIPT"
echo "Config file: $CONFIG_FILE"
echo "Additional arguments: $@"
echo "=========================================="
echo ""

# Display GPU info
nvidia-smi

echo ""
echo "Starting training..."
echo ""

# Run training
python "$TRAIN_SCRIPT" --config "$CONFIG_FILE" "$@"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
