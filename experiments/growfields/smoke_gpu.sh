#!/usr/bin/env bash
# GPU smoke: SoH2, 200 steps, original and monotone config. Prints s/step, peak GPU memory, and metrics.
# Run on the instance from this directory (CWD matters for compute_metrics_per_organ.py). ~2-3 min.
# Usage: bash smoke_gpu.sh [N_STEPS=200]      -> smoke/<cfg>.log, smoke/summary.txt
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"
N=${1:-200}; PY="$D/.venv/bin/python"; GF="$D/growfields"; mkdir -p smoke; : > smoke/summary.txt
for cfg in shared_continuous monotone_continuous; do
  echo "=== $cfg ($N steps) ===" | tee -a smoke/summary.txt
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > smoke/mem_$cfg.csv & MON=$!
  t0=$(date +%s.%N)
  PYTHONUNBUFFERED=1 "$PY" "$GF/train_per_organ.py" --config "$GF/configs/per_organ/$cfg.yaml" \
    --sequences sorghum_highlight_plant2 --points_split_dir "$GF/configs/datasets/TrackPlant3D/point_splits" \
    --output_dir "$D/smoke/results_$cfg" --seed 0 \
    default_model.n_steps=$N default_model.log_interval=100 output.save_plots=false > "smoke/train_$cfg.log" 2>&1
  t1=$(date +%s.%N); kill $MON 2>/dev/null || true
  tt=$(grep -oE "Training time: [0-9.]+ minutes" "smoke/train_$cfg.log" | grep -oE "[0-9.]+" | head -1)
  { echo "wall incl. load/export: $(echo "$t1 - $t0" | bc) s; training time: ${tt:-?} min -> $(echo "${tt:-0} * 60 / $N" | bc -l | cut -c1-6) s/step (Mac CPU single process: 0.138 s/step)"
    echo "peak GPU mem used (MiB, whole card): $(sort -n smoke/mem_$cfg.csv | tail -1)"
    grep -E "monotone velocity|organ [0-9]: base" "smoke/train_$cfg.log" | head -3 || true; } | tee -a smoke/summary.txt
  EXP=$(ls -td "$D/smoke/results_$cfg"/*/ | head -1); EXP=${EXP%/}
  "$PY" "$GF/scripts/compute_metrics_per_organ.py" --experiment_dir "$EXP" --quiet
  "$PY" "$GF/scripts/compute_metrics_fullplant.py" --experiment_dir "$EXP" --quiet
  "$PY" "$GF/scripts/create_latex_table_fullplant.py" --csv "$EXP/fullplant_metrics.csv" --full_table --per_sequence --label "$cfg" | grep -E "^$cfg" | tee -a smoke/summary.txt || true
  "$PY" growth_metrics.py --experiment_dir "$EXP" | tail -3 | tee -a smoke/summary.txt || true
done
echo SMOKE_DONE | tee -a smoke/summary.txt
