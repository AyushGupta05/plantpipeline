#!/usr/bin/env bash
# Train GrowFields on ONE converted sequence (CWD must be this folder: GF configs use the relative dataset path
# data/TrackPlant3D/versions, which here points at our converted PLYs).
#   bash run_gf.sh <sequence> [config=shared_continuous|monotone_continuous] [steps=50000] [protocol=headline|interpolation|extrapolation] [seed=43]
# Same flags as experiments/growfields/campaign_shard.sh: --n_runs 1 --seed 43, output.save_plots=false,
# interpolation = --train_timepoint_stride 2, extrapolation = --train_holdout_last_n 3.
# Output: runs/<config>[_interp|_extrap]/results/fullplant_batch_<config>_ours_aligned[_stride2|_holdlast3]/sequences/<sequence>/
# Log:    runs/<config>[_interp|_extrap]/logs/<sequence>_seed<seed>.log      Env: THREADS (OMP/MKL, default 4), DRY=1
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"
SEQ=${1:?sequence, e.g. pheno4d_maize_M01}; CFG=${2:-shared_continuous}; STEPS=${3:-50000}; PROTO=${4:-headline}; SEED=${5:-43}
THREADS=${THREADS:-4}
PY="$D/../growfields/.venv/bin/python"; GF="$D/../growfields/growfields"
case "$PROTO" in
  headline)      SFX=""; ESFX=""; EXTRA=() ;;
  interpolation) SFX=_interp; ESFX=_stride2;   EXTRA=(--train_timepoint_stride 2) ;;
  extrapolation) SFX=_extrap; ESFX=_holdlast3; EXTRA=(--train_holdout_last_n 3) ;;
  *) echo "unknown protocol $PROTO"; exit 1 ;;
esac
OUT="$D/runs/${CFG}${SFX}"; EXP="$OUT/results/fullplant_batch_${CFG}_ours_aligned${ESFX}"; LOG="$OUT/logs/${SEQ}_seed${SEED}.log"
mkdir -p "$OUT/logs" "$EXP"
OVR=(output.save_plots=false "default_model.n_steps=$STEPS")
[ "$STEPS" -lt 200 ] && OVR+=("default_model.log_interval=$STEPS")
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || OVR+=(device=cpu)
cmd=("$PY" "$GF/train_per_organ.py" --config "$GF/configs/per_organ/$CFG.yaml" --sequences "$SEQ"
     --points_split_dir "$D/point_splits" --experiment_dir "$EXP" --n_runs 1 --seed "$SEED" ${EXTRA[@]+"${EXTRA[@]}"} "${OVR[@]}")
echo "${cmd[*]}"; [ "${DRY:-0}" = 1 ] && exit 0
t0=$(date +%s)
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS "${cmd[@]}" > "$LOG" 2>&1 || { echo "FAILED, see $LOG"; tail -20 "$LOG"; exit 1; }
WALL=$(( $(date +%s) - t0 ))
TRAIN=$(python3 -c "import json,sys;print(round(json.load(open(sys.argv[1]))['aggregate_stats']['training_time_seconds'],1))" "$EXP/sequences/$SEQ/results.json" 2>/dev/null || true)
echo "$(date +%s) $CFG $PROTO $SEQ seed$SEED steps=$STEPS wall=${WALL}s train_s=${TRAIN:-?} s/step=$(awk -v t="${TRAIN:-0}" -v n="$STEPS" 'BEGIN{printf "%.3f", (n>0)?t/n:0}')" | tee -a "$D/runs/jobs.log"
echo "results: $EXP/sequences/$SEQ/results.json"
