#!/usr/bin/env bash
# Run one GrowFields per-organ config on the 8 TrackPlant3D test plants (headline protocol:
# fit on all frames, fixed point split, aligned leaves) on CPU, P sequences in parallel.
#
# Usage: bash run_test_split.sh <config> [N_SEEDS=1] [P=4] [THREADS=3] [EXTRA train_per_organ args...]
#   <config>  name of configs/per_organ/<config>.yaml in the vendored repo (shared_continuous | monotone_continuous)
#   N_SEEDS   1 -> `--n_runs 1 --seed 43` (= seed of run_1 in the paper's --n_runs protocol, 42+1);
#             N>1 -> `--n_runs N --seed 42` (their exact protocol: seeds 43..42+N, run_k/ layout)
#   P         parallel processes (one per sequence); THREADS torch/OMP threads per process (P*THREADS <= 13 on this Mac)
# Outputs: runs/<config>/results/fullplant_batch_<config>_test_aligned/sequences/<seq>/...
#          runs/<config>/<seq>_seed<k>.log      (training log; "Training time" line at the end)
#          runs/<config>/jobs.log               (start/end/wall time per job)
# Then: .venv/bin/python collect_results.py <config>   -> runs/<config>/results.md
set -euo pipefail
CFG=${1:?config name}; N_SEEDS=${2:-1}; P=${3:-4}; THREADS=${4:-3}; shift $(( $# < 4 ? $# : 4 )) || true
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D"   # CWD must be here: configs use dataset_path data/TrackPlant3D/versions (relative)
PY="$D/.venv/bin/python"; GF="$D/growfields"
OUT="$D/runs/$CFG"; EXP="$OUT/results/fullplant_batch_${CFG}_test_aligned"
mkdir -p "$EXP"
# longest sequences first so the tail of the queue is short (frames: Tm1S1 17, Tm1H3 16, MaC2 9, ...)
SEQS=(tomato1_shade_plant1 tomato1_heat_plant3 maize_control_plant2 maize_control_plant3
      sorghum_control_plant2 tobacco_shade_plant3 tobacco_control_plant1 sorghum_highlight_plant2)
if [ "$N_SEEDS" -eq 1 ]; then SEED_ARGS="--n_runs 1 --seed 43"; TAG="seed43"
else SEED_ARGS="--n_runs $N_SEEDS --seed 42"; TAG="seeds43-$((42+N_SEEDS))"; fi

job() {  # $1 = sequence
  local seq=$1 log t0; shift; log="$OUT/${seq}_${TAG}.log"; t0=$(date +%s)
  echo "$(date '+%F %T') START $seq shell_pid=$$ threads=$THREADS" >> "$OUT/jobs.log"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  "$PY" "$GF/train_per_organ.py" --config "$GF/configs/per_organ/$CFG.yaml" \
    --sequences "$seq" --points_split_dir "$GF/configs/datasets/TrackPlant3D/point_splits" \
    --experiment_dir "$EXP" $SEED_ARGS output.save_plots=false "$@" > "$log" 2>&1 \
    && st=OK || st=FAIL
  echo "$(date '+%F %T') END   $seq $st wall=$(( $(date +%s) - t0 ))s" >> "$OUT/jobs.log"
}
export -f job; export OUT TAG THREADS PY GF CFG EXP
echo "$(date '+%F %T') BATCH $CFG seeds=$N_SEEDS P=$P threads=$THREADS extra='$*' pid=$$" >> "$OUT/jobs.log"
# xargs -P runs P jobs at a time; stagger starts so concurrent experiment_config.json writes don't collide
printf '%s\n' "${SEQS[@]}" | xargs -P "$P" -I{} bash -c 'sleep $((RANDOM % 5)); job "$@"' _ {} "$@"
echo "$(date '+%F %T') BATCH $CFG done" >> "$OUT/jobs.log"
