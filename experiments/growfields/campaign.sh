#!/usr/bin/env bash
# Full GrowFields campaign on one GPU: for both configs, headline (all frames, --n_runs 5), then interpolation
# (--train_timepoint_stride 2 --n_runs 3), then extrapolation (--train_holdout_last_n 3 --n_runs 3, 7 plants),
# flags exactly as BENCHMARK.md §3. Stages run in order (headline for both configs first); within a stage P jobs
# (one per sequence) run concurrently on the GPU. Resumable: a sequence whose run_<n_runs>/results.json exists is skipped.
# collect_results.py runs after each stage -> runs/<stage>/results.md.
#
# Start detached (on the instance, from this directory):
#     tmux new-session -d -s train "bash campaign.sh 2>&1 | tee -a runs/campaign.log"
#   or P=3 bash campaign.sh ...        Progress: bash status.sh      Abort: tmux kill-session -t train; pkill -f train_per_organ.py
# Holds runs/campaign.lock (its pid) for the whole queue so the idle-stop cron never fires between runs.
# Layout: runs/<cfg>/           headline   results/fullplant_batch_<cfg>_test_aligned  (same as the Mac runs)
#         runs/<cfg>_interp/    results/fullplant_batch_<cfg>_test_aligned_stride2
#         runs/<cfg>_extrap/    results/fullplant_batch_<cfg>_test_aligned_holdlast3
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"   # CWD must be here (relative dataset_path)
P=${P:-4}; THREADS=${THREADS:-1}
PY="$D/.venv/bin/python"; GF="$D/growfields"
mkdir -p runs; LOCK=runs/campaign.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then echo "campaign already running (pid $(cat "$LOCK"))"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
ALL8=(tomato1_shade_plant1 tomato1_heat_plant3 maize_control_plant2 maize_control_plant3
      sorghum_control_plant2 tobacco_shade_plant3 tobacco_control_plant1 sorghum_highlight_plant2)
EXTRAP7=(tomato1_shade_plant1 tomato1_heat_plant3 maize_control_plant2 maize_control_plant3
         sorghum_control_plant2 tobacco_shade_plant3 tobacco_control_plant1)   # BENCHMARK §3: no sorghum_highlight_plant2

job() {  # $1 seq; env: OUT EXP NRUNS EXTRA
  local seq=$1 log="$OUT/${seq}_seeds43-$((42+NRUNS)).log" t0; t0=$(date +%s)
  if [ -f "$EXP/sequences/$seq/run_$NRUNS/results.json" ]; then echo "$(date '+%F %T') SKIP  $seq (done)" >> "$OUT/jobs.log"; return 0; fi
  echo "$(date '+%F %T') START $seq" >> "$OUT/jobs.log"
  # shellcheck disable=SC2086
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  "$PY" "$GF/train_per_organ.py" --config "$GF/configs/per_organ/$CFG.yaml" --sequences "$seq" \
    --points_split_dir "$GF/configs/datasets/TrackPlant3D/point_splits" --experiment_dir "$EXP" \
    --n_runs "$NRUNS" --seed 42 $EXTRA output.save_plots=false > "$log" 2>&1 && st=OK || st=FAIL
  echo "$(date '+%F %T') END   $seq $st wall=$(( $(date +%s) - t0 ))s" >> "$OUT/jobs.log"
}
export -f job; export PY GF THREADS

stage() {  # $1 cfg  $2 run-dir suffix  $3 exp suffix  $4 n_runs  $5 extra flags  $6.. sequences
  export CFG=$1 NRUNS=$4 EXTRA=$5; local sfx=$2 esfx=$3; shift 5
  export OUT="$D/runs/${CFG}${sfx}" EXP="$D/runs/${CFG}${sfx}/results/fullplant_batch_${CFG}_test_aligned${esfx}"
  mkdir -p "$EXP"
  echo "$(date '+%F %T') STAGE ${CFG}${sfx} n_runs=$NRUNS extra='$EXTRA' P=$P seqs=$#" | tee -a "$OUT/jobs.log"
  printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c 'sleep $((RANDOM % 5)); job "$@"' _ {}
  "$PY" collect_results.py "${CFG}${sfx}" --experiment_dir "$EXP" >> "$OUT/collect.log" 2>&1 \
    && echo "$(date '+%F %T') STAGE ${CFG}${sfx} done -> $OUT/results.md" | tee -a "$OUT/jobs.log" \
    || echo "$(date '+%F %T') STAGE ${CFG}${sfx} collect_results FAILED (see $OUT/collect.log)" | tee -a "$OUT/jobs.log"
}

echo "$(date '+%F %T') CAMPAIGN start pid=$$ P=$P"
for cfg in shared_continuous monotone_continuous; do stage $cfg ""        ""           5 ""                        "${ALL8[@]}"; done
for cfg in shared_continuous monotone_continuous; do stage $cfg _interp   _stride2     3 "--train_timepoint_stride 2" "${ALL8[@]}"; done
for cfg in shared_continuous monotone_continuous; do stage $cfg _extrap   _holdlast3   3 "--train_holdout_last_n 3"   "${EXTRAP7[@]}"; done
echo "$(date '+%F %T') CAMPAIGN done"
