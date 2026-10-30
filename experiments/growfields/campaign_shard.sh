#!/usr/bin/env bash
# One shard of the GrowFields campaign, ONE training process at a time (P=1: on a 4-vCPU g6.xlarge a single process
# runs 0.033-0.042 s/step; 4 concurrent processes each drop to 0.5-0.7 s/step, CPU-bound).
#
# Usage (on the instance, from this directory):
#     SHARD=k/N bash campaign_shard.sh           # k in 0..N-1; e.g. SHARD=0/3 on gf1, 1/3 on gf2, 2/3 on gf3
#     SHARD=0/3 LIST=1 bash campaign_shard.sh    # just print this shard's job list (no training)
#     THREADS=4 (OMP/MKL threads)  DRY=1 (print commands, run nothing)
#   tmux new-session -d -s train "SHARD=0/3 bash campaign_shard.sh 2>&1 | tee -a runs/campaign.log"
#
# Job list (fixed order, 170 jobs = 8.5M steps), BENCHMARK.md §3 flags; GF's --n_runs runs seeds sequentially inside
# one process, so here every (config, sequence, seed) is its own process with `--n_runs 1 --seed <s>`; seed s is
# placed as GF's run_<s-42> (their protocol: --n_runs N --seed 42 -> run_k has seed 42+k), so headline seeds
# 43..47 = run_1..run_5 and interp/extrap seeds 43..45 = run_1..run_3, and collect_results.py / GF's table scripts
# see the usual sequences/<seq>/run_<k>/ layout.
#   1 headline shared_continuous   8 plants x 5 seeds        3 interp shared/monotone  8 x 3  --train_timepoint_stride 2
#   2 headline monotone_continuous 8 x 5                      4 extrap shared/monotone  7 x 3  --train_holdout_last_n 3
# Job i goes to shard i mod N, so each shard gets a balanced mix and all headline jobs come first.
# Resumable: a job whose run_<k>/results.json exists is skipped. A job trains into a scratch experiment dir
# (<exp>/.jobs/<seq>_run<k>, --n_runs 1 layout = sequences/<seq>/) and is moved to <exp>/sequences/<seq>/run_<k> on success.
# Holds runs/campaign.lock (pid) for the whole shard so the idle-stop cron never fires between jobs.
# Logs: runs/<cfg>[_interp|_extrap]/logs/<seq>_seed<s>.log (training), runs/shard.log (epoch START/END/SKIP per job).
# Layout: runs/<cfg>/          results/fullplant_batch_<cfg>_test_aligned
#         runs/<cfg>_interp/   results/fullplant_batch_<cfg>_test_aligned_stride2
#         runs/<cfg>_extrap/   results/fullplant_batch_<cfg>_test_aligned_holdlast3
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"   # CWD must be here (relative dataset_path)
SHARD=${SHARD:?set SHARD=k/N}; K=${SHARD%/*}; N=${SHARD#*/}
THREADS=${THREADS:-4}
PY="$D/.venv/bin/python"; GF="$D/growfields"
ALL8=(tomato1_shade_plant1 tomato1_heat_plant3 maize_control_plant2 maize_control_plant3
      sorghum_control_plant2 tobacco_shade_plant3 tobacco_control_plant1 sorghum_highlight_plant2)
EXTRAP7=(tomato1_shade_plant1 tomato1_heat_plant3 maize_control_plant2 maize_control_plant3
         sorghum_control_plant2 tobacco_shade_plant3 tobacco_control_plant1)   # BENCHMARK §3: no sorghum_highlight_plant2

# --- enumerate: lines "cfg sfx esfx seed extra..." in fixed order ---
JOBS=()
add() {  # $1 cfg $2 sfx $3 esfx $4 nseeds $5 extra $6.. seqs
  local cfg=$1 sfx=$2 esfx=$3 ns=$4 extra=$5 seq s; shift 5
  for seq in "$@"; do for s in $(seq 43 $((42 + ns))); do JOBS+=("$cfg|$sfx|$esfx|$seq|$s|$extra"); done; done
}
for cfg in shared_continuous monotone_continuous; do add $cfg ""      ""         5 ""                          "${ALL8[@]}"; done
for cfg in shared_continuous monotone_continuous; do add $cfg _interp _stride2   3 "--train_timepoint_stride 2" "${ALL8[@]}"; done
for cfg in shared_continuous monotone_continuous; do add $cfg _extrap _holdlast3 3 "--train_holdout_last_n 3"   "${EXTRAP7[@]}"; done
MINE=(); for i in "${!JOBS[@]}"; do [ $((i % N)) -eq "$K" ] && MINE+=("${JOBS[$i]}"); done
echo "shard $SHARD: ${#MINE[@]} of ${#JOBS[@]} jobs, THREADS=$THREADS"
if [ "${LIST:-0}" = 1 ]; then printf '%s\n' "${MINE[@]}" | tr '|' ' '; exit 0; fi

mkdir -p runs; LOCK=runs/campaign.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then echo "campaign already running (pid $(cat "$LOCK"))"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
echo "$(date +%s) SHARD $SHARD start pid=$$ jobs=${#MINE[@]}" >> runs/shard.log
n=0
for j in "${MINE[@]}"; do
  IFS='|' read -r cfg sfx esfx seq s extra <<< "$j"; n=$((n + 1)); k=$((s - 42))
  OUT="$D/runs/${cfg}${sfx}"; EXP="$OUT/results/fullplant_batch_${cfg}_test_aligned${esfx}"
  RUN="$EXP/sequences/$seq/run_$k"; TMP="$EXP/.jobs/${seq}_run$k"; LOG="$OUT/logs/${seq}_seed$s.log"
  id="${cfg}${sfx} $seq seed$s"
  if [ -f "$RUN/results.json" ]; then echo "$(date +%s) SKIP  $n/${#MINE[@]} $id" >> runs/shard.log; continue; fi
  mkdir -p "$OUT/logs" "$EXP/sequences/$seq"; rm -rf "$TMP"; mkdir -p "$TMP"
  echo "$(date +%s) START $n/${#MINE[@]} $id log=$LOG" >> runs/shard.log; t0=$(date +%s)
  # shellcheck disable=SC2086
  cmd=("$PY" "$GF/train_per_organ.py" --config "$GF/configs/per_organ/$cfg.yaml" --sequences "$seq"
       --points_split_dir "$GF/configs/datasets/TrackPlant3D/point_splits" --experiment_dir "$TMP"
       --n_runs 1 --seed "$s" $extra output.save_plots=false)
  if [ "${DRY:-0}" = 1 ]; then echo "${cmd[*]}"; continue; fi
  st=FAIL
  if PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS "${cmd[@]}" > "$LOG" 2>&1 \
     && [ -f "$TMP/sequences/$seq/results.json" ]; then
    rm -rf "$RUN"; mv "$TMP/sequences/$seq" "$RUN"
    [ -f "$EXP/experiment_config.json" ] || mv "$TMP/experiment_config.json" "$EXP/" 2>/dev/null || true
    rm -rf "$TMP"; st=OK
  fi
  echo "$(date +%s) END   $n/${#MINE[@]} $id $st wall=$(( $(date +%s) - t0 ))s" >> runs/shard.log
done
echo "$(date +%s) SHARD $SHARD done" >> runs/shard.log
