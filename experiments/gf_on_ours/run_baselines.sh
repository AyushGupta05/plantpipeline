#!/usr/bin/env bash
# RichardsNet (template nearest) and MonoGrow (z-only, 95k prior, objective/source nearest) on ONE converted sequence,
# then monogrow/metrics.py + GF's metric scripts (score.sh with NEAREST=0).
#   bash run_baselines.sh <sequence> [protocol=headline|interpolation|extrapolation] [extra MonoGrow fit.* overrides...]
# Leakage: the 95k prior (and richards/prior.npz) were fitted on Pheno4D minus M01/M02/M04/T01/T05 and on ALL Sugar4D
# plants (train+val+test), so these baselines are only fair on those five Pheno4D plants. Env: THREADS (default 4),
# RICHARDS=0 / MONOGROW=0 to skip one, SCORE=0 to skip scoring.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"
SEQ=${1:?sequence}; PROTO=${2:-headline}; shift $(( $# > 1 ? 2 : 1 ))
MPY="$D/../monogrow/.venv/bin/python"; PRI="$D/../results/priors"
export OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4}
case "$PROTO" in headline) SFX="" ;; interpolation) SFX=_interp ;; extrapolation) SFX=_extrap ;; *) echo "unknown protocol"; exit 1 ;; esac
RE="$D/runs/richards_nearest${SFX}/results/fullplant_batch_richards_nearest_ours"
ME="$D/runs/monogrow_nearest${SFX}/results/fullplant_batch_monogrow_nearest_ours"
if [ "${RICHARDS:-1}" = 1 ]; then
  mkdir -p "$RE"; t0=$(date +%s)
  "$MPY" baselines/run_richards.py --variant "$PROTO" --template nearest --sequences "$SEQ" --out "$RE" 2>&1 | tail -3
  echo "$(date +%s) richards_nearest $PROTO $SEQ wall=$(( $(date +%s) - t0 ))s" | tee -a "$D/runs/jobs.log"
fi
if [ "${MONOGROW:-1}" = 1 ]; then
  mkdir -p "$ME"; t0=$(date +%s)
  "$MPY" baselines/run_monogrow.py full "fit.variant=$PROTO" fit.source=nearest fit.objective=nearest "fit.sequences=$SEQ" \
      "fit.prior=$PRI/prior_cpu95k.pt" "fit.ckpt=$PRI/ckpt_cpu95k.pt" "fit.out=$ME" "fit.threads=${THREADS:-4}" "$@" 2>&1 | tail -4
  echo "$(date +%s) monogrow_nearest $PROTO $SEQ wall=$(( $(date +%s) - t0 ))s" | tee -a "$D/runs/jobs.log"
fi
if [ "${SCORE:-1}" = 1 ]; then
  [ "${RICHARDS:-1}" = 1 ] && NEAREST=0 bash score.sh "$RE" "$PROTO"
  [ "${MONOGROW:-1}" = 1 ] && NEAREST=0 bash score.sh "$ME" "$PROTO"
fi
