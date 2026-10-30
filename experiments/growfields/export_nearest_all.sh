#!/usr/bin/env bash
# Post-hoc "nearest source" re-export of every finished GrowFields run (no retraining).
#   bash export_nearest_all.sh [RESULTS_ROOT=runs] [SOURCE=nearest] [extra args for export_nearest_source.py]
# Looks for experiment dirs <RESULTS_ROOT>/*/results/*/experiment_config.json (and runs_aws/*/results/*),
# skips dirs that are themselves exports (*_nearest, *_first), and writes <exp>_<SOURCE>/ beside each.
# The variant (headline | interpolation | extrapolation) is read from each run's experiment_config.json.
# Sequences without models/ + results.json (still training) are skipped; rerun later to pick them up
# (already exported sequences are skipped unless --overwrite is passed).
# CWD must be experiments/growfields (relative dataset path in the configs). OMP_NUM_THREADS=2 by default.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="${1:-runs}"; SOURCE="${2:-nearest}"; shift $(( $# > 2 ? 2 : $# )) || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" PYTHONUNBUFFERED=1
PY=.venv/bin/python
for cfg in "$ROOT"/*/results/*/experiment_config.json; do
  [ -f "$cfg" ] || continue
  exp="$(dirname "$cfg")"
  case "$exp" in *_nearest|*_first) continue;; esac
  echo "=== $exp -> ${exp}_${SOURCE}"
  $PY growfields/scripts/export_nearest_source.py --experiment_dir "$exp" --source "$SOURCE" "$@" \
    2>&1 | grep -v "^  Organ \|Processing organ\|Present in\|Normalization (from\|^$"
done
