#!/bin/bash
# usage: run_variant.sh <split> <variant> <family> <template> [tag] [extra fit_test.py args...]
# fits, then scores with monogrow/metrics.py (observed frames + held-out mode) and monogrow/growth_metrics2.py.
set -e
cd "$(dirname "$0")"
SPLIT=$1; VARIANT=$2; FAMILY=$3; TEMPLATE=$4; TAG=${5:-}; shift 5 || shift $#
OUT=$PWD/runs/${SPLIT}_${VARIANT}_${FAMILY}_${TEMPLATE}${TAG}
PY=../monogrow/.venv/bin/python
export OMP_NUM_THREADS=2
rm -rf "$OUT"
$PY fit_test.py --split $SPLIT --variant $VARIANT --family $FAMILY --template $TEMPLATE --out "$OUT" "$@" | tail -1
(cd ../monogrow && $PY metrics.py --experiment_dir "$OUT" | tail -2 | head -1
 if [ "$VARIANT" != headline ]; then $PY metrics.py --experiment_dir "$OUT" --mode $VARIANT | tail -2 | head -1; fi
 $PY growth_metrics2.py --experiment_dir "$OUT" > /dev/null)   # nearest-aware; identical to growth_metrics.py for first/last
