#!/bin/bash
# usage: gf_check.sh <run dir> <plant> [extrapolation|interpolation]
# Copies one plant of a run into the scratchpad and scores it with GrowFields' own scripts (GF venv, CWD =
# experiments/growfields, as compute_metrics_per_organ.py hard-codes data/TrackPlant3D/versions relative to CWD),
# then diffs every row against monogrow/metrics.py (--compare_gf).
set -e
RUN=$(cd "$1" && pwd); PLANT=$2; MODE=${3:-}
GFDIR=$(cd "$(dirname "$0")/../growfields" && pwd); V=$GFDIR/.venv/bin/python; GF=$GFDIR/growfields/scripts
CHK=/private/tmp/claude-501/-Users-ayush-dev-plantpipeline/59224c09-250d-4b23-b25b-9df144709fe3/scratchpad/gf_check_$(basename "$RUN")_$PLANT
rm -rf "$CHK"; mkdir -p "$CHK/sequences"; cp -r "$RUN/sequences/$PLANT" "$CHK/sequences/"
$V - "$RUN/experiment_config.json" "$CHK/experiment_config.json" "$PLANT" <<'PY'
import json, sys; c = json.load(open(sys.argv[1])); c["sequence_names"] = [sys.argv[3]]; json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
FLAG=${MODE:+--$MODE}
cd "$GFDIR"
export OMP_NUM_THREADS=2
$V -I $GF/compute_metrics_fullplant.py --experiment_dir "$CHK" --quiet $FLAG
$V -I $GF/compute_metrics_per_organ.py --experiment_dir "$CHK" --quiet $FLAG
SUF=${MODE:+_$MODE}
$V -I $GF/create_latex_table_fullplant.py --csv "$CHK/fullplant_metrics$SUF.csv" --full_table --per_sequence 2>&1 | grep -E "^\\\\|&|Mean|plant" | tail -6
$V -I $GF/create_latex_table_per_organ.py --csv "$CHK/all_metrics$SUF.csv" --full_table --per_sequence --min_timepoints_per_organ 2 2>&1 | grep -E "&" | tail -4
cd "$GFDIR/../monogrow" && .venv/bin/python metrics.py --experiment_dir "$CHK" --mode ${MODE:-normal} --compare_gf
