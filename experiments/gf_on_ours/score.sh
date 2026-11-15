#!/usr/bin/env bash
# Score a GrowFields-layout experiment dir on our converted data, with GF's own scripts and monogrow/metrics.py,
# then (unless NEAREST=0) re-export the trained GF model with the nearest-source rule and score that too.
#   bash score.sh <experiment_dir> [protocol=headline|interpolation|extrapolation]
# experiment_dir = .../results/fullplant_batch_<cfg>_ours_aligned[...] (contains experiment_config.json + sequences/).
# GF's compute_metrics_*.py read dataset_path from experiment_config.json (relative -> CWD = this folder).
# EPE is NaN / n_tips 0 everywhere: our PLYs carry no leaf tips. Only CD (cd_test, mm^2) is meaningful.
# Output: <experiment_dir>/{fullplant_metrics,all_metrics}[_interpolation|_extrapolation].csv (GF),
#         <experiment_dir>/monogrow_{fullplant,per_organ,summary}[_mode].{csv,json} (metrics.py), stdout summary.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"
EXP="$(cd "${1:?experiment dir}" && pwd)"; PROTO=${2:-headline}
PY="$D/../growfields/.venv/bin/python"; GF="$D/../growfields/growfields"; MPY="$D/../monogrow/.venv/bin/python"
TP3D="$D/data/TrackPlant3D/versions/v2/gt_corrected_v2"
case "$PROTO" in headline) GFFLAG=(); MODE=normal ;; interpolation) GFFLAG=(--interpolation); MODE=interpolation ;;
  extrapolation) GFFLAG=(--extrapolation); MODE=extrapolation ;; *) echo "unknown protocol $PROTO"; exit 1 ;; esac
export OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4}
score_one() {
  local E=$1; echo "== $E ($PROTO)"
  "$PY" "$GF/scripts/compute_metrics_fullplant.py" --experiment_dir "$E" --quiet ${GFFLAG[@]+"${GFFLAG[@]}"} > "$E/gf_fullplant_metrics.log" 2>&1 \
    && echo "   GF fullplant: ok" || { echo "   GF fullplant FAILED (see $E/gf_fullplant_metrics.log)"; tail -5 "$E/gf_fullplant_metrics.log"; }
  "$PY" "$GF/scripts/compute_metrics_per_organ.py" --experiment_dir "$E" --quiet ${GFFLAG[@]+"${GFFLAG[@]}"} > "$E/gf_per_organ_metrics.log" 2>&1 \
    && echo "   GF per-organ: ok" || { echo "   GF per-organ FAILED (see $E/gf_per_organ_metrics.log)"; tail -5 "$E/gf_per_organ_metrics.log"; }
  "$MPY" "$D/../monogrow/metrics.py" --experiment_dir "$E" --tp3d_dir "$TP3D" --mode "$MODE" --compare_gf 2>&1 | grep -v "^$" | sed 's/^/   /'
}
score_one "$EXP"
if [ "${NEAREST:-1}" = 1 ] && [ -z "$(find "$EXP/sequences" -maxdepth 3 -name results.json -exec grep -l '"richards"\|"monogrow"' {} + 2>/dev/null)" ]; then
  "$PY" "$GF/scripts/export_nearest_source.py" --experiment_dir "$EXP" --source nearest --device cpu --overwrite > "${EXP}_nearest.export.log" 2>&1 \
    && score_one "${EXP}_nearest" || { echo "export_nearest_source FAILED (see ${EXP}_nearest.export.log)"; tail -5 "${EXP}_nearest.export.log"; }
fi
