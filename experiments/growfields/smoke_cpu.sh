#!/usr/bin/env bash
# CPU smoke test for GrowFields (ECCV 2026) on macOS/Apple Silicon, no CUDA, no pytorch3d.
# Reproduces what was verified on 2026-10-30 (Apple M5 Pro, Python 3.12, torch 2.15.0 CPU).
#
# Usage:  bash experiments/growfields/smoke_cpu.sh [WORKDIR] [N_STEPS] [SEQUENCE]
#   WORKDIR   fresh directory for venv + clone + data + results (default: $PWD/gf_smoke)
#   N_STEPS   training steps (default 50; paper uses 50000)
#   SEQUENCE  plant sequence (default sorghum_highlight_plant2 = SoH2, shortest test plant, 5 frames)
# Optional env:
#   TP3D_ZIP=/path/TrackPlant3D_v2.zip   use a local copy instead of the ETH download (235 MB)
#
# Everything is installed under WORKDIR; nothing touches the system Python or the repo.
set -euo pipefail

WORK=${1:-"$PWD/gf_smoke"}
N_STEPS=${2:-50}
SEQ=${3:-sorghum_highlight_plant2}
GF_COMMIT=3f8e947            # growfields "code release" commit (the only one as of 2026-10-30)
PSA_COMMIT=99d86f5fb474accf3b001f1207ff50bd2d343fbb   # plant-shape-analysis v0.4.1, pinned by GrowFields

mkdir -p "$WORK"; cd "$WORK"
GF="$WORK/growfields_src"

# 1. venv + CPU torch (growfields pins torch 2.0.1/cu117 + py3.10; newer torch works, README says so)
[ -d .venv ] || uv venv --python 3.12 .venv
PY="$WORK/.venv/bin/python"
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" "numpy<2.4" scipy scikit-learn omegaconf torchdiffeq open3d plotly pandas tqdm pytest \
  "plant-shape-analysis @ git+https://github.com/Joaquin-Gajardo/plant-shape-analysis.git@${PSA_COMMIT}"
# NOTE: growfields/requirements.txt pins numpy==1.23.1 (no py3.12 wheel); PSA v0.4.1 needs scipy+sklearn
# at import time although its pyproject lists only numpy/torch/open3d.

# 2. GrowFields source (editable install so scripts can `import growfields`)
if [ ! -d "$GF" ]; then git clone --quiet https://github.com/Joaquin-Gajardo/growfields "$GF"; fi
(cd "$GF" && git checkout --quiet "$GF_COMMIT")
uv pip install --python "$PY" -e "$GF"

# 3. pytorch3d is Linux/CUDA-only in their pins; only pytorch3d.loss.chamfer_distance is used.
#    Install a pure-torch shim via a .pth file (works under `python -I`).
mkdir -p shim/pytorch3d/loss
cat > shim/pytorch3d/__init__.py <<'PY'
"""CPU stand-in for pytorch3d; only pytorch3d.loss.chamfer_distance is provided."""
__version__ = "0.0.0-cpu-shim"
PY
cat > shim/pytorch3d/loss/__init__.py <<'PY'
"""Pure-torch pytorch3d.loss.chamfer_distance (default args of pytorch3d 0.7.4):
cham = batch_mean( mean_i min_j ||x_i-y_j||^2 + mean_j min_i ||x_i-y_j||^2 );
normals term = batch_mean( mean_i(1-|cos|) + mean_j(1-|cos|) ) over nearest neighbours.
Verified equal (value and gradient) to growfields.utils.metrics.chamfer_loss deterministic path."""
import torch

def _nn(a, b, chunk=4096):
    d_all, i_all = [], []
    for s in range(0, a.shape[0], chunk):
        d = torch.cdist(a[s:s+chunk], b) ** 2
        m, i = d.min(dim=1); d_all.append(m); i_all.append(i)
    return torch.cat(d_all), torch.cat(i_all)

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None,
                     weights=None, batch_reduction="mean", point_reduction="mean", norm=2,
                     single_directional=False, abs_cosine=True):
    assert norm == 2 and point_reduction == "mean" and not single_directional
    assert x_lengths is None and y_lengths is None and weights is None
    if x.dim() == 2: x = x[None]
    if y.dim() == 2: y = y[None]
    dists, nloss = [], []
    for b in range(x.shape[0]):
        dxy, ixy = _nn(x[b], y[b]); dyx, iyx = _nn(y[b], x[b])
        dists.append(dxy.mean() + dyx.mean())
        if x_normals is not None and y_normals is not None:
            nx, ny = x_normals[b], y_normals[b]
            cx = torch.nn.functional.cosine_similarity(nx, ny[ixy], dim=1, eps=1e-6)
            cy = torch.nn.functional.cosine_similarity(ny, nx[iyx], dim=1, eps=1e-6)
            if abs_cosine: cx, cy = cx.abs(), cy.abs()
            nloss.append((1 - cx).mean() + (1 - cy).mean())
    cham = torch.stack(dists); cham_n = torch.stack(nloss) if nloss else None
    if batch_reduction == "mean":
        cham = cham.mean(); cham_n = cham_n.mean() if cham_n is not None else None
    elif batch_reduction == "sum":
        cham = cham.sum(); cham_n = cham_n.sum() if cham_n is not None else None
    return cham, cham_n
PY
SITE=$("$PY" -c "import site;print(site.getsitepackages()[0])")
echo "$WORK/shim" > "$SITE/pytorch3d_shim.pth"
"$PY" -I -c "from pytorch3d.loss import chamfer_distance; import growfields.utils.metrics; print('shim OK')"

# 4. Data: TrackPlant3D v2 (ETH Research Collection, CC BY 4.0) -> WORK/data/TrackPlant3D/versions/v2
#    compute_metrics_per_organ.py hard-codes "data/TrackPlant3D/versions" relative to CWD, so keep CWD=WORK.
DATA="$WORK/data/TrackPlant3D/versions"
if [ ! -d "$DATA/v2/gt_corrected_v2" ]; then
  mkdir -p "$DATA"
  if [ -n "${TP3D_ZIP:-}" ] && [ -f "$TP3D_ZIP" ]; then
    unzip -q -o "$TP3D_ZIP" -d "$DATA"             # zip has top-level v2/
  else
    "$PY" -I -m plant_shape_analysis.utils.download_dataset --version v2 --target-dir "$DATA"
  fi
fi
echo "PLY files: $(find "$DATA/v2/gt_corrected_v2" -name '*.ply' | wc -l) (expect 378)"

# 5. Train GrowFields ("Ours (full)" config) for N_STEPS on one sequence. CPU: DEVICE falls back
#    automatically (growfields/pipeline/config.py). Run with -I so nothing from the checkout dir is imported.
OUT="$WORK/results"
"$PY" -I "$GF/train_per_organ.py" \
  --config "$GF/configs/per_organ/shared_continuous.yaml" \
  --sequences "$SEQ" \
  --points_split_dir "$GF/configs/datasets/TrackPlant3D/point_splits" \
  --dataset_path "$DATA" --output_dir "$OUT" --seed 0 \
  default_model.n_steps="$N_STEPS" default_model.log_interval=10 output.save_plots=false \
  2>&1 | tee "$WORK/train_log.txt" | grep -E "Step 0|Training time|Saved|Organ [0-9]+:|Error|Traceback"
EXP=$(ls -td "$OUT"/*/ | head -1); EXP=${EXP%/}
echo "experiment dir: $EXP"

# 6. Evaluate exactly as the paper tables are produced (metrics from saved PLYs, then table).
"$PY" -I "$GF/scripts/compute_metrics_per_organ.py" --experiment_dir "$EXP" --quiet
"$PY" -I "$GF/scripts/compute_metrics_fullplant.py" --experiment_dir "$EXP" --quiet
"$PY" -I "$GF/scripts/create_latex_table_per_organ.py" --csv "$EXP/all_metrics.csv" \
  --full_table --per_sequence --min_timepoints_per_organ 2 --label "Smoke" | grep -E "^Smoke|Removed"
"$PY" -I "$GF/scripts/create_latex_table_fullplant.py" --csv "$EXP/fullplant_metrics.csv" \
  --full_table --per_sequence --label "Smoke" | grep -E "^Smoke"
echo "CSV: $EXP/all_metrics.csv  $EXP/fullplant_metrics.csv"
