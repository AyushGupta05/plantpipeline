#!/usr/bin/env bash
# Build the GrowFields Python env ON THE INSTANCE (run from the laptop: ./setup_env.sh [name], which ssh-es and runs
# the remote part; or copy and run it there with REMOTE=1). Idempotent. ~10 min without the pytorch3d build,
# ~25 min with it (4 vCPU).
#   venv: ~/plantpipeline/experiments/growfields/.venv  (same layout as the laptop)
#   torch cu128 + numpy scipy scikit-learn open3d omegaconf torchdiffeq pandas plotly tqdm, PSA v0.4.1 (99d86f5),
#   pytorch3d: (1) official wheel index for the installed torch/CUDA, (2) build from source (FORCE_CUDA, sm 8.6/8.9),
#   (3) pure-torch chamfer shim (experiments/growfields/shim, verified on the Mac) via a .pth file.
#   growfields editable; TrackPlant3D_v2.zip extracted to experiments/growfields/data/TrackPlant3D/versions/v2.
set -euo pipefail
if [ "${REMOTE:-0}" != 1 ]; then
  HERE=$(cd "$(dirname "$0")" && pwd); KEY_FILE=${KEY_FILE:-$HOME/.ssh/icp-flow-aws}
  INSTANCE=${INSTANCE:-}   # instance name (.instances/<name>) or empty = .current_instance (MonoGrow box); also accepted as first arg
  if [ -z "$INSTANCE" ] && [ -n "${1:-}" ] && [ -f "$HERE/.instances/$1" ]; then INSTANCE=$1; shift; fi
  if [ -n "$INSTANCE" ]; then REC="$HERE/.instances/$INSTANCE"; else REC="$HERE/.current_instance"; fi
  read -r IID IP _ < "$REC"
  exec ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new ubuntu@"$IP" "REMOTE=1 PYTORCH3D=${PYTORCH3D:-auto} bash -s" < "$0"
fi
# ---------------- remote part ----------------
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
GF=$HOME/plantpipeline/experiments/growfields; cd "$GF"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
[ -d .venv ] || uv venv --python 3.12 .venv
PY=$GF/.venv/bin/python
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" numpy scipy scikit-learn open3d omegaconf torchdiffeq pandas plotly tqdm ninja setuptools wheel
uv pip install --python "$PY" "git+https://github.com/Joaquin-Gajardo/plant-shape-analysis.git@99d86f5fb474accf3b001f1207ff50bd2d343fbb"
uv pip install --python "$PY" -e "$GF/growfields"
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# pytorch3d
SITE=$("$PY" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
if ! "$PY" -c "from pytorch3d.loss import chamfer_distance" 2>/dev/null; then
  TV=$("$PY" -c "import torch,sys;v=torch.__version__.split('+')[0].split('.');print(f'py3{sys.version_info.minor}_cu{torch.version.cuda.replace(\".\",\"\")}_pyt{v[0]}{v[1]}{v[2]}')")
  IDX="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/$TV/download.html"
  ok=0
  if [ "$PYTORCH3D" != shim ]; then
    echo "trying wheel index $IDX"
    uv pip install --python "$PY" --no-deps -f "$IDX" pytorch3d && ok=1 || echo "no prebuilt wheel for $TV"
    if [ $ok = 0 ] && [ "$PYTORCH3D" != wheel-or-shim ]; then
      echo "building pytorch3d from source (~15 min)"
      FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6;8.9" MAX_JOBS=4 uv pip install --python "$PY" --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>&1 | tail -5 && ok=1 || echo "pytorch3d build FAILED"
    fi
  fi
  if [ $ok = 1 ] && "$PY" -c "import torch;from pytorch3d.loss import chamfer_distance;x=torch.rand(1,100,3,device='cuda');print('pytorch3d', chamfer_distance(x,x)[0].item())"; then
    echo "pytorch3d: real"
  else
    uv pip uninstall --python "$PY" pytorch3d 2>/dev/null || true
    echo "$GF/shim" > "$SITE/pytorch3d_shim.pth"; echo "pytorch3d: SHIM ($GF/shim)"
  fi
fi
"$PY" -c "import pytorch3d,torch;from pytorch3d.loss import chamfer_distance;x=torch.rand(1,2048,3,device='cuda');print('chamfer ok',pytorch3d.__version__ if hasattr(pytorch3d,'__version__') else '?', chamfer_distance(x,x*1.01)[0].item())"

# dataset
if [ ! -d data/TrackPlant3D/versions/v2/gt_corrected_v2 ]; then
  mkdir -p data/TrackPlant3D/versions
  unzip -q -o "$HOME/plantpipeline/data/raw/TrackPlant3D_v2.zip" -d data/TrackPlant3D/versions
fi
echo "PLYs: $(find data/TrackPlant3D/versions/v2 -name '*.ply' | wc -l) (expect 378 in gt_corrected_v2 + others; 830 files total)"
"$PY" -c "import plant_shape_analysis, growfields, open3d, omegaconf, torchdiffeq, pandas, sklearn; print('imports ok')"
echo SETUP_ENV_DONE
