#!/usr/bin/env bash
# rsync code + data from this laptop to an instance, same relative layout under ~/plantpipeline.
# Usage: ./sync_data.sh [name] [--code] [--gf-data] [--unified]  (default: all three); INSTANCE=name also works;
#        no name = .current_instance (MonoGrow box). GF boxes need only --code --gf-data.
#   --code     plantpipeline/, pyproject.toml, experiments/growfields (minus .venv, runs, smoke, data, caches)
#   --gf-data  data/raw/TrackPlant3D_v2.zip (235 MB; setup_env.sh extracts it to experiments/growfields/data/TrackPlant3D/versions/v2)
#   --unified  data/unified (185 MB)
# Pull results back: see INSTANCE.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
KEY_FILE=${KEY_FILE:-$HOME/.ssh/icp-flow-aws}
INSTANCE=${INSTANCE:-}   # instance name (.instances/<name>) or empty = .current_instance (MonoGrow box); also accepted as first arg
if [ -z "$INSTANCE" ] && [ -n "${1:-}" ] && [ -f "$HERE/.instances/$1" ]; then INSTANCE=$1; shift; fi
if [ -n "$INSTANCE" ]; then REC="$HERE/.instances/$INSTANCE"; else REC="$HERE/.current_instance"; fi
read -r IID IP _ < "$REC"
SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new"
DEST="ubuntu@$IP:plantpipeline"
R=(rsync -az -e "$SSH")
$SSH ubuntu@"$IP" "mkdir -p plantpipeline/experiments/growfields/runs plantpipeline/data/raw"   # right after launch, bootstrap may not have made them yet
[ $# -eq 0 ] && set -- --code --gf-data --unified
for a in "$@"; do case $a in
  --code)
    "${R[@]}" --exclude __pycache__ --exclude '*.egg-info' "$ROOT/plantpipeline" "$ROOT/pyproject.toml" "$DEST/"
    "${R[@]}" --exclude .venv --exclude runs --exclude smoke --exclude data --exclude __pycache__ --exclude '*.egg-info' \
       --exclude .pytest_cache "$ROOT/experiments/growfields" "$DEST/experiments/"
    "${R[@]}" --exclude .venv --exclude runs --exclude cache --exclude __pycache__ "$ROOT/experiments/monogrow" "$ROOT/experiments/richards" \
       "$ROOT/experiments/DESIGN_MEMO.md" "$ROOT/experiments/RESULTS.md" "$DEST/experiments/";;
  --gf-data)  "${R[@]}" "$ROOT/data/raw/TrackPlant3D_v2.zip" "$DEST/data/raw/";;
  --unified)  "${R[@]}" "$ROOT/data/unified" "$DEST/data/";;
  *) echo "unknown $a"; exit 1;;
esac; done
