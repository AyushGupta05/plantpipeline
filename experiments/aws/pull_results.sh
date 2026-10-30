#!/usr/bin/env bash
# Pull runs/ from every GF shard box into experiments/growfields/runs_aws/<name>/ (no model checkpoints, no scratch
# .jobs dirs), then merge them into experiments/growfields/runs_aws/merged/ (same runs/<stage>/results/<exp>/sequences/
# <seq>/run_<k>/ layout, so collect_results.py can aggregate across shards), and optionally run collect_results.py.
# Usage: ./pull_results.sh [--collect]      (re-runnable; rsync is incremental; merge uses hard links / copies)
# Then (CWD experiments/growfields): .venv/bin/python collect_results.py <stage> --experiment_dir runs_aws/merged/<stage>/results/<exp>
# Stages: shared_continuous monotone_continuous *_interp (exp ..._stride2) *_extrap (exp ..._holdlast3).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); GFD=$(cd "$HERE/../growfields" && pwd); KEY_FILE=${KEY_FILE:-$HOME/.ssh/icp-flow-aws}
SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
DEST="$GFD/runs_aws"; mkdir -p "$DEST"
for rec in "$HERE"/.instances/*; do
  [ -f "$rec" ] || continue; name=$(basename "$rec"); read -r IID IP _ < "$rec"
  echo "== $name ($IP) -> runs_aws/$name/"; mkdir -p "$DEST/$name"
  rsync -az -e "$SSH" --exclude '.jobs/' ubuntu@"$IP":plantpipeline/experiments/growfields/runs/ "$DEST/$name/" \
    || echo "  rsync from $name FAILED (stopped?)"
done
echo "== merging into runs_aws/merged/"
M="$DEST/merged"; mkdir -p "$M"
for shard in "$DEST"/*/; do
  shard=${shard%/}; [ "$(basename "$shard")" = merged ] && continue
  for exp in "$shard"/*/results/*/; do
    [ -d "$exp/sequences" ] || continue
    rel=${exp#"$shard"/}; mkdir -p "$M/$rel"
    [ -f "$M/$rel/experiment_config.json" ] || cp "$exp/experiment_config.json" "$M/$rel/" 2>/dev/null || true
    for run in "$exp"/sequences/*/run_*/; do
      [ -f "$run/results.json" ] || continue
      rrel=${run#"$exp"/}; mkdir -p "$M/$rel/$(dirname "$rrel")"
      rm -rf "$M/$rel/$rrel"; cp -R "$run" "$M/$rel/$rrel"
    done
  done
  for lg in "$shard"/*/logs; do [ -d "$lg" ] && mkdir -p "$M/$(basename "$(dirname "$lg")")/logs" && cp "$lg"/*.log "$M/$(basename "$(dirname "$lg")")/logs/" 2>/dev/null; done
done
for st in "$M"/*/; do st=${st%/}; [ -d "$st/results" ] || continue
  n=$(find "$st/results" -name results.json -path '*/run_*' | wc -l | tr -d ' '); echo "  $(basename "$st"): $n finished runs"
done
if [ "${1:-}" = --collect ]; then
  cd "$GFD"
  for st in "$M"/*/; do st=${st%/}; exp=$(ls -d "$st"/results/*/ 2>/dev/null | head -1); [ -n "$exp" ] || continue
    echo "== collect_results $(basename "$st")"
    mkdir -p "$GFD/runs/$(basename "$st")"   # collect_results writes runs/<config>/results.md
    .venv/bin/python collect_results.py "$(basename "$st")" --experiment_dir "${exp%/}" > "$st/collect.log" 2>&1 \
      && cp "$GFD/runs/$(basename "$st")/results.md" "$st/results.md" && echo "  -> runs_aws/merged/$(basename "$st")/results.md" \
      || { echo "  FAILED, see $st/collect.log"; tail -5 "$st/collect.log"; }
  done
fi
