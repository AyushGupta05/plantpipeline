#!/bin/zsh
# When all 3 GF shards report done (no RUNNING), pull + merge + collect results on the Mac.
cd /Users/ayush/dev/plantpipeline/experiments/aws
while true; do
  s=$(./status_all.sh 2>/dev/null)
  running=$(echo "$s" | grep -c 'shard RUNNING')
  unreach=$(echo "$s" | grep -c unreachable)
  if [ "$running" -eq 0 ] && [ "$unreach" -eq 0 ]; then echo "$(date -u) ALL SHARDS DONE"; ./pull_results.sh --collect > pull_final.out 2>&1 && echo "$(date -u) PULL+COLLECT DONE" || echo "$(date -u) PULL FAILED"; exit 0; fi
  echo "$(date -u) running=$running unreachable=$unreach"; sleep 1800
done
