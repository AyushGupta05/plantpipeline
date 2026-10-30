#!/bin/zsh
# Poll the fleet every 10 min: GPU prior progress/done, shard progress. Appends to fleet_poll.log.
cd /Users/ayush/dev/plantpipeline/experiments/aws
prior_done=0
while true; do
  ts=$(date -u '+%F %T')
  p=$(./ssh.sh 'tail -1 plantpipeline/experiments/monogrow/runs/logs/full_gpu.log' 2>/dev/null | cut -c1-70)
  echo "$ts PRIOR $p"
  if [ $prior_done -eq 0 ] && echo "$p" | grep -q '^done:'; then echo "$ts GPU PRIOR DONE"; prior_done=1; fi
  if (( $(date +%M) < 10 )); then ./status_all.sh 2>/dev/null | grep -E 'jobs|remaining|s/step' | sed "s/^/$ts FLEET /" | head -12; fi
  sleep 600
done
