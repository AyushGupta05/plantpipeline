#!/usr/bin/env bash
# Campaign progress: per run-dir, per sequence: steps done (one "Step NNNNN" line per 200 steps -> 50k = 250 lines
# per run; with --n_runs N a log holds N runs), state, and GPU use. From the Mac:
#   ssh -i ~/.ssh/icp-flow-aws ubuntu@IP 'bash plantpipeline/experiments/growfields/status.sh'
cd "$(dirname "$0")"
[ -e runs/campaign.lock ] && kill -0 "$(cat runs/campaign.lock)" 2>/dev/null && echo "campaign RUNNING (pid $(cat runs/campaign.lock))" || echo "campaign NOT running"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU: /'
echo "training processes: $(pgrep -f train_per_organ.py | wc -l | tr -d " ")"
tot_steps=0
for out in runs/*/; do
  out=${out%/}; [ -f "$out/jobs.log" ] || continue
  n=$(grep -oE 'n_runs=[0-9]+' "$out/jobs.log" | tail -1 | cut -d= -f2); n=${n:-1}
  echo "== $out (n_runs=$n, $(grep -c ' END ' "$out/jobs.log") ended, $(grep -c ' START ' "$out/jobs.log") started) $(grep -q 'STAGE.*done' "$out/jobs.log" && echo COMPLETE)"
  for log in "$out"/*_seed*.log; do
    [ -f "$log" ] || continue
    seq=$(basename "$log" | sed "s/_seeds\{0,1\}[0-9].*//"); s=$(grep -c '^Step' "$log"); steps=$(( s * 200 ))
    done_runs=$(ls -d "$out"/results/*/sequences/"$seq"/run_*/results.json 2>/dev/null | wc -l | tr -d ' ')
    st=$(grep " END   $seq" "$out/jobs.log" | tail -1 | awk '{print $5, $6}')
    printf "  %-26s %7d/%d steps  runs done %d/%d  %s\n" "$seq" "$steps" $((50000 * n)) "$done_runs" "$n" "${st:-running}"
    tot_steps=$(( tot_steps + steps ))
  done
done
echo "total steps logged: $tot_steps / $(( (2*8*5 + 2*8*3 + 2*7*3) * 50000 )) (campaign total, 8.5M)"
tail -3 runs/campaign.log 2>/dev/null
