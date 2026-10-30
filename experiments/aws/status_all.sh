#!/usr/bin/env bash
# Progress of every instance in one call: GF shards (.instances/*) and the MonoGrow box (.current_instance).
# Per GF box: jobs done/total, current job + its steps and s/step (from the job's START epoch in runs/shard.log and
# the "Step NNNNN" lines, one per 200 steps), GPU util/mem, whether the shard holds runs/campaign.lock.
# Usage: ./status_all.sh            (ssh ConnectTimeout 10 s; a box that is stopped/unreachable prints one line)
HERE=$(cd "$(dirname "$0")" && pwd); KEY_FILE=${KEY_FILE:-$HOME/.ssh/icp-flow-aws}
SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes"
REMOTE_GF='cd plantpipeline/experiments/growfields 2>/dev/null || exit 0
now=$(date +%s)
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d " ")
if [ -e runs/campaign.lock ] && kill -0 "$(cat runs/campaign.lock)" 2>/dev/null; then camp="shard RUNNING pid $(cat runs/campaign.lock)"; else camp="shard NOT running"; fi
[ -f runs/shard.log ] || { echo "  $camp; no runs/shard.log; GPU $gpu"; exit 0; }
tot=$(grep -m1 " SHARD .* start " runs/shard.log | sed "s/.*jobs=//"); done_ok=$(grep -c " END .* OK " runs/shard.log); fail=$(grep -c " END .* FAIL " runs/shard.log); skip=$(grep -c " SKIP " runs/shard.log)
echo "  $camp; jobs OK $done_ok + skipped $skip / $tot (failed $fail); GPU $gpu; $(uptime -p)"
last=$(grep " START " runs/shard.log | tail -1); [ -n "$last" ] || exit 0
id=$(echo "$last" | sed "s/^[0-9]* START [0-9/]* //; s/ log=.*//"); log=$(echo "$last" | sed "s/.* log=//"); t0=$(echo "$last" | cut -d" " -f1)
if grep -q " END .* $id " runs/shard.log && [ "$(grep -c " START " runs/shard.log)" -eq "$(grep -c " END " runs/shard.log)" ]; then
  echo "  last job: $id finished ($(grep " END .* $id " runs/shard.log | tail -1 | sed "s/.*seed[0-9]* //"))"; else
  s=$(grep -c "^Step" "$log" 2>/dev/null); steps=$(( s * 200 )); el=$(( now - t0 ))
  if [ "$steps" -gt 0 ]; then sps=$(awk "BEGIN{printf \"%.3f\", $el/$steps}"); eta=$(awk "BEGIN{printf \"%.1f\", (50000-$steps)*$el/$steps/3600}"); else sps=-; eta=-; fi
  echo "  current: $id  $steps/50000 steps in ${el}s  ~$sps s/step  job ETA ${eta}h"
fi
grep " END .* OK " runs/shard.log | tail -3 | awk "{print \"  done: \" \$4\" \"\$5\" \"\$6\" \"\$8}"
if [ "$done_ok" -gt 0 ]; then
  avg=$(grep " END .* OK " runs/shard.log | sed "s/.*wall=//; s/s$//" | awk "{s+=\$1} END{printf \"%.0f\", s/NR}")
  left=$(( tot - done_ok - skip )); echo "  mean wall/job ${avg}s -> remaining $left jobs ~ $(awk "BEGIN{printf \"%.1f\", $left*$avg/3600}")h"
fi'
REMOTE_MG='gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d " ")
L=plantpipeline/experiments/monogrow/runs/logs/full_gpu.log
if pgrep -f "train_prio[r].py" >/dev/null; then st="prior RUNNING"; else st="prior NOT running"; fi
echo "  $st; GPU $gpu; GF procs $(pgrep -fc "train_per_orga[n]"); $(uptime -p)"
[ -f $L ] && tail -1 "$L" | sed "s/^/  /" && grep "^step=" "$L" | tail -11 | sed -n "1p;\$p" | tr "\n" " " | sed "s/^/RATE /"; echo'
for rec in "$HERE"/.instances/* "$HERE/.current_instance"; do
  [ -f "$rec" ] || continue
  name=$(basename "$rec"); [ "$name" = .current_instance ] && name=monogrow
  read -r IID IP TYPE T0 < "$rec"
  echo "== $name  $IID  $IP  $TYPE  launched $T0"
  if [ "$name" = monogrow ]; then
    $SSH ubuntu@"$IP" "$REMOTE_MG" 2>/dev/null | awk '/^RATE /{for(i=2;i<=NF;i++){if($i~/^step=/){n++;split($i,a,"=");st[n]=a[2]} if($i~/^s=/){split($i,b,"=");tm[n]=b[2]}}
        if(n==2&&st[2]>st[1])printf "  recent %.3f s/step (last %d steps); mean %.3f s/step since start\n",(tm[2]-tm[1])/(st[2]-st[1]),st[2]-st[1],tm[2]/st[2]; next} {print}' || echo "  (unreachable)"
  else $SSH ubuntu@"$IP" "$REMOTE_GF" 2>/dev/null || echo "  (unreachable)"; fi
done
