#!/bin/zsh
cd /Users/ayush/dev/plantpipeline/experiments/aws
while true; do
  n=$(./ssh.sh 'grep -h "HALF [AB] ALL DONE" ~/plantpipeline/experiments/monogrow/runs/gpu_test/campaign_*.log 2>/dev/null | wc -l' 2>/dev/null | tr -d ' ')
  f=$(./ssh.sh 'grep -h "FAILED" ~/plantpipeline/experiments/monogrow/runs/gpu_test/campaign_*.log 2>/dev/null | wc -l' 2>/dev/null | tr -d ' ')
  echo "$(date -u) halves done=$n failed=$f"
  [ "$f" != "0" ] && [ -n "$f" ] && { echo "CAMPAIGN HAS FAILURES"; ./ssh.sh 'grep -h FAILED ~/plantpipeline/experiments/monogrow/runs/gpu_test/campaign_*.log | head'; }
  [ "$n" = "2" ] && break
  sleep 600
done
./ssh.sh 'cd ~/plantpipeline/experiments/monogrow && ../growfields/.venv/bin/python runs/gpu_test/summarize.py "headline/interpolation: z only (n_z 300, lr_z 3e-3, lambda_prior 1e-3); extrapolation: + n_theta 1000, lr_theta 1e-5" && ../growfields/.venv/bin/python runs/gpu_test/summarize.py "z only" runs/gpu_test/alt_extrapolation_nt0 && bash runs/gpu_test/gf_check.sh >> runs/gpu_test/RESULTS.md 2>&1; echo SUMMARIZED' 2>&1 | tail -3
IP=$(awk '{print $2}' .current_instance)
rsync -az -e "ssh -i $HOME/.ssh/icp-flow-aws -o StrictHostKeyChecking=accept-new" --exclude 'sequences/' --exclude '*.ply' --exclude '*.pt' ubuntu@$IP:plantpipeline/experiments/monogrow/runs/gpu_val ubuntu@$IP:plantpipeline/experiments/monogrow/runs/gpu_test /Users/ayush/dev/plantpipeline/experiments/monogrow/runs/ && echo "$(date -u) GPU TEST RESULTS SYNCED"
