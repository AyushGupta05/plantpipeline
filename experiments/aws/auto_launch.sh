#!/bin/zsh
# Wait for G-vCPU quota headroom, then bring up the GPU box and start the GrowFields campaign. Idempotent per stage.
cd /Users/ayush/dev/plantpipeline/experiments/aws
log() { echo "$(date -u '+%F %T') $*"; }
until [ -f .current_instance ]; do
  if YES=1 ./launch.sh > launch.out 2>&1; then log "LAUNCHED: $(cat .current_instance)"; else
    if grep -q 'exceed quota\|VcpuLimitExceeded' launch.out; then log "waiting for quota ($(grep '^quota' launch.out))"; sleep 300; else log "LAUNCH FAILED (non-quota):"; tail -20 launch.out; exit 1; fi; fi
done
./sync_data.sh > sync.out 2>&1 && log "SYNC OK" || { log "SYNC FAILED"; tail -20 sync.out; exit 1; }
./setup_env.sh > setup.out 2>&1 && log "SETUP OK" || { log "SETUP FAILED"; tail -40 setup.out; exit 1; }
./ssh.sh 'cd plantpipeline/experiments/growfields && bash smoke_gpu.sh && cat smoke/summary.txt' > smoke.out 2>&1 && { log "SMOKE OK"; cat smoke.out | tail -15; } || { log "SMOKE FAILED"; tail -40 smoke.out; exit 1; }
./ssh.sh 'cd plantpipeline/experiments/growfields && tmux new-session -d -s train "bash campaign.sh 2>&1 | tee -a runs/campaign.log"' && log "CAMPAIGN STARTED" || log "CAMPAIGN START FAILED"
