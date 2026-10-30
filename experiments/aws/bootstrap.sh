#!/usr/bin/env bash
# EC2 user-data for the Deep Learning AMI (Ubuntu 24.04). Runs as root once at first boot (~2 min).
# Only OS-level setup here; the Python env is built by setup_env.sh over ssh AFTER sync_data.sh,
# so failures are visible and the venv layout matches the laptop (experiments/growfields/.venv).
# Log: /var/log/cloud-init-output.log on the instance; ends with BOOTSTRAP_DONE.
set -euxo pipefail
U=ubuntu; H=/home/$U
apt-get update -y && (apt-get install -y tmux rsync htop git unzip nvtop || apt-get install -y tmux rsync htop git unzip)
sudo -u $U bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
grep -q '.local/bin' $H/.bashrc || echo 'export PATH=$HOME/.local/bin:$PATH' >> $H/.bashrc
sudo -u $U mkdir -p $H/plantpipeline/experiments/growfields/runs $H/plantpipeline/data/raw

# Idle auto-stop (instance-initiated shutdown behaviour is "stop", so this stops, not terminates):
# every 10 min, count "idle" when there is no train_per_organ.py / train_prior.py process AND no campaign lock
# (experiments/growfields/runs/campaign.lock, held by campaign.sh for its whole queue, so gaps between
# runs never count) AND no ssh login. 6 consecutive idle ticks = 60 min -> shutdown -h now.
cat > /usr/local/bin/idle-check <<'EOS'
#!/bin/bash
H=/home/ubuntu; LOCK=$H/plantpipeline/experiments/growfields/runs/campaign.lock; f=/var/tmp/idle_count
busy=0
pgrep -f "train_per_organ.py|train_prior.py" >/dev/null && busy=1
[ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null && busy=1   # lock holds the campaign pid; stale lock ignored
[ "$(who | wc -l)" -gt 0 ] && busy=1
[ -e /var/tmp/no-idle-stop ] && busy=1                                           # touch this file to disable
if [ $busy = 1 ]; then n=0; else n=$(( $(cat $f 2>/dev/null || echo 0) + 1 )); fi
echo $n > $f
if [ $n -ge 6 ]; then logger -t idle-check "idle 60 min, stopping"; shutdown -h now; fi
EOS
chmod +x /usr/local/bin/idle-check
echo '*/10 * * * * root /usr/local/bin/idle-check' > /etc/cron.d/idle-check
echo BOOTSTRAP_DONE
