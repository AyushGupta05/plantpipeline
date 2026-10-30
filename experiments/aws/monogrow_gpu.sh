#!/bin/zsh
# After the GF campaign is running on the instance: sync monogrow, start the MonoGrow prior on the GPU in tmux session `prior`.
cd /Users/ayush/dev/plantpipeline/experiments/aws
until grep -q 'CAMPAIGN STARTED' auto_launch.log 2>/dev/null; do sleep 120; done
./sync_data.sh --code > sync_monogrow.out 2>&1
./ssh.sh 'cd plantpipeline/experiments/monogrow && mkdir -p runs/logs && ../growfields/.venv/bin/python -c "import scipy, open3d, torch; print(torch.cuda.is_available())" && tmux new-session -d -s prior "PYTHONUNBUFFERED=1 ../growfields/.venv/bin/python train_prior.py full train.out=runs/full_gpu train.lr_theta=2e-5 train.lambda_g=1e-3 train.grad_clip=1.0 2>&1 | tee runs/logs/full_gpu.log"' && echo "$(date -u) MONOGROW PRIOR STARTED ON GPU" || echo "$(date -u) MONOGROW GPU START FAILED"
