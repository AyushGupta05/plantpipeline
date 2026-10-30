# plantpipeline GPU instances — runbook and record

## Fleet (2026-10-30, 4 x g6.xlarge on-demand, us-east-1a, Project=plantpipeline)
One GrowFields training process per GPU box; the first box is dedicated to the MonoGrow prior. Measured on g6.xlarge:
one GF process alone 0.033-0.053 s/step, ~1.1-1.9 GB GPU; 4 GF + 1 MonoGrow process together drop to 0.5-0.7 s/step each
(4 vCPU saturated, each python ~90 % CPU), so **P=1 per box** and 3 extra boxes instead of more processes.

| name | record file | instance id | public IP | launched (UTC) | job |
|---|---|---|---|---|---|
| monogrow | `.current_instance` | i-02c5554f83e734482 | 44.200.18.8 | 2026-10-30T08:04:59Z | MonoGrow prior, tmux `prior` (`train_prior.py full ... train.out=runs/full_gpu`, log `experiments/monogrow/runs/logs/full_gpu.log`) |
| gf1 | `.instances/gf1` | i-0c415c17d7b3d6bb2 | 98.92.233.106 | 2026-10-30T08:36:55Z | GF shard 0/3 (57 jobs), tmux `train` |
| gf2 | `.instances/gf2` | i-00175be70cdbf8c13 | 44.199.207.165 | 2026-10-30T08:55:16Z | GF shard 1/3 (57 jobs), tmux `train` |
| gf3 | `.instances/gf3` | i-0bcc5b885b6b1001b | 100.48.7.111 | 2026-10-30T08:47:53Z | GF shard 2/3 (56 jobs), tmux `train` |

Quota L-DB2E81BA = 32 vCPU; 16 are used by three unrelated instances (ICP-Flow GPU, instancebaseline, tailwork-gpu2-caps —
never touched), 16 by these four. **Cost: 4 x $0.805 = $3.22/h** (+ 4 x 200 GB gp3 = $0.09/h) while all four run.
Bring-up hiccup (cost ~$0.16): a first gf2 (i-0630179a1f8ccf533, 3.80.222.38) booted fine but port 22 never answered from the laptop,
so it was terminated at 08:49 and relaunched. Real cause, found when gf3 (us-east-1a, 100.48.7.111) showed the same symptom: the carrier NAT
picks the laptop's source IP **per destination**, and those two boxes hashed to NAT addresses not yet in the SG. Adding
37.203.155.171/32 and .172/32 fixed both at once; the SG now allows .169-.174. **If a new box times out on ssh while the others
work, add the next /32 of that pool** (`aws ec2 authorize-security-group-ingress --group-id sg-0cd539950cfd25db4 --protocol tcp --port 22 --cidr 37.203.155.1xx/32`).
The current gf3 is the one launched by the first (interrupted) launch loop at 08:47:53; its record was written by hand.

**Campaign ETA**: 170 jobs x 50k steps x ~0.04-0.05 s/step / 3 boxes ≈ 31-39 h of training (+ ~1 min model export per job) ->
expect the shards to finish **≈ 2026-10-31 16:00-24:00 UTC**; the GF part then costs 3 x ~33 h x $0.805 ≈ **$80**, the MonoGrow box
$0.805/h for as long as the prior runs (0.085 s/step alone vs 0.29 s/step while sharing the GPU with 4 GF processes).
Idle auto-stop (60 min without `train_per_organ.py`/`train_prior.py`/campaign lock/ssh) stops each box by itself afterwards.

### Daily workflow (from `experiments/aws/`)
```bash
./status_all.sh                     # all boxes: jobs done/total, current job, s/step, GPU, MonoGrow prior speed
./ssh.sh gf2 -t tmux attach -t train   # look at a shard (ctrl-b d to detach);  ./ssh.sh = MonoGrow box
./pull_results.sh [--collect]       # rsync runs/ of gf1..3 -> ../growfields/runs_aws/<name>/, merge into runs_aws/merged/,
                                    # --collect runs collect_results.py per stage -> runs_aws/merged/<stage>/results.md
./stop_all.sh [gf1 gf2 ...]         # stop (keeps volumes; confirm);  ./terminate_all.sh [names]  (deletes volumes; type 'terminate')
# relaunch a stopped box: aws ec2 start-instances --instance-ids <id>; then update the IP in .instances/<name>;
# re-run the shard (tmux ... SHARD=k/3 bash campaign_shard.sh): finished run_<k>/results.json are skipped.
```
Sharding (`experiments/growfields/campaign_shard.sh`): fixed 170-job list (headline shared 8x5, headline monotone 8x5, interp
shared/monotone 8x3 `--train_timepoint_stride 2`, extrap shared/monotone 7x3 `--train_holdout_last_n 3`), job i -> shard i mod 3,
so headline jobs come first everywhere and each shard has a balanced mix. One process per job, `--n_runs 1 --seed s`, s = 43..47
(headline) / 43..45 (interp, extrap), placed as GF's `run_<s-42>` (their `--n_runs N --seed 42` protocol gives run_k seed 42+k),
so `collect_results.py` and GF's table scripts aggregate the usual `sequences/<seq>/run_<k>/` layout (verified on the two finished
Mac runs). Per-job logs `runs/<stage>/logs/<seq>_seed<s>.log`, job timings `runs/shard.log`.
If the laptop's NAT IP changes and ssh times out: `aws ec2 authorize-security-group-ingress --group-id sg-0cd539950cfd25db4 --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32`
(launch.sh does this add-only; 37.203.155.169-174 are already allowed; per-destination NAT, see above).

---
## Earlier record (single box, 2026-10-30 01:10-08:34 UTC)

## Status (2026-10-30 01:10 UTC): NOT LAUNCHED — blocked by the G on-demand vCPU quota
`./launch.sh` ran its pre-flight and stopped before creating anything (no instance, no security group, no key pair):

```
quota L-DB2E81BA=16 vCPU, in use=16, requested=4
  i-08db002a48331631d  g5.2xlarge  ICP-Flow GPU          (8 vCPU, since 2026-10-04)
  i-0911a2d093e1e2a30  g5.xlarge   instancebaseline      (4 vCPU, since 2026-10-04)
  i-02a138ff482c2aa8f  g5.xlarge   tailwork-gpu2-caps    (4 vCPU, NEW: launched 2026-10-30 00:37 UTC)
```
The README (written earlier on 2026-10-30) assumed 12/16 in use = room for one `*.xlarge`; the third g5.xlarge
started at 00:37 UTC took that headroom. AWS would refuse `run-instances` with `VcpuLimitExceeded`.
None of these instances were touched. Two ways to unblock (your call, nothing was requested):

1. **Quota increase** (free; this account had 8 -> 16 approved in Aug 2026, usually minutes to 24 h):
   `aws service-quotas request-service-quota-increase --region us-east-1 --service-code ec2 --quota-code L-DB2E81BA --desired-value 32`
   then check `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query Quota.Value`.
2. **Stop one of your own g5.xlarge instances** (`tailwork-gpu2-caps` or `instancebaseline`) while the campaign runs.

Everything else is ready; once there is 4 vCPU of headroom the whole bring-up is:
```bash
cd experiments/aws
YES=1 ./launch.sh            # g6.xlarge on-demand, 200 GB gp3, Project=plantpipeline, shutdown=stop; falls back to g5.xlarge
                             # if g6 has no capacity; writes .current_instance; waits for ssh (~1 min)
./sync_data.sh               # code + TrackPlant3D_v2.zip (235 MB) + data/unified (185 MB) -> ~plantpipeline/ on the instance
./setup_env.sh               # remote: uv venv, torch cu128, PSA, pytorch3d (wheel -> source -> shim), growfields -e, unzip dataset
./ssh.sh 'cd plantpipeline/experiments/growfields && bash smoke_gpu.sh && cat smoke/summary.txt'     # 200-step SoH2 x2 configs
./ssh.sh 'cd plantpipeline/experiments/growfields && tmux new-session -d -s train "bash campaign.sh 2>&1 | tee -a runs/campaign.log"'
```
Then fill in the record below (instance id, IP, launch time, s/step, ETA).

## Record
| Item | Value |
|---|---|
| Instance id | — (not launched) |
| Type / price | g6.xlarge (L4 24 GB, 4 vCPU, 16 GB) on-demand **$0.805/h** us-east-1; fallback g5.xlarge (A10G) $1.006/h |
| Public IP | — |
| Launch time (UTC) | — |
| AMI | ami-055bccd3bd826922c Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04) 20261024 |
| Root volume | 200 GB gp3, DeleteOnTermination, $16/month (~$0.022/h) whether running or stopped |
| Key pair | `icp-flow-mac` = `~/.ssh/icp-flow-aws` (ed25519 fingerprints match: SHA256:RecdPdyAo53Ds+ZP91AJVZ54+gTCW38lF79jpMyWca0). No new key pair created. |
| Security group | `plantpipeline-ssh` (created by launch.sh, SSH/22 from the launching IP only; re-run launch.sh's SG block or `aws ec2 authorize-security-group-ingress` if your IP changes) |
| Tags | Project=plantpipeline (instance + volume), Name=plantpipeline-gpu |
| GPU s/step (smoke) | — (Mac CPU reference: 0.138 s/step single process; 0.30-0.50 s/step with 4 concurrent 3-thread processes) |
| GPU memory / process | — (expected ~2 GB; 4 concurrent should fit easily on 24 GB) |
| Campaign ETA / cost | — see estimate below |

## Campaign (experiments/growfields/campaign.sh)
Queue, in order; P=4 concurrent processes on the one GPU (set `P=` to change):
1. headline `shared_continuous`, 8 plants, `--n_runs 5 --seed 42` (seeds 43-47)  = 8 x 5 x 50k = 2.0M steps
2. headline `monotone_continuous`, same                                               = 2.0M
3. interpolation both configs: `--train_timepoint_stride 2 --n_runs 3`, 8 plants     = 2 x 1.2M
4. extrapolation both configs: `--train_holdout_last_n 3 --n_runs 3`, 7 plants       = 2 x 1.05M
Total **8.5M steps**. Wall-clock = 8.5M x (s/step per process with P concurrent) / P.
Cost at $0.805/h (fill in after the smoke): if 4 concurrent processes each run at 0.04 s/step -> 8.5M x 0.04 / 4 = 24 h -> ~$19;
at 0.08 s/step -> 47 h -> ~$38; CPU-side data/export overhead and the every-200-step eval pass are included in the
measured s/step as long as you measure from the smoke's "Training time" line (smoke_gpu.sh does), but with 4 processes on
4 vCPUs expect the per-process rate to be 1.5-2x slower than the single-process smoke number — measure with
`bash status.sh` twice, 10 min apart, after the first wave starts, and update this table.
Output layout on the instance: `~/plantpipeline/experiments/growfields/runs/{shared_continuous,monotone_continuous}[_interp|_extrap]/`
with `results/fullplant_batch_<cfg>_test_aligned[_stride2|_holdlast3]/`, `<seq>_seeds43-47.log`, `jobs.log`, `results.md`
(written by collect_results.py after each stage), plus `runs/campaign.log` and `runs/campaign.lock` (campaign pid).
Resume after a stop/crash: re-run the tmux command; finished sequences (run_<n>/results.json present) are skipped.
Note: `collect_results.py` scores ALL frames (headline protocol). For the `_interp`/`_extrap` stages the paper's numbers are on
the held-out frames only: afterwards run GrowFields' `scripts/compute_metrics_and_latex_table_fullplant.sh <exp> --interpolation`
and `scripts/compute_metrics_and_latex_table_per_organ.sh <exp> --extrapolation` (BENCHMARK.md §3) on those experiment dirs.

## How to
- **ssh**: `./ssh.sh` (= `ssh -i ~/.ssh/icp-flow-aws ubuntu@$(awk '{print $2}' .current_instance)`), or `./ssh.sh -t tmux attach -t train`.
- **status**: `./ssh.sh 'bash plantpipeline/experiments/growfields/status.sh'` — steps done per sequence (one `Step` line per
  200 steps; 50k x n_runs per log), runs finished, GPU utilisation/memory, whether the campaign process is alive.
- **pull results back** (small: PLYs + CSVs + logs; excludes model checkpoints unless you drop the exclude):
  ```bash
  IP=$(awk '{print $2}' .current_instance)
  rsync -az --info=progress2 -e "ssh -i ~/.ssh/icp-flow-aws" --exclude 'models/' \
    ubuntu@$IP:plantpipeline/experiments/growfields/runs/ ../growfields/runs_aws/
  ```
- **stop** (keeps the 200 GB EBS volume, ~$16/month; compute billing stops; start again with
  `aws ec2 start-instances --instance-ids <id>` — public IP changes, update .current_instance): `./stop.sh`
  (acts only on instances tagged Project=plantpipeline).
- **terminate** (deletes the volume and everything on it — pull results first): `./terminate.sh` (asks for confirmation; same tag filter).
- **idle auto-stop**: cron `/usr/local/bin/idle-check` every 10 min; after 6 consecutive idle ticks (60 min) with no
  `train_per_organ.py` process, no live `runs/campaign.lock` holder, and no ssh login, it runs `shutdown -h now`
  (instance-initiated shutdown behaviour = stop). The campaign lock is held for the entire queue, so gaps between runs
  and the collect_results passes do not count as idle. Disable with `sudo touch /var/tmp/no-idle-stop`.
- **budget guard**: nothing on the account alerts above $10/month (README §0); check `aws ce` or the console.

## Record (filled 2026-10-30 08:21 UTC)
- Instance i-02c5554f83e734482, g6.xlarge (NVIDIA L4 23 GB), us-east-1, public IP 44.200.18.8, launched 2026-10-30T08:04:59Z, on-demand $0.805/h + 200 GB gp3.
- Quota L-DB2E81BA raised 16 -> 32 (approved ~07:55 UTC; EC2 honoured it ~08:05).
- Fixes applied on the way: launch.sh empty-array expansion under bash 3.2; sync_data.sh openrsync `-e` quoting and no `--info=progress2`; SSH rule is per /32 and the laptop sits behind a rotating carrier NAT (37.203.155.169/170/173/174 observed) — add a /32 with `aws ec2 authorize-security-group-ingress --group-id sg-0cd539950cfd25db4 --protocol tcp --port 22 --cidr <ip>/32` when ssh times out.
- Chain: `auto_launch.sh` (launch -> sync -> setup_env -> smoke_gpu -> campaign), then `monogrow_gpu.sh` starts the MonoGrow prior in tmux session `prior`.
- 08:34 UTC: the P=4 GF campaign on this box (tmux `train`, stage shared_continuous, ~10 min in) was killed, its lock and partial `runs/shared_continuous` removed (`runs/campaign_killed_0834.log` kept); the box is now MonoGrow-only and the GF campaign runs sharded on gf1..gf3 (see Fleet). The MonoGrow prior went from 0.29 to ~0.085 s/step. `/usr/local/bin/idle-check` patched to also count `train_prior.py` as busy.
