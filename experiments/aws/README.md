# AWS GPU training setup for plantpipeline

Prepared 2026-10-30 with read-only calls only. **Update 08:50 UTC: four g6.xlarge boxes run (see `INSTANCE.md` Fleet table); the notes below about the quota block are historical.** **Update 01:10 UTC: `launch.sh` was run and stopped at its quota pre-flight
(16/16 G vCPUs in use after a third g5.xlarge `tailwork-gpu2-caps` i-02a138ff482c2aa8f started at 00:37 UTC); nothing was
created. See `INSTANCE.md` for the unblock options and the full bring-up sequence.** Key pair `icp-flow-mac` verified = `~/.ssh/icp-flow-aws`. Account 936102779163, default region **us-east-1** (`~/.aws/config`).
Items marked *(unverified)* were not checked against a live call.

## 0. Read this first: things found in the account
- **Credentials are the account root** (`arn:aws:iam::936102779163:root`, UserId `9361****`). Works, but create an
  IAM user/Identity Center user with `AdministratorAccess` (or EC2+S3+ServiceQuotas) for day-to-day CLI use.
- **Two GPU instances are running now** (not ours): `i-08db002a48331631d` g5.2xlarge "ICP-Flow GPU" and
  `i-0911a2d093e1e2a30` g5.xlarge "instancebaseline", both since 2026-10-04 -> ~$2.22/h, ~$1,600/month.
  Two more g5.xlarge are stopped (RSNAKnee with a 1,200 GB volume; EEW-Chile-worker). Total EBS 1,810 GB
  (~$145/month). Two Elastic IPs (free while attached). Budgets exist ("My Monthly Cost Budget", "my budger") at
  **$10/month each**, which the running instances exceed within hours. No CloudWatch billing alarms, no SNS topics.
- **They also eat the quota**: G on-demand quota is 16 vCPU in us-east-1, 12 are in use -> headroom = 4 vCPU =
  exactly one `*.xlarge`. A g5.2xlarge (8 vCPU) needs a stopped instance or a quota increase.
- SageMaker: 4 Unified Studio domains `InService` (Aug 2026) and 2 default apps `InService` in domains
  `d-tr1mroadyjbb`, `d-icjjrt8cu2ob`. *(unverified whether those apps bill; Studio default JupyterLab spaces do when running.)*
  Not needed for this project.

## 1. Account facts
| Item | Value |
|---|---|
| Default region | us-east-1 |
| Key pairs (us-east-1) | connectbucket, rsaknee, instance-downloader-key, instancebaseline (rsa); **icp-flow-mac** (ed25519; default in `launch.sh`, local file `~/.ssh/icp-flow-aws` via `KEY_FILE`) |
| Default VPC | vpc-09b2bf5ab64c9e8b9 172.31.0.0/16, default subnets in 1a-1f, public IP on launch |
| Security groups | 16 exist (icp-flow-ssh, launch-wizard-1..4, datazone/NFS ones). `launch.sh` creates/updates its own `plantpipeline-ssh` with SSH from your current IP (37.203.155.170 today) |
| S3 buckets | amazon-sagemaker-...-074aad407d12, amazon-sagemaker-...-aki3nv1iv9hcbr, instance-earthquake-data, rsna-knee-data, sameframe-ai-match-dev-frames-...; none for this project |
| EC2 | see section 0 |
| Snapshots / private AMIs | none |
| GPU types offered | us-east-1 & us-west-2 & eu-central-1 & eu-north-1: g4dn, g5, g6, g6e, p4d. eu-west-1: no g6/g6e. **p3.2xlarge no longer exists in us-east-1** (retired). g6e.xlarge not in us-east-1f |

## 2. Service quotas (vCPUs; 0 = cannot launch)
| Quota | us-east-1 | us-west-2 | eu-west-1 |
|---|---|---|---|
| L-DB2E81BA Running On-Demand G and VT | **16** (12 in use) | 0 | 0 |
| L-417A185B Running On-Demand P | 0 | 0 | 0 |
| L-3819A6DF All G and VT Spot requests | **0** | 0 | 0 |
| L-7212CCBC All P Spot requests | 0 | 0 | 0 |

History: L-DB2E81BA was raised 8 -> 16 in Aug 2026 (two closed cases), so increases do get approved on this
account. **Spot needs a request before `SPOT=1` works** (section 6). Typical turnaround: G/VT on-demand or spot
in a region that already has a G quota: minutes to 24 h, often automatic. P instances: manual review, 1-3 days,
frequently reduced or refused for small accounts; not needed here.

## 3. Prices and specs (us-east-1; us-west-2 on-demand identical)
On-demand from `aws pricing get-products`; spot = min/max across AZs from `describe-spot-price-history` on 2026-10-30.

| Type | GPU | GPU mem | vCPU / RAM | OD $/h | Spot us-east-1 | Spot us-west-2 |
|---|---|---|---|---|---|---|
| g4dn.xlarge | T4 | 16 GB | 4 / 16 GB | 0.526 | 0.27-0.30 | 0.25-0.27 |
| g6.xlarge | L4 | 24 GB | 4 / 16 GB | 0.805 | 0.47-0.79 | 0.43-0.68 |
| g5.xlarge | A10G | 24 GB | 4 / 16 GB | 1.006 | 0.52-0.57 | 0.56-0.58 |
| g5.2xlarge | A10G | 24 GB | 8 / 32 GB | 1.212 | 0.62-1.13 | 0.49-0.55 |
| g6e.xlarge | L40S | 48 GB | 4 / 32 GB | 1.861 | 1.26-1.83 | 1.85-1.86 |
| p3.2xlarge | V100 | 16 GB | 8 / 61 GB | 3.06 (list) | not offered | not offered |
| p4d.24xlarge | 8x A100 | 8x40 GB | 96 / 1.1 TB | 21.96 | 16.9-21.6 | 8.9-18.0 |

Recommendation
- **(a) GrowFields-style neural fields (~2 GB each, many sequences in parallel):** `g6.xlarge` on-demand
  ($0.805) fits the current 4-vCPU quota headroom today and is the cheapest 24 GB card; run 6-10 fits
  concurrently on one GPU (CUDA MPS or separate processes). A10G (`g5.xlarge`) has 2x the memory bandwidth and is
  ~20 % faster for these tiny MLPs but costs 25 % more; the 4 vCPUs, not the GPU, will be the limit for
  parallel small jobs either way. Skip g4dn (T4, no bf16).
- **(b) Point-cloud diffusion, 2048-8192 points:** `g5.2xlarge` (A10G 24 GB, 8 vCPU for the data loader) spot
  at ~$0.62 once the spot quota exists; otherwise `g5.xlarge` on-demand is fine at 2048-4096 points with batch
  32-64 in bf16. Move to `g6e.xlarge` (48 GB) only if 8192-point transformer attention at batch >= 32 OOMs.
  p4d is not justified for this scale.

## 4. AMIs (us-east-1, verified 2026-10-30)
| Purpose | ID | Name | Root snapshot |
|---|---|---|---|
| **default in launch.sh** | ami-055bccd3bd826922c | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04) 20261024 | 30 GB |
| alt (lighter, driver+CUDA only) | ami-05414055cdd6e338f | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261027 | 75 GB |
| plain Ubuntu 24.04 | ami-0d6dd9e8265d48906 | ubuntu-noble-24.04-amd64-server-20261023 (also via SSM `/aws/service/canonical/.../current/amd64/hvm/ebs-gp3/ami-id`) | 8 GB |

The DLAMI ships its own torch; `bootstrap.sh` ignores it and builds a uv venv with the cu128 wheel so the env
matches the laptop. CUDA toolkit on the DLAMI is 12.x *(exact minor unverified; check `nvcc --version`)*.
EBS sizing: raw Pheno4D.zip 4.1 GB + extracted ~12 GB + processed/unified < 1 GB + torch venv ~8 GB +
pytorch3d build ~5 GB + checkpoints -> 200 GB gp3 is comfortable ($16/month); 100 GB would do if pressed.

## 5. Scripts (all in this directory, none executed)
| File | What |
|---|---|
| `launch.sh [name]` | Quota check -> SG `plantpipeline-ssh` gets the current IP/32 **added** (never revokes the other NAT /32s) -> `run-instances` (on-demand g6.xlarge in us-east-1a, falls back to g5.xlarge on InsufficientInstanceCapacity; 200 GB gp3, `Project=plantpipeline`, `Name=plantpipeline-<name>`, shutdown behaviour `stop`) -> writes `.instances/<name>` (`id ip type launch_time`), or `.current_instance` when no name (the MonoGrow box) -> waits for ssh. `YES=1` skips the prompt; refuses to overwrite an existing record |
| `bootstrap.sh` | user-data (~2 min): apt tmux/rsync/unzip/nvtop, uv, dirs, cron idle-stop (60 min with no training process, no live campaign lock, no ssh). The Python env moved to `setup_env.sh` |
| `setup_env.sh` | run from the laptop after `sync_data.sh`: builds `experiments/growfields/.venv` on the instance (torch cu128, PSA 99d86f5, pytorch3d wheel -> source -> shim fallback, growfields -e), extracts TrackPlant3D_v2.zip, verifies CUDA |
| `ssh.sh [name] [ssh args]` | ssh to `.instances/<name>` (`INSTANCE=name` also works); no name = `.current_instance` |
| `sync_data.sh [name] [--code] [--gf-data] [--unified]` | rsync to one instance (default: all three groups; GF boxes need only `--code --gf-data`): plantpipeline/ + pyproject.toml + experiments/growfields minus .venv/runs/smoke/data, TrackPlant3D_v2.zip 235 MB, data/unified 185 MB. openrsync-safe (no `--info=progress2`) |
| `setup_env.sh [name]` | builds the venv on that instance (see below) |
| `status_all.sh` | one call: every box's shard progress (jobs done/total, current job, s/step, GPU) and the MonoGrow prior's s/step |
| `pull_results.sh [--collect]` | rsync `runs/` from every GF box into `experiments/growfields/runs_aws/<name>/` (no `models/`), merge into `runs_aws/merged/` (GF `run_<k>` layout across shards), `--collect` runs `collect_results.py` per stage |
| `stop_all.sh [names]` / `terminate_all.sh [names]` | tag-filtered (`Project=plantpipeline`), list + confirm; terminate removes the local record files |
| `stop.sh` / `terminate.sh` | older single-box versions (no listing/confirmation on stop); prefer `*_all.sh` |
| `../growfields/campaign_shard.sh` | `SHARD=k/N bash campaign_shard.sh` on a GF box: its slice of the fixed 170-job list, one process at a time, resumable (see INSTANCE.md) |
| `cost_estimate.md` | scenarios incl. 40 GPU-h g5.xlarge spot vs on-demand |

Workflow (fleet, current): per GF box `YES=1 NAME=gfN ./launch.sh` -> `./sync_data.sh gfN --code --gf-data` -> `./setup_env.sh gfN` ->
`./ssh.sh gfN 'cd plantpipeline/experiments/growfields && tmux new-session -d -s train "SHARD=k/3 bash campaign_shard.sh 2>&1 | tee -a runs/campaign.log"'`;
then `./status_all.sh` -> `./pull_results.sh --collect` -> `./stop_all.sh` or `./terminate_all.sh`. The MonoGrow box (`.current_instance`)
is driven by `./ssh.sh` without a name (`monogrow_gpu.sh` started its `prior` tmux session). Single-box campaign (`campaign.sh`, P=4) is
superseded: 4 processes on 4 vCPUs are CPU-bound (0.5-0.7 s/step each vs 0.04 alone). Full commands and the instance table in `INSTANCE.md`.

## 6. One-time manual steps (checklist)
- [ ] Decide whether the two running g5 instances (ICP-Flow, instancebaseline) should keep running; they are ~$53/day.
- [ ] Confirm budget for this project (cost_estimate.md: ~$50 for a 40 h sweep, ~$80-140 for 100 h diffusion).
- [ ] Spot quota increase (needed for `SPOT=1`). Console: Service Quotas -> EC2 -> "All G and VT Spot Instance
      Requests" -> request 32. Or CLI (not run):
      `aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-3819A6DF --desired-value 32`
      Use-case wording if a case opens: "Academic research (plant 4D point-cloud reconstruction/diffusion, TU Delft).
      Need to run one g5.2xlarge or g6e.xlarge Spot instance (8 vCPU) in us-east-1 for interruptible training jobs
      lasting 1-6 h with checkpointing. Existing on-demand G quota is 16 vCPU."
- [ ] Optional: raise on-demand G quota 16 -> 32 (`--quota-code L-DB2E81BA --desired-value 32`) so a g5.2xlarge fits
      next to existing instances.
- [ ] Billing alarm (not run). First enable "Receive Billing Alerts" in Billing console preferences (root only), then:
      ```
      aws sns create-topic --name billing-alerts --region us-east-1
      aws sns subscribe --topic-arn arn:aws:sns:us-east-1:936102779163:billing-alerts --protocol email --notification-endpoint you@example.com
      aws cloudwatch put-metric-alarm --region us-east-1 --alarm-name monthly-charges-over-150 \
        --namespace AWS/Billing --metric-name EstimatedCharges --dimensions Name=Currency,Value=USD \
        --statistic Maximum --period 21600 --evaluation-periods 1 --threshold 150 \
        --comparison-operator GreaterThanThreshold --alarm-actions arn:aws:sns:us-east-1:936102779163:billing-alerts
      ```
      The two $10 budgets do have 85 %/100 % actual + 100 % forecast notifications and are already in ALARM
      state, so they fire every month and are useless as a signal; raise them to a realistic limit (e.g. $200).
- [ ] Create a non-root IAM identity and `aws configure` a profile for it.
- [ ] Confirm `~/.ssh/icp-flow-aws` is the private key for key pair `icp-flow-mac` (both exist; pairing *(unverified)*).
      `instancebaseline-mac` is also present locally. Else `aws ec2 create-key-pair` (free) and set `KEY_NAME`/`KEY_FILE`.
- [ ] Push the repo to GitHub (remote exists, no commits) so `bootstrap.sh` can clone instead of needing rsync.
