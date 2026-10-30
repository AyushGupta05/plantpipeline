# Cost scenarios (us-east-1, prices observed 2026-10-30, USD)

Spot prices move hourly and differ ~2x between AZs; numbers below use the cheapest AZ seen today.
EBS: gp3 $0.08/GB-month -> 200 GB root = $16/month ($0.022/h) whether running or stopped.

| Scenario | Instance | $/h | Compute | EBS (1 mo) | Total |
|---|---|---|---|---|---|
| 40 GPU-h GrowFields sweep | g6.xlarge on-demand | 0.805 | 32 | 16 | **48** |
| 40 GPU-h | g6.xlarge spot (us-east-1a) | 0.465 | 19 | 16 | **35** |
| 40 GPU-h | g5.xlarge on-demand | 1.006 | 40 | 16 | **56** |
| 40 GPU-h | g5.xlarge spot (us-east-1d) | 0.522 | 21 | 16 | **37** |
| 100 GPU-h diffusion | g5.2xlarge on-demand | 1.212 | 121 | 16 | **137** |
| 100 GPU-h diffusion | g5.2xlarge spot (us-east-1b) | 0.619 | 62 | 16 | **78** |
| 100 GPU-h diffusion, 48 GB | g6e.xlarge on-demand | 1.861 | 186 | 16 | **202** |
| 100 GPU-h diffusion, 48 GB | g6e.xlarge spot (us-east-1c) | 1.260 | 126 | 16 | **142** |
| Forgot to stop for a week | g5.xlarge on-demand | 1.006 | 169 | 4 | **173** |

Notes
- Spot saves 40-50 % on g5/g6 but today's Spot quota is 0 vCPU (request needed, see README 6). On g5.xlarge the
  saving over 40 h is ~$19; not worth a quota wait for a first run, worth it for recurring sweeps.
- Spot interruption: one-time spot is terminated with 2 min warning. Checkpoint every N steps to the root volume
  and rsync results back (or to S3) regularly; `bootstrap.sh` does not do this for you.
- Bootstrap itself costs ~20 min of instance time (~$0.30) per fresh launch; pytorch3d build dominates.
  Snapshotting a bootstrapped volume as a private AMI avoids that (storage $0.05/GB-month, ~$3/month for 60 GB).
- Data transfer: inbound to EC2 free; outbound to the laptop $0.09/GB after 100 GB/month free (results are small).
- **Existing spend (not this project):** two g5 instances were running at inspection time
  (g5.2xlarge "ICP-Flow GPU" since 2026-10-04, g5.xlarge "instancebaseline" since 2026-10-04) = $2.22/h
  = ~$53/day = ~$1,600/month, plus 1,810 GB of EBS (~$145/month) and 2 Elastic IPs. The two budgets are
  $10/month each. Verify these are intentional before adding anything.
