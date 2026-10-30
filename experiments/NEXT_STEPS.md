# Resume checklist (written 2026-10-30 ~22:10 UTC, before the laptop closed)

Everything that matters overnight runs on AWS in tmux and the boxes stop themselves when idle for 60 min.
Nothing on the Mac survives the lid closing; all Mac-side chains were either finished or duplicated on the box.

## What is running on AWS (us-east-1, Project=plantpipeline, $0.805/h each while up)
| box | what | where | expected end |
|---|---|---|---|
| gf1, gf2, gf3 | GrowFields campaign shards (170 jobs: headline 5 seeds × 2 configs, interpolation 3 seeds × 2, extrapolation 3 seeds × 2) | `~/plantpipeline/experiments/growfields/runs/` | ~2026-10-31 17:00–20:00 UTC, then auto-stop |
| monogrow (`.current_instance`) | tmux `obj`: matched-objective val grid + test runs (runs/gpu_val_obj, runs/gpu_test_obj); tmux `prior2`: cosine-decay prior (runs/full_gpu_v2cos, tagged ckpts every 25k); tmux `v2eval`: evaluates that prior's 100k/125k/150k checkpoints on val when it finishes (runs/ablate_prior_len_gpu) | `~/plantpipeline/experiments/monogrow/runs/` | prior ~4–6 h; obj chain a few hours; then auto-stop |

If a box has stopped: `aws ec2 start-instances --instance-ids <id>` (ids in experiments/aws/INSTANCE.md "Fleet" table), wait, then pull.
If ssh times out: the laptop's public IP changed; `aws ec2 authorize-security-group-ingress --region us-east-1 --group-id sg-0cd539950cfd25db4 --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32`.

## Morning commands (from /Users/ayush/dev/plantpipeline/experiments/aws)
```bash
./status_all.sh                                   # shards done? boxes up?
./pull_results.sh --collect                       # GF fleet → growfields/runs_aws/merged + results.md per stage (headline/interp/extrap, both configs)
cd ../growfields && bash export_nearest_all.sh runs_aws/merged nearest   # GF + nearest-source exports (CWD growfields; needs configs symlink, present)
for c in shared_continuous monotone_continuous; do .venv/bin/python collect_results.py $c --experiment_dir runs_aws/merged/$c/results/fullplant_batch_${c}_test_aligned_nearest; cp runs/$c/results.md runs_aws/merged/$c/results_headline_nearest.md; done
#   interpolation/extrapolation stage dirs: see runs_aws/merged/<config>/results/ (names contain interp/extrap); score with
#   ../monogrow/.venv/bin/python ../monogrow/metrics.py --experiment_dir <dir> --mode interpolation|extrapolation --tp3d_dir data/TrackPlant3D/versions/v2/gt_corrected_v2
cd ../aws && IP=$(awk '{print $2}' .current_instance) && rsync -az -e "ssh -i ~/.ssh/icp-flow-aws" --exclude 'sequences/' --exclude '*.ply' --exclude '*.pt' \
  ubuntu@$IP:plantpipeline/experiments/monogrow/runs/gpu_test ubuntu@$IP:plantpipeline/experiments/monogrow/runs/gpu_val_obj ubuntu@$IP:plantpipeline/experiments/monogrow/runs/gpu_test_obj ubuntu@$IP:plantpipeline/experiments/monogrow/runs/ablate_prior_len_gpu ../monogrow/runs/
./stop_all.sh                                     # or terminate_all.sh once results are pulled (EBS ~$16/month per box if only stopped)
```

## Then
1. Fill the `[PENDING: …]` placeholders in experiments/WRITEUP.md from: growfields/runs_aws/merged/*/results*.md (GF 5-seed, GF+monotone, GF+nearest for all three tables), monogrow/runs/gpu_test/RESULTS.md (MonoGrow test, nearest vs first, 3 seeds), monogrow/runs/gpu_test_obj/RESULTS.md (matched objective), monogrow/runs/ablate_prior_len_gpu (prior v2 vs 95k), richards/README.md.
2. Update experiments/RESULTS.md table rows marked pending.
3. Decide the title claim from the numbers (see WRITEUP.md note on "decomposing" vs "beating").
4. Commit the repo (nothing is committed yet; data/ and venvs are git-ignored).
5. Email Bonn (Lasse Klingbeil) about the Pheno4D licence before any publication.

## Known loose ends
- GF's collect_results.py picks the first `results/*` dir; with `_nearest` siblings present pass `--experiment_dir` explicitly.
- growth_metrics.py definitions differ from GF's growth_metrics.py (pooled vs per-plant; nearest references its own source frame); both are reported, do not mix.
- Monotone head guarantees non-decreasing base distance, not Z-extent.
- The 150k GPU prior (runs/full_gpu, no decay) is degraded; use full_cpu95k or the cosine prior.
