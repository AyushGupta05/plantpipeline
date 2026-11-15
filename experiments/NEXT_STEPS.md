# Next steps (updated 2026-11-15)

State: every planned run is complete and pulled; all four AWS boxes were stopped on 31 Oct (terminate them if not already done). The write-up is at v1.0 draft (experiments/WRITEUP.md) with no placeholders. The repo is committed; `git push` is Ayush's step. Backup archives for Google Drive are in `backup/` (git-ignored).

## The one prepared but un-run experiment: GrowFields on our other datasets
Folder: `experiments/gf_on_ours/` (converter from `data/unified` to GrowFields' PLY layout, point splits, run/score scripts, smoke-tested on CPU at 200 steps). Purpose: test whether the ranking found on TrackPlant3D (GrowFields vs +nearest-source vs RichardsNet vs MonoGrow) holds on full-resolution Pheno4D (maize, tomato; 14 plants) and Sugar4D (sugar beet; 48 plants, 16 scans each).

Cost (GrowFields at 50k steps per plant): see `experiments/gf_on_ours/README.md` for the measured s/step; on an L4 (g6.xlarge, ~$0.8/h) expect ~25–40 min per plant, i.e. ~30–40 GPU-hours for all 62 plants × headline, more for interpolation/extrapolation; RichardsNet and MonoGrow fits on the same plants are minutes.

When compute is available:
1. `cd experiments/aws && YES=1 ./launch.sh gf1` (quota L-DB2E81BA is 32 vCPU; three other instances belong to other projects) — or reuse a stopped box with `aws ec2 start-instances`.
2. `./sync_data.sh --code gf1` plus rsync `experiments/gf_on_ours/data` to the box; run `experiments/gf_on_ours/run_gf.sh` per plant in tmux (P=1 per GPU), then `score.sh` and `run_baselines.sh`.
3. Pull with rsync, score locally, add a "cross-dataset" table to RESULTS.md and WRITEUP.md §4.

## Smaller follow-ups
- MonoGrow: longer stage-2 θ fine-tune on the matched one-step objective (the fit-table gap is on monocots); evaluate the cosine prior (results/priors/ckpt_v2cos150k.pt) with the matched objective on test.
- Re-fit GrowFields itself on the one-step (nearest) objective for a fully matched comparison.
- Email Lasse Klingbeil (Bonn) about the Pheno4D licence before any publication.

## Known loose ends
- GF's collect_results.py picks the first `results/*` dir; pass `--experiment_dir` explicitly when `_nearest` siblings exist.
- growth_metrics.py definitions differ from GF's own growth script; report both, do not mix.
- The monotone head guarantees non-decreasing base distance, not Z-extent.
- The 150k no-decay GPU prior is degraded; use the 95k prior or the cosine prior.
