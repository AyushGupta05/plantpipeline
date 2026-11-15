# Smoke test on our converted data (2026-11-15, this Mac, CPU, OMP_NUM_THREADS=4, one process at a time)

Pipeline check only: GrowFields trained for **200 of 50,000 steps** (lr 5e-6), so its CDs are far from converged and
say nothing about the method. Headline protocol (all frames observed, GF's fixed 50/50 point split, seed 43), full-plant
and per-organ **cd_test in mm^2** (GF aggregation: frames -> plant; per organ with the >= 2 scored-frames filter).
EPE is NaN and n_tips 0 everywhere (no tip annotations in Pheno4D / Sugar4D). `monogrow/metrics.py --compare_gf`
matched GF's `fullplant_metrics.csv` / `all_metrics.csv` row for row (max |diff| 0.00e+00) on every run below.

| method | pheno4d_maize_M01 (12 fr, 4 organs) full / organ CD | sugar4d_sugar_beet_plant_002 (16 fr, 19 organs) full / organ CD | fit wall |
|---|---|---|---|
| GrowFields shared_continuous, 200 steps, first source | 3646.57 / 1957.61 | 554.16 / 371.91 | 51 s / 47 s (incl. 9 s dataset load, export) |
| same model, nearest source (`export_nearest_source.py`) | 1229.47 / 660.78 | 230.28 / 187.07 | export + score 34 s for both |
| RichardsNet, template nearest (prior.npz) | 1408.21 / 818.89 | 247.97 / 220.63 | 1 s each |
| MonoGrow z-only, 95k prior, objective=source=nearest | 1173.76 / 659.95 | 243.45 / 214.22 | 153 s / 2283 s |

Leakage: M01 is one of the five Pheno4D plants held out of the MonoGrow/RichardsNet priors (fair); plant_002 was in the
priors' training set (every Sugar4D plant was), so its baseline rows are train-set numbers. GF has no prior.

## Timing

| | pheno4d_maize_M01 | sugar4d_sugar_beet_plant_002 |
|---|---|---|
| GF `training_time_seconds` for 200 steps (1 eval pass incl.) | 39.6 s | 35.8 s |
| => s/step on CPU, 1 process, 4 threads | **0.198** | **0.179** |
| => 50k steps | 2.75 h | 2.5 h |
| whole `run_gf.sh` wall (load tree 9 s + train + export + normals) | 51 s | 47 s |
| MonoGrow fit (1000 z-steps, 1024 pts/organ) | 153 s | 2283 s (18 organ tracks) |

Sugar4D's 19 organs do not slow GF's step: `organs_per_step=10` samples 10 organs per step (8,192 points split over them,
min 256/organ); only the eval pass every 200 steps and the export scale with organs x frames. Reference: 0.138 s/step on
TP3D SoH2 (NOTES.md §3), 0.033-0.042 s/step on an L4 (campaign_shard.sh).

## Files

- GF: `runs/shared_continuous/results/fullplant_batch_shared_continuous_ours_aligned/` (+ `_nearest/`), logs in `runs/shared_continuous/logs/`
- RichardsNet: `runs/richards_nearest/results/fullplant_batch_richards_nearest_ours/`
- MonoGrow: `runs/monogrow_nearest/results/fullplant_batch_monogrow_nearest_ours/`
- each experiment dir: GF `fullplant_metrics.csv`, `all_metrics.csv`, `monogrow_{fullplant,per_organ}.csv`, `monogrow_summary.json`; `runs/jobs.log` has one line per job.

Wire check (20 steps, M01, meaningless CDs): interpolation first/nearest 5259 / 1269 full-plant CD on the odd frames, extrapolation
10670 / 2822 on the last 3 frames (`runs/shared_continuous_{interp,extrap}/`). Not exercised: `monotone_continuous`, Pheno4D tomato
plants (26-41 leaf ids), GPU.
