# experiments/ — map

Everything needed to reproduce the GrowFields comparison. Each subfolder has its own README/NOTES with full detail;
this file is the index. Draft paper: `WRITEUP.md`. Running results table: `RESULTS.md`. Method candidates and math:
`DESIGN_MEMO.md`.

| path | what |
|---|---|
| `DESIGN_MEMO.md` | research memo: GrowFields' weaknesses, Candidates A (MonoGrow), B (TrajDiff, not built), C (RichardsNet), ablation plan, metrics |
| `RESULTS.md` | the one running results table (paper, repro, nearest, RichardsNet, MonoGrow), findings, decomposition plan, compute log |
| `WRITEUP.md` | paper-style draft (v0.3): headline numbers in for every method, MonoGrow under both fitting objectives, cosine prior on val; `[PENDING: ...]` only for the GF interpolation/extrapolation fleet tables and the last 2 monotone plants (shards gf1/gf3 pending retrieval) |
| `NEXT_STEPS.md` | resume checklist: what is running on which AWS box, morning pull/collect commands, decision list |
| `growfields/` | vendored GrowFields (commit `3f8e947`) + our `monotone.patch`, CPU venv, TrackPlant3D v2 data, run/collect/export scripts, `BENCHMARK.md` (protocol + paper numbers), `NOTES.md` (monotone head, Mac runs) |
| `monogrow/` | Candidate A: shared monotone SIREN prior over all training organ tracks, test-time code fitting, our metric scripts (`README.md`) |
| `richards/` | Candidate C: closed-form Richards curves + empirical-Bayes species prior; imports `monogrow/` for data/canon/metrics (`README.md`) |
| `aws/` | GPU fleet scripts and the instance record (`INSTANCE.md`, `README.md`, `cost_estimate.md`) |

## Data flow

```
data/raw/TrackPlant3D_v2.zip ──unzip──> growfields/data/TrackPlant3D/versions/v2/   ──> GrowFields (train_per_organ.py), 8 test / 4 val plants
                                                                                       ──> monogrow/data.py benchmark tracks (same PLYs + GF point_splits)
data/raw/{Pheno4D,Sugar4D,TrackPlant3D} ──`plantpipeline unify`──> data/unified/{trackplant3d,pheno4d,sugar4d}, sequences.json
                                        ──> monogrow/data.py training tracks (1088; TP3D train split only, Pheno4D minus M01/M02/M04/T01/T05, Sugar4D all)
                                        ──> monogrow/train_prior.py  -> runs/<name>/{ckpt.pt, prior.pt, log.csv}
                                        ──> richards/prior.py        -> prior.npz, prior_table.md
monogrow/fit_test.py | richards/fit_test.py ──> GF-layout PLYs ──> monogrow/metrics.py, growth_metrics.py  (and GF's compute_metrics_*.py unchanged)
```

Leakage rule (`monogrow/config.py`): TrackPlant3D val/test plants never enter a prior; Pheno4D M01, M02, M04, T01, T05
are excluded (same physical plants as TrackPlant3D maize / tomato2).

Venvs: `growfields/.venv` (torch CPU, PSA, chamfer shim; built by `growfields/smoke_cpu.sh` logic) and `monogrow/.venv`
(torch, numpy, scipy, open3d). RichardsNet uses the monogrow venv. GrowFields' own metric scripts need the GF venv and
CWD = `experiments/growfields` (relative `data/TrackPlant3D/versions` path).

## Reproducing each row of `RESULTS.md`

All commands from `experiments/`. `OMP_NUM_THREADS=2` was used for every monogrow/richards run on the Mac.

### GrowFields (paper numbers)
No run; parsed from the arXiv HTML and `REPRODUCIBILITY.md`, tabulated in `growfields/BENCHMARK.md` §2.

### GrowFields reproduction, Mac CPU, 1 seed (row "our repro")
```bash
cd growfields
bash launch_headline.sh                      # detached: run_test_split.sh shared_continuous 1 4 3 ; collect ; run_test_split.sh monotone_continuous 1 4 3 ; collect
# or by hand:
bash run_test_split.sh shared_continuous 1 4 3            # 8 test plants, --n_runs 1 --seed 43, P=4 procs x 3 threads, ~5.8 h/plant
.venv/bin/python collect_results.py shared_continuous     # -> runs/shared_continuous/results.md (CD/EPE per plant + Mean, n_tips 192/189, growth metrics)
```
Progress: `grep -c '^Step' runs/shared_continuous/<seq>_seed43.log` (250 = 50k steps); `tail runs/shared_continuous/jobs.log`.

### GrowFields + monotone head (row "GrowFields + monotone head")
Config `growfields/growfields/configs/per_organ/monotone_continuous.yaml` (= `shared_continuous.yaml` +
`velocity_param: monotone, lambda_w: 0, lambda_g: 0, base_eps: 1e-3, g_bias_init: -2.0`; code in `monotone.patch`).
```bash
cd growfields
bash run_test_split.sh monotone_continuous 1 4 3 && .venv/bin/python collect_results.py monotone_continuous
```
Smoke comparison (200 steps, SoH2): `smoke/run_smoke.sh`; numbers in `NOTES.md` §3.

### GrowFields + nearest-source inference (row "GrowFields + nearest-source", no retraining)
```bash
cd growfields
bash export_nearest_all.sh runs nearest       # every finished <exp> -> <exp>_nearest/ ; variant read from experiment_config.json
# one run:  .venv/bin/python growfields/scripts/export_nearest_source.py --experiment_dir runs/shared_continuous/results/fullplant_batch_shared_continuous_test_aligned --source nearest
.venv/bin/python -I growfields/scripts/compute_metrics_fullplant.py --experiment_dir <exp>_nearest [--interpolation|--extrapolation]
.venv/bin/python -I growfields/scripts/compute_metrics_per_organ.py --experiment_dir <exp>_nearest [--interpolation|--extrapolation]
.venv/bin/python -I growfields/scripts/create_latex_table_fullplant.py --csv <exp>_nearest/fullplant_metrics.csv --full_table --per_sequence
```
`--source first` re-exports GrowFields' own prediction from the saved model (sanity check).

### GrowFields GPU campaign, 5/3/3 seeds, both configs (rows pending)
170 jobs: headline shared/monotone 8 plants x 5 seeds; interpolation (`--train_timepoint_stride 2`) 8 x 3; extrapolation
(`--train_holdout_last_n 3`) 7 x 3 (no SoH2). One process per GPU box, sharded `i mod 3`.
```bash
cd aws
YES=1 NAME=gf1 ./launch.sh; ./sync_data.sh gf1 --code --gf-data; ./setup_env.sh gf1        # repeat for gf2, gf3
./ssh.sh gf1 'cd plantpipeline/experiments/growfields && tmux new-session -d -s train "SHARD=0/3 bash campaign_shard.sh 2>&1 | tee -a runs/campaign.log"'
./status_all.sh                               # jobs done/total, s/step, GPU, MonoGrow prior speed
./pull_results.sh --collect                   # -> growfields/runs_aws/<box>/ and merged -> growfields/runs_aws/merged/<stage>/results.md
# held-out tables (collect_results.py scores all frames): GF's compute_metrics_*.py with --interpolation / --extrapolation on
#   runs_aws/merged/<cfg>_interp/results/fullplant_batch_<cfg>_test_aligned_stride2 and ..._extrap/..._holdlast3
./stop_all.sh | ./terminate_all.sh            # terminate deletes volumes: pull first
```
Record of instance ids, IPs, prices, ETA: `aws/INSTANCE.md`.

### MonoGrow prior (stable config)
```bash
cd monogrow; PY=.venv/bin/python
$PY train_prior.py full train.lr_theta=2e-5 train.lambda_g=1e-3 train.grad_clip=1.0 train.out=runs/full_gpu    # the GPU run (aws/monogrow_gpu.sh)
$PY train_prior.py full train.lr_theta=2e-5 train.lambda_g=1e-3 train.grad_clip=1.0 train.lr_decay=cosine train.out=runs/full_cpu_v2cos   # prior v2 (cosine decay)
```
lr_theta 1e-4 (the config default) diverges at ~5k steps; keep 2e-5. Checkpoints: `runs/full_cpu` (Mac, ~95k steps,
healthy reference), `runs/full_gpu` (AWS, 150k, growth head drifted after ~95k, suspect), `runs/tune/headline/ckpt_step50000`
(50k snapshot used for the val tables). Speed: 0.24–0.30 s/step CPU, 0.085 s/step L4.

### MonoGrow fitting variants (val tables in `monogrow/README.md`, `RESULTS.md`)
```bash
cd monogrow; PY=.venv/bin/python; CK=runs/tune/headline/ckpt_step50000      # or runs/full_cpu for the 95k rows
# nearest vs first, z-only (300 steps), val split          -> runs/nearest/{extrapolation,interpolation}[_first]
$PY fit_test.py full fit.variant=extrapolation fit.split=val fit.source=nearest fit.n_z=300 fit.lr_z=3e-3 fit.lambda_prior=1e-3 fit.ckpt=$CK/ckpt.pt fit.prior=$CK/prior.pt fit.out=runs/nearest/extrapolation
$PY fit_test.py full fit.variant=extrapolation fit.split=val fit.source=first   ... fit.out=runs/nearest/extrapolation_first
# prior length ablation (95k CPU prior)                     -> runs/ablate_prior_len/   (runs/ablate_prior_len/run.sh)
# theta fine-tune sweep, first source                       -> runs/val_ft/              (runs/val_ft/run.sh: nt2000_lt2e-5, nt5000_lt2e-5, nt2000_lt1e-4)
#   add fit.n_theta=2000 fit.lr_theta=2e-5
# GPU val grid with nearest                                 -> runs/gpu_val/<prior>/<variant>/   (PRIOR=full_gpu bash runs/gpu_val/run.sh extrapolation)
# test split, nearest then first re-export of the same fit  -> runs/gpu_test/<variant>_{nearest,first}_seed<k>   (bash runs/gpu_test/run.sh headline 0)
# scoring
$PY metrics.py --experiment_dir <fit.out> [--mode interpolation|extrapolation] [--compare_gf]
$PY growth_metrics.py --experiment_dir <fit.out> [--mode all|interpolation|extrapolation]
# sweeps on the val split: tune_val.py --ckpt_dir runs/full_cpu --variant extrapolation --grid small --resume  -> runs/tune/<variant>/summary.md
```
Other flags: `fit.reuse=<dir>` re-exports fitted codes/theta with another `fit.source`; `model.monotone=false`,
`model.step_per_frame=true` ablations; `fit.gt_fill_single_frame=false`.

### RichardsNet (rows "RichardsNet first / last / nearest")
```bash
cd richards; PY=../monogrow/.venv/bin/python
$PY prior.py                                                   # 66 s -> prior.npz, prior_table.md
./run_variant.sh test headline      richards first|last|nearest   # -> runs/test_headline_richards_<template>/
./run_variant.sh test interpolation richards first|last|nearest
./run_variant.sh test extrapolation richards first|last|nearest
./run_variant.sh test extrapolation richards last _nofill --no_gt_fill_single_frame
$PY collect.py --split test                                    # markdown table over runs/
./gf_check.sh runs/test_headline_richards_nearest sorghum_highlight_plant2      # GF's own scripts on one plant + row diff
```
Val-split family/template/λ ablations: same with `val` and `--family beta`, `--lambda_prior 0.1|10|inf`, `--log_space`, `--no_rank`.

## Where results land

| result | location |
|---|---|
| GF Mac runs | `growfields/runs/<config>/results/fullplant_batch_<config>_test_aligned/sequences/<plant>/…`, `runs/<config>/results.md`, logs `runs/<config>/<plant>_seed43.log`, `jobs.log` |
| GF nearest exports | `<exp>_nearest/` beside each experiment dir (`fullplant_metrics.csv`, `all_metrics.csv`) |
| GF AWS campaign | `growfields/runs_aws/<gf1,gf2,gf3>/`, merged `growfields/runs_aws/merged/<stage>/results/<exp>/` + `results.md` |
| MonoGrow priors | `monogrow/runs/{full_cpu,full_gpu,full_cpu_v2cos,tune/*/ckpt_step*}/{ckpt.pt,prior.pt,log.csv}`; logs `monogrow/runs/logs/` |
| MonoGrow fits | `monogrow/runs/{nearest,ablate_prior_len,val_ft,gpu_val,gpu_test,tune}/…` with `*.metrics.txt`, `*.growth.txt`, `monogrow_summary*.json`, `growth_summary.json` |
| RichardsNet | `richards/runs/<split>_<variant>_richards_<template>[tag]/` (same files), `richards/prior_table.md` |
| AWS logs | `aws/fleet_poll.log`, `aws/*.out`, `aws/INSTANCE.md` |

## Caveats (read before quoting any number)

- Headline CD is a fitting score (test points are other samples of observed surfaces); only interpolation /
  extrapolation measure generalisation in time.
- GrowFields' extrapolation protocol normalises time and space over held-out frames and GT-fills organs that emerge in
  held-out frames; we keep the GT-fill for comparability (`growfields/BENCHMARK.md` §1.2).
- GrowFields' Table 4 "Ours" row is per-organ aggregation (149.63 / 18.99); the full-plant aggregation is 117.71 / 21.79.
  Quote the matching aggregation and the 7-plant mean (no SoH2).
- `nearest` source is an inference-time rule available to every method; GrowFields' tables are `first`. Report both.
- The monotone head guarantees non-decreasing base distance ‖x‖, not Z extent; Z-extent violations after θ fine-tuning
  were 0.105 (GT floor 0.035).
- `growth_metrics.py` in `--mode interpolation|extrapolation` with `first` source reports prediction-vs-GT, not field
  violations (use `--mode all`); RichardsNet `last` template violations are the same artefact.
- MonoGrow prior training is unstable at lr_theta 1e-4 and drifts late even at 2e-5; the constant-lr 150k GPU
  checkpoint is degraded (val interp 392 / 20.0), ~95k CPU is the reference and every test number uses it. The cosine-decay
  prior v2 (`runs/full_cpu_v2cos`, 150k) removes the drift on val (extrap 151.8 / 11.5 vs 160.4 / 13.1 at 95k, interp
  57.5 / 8.3 vs 55.4 / 9.1; `results/monogrow_prior_v2cos_val_eval.txt`) but has not been run on test or with the matched
  objective. θ fine-tune lr must stay ≤ 2e-5.
- `monogrow/runs/tune/*` extrapolation/interpolation rows used a mislabelled step-0 snapshot; ignore them.
- GrowFields headline: Mac reproduction is 1 seed on CPU, all 8 plants (0.88 / 1.53); the AWS L4 fleet is 5 seeds, all
  8 plants (0.87 / 1.53; + nearest 0.44 / 1.02) in `growfields/runs_aws/merged/shared_continuous/results_headline{,_nearest}.md`.
  GF interpolation / extrapolation fleet tables are still in flight: shard gf2 (57 jobs) is pulled, shards gf1 / gf3 are
  pending retrieval (`RESULTS.md`, Overnight results); quote the paper's point estimates for those until then.
- GrowFields + monotone head is PARTIAL: 6 of 8 plants (SoH2, TbC1 missing; TbS3 1 seed). Its 6-plant means (0.97 / 1.81;
  + nearest 0.51 / 1.16) are not comparable to 8-plant means; compare per plant.
- MonoGrow validation numbers are single fits. MonoGrow test numbers are 3 seeds with the 95k CPU prior, under two
  fitting objectives: `fit.objective=first` (GF's multi-step from-first-frame CD; 58.91 / 12.95 headline,
  `monogrow/runs/gpu_test/RESULTS.md` copied into `RESULTS.md`) and `fit.objective=nearest` (one-step nearest-frame CD,
  the quantity scored; **13.94 / 10.70** headline, 41.96 / 10.63 interp, 193.4 / 20.5 extrap on 8 plants;
  `results/monogrow_gpu_test_obj_RESULTS.md`). Always state the objective next to a MonoGrow number.
- Matched objective: it beats RichardsNet-nearest on headline CD (13.9 vs 22.1) but not on EPE (10.7 vs 4.6),
  interpolation (42 vs 25) or extrapolation (193 vs 141); GF + nearest (0.44) stays far ahead on the fit tables; per-organ
  7-plant extrapolation 156.5 / 19.3 vs GF 149.6 / 19.0 (first objective 151.7 / 20.1). The matched objective is a
  fitting change available to any method; GF itself has not been re-fitted with it.
- RichardsNet `last` reproduces the template frame exactly (CD 0 there); not comparable to GF's frame-0 rollout.
- Maize in every prior comes from 15 Pheno4D tracks with noisy labels; sugar beet is 65 % of tracks (species-balanced sampling).
- ε_ψ residual MLP, saturation gate, densification, emergence generation, Candidate B: not implemented.
