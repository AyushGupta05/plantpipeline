# MonoGrow (Candidate A of `experiments/DESIGN_MEMO.md`)

One SIREN velocity field shared across all training plant sequences, a monotone-growth head
(`v = softplus(g) x_hat + P_perp(x) w`, so every point's distance to the organ base never decreases), per-organ
auto-decoded codes `z`, a species-conditional Gaussian prior `q(z | species)`, and test-time code fitting evaluated
on the GrowFields benchmark with GrowFields' own metric scripts. Pure numpy/torch; standalone under this folder
(`plantpipeline/` and `experiments/growfields/` are untouched).

```
config.py          dataclasses: DataConfig / ModelConfig / TrainConfig / FitConfig; `smoke` and `full` presets; `a.b=c` overrides
canon.py           numpy port of PSA's 4-stage leaf canonicalisation (+ stem convention), per-frame 4x4 transforms
ply.py             binary PLY reader/writer in GrowFields' field layout (organ_label, is_train, is_leaf_tip, tip_organ_label)
data.py            organ tracks from data/unified (training, leakage rule) and from TrackPlant3D v2 PLYs + GF point splits (benchmark); sampler
model.py           SIREN trunk, monotone head (flag), 1-day Euler integrator (flag: GF one-step-per-frame), chunked squared Chamfer
train_prior.py     auto-decoding training of theta + all z; fits q(z | species); log.csv, ckpt.pt, prior.pt
fit_test.py        z-only (+ optional theta) fitting per test organ, rollout, GT re-posing, PLY export in GF's layout
metrics.py         self-contained CD / tip-EPE / aggregation (frames -> organs -> plants), n_tips check, `--compare_gf`
growth_metrics.py  monotonicity violation rate, negative-growth mass, organ-length MAE (seen / held-out / per horizon)
tests.py           parity checks that need the GrowFields+PSA venv (canonicalisation vs PSA; PLY vs growfields load_ply)
```

## Environment

```bash
cd experiments/monogrow
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu   # CUDA wheel on a GPU box
uv pip install --python .venv/bin/python "numpy<2.4" scipy open3d      # scipy: cKDTree in stage 3; open3d: normals for unified data
```
The code picks CUDA when available, nothing else changes. Data: `data/unified` (built by `plantpipeline unify`) for the
prior; the ETH TrackPlant3D v2 PLYs and GrowFields' `point_splits` for the benchmark, found at the paths in
`DataConfig` (the GrowFields smoke checkout under the session scratchpad; override with `data.tp3d_dir=... data.point_splits=...`).
The GrowFields metric scripts are run from their own venv (`experiments/growfields/smoke_cpu.sh` builds it).
All runs below were made with `OMP_NUM_THREADS=2` (the Mac was running other jobs); `train.threads` / `fit.threads` default to 2.

## Commands

```bash
PY=.venv/bin/python
# 1. prior, CPU smoke (20 organ tracks, 300 steps, 4 organs x 512 pts / step)      -> runs/smoke/{log.csv,ckpt.pt,prior.pt}
$PY train_prior.py smoke
# 1'. prior, full (1088 tracks, 150k steps, 10 organs x 819 pts, 1-day grid, horizon <= 14 d) -> runs/full/
$PY train_prior.py full                      # add e.g. train.steps=50000 train.out=runs/full50k
# 2. test fitting (z <- mu_species, N_z Adam steps on observed frames' train points, roll out, re-pose, write PLYs)
$PY fit_test.py smoke                                                             # SoH2 only, 100 z-steps
$PY fit_test.py full fit.variant=headline      fit.out=runs/full/fit_headline     # 8 test plants, all frames
$PY fit_test.py full fit.variant=interpolation fit.out=runs/full/fit_interp       # frames 0,2,4,... observed
$PY fit_test.py full fit.variant=extrapolation fit.out=runs/full/fit_extrap       # all but the last 3 frames observed
#    options: fit.n_z=1000 fit.lr_z=1e-2 fit.lambda_prior=1e-3 fit.n_theta=2000 fit.lr_theta=1e-6 fit.split=val
#             fit.gt_fill_single_frame=false (predict organs seen once from the prior instead of GF's GT fill)
#             model.step_per_frame=true (GF's one Euler step per frame gap instead of the 1-day grid)
#             fit.source=nearest (predict each frame from the latest observed frame, see "Rollout source")
#             fit.reuse=<other fit.out> (re-export that run's fitted codes / theta with another fit.source; no fitting)
#             fit.lambda_w=1e-3 (penalise the radial term ||P_perp w||^2 during fitting)
# 3. metrics, ours (CSV + summary json next to the PLYs); --mode interpolation|extrapolation for held-out frames
$PY metrics.py --experiment_dir runs/full/fit_headline [--compare_gf]
$PY growth_metrics.py --experiment_dir runs/full/fit_extrap            # monotonicity, length MAE per horizon
# 4. metrics, GrowFields' scripts unchanged (GF venv; CWD must be the GF work dir: compute_metrics_per_organ.py
#    hard-codes data/TrackPlant3D/versions relative to CWD). EXP = any fit.out above.
cd <scratchpad>/gf_smoke; GF=../gf_clone/scripts; V=.venv/bin/python
$V -I $GF/compute_metrics_fullplant.py --experiment_dir $EXP [--interpolation|--extrapolation]
$V -I $GF/compute_metrics_per_organ.py --experiment_dir $EXP [--interpolation|--extrapolation]
$V -I $GF/create_latex_table_fullplant.py  --csv $EXP/fullplant_metrics.csv --full_table --per_sequence
$V -I $GF/create_latex_table_per_organ.py  --csv $EXP/all_metrics.csv --full_table --per_sequence --min_timepoints_per_organ 2
$V    $GF/create_latex_table_heldout.py --metrics_training_times $EXP/all_metrics.csv --metrics_interp_times $EXP/all_metrics_extrapolation.csv \
      --experiment_dir $EXP --metric_type per_organ --per_sequence --full_table --min_timepoints_per_organ 2 --extrapolation   # (no -I: imports siblings)
# 5. parity tests (GF venv)
$V -I /path/to/monogrow/tests.py canon tomato1_shade_plant1
$V -I /path/to/monogrow/tests.py ply $EXP/sequences/<plant>/full_plant/predicted/pred_<plant>_D06.ply
```

## What is implemented (and the decisions that are not in the memo)

* **Canonicalisation** (`canon.py`): PSA's `correct_pca_stem -> align_to_z -> z_rotation_normals -> align_base` ported
  line by line; the composed per-frame 4x4 is stored and inverted for re-posing. Verified against
  `plant_shape_analysis.LeafSequencesDataset(apply_alignment=True)` on SoH2, MaC2, Tm1S1: max |d points| 7.6e-6 mm
  (float32 storage), |d matrix| 1.4e-14, tips identical. Stems: per-frame translation of the lowest-1 % centroid to
  the origin, no rotation (so `x_hat` points away from the stem base; GF keeps stems in plant space).
  Unified npz carry no normals, so for Pheno4D/Sugar4D stage 3 uses Open3D normals oriented to +Z
  (PSA's estimator without the tangent-plane pass, whose global sign flips between frames). TrackPlant3D training
  plants also go through this path (unified data) - their v2 normals are only used for benchmark plants.
* **Tracks / leakage** (`data.py`): 1088 training organ tracks (>= 2 frames, >= 3 points per frame as in PSA):
  TrackPlant3D train split 31 plants (tomato 146, sorghum 44, tobacco 36; TP3D maize is entirely val/test),
  Pheno4D minus M01/M02/M04/T01/T05 (maize 15, tomato 137), Sugar4D all 48 plants (710). TP3D val/test never
  enter the prior. Because sugar beet is 65 % of tracks, sampling is species-balanced (`train.balance_species`).
  Cached as `cache/tracks_<hash>.pkl` (16 s to build).
* **Clocks**: organ age `a = day - first observed day of the organ`, plant age `p = day - first scan day`, both in
  days, normalised by 60 for the network input; never by a held-out range.
* **Model** (`model.py`): input `[x/100 mm, a/60, z (32), species (8), p/60]`; SIREN with GrowFields' convention
  (first sine layer + 5 hidden sine layers of 256, omega 30, GF init) and a 4-d linear head (g, w):
  **341,804 parameters** + 32 per organ. Velocity in (100 mm)/day; `eps = 1e-3` guard on `||x||`.
  Integration: forward Euler on a 1-day grid through every target day (`model.dt_days`), or GF's one step per frame
  gap (`model.step_per_frame=true`). `model.monotone=false` gives the plain 3-d head.
* **Prior training** (`train_prior.py`): per step K organs (points split equally, min 256), 50 % consecutive-frame
  pairs else random `(s, k > s)` within 14 days (Sugar4D spans 56 days), multi-step squared Chamfer averaged over
  frames `s+1..k` (GF divides by the number of frames; the memo writes a sum), computed in (100 mm)^2 units,
  + `lambda_z * mean_i ||z_i||^2` + optional `lambda_w`, `lambda_g`. Adam, lr 1e-4 (theta, incl. species embedding)
  / 1e-3 (z). After training `q(z | s) = N(mu_s, Sigma_s)` with `Sigma_s` shrunk towards the pooled covariance
  (5 pseudo-organs) + 1e-3 ridge; species with no organs get the pooled Gaussian. `prior.pt` holds mu, full cov,
  diagonal cov and counts.
* **Test fitting** (`fit_test.py`): observed frames = all / every 2nd global frame / all but the last 3; only the
  GF point split's `train_indices` are used for fitting; z initialised at `mu_species`, Adam (default lr 1e-2,
  1000 steps) on multi-step CD over the observed frames rolled out from the organ's first observed frame,
  + `lambda_prior (z-mu)^T Sigma^-1 (z-mu)`; all organs of a plant are fitted jointly (one loss per step).
  Stage 2 (`fit.n_theta>0`) fine-tunes theta (reset between plants). Export: all points (train+test) of the first
  observed frame are advected to every later frame of the organ (1-day grid), `is_train` inherited; the GT tip of
  the organ's first tip-annotated frame is advected from that frame (GF does the same); re-posed with the GT
  transform of the target day; tips snapped to the nearest predicted point exactly as GF's writer does.
  Organs with < 2 observed frames get no model and are copied from GT in the full-plant files
  (GF behaviour; `fit.gt_fill_single_frame=false` predicts them from the prior). Frames the variant held out go to
  `full_plant/<mode>/` and `organs/<organ>/<mode>/` like GF's interpolation/extrapolation exports; observed frames to
  the normal dirs. `experiment_config.json` and `results.json` are written so GF's scripts find the dataset and the
  timing columns.
* **Metrics** (`metrics.py`): CD on test points, tip EPE matched by organ id skipping each tip's first-appearance
  frame (frame 0 included), aggregation frames -> organs -> plants -> unweighted mean, organs with < 2 scored rows
  dropped (GF's table scripts do this for the training-frame, interpolation and extrapolation CSVs alike), n_tips
  totals. `--compare_gf` diffs every row against GF's CSVs when present.
* **Growth metrics** (`growth_metrics.py`): length = 99th-percentile canonical Z; violation = decrease > 2 mm between
  consecutive predicted frames (the rollout's source frame counts as the first); negative growth mass; length MAE
  for observed / held-out frames and per extrapolation horizon (+1/+2/+3); GT violation rate as the floor.

## Rollout source (`fit.source`)

`fit.source` is an inference-time option of `fit_test.py` (`growth_metrics.py` reads it from `results.json`; both were
developed as `fit_test2.py` / `growth_metrics2.py` while a sweep was running and folded back on 2026-10-30 12:05 UTC).
Fitting (z, optional theta) is untouched in both modes: multi-step CD over the
observed frames rolled out from the first observed frame, train points only (the per-plant fit losses are
bit-identical between the two sources). Only where the exported prediction of a target frame `t` starts changes:

* `first` (default, GrowFields' protocol): from the organ's first observed frame, every later frame is one long rollout.
* `nearest`: from the latest observed frame `s < t` of that organ. Headline and interpolation: `s = t-1`, i.e. one-day
  predictions (interpolation: an odd frame from the even frame before it); extrapolation: the last observed frame for
  all three held-out frames (horizons +1/+2/+3 days). The organ's first frame is never scored (GF). Source points = all
  points of frame `s`, its `is_train` mask inherited by the PLY (so test points are still scored); integration on the
  1-day grid from `day(s)` to `day(t)`; re-posed with the GT transform of day `t`; GF layout unchanged, so GF's
  `compute_metrics_fullplant.py --interpolation|--extrapolation` runs on the output (verified: identical rows to
  `metrics.py`). A source frame with < 3 train points gives no prediction for its targets (GT fill in the full-plant
  files, no per-organ row), the same rule as organs seen < 2 times. Organs whose first observed frame is later than `t`
  are GT-filled as before.
* **Tip** for `nearest`: the GT tip of the source frame `s` if it is annotated there; else the tip of the latest earlier
  annotated frame `a < s`, advected from `a` through `t` with the same field; else (no annotated tip at or before `s`)
  GF's rule, the first annotated frame after `s` even if held out. The last fallback keeps `n_tips` identical to `first`
  (36 extrapolation / 41 interpolation tip rows on the 4 val plants in both modes); GF's EPE skips a tip's
  first-appearance frame anyway. `first` keeps the current behaviour exactly (first annotated frame >= source).
* `growth_metrics.py` compares each `nearest` prediction with the GT length of its own source frame (`results.json`
  carries `"source"`), so dL / violation rows measure one rollout each instead of a chain of predictions from different
  sources. Caveat for both scripts in `--mode extrapolation|interpolation` with `first`: the chain starts at the GT length
  of the frame before the first held-out prediction, while that prediction was rolled out from the first frame, so the
  "violation" there is prediction-vs-GT error, not a field monotonicity violation (hence 0.23 below; `--mode all` gives 0).

Validation plants, `runs/tune/headline/ckpt_step50000` (50k-step prior), z only (`fit.n_z=300 fit.lr_z=3e-3
fit.lambda_prior=1e-3`), `OMP_NUM_THREADS=2`; held-out frames only, full-plant / per-organ CD (mm^2) and tip EPE (mm)
(`runs/nearest/{extrapolation,interpolation}[_first]`). The earlier `runs/tune/*/fits/nz300_lr0.003_lp0.001_nt0` rows
used a different (step-0-labelled, 02:04) snapshot and are not comparable.

| extrapolation (+1..+3 d) | first: full CD / EPE | first: organ CD / EPE | nearest: full CD / EPE | nearest: organ CD / EPE |
|---|---|---|---|---|
| maize_control_plant1 | 1379.6 / 52.0 | 946.9 / 71.0 | 252.1 / 16.1 | 200.5 / 22.4 |
| sorghum_control_plant1 | 705.1 / 47.0 | 553.2 / 53.4 | 435.8 / 18.4 | 337.1 / 23.9 |
| tobacco_control_plant2 | 46.4 / 13.5 | 54.3 / 12.5 | 21.1 / 7.4 | 28.3 / 7.1 |
| tomato2_control_plant3 | 21.5 / 9.6 | 25.9 / 10.0 | 12.4 / 6.5 | 17.8 / 6.2 |
| **Mean** (n_tips 36 / 36) | **538.2 / 30.5** | **395.1 / 36.7** | **180.3 / 12.1** | **145.9 / 14.9** |
| length MAE held-out, +1/+2/+3 d | 28.2 mm; 22.1 / 29.2 / 33.2 | | 12.9 mm; 6.2 / 14.3 / 18.3 | |

| interpolation (odd frames) | first: full CD / EPE | first: organ CD / EPE | nearest: full CD / EPE | nearest: organ CD / EPE |
|---|---|---|---|---|
| maize_control_plant1 | 1276.8 / 26.9 | 1341.4 / 43.3 | 77.3 / 9.9 | 82.1 / 15.9 |
| sorghum_control_plant1 | 715.0 / 30.9 | 596.0 / 31.9 | 151.6 / 14.6 | 117.1 / 19.0 |
| tobacco_control_plant2 | 41.2 / 8.6 | 43.2 / 8.4 | 13.4 / 4.2 | 14.5 / 4.4 |
| tomato2_control_plant3 | 12.5 / 6.5 | 14.9 / 8.1 | 7.8 / 3.8 | 10.1 / 4.1 |
| **Mean** (n_tips 41 / 39) | **511.4 / 18.3** | **498.9 / 22.9** | **62.5 / 8.1** | **56.0 / 10.8** |
| length MAE held-out | 15.6 mm | | 6.9 mm | |

Both modes read the same fitted codes (`fit_test.py` saves them per plant as `sequences/<plant>/fit_state.pt` with the
fine-tuned theta; `fit.reuse=<fit.out of the nearest run>` re-exports them with `fit.source=first`, verified bit-identical
to a fresh `first` fit on the smoke config); `nearest` is the natural protocol for a growth *field* (predict the next scan
from the latest one) and removes the accumulated drift of a 5-16 day rollout from frame 0. GrowFields' tables are
`first`; report `nearest` alongside, never instead.

## Verified on this Mac (Apple M5 Pro, 2 torch threads, CPU torch 2.15, 2026-10-30)

| step | result |
|---|---|
| canonicalisation parity with PSA (SoH2 19, MaC2 36, Tm1S1 76 leaf frames) | max dpoints 7.6e-6 mm, dM 1.4e-14, dtip 0 |
| PLY round trip through `growfields.utils.io_utils.load_ply` | identical points/labels/is_train/is_leaf_tip/tip_organ_label |
| `train_prior.py smoke` (20 tracks, 300 steps, 4 x 512 pts) | 18 s, 0.059 s/step; CD 37,406 -> 55 mm^2 (batch) |
| `train_prior.py full train.steps=3000 train.points_per_step=4096` (1088 tracks) | 410 s, 0.137 s/step; batch CD ~140-400 mm^2 at the end, still falling |
| full config step on CPU (10 organs x 819 pts, horizon 14 d) | 0.25 s/step -> 150k steps ~ 10 h CPU; GPU estimate 1-3 h [unverified] |
| `fit_test.py smoke` (SoH2, 6 organs, 100 z-steps, 512 pts) | 20 s; export ok |
| GF `compute_metrics_fullplant/per_organ.py` + `create_latex_table_*.py` on our output (headline, interpolation, extrapolation modes) | run unchanged; per-row CD identical to ours (max diff 0), EPE max diff 8e-6 (float32), n_tips identical; aggregates identical to GF's LaTeX tables (e.g. SoH2 smoke 619.20 / 32.66 full, 1377.72 / 27.95 per organ; MaC3 extrapolation 2276.97 / 88.59) |
| n_tips on SoH2 | 11 full plant / 10 per organ, same as the GF smoke run (192/189 need all 8 plants) |
| monotonicity (SoH2 headline, SoH2 interpolation, MaC3 extrapolation; smoke and 3k priors) | predicted violation rate 0.000 and negative-growth mass 0.0 mm in every run; GT floor 4-12 % / 5.5-11.3 mm |
| SoH2 fit with the 3k-step prior, z only 300 steps (107 s) | fit loss CD 222 -> 142; full-plant CD 93.1 mm^2 / EPE 20.6 mm; per-organ 90.4 / 21.7; length MAE 14.6 mm |
| same + stage 2 (200 theta steps, lr 1e-5) | full-plant CD 3.97 / EPE 6.23; per-organ 4.90 / 6.66; length MAE 3.7 mm; still 0 violations (GF: 0.74 / 0.97 after 50k steps per plant) |
| MaC3 extrapolation with the 3k-step prior, z only | observed frames 75.9 / 26.0; held-out 1286 / 113 (GF 321 / 70 with 50k steps); length MAE +1/+2/+3 d: 43 / 68 / 79 mm |
| ablation flags `model.monotone=false`, `model.step_per_frame=true` | train and run (20 steps each) |

Numbers with the 3k-step CPU prior are pipeline evidence only (0.4 % of the planned steps, no tuning).

## Open issues / unverified

* The full prior has not been trained; no GPU run of any script (CUDA path is the standard `.to(device)`, untested).
  Hyper-parameters (lr, lambda_z, lambda_prior, lr_z at test time, horizon cap, species balancing, stage-2 lr) are
  the memo's defaults or first guesses; tune with `fit.split=val`.
* The GrowFields smoke numbers to compare against are in `experiments/growfields/BENCHMARK.md`; the 5-seed GF
  reproduction is another agent's job. `fit_test.py` writes a single run (`run` = 1 in the CSVs); seeds via `fit.seed`.
* Maize in the prior comes only from Pheno4D (TP3D maize plants are all val/test), with Pheno4D's "collar"
  labelling (young leaves carry the stem label). Sugar beet dominates the track count; species balancing is on.
* Stem canonical frame (translation only, base = lowest 1 %) is our convention; GF models stems in plant space.
  Stem thickening is handled by the same radial term (x_hat from the base), not the cross-section variant of the memo.
* Interpolation: an organ whose first observed frame is later than an unobserved frame where it exists is not
  predicted backwards; GF also GT-fills there. Tip start frame follows GF (first tip-annotated frame, even if held out).
* Loss averages the multi-step CD over frames (GF) rather than summing (memo). Saturation gate, densification,
  hypernetwork variant and MetaSDF inner loop are not implemented.
* `metrics.py` reproduces GF's *single-run, per_sequence* aggregation; multi-run std and the `run_<k>` directory
  layout are not produced (write different `fit.out` per seed and average outside).

## Tuning (validation split)

`tune_val.py` sweeps the test-time fitting knobs on the 4 GrowFields **validation** plants (maize_control_plant1,
sorghum_control_plant1, tobacco_control_plant2, tomato2_control_plant3; `fit.split=val`, test plants never touched)
and scores every grid point with `metrics.py` and `growth_metrics.py`:

```bash
OMP_NUM_THREADS=2 $PY tune_val.py --ckpt_dir runs/full_cpu --variant headline      --grid small --resume   # -> runs/tune/headline/
OMP_NUM_THREADS=2 $PY tune_val.py --ckpt_dir runs/full_cpu --variant extrapolation --grid small --resume   # -> runs/tune/extrapolation/
OMP_NUM_THREADS=2 $PY tune_val.py --ckpt_dir runs/full_cpu --variant interpolation --grid full  --resume   # 60 configs
#   --wait_minutes 30          poll for ckpt.pt + prior.pt (a training run still in progress)
#   --override fit.points_per_organ=512 fit.sequences=maize_control_plant1    extra fit_test overrides (cheap probes)
#   --ckpt_step N              if the training step cannot be read from <ckpt_dir>/log.csv or runs/logs/<name>.log
```

* Grid (`--grid full`, 60 configs): `n_z` {300, 1000} x `lr_z` {3e-3, 1e-2} x `lambda_prior` {0, 1e-3, 1e-1} x stage 2
  {off, (n_theta, lr_theta) in {500, 2000} x {1e-6, 1e-5}}. `--grid small` = 6 configs: the defaults
  (300 / 1e-2 / 1e-3, z only), one change at a time along each axis (lr_z 3e-3; lambda_prior 0; lambda_prior 1e-1;
  n_z 1000) and one stage-2 run (500 theta steps, lr 1e-5).
* The checkpoint is snapshotted to `runs/tune/<variant>/ckpt_step<N>/` first (train_prior.py rewrites ckpt.pt / prior.pt
  every `ckpt_every` steps; a half-written file is retried), so every config of one invocation uses the same prior
  and every row of `results.csv` carries `ckpt_step`. Fits go to `runs/tune/<variant>/fits/<config>/` (+ `.log`),
  overwritten on re-run; `results.csv` keeps one row per (config, ckpt_step), `summary.md` ranks the rows per
  checkpoint step by full-plant CD (headline) or held-out full-plant CD (interpolation / extrapolation).
  `--resume` skips configs already scored for the current step, so the same command can be re-run as the prior improves.
* Columns: observed-frame full-plant / per-organ CD and tip EPE, held-out ones for interpolation / extrapolation,
  monotonicity violation rate (+ GT floor), negative-growth mass, length MAE on observed / held-out frames and per
  extrapolation horizon (+1/+2/+3 d), fit seconds (sum of the 4 plants) and wall seconds (fit + scoring).

### Theta fine-tuning sweep on the Mac (val extrapolation, `first` source, 50k-step prior `runs/tune/headline/ckpt_step50000`)

`runs/val_ft/run.sh`: z only (300 steps, lr 3e-3, lambda_prior 1e-3) then stage 2 on theta; held-out frames only, 4 val plants,
`OMP_NUM_THREADS=2`, ~0.3 s/step on the loaded Mac (nt2000: 47 min, nt5000: 108 min for the 4 plants).

| stage 2 | held-out full CD / EPE | held-out organ CD / EPE | maize / sorghum / tobacco / tomato2 full CD | viol. rate (GT) | len MAE held-out; +1/+2/+3 d |
|---|---|---|---|---|---|
| none (z only, from the table above) | 538.2 / 30.5 | 395.1 / 36.7 | 1379.6 / 705.1 / 46.4 / 21.5 | - | 28.2; 22.1 / 29.2 / 33.2 |
| 2000 steps, lr 2e-5 | **323.3 / 31.1** | 268.9 / 38.5 | 704.2 / 513.9 / 60.0 / 15.0 | 0.105 (0.035) | 18.7; 12.4 / 19.9 / 23.9 |
| 5000 steps, lr 2e-5 | 453.7 / 27.7 | 335.7 / 34.5 | 1294.8 / 485.1 / 27.3 / 7.5 | 0.088 (0.035) | 20.3; 15.8 / 21.6 / 23.6 |
| 2000 steps, lr 1e-4 | 273,675 / 391.9 | 425,566 / 391.8 | 130k / 357k / 608k / 7.4 | 0.000 | 582; 488 / 581 / 678 |

Theta fine-tuning at 2e-5 helps (CD -40 % at 2000 steps; 5000 steps over-fits maize again while tobacco / tomato2 keep
improving); lr 1e-4 diverges on 3 of 4 plants (the same instability as prior training at 1e-4), so the GPU grid uses
lr_theta in {1e-5, 2e-5} x n_theta in {1000, 2000, 5000}. The "violation rate" column is the `first`-mode caveat above
(prediction-vs-GT), not a field monotonicity violation. GPU-prior grids with `nearest`: `runs/gpu_val/summary.md`; test split: `runs/gpu_test/RESULTS.md`.

## Objective (`fit.objective`)

Until 2026-10-30 22:00 UTC the per-organ code `z` (and theta in stage 2) was always fitted on GrowFields' objective, the multi-step CD
of one rollout from the organ's first observed frame through all observed frames, while the exported predictions used
`fit.source=nearest` (one-day steps from the latest observed frame): the fitting objective and the quantity scored did not match.
`fit.objective` picks the fitting loss (`fit_test.py`, `fit_codes`):

* `first` (default = previous behaviour, bit-identical sampling order): mean over observed frames `s+1..k` of CD(rollout from the
  first observed frame, frame) + `lambda_prior (z-mu)^T Sigma^-1 (z-mu)`.
* `nearest`: mean over consecutive observed pairs `(s, t)` of CD(frame `s`'s train points rolled out on the 1-day grid to `day(t)`,
  frame `t`'s train points), i.e. exactly what `source=nearest` is scored on (interpolation: 2-day pairs). Same chunked Chamfer,
  prior term, `lambda_w`, optimiser and steps; a mean over pairs (not a sum) so `lambda_prior` keeps its scale relative to `first`.
  Export and metrics are untouched; `fit.reuse` works across objectives.

Smoke on the Mac (tomato2_control_plant3, extrapolation, 95k CPU prior, 20 z-steps, `source=nearest`): fit loss 25.5 (`first`) vs
7.1 mm^2 (`nearest`, one-day CD is smaller by construction); held-out full CD 10.7 vs 17.3 after only 20 steps (not a result).

GPU box, prior `runs/full_cpu95k`, `source=nearest`, 4 val plants, launched 2026-10-30 22:01 UTC in tmux `obj` as one self-contained chain
(`runs/gpu_val_obj/{run.sh,chain.sh,decide.py,summarize.py}`, logs `runs/gpu_val_obj/{headline,interpolation,extrapolation,chain}.log`):
1. val grid per variant (headline / interpolation / extrapolation, three tmux windows in parallel): objective in {first, nearest} x
   {nt0 (z only, n_z 300, lr_z 3e-3, lambda_prior 1e-3), nz1000 (z only, 1000 steps), nt1000 (+ 1000 theta steps, lr 1e-5)}
   -> `runs/gpu_val_obj/summary.md` (both objectives side by side, per plant + mean).
2. `decide.py`: per variant the best `nearest` config must beat the best `first` config on val mean full-plant CD (held-out frames;
   all frames for headline) -> `runs/gpu_val_obj/decision.md`, `winners.txt`.
3. for each winning variant: test split, 3 seeds, `runs/gpu_test/run.sh` with `OUT=runs/gpu_test_obj RECIPE=<winning config>
   fit.objective=nearest` (also re-exports the `first`-source PLYs of the same codes), then `runs/gpu_test/summarize.py ... runs/gpu_test_obj`
   -> `runs/gpu_test_obj/RESULTS.md`. "=== CHAIN ALL DONE" in `chain.log` marks the end.
First numbers (22:12 UTC, z only, held-out full CD / EPE, organ CD / EPE): extrapolation `first` 160.35 / 13.10, 136.69 / 15.49 (identical
to `runs/gpu_val/summary.md` nt0, so `objective=first` reproduces the old fitting exactly) vs `nearest` 148.84 / 12.81, 131.49 / 15.66;
interpolation `first` 55.36 / 9.06, 54.00 / 11.35 vs `nearest` 52.76 / 9.56, 49.93 / 11.74. The matched objective lowers CD by 5-7 %
at z-only; EPE is flat. Expected: val grid ~22:50 UTC (3 variants x 6 configs, GPU shared with `prior2`), test runs ~15 min per z-only variant / ~35 min per
nt1000 variant and seed-triple; everything done by ~2026-10-31 00:30 UTC at the latest.

Pull the results (from `experiments/aws`, the chain does not need the Mac):
```bash
IP=$(awk '{print $2}' .current_instance)
rsync -az -e "ssh -i ~/.ssh/icp-flow-aws" --exclude 'sequences/' --exclude '*.ply' --exclude '*.pt' \
  ubuntu@$IP:plantpipeline/experiments/monogrow/runs/gpu_val_obj ubuntu@$IP:plantpipeline/experiments/monogrow/runs/gpu_test_obj ../monogrow/runs/
```
Then fill in the val table and the decision here from `runs/gpu_val_obj/summary.md` / `decision.md` and `runs/gpu_test_obj/RESULTS.md`.
