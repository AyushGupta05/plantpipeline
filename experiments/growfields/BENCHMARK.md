# GrowFields benchmark: exact evaluation protocol, published numbers, reproduction commands, CPU smoke test

Written 2026-10-30. Sources: GrowFields repo `Joaquin-Gajardo/growfields` at commit `3f8e947` ("code release",
the only commit), its `REPRODUCIBILITY.md`, the arXiv HTML of the paper (`arxiv.org/html/2607.03330v2`,
ECCV 2026), `plant-shape-analysis` (PSA) v0.4.1 = `99d86f5`, and the ETH `TrackPlant3D_v2.zip`
(246,226,495 bytes). Every claim below is tied to a file:function or a paper table; anything not checked
is marked **[unverified]**. Code paths are relative to the growfields repo root. Earlier research notes
(`growfields_trackplant3d.md`, `psa_trackplant3d_api.md`) cover dataset provenance/PSA API and are not repeated.

Abbreviations: CD = Chamfer distance (mm²), EPE = end-point error on leaf tips (mm), GF = GrowFields,
PSA = plant-shape-analysis, TP3D = TrackPlant3D. Sequence codes: Ma=maize, So=sorghum, Tb=tobacco,
Tm1/Tm2=tomato1/tomato2; C/H/S = control/heat-or-highlight/shade; digit = plant number
(`scripts/create_latex_table_fullplant.py:abbreviate_sequence`).

---

## 1. Evaluation protocol (as implemented)

### 1.1 Plants: validation and test

`configs/datasets/TrackPlant3D/data_splits.json` (byte-identical to `v2/splits/data_splits.json` in the ETH zip).
Splits are at plant level; there is **no training split** — every method is fitted per plant sequence
(test-time optimisation), the "val" plants exist only for hyper-parameter tuning.

| split | plants (sequence names) | frames (days) | organ sequences |
|---|---|---|---|
| val (4) | `maize_control_plant1`, `sorghum_control_plant1`, `tobacco_control_plant2`, `tomato2_control_plant3` | — | 24 (20 leaves + 4 stems) |
| test (8) | `maize_control_plant2` (MaC2, 9), `maize_control_plant3` (MaC3, 8), `sorghum_control_plant2` (SoC2, 8), `sorghum_highlight_plant2` (SoH2, 5), `tobacco_control_plant1` (TbC1, 7), `tobacco_shade_plant3` (TbS3, 8), `tomato1_heat_plant3` (Tm1H3, 16), `tomato1_shade_plant1` (Tm1S1, 17) | 78 frames total | 49 (41 leaves + 8 stems); 45 with ≥3 frames |

Frame counts from the paper (Sec. 4.1 / supplementary "Dataset details"). The other 31 TP3D plants are never used.
Day indices come from the filename token `D<nn>` and need not be contiguous (SoH2 has days `[2,4,6,8,10]`,
observed in the smoke run).

### 1.2 What is held out

Three independent hold-outs, all defined by the code:

1. **Points (always).** For every plant and every frame the 10,000 points are split 50/50 at random
   (`scripts/create_train_test_splits.py`, `--train_ratio 0.5 --seed 42`, run against dataset version v2).
   Result: `configs/datasets/TrackPlant3D/point_splits/<plant>_split.json` =
   `{sequence_name, random_seed:42, train_ratio:0.5, created_date, dataset_version:"v2", n_timepoints,
   timepoints:[{timepoint_idx, n_total_points, n_train, n_test, train_indices:[...], test_indices:[...], ...}]}`.
   Indices are positions in that frame's PLY point array (v2 order) — `growfields/pipeline/splits.py:validate_split`
   refuses a split whose `n_total_points` differs from the loaded cloud. **Every annotated leaf tip is forced into
   the test set** (`create_train_test_splits.py` lines 116-134: tips found in train are moved to test).
   Training uses only train points; CD is reported on test points only (`cd_test`).
   Same split file for every method ("fixed and shared across all methods", paper Sec. 4.1).
2. **Frames, interpolation (Table 3 / 6).** `--train_timepoint_stride 2` keeps frame indices 0,2,4,… of the
   split file (`splits.py:create_strided_split_file`); odd frames are "interpolated frames" (strictly between the
   first and last training day, `io_utils.py:save_interpolation_from_organ_models`).
3. **Frames, extrapolation (Table 4).** `--train_holdout_last_n 3` masks the last 3 frames of every organ from
   training (`growfields/models/organ_ablations.py:apply_holdout_last_n`), evaluated only on those frames.
   SoH2 (5 frames) is excluded. **Caveat (REPRODUCIBILITY §6):** the per-organ spatial normalisation and the
   per-organ day range are computed over *all* frames including the held-out ones, and organs that first appear
   in held-out frames are filled with ground truth, so absolute extrapolation errors are optimistic.

Organs: the stem is organ 0, leaves are positive ids persistent over time (`labels` field of the v2 PLYs; the
"w/ model segmentation" ablation uses `predicted_labels` from `v2-autoseg`). There is no per-organ hold-out.

### 1.3 Prediction protocol (what is compared to what)

- Each organ is represented by its **first observed frame's point cloud (train + test points together)** and
  warped forward by multi-step integration through every later frame; the frame-0 `is_train` mask is inherited
  by all predictions ("these models move the initial points rather than producing new ones",
  `organ_model_manager.py:save_organ_predictions` line 1181-1184). Nothing is resampled or densified.
- Rollout starts at the organ's first observed frame, so the first frame of every organ has no prediction and is
  never scored; a plant with T frames yields T-1 scored full-plant frames.
- Per-organ models work in the organ's **canonical frame** (leaves PCA-aligned by PSA
  `LeafSequencesDataset(apply_alignment=True)`: principal axis → Z, base at origin, sign from stem/tip; stem stays
  in plant space) and in **per-organ normalised coordinates**: `x_norm = (x - min_xyz)/scale*2 - 1` with
  `min_xyz`, `scale = max extent` computed from the organ's **train points over all frames**
  (`organ_model_manager.py:_normalize_organ_sequence_traintest`). Predictions are de-normalised back to mm before
  any metric; canonicalisation is a rigid transform, so per-organ CD/EPE in canonical space equal plant-space
  distances (`tests/test_standalone.py` checks every alignment is a rotation).
- Full-plant prediction = concatenation of the de-normalised organ predictions, each mapped back to plant space
  with **that target day's ground-truth rigid pose** (`io_utils.py:save_full_plant_from_organ_models`, inverse
  of PSA `compose_transformations`; for held-out frames `use_pose_anchoring=True` uses the exact per-day GT
  matrix, `save_interpolation_from_organ_models` line ~1796). Tables therefore measure shape + growth, not where
  an organ moves (REPRODUCIBILITY "Held-out frames use their observed poses").
- **GT fill:** an organ present in a target frame but without a model (seen only once, or first appearing in a
  held-out frame) is copied from ground truth into the full-plant prediction
  (`save_full_plant_from_organ_models` lines 1004-1045; `save_interpolation_from_organ_models` ~1487). This
  applies only to full-plant aggregation; per-organ scoring simply has no row for such frames.
- Held-out frames: roll out from frame 0 through the training frames up to the last training frame before the
  unseen day, then integrate with the training Euler sub-step from `t_prev_norm` to
  `t_target_norm = (unseen_day - first_day)/day_range` plus a remainder step
  (`save_interpolation_from_organ_models` lines ~1705-1730). Extrapolation is the same beyond the last training day.
- Time: `t_norm = (day - first_day)/(last_day - first_day)` **per organ** over the organ's full (unstrided) day
  range; Euler step `x += v(x,t)·dt` with `dt = t_norm(next) - t_norm(cur)`, `train_integration_steps=1`
  (`growfields/models/integration.py:make_t_norm_helpers`, `_integrate_segment`).

### 1.4 Metrics, as implemented (`growfields/utils/metrics.py`)

- **CD** — `compute_chamfer_distance(pred_pts, gt_pts)` → `pytorch3d.loss.chamfer_distance(pred[None], gt[None])[0]`
  with pytorch3d defaults (`point_reduction="mean"`, `batch_reduction="mean"`, `norm=2`), i.e.

  `CD(P,G) = mean_{p∈P} min_{g∈G} ||p−g||² + mean_{g∈G} min_{p∈P} ||p−g||²`

  **squared** distances, **sum (not mean) of the two directional means**, units **mm²** (v2 coordinates are mm;
  consistent with paper captions "CD (mm²)"). No normalisation before the metric: both clouds are in mm, plant
  space (full plant) or canonical organ space (per organ). `cd_test` uses only points with `is_train==0` on both
  sides — predicted test points (frame-0 split carried forward) vs GT test points of the target frame
  (`scripts/compute_metrics_per_organ.py:compute_cd_metrics`, `scripts/compute_metrics_fullplant.py` lines 269-284).
  The tables use **`cd_test`**; `cd` (all points) and `cd_train` are also written but never reported
  (`create_latex_table_*.py:cd_test_column` raises if `cd_test` is absent).
- **Number of points:** all of them — per frame ~5,000 GT test points vs the predicted test points of the organ
  (frame-0 count); no subsampling at evaluation. Training samples 8,192 source points per step split equally over
  up to 10 organs (`min 256/organ`).
- **EPE** — `compute_epe(pred_tips, gt_tips) = mean ||pred_tip − gt_tip||₂` in **mm**, tips matched by organ id
  (one tip per leaf). The predicted tip is the GT tip of the organ's first tip-annotated frame, normalised and
  advected by the same field as the points; a tip is scored only on frames **after** its first appearance
  (`compute_metrics_per_organ.py` lines 220-248; `compute_metrics_fullplant.py` lines 308-361). Fixed counts:
  **192** tip measurements in full-plant tables, **189** in per-organ tables (3 lost to the ≥3-frame organ filter);
  234 annotated tips in total (REPRODUCIBILITY "Checking a reproduction").
- **NC** (normal consistency, `1 − normal_loss/2`, mean |cos| over NN pairs both directions), **AccS/AccR**
  (`scene_flow_metrics` from DeformationPyramid, thresholds 0.025/0.05 applied to tip positions in mm, so
  meaningless) are computed and written to the CSVs but **not reported** anywhere.
- CSV columns: per organ `all_metrics.csv` = `sequence, sequence_name(<plant>_leafK|_stem), organ_id,
  organ_type, run, timepoint(day), cd_train, cd_test, nc_train, nc_test, n_points_total, n_points_train,
  n_points_test, epe, accs, accr, n_tips, training_time_seconds, total_model_size_mb`; full plant
  `fullplant_metrics.csv` adds `cd, nc` (all points).

### 1.5 Aggregation (REPRODUCIBILITY "How the numbers are computed"; `create_latex_table_*.py --per_sequence`)

- **Full-plant tables (1, 3, 4-fullplant, 5, 7):** for each (plant, run) average `cd_test`/`epe` over the T-1
  scored frames (EPE over frames that have tips); average over runs → per-plant cell; **Mean = unweighted mean
  over the 8 plants**; the ± on Mean is the std across runs of the per-run plant-average
  (`create_latex_table_fullplant.py:compute_per_sequence_metrics`, `compute_mean_column_run_std`).
- **Per-organ tables (2, 4-per-organ, 6):** average over frames for each (plant, organ, run) → average over the
  plant's organs (equal organ weight within a plant, NaN-safe for EPE) → average over runs → per-plant cell;
  Mean = unweighted mean over plants. Organs with fewer than **2 scored frames** (= 3 observed frames) are
  dropped: `--min_timepoints_per_organ 2`, leaving 45 of 49 organ sequences (35 in stride-2 interpolation runs).
  Not applied to extrapolated-frame rows (at most 3 held-out frames per organ).
  Running without `--per_sequence` weights organs equally and gives 0.80 instead of the paper's 0.87 CD.
- Per-species numbers do not exist in the paper: columns are per plant (two per species), Mean over 8 plants.
- Runs: `--n_runs N` re-seeds model init with `seed+run_idx` (default base 42); the point split is fixed. Paper
  says 5 runs everywhere; in reality `Ours (full)` main rows use 5, interpolation/extrapolation `Ours`, ablations
  and `Ours (per-organ MLPs)` use 3, and NDF/NVFi/DSR/CanFields accidentally reused one seed (REPRODUCIBILITY §1).
- Table 4 mixes aggregations in the paper: COAP row is full-plant aggregation, Ours row is per-organ
  aggregation (REPRODUCIBILITY §3; both versions reproduced in §2.4 below).

---

## 2. Published numbers (paper point estimates; ± over runs in `REPRODUCIBILITY.md`)

Values parsed from the arXiv HTML tables (`S4.T1`-`S4.T5` + supplementary T6, T7). Each cell is `CD / EPE`
(mm² / mm), lower is better. "—" = undefined (DSR has no deformation field).

### 2.1 Table 1 — full-plant reconstruction (all frames seen; 5 runs for Ours full, DPF/COAP; baselines effectively 1 seed)

| Method | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| NDF | 14,496.5 / 73.6 | 1,227.9 / 95.8 | 1,313.4 / 54.9 | 303.0 / 54.3 | 290.2 / 14.0 | 135.6 / 17.9 | 14.1 / 10.4 | 52.7 / 22.0 | 2,229.2 / 42.9 |
| NVFi | 2,178.9 / 72.2 | 1,495.3 / 97.4 | 873.0 / 56.3 | 161.8 / 56.0 | 103.0 / 15.1 | 86.0 / 18.2 | 32.6 / 9.4 | 137.3 / 21.9 | 633.5 / 43.3 |
| DSR | 481.3 / — | 194.9 / — | 186.4 / — | 61.1 / — | 29.9 / — | 25.0 / — | 13.1 / — | 84.0 / — | 134.5 / — |
| CanFields | 2,025.7 / 77.6 | 1,774.8 / 98.9 | 611.4 / 56.4 | 267.6 / 59.5 | 102.1 / 16.6 | 67.3 / 21.1 | 23.0 / 10.4 | 53.5 / 14.4 | 615.7 / 44.4 |
| DPF (single model, pre-finetune) | 8.17 / 23.18 | 5.00 / 5.30 | 1.21 / 5.78 | 0.92 / 57.22 | 0.36 / 7.58 | 0.46 / 16.21 | 0.17 / 1.93 | 3.47 / 3.97 | 2.47 / 15.15 |
| COAP (part-aware) | 4.84 / 7.57 | 2.48 / 3.87 | 1.31 / 7.76 | 1.02 / 5.61 | 1.03 / 3.22 | 0.65 / 2.55 | 0.38 / 1.40 | 1.82 / 1.88 | 1.69 / 4.23 |
| Ours (per-organ MLPs) | 2.41 / 3.68 | 1.52 / 2.07 | 0.87 / 1.47 | 1.01 / 3.44 | 0.70 / 3.28 | 0.48 / 2.67 | 0.13 / 0.74 | 0.33 / 1.59 | 0.93 / 2.37 |
| **Ours (full) = GrowFields** | 2.51 / 2.38 | 1.43 / 2.01 | 0.69 / 1.34 | 0.74 / 0.97 | 0.59 / 1.66 | 0.45 / 1.30 | 0.12 / 0.80 | 0.32 / 1.34 | **0.86 / 1.47** |

REPRODUCIBILITY ± for Mean: DPF 2.47±0.57 / 15.15±0.52; COAP 1.69±0.09 / 4.23±0.83; per-organ MLPs 0.93±0.02 /
2.37±0.21; Ours 0.86±0.02 / 1.47±0.11. CanFields re-evaluated with its training Fourier frequencies (bug fix):
130.1±20.6 / 37.9±2.8 (REPRODUCIBILITY §8). DPF post-finetune is worse (4.75 / 16.99).

### 2.2 Table 2 — per-organ reconstruction (45 organ sequences; organ-averaged within plant)

| Method | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| NDF | 484.6 / 65.9 | 317.0 / 104.4 | 45.9 / 69.7 | 17.6 / 29.8 | 29.5 / 14.0 | 15.2 / 6.8 | 1.5 / 2.7 | 3.6 / 6.8 | 114.4 / 37.5 |
| NVFi | 540.1 / 68.7 | 190.1 / 100.4 | 37.5 / 75.0 | 13.7 / 28.6 | 13.9 / 15.3 | 7.2 / 6.4 | 1.8 / 2.5 | 5.3 / 7.7 | 101.2 / 38.1 |
| DSR | 41.9 / — | 21.6 / — | 26.9 / — | 10.1 / — | 15.3 / — | 6.5 / — | 1.3 / — | 2.1 / — | 15.7 / — |
| CanFields | 245.0 / 52.1 | 355.7 / 77.1 | 174.8 / 55.3 | 294.6 / 31.1 | 18.9 / 7.1 | 6.6 / 4.7 | 2.0 / 2.7 | 6.1 / 5.1 | 138.0 / 29.4 |
| DPF (per organ) | 2.32 / 2.05 | 1.67 / 2.60 | 1.30 / 1.43 | 0.87 / 1.43 | 0.69 / 3.18 | 0.56 / 1.57 | 0.30 / 2.01 | 0.68 / 2.23 | 1.05 / 2.06 |
| COAP | 4.17 / 8.20 | 2.55 / 5.68 | 2.22 / 16.63 | 0.93 / 8.82 | 0.99 / 5.01 | 0.64 / 2.02 | 0.42 / 1.75 | 1.60 / 1.93 | 1.69 / 6.26 |
| Ours (per-organ MLPs) (3 runs) | 2.10 / 3.47 | 1.52 / 2.89 | 1.13 / 1.59 | 0.96 / 2.34 | 0.56 / 4.13 | 0.50 / 1.88 | 0.24 / 1.42 | 0.51 / 1.84 | 0.94 / 2.44 |
| **Ours (full)** | 2.26 / 2.59 | 1.48 / 2.46 | 0.93 / 1.51 | 0.64 / 0.85 | 0.49 / 2.04 | 0.45 / 1.20 | 0.20 / 0.98 | 0.46 / 1.33 | **0.87 / 1.62** |

± (REPRODUCIBILITY): DPF 1.05±0.02 / 2.06±0.20; COAP 1.69±0.11 / 6.26±1.91; per-organ MLPs 0.94±0.02 / 2.44±0.21;
Ours 0.87±0.02 / 1.62±0.14. Significance (Wilcoxon, 45 organs): Ours < COAP on CD and EPE (p<0.001), Ours < DPF
on CD (p<0.001) and EPE (p=0.004).

### 2.3 Table 3 — full-plant interpolation (train on frames 0,2,4,…; `--train_timepoint_stride 2`)

Training frames / Interpolated frames:

| Method | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| COAP, train | 3.20 / 11.87 | 0.82 / 3.08 | 3.04 / 9.60 | 1.11 / 2.20 | 0.91 / 2.39 | 0.56 / 2.34 | 0.15 / 2.46 | 0.59 / 1.40 | 1.30 / 4.42 |
| DPF (per-organ), train | 2.62 / 1.94 | 0.77 / 1.67 | 0.51 / 0.85 | 0.63 / 2.87 | 0.83 / 1.98 | 0.57 / 1.58 | 0.13 / 1.26 | 0.41 / 1.67 | 0.81 / 1.73 |
| Ours (per-organ MLPs), train | 2.59 / 2.21 | 0.97 / 5.49 | 0.59 / 1.28 | 0.72 / 4.73 | 0.77 / 3.44 | 0.47 / 2.05 | 0.12 / 1.14 | 0.37 / 1.93 | 0.82 / 2.78 |
| **Ours (full), train** | 2.24 / 2.14 | 0.70 / 1.72 | 0.42 / 0.71 | 0.44 / 0.56 | 0.57 / 5.39 | 0.39 / 1.70 | 0.10 / 0.58 | 0.35 / 1.24 | 0.65 / 1.75 |
| COAP, interp | 19.30 / 14.63 | 9.57 / 12.97 | 25.42 / 10.20 | 31.07 / 9.97 | 4.16 / 3.90 | 1.88 / 2.82 | 0.83 / 2.55 | 1.62 / 1.63 | 11.73 / 7.33 |
| DPF (per-organ), interp | 17.75 / 8.06 | 8.93 / 4.72 | 18.38 / 5.26 | 29.38 / 5.74 | 3.23 / 3.13 | 1.88 / 1.89 | 0.89 / 1.63 | 1.47 / 1.60 | **10.24 / 4.00** |
| Ours (per-organ MLPs), interp | 18.00 / 8.04 | 9.47 / 13.32 | 19.17 / 5.55 | 29.42 / 9.73 | 2.84 / 4.59 | 1.76 / 2.57 | 0.91 / 1.56 | 1.53 / 2.28 | 10.39 / 5.96 |
| Ours (full), interp | 18.01 / 7.76 | 9.18 / 11.82 | 19.53 / 5.34 | 28.60 / 9.69 | 2.93 / 6.49 | 1.75 / 2.48 | 0.89 / 1.12 | 1.42 / 1.88 | 10.29 / 5.82 |

DPF per-organ is the best interpolator (the paper says so). Table 6 (supplementary, per-organ aggregation of the
same runs): interpolated-frame Means — COAP 12.19 / 6.34, DPF 11.43 / 4.18, per-organ MLPs 11.42 / 4.91,
Ours 11.42 / 4.59; training-frame Means — COAP 1.25 / 4.93, DPF 0.80 / 1.50, per-organ MLPs 0.76 / 2.22,
Ours 0.64 / 1.77 (full per-plant rows in `REPRODUCIBILITY.md` "Supplementary — per-organ interpolation").

### 2.4 Table 4 — extrapolation (train on all but last 3 frames; SoH2 excluded; multi-step rollout from t0)

Paper rows (COAP full-plant aggregation; Ours per-organ aggregation):

| Method | MaC2 | MaC3 | SoC2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | **Mean** |
|---|---|---|---|---|---|---|---|---|
| COAP, training frames | 1.34 / 12.26 | 0.57 / 5.01 | 0.49 / 1.02 | 0.18 / 1.13 | 0.26 / 1.24 | 0.22 / 1.15 | 0.54 / 1.54 | 0.51 / 3.34 |
| Ours, training frames | 1.06 / 1.81 | 0.40 / 0.92 | 0.39 / 1.08 | 0.15 / 0.80 | 0.18 / 1.67 | 0.14 / 0.79 | 0.30 / 1.33 | 0.37 / 1.20 |
| COAP, extrapolated frames | 768.08 / 47.17 | 1632.07 / 103.33 | 620.43 / 42.77 | 79.27 / 9.23 | 15.82 / 5.04 | 8.59 / 5.92 | 33.40 / 6.25 | 451.09 / 31.39 |
| **Ours, extrapolated frames** | 356.13 / 20.96 | 321.08 / 69.77 | 337.39 / 27.04 | 15.38 / 4.06 | 8.09 / 4.14 | 2.80 / 2.52 | 6.56 / 4.46 | **149.63 / 18.99** |

Consistent aggregations (REPRODUCIBILITY Table 4): extrapolated frames, full-plant agg.: COAP 451.09±36.69 /
31.39±2.77 vs Ours 117.71±35.55 / 21.79±0.54; per-organ agg.: COAP 863.59±183.04 / 29.46±3.22 vs Ours
149.63±19.65 / 18.99±0.68. Ours wins every column either way. Note the extrapolated CDs are 2-3 orders of
magnitude above the fitted ones.

### 2.5 Table 5 — ablations (full-plant, 3 runs each except Ours)

| Variant (override) | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| Ours (full) | 2.51 / 2.38 | 1.43 / 2.01 | 0.69 / 1.34 | 0.74 / 0.97 | 0.59 / 1.66 | 0.45 / 1.30 | 0.12 / 0.80 | 0.32 / 1.34 | 0.86 / 1.47 |
| w/o canonicalisation (`dataset.aligned_leaves=false`) | 2.76 / 3.14 | 1.58 / 2.28 | 0.93 / 3.41 | 0.79 / 13.24 | 0.60 / 5.79 | 0.47 / 1.40 | 0.21 / 3.18 | 0.34 / 1.42 | 0.96 / 4.23 |
| w/o latent code (`default_model.conditioning=none`) | 2.77 / 8.04 | 1.76 / 3.70 | 0.83 / 1.43 | 1.00 / 3.32 | 0.83 / 3.72 | 0.61 / 3.31 | 0.15 / 1.28 | 0.47 / 2.72 | 1.05 / 3.44 |
| w/o multi-step CD loss (`default_model.use_multistep_loss=false`) | 2.87 / 2.79 | 1.45 / 2.21 | 0.68 / 1.37 | 0.75 / 1.04 | 0.58 / 3.29 | 0.46 / 1.65 | 0.14 / 1.34 | 0.38 / 1.48 | 0.91 / 1.89 |
| w/o L_code (`default_model.embed_reg_lambda=0`) | 2.52 / 2.46 | 1.54 / 2.14 | 0.71 / 1.38 | 0.74 / 0.99 | 0.61 / 3.42 | 0.45 / 1.06 | 0.12 / 0.85 | 0.32 / 1.62 | 0.88 / 1.74 |
| w/ FiLM conditioning (`default_model.conditioning=film`) | 2.62 / 5.74 | 1.67 / 1.93 | 0.83 / 1.39 | 0.92 / 3.32 | 0.67 / 3.04 | 0.47 / 2.04 | 0.13 / 0.92 | 0.33 / 1.35 | 0.96 / 2.47 |
| w/ model segmentation* (`dataset.version=v2-autoseg dataset.label_field=predicted_labels dataset.exclude_sequences=["sorghum_heat_plant1"]`) | 1.77 / 2.17 | 1.21 / 1.94 | 0.68 / 1.32 | 0.48 / 0.80 | 2.01 / 1.59 | 0.36 / 1.55 | 0.12 / 0.61 | 0.27 / 1.36 | 0.86 / 1.42 |

\* different organ decomposition → not comparable; expect Tm1H3 EPE ~0.2 higher with PSA ≥ v0.4.0 (REPRODUCIBILITY §7).

### 2.6 Table 7 — robustness (full-plant, Ours, 5 runs)

Mean: unperturbed 0.86 / 1.47; noise σ=0.01 (in normalised [-1,1] organ coordinates, i.e. 0.5 % of the organ's
largest extent — the paper's "mm" is wrong, REPRODUCIBILITY §5c) 1.08 / 2.07; σ=0.05 5.24 / 4.97; holes 5×K=15
0.96 / 1.86; holes 5×K=25 1.07 / 1.96. Per-plant rows in `REPRODUCIBILITY.md`.

---

## 3. Reproducing their results: configs, commands, cost

Entry points: `train_per_organ.py` (reads `default_model:` + `per_organ_training:`; all part-aware methods) and
`train_fullplant.py` (reads `model:`; DPF single model). Run from the repo root; the dataset auto-downloads into
`data/TrackPlant3D/versions/v2` via PSA on first use. Output: `results/fullplant_per_organ_batch/
<timestamp>_fullplant_batch_<config>_<split>_aligned[_stride2|_holdlast3]/sequences/<plant>/run_<k>/
{full_plant/{predicted,ground_truth}, organs/<leaf_k|stem>/{predicted,ground_truth}, models/, results.json}`
plus `experiment_config.json` at the batch root. PLYs: `pred_<plant>_D<dd>.ply`, `pred_<plant>_D<dd>_leaf3.ply`
with fields `positions, organ_label, is_leaf_tip, tip_organ_label, is_train, normals`.

| Row | Config (`model_type`) | Key hyper-parameters (from the YAML) | Command |
|---|---|---|---|
| Ours (full) = GrowFields | `configs/per_organ/shared_continuous.yaml` (`shared_continuous`) | SIREN 5 hidden layers × **512** units (paper text says 256; code wins), ω0=30, in=(x,y,z,t)+32-d organ code concat, Adam lr 5e-6 (ReduceLROnPlateau patience 10, factor 0.5, stepped every `log_interval`=200), codes lr 5e-3, init N(0,0.1), λ_code=1e-4 (as `weight.pow(2).mean()`, i.e. 32× weaker than the paper formula), 50,000 steps, 8,192 points/step split over `organs_per_step=10`, 50 % consecutive-frame pairs, multistep CD, Euler 1 sub-step, day-based per-organ t_norm, `aligned_leaves: true`, `normalize: false` | `python train_per_organ.py --config configs/per_organ/shared_continuous.yaml --split test --n_runs 5 --points_split_dir configs/datasets/TrackPlant3D/point_splits` |
| Ours (per-organ MLPs) | `configs/per_organ/continuous.yaml` (`continuous`) | one SIREN per organ, 256×5, 30,000 steps, 4,096 pts, lr 5e-6, multistep CD | same with `--config configs/per_organ/continuous.yaml --n_runs 3` (Table 1 row was trained with a since-deleted `use_spatial_smoothness` regulariser) |
| DPF (per organ) | `configs/per_organ/pairwise.yaml` (`pairwise`) | one SIREN 256×3 per frame pair per organ, 2,000 steps lr 5e-5, 2,048 pts, then joint finetune 3,000 steps (lr 5e-7, x0 lr 5e-6, w_global 2, w_local 0.5) | `--config configs/per_organ/pairwise.yaml --n_runs 5` |
| COAP | `configs/per_organ/encoder_decoder.yaml` (`encoder_decoder_continuous`) | ResnetPointnet encoder (hidden 128, 1,024 pts, lr 1e-3) → 32-d code; ImplicitNet decoder [256]*5 skip at 3, lr 5e-4; 50,000 steps, 4,096 pts, `organs_per_step` 6, multistep CD | `--config configs/per_organ/encoder_decoder.yaml --n_runs 5` |
| DPF (single model, Table 1) | `configs/full_plant/pairwise.yaml` | SIREN 256×5 per frame pair on the whole plant, 2,000 steps lr 1e-5, finetune 3,000 steps lr 1e-6 (w_global 5, w_local 1); `normalize: true`, `aligned_leaves: false`; **reported row = pre-finetune** (`fullplant_metrics_pre.csv`) | `python train_fullplant.py --config configs/full_plant/pairwise.yaml --split test --n_runs 5 --points_split_dir configs/datasets/TrackPlant3D/point_splits` |
| NDF / NVFi / DSR / CanFields | not in this repo | fork `Joaquin-Gajardo/CanFields-TrackPlant3D` at `a865e96` (two bugs since fixed: points reordered before split → train overlapped test; CanFields Fourier frequencies not saved) | not reproducible from here |

Interpolation (Table 3/6): add `--train_timepoint_stride 2` (`--n_runs 3` for Ours, 5 for COAP/DPF).
Extrapolation (Table 4): `--n_runs 3 --train_holdout_last_n 3 --sequences maize_control_plant2 maize_control_plant3
sorghum_control_plant2 tobacco_control_plant1 tobacco_shade_plant3 tomato1_heat_plant3 tomato1_shade_plant1`
(no `--split`: it would override `--sequences`). Ablations: Table-1 command with `--n_runs 3` plus the override in §2.5.
Robustness: `--input_noise_std 0.01|0.05` or `--train_holes_n 5 --train_holes_k 15|25`.

Evaluation (metrics are **never** computed during training; only from saved PLYs):

```bash
bash scripts/compute_metrics_and_latex_table_fullplant.sh  results/fullplant_per_organ_batch/<exp>   # Tables 1,5,7
bash scripts/compute_metrics_and_latex_table_per_organ.sh  results/fullplant_per_organ_batch/<exp>   # Table 2
bash scripts/compute_metrics_and_latex_table_fullplant.sh  <exp> --interpolation                       # Table 3
bash scripts/compute_metrics_and_latex_table_per_organ.sh  <exp> --extrapolation [--skip_sequences sorghum_highlight_plant2]  # Table 4
# individual steps
python scripts/compute_metrics_fullplant.py  --experiment_dir <exp>      # -> fullplant_metrics.csv
python scripts/compute_metrics_per_organ.py  --experiment_dir <exp>      # -> all_metrics.csv (hard-codes data/TrackPlant3D/versions relative to CWD)
python scripts/create_latex_table_fullplant.py --csv <exp>/fullplant_metrics.csv --full_table --per_sequence
python scripts/create_latex_table_per_organ.py --csv <exp>/all_metrics.csv --full_table --min_timepoints_per_organ 2 --per_sequence
python scripts/compute_statistical_significance.py --ours <ours>/all_metrics.csv --coap <coap>/all_metrics.csv --dpf <dpf>/all_metrics.csv
pytest tests/ -v --experiment-dir <exp>/sequences     # invariants: tips are test points, PLY==dataloader, counts==split
```

Integrity check for any reproduction: `fullplant_metrics.csv` run 1 `n_tips.sum() == 192`; `all_metrics.csv`
filtered to organs with ≥2 rows `n_tips.sum() == 189`.

**Compute reported:** none in the paper. Repo: Euler cluster SLURM (`run_euler.sh`: 1 GPU, `gpumem:8G`, 3 CPUs,
24 h wall limit, CUDA 12.2 modules) — GPU model not stated **[unverified]**. `results.json` records
`training_time_seconds` per sequence and the CSVs carry it, but no published value was found. `--deterministic`
costs ~30 % time and 1.7× GPU memory for GrowFields (≤2.2 GB), so the normal run needs well under 2 GB.
Pinned env: Python 3.10, torch 2.0.1+cu117, pytorch3d 0.7.4 (Linux wheel), numpy 1.23.1, PSA `99d86f5`.
Model sizes: GrowFields 1,333,955 params for 6 organs (+32 per extra organ; measured with
`growfields.models.siren.SharedContinuousSiren(6,32,512,5)`), 5.09 MB on disk; FiLM variant 1.49 M; per-organ
MLP 331,011 per organ; DPF per-organ 199,171 per frame pair per organ; DPF single 330,755 per frame pair.
Paper runs are not bitwise repeatable (pytorch3d Chamfer backward uses CUDA atomics); EPE varies ~1 mm between
identical runs; "Read EPE with its std".

---

## 4. CPU smoke test on this Mac (verified 2026-10-30)

Machine: Apple M5 Pro, macOS 26.5, no CUDA; `uv 0.11.14`, Python 3.12.13, torch 2.15.0 (CPU wheel; MPS available
but unused — GF's `DEVICE` picks CUDA or CPU only, `growfields/pipeline/config.py:14`). Everything lives under
the scratchpad: venv `…/scratchpad/gf_smoke/.venv`, source `…/scratchpad/gf_clone`, data
`…/scratchpad/gf_smoke/data/TrackPlant3D/versions/v2`, results `…/scratchpad/gf_smoke/results/`.
Script: `experiments/growfields/smoke_cpu.sh [WORKDIR] [N_STEPS] [SEQUENCE]` (set `TP3D_ZIP=` to reuse the local zip).

What worked unchanged: training (`train_per_organ.py`), prediction export, both metric scripts, both LaTeX table
scripts, PSA leaf canonicalisation of all 43 plants at load time (~2-3 s). `DEVICE` fell back to CPU automatically;
no CUDA assert anywhere (`torch.cuda.manual_seed_all` is a no-op on CPU).

What had to be worked around (no file in the GF/PSA checkouts was modified):

1. **pytorch3d**: pinned as a `py310_cu117` Linux wheel; the only symbol used is `pytorch3d.loss.chamfer_distance`
   (`growfields/utils/metrics.py`, `models/siren_dynamics.py`, `models/encoder_decoder.py`, `tests/test_standalone.py`).
   Replaced by a 40-line pure-torch shim installed through a `.pth` file (so it works under `python -I`).
   Verified identical value and gradient to GF's own deterministic `chamfer_loss` path (`allclose` True), and
   the normals term matches the pytorch3d convention used by `compute_chamfer_distance`.
2. **Python/numpy pins**: `numpy==1.23.1` has no 3.12 wheel; installed `numpy 2.3.5`, `open3d 0.20.0`,
   `scipy`, `scikit-learn` (PSA imports both at module import although its `pyproject` lists only numpy/torch/open3d).
3. **CWD**: `compute_metrics_per_organ.py` hard-codes `data/TrackPlant3D/versions` relative to CWD, so the data
   was extracted under the work dir and all commands run from there with `--dataset_path`/`--output_dir` absolute.
4. Data from the local `TrackPlant3D_v2.zip` (top-level `v2/`, 378 PLYs in `gt_corrected_v2/`) instead of
   PSA's auto-download (also supported by the script).

Commands (abbreviated; `GF` = clone, `PY` = venv python, CWD = work dir):

```bash
$PY -I $GF/train_per_organ.py --config $GF/configs/per_organ/shared_continuous.yaml \
  --sequences sorghum_highlight_plant2 --points_split_dir $GF/configs/datasets/TrackPlant3D/point_splits \
  --dataset_path $WORK/data/TrackPlant3D/versions --output_dir $WORK/results --seed 0 \
  default_model.n_steps=50 default_model.log_interval=10 output.save_plots=false
$PY -I $GF/scripts/compute_metrics_per_organ.py --experiment_dir $EXP --quiet
$PY -I $GF/scripts/compute_metrics_fullplant.py --experiment_dir $EXP --quiet
$PY -I $GF/scripts/create_latex_table_per_organ.py --csv $EXP/all_metrics.csv --full_table --per_sequence --min_timepoints_per_organ 2
$PY -I $GF/scripts/create_latex_table_fullplant.py --csv $EXP/fullplant_metrics.csv --full_table --per_sequence
```

Measured (SoH2: 5 frames, days 2-10, 7 organs, 6.9k-8.6k points per frame (3.8k-4.3k test), 2-4 leaf tips per frame):

| item | value |
|---|---|
| whole 50-step run incl. dataset load, prediction export, normals | 16.3 s wall |
| training 50 steps (5 eval passes at `log_interval=10`) | 8.4 s |
| training 200 steps (1 eval pass) | 25.8 s |
| ⇒ **per training step on CPU** (512×5 SIREN, 8,192 pts over 7 organs) | **≈0.13 s**; eval pass ≈0.4 s |
| ⇒ full 50,000-step paper run for this plant | ≈1.8 h CPU (longer for 16-17-frame Tm1 plants) [extrapolated] |
| `compute_metrics_per_organ.py` / `compute_metrics_fullplant.py` | 2.2 s / 2.4 s |
| metrics after 50 steps (meaningless, pipeline check only) | per-organ Mean CD 42.43 / EPE 14.78 (5 of 6 organs kept; leaf 5 dropped: 1 scored frame); full-plant Mean CD 53.99 / EPE 12.63; n_tips 11 (full) / 10 (per-organ) |

The first-frame-of-organ exclusion, the ≥2-scored-frames organ filter, `cd_test`-only reporting and the GT fill
for leaf 6 (first appears in the last frame) all behaved as described in §1. Log: `…/scratchpad/gf_smoke/train_log.txt`.

Not done: MPS device (would need editing `config.py`/`trackplant3d.py` device defaults — out of scope since GF
code must stay untouched); `pytest tests/test_standalone.py` (imports real pytorch3d for a comparison test; would
trivially pass against the shim) **[unverified]**; full 50k-step run.

---

## 5. Weaknesses to exploit

Stated in the paper (Sec. 5.1) and visible in code:

1. **No learning across plants.** Every table entry is a separate optimisation on one plant's own frames
   (`train_per_organ.py` loops over sequences; a new `OrganModelManager` + fresh `SharedContinuousSiren` per
   sequence and per run). The "cross-organ prior" is only the shared field among the ≤17 organs of one plant.
   Val/test plants are never seen by any other plant's model. A model trained on the other 31 TP3D plants (or
   Pheno4D) that is evaluated zero-/few-shot on the 8 test plants would be a categorically different capability.
2. **Extrapolation is weak and the protocol flatters it.** Held-out-frame CD is 150 mm² vs 0.87 fitted (Table 4),
   with young fast-growing leaves dominating (MaC2/MaC3/SoC2 >300 mm²). The reported numbers already use
   held-out frames for normalisation/time range and GT poses and GT fill for organs emerging in held-out frames
   (REPRODUCIBILITY §6). A fair protocol (normalise on retained frames, predict pose) would be worse.
3. **Emergence is not modelled.** New organs enter at their first observation, initialised from GT
   (`save_full_plant_from_organ_models`: organs without a model are copied from GT; per-organ rows skip them).
   Topology change is handled by composition + GT labels only.
4. **Pose is given, not predicted.** Leaves are placed with the per-day GT rigid transform even at held-out frames
   (`use_pose_anchoring=True`); full-plant metrics isolate shape/growth. Any method predicting plant-space
   positions must either adopt the same anchoring or will be penalised on EPE/CD by pose error.
5. **Requires GT instance segmentation with persistent ids**; with PSegNet+tracking labels the result is
   "not comparable" (Table 5*). Full-plant baselines without segmentation are 2-3 orders of magnitude worse,
   so most of the headline gap is the segmentation input, not the field.
6. **Advection of a fixed point set**: points are only moved (`x + v·dt`), never added, so fast-growing regions
   become sparse (paper limitation 1); Chamfer on test points penalises this weakly (GT test points still find a
   predicted neighbour) — a density-aware metric would expose it.
7. **Interpolation is not better than DPF**: on unseen intermediate frames DPF per-organ wins (10.24 / 4.00 vs
   10.29 / 5.82); the continuous field buys little between frames with 4-5 training timepoints.
8. **Short sequences underconstrain the field** (≤5 frames; SoH2); day-based t_norm with per-organ range means
   organs seen twice get t ∈ {0, 1} only.
9. **Evaluation noise**: EPE rests on 11-55 tips per plant, CUDA-nondeterministic Chamfer moves EPE by ~1 mm
   between identical runs; several published rows are 3-run or 1-seed. Beating 1.47 mm EPE by <0.3 mm would be
   within noise; CD (0.86 ± 0.02) is the stable target.
10. **Cost**: 50k steps per plant per run (1.33 M parameters), 5 runs × 8 plants; ~1.8 h/plant on this CPU, GPU
    time unreported. A single amortised model is cheaper at test time by construction.
11. Minor code/paper mismatches to remember when comparing: width 512 not 256; code L2 penalty 32× weaker than
    stated; noise σ in normalised units not mm; Table 4 mixes aggregations; stride-2 interpolation uses 35, not 45,
    organ sequences.

**Apples-to-apples checklist for a competing model:** use the 8 test plants of `data_splits.json`; load v2 PLYs
through PSA (`PlantSequencesDataset(..., version="v2")`), labels field `labels`; read
`point_splits/<plant>_split.json`, train on `train_indices`, score on `test_indices`; produce for each frame
t ≥ 1 (and for the organ's first frame nothing) a predicted point set per organ (and, for full-plant, placed with
the day's GT pose if you want their Table 1/5 semantics); write PLYs in GF's layout
(`pred_<plant>_D<dd>.ply` with `organ_label, is_train, is_leaf_tip, tip_organ_label`) and run their two
`compute_metrics_*.py` + `create_latex_table_*.py --per_sequence` unchanged, or re-implement
`CD = mean min² + mean min²` (mm²) on test points and tip EPE (mm) skipping first appearance, aggregate
frames → organs → runs → plants → unweighted mean over 8 plants, and verify `n_tips` sums 192/189.
