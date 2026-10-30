# Reproducibility

The paper reports every metric as a point estimate — there was no room for error bars.
This file gives the same paper tables with **mean ± standard deviation over runs**, the command
that trained each one, and the inconsistencies we found tracing the numbers back to the
runs that produced them.

> [!NOTE]
> Paper runs are not bitwise repeatable, because PyTorch3D's Chamfer backward uses CUDA
> atomics. The means here match the paper's, but rerunning a published command gives
> different numbers: CD only slightly, EPE noticeably more. `--deterministic` makes
> same-seed runs bitwise identical; see [inconsistency 4](#known-inconsistenciescaveats).

## How the numbers are computed

- **CD is the `cd_test` column**, Chamfer distance on held-out test points, not the `cd`
  column over all points. Leaf tips are only on test points so the `epe` column is on test points.
- **Full-plant tables:** average over timepoints for each (sequence, run), then mean and
  std across runs.
- **Per-organ tables:** average over timepoints for each (sequence, organ, run), average
  across organs within a plant, then mean and std across runs. Organ sequences with fewer
  than three frames are dropped (`--min_timepoints_per_organ 2`, since the first frame has
  no prediction), leaving **45 of 49** as the paper states (35 in the interpolation runs
  which train on every second frame).
- **Plants are weighted equally**, which is what the captions promise. Running the table
  scripts without `--per_sequence` instead weights organs equally and gives 0.80 where
  Table 2 reports 0.87; both are valid, but only the latter matches the paper.
- **The `Mean` column's ±** is the spread of that average *across runs*: the sequences are
  averaged within each run first, so it says how much the headline number moves if you
  retrain. 
- `n` is the number of runs each row averages.
- **Organs that cannot be predicted take their observed cloud.** Each organ is predicted by
  deforming its first observed cloud, so on that frame its prediction is the observation
  itself. An organ with no earlier training frame to start from (seen only once, or first
  appearing in a held-out frame) is filled with its ground truth. Otherwise the full-plant
  prediction would miss a whole organ and CD would jump for reasons unrelated to the
  model. This applies to every part-aware method alike, and only to the full-plant
  aggregation, since per-organ scores skip these frames.
- **Held-out frames use their observed poses.** As noted in the paper, each organ is predicted in its canonical
  frame and placed back with that day's ground-truth pose, interpolated and extrapolated
  frames included (`use_pose_anchoring=True`). Tables 3 and 4 therefore measure shape and
  growth, not where each organ will move.

## Known inconsistencies/caveats

Found by tracing each published value back to its run. None changes a conclusion, but all
of them matter if you try to reproduce a specific number.

1. **Run counts do not always match the captions.** Every caption says "averaged over five
   independent runs". It holds for the main `Ours (full)` row, but the interpolation and
   extrapolation `Ours` rows, all six ablations, and `Ours (per-organ MLPs)` in Tables 2
   and 3 come from 3-run relaunches. And the four full-plant baselines are effectively
   single-run: across the five runs of NDF, NVFi, DSR and CanFields in Tables 1 and 2 the
   jobs accidentally reused the same random seed, so NDF's and NVFi's EPE is bit-identical
   between them and CD varies only through resampling (CanFields' EPE varies for the reason
   in 8). The `n` column in each table below is the count on disk.
2. **Five rows used a regularizer that no longer exists.** `Ours (per-organ MLPs)` in
   Table 1 and the four robustness runs in the supplementary were trained with
   `use_spatial_smoothness=true`, cut from the code before release because it did not
   consistently improve results, so differences should be minimal and the degradation due
   to holes and other perturbations similar.
3. **The extrapolation table mixes two aggregations:** its COAP row is the full-plant
   aggregation, its `Ours` row the per-organ one. Table 4 below gives both rows under both
   aggregations and marks which pair the paper printed; GrowFields wins every column under
   either one taken consistently.
4. **Paper runs are not bitwise repeatable, because of PyTorch3D nondeterminism.** Its
   Chamfer backward accumulates with CUDA atomics, so even a same-seed rerun lands on a
   slightly different solution. EPE is especially sensitive to this. Chamfer never ties a
   predicted point to a particular true one, so tracked leaf tips can slide along the
   surface at no cost to the loss, and each sequence has only a few tips to average over. CD
   barely moves. `--deterministic` makes same-seed runs bitwise identical, at about 30% more
   training time and 1.7× the peak GPU memory, but the tables were trained without it. Read
   EPE with its std.
5. **Five details differ from the paper's text.** The code is what produced the tables.
   - **5a. Width:** GrowFields uses 512 hidden units (`configs/per_organ/shared_continuous.yaml`),
     not the supplementary's 256. The baselines use 256.
   - **5b. Latent penalty:** `weight.pow(2).mean()` averages over the 32 code dimensions as
     well as over organs, so the effective weight is 32x smaller than the paper's
     $\frac{1}{N}\sum_i \lVert\mathbf{z}_i\rVert_2^2$ with the same $\lambda_{code}$.
   - **5c. Noise σ:** it is added in each organ's normalized [-1, 1] coordinates, not in
     millimetres. The organ's largest extent spans a width of 2 there, so σ = 0.01 is 0.5%
     of it.
   - **5d. Large extrapolation errors:** the paper attributes them partly to organs emerging
     in the held-out frames, but those are filled with ground truth or not scored (see
     [above](#how-the-numbers-are-computed)). The large maize and sorghum errors come from
     young leaves that appeared shortly before the held-out frames and keep growing fast.
   - **5e. Leaf orientation uses the leaf-tip annotations.** The paper aligns each leaf's
     principal axis with Z but not which end is the tip. The code takes that from the stem,
     and lets an annotated tip override it where a frame has one. Tips are optional (leaves
     without any are oriented by the stem alone), only choose the sign of the axis, in the leaf
     alignment preprocessing. They are als used in the *w/ model segmentation* run.
6. **Extrapolation (Table 4) sees the held-out frames during normalization.** Each organ's
   spatial scale and time range are computed over all frames before the last three are held
   out, so growth in the held-out frames shrinks the training coordinates (up to 4.6x for a
   young leaf), and organs first appearing there take their ground truth (see [above](#how-the-numbers-are-computed)).
   GrowFields and COAP share this protocol, so the
   comparison between them holds, but the absolute extrapolation errors are optimistic.
   Normalizing on the retained frames alone would put held-out geometry far outside the
   trained range, so a proper extrapolation protocol is out of scope for this release.
7. **The *w/ model segmentation* row (Table 5) comes out slightly higher in EPE.** Two
   leaf-canonicalization bugs in [`plant-shape-analysis`](https://github.com/joaquin-gajardo/plant-shape-analysis)
   the paper ran are fixed in [`b531e06`](https://github.com/Joaquin-Gajardo/plant-shape-analysis/commit/b531e068013fd62e13fd67df89c7a38a48ef2265)
   (v0.4.0). They cancel each other out on ground-truth labels, so no other table changes, but
   on the automatic labels three leaves of TbS3 and Tm1H3 change orientation. Expect
   Tm1H3's EPE about 0.2 higher; the TbS3 change is within run-to-run spread, and CD is
   unaffected.
8. **The NDF, NVFi, DSR and CanFields rows are not exactly reproducible**, because of two
   bugs since fixed in the [CanFields
   fork](https://github.com/Joaquin-Gajardo/CanFields-TrackPlant3D). (1) Points were
   reordered before the train/test split, so the baselines trained on a random half that
   overlaps the test split (fixed in
   [`f435b94`](https://github.com/Joaquin-Gajardo/CanFields-TrackPlant3D/commit/f435b9459f4b8026adfa496d17a7b1347f5b633d));
   retraining DSR with the fix moves CD by under 1%. (2) CanFields' random Fourier
   frequencies were not saved in its checkpoints, so its meshes and EPE were computed with
   newly drawn ones (fixed in
   [`e7eb21d`](https://github.com/Joaquin-Gajardo/CanFields-TrackPlant3D/commit/e7eb21ddd094b44e27a1c079990a7a655b652ef7)).
   Below, its checkpoints re-evaluated with the training frequencies: five runs of one seed,
   which differ only for CanFields (PyTorch3D's KNN non-determinism, as in 4). Table 2's CD
   rises because five runs that had already diverged dominate the mean. Neither bug changes
   the ranking against GrowFields. Also, NDF's Tm1S1 leaf 8 in Table 2 is scored at epoch
   5000, the only epoch with metrics.

   | CanFields | n | Table 1 CD | Table 1 EPE | Table 2 CD | Table 2 EPE |
   | --- | --- | --- | --- | --- | --- |
   | Published | 5 | 615.7 ±52.1 | 44.4 ±1.8 | 138.0 ±52.4 | 29.4 ±2.1 |
   | Re-evaluated | 5 | 130.1 ±20.6 | 37.9 ±2.8 | 173.6 ±139.4 | 27.8 ±4.1 |

## Training the models

All commands are run from the repository root and assume the dataset is in place (see the
README). Each writes a timestamped directory under `results/fullplant_per_organ_batch/`,
which is what the evaluation commands at the end of this file take. `--n_runs` is set to
what the published row used, as recorded in the `n` column of each table.

The shipped configs carry the paper's hyperparameters, so only the ablations below override anything.

### GrowFields

```bash
# Tables 1, 2 and 5
python train_per_organ.py \
  --config configs/per_organ/shared_continuous.yaml \
  --split test --n_runs 5 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# Table 3, interpolation: train on every second timepoint
python train_per_organ.py \
  --config configs/per_organ/shared_continuous.yaml \
  --split test --n_runs 3 --train_timepoint_stride 2 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# Table 4, extrapolation: hold out the last three timepoints. SoH2 is left out
# because it has only five timepoints in total.
python train_per_organ.py \
  --config configs/per_organ/shared_continuous.yaml \
  --n_runs 3 --train_holdout_last_n 3 \
  --sequences maize_control_plant2 maize_control_plant3 sorghum_control_plant2 \
              tobacco_control_plant1 tobacco_shade_plant3 tomato1_heat_plant3 \
              tomato1_shade_plant1 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits
```

`--sequences` and `--split` are mutually exclusive: `--split` wins if both are given, so
the extrapolation command must not pass it.

The supplementary robustness runs are the Tables 1/2/5 command plus one perturbation flag
— `--input_noise_std 0.01` or `0.05`, or `--train_holes_n 5 --train_holes_k 15` or `25`.

Each Table 5 ablation is the same command with `--n_runs 3` and one change:

| Ablation | Change |
| --- | --- |
| w/o canonicalisation | `dataset.aligned_leaves=false` |
| w/o latent code | `default_model.conditioning=none` |
| w/ FiLM conditioning | `default_model.conditioning=film` |
| w/o multi-step CD loss | `default_model.use_multistep_loss=false` |
| w/o L_code | `default_model.embed_reg_lambda=0` |
| w/ model segmentation | `dataset.version=v2-autoseg dataset.label_field=predicted_labels dataset.exclude_sequences=["sorghum_heat_plant1"]` |

The model-segmentation labels ship with the dataset (`v2-autoseg`); how they were made
(PSegNet retrained on all sequences, then TrackPlant3D tracking) and how to regenerate
them is in plant-shape-analysis's README, under *Automatic organ labels*.
`sorghum_heat_plant1` is excluded because its predicted labels have no stem on some
days, which leaf alignment needs (it's not a val/test sequence, but all sequences get aligned when loading the dataset).

### Part-aware baselines

```bash
# COAP -- Tables 1 and 2
python train_per_organ.py \
  --config configs/per_organ/encoder_decoder.yaml \
  --split test --n_runs 5 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# COAP -- Table 3, interpolation
python train_per_organ.py \
  --config configs/per_organ/encoder_decoder.yaml \
  --split test --n_runs 5 --train_timepoint_stride 2 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# COAP -- Table 4, extrapolation
python train_per_organ.py \
  --config configs/per_organ/encoder_decoder.yaml \
  --n_runs 5 --train_holdout_last_n 3 \
  --sequences maize_control_plant2 maize_control_plant3 sorghum_control_plant2 \
              tobacco_control_plant1 tobacco_shade_plant3 tomato1_heat_plant3 \
              tomato1_shade_plant1 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# Ours (per-organ MLPs) -- Table 2
python train_per_organ.py \
  --config configs/per_organ/continuous.yaml \
  --split test --n_runs 3 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# Ours (per-organ MLPs) -- Table 3, interpolation
python train_per_organ.py \
  --config configs/per_organ/continuous.yaml \
  --split test --n_runs 3 --train_timepoint_stride 2 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# DPF (per-organ) -- Table 2; the shipped config needs no overrides
python train_per_organ.py \
  --config configs/per_organ/pairwise.yaml \
  --split test --n_runs 5 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits

# DPF (per-organ) -- Table 3, interpolation
python train_per_organ.py \
  --config configs/per_organ/pairwise.yaml \
  --split test --n_runs 5 --train_timepoint_stride 2 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits
```

### DPF, single whole-plant model (Table 1)

One model for the whole plant rather than one per organ, so it uses `train_fullplant.py`
and writes to `results/fullplant_single_model_batch/`:

```bash
python train_fullplant.py \
  --config configs/full_plant/pairwise.yaml \
  --split test --n_runs 5 \
  --points_split_dir configs/datasets/TrackPlant3D/point_splits
```

The published Table 1 row is this model's **pre-finetune** evaluation, read from
`fullplant_metrics_pre.csv`; post-finetune is worse (CD 4.75, EPE 16.99).

NDF, NVFi, DSR and CanFields were run from a
[separate repository](https://github.com/Joaquin-Gajardo/CanFields-TrackPlant3D), at commit
[`a865e96`](https://github.com/Joaquin-Gajardo/CanFields-TrackPlant3D/commit/a865e966c4c86d02dc63772f097b518a2bfa3500), and cannot be trained from here.

## Full tables

#### Table 1 — full-plant reconstruction

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPF (pre-finetune) | 5 | 8.17 ±4.65 | 5.00 ±1.65 | 1.21 ±0.09 | 0.92 ±0.07 | 0.36 ±0.03 | 0.46 ±0.09 | 0.17 ±0.01 | 3.47 ±1.61 | 2.47 ±0.57 |
| COAP | 5 | 4.84 ±1.03 | 2.48 ±0.44 | 1.31 ±0.19 | 1.02 ±0.09 | 1.03 ±0.10 | 0.65 ±0.13 | 0.38 ±0.16 | 1.82 ±0.91 | 1.69 ±0.09 |
| Ours (per-organ MLPs) | 5 | 2.41 ±0.08 | 1.52 ±0.03 | 0.87 ±0.08 | 1.01 ±0.04 | 0.70 ±0.03 | 0.48 ±0.03 | 0.13 ±0.01 | 0.33 ±0.01 | 0.93 ±0.02 |
| **Ours (full)** | 5 | 2.51 ±0.15 | 1.43 ±0.05 | 0.69 ±0.02 | 0.74 ±0.03 | 0.59 ±0.02 | 0.45 ±0.03 | 0.12 ±0.01 | 0.32 ±0.02 | 0.86 ±0.02 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPF (pre-finetune) | 5 | 23.18 ±6.37 | 5.30 ±1.29 | 5.78 ±0.75 | 57.22 ±1.51 | 7.58 ±1.13 | 16.21 ±3.61 | 1.93 ±1.54 | 3.97 ±1.18 | 15.15 ±0.52 |
| COAP | 5 | 7.57 ±2.61 | 3.87 ±0.93 | 7.76 ±5.13 | 5.61 ±3.58 | 3.22 ±0.56 | 2.55 ±1.08 | 1.40 ±0.30 | 1.88 ±0.39 | 4.23 ±0.83 |
| Ours (per-organ MLPs) | 5 | 3.68 ±1.10 | 2.07 ±0.19 | 1.47 ±0.37 | 3.44 ±0.62 | 3.28 ±1.10 | 2.67 ±0.75 | 0.74 ±0.13 | 1.59 ±0.43 | 2.37 ±0.21 |
| **Ours (full)** | 5 | 2.38 ±0.17 | 2.01 ±0.34 | 1.34 ±0.12 | 0.97 ±0.15 | 1.66 ±0.24 | 1.30 ±0.32 | 0.80 ±0.12 | 1.34 ±0.10 | 1.47 ±0.11 |


#### Table 2 — per-organ reconstruction

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPF | 5 | 2.32 ±0.07 | 1.67 ±0.02 | 1.30 ±0.13 | 0.87 ±0.02 | 0.69 ±0.04 | 0.56 ±0.03 | 0.30 ±0.04 | 0.68 ±0.08 | 1.05 ±0.02 |
| COAP | 5 | 4.17 ±0.88 | 2.55 ±0.46 | 2.22 ±0.41 | 0.93 ±0.10 | 0.99 ±0.08 | 0.64 ±0.13 | 0.42 ±0.12 | 1.60 ±0.69 | 1.69 ±0.11 |
| Ours (per-organ MLPs) | 3 | 2.10 ±0.06 | 1.52 ±0.03 | 1.13 ±0.09 | 0.96 ±0.09 | 0.56 ±0.01 | 0.50 ±0.04 | 0.24 ±0.03 | 0.51 ±0.02 | 0.94 ±0.02 |
| **Ours (full)** | 5 | 2.26 ±0.17 | 1.48 ±0.08 | 0.93 ±0.01 | 0.64 ±0.02 | 0.49 ±0.01 | 0.45 ±0.03 | 0.20 ±0.03 | 0.46 ±0.02 | 0.87 ±0.02 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPF | 5 | 2.05 ±0.33 | 2.60 ±1.09 | 1.43 ±0.62 | 1.43 ±0.35 | 3.18 ±0.75 | 1.57 ±0.40 | 2.01 ±0.46 | 2.23 ±0.34 | 2.06 ±0.20 |
| COAP | 5 | 8.20 ±3.24 | 5.68 ±1.44 | 16.63 ±14.52 | 8.82 ±4.99 | 5.01 ±0.78 | 2.02 ±0.65 | 1.75 ±0.57 | 1.93 ±0.64 | 6.26 ±1.91 |
| Ours (per-organ MLPs) | 3 | 3.47 ±0.58 | 2.89 ±0.11 | 1.59 ±0.13 | 2.34 ±0.40 | 4.13 ±0.79 | 1.88 ±0.22 | 1.42 ±0.16 | 1.84 ±0.15 | 2.44 ±0.21 |
| **Ours (full)** | 5 | 2.59 ±0.22 | 2.46 ±0.49 | 1.51 ±0.14 | 0.85 ±0.18 | 2.04 ±0.40 | 1.20 ±0.23 | 0.98 ±0.16 | 1.33 ±0.09 | 1.62 ±0.14 |


#### Table 3 — full-plant interpolation, training frames

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 3.20 ±0.56 | 0.82 ±0.10 | 3.04 ±3.07 | 1.11 ±0.53 | 0.91 ±0.09 | 0.56 ±0.06 | 0.15 ±0.01 | 0.59 ±0.17 | 1.30 ±0.42 |
| DPF (per-organ) | 5 | 2.62 ±0.13 | 0.77 ±0.01 | 0.51 ±0.04 | 0.63 ±0.02 | 0.83 ±0.03 | 0.57 ±0.05 | 0.13 ±0.00 | 0.41 ±0.01 | 0.81 ±0.02 |
| Ours (per-organ MLPs) | 3 | 2.59 ±0.13 | 0.97 ±0.06 | 0.59 ±0.03 | 0.72 ±0.07 | 0.77 ±0.04 | 0.47 ±0.04 | 0.12 ±0.00 | 0.37 ±0.02 | 0.82 ±0.02 |
| **Ours (full)** | 3 | 2.24 ±0.14 | 0.70 ±0.01 | 0.42 ±0.01 | 0.44 ±0.01 | 0.57 ±0.02 | 0.39 ±0.02 | 0.10 ±0.00 | 0.35 ±0.01 | 0.65 ±0.02 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 11.87 ±7.36 | 3.08 ±0.90 | 9.60 ±6.28 | 2.20 ±2.40 | 2.39 ±0.99 | 2.34 ±0.77 | 2.46 ±0.24 | 1.40 ±0.04 | 4.42 ±0.94 |
| DPF (per-organ) | 5 | 1.94 ±0.32 | 1.67 ±0.39 | 0.85 ±0.17 | 2.87 ±0.84 | 1.98 ±0.29 | 1.58 ±0.46 | 1.26 ±0.27 | 1.67 ±0.43 | 1.73 ±0.21 |
| Ours (per-organ MLPs) | 3 | 2.21 ±0.59 | 5.49 ±0.90 | 1.28 ±0.46 | 4.73 ±1.07 | 3.44 ±0.23 | 2.05 ±0.46 | 1.14 ±0.21 | 1.93 ±0.22 | 2.78 ±0.21 |
| **Ours (full)** | 3 | 2.14 ±0.06 | 1.72 ±0.20 | 0.71 ±0.10 | 0.56 ±0.06 | 5.39 ±0.90 | 1.70 ±1.02 | 0.58 ±0.12 | 1.24 ±0.14 | 1.75 ±0.04 |


#### Table 3 — full-plant interpolation, interpolated frames

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 19.30 ±0.59 | 9.57 ±0.38 | 25.42 ±6.30 | 31.07 ±1.16 | 4.16 ±0.58 | 1.88 ±0.05 | 0.83 ±0.01 | 1.62 ±0.08 | 11.73 ±0.80 |
| DPF (per-organ) | 5 | 17.75 ±0.27 | 8.93 ±0.04 | 18.38 ±0.35 | 29.38 ±0.21 | 3.23 ±0.05 | 1.88 ±0.05 | 0.89 ±0.01 | 1.47 ±0.01 | 10.24 ±0.05 |
| Ours (per-organ MLPs) | 3 | 18.00 ±0.07 | 9.47 ±0.05 | 19.17 ±0.19 | 29.42 ±0.21 | 2.84 ±0.07 | 1.76 ±0.02 | 0.91 ±0.00 | 1.53 ±0.08 | 10.39 ±0.01 |
| **Ours (full)** | 3 | 18.01 ±0.26 | 9.18 ±0.04 | 19.53 ±0.39 | 28.60 ±0.31 | 2.93 ±0.05 | 1.75 ±0.04 | 0.89 ±0.01 | 1.42 ±0.02 | 10.29 ±0.05 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 14.63 ±5.06 | 12.97 ±0.63 | 10.20 ±5.93 | 9.97 ±0.43 | 3.90 ±0.56 | 2.82 ±0.68 | 2.55 ±0.17 | 1.63 ±0.06 | 7.33 ±0.70 |
| DPF (per-organ) | 5 | 8.06 ±0.11 | 4.72 ±0.03 | 5.26 ±0.09 | 5.74 ±0.15 | 3.13 ±0.22 | 1.89 ±0.38 | 1.63 ±0.28 | 1.60 ±0.28 | 4.00 ±0.03 |
| Ours (per-organ MLPs) | 3 | 8.04 ±0.20 | 13.32 ±1.45 | 5.55 ±0.12 | 9.73 ±0.09 | 4.59 ±0.18 | 2.57 ±0.41 | 1.56 ±0.21 | 2.28 ±0.13 | 5.96 ±0.24 |
| **Ours (full)** | 3 | 7.76 ±0.10 | 11.82 ±0.27 | 5.34 ±0.06 | 9.69 ±0.02 | 6.49 ±1.02 | 2.48 ±0.63 | 1.12 ±0.13 | 1.88 ±0.10 | 5.82 ±0.11 |


#### Table 4 — extrapolation, training frames

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP, full-plant agg. *(paper)* | 5 | 1.34 ±0.11 | 0.57 ±0.02 | 0.49 ±0.04 | 0.18 ±0.01 | 0.26 ±0.05 | 0.22 ±0.09 | 0.54 ±0.21 | 0.51 ±0.03 |
| COAP, per-organ agg. | 5 | 1.33 ±0.12 | 0.51 ±0.02 | 0.55 ±0.05 | 0.17 ±0.01 | 0.29 ±0.06 | 0.27 ±0.07 | 0.60 ±0.23 | 0.53 ±0.04 |
| **Ours**, full-plant agg. | 3 | 1.04 ±0.03 | 0.46 ±0.02 | 0.33 ±0.01 | 0.15 ±0.01 | 0.18 ±0.01 | 0.10 ±0.00 | 0.22 ±0.01 | 0.35 ±0.01 |
| **Ours**, per-organ agg. *(paper)* | 3 | 1.06 ±0.04 | 0.40 ±0.02 | 0.39 ±0.01 | 0.15 ±0.01 | 0.18 ±0.01 | 0.14 ±0.00 | 0.30 ±0.02 | 0.37 ±0.01 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP, full-plant agg. *(paper)* | 5 | 12.26 ±5.35 | 5.01 ±2.42 | 1.02 ±0.14 | 1.13 ±0.28 | 1.24 ±0.39 | 1.15 ±0.15 | 1.54 ±0.14 | 3.34 ±0.72 |
| COAP, per-organ agg. | 5 | 11.96 ±5.52 | 5.01 ±2.42 | 1.59 ±0.25 | 1.28 ±0.28 | 1.39 ±0.49 | 1.39 ±0.21 | 1.64 ±0.13 | 3.47 ±0.71 |
| **Ours**, full-plant agg. | 3 | 1.71 ±0.14 | 0.92 ±0.23 | 0.83 ±0.08 | 0.83 ±0.23 | 1.47 ±0.33 | 0.62 ±0.09 | 1.01 ±0.16 | 1.06 ±0.05 |
| **Ours**, per-organ agg. *(paper)* | 3 | 1.81 ±0.15 | 0.92 ±0.23 | 1.08 ±0.21 | 0.80 ±0.24 | 1.67 ±0.40 | 0.79 ±0.13 | 1.33 ±0.33 | 1.20 ±0.06 |


#### Table 4 — extrapolation, extrapolated frames

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP, full-plant agg. *(paper)* | 5 | 768.08 ±211.38 | 1632.07 ±366.84 | 620.43 ±98.43 | 79.27 ±11.47 | 15.82 ±2.46 | 8.59 ±3.02 | 33.40 ±12.15 | 451.09 ±36.69 |
| COAP, per-organ agg. | 5 | 750.07 ±156.28 | 4216.24 ±1286.59 | 931.03 ±98.50 | 77.58 ±11.09 | 27.54 ±10.79 | 10.14 ±5.00 | 32.54 ±10.88 | 863.59 ±183.04 |
| **Ours**, full-plant agg. | 3 | 301.13 ±169.47 | 217.32 ±28.69 | 233.92 ±70.55 | 52.86 ±0.35 | 6.74 ±0.41 | 3.01 ±0.85 | 8.97 ±1.36 | 117.71 ±35.55 |
| **Ours**, per-organ agg. *(paper)* | 3 | 356.13 ±94.33 | 321.08 ±15.85 | 337.39 ±146.59 | 15.38 ±1.24 | 8.09 ±1.04 | 2.80 ±0.76 | 6.56 ±1.70 | 149.63 ±19.65 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP, full-plant agg. *(paper)* | 5 | 47.17 ±14.06 | 103.33 ±1.85 | 42.77 ±4.78 | 9.23 ±2.19 | 5.04 ±0.68 | 5.92 ±2.22 | 6.25 ±0.92 | 31.39 ±2.77 |
| COAP, per-organ agg. | 5 | 47.17 ±14.06 | 89.10 ±10.20 | 42.77 ±4.78 | 10.37 ±2.38 | 5.05 ±0.51 | 5.56 ±1.95 | 6.25 ±0.92 | 29.46 ±3.22 |
| **Ours**, full-plant agg. | 3 | 20.96 ±4.76 | 89.46 ±1.46 | 27.04 ±1.83 | 3.57 ±0.73 | 4.43 ±0.38 | 2.60 ±0.68 | 4.46 ±0.50 | 21.79 ±0.54 |
| **Ours**, per-organ agg. *(paper)* | 3 | 20.96 ±4.76 | 69.77 ±0.56 | 27.04 ±1.83 | 4.06 ±0.85 | 4.14 ±0.29 | 2.52 ±0.58 | 4.46 ±0.50 | 18.99 ±0.68 |


Two details specific to Table 4: among the per-organ rows the training-frames ones apply
the ≥3-frame organ filter while the extrapolated-frames ones do not, since every organ has
at most three held-out frames; and SoH2 is excluded throughout, having only five
timepoints.
#### Table 5 — ablations (full-plant)

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ours (full)** | 5 | 2.51 ±0.15 | 1.43 ±0.05 | 0.69 ±0.02 | 0.74 ±0.03 | 0.59 ±0.02 | 0.45 ±0.03 | 0.12 ±0.01 | 0.32 ±0.02 | 0.86 ±0.02 |
| w/o canonicalisation | 3 | 2.76 ±0.19 | 1.58 ±0.05 | 0.93 ±0.18 | 0.79 ±0.05 | 0.60 ±0.01 | 0.47 ±0.01 | 0.21 ±0.03 | 0.34 ±0.00 | 0.96 ±0.05 |
| w/o latent code | 3 | 2.77 ±0.09 | 1.76 ±0.02 | 0.83 ±0.05 | 1.00 ±0.04 | 0.83 ±0.03 | 0.61 ±0.01 | 0.15 ±0.00 | 0.47 ±0.01 | 1.05 ±0.01 |
| w/o multi-step CD loss | 3 | 2.87 ±0.37 | 1.45 ±0.07 | 0.68 ±0.00 | 0.75 ±0.04 | 0.58 ±0.02 | 0.46 ±0.02 | 0.14 ±0.01 | 0.38 ±0.02 | 0.91 ±0.06 |
| w/o L_code | 3 | 2.52 ±0.08 | 1.54 ±0.13 | 0.71 ±0.05 | 0.74 ±0.01 | 0.61 ±0.02 | 0.45 ±0.03 | 0.12 ±0.01 | 0.32 ±0.01 | 0.88 ±0.02 |
| w/ FiLM conditioning | 3 | 2.62 ±0.10 | 1.67 ±0.04 | 0.83 ±0.03 | 0.92 ±0.04 | 0.67 ±0.03 | 0.47 ±0.01 | 0.13 ±0.00 | 0.33 ±0.01 | 0.96 ±0.02 |
| w/ model segmentation* | 3 | 1.77 ±0.20 | 1.21 ±0.01 | 0.68 ±0.02 | 0.48 ±0.00 | 2.01 ±0.26 | 0.36 ±0.04 | 0.12 ±0.00 | 0.27 ±0.01 | 0.86 ±0.06 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ours (full)** | 5 | 2.38 ±0.17 | 2.01 ±0.34 | 1.34 ±0.12 | 0.97 ±0.15 | 1.66 ±0.24 | 1.30 ±0.32 | 0.80 ±0.12 | 1.34 ±0.10 | 1.47 ±0.11 |
| w/o canonicalisation | 3 | 3.14 ±0.19 | 2.28 ±0.14 | 3.41 ±0.93 | 13.24 ±0.47 | 5.79 ±0.69 | 1.40 ±0.52 | 3.18 ±0.30 | 1.42 ±0.13 | 4.23 ±0.26 |
| w/o latent code | 3 | 8.04 ±4.31 | 3.70 ±0.29 | 1.43 ±0.22 | 3.32 ±1.60 | 3.72 ±0.98 | 3.31 ±2.14 | 1.28 ±0.25 | 2.72 ±0.50 | 3.44 ±0.91 |
| w/o multi-step CD loss | 3 | 2.79 ±0.29 | 2.21 ±0.18 | 1.37 ±0.06 | 1.04 ±0.06 | 3.29 ±1.45 | 1.65 ±0.07 | 1.34 ±0.56 | 1.48 ±0.19 | 1.89 ±0.21 |
| w/o L_code | 3 | 2.46 ±0.17 | 2.14 ±0.23 | 1.38 ±0.07 | 0.99 ±0.14 | 3.42 ±0.41 | 1.06 ±0.41 | 0.85 ±0.20 | 1.62 ±0.21 | 1.74 ±0.04 |
| w/ FiLM conditioning | 3 | 5.74 ±3.54 | 1.93 ±0.03 | 1.39 ±0.10 | 3.32 ±0.26 | 3.04 ±0.39 | 2.04 ±0.75 | 0.92 ±0.16 | 1.35 ±0.28 | 2.47 ±0.40 |
| w/ model segmentation* | 3 | 2.17 ±0.22 | 1.94 ±0.27 | 1.32 ±0.09 | 0.80 ±0.05 | 1.59 ±0.28 | 1.55 ±0.25 | 0.61 ±0.07 | 1.36 ±0.23 | 1.42 ±0.04 |

\* `w/ model segmentation` is not directly comparable to the ground-truth-segmentation
rows: the organ decomposition differs, so the averaging and the number of leaf-tip
evaluations differ too. It is a reference value, as the paper's caption says.

#### Supplementary — per-organ interpolation, training frames

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 2.94 ±0.53 | 0.82 ±0.11 | 2.77 ±2.43 | 1.28 ±0.36 | 0.71 ±0.07 | 0.51 ±0.09 | 0.19 ±0.01 | 0.77 ±0.18 | 1.25 ±0.32 |
| DPF (per-organ) | 5 | 2.39 ±0.09 | 0.77 ±0.01 | 0.60 ±0.04 | 0.69 ±0.02 | 0.65 ±0.04 | 0.49 ±0.05 | 0.17 ±0.00 | 0.63 ±0.03 | 0.80 ±0.02 |
| Ours (per-organ MLPs) | 3 | 2.34 ±0.11 | 0.79 ±0.01 | 0.64 ±0.03 | 0.69 ±0.01 | 0.53 ±0.02 | 0.38 ±0.03 | 0.16 ±0.01 | 0.57 ±0.04 | 0.76 ±0.01 |
| **Ours (full)** | 3 | 2.06 ±0.10 | 0.67 ±0.02 | 0.48 ±0.01 | 0.52 ±0.01 | 0.42 ±0.01 | 0.32 ±0.02 | 0.14 ±0.00 | 0.53 ±0.02 | 0.64 ±0.02 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 13.36 ±9.81 | 3.31 ±1.28 | 12.56 ±8.66 | 0.84 ±0.55 | 2.39 ±0.99 | 2.62 ±0.79 | 2.64 ±0.39 | 1.69 ±0.10 | 4.93 ±1.41 |
| DPF (per-organ) | 5 | 2.08 ±0.38 | 0.72 ±0.15 | 0.96 ±0.19 | 1.34 ±1.06 | 1.98 ±0.29 | 1.62 ±0.51 | 1.44 ±0.22 | 1.86 ±0.36 | 1.50 ±0.16 |
| Ours (per-organ MLPs) | 3 | 2.40 ±0.79 | 3.58 ±2.30 | 1.81 ±0.62 | 0.69 ±0.24 | 3.44 ±0.23 | 1.86 ±0.51 | 1.65 ±0.28 | 2.32 ±0.26 | 2.22 ±0.38 |
| **Ours (full)** | 3 | 2.24 ±0.07 | 1.39 ±0.34 | 0.76 ±0.19 | 0.31 ±0.08 | 5.39 ±0.90 | 1.84 ±1.00 | 0.67 ±0.18 | 1.54 ±0.21 | 1.77 ±0.13 |


#### Supplementary — per-organ interpolation, interpolated frames

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 21.00 ±0.70 | 10.54 ±0.39 | 22.28 ±4.15 | 35.00 ±1.54 | 3.57 ±0.54 | 2.96 ±0.19 | 0.68 ±0.01 | 1.48 ±0.07 | 12.19 ±0.59 |
| DPF (per-organ) | 5 | 20.29 ±0.38 | 9.96 ±0.09 | 18.93 ±0.35 | 34.38 ±0.57 | 3.09 ±0.11 | 2.68 ±0.10 | 0.72 ±0.01 | 1.37 ±0.03 | 11.43 ±0.09 |
| Ours (per-organ MLPs) | 3 | 20.84 ±0.30 | 10.03 ±0.18 | 18.99 ±0.04 | 34.20 ±0.17 | 2.71 ±0.04 | 2.49 ±0.05 | 0.73 ±0.01 | 1.37 ±0.08 | 11.42 ±0.08 |
| **Ours (full)** | 3 | 21.10 ±0.50 | 10.13 ±0.10 | 19.68 ±0.09 | 32.97 ±0.72 | 2.78 ±0.08 | 2.67 ±0.08 | 0.71 ±0.01 | 1.31 ±0.04 | 11.42 ±0.12 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COAP | 5 | 16.15 ±6.75 | 7.36 ±0.99 | 11.59 ±5.03 | 4.11 ±0.61 | 3.97 ±0.56 | 2.77 ±0.81 | 2.77 ±0.22 | 2.04 ±0.07 | 6.34 ±0.93 |
| DPF (per-organ) | 5 | 8.77 ±0.14 | 5.44 ±0.05 | 6.64 ±0.09 | 3.50 ±0.30 | 3.23 ±0.23 | 2.21 ±0.47 | 1.69 ±0.19 | 1.98 ±0.40 | 4.18 ±0.05 |
| Ours (per-organ MLPs) | 3 | 8.69 ±0.30 | 8.16 ±2.08 | 6.88 ±0.24 | 3.36 ±0.17 | 4.67 ±0.20 | 2.43 ±0.48 | 1.95 ±0.19 | 3.16 ±0.26 | 4.91 ±0.32 |
| **Ours (full)** | 3 | 8.35 ±0.13 | 5.82 ±0.20 | 6.66 ±0.03 | 3.52 ±0.05 | 6.54 ±0.94 | 2.28 ±0.86 | 1.22 ±0.16 | 2.30 ±0.20 | 4.59 ±0.08 |


#### Supplementary — robustness to input perturbations (full-plant)

CD (mm²) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unperturbed (Table 1) | 5 | 2.51 ±0.15 | 1.43 ±0.05 | 0.69 ±0.02 | 0.74 ±0.03 | 0.59 ±0.02 | 0.45 ±0.03 | 0.12 ±0.01 | 0.32 ±0.02 | 0.86 ±0.02 |
| Noise σ = 0.01 | 5 | 3.03 ±0.38 | 2.02 ±0.31 | 1.04 ±0.06 | 0.88 ±0.04 | 0.63 ±0.04 | 0.54 ±0.06 | 0.13 ±0.00 | 0.34 ±0.01 | 1.08 ±0.04 |
| Noise σ = 0.05 | 5 | 15.46 ±3.39 | 10.50 ±2.38 | 6.79 ±0.19 | 3.40 ±0.25 | 3.13 ±1.28 | 1.43 ±0.30 | 0.27 ±0.02 | 0.96 ±0.10 | 5.24 ±0.40 |
| Holes K = 15 | 5 | 2.79 ±0.20 | 1.61 ±0.13 | 0.85 ±0.03 | 0.85 ±0.02 | 0.65 ±0.04 | 0.47 ±0.01 | 0.13 ±0.00 | 0.34 ±0.01 | 0.96 ±0.04 |
| Holes K = 25 | 5 | 3.01 ±0.05 | 1.81 ±0.09 | 1.03 ±0.12 | 0.96 ±0.08 | 0.69 ±0.05 | 0.52 ±0.01 | 0.14 ±0.01 | 0.37 ±0.01 | 1.07 ±0.03 |

EPE (mm) ↓

| Method | n | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unperturbed (Table 1) | 5 | 2.38 ±0.17 | 2.01 ±0.34 | 1.34 ±0.12 | 0.97 ±0.15 | 1.66 ±0.24 | 1.30 ±0.32 | 0.80 ±0.12 | 1.34 ±0.10 | 1.47 ±0.11 |
| Noise σ = 0.01 | 5 | 2.74 ±0.17 | 4.33 ±1.08 | 1.98 ±0.34 | 1.38 ±0.62 | 2.25 ±0.66 | 1.57 ±0.53 | 0.96 ±0.15 | 1.34 ±0.12 | 2.07 ±0.26 |
| Noise σ = 0.05 | 5 | 8.76 ±1.84 | 11.05 ±2.17 | 8.00 ±0.79 | 3.99 ±0.61 | 2.42 ±0.76 | 2.58 ±0.55 | 1.11 ±0.11 | 1.88 ±0.17 | 4.97 ±0.37 |
| Holes K = 15 | 5 | 2.50 ±0.12 | 2.05 ±0.38 | 1.84 ±0.70 | 1.40 ±0.41 | 2.50 ±1.15 | 2.06 ±1.41 | 1.05 ±0.24 | 1.45 ±0.27 | 1.86 ±0.23 |
| Holes K = 25 | 5 | 2.99 ±0.61 | 2.04 ±0.33 | 2.03 ±0.80 | 1.74 ±0.50 | 2.76 ±0.87 | 1.79 ±0.45 | 0.93 ±0.20 | 1.39 ±0.30 | 1.96 ±0.17 |

## Regenerating these numbers

### The tables

Every row above, standard deviations included, comes out of the repository's own scripts.
The wrappers compute the metrics and emit the table in one step:

```bash
# Tables 1 and 5, robustness -- full-plant
bash scripts/compute_metrics_and_latex_table_fullplant.sh <experiment>

# Table 2 -- per-organ
bash scripts/compute_metrics_and_latex_table_per_organ.sh <experiment>

# Table 3 -- add --interpolation for the training + interpolated pair of sub-rows
bash scripts/compute_metrics_and_latex_table_fullplant.sh <experiment> --interpolation

# Table 4 -- extrapolation
bash scripts/compute_metrics_and_latex_table_per_organ.sh <experiment> --extrapolation
```

If an extrapolation directory does contain SoH2, add `--skip_sequences sorghum_highlight_plant2`.

### Statistical significance

The paper's significance claim comes from the same Table 2 metrics:

```bash
python scripts/compute_statistical_significance.py \
  --ours <ours_experiment>/all_metrics.csv \
  --coap <coap_experiment>/all_metrics.csv \
  --dpf  <dpf_experiment>/all_metrics.csv
```

It aggregates the three CSVs exactly as the Table 2 pipeline does, tests the same 45 organ
sequences, and reports Shapiro-Wilk on the paired differences alongside the one-sided
Wilcoxon p-values: GrowFields over COAP on CD and EPE (both *p* < 0.001) and over DPF on
CD (*p* < 0.001) and EPE (*p* = 0.004).

### Checking a reproduction

**Leaf-tip counts are the quickest integrity check.** EPE is evaluated only on annotated leaf tips, and how many of them are scored is fixed: **192** tip measurements for the full-plant tables and **189** for the per-organ ones, the three lost to the >=3-frame organ filter. Both are exact and identical in every run, so they catch a mis-specified split or a dropped sequence immediately:

```python
import pandas as pd

fp = pd.read_csv("<experiment>/fullplant_metrics.csv")
po = pd.read_csv("<experiment>/all_metrics.csv")
keep = po.groupby(["sequence", "organ_id", "run"]).filter(lambda g: len(g) >= 2)

print(int(fp[fp.run == 1].n_tips.sum()))      # 192, full-plant tables
print(int(keep[keep.run == 1].n_tips.sum()))  # 189, per-organ tables
```

> [!NOTE]
The `>= 2` is correct and should not be "fixed" to 3. The paper says "at least three
frames", counting the anchor frame; but the results are based on *evaluated* timepoints, and the
first frame carries no prediction and so no error. `>= 2` reproduces the paper's own chain
exactly, i.e. 49 organ sequences down to 45, and 192 tip measurements down to 189.

Sum `n_tips` rather than counting rows -- in the full-plant frame `n_tips` runs from 0 to 8, so counting rows returns the 70 timepoints instead of 192. Without the filter the per-organ frame also sums to 192: both tables score the same tips, and the filter removes the three that sit on organs with too few frames.

This matters because EPE rests on very little data (between 11 and 55 tip measurements
per sequence, against thousands of points per Chamfer evaluation). That is part of why it is
the noisier of the two metrics, and why a wrong tip count distorts it so much more than it
distorts CD. `compute_epe` warns if the predicted and ground-truth tip counts ever disagree
(`growfields/utils/metrics.py:137`).

### Tests
**The test suite checks the invariants the evaluation depends on.** It runs against a
completed experiment directory rather than being self-contained:

```bash
pytest tests/ -v --experiment-dir <experiment>/sequences
```

It verifies that leaf tips are always test points and never training points, that there is
at most one tip per organ, that the saved ground-truth PLYs match what the dataloader
returns for points, labels and tips, and that point counts agree with the split files. See
`tests/README.md`.

`pytest tests/test_standalone.py` needs no experiment: it checks that `--deterministic`
trains with the same loss and gradient as PyTorch3D, that pose interpolation stays rigid,
and that every organ alignment is a rotation, so per-organ metrics (canonical frame) and
full-plant metrics (plant space) measure the same distances.
