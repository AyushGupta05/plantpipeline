# GrowFields (+ nearest source), RichardsNet and MonoGrow on OUR unified data (Pheno4D, Sugar4D)

GrowFields (GF) only knows TrackPlant3D v2 PLYs read through plant-shape-analysis (PSA). This folder converts
`data/unified` (Pheno4D 14 plants, Sugar4D 48 plants; 8,192 points/scan, labels consistent over time) into a parallel
TrackPlant3D-v2-shaped tree and runs the vendored, unmodified GF code on it with CWD = this folder (GF's configs use
the relative `dataset_path: data/TrackPlant3D/versions`, PSA appends `v2/gt_corrected_v2/<crop>/`, and discovers
sequences from the file names `\d+_(.+)_D\d+\.ply`). Nothing under `experiments/growfields/growfields/`,
`experiments/monogrow/`, `experiments/richards/` or `plantpipeline/` is changed; venvs are reused
(`../growfields/.venv` for GF, `../monogrow/.venv` for the baselines and `metrics.py`).

```
convert.py              data/unified npz -> data/TrackPlant3D/versions/v2/gt_corrected_v2/<dataset>_<species>/<N>_<seq>_D<dd>.ply
                        + point_splits/<seq>_split.json (GF's 50/50 split, seed 42) + data_splits.json + MANIFEST.md
run_gf.sh               GF shared_continuous / monotone_continuous on one sequence, 3 protocols (campaign_shard.sh flags)
score.sh                GF compute_metrics_{fullplant,per_organ}.py + monogrow/metrics.py (--compare_gf) + export_nearest_source.py and re-score
run_baselines.sh        RichardsNet (template nearest) + MonoGrow (z-only, 95k prior, objective/source nearest) + scoring
baselines/_ours.py      shim: monogrow/data.py pointed at our tree, species parsed from our sequence names (no upstream edits)
baselines/run_{richards,monogrow}.py   thin wrappers around richards/fit_test.py and monogrow/fit_test.py
runs/                   results (gitignored); runs/jobs.log = one line per job with wall time and s/step; runs/smoke_results.md
```

Sequence names are `<dataset>_<species>_<sequence_id>`: `pheno4d_maize_M01`, `pheno4d_tomato_T05`,
`sugar4d_sugar_beet_plant_002`. Full list, frames, organ counts: `MANIFEST.md`.

## Conversion (what changes vs. the unified contract)

* labels: unified stem `1` -> GF organ `0`, leaf `k >= 2` -> `k-1`; unified `0` (unknown) is dropped (never occurs in these two datasets);
* geometry: the unified plant-centred mm frame (stem base at the origin, z up) is written as is; GF canonicalises every
  leaf itself (PSA PCA alignment) and keeps the stem in plant space, so no further alignment is needed;
* normals: Open3D estimate oriented towards +Z (what `monogrow/canon.py` does; PSA uses them only for the per-frame
  z-rotation stage of leaf alignment, which is otherwise the identity);
* `is_leaf_tip = 0` everywhere: there are no tip annotations, hence **EPE is undefined (NaN, n_tips 0); only CD is scored**;
* `D<dd>` = `int(day)` of the unified npz (Pheno4D calendar days 0..20, Sugar4D DAS-37 = 0..56); PSA sorts frames by it;
* frames without stem points are left out (PSA raises when a day has no stem to orient its leaves): one scan,
  `sugar4d_sugar_beet_plant_004` DAS 40; all other 989 scans are kept;
* PLY layout is byte-compatible with the ETH v2 files (x y z double, nx ny nz float, organ_label int, is_leaf_tip uchar);
* point splits: exact re-implementation of GF `scripts/create_train_test_splits.py` (`RandomState(42)` per sequence,
  `rng.permutation(n)` per frame, first `int(0.5 n)` = train; no tips to move). GF validates `n_total_points` per frame;
* `data_splits.json` lists val/test/train plants from `data/unified/sequences.json` (Pheno4D: 06 val / 07 test;
  Sugar4D: shipped 12/12 val/test). GF reads plant splits only from its own repo copy, so pass sequences explicitly with
  `run_gf.sh <sequence>` (as `campaign_shard.sh` does); this file is for our scripts and bookkeeping.

## Which plants are fair for which method

* **GrowFields / GrowFields+nearest**: per-plant test-time optimisation, nothing is learned across plants -> any of the 62
  plants is fair. (Pheno4D maize M01/M02/M04 and tomato T01/T05 are the same physical plants as TrackPlant3D's maize /
  tomato2 benchmark plants, at 8,192 instead of ~10k points and with Pheno4D's own labels.)
* **MonoGrow (95k prior `experiments/results/priors/{ckpt,prior}_cpu95k.pt`) and RichardsNet (`richards/prior.npz`)**:
  both priors were fitted with `monogrow/data.py load_training_tracks` under `DataConfig`: TrackPlant3D train split,
  Pheno4D **minus M01, M02, M04, T01, T05**, and Sugar4D with `other_splits = (train, val, test)`, i.e. **all 48 Sugar4D
  plants** (checked in `ckpt_cpu95k.pt["config"]["data"]` and `results/priors/config_v2cos.json`).
  => fair test plants for the priors: **`pheno4d_maize_M01`, `pheno4d_maize_M02`, `pheno4d_maize_M04`,
  `pheno4d_tomato_T01`, `pheno4d_tomato_T05`** only. Every Sugar4D plant (including the test split) and the other nine
  Pheno4D plants were in the prior's training set; numbers on them are train-set numbers for MonoGrow/RichardsNet.
  A fair Sugar4D comparison needs a prior retrained with `data.other_splits=train` (or `train,val`) for Sugar4D.

## Commands (all from this folder)

```bash
cd experiments/gf_on_ours
../growfields/.venv/bin/python convert.py                       # 6 s: 990 PLYs (321 MB), point_splits/, data_splits.json, MANIFEST.md
bash run_gf.sh pheno4d_maize_M01                                # GF shared_continuous, 50k steps, headline (seed 43)
bash run_gf.sh pheno4d_maize_M01 monotone_continuous 50000 interpolation   # stride-2 protocol
bash run_gf.sh pheno4d_maize_M01 shared_continuous 50000 extrapolation     # holdout last 3
bash score.sh runs/shared_continuous/results/fullplant_batch_shared_continuous_ours_aligned            # all plants in the dir
bash score.sh runs/shared_continuous_interp/results/fullplant_batch_shared_continuous_ours_aligned_stride2 interpolation
bash run_baselines.sh pheno4d_maize_M01 [headline|interpolation|extrapolation] [fit.n_z=200 ...]   # RichardsNet + MonoGrow + scores
THREADS=4 ... ; DRY=1 bash run_gf.sh ... prints the command; NEAREST=0 bash score.sh skips the nearest-source export
```
`score.sh` writes GF's `fullplant_metrics*.csv` / `all_metrics*.csv` and `metrics.py`'s `monogrow_*` CSV/JSON into the
experiment dir, prints the per-plant table (full-plant / per-organ CD, mm^2, `cd_test` on test points only) and the
`--compare_gf` diff (0.00 on every row in the smoke test). Protocol semantics are GF's (BENCHMARK.md §1): first frame of
every organ unscored, interpolation scores odd frames, extrapolation the last 3, GT fill for organs seen < 2 times.

## Smoke test (CPU, `OMP_NUM_THREADS=4`, this Mac) -> `runs/smoke_results.md`

See that file for the numbers. Per-step cost of GF `shared_continuous` (512x5 SIREN, 8,192 points/step over <= 10
organs; `training_time_seconds` of `results.json` / 200 steps, which includes one eval pass over all organs x frames):

| plant | frames | organs | s/step (CPU, 1 process, 4 threads) |
|---|---|---|---|
| `pheno4d_maize_M01` | 12 | 4 | 0.20 |
| `sugar4d_sugar_beet_plant_002` | 16 | 19 (`organs_per_step=10`: each step samples 10 of them) | 0.18 |

Per-organ sampling makes the step cost independent of the organ count; the eval pass every `log_interval=200` steps
and the final export grow with organs x frames (Pheno4D tomato plants have 26-41 leaf ids over 20 frames: unmeasured,
expect noticeably slower eval passes). Reference: 0.138 s/step on TP3D SoH2 alone (NOTES.md §3), 0.033-0.042 s/step on
a g6.xlarge L4 (campaign_shard.sh header).

## Cost of a full run (50,000 steps per plant per run, 1 seed)

| set | plants | steps | CPU @0.19 s/step | L4 @0.04-0.06 s/step |
|---|---|---|---|---|
| prior-fair Pheno4D (M01 M02 M04 T01 T05) | 5 | 250k | 13 h | 2.8-4.2 h |
| all Pheno4D | 14 | 700k | 37 h | 7.8-11.7 h |
| Sugar4D test split | 12 | 600k | 32 h | 6.7-10 h |
| all 62 plants, one config, headline | 62 | 3.1M | 164 h | 34-52 h |
| all 62, 2 configs x 3 protocols (extrapolation: 62) | 372 jobs | 18.6M | ~980 h | 207-310 h (~$170-250 at g6.xlarge on-demand) |

Baselines: RichardsNet ~1 s/plant; MonoGrow z-only fit (1000 z-steps, 1024 pts/organ, chunked Chamfer on CPU) 2.5 min for
M01 (4 organs) but 38 min for Sugar4D plant_002 (18 organ tracks, 16 frames) - pass `fit.n_z=200` / `fit.points_per_organ=512`
for probes, or run it on the GPU box. One GF process uses ~1.3-1.8 cores; several processes per box slow each other
down (NOTES.md §6.1), so prefer one process per 4 vCPUs as `campaign_shard.sh` does.

## Caveats

* No leaf tips -> no EPE, no `n_tips` integrity check; the paper's second metric is unavailable here, CD only.
* Label remap 1->0 / k->k-1 and the organ-id semantics: Pheno4D maize uses the collar convention (young leaves carry the
  stem label until the collar is visible) and unlabelled Pheno4D days carry 1-NN transferred labels (`labelled=0` in the
  unified manifest); Sugar4D leaf ids drop out as leaves die. GF handles organs appearing/disappearing (first frame of an
  organ unscored, GT fill), but label noise lands directly in the per-organ CD.
* 8,192 points per scan (FPS) vs TrackPlant3D's ~10k: GF trains on 4,096 points per frame and scores on 4,096 test
  points (vs ~5,000); CD values are not comparable across datasets anyway (different plant sizes, mm^2).
* Sugar4D cadence is 3-4 days (DAS 37..93) vs daily Pheno4D; GF's per-organ day-based `t_norm` absorbs that, but one
  Euler step per frame gap covers more growth, and MonoGrow's 1-day grid takes 3-4 steps per gap.
* `sugar4d_sugar_beet_plant_004` lost its DAS-40 scan (no stem points); all other sequences are complete.
* GF's `--split val|test` reads the vendored repo's `data_splits.json` (TrackPlant3D plants) and must not be used here;
  always pass `--sequences` (what `run_gf.sh` does).
* `compute_metrics_*.py` and training load the whole converted tree (62 plants, PSA leaf alignment, ~9 s) at start.
* Interpolation / extrapolation were only wire-checked (20-step runs on M01: training, GF scoring with the held-out flags and the
  nearest-source export all run; numbers meaningless). Not verified: `monotone_continuous`, Pheno4D tomato plants (26-41 leaf ids),
  multi-seed layout, GPU.
