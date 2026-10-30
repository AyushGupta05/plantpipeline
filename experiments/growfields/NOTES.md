# GrowFields reproduction + monotone velocity model: working notes

Written 2026-10-30. Companion to `BENCHMARK.md` (protocol, published numbers) and `../DESIGN_MEMO.md`
§3/§7 (Candidate A "MonoGrow", first experiment). Everything here is under `experiments/growfields/`.

## 1. Layout

| path | what |
|---|---|
| `growfields/` | vendored `Joaquin-Gajardo/growfields` (MIT), commit **`3f8e9477ec1f88a9154c146bb654e23f4ae21df1`** ("code release", 2026-10-14, the only commit). Nested `.git` removed so it is plain project code; upstream state = that commit, our changes = `monotone.patch`. |
| `.venv/` | uv venv, Python 3.12.13, torch 2.15.0 CPU, numpy 2.3.5, scipy, scikit-learn, open3d 0.20, omegaconf, torchdiffeq, pandas, plotly, PSA v0.4.1 (`99d86f5`), `growfields` installed editable from `growfields/`. Gitignored. |
| `shim/pytorch3d/` | the pure-torch `pytorch3d.loss.chamfer_distance` stand-in from `smoke_cpu.sh`, byte-identical; wired in via `.venv/lib/python3.12/site-packages/pytorch3d_shim.pth` (so it also works under `python -I`). |
| `data/TrackPlant3D/versions/v2/` | the one extracted copy of `/Users/ayush/dev/plantpipeline/data/raw/TrackPlant3D_v2.zip` (378 PLYs). Gitignored. All GF configs use the relative `dataset_path: data/TrackPlant3D/versions`, and `scripts/compute_metrics_per_organ.py` hard-codes the same relative path, so **every command runs with CWD = `experiments/growfields/`**; PSA's auto-download only triggers when that directory is missing (`plant_shape_analysis/dataloaders/trackplant3D.py:122`), so nothing is re-downloaded. |
| `smoke/` | 200-step SoH2 comparison (`smoke/run_smoke.sh`, logs, results). Gitignored. |
| `runs/<config>/` | full runs: `results/fullplant_batch_<config>_test_aligned/` (GF experiment dir), `<seq>_seed43.log`, `jobs.log`, `results.md`. Gitignored. |
| `run_test_split.sh` | 8 test plants × N seeds, P parallel processes, THREADS torch threads each (§5). |
| `collect_results.py`, `growth_metrics.py` | metrics → `runs/<config>/results.md` (§5). |
| `launch_headline.sh` | detached launcher used for the runs in §6. |
| `monotone.patch` | unified diff of every vendored file we changed (regenerate with the loop in the shell history: `diff -u <upstream> <ours>` per file). |

`.gitignore` (repo root) gained `experiments/growfields/{.venv,data,runs,smoke}/`.

## 2. Monotone velocity model (what changed, file by file)

Model option, not a new `model_type`: `model_type` stays `shared_continuous`, so every dispatch site in
`organ_model_manager.py`, `io_utils.py`, the metrics scripts and the save/load paths is untouched and
`default_model.velocity_param: free` (default everywhere) is bit-for-bit the original code path.

1. **`growfields/models/siren.py`** — new class `MonotoneSharedContinuousSiren(SharedContinuousSiren)` (appended,
   nothing above it modified).
   - Same trunk as the parent (`Siren(4+embed_dim, h_f, h_layers, out)` with ω0 = 30, same embedding/`conditioning`/
     `embed_init`/`organ_id_remap` handling), but the head is 4-d: `[g, w_x, w_y, w_z]`. Param count 1,334,468 vs
     1,333,955 (+513 = one extra output row) for 6 organs, 512×5.
   - `forward(x, t, organ_id)`: `r = x − organ_base[organ]`, `r̂ = r/‖r‖` (set to 0 where `‖r‖ < base_eps`),
     `P⊥w = w − (w·r̂) r̂`, **`v = softplus(g)·r̂ + P⊥w`**. Hence `d‖r‖/dt = softplus(g) ≥ 0`: distance from the
     organ base is non-decreasing along the flow; bending/twisting lives in `P⊥w`.
   - `organ_base`: registered buffer `[num_organs, 3]`, indexed by dense organ index, in the organ's *normalised*
     model coordinates (`x_norm = (x − min_xyz)/scale·2 − 1`; isotropic, so directions are preserved). Saved in the
     state dict (`models/model_shared_continuous.pth`), so loading needs no extra file.
   - `g_bias_init` (−2.0): initial bias of the g output so growth starts at softplus(−2) ≈ 0.13 instead of 0.69.
   - Regulariser hooks: when `collect_reg` is True and the module is in train mode, each forward appends
     `(mean softplus(g), mean ‖P⊥w‖²)` to a list; `pop_reg_terms()` returns and clears it. Zero overhead when off.
2. **`growfields/models/siren_dynamics.py: train_shared_continuous`** — new kwargs `velocity_param="free"`,
   `lambda_w=0`, `lambda_g=0`, `base_eps=1e-3`, `g_bias_init=-2.0`. If `velocity_param == "monotone"` it builds the
   monotone class and fills `organ_base`: **leaves** (PSA canonical frame, base at the origin) get the normalised
   image of the origin, `−min_xyz/scale·2 − 1`; the **stem (organ 0, stays in plant space)** gets the soil point of
   its first training frame `(median x, median y, min z)`. Adds `lambda_g·mean(softplus g) + lambda_w·mean‖P⊥w‖²`
   (averaged over every velocity evaluation of the step) to `loss`, logged as `loss_logs["monotone_reg"]`.
   Integration (`_integrate_segment`), Chamfer loss, sampling, optimiser, scheduler, diagnostics: unchanged.
   Assumes `dataset.aligned_leaves: true` (as in both configs); with unaligned leaves the leaf base would be wrong.
3. **`growfields/models/organ_model_manager.py: _shared_continuous_kwargs`** — forwards the five config keys
   (defaults = original behaviour).
4. **`scripts/generate_interpolation.py: _load_shared_continuous`** — instantiates the monotone class when the saved
   config has `velocity_param: monotone` (7 lines). Not exercised yet **[unverified]**.
5. **`configs/per_organ/monotone_continuous.yaml`** — copy of `shared_continuous.yaml` + the block
   `velocity_param: monotone, lambda_w: 0.0, lambda_g: 0.0, base_eps: 1.0e-3, g_bias_init: -2.0` (and a 2-line header
   comment). `diff` against `shared_continuous.yaml` shows nothing else.

Not changed: integration, losses, point/frame splits, metrics scripts, table scripts, output layout.
Unit checks (`python -c`, see shell history): radial velocity component ≥ 0 on random points (min 0.118 at init),
state-dict round trip of `organ_base`, gradients flow through the reg terms, tensor organ ids are remapped.

## 3. CPU smoke comparison, SoH2, 200 steps (`smoke/run_smoke.sh`, OMP_NUM_THREADS=3, single process)

Same command for both configs: `train_per_organ.py --config configs/per_organ/<cfg>.yaml --sequences
sorghum_highlight_plant2 --points_split_dir … --seed 0 default_model.n_steps=200 default_model.log_interval=100
output.save_plots=false`, then `compute_metrics_per_organ.py`, `compute_metrics_fullplant.py`, both
`create_latex_table_*.py --per_sequence`, and `growth_metrics.py --tol 0`.

| | shared_continuous (original) | monotone_continuous |
|---|---|---|
| training time, 200 steps (incl. 2 eval passes) | 27.6 s (**0.138 s/step**) | 27.7 s (0.139 s/step) |
| whole run wall (load, train, export, normals) | 63 s | 35 s (dataset cache warm) |
| train CD at step 100 / MeanEvalCD | 0.0201 / 26.7 | 0.0431 / 43.5 |
| per-organ Mean CD / EPE (5 organs kept) | 16.07 / 9.36 | 26.22 / 16.92 |
| full-plant Mean CD / EPE | 19.55 / 7.79 | 26.68 / 14.37 |
| n_tips full / per-organ | 11 / 10 | 11 / 10 |
| organ-length violations (tol 0 mm) over 12 consecutive predicted-frame pairs | 2/12 (0.167) | **0/12** |
| GT violation floor (tol 0) | 0.25 | 0.25 |
| organ-length MAE vs GT (mm) | 5.24 | 16.41 |

Predicted p99 axial extent (mm) per frame, monotone model: leaf_1 35.3→37.0→38.7→40.2, leaf_2 67.4→69.4→71.2→72.8,
leaf_3 64.8→68.7→71.8, leaf_4 48.7→56.5, stem 23.9→25.3→26.8→28.2 — non-decreasing everywhere, as the construction
guarantees. (Original: leaf_2 68.0→67.6→67.3→68.1, stem fine.) After only 200 steps the monotone model under-grows
(MAE 16 mm vs 5 mm): it starts with small growth (g bias −2) and lr is 5e-6; whether it catches up by 50k steps is
exactly what the full runs will tell. Both metric CSVs pass the structural checks of BENCHMARK §4.

## 4. Pipeline dry run (all 8 test plants, 20 steps, P=4, 3 threads)

`bash run_test_split.sh shared_continuous 1 4 3 default_model.n_steps=20 default_model.log_interval=10` → 44 s wall,
8/8 OK, then `collect_results.py` produced `results.md` with full-plant `n_tips = 192` and per-organ `n_tips = 189`
(exactly BENCHMARK §3's integrity check; 45 of 49 organ sequences kept). Deleted afterwards (`rm -rf runs/shared_continuous`).

## 5. Scripts

- `run_test_split.sh <config> [N_SEEDS=1] [P=4] [THREADS=3] [extra key=value overrides]` — one `train_per_organ.py`
  process per sequence (longest first), `xargs -P`, `OMP_NUM_THREADS=MKL_NUM_THREADS=THREADS`, all into one GF
  experiment dir `runs/<config>/results/fullplant_batch_<config>_test_aligned` via `--experiment_dir`. Seeds: N=1 →
  `--n_runs 1 --seed 43` (the seed of run_1 in the paper's `--n_runs` protocol, 42+1; single-run layout
  `sequences/<seq>/`); N>1 → `--n_runs N --seed 42` (their exact protocol, `run_k/` layout, seeds run sequentially
  inside the process). Logs `runs/<config>/<seq>_seed43.log`, job timings `runs/<config>/jobs.log`. Starts are
  staggered 0-4 s because every process rewrites the shared `experiment_config.json` (same content).
- `collect_results.py <config>` — runs GF's `compute_metrics_per_organ.py`, `compute_metrics_fullplant.py`,
  `create_latex_table_{fullplant,per_organ}.py --per_sequence`, and `growth_metrics.py`; writes
  `runs/<config>/results.md`: full-plant and per-organ CD/EPE per plant + unweighted Mean (REPRODUCIBILITY
  aggregation, re-implemented in pandas and cross-checked against the LaTeX rows, which are included verbatim), paper
  numbers beside them, n_tips integrity, training time, and the growth metrics.
- `growth_metrics.py --experiment_dir <exp> [--tol 2]` — organ length L = 99th-percentile axial (z) extent per
  exported organ PLY in the canonical organ frame (leaves canonical, stem plant space, z up, mm). Violation rate =
  fraction of consecutive exported-frame pairs with `L(t+1) < L(t) − tol` (memo: tol 2 mm), GT floor computed the same
  way, negative-growth mass Σ max(0, L(t) − L(t+1)), length MAE = mean |L_pred − L_gt|. Caveat: the organ's first
  frame is never exported by GF (rollout starts there), so pairs are between predicted frames only; organs seen twice
  contribute no pair.

## 6. Background runs launched 2026-10-30 00:55 GMT (this Mac, CPU)

(A first launch at 00:50, master pid 18540, was killed after 5 min because Python's block-buffered stdout made the
logs unreadable; `run_test_split.sh` now sets `PYTHONUNBUFFERED=1` and `runs/` was wiped before relaunching.)

`bash launch_headline.sh` (P=4, THREADS=3, N_SEEDS=1) → detached master **pid 21068** (own session/process group,
parent launchd; also in `runs/launch_headline.pid`), running:

```
bash run_test_split.sh shared_continuous 1 4 3 && .venv/bin/python collect_results.py shared_continuous ;
bash run_test_split.sh monotone_continuous 1 4 3 && .venv/bin/python collect_results.py monotone_continuous
```

i.e. the ORIGINAL config on the 8 test plants, 1 seed (43), headline protocol (all frames, fixed point split,
aligned leaves, 50,000 steps, their defaults, only `output.save_plots=false`), 4 sequences at a time × 3 threads
(12 of 15 cores); the monotone config is queued after it with the same settings, so total load never exceeds 12
cores. Each worker command line (from `jobs.log` / `ps`):

```
cd experiments/growfields && OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 .venv/bin/python growfields/train_per_organ.py \
  --config growfields/configs/per_organ/shared_continuous.yaml --sequences <seq> \
  --points_split_dir growfields/configs/datasets/TrackPlant3D/point_splits \
  --experiment_dir runs/shared_continuous/results/fullplant_batch_shared_continuous_test_aligned \
  --n_runs 1 --seed 43 output.save_plots=false
```
(plus `PYTHONUNBUFFERED=1`.) First wave: tomato1_shade_plant1 (python pid 21130), maize_control_plant3 (21131),
maize_control_plant2 (21138), tomato1_heat_plant3 (21216); queue: sorghum_control_plant2, tobacco_shade_plant3, tobacco_control_plant1,
sorghum_highlight_plant2; then the 8 monotone jobs. Master log `runs/launch_headline.log`.

Measured per-step time with 4 concurrent 3-thread processes: see §6.1. Progress: `grep -c '^Step' runs/<config>/<seq>_seed43.log`
(one line per 200 steps; 250 = done), `tail runs/*/jobs.log`. Kill everything: `kill -- -21068`.

### 6.1 Measured speed and expected finish

Step lines (`Step NNNNN`, one per 200 steps) counted over a 300 s window (00:56:43 → 01:01:43) with all four workers
running, 3 threads each:

| plant (frames, organs) | step lines in 300 s | ≈ steps | ≈ s/step | ≈ h for 50k steps |
|---|---|---|---|---|
| MaC2 (9, 6) | 5 | 1000 | 0.30 | 4.2 |
| MaC3 (8, 5) | 4 | 800 | 0.38 | 5.2 |
| Tm1S1 (17, 9) | 4 | 800 | 0.38 | 5.2 |
| Tm1H3 (16, 4) | 3 | 600 | 0.50 | 7.0 |

(±200 steps granularity.) So **≈0.3–0.5 s/step per process with 4 concurrent processes**, vs 0.138 s/step for one
process alone (§3) and 0.13 in BENCHMARK §4: the 4 processes share 5 performance + 10 efficiency cores and each
sits at only ~120-135 % CPU (`ps`), i.e. the SIREN matmuls at 8,192 points do not use 3 threads well; the eval pass
every 200 steps (all organs, all frames) also grows with frames × organs (Tm1 plants). Expected: wave 1 (4 longest
plants) ≈ 5-7 h, wave 2 (4 shorter plants) ≈ 4-5 h → **original config done ≈ 10-12 h after 00:55, i.e. ~11:00-13:00
GMT 30 Oct**; the monotone batch (same speed in §3) then takes another ~10-12 h → **≈ 21:00 30 Oct – 01:00 31 Oct
GMT**. `collect_results.py` runs automatically after each batch (`runs/<config>/results.md`). If throughput matters
more than the 12-core cap, `P=6 THREADS=2 bash launch_headline.sh` would likely be faster given the low per-process
utilisation **[untested]**.

## 7. What is verified / unverified

Verified: venv + shim + data path; original and monotone configs train, export, and evaluate on CPU; monotone
predictions are non-decreasing in organ length (0 violations at tol 0 on SoH2); both metric scripts and table scripts
run on the vendored copy; parallel 8-plant pipeline + `collect_results.py` end-to-end (20-step dry run, n_tips
192/189); detached launch survives the launching shell (ppid 1).
Unverified: the 50k-step numbers themselves (running); `generate_interpolation.py` with the monotone class;
`lambda_w`/`lambda_g` > 0 beyond a one-step gradient check; `--n_runs > 1` path of `run_test_split.sh` (only N=1
exercised); `pytest tests/` on the vendored copy.
