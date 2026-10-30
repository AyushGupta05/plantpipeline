# RichardsNet (Candidate C of `experiments/DESIGN_MEMO.md`, section 5)

Explicit per-organ growth curves + species hierarchy, CPU-only, seconds per plant. Each organ's length trajectory
L(a) (a = days since the organ's first observed frame) follows a 4-parameter Richards curve (or Yin's beta-sigmoid),
width follows the allometry W = alpha L^beta, and every frame is decoded in closed form from one template frame,
`x(a) = diag(W/W_tpl, W/W_tpl, L/L_tpl) x_tpl`, in the canonical organ frame. Curve parameters are MAP estimates
under a species(/leaf-rank) Gaussian prior fitted on the training organ tracks (empirical Bayes). Evaluated on the
GrowFields benchmark with exactly MonoGrow's output format (GF-layout PLYs), `monogrow/metrics.py`,
`monogrow/growth_metrics.py` and GrowFields' own `compute_metrics_*.py`. The shared residual MLP eps_psi of the memo
is NOT implemented (`--residual_mlp` flag reserved, raises).

```
curves.py       Richards / beta-sigmoid L(a) on an unconstrained 4-vector theta, length/width measures, closed-form decode
common.py       thread limits (2), sys.path -> ../monogrow (canon, data, ply, config, metrics are imported, not copied),
                organ size series, leaf ranks, MAP fit (scipy least_squares 'trf' on [data residuals; chol(prec)(theta-mu)])
prior.py        fits every training track, pools per species/{leaf,stem} -> prior.npz + prior_table.md
fit_test.py     8 test plants x {headline, interpolation, extrapolation}: MAP fit, decode, GT re-pose, GF-layout PLYs
run_variant.sh  fit + monogrow/metrics.py (+ held-out mode) + monogrow/growth_metrics2.py (nearest-aware) for one configuration
collect.py      markdown table over runs/
gf_check.sh     one plant through GrowFields' compute_metrics_* / create_latex_table_* (GF venv) + --compare_gf diff
```

## Commands (monogrow venv; `OMP_NUM_THREADS=2`, set by common.py)

```bash
cd experiments/richards; PY=../monogrow/.venv/bin/python
$PY prior.py                                          # 66 s: 1088 training tracks, 925 with >= 4 frames -> prior.npz, prior_table.md
./run_variant.sh test headline      richards last     # fit 8 plants (0.14 s of curve fitting, ~6 s with PLY writing) + metrics (~1 min)
./run_variant.sh test interpolation richards last
./run_variant.sh test extrapolation richards last
./run_variant.sh test headline      richards nearest  # template = latest observed frame before each target (MonoGrow's fit.source=nearest)
#   fit_test.py flags: --family richards|beta  --template first|last|nearest  --lambda_prior 1 (inf = prior mean only)
#   --sigma_L 5 (mm; --log_space for relative residuals)  --no_rank  --no_gt_fill_single_frame  --split val
$PY collect.py --split test                           # table below
./gf_check.sh runs/test_headline_richards_last sorghum_highlight_plant2                 # GrowFields' scripts, GF venv
./gf_check.sh runs/test_extrapolation_richards_last maize_control_plant3 extrapolation
```

## What is implemented (decisions not in the memo)

* **Curves** (`curves.py`): Richards theta = (log A, log k, a_m, log nu); beta-sigmoid theta = (log L_m, log a_e,
  logit-ish a_m/a_e, log a_0) where a_0 is the unknown organ age at first observation (the beta curve is 0 at true age 0,
  organs are not). Lengths = 99th-percentile canonical Z, widths = 99th-percentile sqrt(x^2+y^2); stems use MonoGrow's
  stem frame (base at origin, no rotation), so stem "length" is height.
* **Prior** (`prior.py`): tracks with >= 4 frames (925 of 1088; MonoGrow's leakage rule: TP3D train split, Pheno4D minus
  M01/M02/M04/T01/T05, Sugar4D). Stage 0: robust (soft-L1) unregularised fit per track; pool per species/{leaf,stem} with
  3-MAD clipping, shrink mu/Sigma towards the organ-type pool (5 pseudo-organs), ridge floor on Sigma; then 2 rounds of
  refit-with-prior + re-pool (empirical Bayes; the posterior covariance of each organ is not added back, so Sigma is
  slightly over-shrunk). Rank effect: ridge slope of theta on (rank - mean rank), rank = order of first appearance of the
  leaf in the plant (>= 10 ranked leaves per group, else 0). Allometry beta per group from within-organ demeaned
  regression of log W on log L; per test organ beta_i = (S_xy + lam beta_s)/(S_xx + lam), lam = sigma_eps^2/sigma_beta^2.
  Phyllochron Gaussian per species (reported only; no emergence generation). Table: `prior_table.md`.
  Maize comes only from Pheno4D (8 leaves, 4 stems; TP3D maize is all val/test) and Pheno4D labels are noisy.
* **Test fitting** (`fit_test.py`): observed frames = all / every 2nd global frame / all but the last 3 (GF protocols);
  only the point split's train points are measured; organs with 1 observed frame take the prior mean (shape only -
  the level cancels in the ratio); prior mean shifted by the rank slope with the rank computed from observed frames only.
  Template `first` = GF's protocol source frame; `last` = last OBSERVED frame with inverse scaling for earlier frames
  (memo section 5; in extrapolation this is the last training frame, so nothing held out is used). is_train is inherited
  from the template frame; the GT tip of the organ's first tip-annotated frame (from the first observed frame on, held
  out or not, as GF/MonoGrow do) is scaled the same way. Template `nearest` = MonoGrow's `fit.source=nearest`: target
  frame t is decoded from the organ's latest observed frame s < t (headline / interpolation: s = t-1 global frame, i.e.
  one-day predictions, interpolation odd frames from the even frame before; extrapolation: the last observed frame for
  all three held-out frames, identical points to `last`), scaled by the fitted curve's ratio L(a_t)/L(a_s) (width
  (L ratio)^beta) applied to the measured points of s; is_train inherited from s; tip = GT tip of s if annotated there,
  else the latest earlier annotated frame scaled from there, else GF's rule (first annotated frame after s), as in
  `monogrow/fit_test2.py`; `results.json` carries `"source"` so `growth_metrics2.py` references each prediction to its
  own source frame. No min-train-points rule on the source frame (richards measures from all points when a frame has
  no train points). Export, GT fill rule for organs seen < 2 times, directory layout, `experiment_config.json` /
  `results.json` are identical to `monogrow/fit_test.py` (`--template first|last` outputs are bit-identical to before).

## Results, test split (8 plants, Richards, template = last observed frame; CD mm^2 / EPE mm, single deterministic run)

GrowFields for reference (paper, 5 seeds): headline 0.86 / 1.47, interpolation 10.29 / 5.82, extrapolation 149.63 / 18.99.

| variant | frames scored | full-plant CD / EPE | per-organ CD / EPE | n_tips | monotonicity viol. (GT floor) | neg. growth mass mm (GT) | length MAE mm obs / held-out | MAE +1/+2/+3 d |
|---|---|---|---|---|---|---|---|---|
| headline | all | **23.02 / 5.84** | 18.70 / 6.06 | 192 / 189 | 0.006 (0.040) | 9.4 (86.5) | 2.0 / - | - |
| interpolation | observed (even) | 14.37 / 5.47 | 14.05 / 5.94 | 85 / 79 | 0.006 (0.042) | 13.6 (86.5) | 1.6 / 3.2 | 4 / - / - |
| interpolation | held-out (odd) | **20.79 / 7.14** | 19.43 / 8.29 | 93 / 89 | | | | |
| extrapolation | observed | 10.90 / 3.99 | 9.26 / 4.97 | 102 / 98 | 0.007 (0.043) | 13.8 (86.5) | 1.7 / 8.7 | 3 / 10 / 13 |
| extrapolation | held-out (last 3) | **141.48 / 16.36** | 99.55 / 14.38 | 71 / 71 | | | | |

The extrapolation mean above is over all 8 plants; GF's Table 4 excludes 5-frame SoH2 - over the other 7 plants ours is
157.66 / 17.85 (full plant), 106.96 / 15.71 (per organ). Monotonicity violations with the `last` template are a metric
artefact: `growth_metrics.py` assumes the prediction equals GT at the rollout source frame, whereas our first predicted
frame is the inverse-scaled last frame; with `--template first` the rate is exactly 0 in every variant (table below).

Per plant (full-plant CD / EPE; per-organ in `runs/test_*/monogrow_summary*.json`):

| | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
|---|---|---|---|---|---|---|---|---|---|
| headline | 28.68 / 10.69 | 59.89 / 9.81 | 61.85 / 6.81 | 17.26 / 5.98 | 3.80 / 3.37 | 3.94 / 4.43 | 1.92 / 2.29 | 6.84 / 3.35 | 23.02 / 5.84 |
| interpolation (held-out) | 26.16 / 10.79 | 39.92 / 15.68 | 58.22 / 7.34 | 22.88 / 7.84 | 5.75 / 4.13 | 4.20 / 6.14 | 1.77 / 1.90 | 7.38 / 3.30 | 20.79 / 7.14 |
| extrapolation (held-out) | 264.42 / 21.02 | 699.64 / 69.14 | 115.89 / 16.49 | 28.18 / 5.92 | 14.65 / 5.00 | 5.27 / 6.35 | 0.98 / 1.78 | 2.79 / 5.20 | 141.48 / 16.36 |
| **nearest**: headline | 30.45 / 8.40 | 19.42 / 8.09 | 68.45 / 7.14 | 48.45 / 6.22 | 4.22 / 2.67 | 2.40 / 1.74 | 1.06 / 1.17 | 2.17 / 1.47 | 22.08 / 4.61 |
| **nearest**: interpolation (held-out) | 32.33 / 7.03 | 32.07 / 14.78 | 65.22 / 7.48 | 60.04 / 6.63 | 4.26 / 3.36 | 3.00 / 1.75 | 1.07 / 1.19 | 2.67 / 1.63 | 25.08 / 5.48 |
| **nearest**: extrapolation (held-out) | 264.42 / 14.24 | 699.64 / 67.29 | 115.89 / 13.13 | 28.18 / 6.32 | 14.65 / 4.31 | 5.27 / 3.07 | 0.98 / 1.29 | 2.79 / 1.65 | 141.48 / 13.91 |

Same with GF's protocol template (`--template first`, rollout source = first observed frame), with the per-target
template (`--template nearest`, source = latest observed frame before each target, i.e. 1-day predictions on headline /
interpolation and the last observed frame on extrapolation) and other test-split runs (`$PY collect.py --split test`):

| run | frames | full CD / EPE | organ CD / EPE | viol. rate | length MAE obs / held-out | MAE +1/+2/+3 d |
|---|---|---|---|---|---|---|
| headline, first | all | 112.09 / 8.51 | 126.17 / 10.26 | 0.000 | 6.1 / - | - |
| interpolation, first | held-out | 99.70 / 8.69 | 124.48 / 9.83 | 0.000 | 6.3 / 6.3 | 8 / - / - |
| extrapolation, first | held-out | 238.18 / 17.93 | 244.17 / 15.67 | 0.000 | 5.2 / 11.9 | 8 / 12 / 16 |
| headline, **nearest** | all | 22.08 / 4.61 | 17.83 / 5.50 | 0.000 | 2.5 / - | - |
| interpolation, **nearest** | observed (even) | 48.00 / 5.91 | 49.29 / 5.54 | 0.000 | 3.0 / 2.9 | 4 / - / - |
| interpolation, **nearest** | held-out (odd) | 25.08 / 5.48 | 22.16 / 6.63 | 0.000 | 3.0 / 2.9 | 4 / - / - |
| extrapolation, **nearest** | observed | 27.98 / 3.70 | 18.63 / 4.21 | 0.000 | 2.4 / 8.7 | 3 / 10 / 13 |
| extrapolation, **nearest** | held-out (last 3) | 141.48 / 13.91 | 99.55 / 11.95 | 0.000 | 2.4 / 8.7 | 3 / 10 / 13 |
| extrapolation, last, `--no_gt_fill_single_frame` (1-frame organs predicted from the prior mean instead of GT-filled) | held-out | 209.14 / 17.65 | 195.20 / 18.42 | 0.006 | 1.7 / 10.5 | 5 / 12 / 15 |

Nearest vs last vs first (test means, full-plant / per-organ CD / EPE): headline 22.08 / 4.61, 17.83 / 5.50 (nearest) vs
23.02 / 5.84, 18.70 / 6.06 (last) vs 112.09 / 8.51, 126.17 / 10.26 (first); interpolation held-out 25.08 / 5.48, 22.16 / 6.63
vs 20.79 / 7.14, 19.43 / 8.29 vs 99.70 / 8.69, 124.48 / 9.83; extrapolation held-out 141.48 / 13.91, 99.55 / 11.95 vs
141.48 / 16.36, 99.55 / 14.38 vs 238.18 / 17.93, 244.17 / 15.67. On extrapolation the nearest and last point clouds are the
same (same source frame), only the tip differs (nearest scales the source frame's own GT tip, last scales the organ's
first annotated tip through the whole track), which is the whole EPE gain. On the headline table nearest predicts every
frame one day ahead from measured points (no frame is reproduced exactly, unlike last), and still matches last's CD and
beats its EPE; on interpolation held-out it loses 4 mm^2 CD to last (the source is 1 day away instead of the late,
fully-grown frame scaled down; maize/sorghum leaves unfolding between consecutive days hurt the 1-day step most) but wins
EPE. Monotonicity / negative-growth rows are 0 for nearest in every variant with `growth_metrics2.py`'s per-source
reference (interpolation observed frames: n_tips 87 / 80 vs 85 / 79 for first/last - nearest's tip search also reaches
annotated frames before the first observed frame; held-out n_tips are identical).

Reading: tomato/tobacco are close to GF on the headline table (1.9-6.8 mm^2 vs GF's per-plant ~0.5-1); maize/sorghum
are 10-60x worse because young maize/sorghum leaves unfold in the canonical frame (SoH2 leaf 2: width 6 -> 26 mm while
length stays 66 mm) and axis-aligned scaling cannot represent that - exactly the con the memo predicts, and what
eps_psi is for. Even the *oracle* scaling (true L and W ratios) leaves 57-270 mm^2 on that leaf. Length MAE is small
(2 mm on observed frames, 3 mm interpolated, 3/10/13 mm at +1/+2/+3 held-out frames), extrapolation is bounded and
saturating, and the extrapolation CD on held-out frames (141 over 8 plants / 158 over GF's 7) is in the range of GF's
149.63 at ~1e-5 of the compute. Headline interpretation caveat for the `last` template: the template frame itself is
reproduced exactly (CD 0 on that frame, EPE 0 for its tip), which GF's rollout-from-frame-0 never does.

## Validation split (4 plants), used to choose family and template (full-plant CD / EPE; length MAE mm obs / held-out)

| variant | Richards, first | Richards, last | Richards, nearest (added later) | beta, first | beta, last |
|---|---|---|---|---|---|
| headline | 119.67 / 8.29; 4.5 | **65.51 / 4.25; 2.2** | 34.96 / 4.16; 2.9 | 136.14 / 10.53; 6.7 | 63.49 / 6.88; 2.6 |
| interpolation, held-out | 112.32 / 7.51; 6.0 / 5.2 | **65.49 / 4.15; 1.6 / 3.0** | 28.39 / 3.76; 3.3 / 3.0 | 130.42 / 9.01; 8.1 / 6.6 | 71.77 / 6.26; 2.1 / 3.5 |
| extrapolation, held-out | 260.83 / 15.26; 3.9 / 11.0 | **162.61 / 11.85; 1.7 / 7.4** | 162.61 / 8.54; 3.3 / 7.4 | 297.01 / 16.20; 4.1 / 12.0 | 195.34 / 13.10; 1.8 / 9.0 |

Richards wins every cell on EPE and length MAE and all but one on CD; `last` template wins everywhere among the
templates available when the choice was made (bold). `nearest` (Richards only, run afterwards) beats `last` on every val
cell: per-organ CD / EPE headline 29.88 / 4.77 vs 46.44 / 4.70, interpolation held-out 22.95 / 3.99 vs 48.52 / 4.53,
extrapolation held-out 114.91 / 11.06 vs 114.91 / 14.63 (same points, better tip); per plant the gain is on maize
(headline 40.0 vs 68.4) and sorghum (94.6 vs 186.0). Its observed-frame length MAE is higher (2.9 vs 2.2 mm) because
`last` reproduces the last observed frame exactly and `nearest` predicts every frame. Ablations on val
extrapolation (Richards, last; held-out full CD / EPE; MAE +1/+2/+3): default lambda_prior=1: 162.61 / 11.85, 4/7/11;
lambda 0.1: 158.15 / 11.33; lambda 10: 161.74 / 11.89; log-space residuals: 164.58 / 11.72; `--no_rank`: 172.82 / 12.31
(4/8/12); prior mean only (no fitting): 139.70 / 34.44 (3/8/11) - lower CD by luck of the shape, but EPE triples and the
observed-frame CD goes 4.9 -> 20.9. Defaults were kept at lambda = 1, mm residuals, rank on.

## Timing (Apple M5 Pro, 2 threads, machine loaded by other jobs)

prior.py 66 s (both families, 3 EB passes over 925 tracks). fit_test.py: 0.10-0.16 s of curve fitting for all 8 plants
per variant, 5-8 s wall including canonicalisation and PLY writing; monogrow/metrics.py ~30-60 s per run;
growth_metrics.py ~10 s. GF's compute_metrics on one plant ~30 s. Model size: 5 scalars per organ.

## Verified

* End to end on all 8 test plants x 3 variants x 3 templates (and the 4 val plants x 3 variants x 2 families x 2
  templates + Richards nearest); n_tips 192 / 189 on the headline runs as GF requires. After adding `nearest`,
  `--template last` on tomato2_control_plant3 (all 3 variants) re-produced the earlier PLYs byte-identically.
* GrowFields' `compute_metrics_fullplant.py`, `compute_metrics_per_organ.py`, `create_latex_table_{fullplant,per_organ}.py`
  run unchanged (GF venv, CWD `experiments/growfields`) on SoH2 headline and MaC3 extrapolation outputs; every CD row
  identical to `monogrow/metrics.py`, EPE max diff 7e-6 (float32), n_tips identical (`gf_check.sh`).

## Unverified / open

* No multi-seed: the method is deterministic (no randomness anywhere), so a single run is the result.
* `create_latex_table_heldout.py` was not run (MonoGrow verified it on the same layout).
* Prior quality for maize rests on 8 Pheno4D leaf tracks with noisy labels; sugar beet (65 % of tracks) only matters
  through the organ-type shrinkage target. Hyper-parameters (sigma_L = 5 mm, lambda = 1, 5 pseudo-organs, ridge floor)
  are first guesses checked only by the val ablations above.
* Monotonicity numbers for the `last` template are confounded by the metric's source-frame assumption (see above);
  the curve itself is monotone by construction (Richards with k > 0).
* eps_psi (shared residual MLP), Chamfer-based refinement of (A, k), densification, emergence generation: not done.
