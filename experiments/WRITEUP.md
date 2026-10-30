# Growth priors and nearest-frame inference for 4D plant modelling: decomposing GrowFields' benchmark

Draft v0.2, 2026-10-30 (22:30 UTC). Workshop-length. Status: **headline results in for every method on the test
split** (GrowFields 5 seeds, GrowFields + nearest 5 seeds, GrowFields + monotone head partial 6/8 plants, RichardsNet,
MonoGrow 3 seeds). Cells still marked `[PENDING: ...]` are runs in flight on AWS: the GrowFields interpolation /
extrapolation fleet tables, the last two monotone-head plants, the matched-objective MonoGrow fit and the cosine-decay
prior (`NEXT_STEPS.md`). Every number in this draft traces to a file in `experiments/`, cited as (file, section). No
number here was produced for this document; nothing is invented.

The title stays "decomposing", not "beating": on the test split MonoGrow ties GrowFields on per-organ extrapolation
Chamfer and does not close the gap on the fit tables (§4.1, §6).

---

## Abstract

GrowFields (Gajardo et al., ECCV 2026) fits a per-plant SIREN velocity field to a tracked, organ-segmented 4D point-cloud
sequence and reports squared Chamfer distance (CD, mm²) and leaf-tip end-point error (EPE, mm) on the TrackPlant3D test
split: 0.86 / 1.47 when all frames are seen, 10.29 / 5.82 on interpolated frames and 149.63 / 18.99 on three
extrapolated frames (`growfields/BENCHMARK.md`, §2). We reproduce the headline on an L4 fleet with 5 seeds at
0.87 / 1.53 (`growfields/runs_aws/merged/shared_continuous/results_headline.md`) and take the benchmark apart along
three axes. (i) *Inference*: GrowFields rolls every organ out from its first observed frame; predicting each frame
from the nearest observed frame instead is protocol-legal, needs no retraining, and on the same saved 5-seed models
halves GrowFields' headline to **0.44 / 1.02** (`results_headline_nearest.md`, same directory). This inference rule is
the dominant effect for every method we tested: for MonoGrow it is 385 → 59 mm² headline, 351 → 59 interpolation and
580 → 178 extrapolation (`RESULTS.md`, table). (ii) *Constraint*: a velocity factorisation v = softplus(g)·x̂ + P⊥w
makes the distance of every point from the organ base non-decreasing by construction, at a cost of 513 extra
parameters inside GrowFields' own code (`growfields/NOTES.md`, §2). On the 6 of 8 test plants finished so far it
trades 4–16 % of headline CD (0.97 / 1.81 first-source, 0.51 / 1.16 nearest, against 0.93 and 0.44 for the original
field on the same six plants) for three times fewer Z-extent violations (0.017 vs 0.057, GT floor 0.049)
(`growfields/runs_aws/merged/monotone_continuous/results_headline{,_nearest}.md`). (iii)
*Prior*: MonoGrow, a single monotone field shared across 1,088 organ tracks from TrackPlant3D-train, Pheno4D and
Sugar4D with auto-decoded organ codes and a species-conditional Gaussian prior, fitted at test time by 300 code steps
per plant (about one minute on an L4) (`monogrow/README.md`). On the test split, 3 seeds, it reaches 58.91 ± 0.41 /
12.95 ± 0.30 headline, 59.06 ± 0.84 / 10.87 ± 0.08 interpolation and 151.67 / 20.12 on GrowFields' 7-plant per-organ
extrapolation table against GrowFields' 149.63 / 18.99, with zero monotonicity violations (`RESULTS.md`, table, from
`monogrow/runs/gpu_test/RESULTS.md`). It ties GrowFields on per-organ extrapolation and does not close the gap on the
fit tables. A closed-form structured baseline, RichardsNet (four-parameter Richards curves per organ, empirical-Bayes
species prior, 0.1 s for all eight test plants) with nearest-frame templates beats MonoGrow on all three tables and
beats GrowFields on extrapolation tip error (141.48 / 13.91 on 8 plants) while tying its per-organ Chamfer
(`richards/README.md`, Results). We report the gains as a decomposition chain and state plainly what the headline
metric measures, where GrowFields' extrapolation protocol leaks information, and which cells are still in flight.

---

## 1. Introduction

Modelling how a plant's organs grow over days from a sequence of 3D scans is the 4D counterpart of plant phenotyping:
the model must reproduce each observed scan, interpolate between scans, and ideally extrapolate growth beyond the last
one. GrowFields is the first method evaluated on a public organ-level benchmark for this task, TrackPlant3D v2
(43 plants, 378 scans, five species; `growfields/BENCHMARK.md`, §1.1). Its protocol is precise and we adopt it
unchanged: 8 test plants, a fixed 50/50 random point split per frame with all annotated leaf tips in the test half,
CD on test points only, EPE on tips, predictions re-posed with the per-day ground-truth rigid transform, and three
settings — all frames observed (headline), every second frame observed (interpolation), last three frames held out
(extrapolation) (`growfields/BENCHMARK.md`, §1.2–1.5).

Four structural choices in GrowFields leave room that a competing method can occupy without changing the metric
(`DESIGN_MEMO.md`, §1; `growfields/BENCHMARK.md`, §5):

1. **First-frame rollout.** Every frame t ≥ 1 of an organ is predicted by advecting the organ's *first* observed point
   cloud through all intermediate frames (`growfields/BENCHMARK.md`, §1.3). On the headline table the frames are
   observed, so this is a self-imposed handicap: the model has seen frame t−1 but predicts frame t from frame 0.
2. **Per-plant fitting.** θ and the organ codes are re-optimised from scratch for 50,000 Adam steps on each of the
   8 test plants; the other 31 TrackPlant3D plants and all other datasets are unused (`growfields/BENCHMARK.md`, §5.1).
   Short tracks (2–4 frames) are under-constrained; the paper names the 5-frame SoH2 as a failure case.
3. **Unconstrained velocity.** Nothing in f_θ prevents an organ from shrinking, folding or oscillating; in
   extrapolation the Euler rollout continues at the last fitted velocity, unbounded (`DESIGN_MEMO.md`, §1.2).
4. **Emergence is not modelled.** An organ that first appears in a held-out frame is copied from ground truth into
   the full-plant prediction; the per-organ day range and spatial normalisation are computed over all frames including
   held-out ones (`growfields/BENCHMARK.md`, §1.2 caveat; GrowFields `REPRODUCIBILITY.md` §6 as cited there).

Our contribution is to measure each of these separately, on GrowFields' own code and metric scripts, and to report the
gains as a chain: GF (paper, first-frame rollout) → GF + nearest-frame inference → GF + monotone head → MonoGrow
(cross-plant prior + monotone + nearest, code-only fitting) → MonoGrow + per-plant θ fine-tuning, with RichardsNet as
the structured baseline on every table (`RESULTS.md`, Decomposition plan).

---

## 2. Method

### 2.1 Nearest-observed-frame inference

Let an organ be observed on days d₀ < d₁ < … and let t be a target frame. GrowFields exports the prediction for t by
integrating the field from d₀ through every intermediate observed day (`growfields/BENCHMARK.md`, §1.3). We define the
*nearest* source rule: predict frame t from the latest *observed* frame s < t of that organ. Under the headline and
interpolation protocols this makes every prediction a one-scan-ahead prediction (for interpolation, an odd frame from
the even frame before it); under extrapolation the source is the last observed frame for all three held-out frames,
i.e. horizons of +1, +2, +3 scan days (`monogrow/README.md`, Rollout source). Source points are all points of frame s
with their `is_train` mask inherited, so scored points remain the held-out test points; the organ's first frame is
never scored, as in GrowFields. Fitting is untouched in both modes — the per-plant losses are bit-identical — only the
export changes, so the rule is available to *any* method, including GrowFields itself, for which we implemented it as
a post-hoc re-export of saved models (`growfields/export_nearest_all.sh`; `growfields/growfields/scripts/export_nearest_source.py`).
The rule is protocol-legal: only observed frames are used (`RESULTS.md`, MonoGrow findings).

The tip used for EPE follows the same rule: the GT tip of the source frame if annotated there, else the tip of the
latest earlier annotated frame advected through t, else GrowFields' own rule; `n_tips` is identical in both modes
(`monogrow/README.md`, Rollout source).

### 2.2 Monotone velocity factorisation

In the canonical organ frame (PSA leaf canonicalisation: base at the origin, principal axis = +Z; stems with the base
at the soil point), write x̂ = x/‖x‖ and P⊥(x) = I − x̂x̂ᵀ, and factor the velocity as

  v(x, a, c) = softplus(g_θ(x, a, c)) · x̂ + P⊥(x) w_θ(x, a, c).

Then d‖x‖/dt = x̂ᵀv = softplus(g) ≥ 0: **the distance of every point from the organ base is non-decreasing along the
flow**, while bending and twisting remain representable through the tangential residual w (`DESIGN_MEMO.md`, §3).
We state the guarantee exactly: it is on base distance ‖x‖, **not** on axial (Z) extent. The tangential term can rotate
a leaf tip so that its Z coordinate decreases while its base distance grows; after θ fine-tuning on the validation
plants the Z-extent violation rate rose to 0.105 against a GT floor of 0.035 (`RESULTS.md`, MonoGrow findings). If
Z-monotonicity is wanted, a λ_w penalty on ‖P⊥w‖² or a Z-monotone parameterisation is needed; both are open.

Inside GrowFields' code the head is a drop-in: the 3-d velocity output becomes a 4-d [g, w] output, an `organ_base`
buffer stores each organ's base in the model's normalised coordinates, and `velocity_param: free` reproduces the
original bit-for-bit. Parameter count 1,334,468 vs 1,333,955 for six organs (`growfields/NOTES.md`, §2). Integration,
losses, splits and metric scripts are unchanged.

### 2.3 MonoGrow: cross-plant prior with auto-decoded codes

MonoGrow is one SIREN trunk (first sine layer + 5 hidden sine layers of 256, ω₀ = 30, GrowFields' initialisation) with
the 4-d monotone head, input [x/100 mm, a/60, z, s, p/60] where a is organ age in days since first observation, p is
plant age in days, z ∈ ℝ³² is an auto-decoded organ code and s an 8-d species embedding (5 species); 341,804
parameters plus 32 per organ (`monogrow/README.md`, What is implemented). Clocks are absolute days, never normalised
by a held-out range. Integration is forward Euler on a 1-day grid (GrowFields' one-step-per-frame-gap is an ablation flag).

**Prior training.** All 1,088 training organ tracks (≥ 2 frames, ≥ 3 points per frame) from TrackPlant3D train split
(31 plants: tomato 146, sorghum 44, tobacco 36 tracks), Pheno4D minus the plants that are the same physical plants as
TrackPlant3D test sequences (maize 15, tomato 137) and Sugar4D (710) are canonicalised identically and fitted jointly
over θ and all z_i (DeepSDF auto-decoding) with GrowFields' multi-step squared Chamfer over frames s+1..k, 50 %
consecutive-frame pairs, horizon ≤ 14 days, species-balanced sampling, λ_z‖z‖² and optional λ_w, λ_g penalties
(`monogrow/README.md`, What is implemented). After training, q(z | s) = N(μ_s, Σ_s) is fitted to the codes with
shrinkage towards the pooled covariance.

**Test-time fitting.** Freeze θ. For each test plant initialise every organ code at μ_s and run N_z Adam steps on the
multi-step CD over the frames the protocol allows, on train points only, plus λ_prior (z−μ_s)ᵀΣ_s⁻¹(z−μ_s); all
organs of a plant are fitted jointly. Optional stage 2 unfreezes θ for N_θ steps at a small learning rate, which
recovers GrowFields' per-plant regime as a special case (`DESIGN_MEMO.md`, §3; `monogrow/README.md`). Export is in
GrowFields' PLY layout and GrowFields' unmodified metric scripts score it: per-row CD identical, EPE within 8e-6 mm,
`n_tips` identical (`monogrow/README.md`, Verified).

### 2.4 RichardsNet: the structured baseline

Each organ's length L(a) (99th-percentile canonical Z) follows a four-parameter Richards curve, width follows the
allometry W = αL^β, and every frame is decoded in closed form from one template frame by axis-aligned scaling
diag(W/W_tpl, W/W_tpl, L/L_tpl). Curve parameters are MAP estimates under a species/{leaf, stem} Gaussian prior with a
leaf-rank slope, fitted by empirical Bayes on 925 training tracks with ≥ 4 frames (`richards/README.md`). The
template is `first` (GrowFields' source frame), `last` (last observed frame, inverse-scaled for earlier frames) or
`nearest` (MonoGrow's rule). The shared residual MLP ε_ψ of the design memo is not implemented. RichardsNet is
deterministic and costs 0.10–0.16 s of curve fitting for all eight test plants (`richards/README.md`, Timing).

---

## 3. Experimental setup

**Datasets and splits.** Benchmark: TrackPlant3D v2 (ETH zip, 378 PLYs), GrowFields' `data_splits.json` (4 validation
plants, 8 test plants) and `point_splits/<plant>_split.json` (`growfields/BENCHMARK.md`, §1.1–1.2). Test plants and
frame counts: MaC2 (9), MaC3 (8), SoC2 (8), SoH2 (5), TbC1 (7), TbS3 (8), Tm1H3 (16), Tm1S1 (17); 49 organ sequences,
45 with ≥ 3 frames. Prior training data: `data/unified` built by `plantpipeline unify` from TrackPlant3D train,
Pheno4D and Sugar4D (`monogrow/README.md`, Environment).

**Leakage rule.** TrackPlant3D validation and test plants never enter any prior. Pheno4D plants M01, M02, M04, T01 and
T05 are excluded because they are the same physical plants as TrackPlant3D maize / tomato2 sequences
(`monogrow/config.py`, `PHENO4D_LEAK`; `monogrow/README.md`, Tracks / leakage). Consequence: maize in the prior comes
only from 15 Pheno4D tracks with noisy "collar" labelling.

**Metrics.** CD = sum of both directional mean squared nearest-neighbour distances on test points, mm²; EPE = mean
tip error, mm; aggregation frames → organs → plants → unweighted mean over plants, organs with < 2 scored frames
dropped in per-organ tables; `n_tips` must be 192 (full plant) / 189 (per organ) on the headline protocol
(`growfields/BENCHMARK.md`, §1.4–1.5). Growth-plausibility metrics (ours, computable from any method's PLYs): organ
length violation rate (decrease > 2 mm between consecutive predicted frames), negative-growth mass (mm), organ-length
MAE on observed and held-out frames and per extrapolation horizon, with the GT's own violation rate as the floor
(`monogrow/growth_metrics.py`; `DESIGN_MEMO.md`, §7). For `nearest` predictions each frame is referenced to the GT
length of its own source frame (`monogrow/README.md`, Rollout source).

**Seeds.** GrowFields paper: 5 seeds for the headline, 3 for interpolation/extrapolation (`growfields/BENCHMARK.md`,
§1.5). Our Mac reproduction: 1 seed (43 = GrowFields' run_1). AWS campaign: 5 / 3 / 3 seeds for both GrowFields
configs (`growfields/campaign_shard.sh`); the 5-seed headline for both configs is in, interpolation / extrapolation
are still running (`NEXT_STEPS.md`). MonoGrow validation numbers: single fit per configuration; MonoGrow test numbers:
3 seeds, 95k-step CPU prior (`RESULTS.md`, table). RichardsNet: deterministic.

**Compute.** Mac (Apple M5 Pro, CPU torch): GrowFields 0.138 s/step single process, 0.3–0.5 s/step with four
concurrent processes, ≈ 5.8 h per plant; MonoGrow prior 0.24–0.30 s/step (`RESULTS.md`, Compute log;
`growfields/NOTES.md`, §6.1). AWS: four g6.xlarge (NVIDIA L4, $0.805/h each, $3.22/h total); GrowFields
0.033–0.053 s/step per box with one process per GPU (four concurrent processes on the 4-vCPU box fall to 0.5–0.7
s/step, so the campaign is sharded over three boxes); MonoGrow prior 0.085–0.087 s/step, 150k steps in 13,091 s
(`aws/INSTANCE.md`, Fleet; `aws/fleet_poll.log`). Estimated campaign cost ≈ $80–90 for 170 GrowFields runs
(`RESULTS.md`, Compute log).

---

## 4. Results

### 4.1 Decomposition chain, test split

CD mm² / EPE mm, full-plant aggregation unless noted. "paper" rows are GrowFields' published point estimates
(`growfields/BENCHMARK.md`, §2); everything else is ours. Extrapolation means are over GrowFields' 7 plants (SoH2
excluded) unless marked "8 plants".

| # | Model | Inference | Fitting per plant | Headline | Interpolation (held-out) | Extrapolation (held-out) | Source |
|---|---|---|---|---|---|---|---|
| 0a | GrowFields (paper, 5 seeds) | first | 50k steps | **0.86 ± 0.02 / 1.47 ± 0.11** | 10.29 / 5.82 (3 seeds) | 149.63 ± 19.65 / 18.99 ± 0.68 (per-organ agg.); 117.71 ± 35.55 / 21.79 ± 0.54 (full-plant agg.) | BENCHMARK §2.1, 2.3, 2.4 |
| 0b | DPF per-organ (paper) | — | — | 2.47 / 15.15 (single model) | 10.24 / 4.00 | — | BENCHMARK §2.1, 2.3 |
| 0c | COAP (paper) | — | — | 1.69 / 4.23 | 11.73 / 7.33 | 451.09 / 31.39 | BENCHMARK §2.1, 2.3, 2.4 |
| 1 | GrowFields repro, Mac CPU, 1 seed, 8 plants | first | 50k | **0.88 / 1.53** (per-organ 0.89 / 1.60); MaC2 2.65 / 2.81, MaC3 1.40 / 1.79, Tm1S1 0.32 / 1.25 (paper: 2.51 / 2.38, 1.43 / 2.01, 0.32 / 1.34) | — | — | RESULTS.md table; growfields/runs/shared_continuous |
| 1' | GrowFields repro, AWS L4 fleet, 5 seeds, 8 plants | first | 50k | **0.87 ± 0.02 / 1.53 ± 0.06** (per-organ 0.88 / 1.67); every plant within ±0.1 CD of the paper (MaC2 2.49 / 2.38, MaC3 1.52 / 2.37, SoC2 0.68 / 1.33, SoH2 0.75 / 0.99, TbC1 0.58 / 1.62, TbS3 0.46 / 1.17, Tm1H3 0.12 / 0.92, Tm1S1 0.33 / 1.50) | `[PENDING: GF fleet interpolation, 3 seeds]` | `[PENDING: GF fleet extrapolation, 3 seeds]` | runs_aws/merged/shared_continuous/results_headline.md |
| 2 | GrowFields + nearest-source export (same saved models as row 1') | **nearest** | 50k (no retraining) | **0.44 ± 0.01 / 1.02 ± 0.04** (per-organ 0.48 / 1.03); MaC2 1.08 / 1.58, MaC3 0.67 / 1.58, SoC2 0.44 / 1.05, SoH2 0.69 / 0.92, TbC1 0.19 / 1.10, TbS3 0.14 / 0.66, Tm1H3 0.09 / 0.45, Tm1S1 0.20 / 0.79 | `[PENDING: GF fleet interpolation, nearest export]` | `[PENDING: GF fleet extrapolation, nearest export]` | runs_aws/merged/shared_continuous/results_headline_nearest.md |
| 2' | same, Mac 1-seed models (row 1) | **nearest** | 50k (no retraining) | 0.45 / 1.10 (per-organ 0.49 / 1.08); MaC2 1.16 / 1.84, MaC3 0.64 / 1.33, Tm1S1 0.19 / 0.73 | — | — | growfields/runs/shared_continuous/results_nearest_metrics.txt |
| 3 | GrowFields + monotone head (`monotone_continuous`), 5 seeds, **partial: 6 of 8 plants** (SoH2, TbC1 missing; TbS3 1 seed) | first | 50k | 0.97 ± 0.08 / 1.81 ± 0.06 (per-organ 1.02 / 2.01); 6-plant mean, not comparable to the 8-plant rows above | `[PENDING: GF+monotone fleet interpolation]` | `[PENDING: GF+monotone fleet extrapolation]` | runs_aws/merged/monotone_continuous/results_headline.md |
| 3' | GrowFields + monotone head + nearest export, same partial 6 plants | **nearest** | 50k (no retraining) | 0.51 ± 0.04 / 1.16 ± 0.04 (per-organ 0.57 / 1.18); `[PENDING: GF+monotone, remaining 2 plants]` | `[PENDING: GF+monotone fleet interpolation, nearest]` | `[PENDING: GF+monotone fleet extrapolation, nearest]` | runs_aws/merged/monotone_continuous/results_headline_nearest.md |
| 4a | RichardsNet, first-frame template | first | closed form, 0.1 s total | 112.09 / 8.51 | 99.70 / 8.69 | 238.18 / 17.93 (8 plants) | richards/README Results |
| 4b | RichardsNet, last-observed template | last | closed form | 23.02 / 5.84 * | 20.79 / 7.14 * | 141.48 / 16.36 (8 plants); 157.66 / 17.85 (7 plants); per-organ 7-plant 106.96 / 15.71 | richards/README Results |
| 4c | RichardsNet, nearest template | **nearest** | closed form | 22.08 / 4.61 (per-organ 17.83 / 5.50) | 25.08 / 5.48 (per-organ 22.16 / 6.63) | 141.48 / 13.91 (8 plants; per-organ 99.55 / 11.95); 7-plant CD 157.66 | richards/README Results; RESULTS.md table |
| 5 | MonoGrow, 95k-step shared prior, z-only (300 steps; extrapolation + 1000 θ steps @ 1e-5), 3 seeds | **nearest** | ~300 z steps, ~1 min on L4 | 58.91 ± 0.41 / 12.95 ± 0.30 | 59.06 ± 0.84 / 10.87 ± 0.08 | 177.81 ± 36.97 / 21.33 ± 0.50 (8 plants); 7 plants 191.97 / 22.38 full-plant, **151.67 / 20.12 per-organ** (GF per-organ 149.63 / 18.99) | RESULTS.md table (from monogrow/runs/gpu_test/RESULTS.md on the GPU box) |
| 6 | MonoGrow, same fitted codes as row 5, first-frame rollout | first | same | 385.48 / 27.79 | 350.81 / 25.02 | 579.67 / 40.81 | RESULTS.md table (gpu_test) |
| 7 | MonoGrow, matched objective (z fitted with the one-step nearest-source CD it is scored with) | nearest | 300 z | `[PENDING: matched-objective test runs, gpu_test_obj]` | `[PENDING: matched-objective test runs]` | `[PENDING: matched-objective test runs]` | NEXT_STEPS.md |
| 8 | MonoGrow, cosine-decay prior v2 (100k / 125k / 150k tagged checkpoints) | nearest | 300 z | `[PENDING: cosine-prior evaluation, ablate_prior_len_gpu]` | `[PENDING: cosine-prior evaluation]` | `[PENDING: cosine-prior evaluation]` | NEXT_STEPS.md |

\* With the `last` template the template frame itself is reproduced exactly (CD 0 on that frame), which GrowFields'
frame-0 rollout never does; rows 4b are therefore not directly comparable to row 0a (`richards/README.md`, Results).

Reading the chain. **Row 1' against row 0a**: the 5-seed L4 reproduction is 0.87 / 1.53 against the paper's
0.86 / 1.47, with every plant within ±0.1 mm² CD, so the fleet is a faithful reference (`results_headline.md`). **Row 2
against row 1'**: with no retraining, the nearest-source rule halves GrowFields' own headline on the same saved models,
0.87 → 0.44 CD and 1.53 → 1.02 EPE, and improves every one of the 8 plants (MaC2 2.49 → 1.08, TbC1 0.58 → 0.19, TbS3
0.46 → 0.14) (`results_headline_nearest.md`). The Mac single-seed export agrees (0.45 / 1.10, row 2'). This is the bar
any learned prior must beat, and it is a change to inference, not to the model. **Row 3 against row 1'** (same 6
plants only): the monotone head costs 4–16 % of headline CD, about 10 % on average (first-source 0.97 vs 0.93 for
the same six plants of row 1'; nearest 0.51 vs 0.44 for the same six plants of row 2; subset means computed from the
per-plant rows of the two `results_headline*.md` files) and buys three times fewer Z-extent violations, 0.017 vs
0.057 with nearest export, against a GT floor of 0.049 on those plants (§4.4). **Rows 5–6 against row 2**: for
MonoGrow the inference rule is worth 6.5× on the headline (385 → 59), 5.9× on interpolation (351 → 59) and 3.3× on
extrapolation (580 → 178), the same direction as for GrowFields and larger because the code-only fit is coarser.
**Row 5 against rows 0a and 4c**: MonoGrow ties GrowFields on GrowFields' own per-organ 7-plant extrapolation table
(151.67 / 20.12 vs 149.63 / 18.99) with zero monotonicity violations and 25× cheaper fitting, but is 68× worse on the
headline and 5.7× worse on interpolation; RichardsNet-nearest beats MonoGrow on all three tables and beats GrowFields
on extrapolation EPE (13.91 vs 18.99) (`RESULTS.md`, Where things stand).

### 4.2 RichardsNet per plant, test split

Full-plant CD / EPE (`richards/README.md`, Results, per-plant table). GrowFields paper per-plant headline for
reference: MaC2 2.51 / 2.38, MaC3 1.43 / 2.01, SoC2 0.69 / 1.34, SoH2 0.74 / 0.97, TbC1 0.59 / 1.66, TbS3 0.45 / 1.30,
Tm1H3 0.12 / 0.80, Tm1S1 0.32 / 1.34 (`growfields/BENCHMARK.md`, §2.1); extrapolation (per-organ agg.) MaC2 356.13 /
20.96, MaC3 321.08 / 69.77, SoC2 337.39 / 27.04, TbC1 15.38 / 4.06, TbS3 8.09 / 4.14, Tm1H3 2.80 / 2.52, Tm1S1 6.56 /
4.46 (`growfields/BENCHMARK.md`, §2.4).

| RichardsNet | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
|---|---|---|---|---|---|---|---|---|---|
| headline, last | 28.68 / 10.69 | 59.89 / 9.81 | 61.85 / 6.81 | 17.26 / 5.98 | 3.80 / 3.37 | 3.94 / 4.43 | 1.92 / 2.29 | 6.84 / 3.35 | 23.02 / 5.84 |
| headline, nearest | 30.45 / 8.40 | 19.42 / 8.09 | 68.45 / 7.14 | 48.45 / 6.22 | 4.22 / 2.67 | 2.40 / 1.74 | 1.06 / 1.17 | 2.17 / 1.47 | 22.08 / 4.61 |
| interpolation held-out, last | 26.16 / 10.79 | 39.92 / 15.68 | 58.22 / 7.34 | 22.88 / 7.84 | 5.75 / 4.13 | 4.20 / 6.14 | 1.77 / 1.90 | 7.38 / 3.30 | 20.79 / 7.14 |
| interpolation held-out, nearest | 32.33 / 7.03 | 32.07 / 14.78 | 65.22 / 7.48 | 60.04 / 6.63 | 4.26 / 3.36 | 3.00 / 1.75 | 1.07 / 1.19 | 2.67 / 1.63 | 25.08 / 5.48 |
| extrapolation held-out, last | 264.42 / 21.02 | 699.64 / 69.14 | 115.89 / 16.49 | 28.18 / 5.92 | 14.65 / 5.00 | 5.27 / 6.35 | 0.98 / 1.78 | 2.79 / 5.20 | 141.48 / 16.36 |
| extrapolation held-out, nearest | 264.42 / 14.24 | 699.64 / 67.29 | 115.89 / 13.13 | 28.18 / 6.32 | 14.65 / 4.31 | 5.27 / 3.07 | 0.98 / 1.29 | 2.79 / 1.65 | 141.48 / 13.91 |

Tomato and tobacco are within 1–7 mm² of GrowFields on the headline table; maize and sorghum are 10–60× worse because
young monocot leaves unfold in the canonical frame (SoH2 leaf 2: width 6 → 26 mm at constant length 66 mm) and
axis-aligned scaling cannot represent that — even oracle L and W ratios leave 57–270 mm² on that leaf
(`richards/README.md`, Reading). On extrapolation the nearest and last point clouds coincide (same source frame); the
EPE gain is entirely from scaling the source frame's own GT tip instead of the organ's first tip through the whole
track. The per-organ 7-plant extrapolation CD, 106.96 (`last`), is below GrowFields' per-organ 149.63 at ~1e-5 of the
compute; the full-plant 7-plant CD 157.66 is above GrowFields' full-plant 117.71, so "ties CD" depends on the
aggregation and we report both (`richards/README.md`, Results; `growfields/BENCHMARK.md`, §2.4).

Validation-split ablations (Richards, `last`, extrapolation held-out full CD / EPE): λ_prior 1 (default) 162.61 /
11.85; 0.1: 158.15 / 11.33; 10: 161.74 / 11.89; log-space residuals 164.58 / 11.72; no rank effect 172.82 / 12.31;
prior mean only, no fitting 139.70 / 34.44 (`richards/README.md`, Validation split). Richards beats the beta-sigmoid
in every validation cell on EPE and length MAE.

### 4.3 MonoGrow, validation split

The recipe used on the test split (§4.3b) was chosen on the 4 GrowFields validation plants (maize_control_plant1,
sorghum_control_plant1, tobacco_control_plant2, tomato2_control_plant3), held-out frames only, z-only fitting with
`fit.n_z=300 fit.lr_z=3e-3 fit.lambda_prior=1e-3` unless stated (`monogrow/README.md`, Rollout source table;
`RESULTS.md`, MonoGrow findings). All numbers in this subsection are validation, single fits.

**Inference rule (50k-step CPU prior, z-only):**

| Setting | first: full CD / EPE | nearest: full CD / EPE | first: organ CD / EPE | nearest: organ CD / EPE |
|---|---|---|---|---|
| extrapolation, maize | 1379.6 / 52.0 | 252.1 / 16.1 | 946.9 / 71.0 | 200.5 / 22.4 |
| extrapolation, sorghum | 705.1 / 47.0 | 435.8 / 18.4 | 553.2 / 53.4 | 337.1 / 23.9 |
| extrapolation, tobacco | 46.4 / 13.5 | 21.1 / 7.4 | 54.3 / 12.5 | 28.3 / 7.1 |
| extrapolation, tomato2 | 21.5 / 9.6 | 12.4 / 6.5 | 25.9 / 10.0 | 17.8 / 6.2 |
| **extrapolation, mean** | **538.2 / 30.5** | **180.3 / 12.1** | 395.1 / 36.7 | 145.9 / 14.9 |
| extrapolation, length MAE held-out; +1/+2/+3 d | 28.2 mm; 22.1 / 29.2 / 33.2 | 12.9 mm; 6.2 / 14.3 / 18.3 | | |
| **interpolation, mean** | **511.4 / 18.3** | **62.5 / 8.1** | 498.9 / 22.9 | 56.0 / 10.8 |
| interpolation, length MAE held-out | 15.6 mm | 6.9 mm | | |

`n_tips` is identical in both modes (36 extrapolation / 41 interpolation full-plant rows) and GrowFields' metric
script agrees row-wise (`monogrow/README.md`, Rollout source).

**Prior length (nearest, z-only):** 50k-step prior → 95k-step prior: extrapolation 180.3 / 12.1 → **131.5 / 12.6**
(maize 216.37 / 16.66, sorghum 262.34 / 21.42, tobacco 35.74 / 5.47, tomato2 11.61 / 6.69; per-organ 115.61 / 15.27);
interpolation 62.5 / 8.1 → 55.9 / 8.9 (per-organ 50.04 / 11.48) (`RESULTS.md`, MonoGrow findings;
`monogrow/runs/ablate_prior_len/{extrapolation,interpolation}_cpu95k.metrics.txt`). RichardsNet on the same
validation extrapolation split: 162.61 / 11.85 (`last`), 162.61 / 8.54 (`nearest`) (`richards/README.md`, Validation
split). MonoGrow z-only with the 95k prior therefore beats RichardsNet on validation extrapolation CD at equal
protocol and loses on EPE. The 150k GPU prior is confirmed degraded, as the compute log predicted (§5): validation
interpolation, nearest, z-only, 392 / 20.0 against 55.4 / 9.1 for the 95k CPU prior, so every test number below uses
the 95k checkpoint and the cosine-decay prior v2 is the fix (`RESULTS.md`, MonoGrow findings, "150k GPU prior").

**Headline, z-only, 50k prior** (observed frames, full-plant CD): tomato2 13.4, tobacco 71.9, sorghum 869, maize
1,608, mean 641 at lr_z 1e-2; mean 324 at lr_z 3e-3 (`RESULTS.md`, MonoGrow findings). Monocots with few training
tracks (maize 15, sorghum 44) are not covered by the code prior; per-plant θ fine-tuning is required to approach
GrowFields' regime on the headline table.

**θ fine-tuning sweep (first source, 50k prior, extrapolation held-out):**

| Stage 2 | full CD / EPE | organ CD / EPE | maize / sorghum / tobacco / tomato2 full CD | Z-extent viol. (GT floor) | length MAE; +1/+2/+3 d |
|---|---|---|---|---|---|
| none (z only) | 538.2 / 30.5 | 395.1 / 36.7 | 1379.6 / 705.1 / 46.4 / 21.5 | — | 28.2; 22.1 / 29.2 / 33.2 |
| 2000 steps, lr 2e-5 | **323.3 / 31.1** | 268.9 / 38.5 | 704.2 / 513.9 / 60.0 / 15.0 | 0.105 (0.035) | 18.7; 12.4 / 19.9 / 23.9 |
| 5000 steps, lr 2e-5 | 453.7 / 27.7 | 335.7 / 34.5 | 1294.8 / 485.1 / 27.3 / 7.5 | 0.088 (0.035) | 20.3; 15.8 / 21.6 / 23.6 |
| 2000 steps, lr 1e-4 | 273,675 / 391.9 | 425,566 / 391.8 | 130k / 357k / 608k / 7.4 | 0.000 | 582; 488 / 581 / 678 |

(`monogrow/README.md`, Theta fine-tuning sweep.) Fine-tuning at 2e-5 lowers CD by 40 % at 2,000 steps; 5,000 steps
over-fits maize while tobacco and tomato2 keep improving; lr 1e-4 diverges on three of four plants, the same
instability as prior training at 1e-4 (§5).

**GPU validation grid (95k prior, nearest source; lr_θ ∈ {1e-5, 2e-5} × N_θ ∈ {1000, 2000, 5000}):** on
interpolation held-out frames z-only is best at 55.4 / 9.1 and every θ fine-tune configuration is worse (68–339 mm²).
On extrapolation, N_θ = 1000 at lr 1e-5 gives 149.8 / 14.0 against 160.4 / 13.1 for z-only, a marginal CD gain;
lr 2e-5 is harmful and 5k steps at 2e-5 diverges. The Z-extent violation rate is 0.000 in every cell. The test recipe
follows: z-only (300 steps, about 1 min per plant on an L4) for headline and interpolation, plus 1000 θ steps at 1e-5
for extrapolation (`RESULTS.md`, MonoGrow findings, "GPU val grid"; `monogrow/runs/gpu_val/run.sh`).

**Pipeline evidence with a 3k-step prior (SoH2, test plant, 0.4 % of planned steps):** z-only 93.1 / 20.6; plus 200 θ
steps at lr 1e-5: 3.97 / 6.23 with 0 base-distance violations (GrowFields 0.74 / 0.97 after 50k steps)
(`monogrow/README.md`, Verified). These are not results; they show the stage-2 path works.

### 4.3b MonoGrow, test split (3 seeds)

Recipe from the validation grid, 95k-step CPU prior, `nearest` source; headline and interpolation z-only (300 steps),
extrapolation + 1000 θ steps at lr 1e-5; seeds 0–2; GrowFields' unmodified metric scripts. All numbers below are from
`RESULTS.md` (table rows "MonoGrow, nearest source" and "MonoGrow, first-frame source"), which copies
`monogrow/runs/gpu_test/RESULTS.md` from the GPU box.

| MonoGrow, test | Headline | Interpolation (held-out) | Extrapolation (held-out) |
|---|---|---|---|
| nearest source, full-plant, 8 plants | **58.91 ± 0.41 / 12.95 ± 0.30** | **59.06 ± 0.84 / 10.87 ± 0.08** | 177.81 ± 36.97 / 21.33 ± 0.50 |
| nearest source, 7 plants (GF's extrapolation set) | | | 191.97 / 22.38 full-plant; **151.67 / 20.12 per-organ** (GF paper per-organ 149.63 / 18.99, full-plant 117.71 / 21.79) |
| first-frame source, same fitted codes | 385.48 / 27.79 | 350.81 / 25.02 | 579.67 / 40.81 |
| Z-extent violation rate (GT floor) | 0.000 (0.040) | 0.000 | 0.000 |

Three readings. (i) Inference dominates again: the same codes rolled out from the first frame are 6.5× / 5.9× / 3.3×
worse on the three tables. (ii) On GrowFields' own per-organ extrapolation table MonoGrow ties GrowFields (151.67 vs
149.63 CD, 20.12 vs 18.99 EPE) with no per-plant velocity field, zero monotonicity violations and about one minute of
fitting per plant against 20–45 minutes for GrowFields on the same GPU (`results_headline.md`, training times); the
seed spread on extrapolation (± 36.97) is large and driven by the monocots. (iii) It does not close the fit gap:
58.91 against GrowFields' 0.87 (first) / 0.44 (nearest) on the headline, 59.06 against 10.29 on interpolation. The
cross-plant prior with 32 free numbers per organ cannot reproduce observed frames to sub-mm² Chamfer, which is what
those two tables reward (§5.1). RichardsNet with nearest templates is better than MonoGrow on all three tables (22.08 /
4.61, 25.08 / 5.48, 141.48 / 13.91 on 8 plants; row 4c), so the learned field has so far not earned its parameters
over four Richards parameters per organ. Two fixes are running: fitting z with the one-step nearest-source objective it
is scored with instead of the multi-step from-first-frame CD, and the cosine-decay prior (`NEXT_STEPS.md`; rows 7–8).

### 4.4 Growth-plausibility metrics

| Model / run | Violation rate (GT floor) | Negative-growth mass mm (GT) | Length MAE mm observed / held-out | Source |
|---|---|---|---|---|
| GrowFields original, SoH2, 200-step smoke (12 frame pairs) | 0.167 (0.25, tol 0) | — | 5.24 / — | NOTES.md §3 |
| GrowFields + monotone head, SoH2, 200-step smoke | **0.000** (0.25, tol 0) | — | 16.41 / — (under-grows at 200 steps) | NOTES.md §3 |
| GrowFields original, 50k, 5 seeds, 8 plants, first source (headline, per-organ) | 0.055 (0.062) | 1.66 | 0.90 / — | runs_aws/merged/shared_continuous/results_headline.md |
| GrowFields original + nearest export, same models | 0.057 (0.062) | 1.62 | 0.73 / — | runs_aws/merged/shared_continuous/results_headline_nearest.md |
| GrowFields + monotone head, 50k, 5 seeds, **6 of 8 plants**, first source | **0.000** (0.049) | 0.00 | 1.08 / — | runs_aws/merged/monotone_continuous/results_headline.md |
| GrowFields + monotone head + nearest export, same 6 plants | 0.017 (0.049) | 0.93 | 0.92 / — | runs_aws/merged/monotone_continuous/results_headline_nearest.md |
| MonoGrow, test, nearest, 3 seeds (headline / interp / extrap) | 0.000 (0.040) | — | — | RESULTS.md table (gpu_test) |
| RichardsNet, first template, any variant | 0.000 | — | 6.1 / — (headline); 5.2 / 11.9 (extrap) | richards/README Results |
| RichardsNet, nearest, headline / extrap | 0.000 | 0 | 2.5 / — ; 2.4 / 8.7 (+1/+2/+3: 3 / 10 / 13) | richards/README Results |
| RichardsNet, last, headline | 0.006 (0.040) † | 9.4 (86.5) † | 2.0 / — | richards/README Results |
| MonoGrow, all smoke / 3k runs, base-distance | 0.000 | 0.0 | — | monogrow/README Verified |
| MonoGrow + θ 2k, val extrap, Z-extent, first source | 0.105 (0.035) ‡ | — | 18.7 (held-out) | monogrow/README sweep |

† Metric artefact: the `last` template inverse-scales from a later frame, so the chain's first "pair" compares a
prediction to GT rather than two predictions; the Richards curve itself is monotone (k > 0) (`richards/README.md`).
‡ In `first` mode on held-out frames the chain starts at the GT length of the frame before the first held-out
prediction, so this row is prediction-vs-GT error, not a field monotonicity violation; `--mode all` gives 0
(`monogrow/README.md`, Rollout source caveat). Base-distance monotonicity is the guaranteed property; Z-extent is not.

On the finished 5-seed fleet runs the picture is: GrowFields' original field shrinks organs at about the rate the
ground truth does (0.055 vs floor 0.062, 1.66 mm of negative-growth mass per organ pair); the monotone head removes
every violation on the 6 plants finished so far (0.000, 0.00 mm) at a cost of 0.18 mm in observed-frame length MAE
(1.08 vs 0.90). With nearest export each frame is referenced to the GT length of its own source frame, so the
monotone-head rate of 0.017 is prediction-vs-GT noise around a monotone field, not field violations; it is still three
times below the original field's 0.057 under the same export (`results_headline{,_nearest}.md`, growth_metrics
output, both configs). MonoGrow is at 0.000 on all three test tables against a GT floor of 0.040 (`RESULTS.md`, table).

---

## 5. Limitations and honesty notes

1. **The headline CD is a fitting score.** Test points are other random samples of the same observed surfaces, so the
   headline table rewards reproducing observed frames densely; only interpolation and extrapolation measure
   generalisation in time (`DESIGN_MEMO.md`, §0). Any method that uses the nearest observed frame as its source will
   look strong on the headline table for that reason alone.
2. **Nearest-source is an inference change available to all methods.** It is not part of our model. We report it for
   GrowFields itself (row 2) precisely so that the chain separates inference strategy from the monotone prior and from
   the cross-plant prior. GrowFields' published tables are `first`; we report `nearest` alongside, never instead
   (`monogrow/README.md`, Rollout source).
3. **GrowFields' extrapolation protocol leaks.** Per-organ time normalisation and spatial normalisation use all frames
   including the held-out ones, and organs that first appear in held-out frames are GT-filled, so published and
   reproduced extrapolation numbers are optimistic (`growfields/BENCHMARK.md`, §1.2 caveat; `RESULTS.md`, Reading so
   far). Our methods use absolute day clocks and never normalise by a held-out range, but we keep GrowFields' GT-fill
   rule for comparability (a `--no_gt_fill_single_frame` variant of RichardsNet gives 209.14 / 17.65 instead of
   141.48 / 16.36 on 8-plant extrapolation, `richards/README.md`, Results).
4. **The monotonicity guarantee is on base distance, not Z extent** (§2.2). The Z-extent violation rate after θ
   fine-tuning (0.105 vs GT floor 0.035) must be reported, not hidden behind the guarantee (`RESULTS.md`).
5. **Prior-training instability.** At lr_θ 1e-4 the prior diverged at step ≈ 5,000 and, with gradient clipping,
   drifted from ≈ 7,000 (growth head g 0.05 → 0.21). The 6k-step sweep chose lr_θ 2e-5, λ_g 1e-3, clip 1.0 (batch CD
   25.9 mm² vs 35.3 for the alternative). Even at 2e-5 the growth head drifts late: on the GPU run g rose
   monotonically 0.016 (95k) → 0.107 (145k) with batch CD trending up, so the 150k checkpoint is suspect and the ≈ 95k
   CPU checkpoint is the healthy reference; a v2 prior with cosine decay to 0 and tagged checkpoints is running
   (`RESULTS.md`, Compute log; `aws/fleet_poll.log`). Any MonoGrow test number must state which checkpoint it used.
6. **Seeds and partial tables.** The GrowFields headline rows (1', 2) are 5 seeds on the L4 fleet and reproduce the
   paper within its own ± (0.02 CD / 0.06 EPE ours vs 0.02 / 0.11 paper; `results_headline.md`;
   `growfields/BENCHMARK.md`, §1.5). The GrowFields interpolation and extrapolation rows are still running, so the
   only extrapolation comparison available today is against the paper's point estimates. The monotone-head rows are
   **6 of 8 plants** (TbS3 with one seed) and their means are not comparable to 8-plant means; the Mac rows are one
   seed (43) on CPU. EPE varies ~1 mm between identical CUDA runs (`growfields/BENCHMARK.md`, §5.9).
7. **MonoGrow's test numbers are 3 seeds with a recipe tuned on validation plants** whose species mix differs (val has
   tomato2, test has tomato1). The headline gap to GrowFields (58.91 vs 0.44–0.87) is a model limitation, not a seed
   effect (± 0.41); the extrapolation seed spread (± 36.97) is large and monocot-driven. MonoGrow does not beat
   GrowFields on any fit table and ties it only on per-organ extrapolation (`RESULTS.md`, table).
8. **Species coverage of the prior is uneven.** Maize rests on 15 Pheno4D tracks with noisy labels; sugar beet is 65 %
   of tracks (species-balanced sampling mitigates this) (`monogrow/README.md`, Open issues).
9. **RichardsNet's headline failure is structural**: axis-aligned scaling cannot represent monocot leaf unfolding; the
   residual MLP ε_ψ, densification and emergence generation are not implemented (`richards/README.md`, Unverified).
10. **Not implemented from the design memo**: the saturation gate, 4× source densification, the hypernetwork and
    MetaSDF variants, and Candidate B (trajectory diffusion with an emergence model) (`monogrow/README.md`, Open issues).
11. **Compute reporting.** GrowFields reports no compute; our timings are on an Apple M5 Pro and an NVIDIA L4 and are
    not comparable across the two (`growfields/BENCHMARK.md`, §3; `RESULTS.md`, Compute log).

---

## 6. Conclusion

On GrowFields' own benchmark, the single largest lever is not a model. Predicting each frame from the nearest observed
frame instead of the first one halves GrowFields' 5-seed headline on the same saved models, 0.87 → 0.44 mm² CD and
1.53 → 1.02 mm EPE, improving all 8 plants with no retraining (`results_headline_nearest.md`), and it is the dominant
effect for every method we ran: 6.5× / 5.9× / 3.3× on MonoGrow's three test tables (`RESULTS.md`, table). Any future
comparison on this benchmark should state the source rule next to the number.

The other two axes are smaller and honestly mixed. The monotone velocity head, dropped into GrowFields' own code for
513 parameters, trades 4–16 % of headline CD (0.51 vs 0.44 under nearest export on the same 6 of 8 plants finished
so far) for three times fewer Z-extent violations (0.017 vs 0.057) and zero under first-frame rollout, with an exact but
narrow guarantee on base distance (`monotone_continuous/results_headline{,_nearest}.md`). The cross-plant prior,
MonoGrow, fitted by 300 code steps per plant in about a minute, ties GrowFields on its per-organ 7-plant extrapolation
table (151.67 / 20.12 vs 149.63 / 18.99) with zero violations and 3-seed spreads reported, but does not close the gap
on the fit tables (58.91 headline, 59.06 interpolation) and is beaten on all three tables by the four-parameter
RichardsNet with nearest templates, which itself beats GrowFields on extrapolation tip error (13.91 vs 18.99 mm) and
ties its per-organ Chamfer at ~1e-5 of the compute (`richards/README.md`, Results). The title therefore stays
"decomposing": the chain in §4.1 attributes GrowFields' headline to inference strategy first, per-plant fitting
second, and shows that neither a monotone constraint nor a cross-plant prior has yet substituted for the latter. The
cells still in flight — GrowFields' own interpolation and extrapolation with the nearest rule, the last two
monotone-head plants, the matched-objective fit and the cosine prior (`NEXT_STEPS.md`) — will sharpen the
extrapolation column, where the structured and learned priors are closest to GrowFields, and are not expected to
change the fit-table reading.

---

## Appendix A. Reproducibility (commands)

All paths relative to `experiments/`. Full detail and caveats in `experiments/README.md`.

```bash
# GrowFields reproduction (CPU, 1 seed, headline), then monotone config; CWD must be experiments/growfields
cd growfields && bash launch_headline.sh                          # = run_test_split.sh shared_continuous 1 4 3; then monotone_continuous
.venv/bin/python collect_results.py shared_continuous             # -> runs/shared_continuous/results.md
# GrowFields nearest-source re-export of every finished run (no retraining)
bash export_nearest_all.sh runs nearest                           # -> <exp>_nearest/ beside each experiment dir
# GrowFields GPU campaign (AWS, 170 jobs, sharded over gf1..gf3)
cd ../aws && YES=1 NAME=gf1 ./launch.sh && ./sync_data.sh gf1 --code --gf-data && ./setup_env.sh gf1
./ssh.sh gf1 'cd plantpipeline/experiments/growfields && tmux new-session -d -s train "SHARD=0/3 bash campaign_shard.sh 2>&1 | tee -a runs/campaign.log"'
./status_all.sh; ./pull_results.sh --collect                      # -> growfields/runs_aws/merged/<stage>/results.md

# MonoGrow prior (stable config) and fitting
cd ../monogrow; PY=.venv/bin/python
$PY train_prior.py full train.lr_theta=2e-5 train.lambda_g=1e-3 train.grad_clip=1.0 train.out=runs/full_gpu   # GPU; CPU ~10 h/150k
$PY fit_test.py full fit.variant=extrapolation fit.split=val fit.source=nearest fit.n_z=300 fit.lr_z=3e-3 fit.lambda_prior=1e-3 \
    fit.ckpt=runs/full_cpu/ckpt.pt fit.prior=runs/full_cpu/prior.pt fit.out=runs/ablate_prior_len/extrapolation_cpu95k
$PY metrics.py --experiment_dir runs/ablate_prior_len/extrapolation_cpu95k --mode extrapolation
$PY growth_metrics.py --experiment_dir runs/ablate_prior_len/extrapolation_cpu95k --mode extrapolation
bash runs/gpu_test/run.sh headline 0                              # test split, nearest then first (reuse), seed 0

# RichardsNet
cd ../richards; PY=../monogrow/.venv/bin/python
$PY prior.py && ./run_variant.sh test headline richards nearest && ./run_variant.sh test extrapolation richards nearest
$PY collect.py --split test
```
