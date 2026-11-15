# Growth priors and nearest-frame inference for 4D plant modelling: decomposing GrowFields' benchmark

Draft v1.0, 2026-11-15. Workshop-length. Status: **every run complete; every cell of every table filled on the test
split.** GrowFields and GrowFields + monotone head: 5 seeds headline, 3 seeds interpolation and extrapolation, all 8
test plants, both inference rules, 170 L4 runs (`results/gf_fleet/*.txt`; `RESULTS.md`, Final GrowFields fleet
tables). MonoGrow: 3 seeds under both fitting objectives (`results/monogrow_gpu_test_obj_RESULTS.md`). RichardsNet:
deterministic (`richards/README.md`). New since v0.3: the GrowFields interpolation and extrapolation fleet tables for
both configurations and both inference rules, and the monotone head on all 8 plants (shards gf1 and gf3 retrieved).
Every number in this draft traces to a file in `experiments/`, cited as (file, section). No number here was produced
for this document; nothing is invented.

The title stays "decomposing", not "beating": GrowFields reproduces on all three tables; the nearest-source rule halves
its headline error and roughly halves its tip error on every table, with Chamfer flat on interpolation and mixed on
extrapolation; the monotone head buys guaranteed base-distance monotonicity for ~6 % headline CD; MonoGrow with the
matched fitting objective passes RichardsNet on headline Chamfer (13.94 vs 22.08) but not on tip error (10.70 vs 4.61),
remains 32× behind GrowFields + nearest on the headline table (0.44), 4× behind GrowFields on interpolation (41.96 vs
10.32) and ties it only on per-organ extrapolation (§4.1, §6).

---

## Abstract

GrowFields (Gajardo et al., ECCV 2026) fits a per-plant SIREN velocity field to a tracked, organ-segmented 4D point-cloud
sequence and reports squared Chamfer distance (CD, mm²) and leaf-tip end-point error (EPE, mm) on the TrackPlant3D test
split: 0.86 / 1.47 when all frames are seen, 10.29 / 5.82 on interpolated frames and 149.63 / 18.99 on three
extrapolated frames (`growfields/BENCHMARK.md`, §2). We reproduce all three tables on an L4 fleet (5 / 3 / 3 seeds,
8 plants): 0.87 ± 0.02 / 1.53 ± 0.06 headline, 10.32 ± 0.08 / 5.38 ± 0.14 interpolation, 98.82 ± 12.72 / 21.21 ± 0.52
extrapolation full-plant (133.10 ± 17.60 / 18.49 ± 0.35 per-organ, against the paper's 149.63 / 18.99)
(`results/gf_fleet/shared_continuous{,_interp,_extrap}_first.txt`) and take the benchmark apart along three axes.
(i) *Inference*: GrowFields rolls every organ out from its first observed frame; predicting each frame from the nearest
observed frame instead is protocol-legal, needs no retraining, and on the same saved models halves GrowFields' headline
to **0.44 ± 0.01 / 1.02 ± 0.04**, leaves interpolation Chamfer flat (10.93 ± 0.07) while cutting tip error 5.38 → **3.65
± 0.04** (below DPF's 4.00), and on extrapolation halves tip error 21.21 → **11.70 ± 0.68** while Chamfer is mixed
(133.88 ± 22.72 vs 98.82 ± 12.72 full-plant; 129.48 ± 16.97 vs 133.10 per-organ)
(`results/gf_fleet/shared_continuous{,_interp,_extrap}_nearest.txt`). The honest claim is therefore tip accuracy and
short-horizon fit, not extrapolation Chamfer. The rule is the dominant effect for every method we tested: for MonoGrow
it is 385 → 59 mm² headline, 351 → 59 interpolation and 580 → 178 extrapolation (`RESULTS.md`, table).
(ii) *Constraint*: a velocity factorisation v = softplus(g)·x̂ + P⊥w makes the distance of every point from the organ
base non-decreasing by construction, at a cost of 513 extra parameters inside GrowFields' own code
(`growfields/NOTES.md`, §2). On all 8 test plants it costs ~6 % of headline CD (0.92 ± 0.02 / 2.04 ± 0.13 vs 0.87 /
1.53 first-source; 0.58 ± 0.01 / 1.37 ± 0.03 vs 0.44 / 1.02 nearest), is equal on interpolation (10.25 ± 0.05 vs 10.32;
nearest 10.70 ± 0.04 / 3.68 ± 0.01 vs 10.93 / 3.65) and on per-organ extrapolation (136.29 ± 14.38 / 18.06 vs 133.10 /
18.49), is worse on full-plant extrapolation (128.31 ± 11.25 vs 98.82; nearest 170.48 ± 18.82 / 12.24 ± 1.86 vs
133.88 / 11.70), and gives guaranteed base-distance monotonicity with 2–3× fewer Z-extent violations under nearest
export (headline 0.028 vs 0.060, GT floor 0.040; interpolation 0.007 vs 0.021; extrapolation 0.206 vs 0.265)
(`results/gf_fleet/monotone_continuous*.txt`, `results/gf_fleet/*_growth_nearest.txt`). (iii)
*Prior*: MonoGrow, a single monotone field shared across 1,088 organ tracks from TrackPlant3D-train, Pheno4D and
Sugar4D with auto-decoded organ codes and a species-conditional Gaussian prior, fitted at test time by a few hundred
code steps per plant (one to four minutes on an L4) (`monogrow/README.md`). Fitted with GrowFields' multi-step
from-first-frame loss it reaches 58.91 ± 0.41 / 12.95 ± 0.30 headline, 59.06 ± 0.84 / 10.87 ± 0.08 interpolation and
151.67 / 20.12 on GrowFields' 7-plant per-organ extrapolation table against the paper's 149.63 / 18.99 and our fleet's
133.10 ± 17.60 / 18.49, 3 seeds, zero monotonicity violations (`RESULTS.md`, table). Fitting the codes instead on the
one-step nearest-frame Chamfer they are scored with (`fit.objective=nearest`, §2.3) lowers the headline to **13.94 ±
2.90 / 10.70 ± 0.33** and interpolation to **41.96 ± 3.37 / 10.63 ± 0.38** and leaves extrapolation at 193.40 ± 6.01 /
20.50 ± 0.55 on 8 plants (per-organ 148.32 / 18.03; 7-plant per-organ 156.46 / 19.25)
(`results/monogrow_gpu_test_obj_RESULTS.md`). The objective/inference mismatch, not the prior, was most of the headline
gap, yet MonoGrow ties GrowFields only on per-organ extrapolation and trails it by an order of magnitude on both fit
tables (13.94 vs 0.44–0.87; 41.96 vs 10.32). A closed-form structured baseline, RichardsNet (four-parameter Richards
curves per organ, empirical-Bayes species prior, 0.1 s for all eight test plants) with nearest-frame templates reaches
22.08 / 4.61, 25.08 / 5.48 and 141.48 / 13.91 (8 plants): behind MonoGrow on headline Chamfer but ahead on headline tip
error, on interpolation and on extrapolation, and ahead of GrowFields' first-frame extrapolation tip error (13.91 vs
21.21 full-plant) though not of GrowFields + nearest (11.70) (`richards/README.md`, Results). A cosine-decay
learning-rate schedule removes the late-training drift of the shared prior (validation extrapolation 151.79 / 11.49 at
150k steps against 160.35 / 13.10 for the 95k checkpoint used on test, `results/monogrow_prior_v2cos_val_eval.txt`) and
is not yet combined with the matched objective. We report the gains as a decomposition chain and state plainly what the
headline metric measures and where GrowFields' extrapolation protocol leaks information.

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

**Matched fitting objective (`fit.objective`).** Until the overnight runs the codes were always fitted on GrowFields'
loss, the multi-step CD of one rollout from the organ's first observed frame through all observed frames, while the
export predicted each frame from its nearest observed frame: the quantity optimised and the quantity scored did not
match (`monogrow/README.md`, Objective). `fit.objective=nearest` replaces the loss by the mean over consecutive observed
pairs (s, t) of CD between frame s's train points rolled out on the 1-day grid to day(t) and frame t's train points —
exactly what `source=nearest` is scored on (under interpolation the pairs are two days apart) — with the same chunked
Chamfer, the same λ_prior term (a mean over pairs, not a sum, so λ_prior keeps its scale relative to `first`), the same
optimiser and step counts and the same optional stage-2 θ fine-tune; `first` stays the default and is bit-identical to
the previous behaviour, and export and metrics are untouched (`monogrow/README.md`, Objective). The mismatch matters
because the first-frame loss spends the 32 code dimensions on a long rollout that inference never uses, and the
one-step loss has a different optimum: on the validation plants it moves the headline from 59.23 to 3.50 mm² at equal
step budget (§4.3, `results/monogrow_gpu_val_obj_decision.md`).

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
§1.5). Our Mac reproduction: 1 seed (43 = GrowFields' run_1). AWS campaign: 5 seeds headline, 3 seeds interpolation
and 3 seeds extrapolation for both GrowFields configurations, all 8 test plants (7 for extrapolation, GrowFields' set),
170 runs in total, every run complete and every table reported as mean ± std over seeds
(`growfields/campaign_shard.sh`; `results/gf_fleet/*.txt`; `RESULTS.md`, Final GrowFields fleet tables).
MonoGrow validation numbers: single fit per configuration; MonoGrow test numbers: 3 seeds, 95k-step CPU prior, under
both fitting objectives (`RESULTS.md`, table; `results/monogrow_gpu_test_obj_RESULTS.md`). RichardsNet: deterministic.

**Compute.** Mac (Apple M5 Pro, CPU torch): GrowFields 0.138 s/step single process, 0.3–0.5 s/step with four
concurrent processes, ≈ 5.8 h per plant; MonoGrow prior 0.24–0.30 s/step (`RESULTS.md`, Compute log;
`growfields/NOTES.md`, §6.1). AWS: four g6.xlarge (NVIDIA L4, $0.805/h each, $3.22/h total); GrowFields
0.033–0.053 s/step per box with one process per GPU (four concurrent processes on the 4-vCPU box fall to 0.5–0.7
s/step, so the campaign is sharded over three boxes); MonoGrow prior 0.085–0.087 s/step, 150k steps in 13,091 s
(`aws/INSTANCE.md`, Fleet; `aws/fleet_poll.log`). The GrowFields campaign was 170 L4 runs of 50k steps (~25 min
each, ≈ 90 GPU-hours) on four g6.xlarge boxes for ≈ 30 h, ≈ $100 including the MonoGrow prior box
(`RESULTS.md`, Compute log; `aws/INSTANCE.md`).

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
| 1' | GrowFields repro, AWS L4 fleet, 5 / 3 / 3 seeds, 8 plants | first | 50k | **0.87 ± 0.02 / 1.53 ± 0.06** (per-organ 0.88 / 1.67); every plant within ±0.1 CD of the paper (MaC2 2.49 / 2.38, MaC3 1.52 / 2.37, SoC2 0.68 / 1.33, SoH2 0.75 / 0.99, TbC1 0.58 / 1.62, TbS3 0.46 / 1.17, Tm1H3 0.12 / 0.92, Tm1S1 0.33 / 1.50) | **10.32 ± 0.08 / 5.38 ± 0.14** (per-organ 11.46 / 4.14); MaC2 18.52, MaC3 9.24, SoC2 19.21, SoH2 28.54, TbC1 2.98, TbS3 1.77, Tm1H3 0.89, Tm1S1 1.42 | **98.82 ± 12.72 / 21.21 ± 0.52** full-plant; **133.10 ± 17.60 / 18.49 ± 0.35** per-organ (paper 117.71 / 21.79 and 149.63 / 18.99); MaC2 156.59, MaC3 215.62, SoC2 242.42, TbC1 57.73, TbS3 6.76, Tm1H3 3.19, Tm1S1 9.42 | results/gf_fleet/shared_continuous{,_interp,_extrap}_first.txt |
| 2 | GrowFields + nearest-source export (same saved models as row 1') | **nearest** | 50k (no retraining) | **0.44 ± 0.01 / 1.02 ± 0.04** (per-organ 0.48 / 1.03); MaC2 1.08 / 1.58, MaC3 0.67 / 1.58, SoC2 0.44 / 1.05, SoH2 0.69 / 0.92, TbC1 0.19 / 1.10, TbS3 0.14 / 0.66, Tm1H3 0.09 / 0.45, Tm1S1 0.20 / 0.79 | 10.93 ± 0.07 / **3.65 ± 0.04** (per-organ 10.97 / 3.77); CD flat, EPE 5.38 → 3.65 on 7 of 8 plants (SoH2 9.64 → 5.33, MaC3 11.72 → 4.85) | 133.88 ± 22.72 / **11.70 ± 0.68** full-plant; 129.48 ± 16.97 / 11.42 ± 0.55 per-organ; CD mixed (MaC2 157 → 245, MaC3 216 → 318, SoC2 242 → 348; TbC1 58 → 12, Tm1S1 9.4 → 4.6), EPE halved on every plant | results/gf_fleet/shared_continuous{,_interp,_extrap}_nearest.txt |
| 2' | same, Mac 1-seed models (row 1) | **nearest** | 50k (no retraining) | 0.45 / 1.10 (per-organ 0.49 / 1.08); MaC2 1.16 / 1.84, MaC3 0.64 / 1.33, Tm1S1 0.19 / 0.73 | — | — | growfields/runs/shared_continuous/results_nearest_metrics.txt |
| 3 | GrowFields + monotone head (`monotone_continuous`), 5 / 3 / 3 seeds, 8 plants | first | 50k | 0.92 ± 0.02 / 2.04 ± 0.13 (per-organ 0.93 / 2.18); MaC2 2.48 / 2.36, MaC3 1.54 / 2.55, SoC2 0.84 / 1.94, SoH2 0.97 / 2.88, TbC1 0.63 / 1.81, TbS3 0.47 / 1.99, Tm1H3 0.13 / 1.55, Tm1S1 0.33 / 1.22 | 10.25 ± 0.05 / 5.61 ± 0.08 (per-organ 11.01 / 4.49) | 128.31 ± 11.25 / 21.00 ± 0.68 full-plant; 136.29 ± 14.38 / 18.06 ± 0.71 per-organ; MaC2 336.38, MaC3 316.34, SoC2 172.69, TbC1 55.82, TbS3 5.70, Tm1H3 2.73, Tm1S1 8.54 | results/gf_fleet/monotone_continuous{,_interp,_extrap}_first.txt |
| 3' | GrowFields + monotone head + nearest export (same saved models as row 3) | **nearest** | 50k (no retraining) | 0.58 ± 0.01 / 1.37 ± 0.03 (per-organ 0.67 / 1.45); MaC2 1.09 / 1.72, MaC3 0.69 / 1.70, SoC2 0.80 / 1.55, SoH2 1.33 / 2.17, TbC1 0.24 / 1.69, TbS3 0.15 / 0.79, Tm1H3 0.10 / 0.63, Tm1S1 0.21 / 0.73 | 10.70 ± 0.04 / 3.68 ± 0.01 (per-organ 10.55 / 3.75) | 170.48 ± 18.82 / 12.24 ± 1.86 full-plant; 142.34 ± 12.81 / 11.64 ± 1.49 per-organ; MaC2 445.09, MaC3 472.82, SoC2 249.64, TbC1 14.61, TbS3 5.33, Tm1H3 2.24, Tm1S1 3.61 | results/gf_fleet/monotone_continuous{,_interp,_extrap}_nearest.txt |
| 4a | RichardsNet, first-frame template | first | closed form, 0.1 s total | 112.09 / 8.51 | 99.70 / 8.69 | 238.18 / 17.93 (8 plants) | richards/README Results |
| 4b | RichardsNet, last-observed template | last | closed form | 23.02 / 5.84 * | 20.79 / 7.14 * | 141.48 / 16.36 (8 plants); 157.66 / 17.85 (7 plants); per-organ 7-plant 106.96 / 15.71 | richards/README Results |
| 4c | RichardsNet, nearest template | **nearest** | closed form | 22.08 / 4.61 (per-organ 17.83 / 5.50) | 25.08 / 5.48 (per-organ 22.16 / 6.63) | 141.48 / 13.91 (8 plants; per-organ 99.55 / 11.95); 7-plant CD 157.66 | richards/README Results; RESULTS.md table |
| 5 | MonoGrow, 95k-step shared prior, z-only (300 steps; extrapolation + 1000 θ steps @ 1e-5), 3 seeds | **nearest** | ~300 z steps, ~1 min on L4 | 58.91 ± 0.41 / 12.95 ± 0.30 | 59.06 ± 0.84 / 10.87 ± 0.08 | 177.81 ± 36.97 / 21.33 ± 0.50 (8 plants); 7 plants 191.97 / 22.38 full-plant, **151.67 / 20.12 per-organ** (GF per-organ 149.63 / 18.99) | RESULTS.md table (from monogrow/runs/gpu_test/RESULTS.md on the GPU box) |
| 6 | MonoGrow, same fitted codes as row 5, first-frame rollout | first | same | 385.48 / 27.79 | 350.81 / 25.02 | 579.67 / 40.81 | RESULTS.md table (gpu_test) |
| 7 | MonoGrow, **matched objective** (`fit.objective=nearest`, §2.3), 95k prior, 3 seeds; headline / interpolation z 300 + 1000 θ @ 1e-5, extrapolation z 1000 | **nearest** | ~1.3k steps, 3–4 min on L4 | **13.94 ± 2.90 / 10.70 ± 0.33** (per-organ 12.99 / 12.99); MaC2 4.58 / 13.04, MaC3 21.96 / 29.96, SoC2 40.40 / 17.25, SoH2 41.84 / 15.01, TbC1 1.25 / 4.40, TbS3 0.98 / 2.87, Tm1H3 0.16 / 1.32, Tm1S1 0.35 / 1.73 | **41.96 ± 3.37 / 10.63 ± 0.38** (per-organ 43.18 / 11.11) | 193.40 ± 6.01 / 20.50 ± 0.55 (8 plants; per-organ 148.32 / 18.03); 7 plants 210.92 / 21.87 full-plant, 156.46 / 19.25 per-organ | results/monogrow_gpu_test_obj_RESULTS.md; recipe results/monogrow_gpu_val_obj_decision.md |
| 8 | MonoGrow, cosine-decay prior v2, 150k checkpoint, first objective, z-only — **validation split only (4 plants), not test**; shown for the prior, outside the chain | nearest | 300 z | not evaluated | val 57.51 / 8.31 (per-organ 52.11 / 11.26); 95k prior, same recipe, 55.36 / 9.06 | val 151.79 / 11.49 (per-organ 121.86 / 14.28); 95k prior, same recipe, 160.35 / 13.10; 100k / 125k checkpoints 185.22 / 12.05, 180.88 / 12.75 | results/monogrow_prior_v2cos_val_eval.txt; 95k reference results/monogrow_gpu_val_obj_summary.md (nt0, first) |

\* With the `last` template the template frame itself is reproduced exactly (CD 0 on that frame), which GrowFields'
frame-0 rollout never does; rows 4b are therefore not directly comparable to row 0a (`richards/README.md`, Results).

Reading the chain. **Row 1' against row 0a**: GrowFields reproduces on all three tables. The L4 fleet gives 0.87 ±
0.02 / 1.53 ± 0.06 against the paper's 0.86 / 1.47 with every plant within ±0.1 mm² CD; 10.32 ± 0.08 / 5.38 ± 0.14
against 10.29 / 5.82 on interpolation; and 98.82 ± 12.72 / 21.21 ± 0.52 full-plant, 133.10 ± 17.60 / 18.49 ± 0.35
per-organ, against 117.71 / 21.79 and 149.63 / 18.99 on extrapolation, i.e. somewhat *better* Chamfer than published,
within the paper's own ± 19.65 / ± 35.55 (`results/gf_fleet/shared_continuous{,_interp,_extrap}_first.txt`). The fleet
is a faithful reference, and from here on GrowFields comparisons use the fleet numbers, not the paper's. **Row 2
against row 1'**: with no retraining, the nearest-source rule halves GrowFields' own headline on the same saved models,
0.87 → 0.44 CD and 1.53 → 1.02 EPE, improving every one of the 8 plants (MaC2 2.49 → 1.08, TbC1 0.58 → 0.19, TbS3
0.46 → 0.14) (`shared_continuous_nearest.txt`); the Mac single-seed export agrees (0.45 / 1.10, row 2'). On
interpolation Chamfer is flat (10.93 ± 0.07 vs 10.32 ± 0.08, slightly worse) while tip error drops 5.38 → 3.65 ± 0.04,
below DPF's 4.00 and the best interpolation EPE on the benchmark (`shared_continuous_interp_nearest.txt`). On
extrapolation tip error again roughly halves, 21.21 → 11.70 ± 0.68 full-plant and 18.49 → 11.42 per-organ, on every
plant (MaC3 88.5 → 27.5), while Chamfer is mixed: full-plant 133.88 ± 22.72 vs 98.82 ± 12.72 (worse, driven by the three
monocots where the last frame's velocity is extrapolated for +1..+3 days) and per-organ 129.48 ± 16.97 vs 133.10 ±
17.60 (equal) (`shared_continuous_extrap_nearest.txt`). The accurate claim for the rule is therefore **tip accuracy
on every table and short-horizon fit**: it is the bar any learned prior must beat on the headline, and it is a change to
inference, not to the model. **Row 3 against row 1', row 3' against row 2** (all 8 plants): the monotone head costs
~6 % of headline CD under first-source rollout (0.92 ± 0.02 vs 0.87 ± 0.02) and about a third under nearest export
(0.58 ± 0.01 vs 0.44 ± 0.01), with headline EPE also worse (2.04 vs 1.53; 1.37 vs 1.02); it is equal on interpolation
(10.25 ± 0.05 vs 10.32; nearest 10.70 ± 0.04 / 3.68 ± 0.01 vs 10.93 / 3.65) and on per-organ extrapolation (136.29 ±
14.38 / 18.06 vs 133.10 / 18.49; nearest 142.34 ± 12.81 / 11.64 vs 129.48 / 11.42), and worse on full-plant
extrapolation (128.31 ± 11.25 vs 98.82 ± 12.72; nearest 170.48 ± 18.82 vs 133.88 ± 22.72, within the combined seed
spread). In return it guarantees non-decreasing base distance and, measured as Z-extent violations under nearest export,
cuts the violation rate 2–3× on every table: headline 0.028 vs 0.060 (GT floor 0.040), interpolation 0.007 vs 0.021
(0.035), extrapolation 0.206 vs 0.265 (0.039) (`results/gf_fleet/{shared,monotone}_continuous*_growth_nearest.txt`,
§4.4). **Rows 5–6 against row 2**: for MonoGrow the inference rule is worth 6.5× on the headline (385 → 59), 5.9× on
interpolation (351 → 59) and 3.3× on extrapolation (580 → 178), the same direction as for GrowFields and larger
because the code-only fit is coarser. **Row 5 against rows 1' and 4c**: MonoGrow ties the paper's per-organ 7-plant
extrapolation (151.67 / 20.12 vs 149.63 / 18.99) and is about one seed-std above the fleet's (133.10 ± 17.60 / 18.49)
with zero monotonicity violations and 25× cheaper fitting, but is 68× worse on the headline and 5.7× worse on
interpolation; RichardsNet-nearest beats first-objective MonoGrow on all three tables and beats GrowFields' first-frame
extrapolation EPE (13.91 vs 21.21 full-plant, 18.49 per-organ) though not GrowFields + nearest (11.70 / 11.42)
(`RESULTS.md`, Where things stand; `richards/README.md`, Results). **Row 7 against rows 5, 4c and 2**: fitting the codes
on the one-step objective they are scored with cuts the headline 4.2× (58.91 → 13.94) and interpolation 1.4× (59.06 →
41.96) at the same inference rule and prior, and leaves extrapolation inside the first objective's seed spread (193.40 ±
6.01 vs 177.81 ± 36.97 on 8 plants; 7-plant per-organ 156.46 ± 10.10 vs 151.67, fleet GrowFields 133.10 ± 17.60, paper
149.63). MonoGrow now beats RichardsNet-nearest on headline CD (13.94 vs 22.08) but not on headline EPE (10.70 vs
4.61), interpolation (41.96 vs 25.08) or extrapolation (193.40 vs 141.48), and stays 32× behind GrowFields + nearest on
the headline (0.44), 16× behind GrowFields itself (0.87) and 4× behind GrowFields on interpolation (10.32)
(`results/monogrow_gpu_test_obj_RESULTS.md`, Summary and Reference; `results/gf_fleet/`).
**Row 8**: the cosine-decay schedule removes the late-training degradation of the shared prior (validation
extrapolation 151.79 / 11.49 at 150k against 160.35 / 13.10 for the 95k checkpoint, and against 392 / 20.0
interpolation for the drifted constant-lr 150k checkpoint), with the 100k and 125k checkpoints worse than 150k, so the
fix is the decay and not an earlier stop; it has not been run on test or combined with the matched objective
(`results/monogrow_prior_v2cos_val_eval.txt`; `RESULTS.md`, Overnight results).

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
track. The per-organ 7-plant extrapolation CD, 106.96 (`last`), is below GrowFields' per-organ 149.63 (paper) and
133.10 ± 17.60 (our fleet) at ~1e-5 of the compute; the full-plant 7-plant CD 157.66 is above GrowFields' full-plant
117.71 (paper) and 98.82 ± 12.72 (fleet), so "ties CD" depends on the aggregation and we report both
(`richards/README.md`, Results; `growfields/BENCHMARK.md`, §2.4; `results/gf_fleet/shared_continuous_extrap_first.txt`).

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
the 95k checkpoint; the cosine-decay prior v2 (below) removes the degradation (`RESULTS.md`, MonoGrow findings, "150k
GPU prior").

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

**Fitting objective (95k prior, nearest source; `first` vs `nearest` objective × {nt0: z 300; nz1000: z 1000; nt1000:
z 300 + 1000 θ @ 1e-5}; single fits):** the matched objective wins every variant on validation full-plant CD: headline
59.23 (`first`, nz1000) → **3.50 / 7.33** (`nearest`, nt1000; per plant maize 3.1, sorghum 9.3, tobacco 1.4, tomato2
0.2); interpolation held-out 55.36 (`first`, nt0) → **38.99 / 8.43** (`nearest`, nt1000); extrapolation held-out 149.76
(`first`, nt1000) → **129.29 / 13.10** (`nearest`, nz1000). Under the matched objective θ fine-tuning helps on the fit
tables (z-only `nearest` gives 44.51 headline, 52.76 interpolation) and hurts on extrapolation (nt1000 142.05 vs nz1000
129.29); the violation rate is 0.000 in every cell. These three winners are the test recipe of §4.3b
(`results/monogrow_gpu_val_obj_summary.md`; `results/monogrow_gpu_val_obj_decision.md`).

**Cosine-decay prior v2 (150k steps, lr decayed to 0, tagged checkpoints; z-only 300, nearest source, `first`
objective):** validation extrapolation held-out 185.22 / 12.05 (100k), 180.88 / 12.75 (125k), **151.79 / 11.49**
(150k; per-organ 121.86 / 14.28) against 160.35 / 13.10 for the constant-lr 95k checkpoint; interpolation held-out
62.24 / 8.14, 64.86 / 8.54, **57.51 / 8.31** (150k; per-organ 52.11 / 11.26) against 55.36 / 9.06. The constant-lr 150k
checkpoint was at 392 / 20.0 on the same interpolation split, so the decay removes the late-training degradation, and
the checkpoints improve monotonically to 150k, so the gain is from the schedule and not from stopping earlier. Not yet
combined with the matched objective and not run on test (`results/monogrow_prior_v2cos_val_eval.txt`; `RESULTS.md`,
Overnight results).

**Pipeline evidence with a 3k-step prior (SoH2, test plant, 0.4 % of planned steps):** z-only 93.1 / 20.6; plus 200 θ
steps at lr 1e-5: 3.97 / 6.23 with 0 base-distance violations (GrowFields 0.74 / 0.97 after 50k steps)
(`monogrow/README.md`, Verified). These are not results; they show the stage-2 path works.

### 4.3b MonoGrow, test split (3 seeds, both fitting objectives)

95k-step CPU prior, `nearest` source, seeds 0–2 (point-sampling seed; z starts at μ_s, so the spread is sampling noise
only), GrowFields' unmodified metric scripts. **First objective** (GrowFields' multi-step from-first-frame CD), recipe
from the validation grid: headline and interpolation z-only (300 steps), extrapolation + 1000 θ steps at lr 1e-5
(`RESULTS.md`, table rows "MonoGrow, nearest source" and "MonoGrow, first-frame source", copied from
`monogrow/runs/gpu_test/RESULTS.md`). **Matched objective** (`fit.objective=nearest`, §2.3), recipe from the validation
objective grid (§4.3): headline and interpolation z 300 + 1000 θ steps at lr 1e-5, extrapolation z-only 1000 steps,
3–4 min per plant on the L4 (`results/monogrow_gpu_test_obj_RESULTS.md`).

| MonoGrow, test, first objective | Headline | Interpolation (held-out) | Extrapolation (held-out) |
|---|---|---|---|
| nearest source, full-plant, 8 plants | 58.91 ± 0.41 / 12.95 ± 0.30 | 59.06 ± 0.84 / 10.87 ± 0.08 | 177.81 ± 36.97 / 21.33 ± 0.50 |
| nearest source, 7 plants (GF's extrapolation set) | | | 191.97 / 22.38 full-plant; 151.67 / 20.12 per-organ (GF paper per-organ 149.63 / 18.99, full-plant 117.71 / 21.79) |
| first-frame source, same fitted codes | 385.48 / 27.79 | 350.81 / 25.02 | 579.67 / 40.81 |
| Z-extent violation rate (GT floor) | 0.000 (0.040) | 0.000 | 0.000 |

| MonoGrow, test, matched objective | Headline | Interpolation (held-out) | Extrapolation (held-out) |
|---|---|---|---|
| nearest source, full-plant, 8 plants | **13.94 ± 2.90 / 10.70 ± 0.33** | **41.96 ± 3.37 / 10.63 ± 0.38** | 193.40 ± 6.01 / 20.50 ± 0.55 |
| nearest source, per-organ, 8 plants | 12.99 ± 1.69 / 12.99 ± 0.24 | 43.18 ± 2.82 / 11.11 ± 0.24 | 148.32 ± 8.43 / 18.03 ± 0.39 |
| nearest source, 7 plants (GF's extrapolation set) | | | 210.92 ± 7.09 / 21.87 ± 0.68 full-plant; 156.46 ± 10.10 / 19.25 ± 0.53 per-organ |
| first-frame source, same fitted codes / θ | 118.89 ± 12.29 / 23.17 ± 1.68 | 142.85 ± 4.52 / 23.55 ± 1.39 | 564.40 ± 51.32 / 37.34 ± 1.19 |
| Z-extent violation rate (GT floor) | 0.000 (0.040) | 0.000 (0.039) | 0.000 (0.040) |
| length MAE mm, observed / held-out | 2.4 / — | 7.8 / 5.9 | 3.9 / 15.3 (+1/+2/+3 d: 7.5 / 16.7 / 21.9) |

Per plant, matched objective, nearest source, full-plant CD / EPE (mean over seeds; held-out frames for interpolation
and extrapolation) (`results/monogrow_gpu_test_obj_RESULTS.md`, Per plant):

| | MaC2 | MaC3 | SoC2 | SoH2 | TbC1 | TbS3 | Tm1H3 | Tm1S1 | Mean |
|---|---|---|---|---|---|---|---|---|---|
| headline | 4.58 / 13.04 | 21.96 / 29.96 | 40.40 / 17.25 | 41.84 / 15.01 | 1.25 / 4.40 | 0.98 / 2.87 | 0.16 / 1.32 | 0.35 / 1.73 | 13.94 / 10.70 |
| interpolation | 68.43 / 17.01 | 45.92 / 23.87 | 88.29 / 11.80 | 81.59 / 12.21 | 23.00 / 7.98 | 19.17 / 6.18 | 3.62 / 2.62 | 5.65 / 3.37 | 41.96 / 10.63 |
| extrapolation | 267.70 / 21.07 | 907.57 / 84.75 | 175.70 / 17.41 | 70.75 / 10.91 | 46.27 / 10.10 | 40.19 / 10.19 | 19.79 / 4.55 | 19.20 / 4.97 | 193.40 / 20.50 |

Four readings. (i) Inference dominates again: under the first objective the same codes rolled out from the first frame
are 6.5× / 5.9× / 3.3× worse on the three tables; under the matched objective 8.5× / 3.4× / 2.9×. (ii) The
objective/inference mismatch was most of MonoGrow's headline gap: matching the fitting loss to the one-step rule cuts
the headline 4.2× (58.91 → 13.94) and interpolation 1.4× (59.06 → 41.96) with the same prior and the same 32 numbers
per organ, and the per-plant pattern follows the prior's species coverage — tomato and tobacco at 0.16–1.25 mm², maize
and sorghum at 4.6–42 (§3, Leakage rule). The headline EPE barely moves (12.95 → 10.70) and stays above RichardsNet's
4.61: the matched loss places surfaces, not tips. (iii) On GrowFields' own per-organ 7-plant extrapolation table MonoGrow
ties the paper's GrowFields under either objective (151.67 / 20.12 first objective, 156.46 / 19.25 matched, paper
149.63 / 18.99) and sits about one seed-std above our fleet's 133.10 ± 17.60 / 18.49
(`results/gf_fleet/shared_continuous_extrap_first.txt`), with zero monotonicity violations and one to four minutes of fitting per plant against 20–45 minutes for
GrowFields on the same GPU (`results_headline.md`, training times); the matched objective is slightly worse on 8-plant
full-plant extrapolation (193.40 vs 177.81) but inside the first objective's seed spread (± 36.97 → ± 6.01), and MaC3
alone (907.57 / 84.75) is almost half of the 8-plant sum. (iv) It still does not close the fit gap: 13.94 against
GrowFields' 0.87 (first) / 0.44 (nearest) on the headline, 41.96 against 10.32 (fleet; paper 10.29) on interpolation.
RichardsNet with
nearest templates is now behind MonoGrow on headline CD (22.08 vs 13.94) but ahead on headline EPE (4.61 vs 10.70),
interpolation (25.08 / 5.48 vs 41.96 / 10.63) and extrapolation (141.48 / 13.91 vs 193.40 / 20.50 on 8 plants; row
4c). The cosine-decay prior (§4.3, row 8) has not been combined with the matched objective; that combination is the next
MonoGrow run (`RESULTS.md`, Overnight results).

### 4.4 Growth-plausibility metrics

| Model / run | Violation rate (GT floor) | Negative-growth mass mm (GT) | Length MAE mm observed / held-out | Source |
|---|---|---|---|---|
| GrowFields original, SoH2, 200-step smoke (12 frame pairs) | 0.167 (0.25, tol 0) | — | 5.24 / — | NOTES.md §3 |
| GrowFields + monotone head, SoH2, 200-step smoke | **0.000** (0.25, tol 0) | — | 16.41 / — (under-grows at 200 steps) | NOTES.md §3 |
| GrowFields original, 50k, 5 seeds, 8 plants, first source (headline, per-organ) | 0.055 (0.062) | 1.66 | 0.90 / — | runs_aws/merged/shared_continuous/results_headline.md |
| GrowFields + monotone head, 50k, first source, `--mode all` (6-plant intermediate file; 0 by construction on any plant) | **0.000** (0.049) | 0.00 | 1.08 / — | runs_aws/merged/monotone_continuous/results_headline.md |
| GrowFields original + nearest export, headline, 5 seeds, 8 plants (325 organ frames) | 0.060 ± 0.002 (0.040) | 118.8 total (GT 86.5) | 0.95 / — | results/gf_fleet/shared_continuous_growth_nearest.txt |
| GrowFields + monotone head + nearest export, headline, 5 seeds, 8 plants | **0.028 ± 0.000** (0.040) | 37.1 total (86.5) | 1.17 / — | results/gf_fleet/monotone_continuous_growth_nearest.txt |
| GrowFields original + nearest, interpolation held-out, 3 seeds (145 organ frames) | 0.021 (0.035) | 28.2 total (38.1) | — / 2.30 | results/gf_fleet/shared_continuous_interp_growth_nearest.txt |
| GrowFields + monotone head + nearest, interpolation held-out, 3 seeds | **0.007** (0.035) | 9.8 total (38.1) | — / 2.32 | results/gf_fleet/monotone_continuous_interp_growth_nearest.txt |
| GrowFields original + nearest, extrapolation held-out, 3 seeds (102 organ frames) ‡ | 0.265 ± 0.026 (0.039) | 314.2 total (26.6) | 7.51 (held-out, chain vs GT) | results/gf_fleet/shared_continuous_extrap_growth_nearest.txt |
| GrowFields + monotone head + nearest, extrapolation held-out, 3 seeds ‡ | **0.206 ± 0.017** (0.039) | 285.5 total (26.6) | 7.18 (held-out, chain vs GT) | results/gf_fleet/monotone_continuous_extrap_growth_nearest.txt |
| MonoGrow, test, nearest, 3 seeds, first objective (headline / interp / extrap) | 0.000 (0.040) | — | — | RESULTS.md table (gpu_test) |
| MonoGrow, test, nearest, 3 seeds, matched objective (headline / interp / extrap) | 0.000 (0.040 / 0.039 / 0.040) | 0.0 / 0.1 / 0.0 (86.5 / 78.6 / 81.6) | 2.4 / — ; 7.8 / 5.9 ; 3.9 / 15.3 | results/monogrow_gpu_test_obj_RESULTS.md |
| RichardsNet, first template, any variant | 0.000 | — | 6.1 / — (headline); 5.2 / 11.9 (extrap) | richards/README Results |
| RichardsNet, nearest, headline / extrap | 0.000 | 0 | 2.5 / — ; 2.4 / 8.7 (+1/+2/+3: 3 / 10 / 13) | richards/README Results |
| RichardsNet, last, headline | 0.006 (0.040) † | 9.4 (86.5) † | 2.0 / — | richards/README Results |
| MonoGrow, all smoke / 3k runs, base-distance | 0.000 | 0.0 | — | monogrow/README Verified |
| MonoGrow + θ 2k, val extrap, Z-extent, first source | 0.105 (0.035) ‡ | — | 18.7 (held-out) | monogrow/README sweep |

† Metric artefact: the `last` template inverse-scales from a later frame, so the chain's first "pair" compares a
prediction to GT rather than two predictions; the Richards curve itself is monotone (k > 0) (`richards/README.md`).
‡ On held-out frames the chain starts at the GT length of the frame before the first held-out prediction (and, in
`nearest` mode, every frame is referenced to the GT length of its own source frame; under extrapolation all three
held-out frames share the last observed frame as source), so these rows are prediction-vs-GT error, not field
monotonicity violations; `--mode all` gives 0 for the monotone head (`monogrow/README.md`, Rollout source caveat).
Base-distance monotonicity is the guaranteed property; Z-extent is not.

On the complete fleet the picture is: under first-source rollout GrowFields' original field shrinks organs at about the
rate the ground truth does (0.055 vs floor 0.062, 1.66 mm of negative-growth mass per organ pair), and the monotone
head removes every violation by construction (0.000) at a cost of ~0.2 mm in observed-frame length MAE
(`results_headline.md`, both configs). Under nearest export each frame is referenced to the GT length of its own source
frame, so the rates are prediction-vs-GT noise around the field rather than field violations, and the GT floor itself
is 0.040 / 0.035 / 0.039; measured that way the monotone head is 2–3× below the original field on every table:
headline 0.028 vs 0.060 (negative-growth mass 37 vs 119 mm, GT 86), interpolation 0.007 vs 0.021 (10 vs 28 mm, GT 38),
extrapolation 0.206 vs 0.265 (285 vs 314 mm, GT 27) (`results/gf_fleet/*_growth_nearest.txt`). The extrapolation rates
are high for both fields because all three held-out frames are predicted from the same last observed frame and the
chain compares three independent predictions to one another and to that frame; they are not comparable to the headline
rates, and the metric caveat (‡) applies. Held-out length MAE is equal for the two heads (interpolation 2.30 vs 2.32 mm,
extrapolation 7.51 vs 7.18 mm), so the constraint does not cost length accuracy where the field generalises. MonoGrow is
at 0.000 on all three test tables under both fitting objectives against a GT floor of 0.039–0.040 (`RESULTS.md`,
table; `results/monogrow_gpu_test_obj_RESULTS.md`).

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
   CPU checkpoint is the healthy reference used for every test number (`RESULTS.md`, Compute log; `aws/fleet_poll.log`).
   The v2 prior with cosine decay to 0 removes the drift on validation (extrapolation 151.79 / 11.49 at 150k vs
   160.35 / 13.10 at 95k; interpolation 57.51 / 8.31 vs 55.36 / 9.06, `results/monogrow_prior_v2cos_val_eval.txt`) but
   has not been used on test. Any MonoGrow number must state which checkpoint and which fitting objective it used.
6. **Seeds.** The GrowFields and monotone-head rows (1', 2, 3, 3') are 5 seeds headline and 3 seeds interpolation /
   extrapolation on the L4 fleet, all 8 plants (7 for extrapolation), reported as mean ± std over seeds
   (`results/gf_fleet/*.txt`); the headline reproduces the paper within its own ± (0.02 CD / 0.06 EPE ours vs 0.02 /
   0.11 paper; `growfields/BENCHMARK.md`, §1.5). Our extrapolation Chamfer is lower than the paper's (98.82 vs 117.71
   full-plant, 133.10 vs 149.63 per-organ) but inside the paper's own ± 35.55 / ± 19.65; we have not identified a
   cause and quote both. The extrapolation seed spread is large (± 12–23 mm² on means of 99–170), so the full-plant
   Chamfer differences between rows 1', 2, 3 and 3' on that table are at most ~1–2 σ and we do not claim them; the
   EPE differences (21 → 12 mm, ± 0.5–1.9) are not in doubt. The Mac rows are one seed (43) on CPU. EPE varies ~1 mm
   between identical CUDA runs (`growfields/BENCHMARK.md`, §5.9).
7. **MonoGrow's test numbers are 3 seeds with recipes tuned on validation plants** whose species mix differs (val has
   tomato2, test has tomato1), and the matched-objective recipe was chosen on the same four plants after the first
   objective's test runs were in. The headline gap to GrowFields (13.94 vs 0.44–0.87) is a model limitation, not a
   seed effect (± 2.90); the first objective's extrapolation spread (± 36.97) was monocot-driven and the matched
   objective's (± 6.01) is small. MonoGrow beats RichardsNet-nearest on headline CD only, does not beat GrowFields on
   any fit table and ties it only on per-organ extrapolation (`results/monogrow_gpu_test_obj_RESULTS.md`; `RESULTS.md`,
   table).
8. **The matched objective is a fitting change, not a model change.** Like the nearest-source rule it is available to
   any method, including GrowFields' own 50k-step per-plant fit; we have not run GrowFields with a one-step-pair loss,
   so rows 1'–3' remain `first`-objective fits (`monogrow/README.md`, Objective).
9. **Species coverage of the prior is uneven.** Maize rests on 15 Pheno4D tracks with noisy labels; sugar beet is 65 %
   of tracks (species-balanced sampling mitigates this) (`monogrow/README.md`, Open issues).
10. **RichardsNet's headline failure is structural**: axis-aligned scaling cannot represent monocot leaf unfolding; the
   residual MLP ε_ψ, densification and emergence generation are not implemented (`richards/README.md`, Unverified).
11. **Not implemented from the design memo**: the saturation gate, 4× source densification, the hypernetwork and
    MetaSDF variants, and Candidate B (trajectory diffusion with an emergence model) (`monogrow/README.md`, Open issues).
12. **Compute reporting.** GrowFields reports no compute; our timings are on an Apple M5 Pro and an NVIDIA L4 and are
    not comparable across the two (`growfields/BENCHMARK.md`, §3; `RESULTS.md`, Compute log).

---

## 6. Conclusion

GrowFields reproduces. On an L4 fleet with 5 / 3 / 3 seeds and all 8 test plants we obtain 0.87 ± 0.02 / 1.53 ± 0.06,
10.32 ± 0.08 / 5.38 ± 0.14 and 98.82 ± 12.72 / 21.21 ± 0.52 (133.10 ± 17.60 / 18.49 per-organ) against the paper's
0.86 / 1.47, 10.29 / 5.82 and 117.71 / 21.79 (149.63 / 18.99) (`results/gf_fleet/shared_continuous*_first.txt`). On
that benchmark the single largest lever is not a model. Predicting each frame from the nearest observed frame instead
of the first one halves GrowFields' headline on the same saved models, 0.87 → 0.44 mm² CD and 1.53 → 1.02 mm EPE,
improving all 8 plants with no retraining, and roughly halves tip error on every table (5.38 → 3.65 interpolation,
21.21 → 11.70 extrapolation) while Chamfer is flat on interpolation (10.93 vs 10.32) and mixed on extrapolation (133.88
± 22.72 vs 98.82 ± 12.72 full-plant, 129.48 vs 133.10 per-organ) (`results/gf_fleet/shared_continuous*_nearest.txt`).
The claim we make for it is tip accuracy and short-horizon fit, not extrapolation Chamfer. It is the dominant effect
for every method we ran: 6.5× / 5.9× / 3.3× on MonoGrow's three test tables (`RESULTS.md`, table). Any future
comparison on this benchmark should state the source rule next to the number.

The other two axes are smaller and honestly mixed. The monotone velocity head, dropped into GrowFields' own code for
513 parameters, costs ~6 % of headline CD (0.92 vs 0.87; 0.58 vs 0.44 under nearest export, where the gap is a third)
and some headline tip error (2.04 vs 1.53), is equal on interpolation (10.25 vs 10.32) and per-organ extrapolation
(136.29 vs 133.10), worse on full-plant extrapolation (128.31 ± 11.25 vs 98.82 ± 12.72, under 2 σ), and gives an exact guarantee on base
distance with 2–3× fewer Z-extent violations on every table (0.028 vs 0.060 headline, 0.007 vs 0.021 interpolation,
0.206 vs 0.265 extrapolation) (`results/gf_fleet/monotone_continuous*.txt`, `*_growth_nearest.txt`). The cross-plant
prior, MonoGrow, fitted by a few hundred code steps per plant in one to four minutes, ties the paper's per-organ
7-plant extrapolation (151.67 / 20.12 with GrowFields' fitting loss, 156.46 / 19.25 with the matched one-step loss, vs
149.63 / 18.99; our fleet 133.10 ± 17.60 / 18.49) with zero violations and 3-seed spreads reported. Its fit-table
numbers in v0.2 rested on a mismatch: the codes were fitted on a multi-step rollout from the first frame and scored on
one-step predictions from the nearest frame. Fitting them on the quantity scored moves the headline from 58.91 to
13.94 mm² and interpolation from 59.06 to 41.96 (`results/monogrow_gpu_test_obj_RESULTS.md`), now below the
four-parameter RichardsNet on headline Chamfer (22.08) but not on tip error (4.61 vs 10.70), interpolation (25.08 /
5.48) or extrapolation (141.48 / 13.91 vs 193.40 / 20.50 on 8 plants); RichardsNet itself beats GrowFields'
first-frame extrapolation tip error (13.91 vs 21.21 mm) at ~1e-5 of the compute but not GrowFields + nearest (11.70)
(`richards/README.md`, Results). Both priors trail GrowFields by an order of magnitude on the fit tables: GrowFields +
nearest at 0.44 is 32× ahead of MonoGrow on the headline and GrowFields 4× ahead on interpolation (10.32 vs 41.96).
The cosine-decay schedule fixes the shared prior's late-training drift on validation (extrapolation 151.79 / 11.49 vs
160.35 / 13.10, `results/monogrow_prior_v2cos_val_eval.txt`) and has not yet been combined with the matched objective.

The title therefore stays "decomposing": the chain in §4.1 attributes GrowFields' headline to inference strategy
first, per-plant fitting second, and shows that neither a monotone constraint nor a cross-plant prior has substituted
for the latter — the one new lever, matching the fitting objective to the inference rule, is itself a fitting change
available to any method. To beat GrowFields + nearest on the fit tables a follow-up needs per-plant capacity on the
order of GrowFields' 1.3 M parameters fitted with the one-step objective (stage-2 θ fine-tuning of MonoGrow for
thousands rather than 1,000 steps, or GrowFields itself re-fitted on the matched loss, neither run here): the per-plant
pattern of §4.3b, tomato and tobacco at 0.16–1.25 mm² and maize and sorghum at 4.6–42, says the remaining gap is on
the monocots the prior covers worst, where 32 code dimensions per organ have not sufficed under either objective. The
planned extension evaluates GrowFields and GrowFields + nearest on our own Pheno4D full-resolution and
Sugar4D sequences (`experiments/gf_on_ours`, being prepared; no runs yet), where tracks are longer and the headline
protocol's fitting-score character can be separated from temporal generalisation.

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
# matched objective: add fit.objective=nearest (val grid + decision: runs/gpu_val_obj/{run.sh,chain.sh,decide.py};
#   test recipes per variant in the header of results/monogrow_gpu_test_obj_RESULTS.md, runs in runs/gpu_test_obj/)

# RichardsNet
cd ../richards; PY=../monogrow/.venv/bin/python
$PY prior.py && ./run_variant.sh test headline richards nearest && ./run_variant.sh test extrapolation richards nearest
$PY collect.py --split test
```
