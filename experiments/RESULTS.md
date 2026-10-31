# Results on the GrowFields benchmark (TrackPlant3D test split)

CD = squared Chamfer on test points, mm² (sum of both directional means). EPE = leaf-tip endpoint error, mm.
Headline = fit on all frames, scored on held-out points. Interpolation = train on frames 0,2,4,…, scored on held-out frames.
Extrapolation = hold out last 3 frames (GF's table uses 7 plants, SoH2 excluded). Full-plant aggregation unless noted.
All numbers below are ours unless marked "paper".

| Model | Steps / plant | Headline CD / EPE | Interp CD / EPE | Extrap CD / EPE (7 plants) | Monotone violations | Notes |
|---|---|---|---|---|---|---|
| GrowFields (paper, 5 seeds) | 50k | 0.86 / 1.47 | 10.29 / 5.82 | 149.63 / 18.99 | not reported | per-plant fitting, 1.33 M params |
| DPF per-organ (paper) | — | 2.47 / 15.15 (single) | 10.24 / 4.00 | — | — | |
| COAP (paper) | — | 1.69 / 4.23 | — | 451.09 / 31.39 | — | |
| GrowFields (our repro, Mac CPU, 1 seed, all 8 plants) | 50k | **0.88 / 1.53** (per-organ 0.89 / 1.60); MaC2 2.65/2.81 (paper 2.51/2.38), MaC3 1.40/1.79 (1.43/2.01), SoC2 0.69, SoH2 0.78, TbC1 0.57, TbS3 0.53, Tm1H3 0.12, Tm1S1 0.32 | — | — | — | runs/shared_continuous; 3–6.5 h per plant on CPU; matches paper 0.86/1.47 |
| GrowFields (our repro, AWS L4 fleet, 5/3/3 seeds, all 8 plants) | 50k | **0.87 ± 0.02 / 1.53 ± 0.06** (per plant within ±0.1 of paper) | **10.32 ± 0.08 / 5.38 ± 0.14** (per-organ 11.46 / 4.14; paper 10.29 / 5.82) | **98.82 ± 12.72 / 21.21 ± 0.52** full, **133.10 ± 17.60 / 18.49 ± 0.35** per-organ (paper 117.71 / 21.79 and 149.63 / 18.99) | — | runs_aws/merged/shared_continuous; ~25 min per run on L4 |
| GrowFields + nearest-source inference (AWS fleet, same models, 5/3/3 seeds) | 50k | **0.44 ± 0.01 / 1.02 ± 0.04** (per-organ 0.48 / 1.03) | 10.93 ± 0.07 / **3.65 ± 0.04** (per-organ 10.97 / 3.77; beats DPF's 4.00 EPE) | 133.88 ± 22.72 / **11.70 ± 0.68** full, 129.48 ± 16.97 / 11.42 per-organ (CD mixed, EPE halved) | 0.057 (GT 0.062) | runs_aws/merged/shared_continuous/results_headline_nearest.md; same models as the 0.87 row |
| GrowFields + monotone head (ours, 5/3/3 seeds, 8 plants) | 50k | 0.92 ± 0.02 / 2.04 ± 0.13 (per-organ 0.93 / 2.18) | 10.25 ± 0.05 / 5.61 ± 0.08 (per-organ 11.01 / 4.49) | 128.31 ± 11.25 / 21.00 ± 0.68 full, 136.29 ± 14.38 / 18.06 per-organ | see results/gf_fleet/*growth* | same params; fit cost ~6% headline CD, equal interp, extrap per-organ equal |
| GrowFields + monotone head + nearest source | 50k | 0.58 ± 0.01 / 1.37 ± 0.03 (per-organ 0.67 / 1.45) | 10.70 ± 0.04 / 3.68 ± 0.01 (per-organ 10.55 / 3.75) | 170.48 ± 18.82 / 12.24 ± 1.86 full, 142.34 ± 12.81 / 11.64 per-organ | see results/gf_fleet/*growth* | results/gf_fleet/ |
| GrowFields + nearest-source inference (ours, post-hoc from saved GF models, 1 seed, all 8 plants) | 50k | **0.45 / 1.10** (per-organ 0.49 / 1.08); MaC2 1.16/1.84 (first 2.65/2.81), MaC3 0.64/1.33 (1.40/1.79), Tm1S1 0.19/0.73 (0.32/1.25), SoC2 0.44, SoH2 0.72, TbC1 0.19, TbS3 0.14, Tm1H3 0.09 | pending | pending | — | export_nearest_source.py; no retraining; the bar MonoGrow must beat |
| RichardsNet (ours, first-frame template) | 0 (closed form) | 112.09 / 8.51 | 99.70 / 8.69 | 238.18 / 17.93 | 0.000 | 4 params/organ, 0.1 s total |
| RichardsNet (ours, last-observed template) | 0 | 23.02 / 5.84 * | 20.79 / 7.14 * | 157.66 / 17.85 | ~0 | * template frame scored at 0, not comparable to GF's frame-0 rollout |
| RichardsNet (ours, nearest-frame template) | 0 | 22.08 / 4.61 | 25.08 / 5.48 | 141.48 / 13.91 (8 plants; 7-plant CD 157.66) | 0.000 | every frame a 1-day prediction; beats GF on extrapolation EPE, ties CD |
| MonoGrow, nearest source (95k prior, z-only 300 steps; extrap +1000 θ @1e-5; 3 seeds, test) | 300 z (~1 min/plant) | 58.91 ± 0.41 / 12.95 ± 0.30 | 59.06 ± 0.84 / 10.87 ± 0.08 (held-out) | 177.81 ± 36.97 / 21.33 ± 0.50 (8 plants); 7 plants 191.97 / 22.38 full, **151.67 / 20.12 per-organ** (GF per-organ 149.63 / 18.99) | 0.000 (GT 0.040) | runs/gpu_test/RESULTS.md; ties GF on per-organ extrapolation, loses on fit tables; RichardsNet-nearest is better on all three |
| **MonoGrow, matched one-step objective** (fit.objective=nearest; 95k prior; headline/interp +1000 θ @1e-5, extrap n_z 1000; 3 seeds, test) | ~1k steps (3–4 min/plant) | **13.94 ± 2.90 / 10.70 ± 0.33** (per-organ 12.99 / 12.99) | **41.96 ± 3.37 / 10.63 ± 0.38** (held-out; per-organ 43.18 / 11.11) | 193.40 ± 6.01 / 20.50 ± 0.55 (8 plants; per-organ 148.32 / 18.03) | 0.000 (GT 0.040) | results/monogrow_gpu_test_obj_RESULTS.md; val: headline 59→3.5, interp 55→39, extrap 150→129 |
| MonoGrow, first-frame source (same codes) | 300 z | 385.48 / 27.79 | 350.81 / 25.02 | 579.67 / 40.81 | 0.000 | shows the nearest-source effect for MonoGrow: 6.5x headline, 5.9x interp, 3.3x extrap |

## Reading so far (2026-10-30)
- The structured curve model already matches GrowFields on extrapolation CD and beats it on EPE with no learning at test time.
  Its headline fit is poor because axis-aligned scaling cannot represent leaf unfolding (SoH2 leaf 2: width 6 → 26 mm at constant length).
- The monotone head in GrowFields' own code gives zero shrinking organs at no parameter cost (smoke run); full numbers pending.
- Protocol caveat we must state in any write-up: GF's extrapolation normalises time over held-out frames and GT-fills emerging organs.

## MonoGrow findings (2026-10-30, 50k-step prior, validation plants)
- Code-only test-time fitting (z, 300 steps) on the 4 val plants, headline: tomato2 13.4 mm², tobacco 71.9, sorghum 869, maize 1,608 (mean 641). lr_z 3e-3 beats 1e-2 (324 vs 641 mean).
  Monocots with few training tracks (maize 15, sorghum 44) are not covered by the code prior; field fine-tuning per plant is required to reach GrowFields' regime.
- Stage-2 θ fine-tuning, 2k steps lr 2e-5, val extrapolation held-out frames: maize 704, sorghum 514, tobacco 60, tomato2 15 (mean 323 / EPE 31.1). RichardsNet on the same split: 162.6 / 11.9. So MonoGrow-as-is loses to the explicit curve model on extrapolation; the monocots dominate.
- Monotonicity caveat: the velocity factorisation guarantees non-decreasing base distance ||x||, NOT non-decreasing axial (Z) extent — tangential w can rotate tips. After θ fine-tuning the Z-extent violation rate rose to 0.105 (GT floor 0.035). Report base-distance monotonicity as the guaranteed property; add a λ_w penalty or a Z-monotone parameterisation if Z-extent matters.
- 5k-step and lr 1e-4 fine-tunes running; decision pending on a hybrid (Richards growth backbone + learned residual field).
- **Rollout source = nearest observed frame** (fit.source=nearest; same 50k prior, z-only, val plants, held-out frames):
  extrapolation 538.2/30.5 (first) → **180.3/12.1** (nearest); per plant maize 1380→252, sorghum 705→436, tobacco 46→21, tomato2 21.5→12.4; held-out length MAE 28.2→12.9 mm (+1/+2/+3 d: 6.2/14.3/18.3).
  interpolation 511.4/18.3 → **62.5/8.1**; length MAE 15.6→6.9 mm. n_tips unchanged; GF's own metric script agrees row-wise.
  Protocol-legal (only observed frames used). GF always rolls out from frame 0 — this is its structural weakness; plan a "GF + nearest source" ablation so the comparison decomposes into inference strategy / monotone prior / cross-plant prior.
- θ fine-tune lr sweep (first-source, val extrapolation): 2k@2e-5 → 323; 5k@2e-5 → 454 (maize worse); 2k@1e-4 → 2.7e5 (blows up, same instability as prior training at 1e-4). Fine-tune lr must stay ≤ 2e-5.
- **Prior length** (nearest source, z-only, val extrapolation held-out): 50k-step prior 180.3/12.1 → 95k-step prior **131.5/12.6** (maize 216, sorghum 262, tobacco 36, tomato2 11.6). MonoGrow z-only now beats RichardsNet-last (162.6/11.9) on CD at equal protocol. The 150k GPU prior is pending. Interpolation (same settings): 50k prior 62.5/8.1 → 95k prior 55.9/8.9.
- **GPU val grid (95k prior, nearest source)**: interpolation held-out: z-only 55.4/9.1 best; every θ fine-tune config worse (68–339). Extrapolation: nt1000 lr1e-5 149.8/14.0 vs z-only 160.4/13.1 (marginal); lr 2e-5 harmful, 5k@2e-5 diverges. Violation rate 0.000 everywhere. Recipe: z-only (300 steps, ~1 min/plant on L4) for headline+interpolation; +1000 θ steps @1e-5 for extrapolation (alt z-only also run). Test campaign launched 20:57 UTC.
- Objective/inference mismatch to fix next: z is fitted with the multi-step from-first-frame CD, inference is one-step from the nearest frame.
- 150k GPU prior confirmed degraded: val interpolation nearest z-only 392/20.0 vs 95k CPU prior 55.4/9.1 (GPU box gpu_val). Use the 95k prior; prior v2 (cosine) is the fix.
- Caveat found: the tuning dry-run's extrapolation/interpolation rows (runs/tune/*) used an early snapshot mislabelled step 0; ignore those rows.

## Final GrowFields fleet tables (2026-10-31 14:44 UTC, all 170 runs; results/gf_fleet/)
- GF reproduces on all three tables: headline 0.87/1.53, interpolation 10.32/5.38, extrapolation 98.8/21.2 full-plant (133.1/18.5 per-organ) vs paper 0.86/1.47, 10.29/5.82, 117.7/21.8 (149.6/19.0).
- Nearest-source inference on GF's own models: headline halved (0.44/1.02); interpolation CD flat (10.93) but EPE 5.38→3.65 (below DPF's 4.00); extrapolation CD mixed (133.9 vs 98.8, ±23) with EPE 21.2→11.7. Tip accuracy improves everywhere; Chamfer improves where the horizon is short.
- Monotone head: ~6% headline CD cost (0.92 vs 0.87), interpolation equal (10.25 vs 10.32), extrapolation per-organ equal (136.3 vs 133.1); with nearest source 0.58 / 10.70 / 170.5 (EPE 1.37 / 3.68 / 12.24).

## Overnight results (2026-10-31)
- Matched objective (fit codes/θ on one-step nearest-frame CD, the quantity inference uses): val headline 59.2→3.50, interpolation 55.4→39.0, extrapolation 149.8→129.3 (results/monogrow_gpu_val_obj_decision.md). Test (3 seeds): headline 13.94/10.70 (was 58.9/12.95), interpolation 41.96/10.63 (was 59.1/10.9), extrapolation 193.4/20.5 (was 177.8/21.3). MonoGrow now beats RichardsNet-nearest on headline CD (13.9 vs 22.1) but not on tip EPE (10.7 vs 4.6) and not on interpolation (42 vs 25); GF+nearest (0.44) remains far ahead on the fit tables.
- Cosine-decay prior (150k, no late drift; log in results/priors/log_v2cos.csv): val z-only nearest extrapolation 151.8/11.5 (95k prior 160.4/13.1), interpolation 57.5/8.3 (55.4/9.1); checkpoints at 100k/125k worse than 150k, so the decay fixed the degradation. Not yet combined with the matched objective on test.
- Fleet: gf2 shard (57 jobs) complete and pulled with models; gf1/gf3 unreachable at 13:00 UTC (auto-stopped after finishing, presumably) — their shards (interp/extrap jobs for shards 0 and 2, monotone headline remainder) must be started and pulled before termination; AWS CLI session expired (`aws login` needed).

## Where things stand (2026-10-30 22:20 UTC)
- Inference strategy (nearest observed frame) is the dominant effect for every method: GF 0.87→0.44 headline; MonoGrow 385→59 headline, 351→59 interp, 580→178 extrap.
- RichardsNet-nearest (4 params/organ) beats MonoGrow on all three tables and beats GF on extrapolation EPE; GF (+nearest) remains best on the fit tables.
- MonoGrow's cross-plant prior gives zero violations and 25x cheaper fitting but has not closed the fit gap; the matched-objective fix and the cosine prior are running overnight on the GPU box.

## Decomposition plan (for the write-up)
Gains are reported as a chain: GF (paper, frame-0 rollout) → GF + nearest-source inference → GF + monotone head (+ nearest) → MonoGrow (cross-plant prior + monotone + nearest, z-only) → MonoGrow + per-plant θ fine-tune. RichardsNet is the structured baseline on every table.

## Compute log
- Mac M5 Pro CPU: GF 0.138 s/step single process, 0.3–0.5 s/step with 4 concurrent. MonoGrow prior 0.24–0.30 s/step.
- MonoGrow prior stability: lr_theta 1e-4 diverged at step ~5000 (blow-up) and, with grad clipping, drifted from step ~7000 (growth head g 0.05 → 0.21). 6k-step sweep: A (lr_theta 2e-5, lambda_g 1e-3, clip 1.0) final batch CD 25.9 mm², B (5e-5, 1e-2, clip 0.5) 35.3 mm², both stable. Full run uses A. Even at 2e-5 the growth head drifts late: on the GPU run g rose monotonically 0.016 (95k) → 0.107 (145k) with batch CD trending up; the 150k checkpoint is suspect, the ~95k CPU checkpoint is the healthy reference. Prior v2 (cosine lr decay to 0, tagged checkpoints every 25k) started on the Mac 12:08 UTC.
- AWS: quota raised to 32 (approved ~07:55 UTC). g6.xlarge (L4) i-02c5554f83e734482 launched 08:05 UTC. GF single process 0.033–0.042 s/step (3–4x the Mac); but 4 GF + 1 MonoGrow processes on the 4-vCPU box fall to 0.5–0.7 s/step each (CPU-bound + GPU context thrash), so concurrency on one box is a net loss. Decision 08:35 UTC: one process per GPU, campaign sharded over 3 more g6.xlarge (~$3.2/h total, ~31 h, ~$80 for 170 runs); the first box is dedicated to the MonoGrow prior.
- Fleet (08:37–09:02 UTC): gf1 i-0c415c17d7b3d6bb2, gf2 i-00175be70cdbf8c13, gf3 i-0bcc5b885b6b1001b (shards 0–2 of 170 jobs, P=1, 0.045–0.065 s/step), monogrow i-02c5554f83e734482 (prior alone: 0.085 s/step, ~3.5 h for 150k). ETA GF campaign ≈ 2026-10-31 20:00–2026-11-01 03:00 UTC, ≈ $90. One extra box launched by mistake and terminated after 12 min (~$0.16).
