# MonoGrow test split (8 GrowFields test plants), prior `runs/full_cpu95k` (95k-step CPU prior; the 150k GPU prior is degraded, runs/gpu_val/summary.md) — gpu_test_obj

Recipe (chosen on the val plants, runs/gpu_val/summary.md): fit.objective=nearest; per-variant recipes in runs/gpu_val_obj/decision.md: headline fit.n_z=300 fit.lr_z=3e-3 fit.lambda_prior=1e-3 fit.n_theta=1000 fit.lr_theta=1e-5;interpolation fit.n_z=300 fit.lr_z=3e-3 fit.lambda_prior=1e-3 fit.n_theta=1000 fit.lr_theta=1e-5;extrapolation fit.n_z=1000 fit.lr_z=3e-3 fit.lambda_prior=1e-3 fit.n_theta=0;. Mean ± std over seeds (fit.seed = point-sampling seed; z starts at
mu_species, so the std is sampling noise only). CD mm^2 / EPE mm. `nearest` = each frame predicted from the latest observed frame
(one-day steps; extrapolation: +1/+2/+3 d from the last observed frame); `first` = GrowFields' protocol (rollout from the organ's first observed frame).
Both sources share the same fitted codes / fine-tuned theta (fit.reuse). Headline = all frames observed, scored on frames >= 1.

## Summary

| variant | source | seeds | frames | full-plant CD / EPE | per-organ CD / EPE | n_tips full / organ | viol. rate (GT) | neg. growth mass mm (GT) | len MAE mm obs / held-out | MAE +1/+2/+3 d | fit s (8 plants) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| extrapolation | first | 3 | observed | 122.96 ± 11.46 / 17.12 ± 1.37 | 122.15 ± 8.88 / 21.81 ± 1.61 | 102 / 98 | 0.000 ± 0.000 (0.043) | 0.0 ± 0.0 (86.5) | 14.5 ± 0.2 / 33.9 ± 0.1 | 27.5 ± 0.4 / 35.0 ± 0.2 / 39.2 ± 0.5 | 741 ± 3 |
| extrapolation | first | 3 | **held-out** | **564.40 ± 51.32 / 37.34 ± 1.19** | 435.70 ± 46.78 / 35.97 ± 1.71 | 71 / 71 | | | | | |
| extrapolation | nearest | 3 | observed | 32.00 ± 0.85 / 9.62 ± 0.20 | 26.62 ± 1.51 / 12.53 ± 0.11 | 102 / 98 | 0.000 ± 0.000 (0.040) | 0.0 ± 0.0 (81.6) | 3.9 ± 0.2 / 15.3 ± 0.2 | 7.5 ± 0.3 / 16.7 ± 0.4 / 21.9 ± 0.3 | 741 ± 3 |
| extrapolation | nearest | 3 | **held-out** | **193.40 ± 6.01 / 20.50 ± 0.55** | 148.32 ± 8.43 / 18.03 ± 0.39 | 71 / 71 | | | | | |
| headline | first | 3 | all | 118.89 ± 12.29 / 23.17 ± 1.68 | 108.97 ± 13.56 / 29.40 ± 1.87 | 192 / 189 | 0.000 ± 0.000 (0.040) | 0.5 ± 0.1 (86.5) | 11.3 ± 0.4 / - | - | 1773 ± 17 |
| headline | nearest | 3 | all | 13.94 ± 2.90 / 10.70 ± 0.33 | 12.99 ± 1.69 / 12.99 ± 0.24 | 192 / 189 | 0.000 ± 0.000 (0.040) | 0.0 ± 0.0 (86.5) | 2.4 ± 0.1 / - | - | 1773 ± 17 |
| interpolation | first | 3 | observed | 157.23 ± 5.05 / 27.27 ± 2.36 | 161.30 ± 2.77 / 25.17 ± 2.79 | 85 / 79 | 0.000 ± 0.000 (0.042) | 0.0 ± 0.0 (86.5) | 14.9 ± 0.1 / 14.4 ± 0.2 | - | 1288 ± 9 |
| interpolation | first | 3 | **held-out** | **142.85 ± 4.52 / 23.55 ± 1.39** | 143.34 ± 1.62 / 27.60 ± 1.47 | 93 / 89 | | | | | |
| interpolation | nearest | 3 | observed | 73.47 ± 5.60 / 19.61 ± 1.00 | 72.96 ± 3.37 / 15.18 ± 0.98 | 87 / 80 | 0.000 ± 0.000 (0.039) | 0.1 ± 0.1 (78.6) | 7.8 ± 0.2 / 5.9 ± 0.5 | - | 1288 ± 9 |
| interpolation | nearest | 3 | **held-out** | **41.96 ± 3.37 / 10.63 ± 0.38** | 43.18 ± 2.82 / 11.11 ± 0.24 | 93 / 89 | | | | | |

Reference (test split): GrowFields paper (BENCHMARK.md section 2; `first` protocol, 50k steps per plant, 5/3 seeds): headline 0.86 ± 0.02 / 1.47 ± 0.11 full-plant,
0.87 / 1.62 per-organ; interpolation held-out 10.29 / 5.82 full-plant (11.42 / 4.59 per-organ); extrapolation held-out (7 plants, no SoH2)
117.71 ± 35.55 / 21.79 ± 0.54 full-plant, 149.63 ± 19.65 / 18.99 ± 0.68 per-organ. RichardsNet (experiments/richards/README.md, template = last
observed frame, deterministic): headline 23.02 / 5.84, interpolation held-out 20.79 / 7.14, extrapolation held-out 141.48 / 16.36 (8 plants) /
157.66 / 17.85 (GF's 7); with template = first: 112.09 / 8.51, 99.70 / 8.69, 238.18 / 17.93; with template = nearest (same protocol as our
`nearest`, full-plant / per-organ): headline 22.08 / 4.61, 17.83 / 5.50; interpolation held-out 25.08 / 5.48, 22.16 / 6.63; extrapolation held-out
141.48 / 13.91, 99.55 / 11.95 (8 plants).

## Per plant (full-plant CD / EPE, mean over seeds; held-out frames for interpolation / extrapolation)

| variant, source | maize_control_plant2 | maize_control_plant3 | sorghum_control_plant2 | sorghum_highlight_plant2 | tobacco_control_plant1 | tobacco_shade_plant3 | tomato1_heat_plant3 | tomato1_shade_plant1 | Mean |
|---|---|---|---|---|---|---|---|---|---|
| extrapolation, first | 2401.49 / 87.98 | 870.80 / 105.52 | 542.17 / 47.77 | 70.52 / 11.38 | 125.96 / 9.31 | 259.05 / 15.49 | 156.19 / 13.20 | 89.00 / 8.05 | 564.40 / 37.34 |
| extrapolation, nearest | 267.70 / 21.07 | 907.57 / 84.75 | 175.70 / 17.41 | 70.75 / 10.91 | 46.27 / 10.10 | 40.19 / 10.19 | 19.79 / 4.55 | 19.20 / 4.97 | 193.40 / 20.50 |
| headline, first | 375.64 / 45.08 | 256.81 / 65.03 | 200.94 / 37.15 | 88.21 / 19.76 | 7.42 / 5.74 | 12.34 / 5.04 | 2.12 / 2.83 | 7.66 / 4.72 | 118.89 / 23.17 |
| headline, nearest | 4.58 / 13.04 | 21.96 / 29.96 | 40.40 / 17.25 | 41.84 / 15.01 | 1.25 / 4.40 | 0.98 / 2.87 | 0.16 / 1.32 | 0.35 / 1.73 | 13.94 / 10.70 |
| interpolation, first | 295.61 / 48.47 | 152.27 / 64.99 | 358.75 / 34.19 | 108.20 / 13.14 | 79.19 / 9.42 | 126.17 / 9.19 | 8.49 / 3.88 | 14.09 / 5.16 | 142.85 / 23.55 |
| interpolation, nearest | 68.43 / 17.01 | 45.92 / 23.87 | 88.29 / 11.80 | 81.59 / 12.21 | 23.00 / 7.98 | 19.17 / 6.18 | 3.62 / 2.62 | 5.65 / 3.37 | 41.96 / 10.63 |

GF's extrapolation table excludes 5-frame sorghum_highlight_plant2; the extrapolation mean over the other 7 plants (GF: full-plant 117.71 / 21.79, per-organ 149.63 / 18.99):

* extrapolation, first, full: 634.95 ± 58.38 / 41.05 ± 1.50
* extrapolation, first, organ: 482.71 ± 52.90 / 39.60 ± 2.27
* extrapolation, nearest, full: 210.92 ± 7.09 / 21.87 ± 0.68
* extrapolation, nearest, organ: 156.46 ± 10.10 / 19.25 ± 0.53
