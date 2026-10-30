# MonoGrow test split (8 GrowFields test plants), prior `runs/full_cpu95k` (95k-step CPU prior; the 150k GPU prior is degraded, runs/gpu_val/summary.md) — gpu_test

Recipe (chosen on the val plants, runs/gpu_val/summary.md): headline/interpolation: z only (n_z 300, lr_z 3e-3, lambda_prior 1e-3); extrapolation: + n_theta 1000, lr_theta 1e-5. Mean ± std over seeds (fit.seed = point-sampling seed; z starts at
mu_species, so the std is sampling noise only). CD mm^2 / EPE mm. `nearest` = each frame predicted from the latest observed frame
(one-day steps; extrapolation: +1/+2/+3 d from the last observed frame); `first` = GrowFields' protocol (rollout from the organ's first observed frame).
Both sources share the same fitted codes / fine-tuned theta (fit.reuse). Headline = all frames observed, scored on frames >= 1.

## Summary

| variant | source | seeds | frames | full-plant CD / EPE | per-organ CD / EPE | n_tips full / organ | viol. rate (GT) | neg. growth mass mm (GT) | len MAE mm obs / held-out | MAE +1/+2/+3 d | fit s (8 plants) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| extrapolation | first | 3 | observed | 163.29 ± 17.99 / 19.57 ± 1.62 | 183.40 ± 12.29 / 24.64 ± 1.86 | 102 / 98 | 0.000 ± 0.000 (0.043) | 0.3 ± 0.4 (86.5) | 21.9 ± 2.1 / 37.7 ± 1.2 | 31.5 ± 0.7 / 38.4 ± 1.8 / 43.1 ± 1.2 | 943 ± 24 |
| extrapolation | first | 3 | **held-out** | **579.67 ± 54.66 / 40.81 ± 0.69** | 498.04 ± 45.51 / 39.84 ± 0.68 | 71 / 71 | | | | | |
| extrapolation | nearest | 3 | observed | 51.94 ± 2.13 / 10.98 ± 0.97 | 65.83 ± 5.97 / 14.37 ± 0.80 | 102 / 98 | 0.000 ± 0.000 (0.040) | 0.2 ± 0.2 (81.6) | 10.0 ± 1.0 / 16.5 ± 0.9 | 8.3 ± 0.7 / 17.7 ± 1.2 / 23.5 ± 0.9 | 943 ± 24 |
| extrapolation | nearest | 3 | **held-out** | **177.81 ± 36.97 / 21.33 ± 0.50** | 147.73 ± 24.77 / 19.16 ± 0.29 | 71 / 71 | | | | | |
| headline | first | 3 | all | 385.48 ± 8.77 / 27.79 ± 0.54 | 431.44 ± 18.87 / 35.13 ± 0.71 | 192 / 189 | 0.000 ± 0.000 (0.040) | 0.4 ± 0.3 (86.5) | 26.8 ± 1.0 / - | - | 388 ± 2 |
| headline | nearest | 3 | all | 58.91 ± 0.41 / 12.95 ± 0.30 | 60.22 ± 0.62 / 15.21 ± 0.35 | 192 / 189 | 0.000 ± 0.000 (0.040) | 0.1 ± 0.1 (86.5) | 8.8 ± 0.3 / - | - | 388 ± 2 |
| interpolation | first | 3 | observed | 377.68 ± 7.27 / 28.59 ± 1.08 | 336.61 ± 7.86 / 26.76 ± 1.88 | 85 / 79 | 0.000 ± 0.000 (0.042) | 0.4 ± 0.5 (86.5) | 27.3 ± 1.7 / 25.5 ± 1.3 | - | 292 ± 3 |
| interpolation | first | 3 | **held-out** | **350.81 ± 9.28 / 25.02 ± 1.04** | 342.99 ± 15.93 / 29.72 ± 1.16 | 93 / 89 | | | | | |
| interpolation | nearest | 3 | observed | 147.34 ± 4.08 / 20.96 ± 0.62 | 109.72 ± 2.89 / 16.89 ± 1.11 | 87 / 80 | 0.000 ± 0.000 (0.039) | 0.0 ± 0.1 (78.6) | 14.9 ± 0.5 / 8.8 ± 0.6 | - | 292 ± 3 |
| interpolation | nearest | 3 | **held-out** | **59.06 ± 0.84 / 10.87 ± 0.08** | 52.89 ± 1.56 / 11.53 ± 0.06 | 93 / 89 | | | | | |

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
| extrapolation, first | 1608.96 / 86.15 | 987.73 / 112.11 | 735.89 / 53.15 | 78.60 / 13.05 | 176.87 / 14.69 | 472.35 / 16.22 | 345.96 / 17.92 | 231.01 / 13.17 | 579.67 / 40.81 |
| extrapolation, nearest | 351.58 / 21.43 | 623.84 / 81.81 | 196.11 / 19.59 | 78.67 / 13.99 | 51.30 / 11.66 | 61.41 / 9.74 | 30.84 / 6.19 | 28.71 / 6.27 | 177.81 / 21.33 |
| headline, first | 1271.87 / 53.04 | 574.96 / 73.43 | 574.81 / 37.83 | 135.01 / 17.98 | 118.04 / 10.41 | 231.25 / 8.91 | 109.52 / 14.32 | 68.37 / 6.42 | 385.48 / 27.79 |
| headline, nearest | 97.24 / 16.39 | 81.88 / 30.82 | 127.46 / 18.72 | 74.91 / 14.46 | 25.84 / 7.14 | 29.48 / 6.99 | 16.58 / 4.94 | 17.87 / 4.13 | 58.91 / 12.95 |
| interpolation, first | 1151.13 / 50.06 | 524.14 / 64.06 | 536.83 / 34.88 | 102.25 / 14.66 | 102.13 / 8.08 | 216.14 / 14.19 | 111.34 / 8.49 | 62.50 / 5.72 | 350.81 / 25.02 |
| interpolation, nearest | 139.43 / 18.63 | 55.33 / 23.63 | 114.97 / 12.03 | 75.12 / 12.03 | 26.55 / 5.81 | 29.61 / 6.68 | 14.28 / 4.04 | 17.19 / 4.12 | 59.06 / 10.87 |

GF's extrapolation table excludes 5-frame sorghum_highlight_plant2; the extrapolation mean over the other 7 plants (GF: full-plant 117.71 / 21.79, per-organ 149.63 / 18.99):

* extrapolation, first, full: 651.25 ± 62.28 / 44.77 ± 0.74
* extrapolation, first, organ: 551.57 ± 51.20 / 43.90 ± 0.70
* extrapolation, nearest, full: 191.97 ± 42.96 / 22.38 ± 0.61
* extrapolation, nearest, organ: 151.67 ± 29.47 / 20.12 ± 0.37
=== GF compute_metrics_fullplant.py  on /home/ubuntu/plantpipeline/experiments/monogrow/runs/gpu_test/headline_nearest_seed0
Runtime and Storage:
  Mean training time: 54.0s (0.9 min)
  Mean model size: 1.30 MB
=== GF compute_metrics_fullplant.py --interpolation on /home/ubuntu/plantpipeline/experiments/monogrow/runs/gpu_test/interpolation_nearest_seed0
Runtime and Storage:
  Mean training time: 40.4s (0.7 min)
  Mean model size: 1.30 MB
=== GF compute_metrics_fullplant.py --extrapolation on /home/ubuntu/plantpipeline/experiments/monogrow/runs/gpu_test/extrapolation_nearest_seed0
Runtime and Storage:
  Mean training time: 121.3s (2.0 min)
  Mean model size: 1.30 MB
