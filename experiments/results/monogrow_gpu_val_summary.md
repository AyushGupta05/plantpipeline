# MonoGrow validation grid, 4 GrowFields val plants, fit.source=nearest (priors: runs/<prior>/ckpt.pt, one section per prior and variant)

Common: n_z 300, lr_z 3e-3, lambda_prior 1e-3, 1024 pts/organ, seed 0. CD mm^2 / EPE mm. 'held-out' = the variant's held-out frames
(interpolation: odd frames; extrapolation: last 3), 'obs' = observed frames. Violation rate from growth_metrics --mode all
(GT floor in brackets), length MAE on held-out frames from --mode <variant>. GPU s/step = fit seconds / (n_z + n_theta), summed over the 4 plants.

Baselines to beat: `nt0` (z only). The same z-only nearest recipe on the Mac's CPU prior gave val extrapolation held-out 131.5 / 12.6
(maize 216, sorghum 262, tobacco 36, tomato2 11.6) at step ~95k; on the 50k-step prior 180.3 / 12.1 (README, Rollout source).
lr_theta 1e-4 is not in the grid: it diverges (Mac sweep, val extrapolation CD 2.7e5; same instability as prior training at 1e-4).

## prior `full_cpu95k`, interpolation

| rank | config | held-out full CD / EPE | held-out organ CD / EPE | obs full CD / EPE | obs organ CD / EPE | n_tips held-out (full/organ) | viol. rate (GT) | neg. mass mm (GT) | len MAE held-out | len MAE obs | MAE +1/+2/+3 | fit s (4 plants) | GPU s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nt0 | 55.36 / 9.06 | 54.00 / 11.35 | 134.30 / 18.01 | 133.85 / 20.94 | 41 / 39 | 0.000 (0.045) | 0.0 (46.0) | 10.0 | 17.4 | - | 134 | 0.111 |
| 2 | nt1000_lt1e-5 | 68.04 / 10.63 | 67.76 / 13.08 | 161.22 / 18.39 | 153.30 / 19.86 | 41 / 39 | 0.000 (0.045) | 0.0 (46.0) | 14.1 | 19.8 | - | 586 | 0.113 |
| 3 | nt1000_lt2e-5 | 68.20 / 10.37 | 79.78 / 12.63 | 135.72 / 21.51 | 174.01 / 24.47 | 41 / 39 | 0.000 (0.045) | 0.0 (46.0) | 14.3 | 21.0 | - | 579 | 0.111 |
| 4 | nt2000_lt2e-5 | 77.26 / 9.52 | 72.84 / 12.51 | 170.20 / 19.96 | 176.07 / 22.55 | 41 / 39 | 0.000 (0.045) | 0.0 (46.0) | 12.6 | 18.5 | - | 1009 | 0.110 |
| 5 | nt2000_lt1e-5 | 92.95 / 10.76 | 108.52 / 13.25 | 204.29 / 20.35 | 224.78 / 21.67 | 41 / 39 | 0.000 (0.045) | 0.0 (46.0) | 12.9 | 19.4 | - | 1027 | 0.112 |
| 6 | nt5000_lt1e-5 | 98.17 / 11.25 | 89.71 / 13.83 | 242.08 / 20.41 | 241.30 / 21.76 | 41 / 39 | 0.000 (0.045) | 0.0 (46.0) | 10.5 | 17.2 | - | 2358 | 0.111 |
| 7 | nt5000_lt2e-5 | 339.44 / 16.99 | 408.06 / 21.96 | 890.70 / 34.77 | 1145.86 / 34.78 | 41 / 39 | 0.008 (0.045) | 7.8 (46.0) | 33.1 | 43.3 | - | 2309 | 0.109 |

Per plant, held-out full CD / EPE (organ CD / EPE):

| config | maize_control_plant1 | sorghum_control_plant1 | tobacco_control_plant2 | tomato2_control_plant3 |
|---|---|---|---|---|
| nt0 | 65.5 / 12.8 (63.2 / 17.8) | 122.6 / 14.7 (106.9 / 18.5) | 21.9 / 5.5 (30.9 / 5.5) | 11.4 / 3.3 (14.9 / 3.7) |
| nt1000_lt1e-5 | 66.2 / 11.7 (65.4 / 17.3) | 124.1 / 18.2 (106.7 / 21.1) | 38.4 / 6.5 (60.5 / 8.4) | 43.4 / 6.2 (38.4 / 5.4) |
| nt1000_lt2e-5 | 100.9 / 11.2 (151.1 / 16.9) | 112.0 / 16.3 (97.5 / 20.2) | 40.3 / 9.0 (48.5 / 7.9) | 19.6 / 4.9 (22.0 / 5.4) |
| nt2000_lt2e-5 | 71.1 / 11.2 (62.2 / 17.2) | 116.5 / 11.3 (93.8 / 16.9) | 84.6 / 10.2 (102.7 / 10.5) | 36.8 / 5.3 (32.7 / 5.4) |
| nt2000_lt1e-5 | 66.7 / 12.2 (72.8 / 18.0) | 142.1 / 14.4 (109.8 / 19.2) | 122.7 / 8.1 (209.7 / 8.1) | 40.2 / 8.4 (41.7 / 7.6) |
| nt5000_lt1e-5 | 68.0 / 12.5 (71.1 / 17.7) | 240.1 / 16.4 (185.8 / 20.3) | 60.8 / 10.6 (70.6 / 11.5) | 23.8 / 5.5 (31.4 / 5.7) |
| nt5000_lt2e-5 | 84.1 / 12.0 (77.9 / 17.1) | 120.9 / 15.1 (112.5 / 19.3) | 762.2 / 29.6 (1068.2 / 41.0) | 390.6 / 11.3 (373.6 / 10.4) |

## prior `full_cpu95k`, extrapolation

| rank | config | held-out full CD / EPE | held-out organ CD / EPE | obs full CD / EPE | obs organ CD / EPE | n_tips held-out (full/organ) | viol. rate (GT) | neg. mass mm (GT) | len MAE held-out | len MAE obs | MAE +1/+2/+3 | fit s (4 plants) | GPU s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nt1000_lt1e-5 | 149.76 / 14.02 | 131.08 / 16.38 | 53.26 / 10.36 | 52.14 / 11.52 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 17.2 | 7.9 | 9.9 / 19.8 / 22.0 | 383 | 0.074 |
| 2 | nt5000_lt1e-5 | 152.31 / 13.93 | 134.54 / 16.99 | 93.89 / 11.49 | 79.39 / 12.19 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 16.3 | 11.6 | 8.9 / 17.9 / 22.3 | 1561 | 0.074 |
| 3 | nt2000_lt1e-5 | 153.54 / 12.51 | 144.14 / 15.69 | 73.66 / 11.42 | 66.43 / 11.63 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 17.0 | 11.2 | 9.2 / 19.0 / 22.8 | 669 | 0.073 |
| 4 | nt0 | 160.35 / 13.10 | 136.69 / 15.49 | 54.61 / 10.15 | 50.20 / 10.48 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 17.5 | 8.6 | 10.8 / 18.7 / 23.0 | 89 | 0.074 |
| 5 | nt2000_lt2e-5 | 200.70 / 17.58 | 205.60 / 20.16 | 110.83 / 13.50 | 99.02 / 13.73 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 18.5 | 13.5 | 11.3 / 19.6 / 24.5 | 675 | 0.073 |
| 6 | nt1000_lt2e-5 | 239.24 / 18.04 | 245.36 / 20.13 | 97.40 / 13.47 | 92.31 / 13.44 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 18.2 | 11.1 | 11.0 / 19.9 / 23.8 | 388 | 0.075 |
| 7 | nt5000_lt2e-5 | 256.26 / 20.86 | 337.13 / 23.36 | 137.39 / 13.77 | 139.42 / 14.95 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 25.3 | 11.0 | 14.1 / 28.4 / 33.4 | 1562 | 0.074 |

Per plant, held-out full CD / EPE (organ CD / EPE):

| config | maize_control_plant1 | sorghum_control_plant1 | tobacco_control_plant2 | tomato2_control_plant3 |
|---|---|---|---|---|
| nt1000_lt1e-5 | 252.1 / 17.7 (205.8 / 23.8) | 297.6 / 23.3 (237.1 / 27.4) | 35.6 / 8.2 (62.0 / 8.2) | 13.7 / 6.9 (19.4 / 6.2) |
| nt5000_lt1e-5 | 303.2 / 16.5 (201.2 / 22.8) | 225.6 / 23.5 (221.5 / 28.6) | 68.6 / 9.2 (100.8 / 9.7) | 11.8 / 6.5 (14.8 / 6.8) |
| nt2000_lt1e-5 | 260.9 / 16.5 (202.0 / 23.6) | 285.6 / 18.3 (266.6 / 23.9) | 51.8 / 8.2 (86.1 / 8.2) | 15.9 / 7.1 (21.9 / 7.0) |
| nt0 | 244.3 / 17.8 (168.7 / 23.3) | 339.3 / 19.5 (274.8 / 24.6) | 34.5 / 7.4 (73.5 / 7.2) | 23.2 / 7.7 (29.8 / 6.9) |
| nt2000_lt2e-5 | 329.5 / 22.2 (240.0 / 26.7) | 326.1 / 20.9 (339.4 / 26.2) | 131.5 / 16.4 (223.3 / 18.0) | 15.8 / 10.8 (19.6 / 9.7) |
| nt1000_lt2e-5 | 337.5 / 19.3 (231.0 / 24.8) | 421.1 / 30.5 (451.4 / 33.6) | 181.2 / 16.8 (277.9 / 16.8) | 17.2 / 5.5 (21.1 / 5.3) |
| nt5000_lt2e-5 | 235.4 / 24.5 (166.1 / 29.3) | 441.2 / 28.7 (484.4 / 33.5) | 337.3 / 24.4 (683.1 / 24.9) | 11.1 / 5.9 (14.8 / 5.7) |

## prior `full_gpu`, interpolation

| rank | config | held-out full CD / EPE | held-out organ CD / EPE | obs full CD / EPE | obs organ CD / EPE | n_tips held-out (full/organ) | viol. rate (GT) | neg. mass mm (GT) | len MAE held-out | len MAE obs | MAE +1/+2/+3 | fit s (4 plants) | GPU s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nt0 | 392.24 / 20.03 | 450.75 / 20.78 | 1272.53 / 39.82 | 1614.77 / 39.36 | 41 / 39 | 0.000 (0.045) | 0.0 (46.0) | 58.6 | 84.6 | - | 137 | 0.114 |

Per plant, held-out full CD / EPE (organ CD / EPE):

| config | maize_control_plant1 | sorghum_control_plant1 | tobacco_control_plant2 | tomato2_control_plant3 |
|---|---|---|---|---|
| nt0 | 108.0 / 18.1 (124.9 / 19.8) | 463.2 / 24.0 (505.6 / 24.9) | 774.8 / 25.2 (937.1 / 25.0) | 223.0 / 12.8 (235.4 / 13.4) |

## prior `full_gpu`, extrapolation

| rank | config | held-out full CD / EPE | held-out organ CD / EPE | obs full CD / EPE | obs organ CD / EPE | n_tips held-out (full/organ) | viol. rate (GT) | neg. mass mm (GT) | len MAE held-out | len MAE obs | MAE +1/+2/+3 | fit s (4 plants) | GPU s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nt0 | 1191.20 / 40.38 | 1895.77 / 38.82 | 394.32 / 24.68 | 475.05 / 24.34 | 36 / 36 | 0.000 (0.047) | 0.0 (38.2) | 86.3 | 56.8 | 56.2 / 87.8 / 114.9 | 89 | 0.074 |

Per plant, held-out full CD / EPE (organ CD / EPE):

| config | maize_control_plant1 | sorghum_control_plant1 | tobacco_control_plant2 | tomato2_control_plant3 |
|---|---|---|---|---|
| nt0 | 265.1 / 38.9 (426.0 / 39.3) | 1289.0 / 65.5 (2354.4 / 58.7) | 2593.8 / 32.1 (3943.4 / 32.9) | 616.9 / 25.0 (859.4 / 24.3) |
