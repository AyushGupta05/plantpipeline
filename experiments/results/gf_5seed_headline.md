# shared_continuous: results
Generated 2026-10-30 20:59. Experiment dir: `runs_aws/merged/shared_continuous/results/fullplant_batch_shared_continuous_test_aligned`.
Sequences present: 8 of 8 test plants. Integrity: full-plant n_tips sum = 192 (192 expected for 8 plants, 1 run), per-organ = 189 (189 expected).

## Full-plant (Table 1 aggregation, cd_test over all frames t>=1)
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips |
|---|---|---|---|---|---|
| MaC2 | 5 run(s) | 2.49 | 2.38 | 2.51 / 2.38 | 31 |
| MaC3 | 5 run(s) | 1.52 | 2.37 | 1.43 / 2.01 | 10 |
| SoC2 | 5 run(s) | 0.68 | 1.33 | 0.69 / 1.34 | 18 |
| SoH2 | 5 run(s) | 0.75 | 0.99 | 0.74 / 0.97 | 11 |
| TbC1 | 5 run(s) | 0.58 | 1.62 | 0.59 / 1.66 | 15 |
| TbS3 | 5 run(s) | 0.46 | 1.17 | 0.45 / 1.30 | 16 |
| Tm1H3 | 5 run(s) | 0.12 | 0.92 | 0.12 / 0.80 | 36 |
| Tm1S1 | 5 run(s) | 0.33 | 1.50 | 0.32 / 1.34 | 55 |
| **Mean (8 plants)** | | **0.87** | **1.53** | 0.86 / 1.47 | 192 |

Training time per plant (s, first run): MaC2 1610, MaC3 1401, SoC2 1461, SoH2 1223, TbC1 1605, TbS3 1728, Tm1H3 2720, Tm1S1 2698

## Per-organ (Table 2 aggregation, organs with >= 2 scored frames) + growth metrics
Growth metrics from `growth_metrics.py`: organ length L = 99th-percentile axial (z) extent per exported frame in the canonical organ frame; violation = L(t+1) < L(t) - 2 mm over consecutive exported frames; GT floor in parentheses; length MAE = mean |L_pred - L_gt| (mm).
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips | viol. rate (GT floor) | neg. mass mm | length MAE mm |
|---|---|---|---|---|---|---|---|---|
| MaC2 | 5 organs | 2.22 | 2.57 | 2.26 / 2.59 | 30 | 0.152 (0.182) | 3.96 | 1.40 |
| MaC3 | 4 organs | 1.59 | 2.84 | 1.48 / 2.46 | 9 | 0.019 (0.048) | 0.95 | 1.23 |
| SoC2 | 5 organs | 0.90 | 1.53 | 0.93 / 1.51 | 18 | 0.052 (0.043) | 3.34 | 1.14 |
| SoH2 | 5 organs | 0.66 | 0.86 | 0.64 / 0.85 | 10 | 0.167 (0.167) | 1.71 | 2.29 |
| TbC1 | 6 organs | 0.52 | 1.81 | 0.49 / 2.04 | 15 | 0.037 (0.037) | 1.13 | 0.27 |
| TbS3 | 7 organs | 0.46 | 1.02 | 0.45 / 1.20 | 16 | 0.000 (0.000) | 0.49 | 0.37 |
| Tm1H3 | 4 organs | 0.21 | 1.17 | 0.20 / 0.98 | 36 | 0.012 (0.020) | 1.43 | 0.16 |
| Tm1S1 | 9 organs | 0.48 | 1.55 | 0.46 / 1.33 | 55 | 0.000 (0.000) | 0.27 | 0.38 |
| **Mean (8 plants)** | | **0.88** | **1.67** | 0.87 / 1.62 | 189 | 0.055 (0.062) | 1.66 | 0.90 |

## GrowFields' own LaTeX table rows (`create_latex_table_*.py --per_sequence`; full plant, then per organ)
```
shared_continuous & 2.49\std{0.18} & 2.38\std{0.25} & 1.52\std{0.07} & 2.37\std{0.45} & 0.68\std{0.02} & 1.33\std{0.11} & 0.75\std{0.03} & 0.99\std{0.07} & 0.58\std{0.02} & 1.62\std{0.40} & 0.46\std{0.03} & 1.17\std{0.30} & 0.12\std{0.01} & 0.92\std{0.31} & 0.33\std{0.02} & 1.50\std{0.17} & 0.87\std{0.02} & 1.53\std{0.06} \\
  Removed 4 organ sequences with insufficient timepoints:
    maize_control_plant2, organ_id=5: 1-1 timepoints (avg: 1.0, 5 runs)
    maize_control_plant3, organ_id=4: 1-1 timepoints (avg: 1.0, 5 runs)
    sorghum_highlight_plant2, organ_id=5: 1-1 timepoints (avg: 1.0, 5 runs)
    tobacco_control_plant1, organ_id=6: 1-1 timepoints (avg: 1.0, 5 runs)
shared_continuous & 2.22\std{0.17} & 2.57\std{0.32} & 1.59\std{0.08} & 2.84\std{0.63} & 0.90\std{0.02} & 1.53\std{0.29} & 0.66\std{0.01} & 0.86\std{0.10} & 0.52\std{0.08} & 1.81\std{0.26} & 0.46\std{0.03} & 1.02\std{0.25} & 0.21\std{0.02} & 1.17\std{0.47} & 0.48\std{0.02} & 1.55\std{0.20} & 0.88\std{0.02} & 1.67\std{0.13} \\
```

## growth_metrics.py output
```
viol_rate  viol_rate_gt  neg_mass  len_mae  n_pairs  n_organs
sequence                                                                               
maize_control_plant2         0.1515        0.1818    3.9635   1.3988     33.0     6.000
maize_control_plant3         0.0190        0.0476    0.9453   1.2269     21.0     5.000
sorghum_control_plant2       0.0522        0.0435    3.3439   1.1440     23.0     5.000
sorghum_highlight_plant2     0.1667        0.1667    1.7110   2.2928     12.0     6.000
tobacco_control_plant1       0.0370        0.0370    1.1254   0.2653     27.0     7.000
tobacco_shade_plant3         0.0000        0.0000    0.4911   0.3708     34.0     7.000
tomato1_heat_plant3          0.0118        0.0196    1.4293   0.1563     51.0     4.000
tomato1_shade_plant1         0.0000        0.0000    0.2728   0.3786     75.0     9.000
Mean                         0.0548        0.0620    1.6603   0.9042     34.5     6.125

rows: 1625 -> runs_aws/merged/shared_continuous/results/fullplant_batch_shared_continuous_test_aligned/growth_metrics.csv
```

## Logs
- `runs/shared_continuous/jobs.log`
- `runs/shared_continuous/maize_control_plant2_seed43.log`
- `runs/shared_continuous/maize_control_plant3_seed43.log`
- `runs/shared_continuous/sorghum_control_plant2_seed43.log`
- `runs/shared_continuous/sorghum_highlight_plant2_seed43.log`
- `runs/shared_continuous/tobacco_control_plant1_seed43.log`
- `runs/shared_continuous/tobacco_shade_plant3_seed43.log`
- `runs/shared_continuous/tomato1_heat_plant3_seed43.log`
- `runs/shared_continuous/tomato1_shade_plant1_seed43.log`
