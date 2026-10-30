# shared_continuous: results
Generated 2026-10-30 21:07. Experiment dir: `runs_aws/merged/shared_continuous/results/fullplant_batch_shared_continuous_test_aligned_nearest`.
Sequences present: 8 of 8 test plants. Integrity: full-plant n_tips sum = 192 (192 expected for 8 plants, 1 run), per-organ = 189 (189 expected).

## Full-plant (Table 1 aggregation, cd_test over all frames t>=1)
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips |
|---|---|---|---|---|---|
| MaC2 | 5 run(s) | 1.08 | 1.58 | 2.51 / 2.38 | 31 |
| MaC3 | 5 run(s) | 0.67 | 1.58 | 1.43 / 2.01 | 10 |
| SoC2 | 5 run(s) | 0.44 | 1.05 | 0.69 / 1.34 | 18 |
| SoH2 | 5 run(s) | 0.69 | 0.92 | 0.74 / 0.97 | 11 |
| TbC1 | 5 run(s) | 0.19 | 1.10 | 0.59 / 1.66 | 15 |
| TbS3 | 5 run(s) | 0.14 | 0.66 | 0.45 / 1.30 | 16 |
| Tm1H3 | 5 run(s) | 0.09 | 0.45 | 0.12 / 0.80 | 36 |
| Tm1S1 | 5 run(s) | 0.20 | 0.79 | 0.32 / 1.34 | 55 |
| **Mean (8 plants)** | | **0.44** | **1.02** | 0.86 / 1.47 | 192 |

Training time per plant (s, first run): MaC2 1610, MaC3 1401, SoC2 1461, SoH2 1223, TbC1 1605, TbS3 1728, Tm1H3 2720, Tm1S1 2698

## Per-organ (Table 2 aggregation, organs with >= 2 scored frames) + growth metrics
Growth metrics from `growth_metrics.py`: organ length L = 99th-percentile axial (z) extent per exported frame in the canonical organ frame; violation = L(t+1) < L(t) - 2 mm over consecutive exported frames; GT floor in parentheses; length MAE = mean |L_pred - L_gt| (mm).
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips | viol. rate (GT floor) | neg. mass mm | length MAE mm |
|---|---|---|---|---|---|---|---|---|
| MaC2 | 5 organs | 1.11 | 1.65 | 2.26 / 2.59 | 30 | 0.152 (0.182) | 3.83 | 1.18 |
| MaC3 | 4 organs | 0.79 | 1.60 | 1.48 / 2.46 | 9 | 0.019 (0.048) | 0.94 | 0.81 |
| SoC2 | 5 organs | 0.59 | 1.05 | 0.93 / 1.51 | 18 | 0.078 (0.043) | 3.71 | 0.88 |
| SoH2 | 5 organs | 0.57 | 0.74 | 0.64 / 0.85 | 10 | 0.167 (0.167) | 1.95 | 2.10 |
| TbC1 | 6 organs | 0.23 | 1.24 | 0.49 / 2.04 | 15 | 0.037 (0.037) | 0.98 | 0.18 |
| TbS3 | 7 organs | 0.19 | 0.75 | 0.45 / 1.20 | 16 | 0.000 (0.000) | 0.37 | 0.25 |
| Tm1H3 | 4 organs | 0.11 | 0.45 | 0.20 / 0.98 | 36 | 0.000 (0.020) | 0.98 | 0.15 |
| Tm1S1 | 9 organs | 0.28 | 0.72 | 0.46 / 1.33 | 55 | 0.000 (0.000) | 0.20 | 0.33 |
| **Mean (8 plants)** | | **0.48** | **1.03** | 0.87 / 1.62 | 189 | 0.057 (0.062) | 1.62 | 0.73 |

## GrowFields' own LaTeX table rows (`create_latex_table_*.py --per_sequence`; full plant, then per organ)
```
shared_continuous & 1.08\std{0.07} & 1.58\std{0.10} & 0.67\std{0.03} & 1.58\std{0.30} & 0.44\std{0.00} & 1.05\std{0.05} & 0.69\std{0.03} & 0.92\std{0.06} & 0.19\std{0.00} & 1.10\std{0.06} & 0.14\std{0.00} & 0.66\std{0.07} & 0.09\std{0.00} & 0.45\std{0.03} & 0.20\std{0.01} & 0.79\std{0.04} & 0.44\std{0.01} & 1.02\std{0.04} \\
  Removed 4 organ sequences with insufficient timepoints:
    maize_control_plant2, organ_id=5: 1-1 timepoints (avg: 1.0, 5 runs)
    maize_control_plant3, organ_id=4: 1-1 timepoints (avg: 1.0, 5 runs)
    sorghum_highlight_plant2, organ_id=5: 1-1 timepoints (avg: 1.0, 5 runs)
    tobacco_control_plant1, organ_id=6: 1-1 timepoints (avg: 1.0, 5 runs)
shared_continuous & 1.11\std{0.07} & 1.65\std{0.11} & 0.79\std{0.03} & 1.60\std{0.34} & 0.59\std{0.01} & 1.05\std{0.11} & 0.57\std{0.00} & 0.74\std{0.05} & 0.23\std{0.01} & 1.24\std{0.06} & 0.19\std{0.00} & 0.75\std{0.16} & 0.11\std{0.00} & 0.45\std{0.03} & 0.28\std{0.01} & 0.72\std{0.03} & 0.48\std{0.01} & 1.03\std{0.06} \\
```

## growth_metrics.py output
```
viol_rate  viol_rate_gt  neg_mass  len_mae  n_pairs  n_organs
sequence                                                                               
maize_control_plant2         0.1515        0.1818    3.8322   1.1769     33.0     6.000
maize_control_plant3         0.0190        0.0476    0.9381   0.8050     21.0     5.000
sorghum_control_plant2       0.0783        0.0435    3.7050   0.8818     23.0     5.000
sorghum_highlight_plant2     0.1667        0.1667    1.9465   2.0995     12.0     6.000
tobacco_control_plant1       0.0370        0.0370    0.9849   0.1844     27.0     7.000
tobacco_shade_plant3         0.0000        0.0000    0.3686   0.2493     34.0     7.000
tomato1_heat_plant3          0.0000        0.0196    0.9788   0.1490     51.0     4.000
tomato1_shade_plant1         0.0000        0.0000    0.2026   0.3259     75.0     9.000
Mean                         0.0566        0.0620    1.6196   0.7340     34.5     6.125

rows: 1625 -> runs_aws/merged/shared_continuous/results/fullplant_batch_shared_continuous_test_aligned_nearest/growth_metrics.csv
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
