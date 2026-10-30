# monotone_continuous: results
Generated 2026-10-30 20:59. Experiment dir: `runs_aws/merged/monotone_continuous/results/fullplant_batch_monotone_continuous_test_aligned`.
Sequences present: 6 of 8 test plants. Integrity: full-plant n_tips sum = 166 (192 expected for 8 plants, 1 run), per-organ = 164 (189 expected).

## Full-plant (Table 1 aggregation, cd_test over all frames t>=1)
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips |
|---|---|---|---|---|---|
| MaC2 | 5 run(s) | 2.48 | 2.36 | 2.51 / 2.38 | 31 |
| MaC3 | 5 run(s) | 1.54 | 2.55 | 1.43 / 2.01 | 10 |
| SoC2 | 4 run(s) | 0.85 | 1.99 | 0.69 / 1.34 | 18 |
| TbS3 | 1 run(s) | 0.48 | 1.17 | 0.45 / 1.30 | 16 |
| Tm1H3 | 5 run(s) | 0.13 | 1.55 | 0.12 / 0.80 | 36 |
| Tm1S1 | 5 run(s) | 0.33 | 1.22 | 0.32 / 1.34 | 55 |
| **Mean (6 plants)** | | **0.97** | **1.81** | 0.86 / 1.47 | 166 |

Training time per plant (s, first run): MaC2 1779, MaC3 1489, SoC2 1554, TbS3 1959, Tm1H3 2794, Tm1S1 3123

## Per-organ (Table 2 aggregation, organs with >= 2 scored frames) + growth metrics
Growth metrics from `growth_metrics.py`: organ length L = 99th-percentile axial (z) extent per exported frame in the canonical organ frame; violation = L(t+1) < L(t) - 2 mm over consecutive exported frames; GT floor in parentheses; length MAE = mean |L_pred - L_gt| (mm).
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips | viol. rate (GT floor) | neg. mass mm | length MAE mm |
|---|---|---|---|---|---|---|---|---|
| MaC2 | 5 organs | 2.28 | 2.57 | 2.26 / 2.59 | 30 | 0.000 (0.182) | 0.00 | 2.00 |
| MaC3 | 4 organs | 1.60 | 2.97 | 1.48 / 2.46 | 9 | 0.000 (0.048) | 0.00 | 1.55 |
| SoC2 | 5 organs | 1.04 | 2.02 | 0.93 / 1.51 | 18 | 0.000 (0.043) | 0.00 | 1.86 |
| TbS3 | 7 organs | 0.47 | 1.01 | 0.45 / 1.20 | 16 | 0.000 (0.000) | 0.00 | 0.43 |
| Tm1H3 | 4 organs | 0.22 | 2.18 | 0.20 / 0.98 | 36 | 0.000 (0.020) | 0.00 | 0.27 |
| Tm1S1 | 9 organs | 0.47 | 1.28 | 0.46 / 1.33 | 55 | 0.000 (0.000) | 0.00 | 0.39 |
| **Mean (6 plants)** | | **1.02** | **2.01** | 0.87 / 1.62 | 164 | 0.000 (0.049) | 0.00 | 1.08 |

## GrowFields' own LaTeX table rows (`create_latex_table_*.py --per_sequence`; full plant, then per organ)
```
monotone_continuous & 2.48\std{0.18} & 2.36\std{0.22} & 1.54\std{0.10} & 2.55\std{0.37} & 0.85\std{0.02} & 1.99\std{0.10} & 0.48 & 1.17 & 0.13\std{0.01} & 1.55\std{0.10} & 0.33\std{0.01} & 1.22\std{0.11} & 0.97\std{0.08} & 1.81\std{0.06} \\
  Removed 2 organ sequences with insufficient timepoints:
    maize_control_plant2, organ_id=5: 1-1 timepoints (avg: 1.0, 5 runs)
    maize_control_plant3, organ_id=4: 1-1 timepoints (avg: 1.0, 5 runs)
monotone_continuous & 2.28\std{0.23} & 2.57\std{0.24} & 1.60\std{0.13} & 2.97\std{0.38} & 1.04\std{0.03} & 2.02\std{0.18} & 0.47 & 1.01 & 0.22\std{0.03} & 2.18\std{0.20} & 0.47\std{0.02} & 1.28\std{0.09} & 1.02\std{0.08} & 2.01\std{0.14} \\
```

## growth_metrics.py output
```
viol_rate  viol_rate_gt  neg_mass  len_mae  n_pairs  n_organs
sequence                                                                             
maize_control_plant2          0.0        0.1818    0.0035   1.9954     33.0       6.0
maize_control_plant3          0.0        0.0476    0.0000   1.5461     21.0       5.0
sorghum_control_plant2        0.0        0.0435    0.0000   1.8571     23.0       5.0
tobacco_shade_plant3          0.0        0.0000    0.0016   0.4277     34.0       7.0
tomato1_heat_plant3           0.0        0.0196    0.0004   0.2683     51.0       4.0
tomato1_shade_plant1          0.0        0.0000    0.0000   0.3931     75.0       9.0
Mean                          0.0        0.0488    0.0009   1.0813     39.5       6.0

rows: 1173 -> runs_aws/merged/monotone_continuous/results/fullplant_batch_monotone_continuous_test_aligned/growth_metrics.csv
```

## Logs
- `runs/monotone_continuous/jobs.log`
- `runs/monotone_continuous/maize_control_plant2_seed43.log`
- `runs/monotone_continuous/maize_control_plant3_seed43.log`
- `runs/monotone_continuous/tomato1_heat_plant3_seed43.log`
- `runs/monotone_continuous/tomato1_shade_plant1_seed43.log`
