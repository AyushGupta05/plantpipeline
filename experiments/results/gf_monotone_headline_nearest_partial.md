# monotone_continuous: results
Generated 2026-10-30 21:07. Experiment dir: `runs_aws/merged/monotone_continuous/results/fullplant_batch_monotone_continuous_test_aligned_nearest`.
Sequences present: 6 of 8 test plants. Integrity: full-plant n_tips sum = 166 (192 expected for 8 plants, 1 run), per-organ = 164 (189 expected).

## Full-plant (Table 1 aggregation, cd_test over all frames t>=1)
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips |
|---|---|---|---|---|---|
| MaC2 | 5 run(s) | 1.09 | 1.72 | 2.51 / 2.38 | 31 |
| MaC3 | 5 run(s) | 0.69 | 1.70 | 1.43 / 2.01 | 10 |
| SoC2 | 4 run(s) | 0.81 | 1.54 | 0.69 / 1.34 | 18 |
| TbS3 | 1 run(s) | 0.15 | 0.67 | 0.45 / 1.30 | 16 |
| Tm1H3 | 5 run(s) | 0.10 | 0.63 | 0.12 / 0.80 | 36 |
| Tm1S1 | 5 run(s) | 0.21 | 0.73 | 0.32 / 1.34 | 55 |
| **Mean (6 plants)** | | **0.51** | **1.16** | 0.86 / 1.47 | 166 |

Training time per plant (s, first run): MaC2 1779, MaC3 1489, SoC2 1554, TbS3 1959, Tm1H3 2794, Tm1S1 3123

## Per-organ (Table 2 aggregation, organs with >= 2 scored frames) + growth metrics
Growth metrics from `growth_metrics.py`: organ length L = 99th-percentile axial (z) extent per exported frame in the canonical organ frame; violation = L(t+1) < L(t) - 2 mm over consecutive exported frames; GT floor in parentheses; length MAE = mean |L_pred - L_gt| (mm).
| plant | frames/organs scored | CD test (mm²) | EPE (mm) | paper CD / EPE | n_tips | viol. rate (GT floor) | neg. mass mm | length MAE mm |
|---|---|---|---|---|---|---|---|---|
| MaC2 | 5 organs | 1.21 | 1.78 | 2.26 / 2.59 | 30 | 0.061 (0.182) | 1.40 | 1.71 |
| MaC3 | 4 organs | 0.81 | 1.79 | 1.48 / 2.46 | 9 | 0.000 (0.048) | 0.39 | 1.32 |
| SoC2 | 5 organs | 0.80 | 1.49 | 0.93 / 1.51 | 18 | 0.043 (0.043) | 2.51 | 1.61 |
| TbS3 | 7 organs | 0.21 | 0.73 | 0.45 / 1.20 | 16 | 0.000 (0.000) | 0.05 | 0.24 |
| Tm1H3 | 4 organs | 0.11 | 0.63 | 0.20 / 0.98 | 36 | 0.000 (0.020) | 0.84 | 0.29 |
| Tm1S1 | 9 organs | 0.29 | 0.66 | 0.46 / 1.33 | 55 | 0.000 (0.000) | 0.39 | 0.33 |
| **Mean (6 plants)** | | **0.57** | **1.18** | 0.87 / 1.62 | 164 | 0.017 (0.049) | 0.93 | 0.92 |

## GrowFields' own LaTeX table rows (`create_latex_table_*.py --per_sequence`; full plant, then per organ)
```
monotone_continuous & 1.09\std{0.10} & 1.72\std{0.16} & 0.69\std{0.04} & 1.70\std{0.14} & 0.81\std{0.00} & 1.54\std{0.04} & 0.15 & 0.67 & 0.10\std{0.00} & 0.63\std{0.02} & 0.21\std{0.01} & 0.73\std{0.04} & 0.51\std{0.04} & 1.16\std{0.04} \\
  Removed 2 organ sequences with insufficient timepoints:
    maize_control_plant2, organ_id=5: 1-1 timepoints (avg: 1.0, 5 runs)
    maize_control_plant3, organ_id=4: 1-1 timepoints (avg: 1.0, 5 runs)
monotone_continuous & 1.21\std{0.10} & 1.78\std{0.17} & 0.81\std{0.05} & 1.79\std{0.10} & 0.80\std{0.01} & 1.49\std{0.05} & 0.21 & 0.73 & 0.11\std{0.00} & 0.63\std{0.02} & 0.29\std{0.01} & 0.66\std{0.03} & 0.57\std{0.04} & 1.18\std{0.04} \\
```

## growth_metrics.py output
```
viol_rate  viol_rate_gt  neg_mass  len_mae  n_pairs  n_organs
sequence                                                                             
maize_control_plant2       0.0606        0.1818    1.3984   1.7114     33.0       6.0
maize_control_plant3       0.0000        0.0476    0.3873   1.3170     21.0       5.0
sorghum_control_plant2     0.0435        0.0435    2.5057   1.6078     23.0       5.0
tobacco_shade_plant3       0.0000        0.0000    0.0496   0.2372     34.0       7.0
tomato1_heat_plant3        0.0000        0.0196    0.8379   0.2946     51.0       4.0
tomato1_shade_plant1       0.0000        0.0000    0.3949   0.3260     75.0       9.0
Mean                       0.0173        0.0488    0.9290   0.9156     39.5       6.0

rows: 1173 -> runs_aws/merged/monotone_continuous/results/fullplant_batch_monotone_continuous_test_aligned_nearest/growth_metrics.csv
```

## Logs
- `runs/monotone_continuous/jobs.log`
- `runs/monotone_continuous/maize_control_plant2_seed43.log`
- `runs/monotone_continuous/maize_control_plant3_seed43.log`
- `runs/monotone_continuous/tomato1_heat_plant3_seed43.log`
- `runs/monotone_continuous/tomato1_shade_plant1_seed43.log`
