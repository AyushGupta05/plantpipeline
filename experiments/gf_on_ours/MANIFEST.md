# Converted sequences (2026-11-15), 990 PLYs under `data/TrackPlant3D/versions/v2/gt_corrected_v2`

Labels: unified stem 1 -> 0, leaf k -> k-1, unknown 0 dropped. `leaves` = distinct leaf ids over the sequence. `skipped frames` = scans without stem points (PSA cannot orient their leaves), left out of the sequence. `prior-fair` = not in MonoGrow's 95k prior training set (only Pheno4D M01/M02/M04/T01/T05; every Sugar4D plant was used).

| sequence | dataset | species | split | frames | days | leaves | pts/frame | dropped pts | skipped frames (no stem) | prior-fair |
|---|---|---|---|---|---|---|---|---|---|---|
| pheno4d_maize_M01 | pheno4d | maize | train | 12 | 0-12 | 3 | 8192-8192 | 0 | - | yes |
| pheno4d_maize_M02 | pheno4d | maize | train | 12 | 0-12 | 4 | 8192-8192 | 0 | - | yes |
| pheno4d_maize_M03 | pheno4d | maize | train | 11 | 0-12 | 3 | 8192-8192 | 0 | - | no |
| pheno4d_maize_M04 | pheno4d | maize | train | 12 | 0-12 | 3 | 8192-8192 | 0 | - | yes |
| pheno4d_maize_M05 | pheno4d | maize | train | 12 | 0-12 | 3 | 8192-8192 | 0 | - | no |
| pheno4d_maize_M06 | pheno4d | maize | val | 12 | 0-12 | 2 | 8192-8192 | 0 | - | no |
| pheno4d_maize_M07 | pheno4d | maize | test | 12 | 0-12 | 3 | 8192-8192 | 0 | - | no |
| pheno4d_tomato_T01 | pheno4d | tomato | train | 20 | 0-20 | 41 | 8192-8192 | 0 | - | yes |
| pheno4d_tomato_T02 | pheno4d | tomato | train | 20 | 0-20 | 29 | 8192-8192 | 0 | - | no |
| pheno4d_tomato_T03 | pheno4d | tomato | train | 20 | 0-20 | 31 | 8192-8192 | 0 | - | no |
| pheno4d_tomato_T04 | pheno4d | tomato | train | 20 | 0-20 | 29 | 8192-8192 | 0 | - | no |
| pheno4d_tomato_T05 | pheno4d | tomato | train | 20 | 0-20 | 38 | 8192-8192 | 0 | - | yes |
| pheno4d_tomato_T06 | pheno4d | tomato | val | 20 | 0-20 | 34 | 8192-8192 | 0 | - | no |
| pheno4d_tomato_T07 | pheno4d | tomato | test | 20 | 0-20 | 26 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_001 | sugar4d | sugar_beet | val | 16 | 0-56 | 13 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_002 | sugar4d | sugar_beet | test | 16 | 0-56 | 18 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_003 | sugar4d | sugar_beet | train | 16 | 0-56 | 15 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_004 | sugar4d | sugar_beet | val | 15 | 0-56 | 10 | 8192-8192 | 0 | plant_004_das40 | no |
| sugar4d_sugar_beet_plant_005 | sugar4d | sugar_beet | train | 16 | 0-56 | 10 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_006 | sugar4d | sugar_beet | train | 16 | 0-56 | 13 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_007 | sugar4d | sugar_beet | test | 16 | 0-56 | 16 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_008 | sugar4d | sugar_beet | train | 16 | 0-56 | 16 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_009 | sugar4d | sugar_beet | train | 16 | 0-56 | 16 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_010 | sugar4d | sugar_beet | test | 16 | 0-56 | 15 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_011 | sugar4d | sugar_beet | val | 16 | 0-56 | 16 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_012 | sugar4d | sugar_beet | test | 16 | 0-56 | 13 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_013 | sugar4d | sugar_beet | train | 16 | 0-56 | 17 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_014 | sugar4d | sugar_beet | train | 16 | 0-56 | 12 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_015 | sugar4d | sugar_beet | train | 16 | 0-56 | 13 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_016 | sugar4d | sugar_beet | test | 16 | 0-56 | 14 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_017 | sugar4d | sugar_beet | train | 16 | 0-56 | 17 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_018 | sugar4d | sugar_beet | train | 16 | 0-56 | 11 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_019 | sugar4d | sugar_beet | val | 16 | 0-56 | 15 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_020 | sugar4d | sugar_beet | train | 16 | 0-56 | 18 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_021 | sugar4d | sugar_beet | train | 16 | 0-56 | 15 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_022 | sugar4d | sugar_beet | train | 16 | 0-56 | 12 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_023 | sugar4d | sugar_beet | test | 16 | 0-56 | 17 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_024 | sugar4d | sugar_beet | val | 16 | 0-56 | 10 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_025 | sugar4d | sugar_beet | val | 16 | 0-56 | 11 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_026 | sugar4d | sugar_beet | test | 16 | 0-56 | 11 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_027 | sugar4d | sugar_beet | val | 16 | 0-56 | 17 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_028 | sugar4d | sugar_beet | train | 16 | 0-56 | 14 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_029 | sugar4d | sugar_beet | test | 16 | 0-56 | 11 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_030 | sugar4d | sugar_beet | train | 16 | 0-56 | 18 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_031 | sugar4d | sugar_beet | train | 16 | 0-56 | 16 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_032 | sugar4d | sugar_beet | val | 16 | 0-56 | 14 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_033 | sugar4d | sugar_beet | train | 16 | 0-56 | 14 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_034 | sugar4d | sugar_beet | train | 16 | 0-56 | 12 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_035 | sugar4d | sugar_beet | val | 16 | 0-56 | 14 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_036 | sugar4d | sugar_beet | val | 16 | 0-56 | 11 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_037 | sugar4d | sugar_beet | val | 16 | 0-56 | 10 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_038 | sugar4d | sugar_beet | train | 16 | 0-56 | 14 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_039 | sugar4d | sugar_beet | train | 16 | 0-56 | 15 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_040 | sugar4d | sugar_beet | test | 16 | 0-56 | 11 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_041 | sugar4d | sugar_beet | test | 16 | 0-56 | 16 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_042 | sugar4d | sugar_beet | train | 16 | 0-56 | 12 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_043 | sugar4d | sugar_beet | train | 16 | 0-56 | 17 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_044 | sugar4d | sugar_beet | val | 16 | 0-56 | 16 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_045 | sugar4d | sugar_beet | test | 16 | 0-56 | 19 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_046 | sugar4d | sugar_beet | test | 16 | 0-56 | 17 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_047 | sugar4d | sugar_beet | train | 16 | 0-56 | 14 | 8192-8192 | 0 | - | no |
| sugar4d_sugar_beet_plant_048 | sugar4d | sugar_beet | train | 16 | 0-56 | 9 | 8192-8192 | 0 | - | no |
