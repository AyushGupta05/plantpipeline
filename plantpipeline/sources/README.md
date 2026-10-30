# Unified 4D plant format (`data/unified/`)

Several 4D plant point-cloud datasets, one layout, so a growth model can train across them.
Built by `plantpipeline unify` from `plantpipeline/sources/` (one short module per dataset).

## Contract

```
data/unified/<dataset>/<sequence_id>/<scan_id>.npz
data/unified/manifest.csv      one row per scan
data/unified/sequences.json    one entry per sequence
```

Per-scan npz keys:

| key | type | meaning |
|---|---|---|
| `points` | float32 (N, 3) | millimetres, plant-centred: stem/plant base at the origin, z up. N varies per dataset (see below); the dataset class subsamples |
| `labels` | int16 (N,) | `0` unknown, `1` stem, `2..` leaf instance id, consistent over time within a sequence when the source provides it (all three sources do) |
| `day` | float | days since the sequence's first scan |
| `t_norm` | float | `day / last day` of the sequence, in [0, 1] (0 for a single-frame sequence) |

`manifest.csv` columns: `scan_id, dataset, species, treatment, sequence_id, frame (0-based order), day, t_norm,
source_time (the source's own time stamp: date / D index / DAS), split, labelled (1 = ground-truth labels on
this scan, 0 = transferred), n_points, n_leaves, height_mm, path`.

`sequences.json` entries: `dataset, species, treatment, sequence_id, split, frames (ordered scan ids), days,
n_frames, has_consistent_labels, notes (units / source frame / what was shifted)`.

Splits are by **sequence** (never by scan), taken from each source:

| dataset | split rule |
|---|---|
| pheno4d | plants 01-05 train, 06 val, 07 test (as in `process.py`) |
| trackplant3d | GrowFields `v2/splits/data_splits.json` exactly: 4 val plants, 8 test plants, the other 31 train |
| sugar4d | shipped `splits/{train,val,test}.txt` (24 / 12 / 12 plants) |

## Sources

### pheno4d (`sources/pheno4d.py`) - maize + tomato, Bonn laser arm
Copied from `data/processed/` (output of `plantpipeline process`; `process.py` is unchanged). Already mm,
plant-centred (xy = plant centroid, z = soil surface), FPS to 8192 points, labels 1 stem / 2.. leaf. Leaf ids
persist across days in the Pheno4D annotation; unlabelled days carry 1-NN transferred labels (`labelled = 0`).
Sequence = plant (`M01`..`T07`), 14 sequences, 223 scans, daily, up to 20 days.

### trackplant3d (`sources/trackplant3d.py`) - maize, sorghum, tobacco, tomato (2 cultivars), ETH v2
Raw: `data/raw/TrackPlant3D_v2.zip` (MD5 `d07236ef8c42298c4518082aa5e0699e`), `v2/gt_corrected_v2/<crop>/*.ply`,
binary PLY with `x y z` (double, mm), normals, `organ_label` (0 stem, 1.. leaf, consistent over time), `is_leaf_tip`.
Clouds are sparse (4,655-12,279 points, kept as is, no FPS up), already temporally aligned by GrowFields'
stem-based ICP, z up, stem base at z = 0. We only (a) shift xy by one vector per sequence so the first frame's
stem base (mean of the lowest 10 % stem points) is at the origin (after that every frame's stem base is within
~10 mm of the origin), and (b) remap labels: stem 0 -> 1, leaf k -> k + 1. `day` = the `D<dd>` index in the file
name (daily or 2-day cadence). Treatment comes from the sequence name (`control1/2` -> `control`); the cultivar
(`tomato1`/`tomato2`) is kept in `sequence_id` and `notes`. Normals and leaf tips are not carried over.
43 sequences, 378 scans. Note the v2 `transformations/*.json` are documented as wrong upstream; not used.

### sugar4d (`sources/sugar4d.py`) - sugar beet, Bonn / IfZ Goettingen terrestrial LiDAR
Raw: `data/raw/sugar4d.zip` (6.27 GB, bonndata file id 54550, `https://bonndata.uni-bonn.de/api/access/datafile/54550`,
CC BY 4.0; downloaded with `curl -L -C - --retry 20 ...`). `point_clouds/<DAS>_das/<DAS>_das_plant_NNN.ply`,
16 time points (DAS 37..93, semiweekly) x 48 plants, binary PLY in **metres** in a greenhouse-fixed frame with
the ground plane removed, fields `x y z nx ny nz label_semantic label_instance`.
Mapping: metres -> mm; `label_semantic` 0 taproot and 1 core -> stem (1); 2 leaf -> `label_instance + 2`
(instances 0..19 are consistent over time); a leaf point without an instance (< 0) -> 0 unknown. Each sequence is
shifted by one vector: the first frame's plant base (xy = centroid of stem points, z = lowest stem point), so
growth and any pot movement stay visible. Dense clouds (15k-500k points) are FPS-downsampled to `--n-points`
(8192) with `sample.fps`. `day` = DAS - first DAS of the plant (37), `source_time` = `DAS<n>`. The 12 genotypes
exist only as an image (`supplements/experiment_layout.png`), so `treatment = control` for all plants.
48 sequences, 768 scans.

### Skipped: Bonn 4D plant registration zip (212 MB)
xyz-only, soil removed, 6 plants that are the same scans as Pheno4D M01/M02/M04 and T01/T05 - redundant with
the labelled Pheno4D copy already in the unified set.

## Usage

```
plantpipeline unify                                   # all three sources -> data/unified
plantpipeline unify --datasets pheno4d trackplant3d   # subset
plantpipeline view maize_control_plant2 --out data/unified/trackplant3d            # Open3D window, n/p = next/prev frame
plantpipeline view plant_001 --out data/unified/sugar4d --png snapshots/sugar_001  # PNG per frame
```

```python
from plantpipeline.sequences import PlantSequences
ds = PlantSequences(Path("data/unified"), split="train", n=2048, datasets=["trackplant3d", "sugar4d"])
item = ds[0]   # points (T, n, 3) in mm/scale_mm, labels (T, n), days (T,), t_norm (T,), meta dict
```
T differs per sequence, so use `batch_size=1` or a custom `collate_fn`. `PlantScans` in `dataset.py` is unchanged
(per-scan, Pheno4D processed layout).

## Verification (2026-10-30)

`plantpipeline unify` (all three sources, ~4 min, almost all of it inflating + FPS-ing the 9.8 GB of Sugar4D PLYs):

```
1369 scans, 105 sequences
  by dataset: scans {pheno4d: 223, trackplant3d: 378, sugar4d: 768} | sequences {pheno4d: 14, trackplant3d: 43, sugar4d: 48}
  by species: scans {maize: 109, tomato: 352, sorghum: 82, tobacco: 58, sugar_beet: 768} | sequences {maize: 10, tomato: 27, sorghum: 12, tobacco: 8, sugar_beet: 48}
  by split:   scans {train: 810, val: 257, test: 302} | sequences {train: 65, val: 18, test: 22}
  pheno4d:      points/scan 8192,       frames/seq 11-20, day range 0-20
  trackplant3d: points/scan 4655-12279, frames/seq 4-19,  day range 0-18
  sugar4d:      points/scan 8192,       frames/seq 16,    day range 0-56
```
`data/unified/` is 185 MB. Checks done:
- manifest rows == sum of `n_frames` in sequences.json; every `path` exists; `PlantSequences` returns
  `(T, n, 3)` / `(T, n)` / `(T,)` tensors for Pheno4D, TrackPlant3D and Sugar4D items.
- TrackPlant3D: after the per-sequence xy shift the per-frame stem base is within 10 mm of the origin
  (worst: `sorghum_highlight_plant1`). Sugar4D: plant base drifts < 15 mm (xy and z) over the 56 days in the
  greenhouse frame, so one origin per sequence is adequate; no leaf points lack an instance id (0 unknown never
  occurs); leaf ids persist over frames (e.g. plant_001: ids 2-5 at DAS 37, old ids drop out as leaves die,
  new ones are appended, up to id 14).
- Rendered with `plantpipeline view ... --png` (in `snapshots/`): `tp3d_maize_control_plant2`,
  `tp3d_tomato1_shade_plant1`, `unified_M01`, `sugar_plant_001` - stem grey at the origin, leaves coloured by
  instance, z up, growth visible frame to frame.

Known source quirks (left as is):
- Pheno4D maize uses the "collar" convention: a young leaf gets its own id only once its collar is visible, so the
  newest leaves carry the stem label (1) on GT days too (see `M01_0325`).
- TrackPlant3D's Pheno4D-derived sequences (maize, tomato2) are the same plants as part of `pheno4d` at
  10k points; keep that in mind when mixing both datasets in one split (sequence-level splits differ:
  GrowFields puts `maize_control_plant1` in val, Pheno4D puts M06 in val).
- Sugar4D genotype is not machine-readable; `treatment` is `control` for all 48 plants.
