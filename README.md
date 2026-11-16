# plantpipeline

Data pipeline and experiments for time-conditioned modelling of growing plants from laser scans
(project with Yancong Lin, University of Nottingham).

- `plantpipeline/` — the data pipeline: Pheno4D ingest (soil removal, farthest-point sampling, plant-centred frames),
  converters that merge Pheno4D, TrackPlant3D v2 and Sugar4D into one format (`data/unified`), PyTorch datasets, a viewer.
- `experiments/` — the GrowFields benchmark work: faithful reproduction, nearest-source inference, monotone velocity head,
  the MonoGrow cross-plant prior, the RichardsNet growth-curve baseline, AWS fleet scripts, all result tables
  (`experiments/results/`) and the write-up (`experiments/WRITEUP.md`). Start at `experiments/README.md` and
  `experiments/NEXT_STEPS.md`.

## Setup

```bash
uv sync --extra torch --extra view
mkdir -p data/raw && curl -L -o data/raw/Pheno4D.zip https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip
```

## Pipeline

```bash
plantpipeline inventory data/raw/Pheno4D.zip
plantpipeline process data/raw/Pheno4D.zip --out data/processed
plantpipeline unify                      # needs data/raw/TrackPlant3D_v2.zip and sugar4d.zip too (see plantpipeline/sources/README.md)
plantpipeline view M01                   # Open3D window; n / p step through days
```

Output per scan: `points (N,3) float32 mm`, `labels (N,) int16` (1 stem, 2.. leaves), `day`; plus `manifest.csv` and `sequences.json`.
Data, run outputs and virtual environments are git-ignored.
