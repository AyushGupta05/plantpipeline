# Configuration System

This directory holds the YAML configs for the two training entry points, read with
OmegaConf. Configs are grouped by the entry point that reads them -- the two are not
interchangeable, since `train_fullplant.py` reads a `model:` section while
`train_per_organ.py` reads `default_model:` and `per_organ_training:`.

## Quick Start

### Using Config Files

```bash
# Train per-organ (one model per organ) -- the method in the paper
python train_per_organ.py \
    --config configs/per_organ/shared_continuous.yaml \
    --sequences maize_control_plant1

# Train one model on the whole plant, pairwise (the DPF baseline)
python train_fullplant.py \
    --config configs/full_plant/pairwise.yaml \
    dataset.sequence=maize_control_plant1

# Train one model on the whole plant, continuous
python train_fullplant.py \
    --config configs/full_plant/continuous.yaml \
    dataset.sequence=maize_control_plant1
```

Note the two entry points select a sequence differently: `train_per_organ.py` takes
`--sequences` (or `--split`, which overrides it), `train_fullplant.py` takes the
`dataset.sequence` override. Run either with `--help` for the full flag list.

### Using CLI Overrides Only

```bash
# No config file - use defaults with CLI overrides
python train_fullplant.py \
    dataset.sequence=maize_control_plant1 \
    model.model_type=continuous \
    model.n_steps=100000
```

### Combining Config Files + CLI Overrides

```bash
# Load config and override specific parameters
python train_fullplant.py \
    --config configs/full_plant/pairwise.yaml \
    dataset.sequence=maize_control_plant2 \
    model.n_steps=5000 \
    model.lr=5.0e-6
```

The per-organ configs reproduce the paper's hyperparameters as they stand, so a bare
`--config` plus `--split` and `--n_runs` needs no overrides.

## Directory Structure

```
configs/
├── full_plant/                      # -> python train_fullplant.py
│   ├── pairwise.yaml                #    DPF single-model, the whole plant at once
│   └── continuous.yaml              #    untuned, no paper table uses it
├── per_organ/                       # -> python train_per_organ.py
│   ├── shared_continuous.yaml       #    GrowFields, the paper's method
│   ├── continuous.yaml              #    Ours (per-organ MLPs)
│   ├── pairwise.yaml                #    DPF per-organ
│   └── encoder_decoder.yaml         #    COAP
├── datasets/
│   └── TrackPlant3D/                # dataset-specific, not overridable
│       ├── data_splits.json         #   which plants are val / test
│       └── point_splits/            #   per-point train/test splits
└── README.md                        # This file
```

`dataset.dataset_path` changes where the TrackPlant3D loader reads data; it does
not select a different loader or support arbitrary file formats. Another dataset
needs compatible preprocessing or an adapter, and point splits. See
[Using your own data](../README.md#using-your-own-data).

## Configuration Structure

### Top-Level Config

```yaml
dataset:          # Dataset configuration
  ...
model:            # Model configuration -- train_fullplant.py only
  ...
default_model:    # Model configuration -- train_per_organ.py only
  ...
output:           # Output and logging configuration
  ...
wandb:            # Weights & Biases logging
  ...
device: cuda
list_sequences: false
```

### Dataset Config

```yaml
dataset:
  dataset_path: data/TrackPlant3D/versions
  sequence: maize_control_plant1     # train_fullplant.py only
  version: v2
  label_field: labels                # "labels" = ground-truth organ labels,
                                     # "predicted_labels" = PSegNet auto-segmentation
  aligned_leaves: true               # true for per-organ, false for full-plant
  normalize: false                   # must be false for per-organ training
  train_ratio: 1.0                   # Ignored when split_file is set
  split_file: null                   # Pre-computed point split JSON, full-plant configs
  min_timepoints: 1
  exclude_sequences: []              # Sequence names to skip entirely
```

The two defaults that differ by entry point are `aligned_leaves` and `normalize`: the
per-organ configs train in aligned leaf space with normalisation off, the full-plant
configs in plant space with it on. Both are what the paper used for their respective
methods, so leave them alone unless you mean to change the method.

Holding out timepoints is a **CLI flag, not a config key**: pass
`--train_timepoint_stride 2` to either entry point. There is no `dataset.timepoint_stride`
override; `train_fullplant.py` alone falls back to a `dataset.stride` key when the flag is
absent, and `train_per_organ.py` reads the flag only.

### Pairwise Model Config (`train_fullplant.py`)

`model_type: pairwise` is Dynamic Point Fields (DPF): one network per frame pair, chained
to roll the plant forward, then fine-tuned jointly. On the whole plant this is the paper's
"DPF Single-Model" baseline.

```yaml
model:
  model_type: pairwise
  hidden_features: 256
  hidden_layers: 5
  omega_0: 30.0

  # Training
  n_steps: 2000
  n_samples: 2048                    # Points sampled per training iteration
  lr: 1.0e-5
  log_interval: 200

  # Fine-tuning
  finetune_enabled: true
  finetune_steps: 3000
  finetune_lr: 1.0e-6
  w_global: 5.0
  w_local: 1.0
```

### Continuous Model Config (`train_fullplant.py`)

```yaml
model:
  model_type: continuous
  hidden_features: 256
  hidden_layers: 5
  first_omega_0: 30.0
  hidden_omega_0: 30.0

  # Training
  n_steps: 50000
  n_samples: 2048
  lr: 1.0e-5
  log_interval: 200

  # Day-based t_norm with velocity*dt (true) or frame-index-based with dt=1 (false)
  use_day_based_tnorm: true
  use_multistep_loss: false          # Chamfer at every intermediate frame
  use_odeint: false                  # off = Euler; see the config file for the solver keys
```

Note the header of `full_plant/continuous.yaml`: its values were never tuned and no paper
table uses it. The per-organ configs are the ones that reproduce the paper.

### Per-Organ Model Config (`train_per_organ.py`)

Same shape, under `default_model:` instead of `model:`, applying to every organ. The
`model_type` picks the method:

| `model_type` | Method | Config |
| --- | --- | --- |
| `shared_continuous` | GrowFields, one shared organ-conditioned field | `shared_continuous.yaml` |
| `continuous` | one independent continuous field per organ | `continuous.yaml` |
| `pairwise` | **DPF** (Dynamic Point Fields), per organ | `pairwise.yaml` |
| `encoder_decoder_continuous` | COAP-style encoder-decoder | `encoder_decoder.yaml` |

Each config adds only the keys its own `model_type` needs -- organ embeddings and
`organs_per_step` for `shared_continuous`, `latent_dim` and `lr_encoder` for
`encoder_decoder_continuous` -- so read the config file itself rather than a list here.
Keys a trainer does not read are silently ignored: `organs_per_step` means nothing to
`continuous` or `pairwise`, which train one model per organ.

```yaml
default_model:
  model_type: shared_continuous
  n_steps: 50000
  n_samples: 8192
  lr: 5.0e-6
  ...

per_organ_training:
  enabled: true
```

`per_organ_training.organ_configs` optionally overrides `default_model` for named organs,
keyed `stem`, `leaf_1`, `leaf_2`, ... -- any organ not listed uses `default_model`
unchanged:

```yaml
per_organ_training:
  enabled: true
  organ_configs:
    stem:
      n_steps: 50000
      lr: 1.0e-5
```

This applies to the two per-organ types, `pairwise` and `continuous`, which build one
model per organ. `shared_continuous` and `encoder_decoder_continuous` train a single model
over all organs from `default_model`, so there is nothing per-organ to override.

### Output Config

```yaml
output:
  output_dir: results/fullplant_single_model_batch
  save_predictions: true
  save_models: true
  save_normals: true                 # estimate normals for predicted clouds; see below
  save_full_plant: true              # per-organ only: concatenate organs into a plant
  save_plots: true
  verbose: true
```

`train_per_organ.py` **ignores `output.output_dir`** and writes to
`results/fullplant_per_organ_batch/<timestamp>_fullplant_batch_<config_name>` unless you
pass `--output_dir`. The key is still in the per-organ configs but has no effect there.

The models predict positions only, so `save_normals` fits normals to each predicted organ
with Open3D and flips each one to agree with the nearest ground-truth normal of that frame.
They are for rendering: no metric depends on their sign (NC uses `|cos|`), and CD and EPE
ignore them. Since they follow the ground truth, they also follow its flips: for about 16%
of consecutive organ day-pairs, the dataset normals face the opposite side of the organ the
next day. Turn it off to skip the estimation; ground-truth PLYs always carry the
dataset's normals. `scripts/generate_interpolation.py` takes `--no-save_normals` for the
same purpose.

## Common Use Cases

### 1. Quick Experiment with Different Sequences

```bash
# Same config, different sequences
for seq in maize_control_plant1 maize_control_plant2 maize_heat_plant1; do
    python train_fullplant.py \
        --config configs/full_plant/pairwise.yaml \
        dataset.sequence=$seq
done
```

### 2. Hyperparameter Sweep

```bash
# Try different learning rates
for lr in 1e-4 5e-5 1e-5 5e-6; do
    python train_per_organ.py \
        --config configs/per_organ/shared_continuous.yaml \
        --sequences maize_control_plant1 \
        default_model.lr=$lr
done
```

### 3. Pairwise with Fine-tuning

```bash
python train_fullplant.py \
    --config configs/full_plant/pairwise.yaml \
    dataset.sequence=maize_control_plant1 \
    model.finetune_steps=5000 \
    model.w_global=10.0
```

### 4. Interpolation Experiment

```bash
# Train on every other timepoint, holding out the rest
python train_per_organ.py \
    --config configs/per_organ/continuous.yaml \
    --split test \
    --points_split_dir configs/datasets/TrackPlant3D/point_splits \
    --train_timepoint_stride 2
```

### 5. List Available Sequences

```bash
# List all plant sequences
python train_fullplant.py --list_sequences

# Or via OmegaConf
python train_fullplant.py list_sequences=true
```

## Creating Custom Configs

### Option 1: Create a New YAML File

```bash
# Copy and modify an existing config
cp configs/per_organ/shared_continuous.yaml configs/per_organ/my_experiment.yaml

# Edit your config
# vim configs/per_organ/my_experiment.yaml

# Run with your config
python train_per_organ.py --config configs/per_organ/my_experiment.yaml --sequences maize_control_plant1
```

### Option 2: Use CLI Overrides

For quick experiments, just override parameters on the command line without creating a new file.

## Tips

1. **Start from the config closest to what you want**: `shared_continuous.yaml` is the
   paper's method, `pairwise.yaml` the DPF per-organ baseline.
2. **Override what you need**: CLI overrides for quick parameter changes, a new YAML file
   for anything you will run more than a few times.
3. **Check the saved config**: each run writes its resolved config to
   `<results_dir>/config.yaml`, and a batch run also writes `experiment_config.json` at
   the batch root.
4. **Mind the entry point**: `model:` keys do nothing in a per-organ config and
   `default_model:` keys do nothing in a full-plant one.

## OmegaConf Syntax

```bash
# Nested parameters use dot notation
model.lr=1e-5                 # train_fullplant.py
default_model.lr=1e-5         # train_per_organ.py

# Booleans
model.finetune_enabled=true
dataset.aligned_leaves=false

# Strings
dataset.sequence=maize_control_plant1
output.output_dir=results/my_exp

# Lists
dataset.exclude_sequences=[maize_control_plant1,tomato1_heat_plant3]
```

## Troubleshooting

### "Please specify dataset.sequence"

`train_fullplant.py` needs a sequence name:
```bash
python train_fullplant.py \
    --config configs/full_plant/pairwise.yaml \
    dataset.sequence=maize_control_plant1
```

### "Unknown model type"

For `train_fullplant.py`, `model.model_type` must be `pairwise` or `continuous`. For
`train_per_organ.py`, `default_model.model_type` must be one of the keys of
`ORGAN_MODEL_FACTORIES` in `growfields/models/organ_model_manager.py`: `pairwise`,
`continuous`, `shared_continuous` or `encoder_decoder_continuous`. Note `run_euler.sh`
takes the shorter `encoder_decoder` as its first argument, but that selects a config file
rather than setting `model_type`.

### My override did nothing

Check which section the key belongs to for the entry point you are running -- see the Tips
above -- and that you spelled the key right. A leftover argument without `=` is rejected,
but a **mistyped key is not**: `dataset.sequenc=maize` merges in as a new key, leaves
`dataset.sequence` at its default and trains on the wrong thing silently. The configs are
deliberately sparse -- some documented overrides appear in none of them -- so the keys
cannot be validated against them without rejecting working commands. See the note in
`merge_cli_overrides` in `growfields/pipeline/config.py`.

### Config not loading

Config paths are resolved relative to where you run the script, so run from the repository
root:
```bash
python train_per_organ.py --config configs/per_organ/shared_continuous.yaml
```
