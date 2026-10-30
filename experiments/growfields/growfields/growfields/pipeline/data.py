"""Dataset construction and per-sequence array preparation."""

import torch

from growfields.datasets.trackplant3d import TrackPlant3DDataset


def load_plant_dataset(
    dataset_cfg,
    sequence_name,
    *,
    split_file,
    timepoint_stride,
    normalize,
    device,
):
    """Build the plant-mode TrackPlant3D dataset for one sequence.

    Both entry points go through here so the constructor arguments cannot drift
    apart again -- `label_field` was previously honoured by the batch runner but
    silently dropped by the single-model script.

    Args:
        dataset_cfg: The `dataset` section of the config.
        sequence_name: Sequence to load.
        split_file: Train/test point split to apply, or None.
        timepoint_stride: 1 = every timepoint.
        normalize: Whether the dataset applies global normalization. When False
            it exposes no `global_min`/`global_scale`.
        device: Passed explicitly by the callers; select a GPU with
            `CUDA_VISIBLE_DEVICES`.
    """
    return TrackPlant3DDataset(
        dataset_path=dataset_cfg.dataset_path,
        sequence_name=sequence_name,
        version=dataset_cfg.version,
        mode="plant",
        train_ratio=dataset_cfg.train_ratio,
        split_file=split_file,
        aligned_leaves=dataset_cfg.aligned_leaves,
        label_field=dataset_cfg.get("label_field", "labels"),
        normalize=normalize,
        timepoint_stride=timepoint_stride,
        exclude_sequences=dataset_cfg.get("exclude_sequences", None),
        device=device,
    )


def combine_train_test_clouds(dataset):
    """Concatenate train and test points per timepoint, tracking which is which.

    The full plant is predicted from all points, but metrics are reported
    separately for train and test, so the split has to survive concatenation.

    Returns:
        (clouds, labels, normals, is_train_masks), each a list over timepoints.
        Clouds/labels/normals carry a leading batch dimension; `is_train_masks`
        holds 1 for train points and 0 for test.
    """
    clouds, labels, normals, is_train_masks = [], [], [], []

    for t in range(len(dataset.train_pts)):
        train_pts, test_pts = dataset.train_pts[t], dataset.test_pts[t]

        clouds.append(torch.cat([train_pts, test_pts], dim=0).unsqueeze(0))
        labels.append(
            torch.cat([dataset.train_labels[t], dataset.test_labels[t]], dim=0).unsqueeze(0)
        )
        normals.append(
            torch.cat([dataset.train_normals[t], dataset.test_normals[t]], dim=0).unsqueeze(0)
        )
        is_train_masks.append(
            torch.cat(
                [
                    torch.ones(len(train_pts), dtype=torch.int32, device=train_pts.device),
                    torch.zeros(len(test_pts), dtype=torch.int32, device=test_pts.device),
                ]
            )
        )

    return clouds, labels, normals, is_train_masks


def get_day_range(dataset):
    """Return (days, full_day_range) for a sequence.

    `days` are the timepoints actually loaded (affected by stride); the range is
    taken from the *unstrided* plant timeseries, so t_norm stays comparable
    across runs with different strides.
    """
    days = [tp.get("day", i) for i, tp in enumerate(dataset.sequence_data["timepoints"])]

    full_plant_sequence = dataset.plant_dataset.get_timeseries_by_sequence_name(
        dataset.sequence_name
    )
    all_days = [tp["day"] for tp in full_plant_sequence["timepoints"]]

    return days, (min(all_days), max(all_days))


def get_normalization_params(dataset, device):
    """Return (min_xyz, scale) tensors, or (None, None) when unnormalized.

    `global_min`/`global_scale` are only set when the dataset was built with
    `normalize=True`, so their absence is how an unnormalized dataset presents.
    """
    if not (hasattr(dataset, "global_min") and hasattr(dataset, "global_scale")):
        return None, None

    min_xyz = (
        torch.tensor(dataset.global_min, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    scale = torch.tensor([[dataset.global_scale]], dtype=torch.float32).to(device)
    return min_xyz, scale
