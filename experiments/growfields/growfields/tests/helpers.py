"""Reusable helpers for PLY verification tests."""

import glob
import os
import re

import numpy as np


def find_training_organ_files(organ_dir, kind="predicted", prefix="pred"):
    """Find training (seen) organ files, supporting both direct and finetuned layouts.

    Direct layout:       organ_dir/predicted/pred_*.ply
    Finetuned layout:    organ_dir/post_finetune/predicted/pred_*.ply
                      or organ_dir/pre_finetune/predicted/pred_*.ply
    """
    # Direct layout
    files = find_ply(organ_dir, f"{kind}/{prefix}_*.ply")
    files = [f for f in files if "interpolation" not in f]
    if files:
        return files

    # Finetuned layout — prefer post_finetune
    for variant in ("post_finetune", "pre_finetune"):
        files = find_ply(organ_dir, f"{variant}/{kind}/{prefix}_*.ply")
        if files:
            return files

    return []


def build_day_leaf_tip_map(leaf_sequences):
    """Build day -> {leaf_id: leaf_tip_position} from leaf sequences."""
    day_leaf_tips = {}
    for leaf_seq in leaf_sequences:
        leaf_id = leaf_seq["leaf_id"]
        for ltp in leaf_seq["timepoints"]:
            tip = ltp.get("leaf_tip")
            if tip is not None:
                day = ltp["day"]
                if day not in day_leaf_tips:
                    day_leaf_tips[day] = {}
                day_leaf_tips[day][leaf_id] = np.asarray(tip)
    return day_leaf_tips


def find_ply(base_dir, pattern):
    """Glob for PLY files under *base_dir* matching *pattern*.

    Returns a sorted list of absolute paths.
    """
    matches = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    return sorted(matches)


def day_from_filename(path):
    """Extract the day number from a PLY filename like ``..._D05.ply``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    for part in reversed(stem.split("_")):
        if part.startswith("D") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"Cannot extract day from {path}")


def day_from_filename_float(path):
    """Extract integer day from generate_interpolation.py filenames like ``..._D3.00_F0003.ply``.

    Returns the integer day if the float value is whole, otherwise None
    (non-integer days occur when n_interp > 1 and the intermediate frame does
    not fall on an integer day).
    """
    m = re.search(r"_D(\d+(?:\.\d+)?)_F\d+(?:_\w+)?\.ply$", str(path))
    if m is None:
        return None
    val = float(m.group(1))
    return int(val) if val == int(val) else None
