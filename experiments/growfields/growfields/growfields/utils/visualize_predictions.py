#!/usr/bin/env python3
"""
Visualize PLY-based prediction/ground-truth sequences with Open3D.

Supports both single-organ (leaf) and full-plant visualization modes.
Point at a directory of PLY files and this script will load, color, and
display the growth sequence with optional leaf-tip spheres and connections.

Usage examples:
    # Leaf sequence (auto-detected type)
    python growfields/utils/visualize_predictions.py path/to/organs/leaf_1/predicted/

    # Full plant sequence
    python growfields/utils/visualize_predictions.py path/to/full_plant/predicted/ --type plant

    # Show only train points with organ coloring
    python growfields/utils/visualize_predictions.py path/to/full_plant/predicted/ --type plant --points train --color-by-organ

    # Compare predicted vs ground truth (ply_dir contains predicted/ and ground_truth/ subdirs)
    python growfields/utils/visualize_predictions.py path/to/full_plant/ --mode compare

    # Compare on a leaf organ directory
    python growfields/utils/visualize_predictions.py path/to/organs/leaf_1/ --mode compare
"""

import argparse
import colorsys
import re
from pathlib import Path

import numpy as np
import open3d as o3d

# Color mapping for organ labels (matches TrackPlant3D paper figures)
COLOR_MAP = {
    0: np.array([0.0, 1.0, 0.0]),  # Green  (stem)
    1: np.array([1.0, 0.0, 0.0]),  # Red
    2: np.array([1.0, 1.0, 0.0]),  # Yellow
    3: np.array([0.0, 0.0, 1.0]),  # Blue
    4: np.array([1.0, 0.65, 0.0]),  # Orange
    5: np.array([0.5, 0.0, 0.5]),  # Purple
    6: np.array([0.0, 1.0, 1.0]),  # Cyan
    7: np.array([1.0, 0.0, 1.0]),  # Magenta
    8: np.array([0.5, 1.0, 0.0]),  # Lime
    9: np.array([1.0, 0.75, 0.8]),  # Pink
}

# Regex to extract day from PLY filename: matches _D{dd} (one or more digits)
_DAY_PATTERN = re.compile(r"_D(\d+)")


def _get_organ_color(organ_id):
    """Get RGB color for an organ id, generating one if beyond COLOR_MAP."""
    if organ_id in COLOR_MAP:
        return COLOR_MAP[organ_id]
    hue = ((organ_id - 1) * 0.15) % 1.0
    return np.array(colorsys.hsv_to_rgb(hue, 0.7, 0.9))


def load_ply_sequence(directory, points_filter="all"):
    """
    Load all PLY files from *directory*, sorted by day.

    Parameters
    ----------
    directory : str or Path
        Directory containing ``*.ply`` files whose names include ``_D{dd}``.
    points_filter : str
        ``"all"`` (default), ``"train"``, or ``"test"`` — keep only points
        whose ``is_train`` attribute matches (1 for train, 0 for test).

    Returns
    -------
    list[dict]
        Each dict has keys:
        ``day``, ``points``, ``normals``, ``labels``, ``is_train``,
        ``leaf_tips`` (list of ``(organ_id, position)`` tuples).
    """
    ply_dir = Path(directory)
    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {ply_dir}")

    sequence = []
    for ply_file in ply_files:
        # Extract day from filename
        m = _DAY_PATTERN.search(ply_file.stem)
        if m is None:
            print(f"Warning: cannot parse day from '{ply_file.name}', skipping")
            continue
        day = int(m.group(1))

        # Read with tensor API to access custom attributes
        pcd = o3d.t.io.read_point_cloud(str(ply_file))
        points = pcd.point.positions.numpy()

        normals = None
        if "normals" in pcd.point:
            normals = pcd.point["normals"].numpy()

        labels = None
        if "organ_label" in pcd.point:
            labels = pcd.point["organ_label"].numpy().flatten()

        is_train = None
        if "is_train" in pcd.point:
            is_train = pcd.point["is_train"].numpy().flatten()

        # Extract leaf tips
        leaf_tips = []
        if "is_leaf_tip" in pcd.point:
            tip_mask = pcd.point["is_leaf_tip"].numpy().flatten()
            tip_indices = np.where(tip_mask == 1)[0]
            if len(tip_indices) > 0:
                tip_organ_labels = None
                if "tip_organ_label" in pcd.point:
                    tip_organ_labels = pcd.point["tip_organ_label"].numpy().flatten()
                for idx in tip_indices:
                    organ_id = int(tip_organ_labels[idx]) if tip_organ_labels is not None else 0
                    leaf_tips.append((organ_id, points[idx].copy()))

        # Apply points filter
        if points_filter != "all" and is_train is not None:
            if points_filter == "train":
                mask = is_train == 1
            elif points_filter == "test":
                mask = is_train == 0
            else:
                raise ValueError(f"Unknown points_filter: {points_filter}")
            points = points[mask]
            if normals is not None:
                normals = normals[mask]
            if labels is not None:
                labels = labels[mask]
            is_train = is_train[mask]
            # Filter leaf tips: keep only those whose nearest point survived
            # (tips are snapped to nearest point, so just check if tip index survived)
            # Simpler: keep all tips (they are positions, not filtered)

        sequence.append({
            "day": day,
            "points": points,
            "normals": normals,
            "labels": labels,
            "is_train": is_train,
            "leaf_tips": leaf_tips,
        })

    # Sort by day
    sequence.sort(key=lambda x: x["day"])
    return sequence


def create_correspondence_lines(correspondences, color=None):
    """
    Create line connections between consecutive corresponding points.

    Parameters
    ----------
    correspondences : list
        List of 3-D positions (or None for missing timepoints).
    color : array-like or None
        Fixed RGB colour for all lines. If None, uses a gradient.

    Returns
    -------
    o3d.geometry.LineSet or None
    """
    valid_points = []
    valid_indices = []
    for i, pos in enumerate(correspondences):
        if pos is not None:
            valid_points.append(pos)
            valid_indices.append(i)

    if len(valid_points) < 2:
        return None

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(valid_points)

    lines = []
    colors = []
    for i in range(len(valid_points) - 1):
        lines.append([i, i + 1])
        if color is not None:
            colors.append(np.asarray(color))
        else:
            hue = valid_indices[i] / max(1, len(valid_points) - 1)
            colors.append(np.array(colorsys.hsv_to_rgb(hue * 0.8, 0.7, 0.8)))

    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_leaf_sequence(
    ply_dir,
    spacing=30.0,
    show_leaf_tips=True,
    show_connections=False,
    show_coordinate_frame=False,
    points_filter="all",
    window_name="Leaf Growth Sequence",
    leaf_tip_radius=0.5,
):
    """
    Visualize a single organ's growth over time.

    Loads PLY files from *ply_dir* (e.g. ``organs/leaf_1/predicted/``) and
    displays the point clouds spaced along the Y axis, coloured with an HSV
    gradient over time.

    Parameters
    ----------
    ply_dir : str or Path
        Directory containing PLY files for one organ.
    spacing : float
        Distance between consecutive timepoints along Y.
    show_leaf_tips : bool
        Render leaf tips as red spheres.
    show_connections : bool
        Draw lines connecting leaf tips across time.
    show_coordinate_frame : bool
        Show an axis frame at the origin.
    points_filter : str
        ``"all"``, ``"train"``, or ``"test"``.
    window_name : str
        Title for the Open3D window.
    leaf_tip_radius : float
        Radius of the leaf-tip sphere.
    """
    sequence = load_ply_sequence(ply_dir, points_filter=points_filter)
    n = len(sequence)
    if n == 0:
        print("No timepoints to visualize.")
        return

    # HSV colour per timepoint
    tp_colors = [
        np.array(colorsys.hsv_to_rgb(i / max(1, n), 0.8, 0.9))
        for i in range(n)
    ]

    geometries = []
    all_points = []
    tip_positions = []  # for connections

    for i, tp in enumerate(sequence):
        points = tp["points"]
        if len(points) == 0:
            tip_positions.append(None)
            continue

        translated = points.copy()
        translated[:, 1] += i * spacing

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(translated)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(tp_colors[i], (len(translated), 1))
        )
        if tp["normals"] is not None:
            pcd.normals = o3d.utility.Vector3dVector(tp["normals"])
        geometries.append(pcd)
        all_points.append(translated)

        # Leaf tips
        if tp["leaf_tips"]:
            # For a single-organ directory there is typically one tip
            _, tip_pos = tp["leaf_tips"][0]
            tip_translated = tip_pos.copy()
            tip_translated[1] += i * spacing

            if show_leaf_tips:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=leaf_tip_radius)
                sphere.translate(tip_translated)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(sphere)

            tip_positions.append(tip_translated)
        else:
            tip_positions.append(None)

    # Connections
    if show_connections:
        ls = create_correspondence_lines(tip_positions)
        if ls is not None:
            geometries.append(ls)

    # Coordinate frame
    if show_coordinate_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        geometries.append(frame)

    # Camera lookat
    if all_points:
        concat = np.vstack(all_points)
        mean_pt = concat.mean(axis=0)
        center_y = (n - 1) * spacing / 2
        lookat = np.array([mean_pt[0], center_y, mean_pt[2]])
    else:
        lookat = np.zeros(3)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        up=[0, 0, 1],
        front=[1, 0, 0],
        lookat=lookat,
    )


def visualize_plant_sequence(
    ply_dir,
    spacing=30.0,
    show_leaf_tips=True,
    show_connections=False,
    show_coordinate_frame=False,
    color_by_organ=True,
    points_filter="all",
    window_name="Plant Growth Sequence",
    leaf_tip_radius=1.0,
):
    """
    Visualize a full plant's growth over time.

    Loads PLY files from *ply_dir* (e.g. ``full_plant/predicted/``) and
    displays the point clouds spaced along Y, coloured either by organ label
    or by timepoint.

    Parameters
    ----------
    ply_dir : str or Path
        Directory containing full-plant PLY files.
    spacing : float
        Distance between consecutive timepoints along Y.
    show_leaf_tips : bool
        Render leaf tips as spheres coloured by organ.
    show_connections : bool
        Draw per-organ lines connecting leaf tips across time.
    show_coordinate_frame : bool
        Show an axis frame at the origin.
    color_by_organ : bool
        If True, colour points by organ label; otherwise by timepoint.
    points_filter : str
        ``"all"``, ``"train"``, or ``"test"``.
    window_name : str
        Title for the Open3D window.
    leaf_tip_radius : float
        Radius of the leaf-tip spheres.
    """
    sequence = load_ply_sequence(ply_dir, points_filter=points_filter)
    n = len(sequence)
    if n == 0:
        print("No timepoints to visualize.")
        return

    # Timepoint colours (used when color_by_organ is False)
    tp_colors = [
        np.array(colorsys.hsv_to_rgb(i / max(1, n - 1), 0.8, 0.9))
        for i in range(n)
    ]

    geometries = []
    all_points = []
    all_leaf_tips = []  # list of list[(organ_id, position)]

    for i, tp in enumerate(sequence):
        points = tp["points"]
        if len(points) == 0:
            all_leaf_tips.append([])
            continue

        translated = points.copy()
        translated[:, 1] += i * spacing

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(translated)

        # Colouring
        if color_by_organ and tp["labels"] is not None:
            point_colors = np.zeros((len(points), 3))
            for organ_id in np.unique(tp["labels"]):
                mask = tp["labels"] == organ_id
                point_colors[mask] = _get_organ_color(int(organ_id))
        else:
            point_colors = np.tile(tp_colors[i], (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(point_colors)

        if tp["normals"] is not None:
            pcd.normals = o3d.utility.Vector3dVector(tp["normals"])
        geometries.append(pcd)
        all_points.append(translated)

        # Leaf tips
        timepoint_tips = []
        for organ_id, tip_pos in tp["leaf_tips"]:
            tip_translated = tip_pos.copy()
            tip_translated[1] += i * spacing

            if show_leaf_tips:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=leaf_tip_radius)
                sphere.translate(tip_translated)
                sphere.paint_uniform_color(_get_organ_color(organ_id))
                geometries.append(sphere)

            timepoint_tips.append((organ_id, tip_translated))
        all_leaf_tips.append(timepoint_tips)

    # Connections: per-organ trajectories
    if show_connections and len(all_leaf_tips) > 1:
        all_organ_ids = set()
        for tips in all_leaf_tips:
            for oid, _ in tips:
                all_organ_ids.add(oid)

        for organ_id in sorted(all_organ_ids):
            trajectory = []
            for tips in all_leaf_tips:
                found = None
                for oid, pos in tips:
                    if oid == organ_id:
                        found = pos
                        break
                trajectory.append(found)

            line_color = _get_organ_color(organ_id)
            ls = create_correspondence_lines(trajectory, color=line_color)
            if ls is not None:
                geometries.append(ls)

    # Coordinate frame
    if show_coordinate_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
        geometries.append(frame)

    # Camera lookat
    if all_points:
        concat = np.vstack(all_points)
        mean_pt = concat.mean(axis=0)
        center_y = (n - 1) * spacing / 2
        lookat = np.array([mean_pt[0], center_y, mean_pt[2]])
    else:
        lookat = np.zeros(3)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        up=[0, 0, 1],
        front=[1, 0, 0],
        lookat=lookat,
    )


def visualize_compare(
    ply_dir,
    spacing=30.0,
    show_coordinate_frame=False,
    points_filter="all",
    window_name="Predicted vs Ground Truth",
    pred_color=None,
    gt_color=None,
):
    """
    Overlay predicted and ground-truth point clouds for each timepoint.

    *ply_dir* must contain ``predicted/`` and ``ground_truth/`` subdirectories.
    Both sequences are loaded, matched by day, and rendered overlaid with
    distinct colours so the fitting quality can be assessed visually.

    Parameters
    ----------
    ply_dir : str or Path
        Parent directory containing ``predicted/`` and ``ground_truth/`` subdirs.
    spacing : float
        Distance between consecutive timepoints along Y.
    show_coordinate_frame : bool
        Show an axis frame at the origin.
    points_filter : str
        ``"all"``, ``"train"``, or ``"test"``.
    window_name : str
        Title for the Open3D window.
    pred_color : array-like or None
        RGB [0-1] colour for predicted points (default: blue).
    gt_color : array-like or None
        RGB [0-1] colour for ground-truth points (default: red).
    """
    ply_dir = Path(ply_dir)
    pred_dir = ply_dir / "predicted"
    gt_dir = ply_dir / "ground_truth"

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"No predicted/ subdirectory in {ply_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"No ground_truth/ subdirectory in {ply_dir}")

    pred_seq = load_ply_sequence(pred_dir, points_filter=points_filter)
    gt_seq = load_ply_sequence(gt_dir, points_filter=points_filter)

    # Index by day for matching
    pred_by_day = {tp["day"]: tp for tp in pred_seq}
    gt_by_day = {tp["day"]: tp for tp in gt_seq}
    all_days = sorted(set(pred_by_day.keys()) | set(gt_by_day.keys()))

    if not all_days:
        print("No timepoints to visualize.")
        return

    if pred_color is None:
        pred_color = np.array([0.2, 0.4, 1.0])  # blue
    else:
        pred_color = np.asarray(pred_color)
    if gt_color is None:
        gt_color = np.array([1.0, 0.2, 0.2])  # red
    else:
        gt_color = np.asarray(gt_color)

    geometries = []
    all_points = []
    n = len(all_days)

    for i, day in enumerate(all_days):
        y_offset = i * spacing

        # Ground truth
        if day in gt_by_day:
            pts = gt_by_day[day]["points"]
            if len(pts) > 0:
                translated = pts.copy()
                translated[:, 1] += y_offset
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(translated)
                pcd.colors = o3d.utility.Vector3dVector(np.tile(gt_color, (len(pts), 1)))
                if gt_by_day[day]["normals"] is not None:
                    pcd.normals = o3d.utility.Vector3dVector(gt_by_day[day]["normals"])
                geometries.append(pcd)
                all_points.append(translated)

        # Predicted
        if day in pred_by_day:
            pts = pred_by_day[day]["points"]
            if len(pts) > 0:
                translated = pts.copy()
                translated[:, 1] += y_offset
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(translated)
                pcd.colors = o3d.utility.Vector3dVector(np.tile(pred_color, (len(pts), 1)))
                if pred_by_day[day]["normals"] is not None:
                    pcd.normals = o3d.utility.Vector3dVector(pred_by_day[day]["normals"])
                geometries.append(pcd)
                all_points.append(translated)

    if show_coordinate_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        geometries.append(frame)

    # Camera lookat
    if all_points:
        concat = np.vstack(all_points)
        mean_pt = concat.mean(axis=0)
        center_y = (n - 1) * spacing / 2
        lookat = np.array([mean_pt[0], center_y, mean_pt[2]])
    else:
        lookat = np.zeros(3)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        up=[0, 0, 1],
        front=[1, 0, 0],
        lookat=lookat,
    )


def _detect_type(sequence):
    """Return ``"leaf"`` if all timepoints have a single organ label, else ``"plant"``."""
    for tp in sequence:
        if tp["labels"] is not None and len(np.unique(tp["labels"])) > 1:
            return "plant"
    return "leaf"


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PLY prediction / ground-truth sequences"
    )
    parser.add_argument(
        "ply_dir",
        type=str,
        help="Path to directory containing PLY files (e.g. .../predicted/), "
        "or parent dir with predicted/ & ground_truth/ subdirs for compare mode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "compare"],
        default="normal",
        help="normal: visualize a single sequence; compare: overlay predicted/ and ground_truth/ (default: normal)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["leaf", "plant", "auto"],
        default="auto",
        dest="seq_type",
        help="Sequence type: leaf (single organ), plant (full plant), or auto-detect (default: auto)",
    )
    parser.add_argument(
        "--points",
        type=str,
        choices=["all", "train", "test"],
        default="all",
        help="Filter points by train/test split (default: all)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=30.0,
        help="Spacing between timepoints along Y axis (default: 30.0)",
    )
    parser.add_argument(
        "--no-tips",
        action="store_true",
        help="Hide leaf tip spheres (normal mode)",
    )
    parser.add_argument(
        "--no-connections",
        action="store_true",
        help="Hide connections between leaf tips across time (normal mode)",
    )
    parser.add_argument(
        "--show-coord-frame",
        action="store_true",
        help="Show coordinate frame at origin",
    )
    parser.add_argument(
        "--color-by-organ",
        action="store_true",
        default=True,
        help="Color points by organ label (normal+plant, default)",
    )
    parser.add_argument(
        "--color-by-time",
        action="store_true",
        help="Color points by timepoint instead of organ (normal+plant)",
    )
    parser.add_argument(
        "--leaf-tip-radius",
        type=float,
        default=0.5,
        help="Radius of leaf tip spheres (default: 0.5)",
    )

    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    if not ply_dir.is_dir():
        parser.error(f"Not a directory: {ply_dir}")

    mode = args.mode
    seq_type = args.seq_type

    # In compare mode, ply_dir is the parent with predicted/ & ground_truth/
    if mode == "compare":
        visualize_compare(
            ply_dir,
            spacing=args.spacing,
            show_coordinate_frame=args.show_coord_frame,
            points_filter=args.points,
        )
    else:
        # Resolve type
        if seq_type == "auto":
            seq = load_ply_sequence(ply_dir, points_filter=args.points)
            seq_type = _detect_type(seq)
            print(f"Auto-detected type: {seq_type}")

        color_by_organ = args.color_by_organ and not args.color_by_time

        if seq_type == "leaf":
            visualize_leaf_sequence(
                ply_dir,
                spacing=args.spacing,
                show_leaf_tips=not args.no_tips,
                show_connections=not args.no_connections,
                show_coordinate_frame=args.show_coord_frame,
                points_filter=args.points,
                leaf_tip_radius=args.leaf_tip_radius,
            )
        else:
            visualize_plant_sequence(
                ply_dir,
                spacing=args.spacing,
                show_leaf_tips=not args.no_tips,
                show_connections=not args.no_connections,
                show_coordinate_frame=args.show_coord_frame,
                color_by_organ=color_by_organ,
                points_filter=args.points,
                leaf_tip_radius=args.leaf_tip_radius,
            )


if __name__ == "__main__":
    main()
