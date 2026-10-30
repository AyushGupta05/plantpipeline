#!/usr/bin/env python3
"""
Compute per-leaf surface area from dense interpolation PLY files.

Extracts per-leaf point clouds from full-plant PLYs with organ_label attributes,
meshes each leaf via trimmed Poisson reconstruction, and computes surface area.
Compares predicted vs GT leaf areas.

Usage:
    # Basic usage on one sequence
    python scripts/compute_leaf_areas.py \
        --results_dir results/fullplant_per_organ_batch/20260302_... \
        --sequences maize_control_plant2

    # Save meshes for visual inspection
    python scripts/compute_leaf_areas.py \
        --results_dir results/fullplant_per_organ_batch/20260302_... \
        --save_meshes

    # Include stem area
    python scripts/compute_leaf_areas.py \
        --results_dir results/fullplant_per_organ_batch/20260302_... \
        --include_stem
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

def load_ply_with_organs(path):
    """Read PLY file and extract points, organ_label, and normals.

    Returns:
        tuple: (points, organ_labels, normals) — normals may be None
    """
    pcd = o3d.t.io.read_point_cloud(str(path))
    points = pcd.point.positions.numpy()

    organ_labels = None
    if "organ_label" in pcd.point:
        organ_labels = pcd.point.organ_label.numpy().flatten()

    normals = None
    if "normals" in pcd.point:
        normals = pcd.point.normals.numpy()

    return points, organ_labels, normals

def extract_organ_point_clouds(points, organ_labels, normals, skip_stem=True):
    """Split point cloud by organ_label.

    Args:
        points: (N, 3) array
        organ_labels: (N,) array of organ IDs (0=stem, >0=leaves)
        normals: (N, 3) array or None
        skip_stem: If True, exclude organ_label=0

    Returns:
        dict[int, tuple]: organ_id -> (points, normals) where normals may be None
    """
    if organ_labels is None:
        return {}

    organs = {}
    for organ_id in np.unique(organ_labels):
        organ_id = int(organ_id)
        if skip_stem and organ_id == 0:
            continue
        mask = organ_labels == organ_id
        organ_pts = points[mask]
        organ_normals = normals[mask] if normals is not None else None
        organs[organ_id] = (organ_pts, organ_normals)

    return organs

def _prepare_pcd(points, normals):
    """Create Open3D point cloud with normals estimated if needed."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )

    pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd

def _mesh_poisson(pcd, poisson_depth=8, density_percentile=10, distance_multiplier=10.0):
    """Poisson reconstruction with density and distance trimming."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    densities = np.asarray(densities)

    if len(densities) == 0:
        return None

    # Density-based trimming
    density_threshold = np.percentile(densities, density_percentile)
    mesh.remove_vertices_by_mask(densities < density_threshold)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    # Distance-based trimming
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    distances = np.zeros(len(mesh_vertices))
    for i, v in enumerate(mesh_vertices):
        _, _, dist2 = pcd_tree.search_knn_vector_3d(v, 1)
        distances[i] = np.sqrt(dist2[0])

    dist_threshold = distances.mean() * distance_multiplier
    mesh.remove_vertices_by_mask(distances > dist_threshold)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    return mesh

def _mesh_ball_pivoting(pcd, radius_scaling=5):
    """Ball pivoting reconstruction with adaptive radii."""
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist / 2
    radii = [(1 / radius_scaling) * radius, radius, radius_scaling * radius]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    return mesh

def mesh_leaf_and_compute_area(
    points, normals, method="poisson",
    poisson_depth=8, density_percentile=10, distance_multiplier=10.0,
    radius_scaling=5,
):
    """Mesh a leaf point cloud and compute area.

    Args:
        points: (N, 3) array
        normals: (N, 3) array or None (will be estimated if None)
        method: "poisson" or "ball_pivoting"
        poisson_depth: Poisson reconstruction depth (poisson only)
        density_percentile: Density trim percentile (poisson only)
        distance_multiplier: Distance trim multiplier (poisson only)
        radius_scaling: Scaling factor for ball pivoting radii (ball_pivoting only)

    Returns:
        tuple: (area, mesh) or (None, None) on failure
    """
    try:
        pcd = _prepare_pcd(points, normals)

        if method == "ball_pivoting":
            mesh = _mesh_ball_pivoting(pcd, radius_scaling=radius_scaling)
        else:
            mesh = _mesh_poisson(
                pcd,
                poisson_depth=poisson_depth,
                density_percentile=density_percentile,
                distance_multiplier=distance_multiplier,
            )

        if mesh is None:
            return None, None

        area = mesh.get_surface_area()
        return area, mesh

    except Exception as e:
        print(f"    Warning: Meshing failed: {e}")
        return None, None

def parse_interpolation_filename(filename):
    """Extract (day_float, frame_idx, split_tag) from interpolation PLY filenames.

    Handles formats:
        - seq_D12.50_F0003.ply          -> (12.5, 3, None)
        - seq_D12.50_F0003_train.ply    -> (12.5, 3, 'train')
        - pred_seq_D01.ply              -> (1.0, None, None)
        - gt_seq_D01.ply                -> (1.0, None, None)

    Returns:
        tuple: (day_float, frame_idx, split_tag) — frame_idx and split_tag may be None
    """
    # Try new format with frame index: _D<day>_F<frame>[_split].ply
    match = re.search(r"_D(\d+(?:\.\d+)?)_F(\d+)(?:_(train|eval))?\.(ply)$", filename)
    if match:
        day = float(match.group(1))
        frame_idx = int(match.group(2))
        split_tag = match.group(3)
        return day, frame_idx, split_tag

    # Try old format: pred_seq_D<day>.ply or gt_seq_D<day>.ply
    match = re.search(r"_D(\d+(?:\.\d+)?)\.(ply)$", filename)
    if match:
        day = float(match.group(1))
        return day, None, None

    return None, None, None

def process_sequence(seq_dir, sequence_name, args):
    """Process all matched pred/GT frames in a sequence directory.

    Looks for predicted/ and ground_truth/ subdirectories.

    Args:
        seq_dir: Path to sequence interpolation directory
        sequence_name: Name of the sequence
        args: Parsed CLI arguments

    Returns:
        list[dict]: Rows for the output CSV
    """
    pred_dir = os.path.join(seq_dir, "predicted")
    gt_dir = os.path.join(seq_dir, "ground_truth")

    if not os.path.isdir(pred_dir):
        print(f"  No predicted/ directory found in {seq_dir}")
        return []

    has_gt = os.path.isdir(gt_dir)

    # List PLY files
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".ply")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".ply")]) if has_gt else []

    if not pred_files:
        print(f"  No prediction PLY files found in {pred_dir}")
        return []

    # Parse filenames and index by day
    pred_by_day = {}
    for f in pred_files:
        day, frame_idx, split_tag = parse_interpolation_filename(f)
        if day is not None:
            pred_by_day[day] = (f, frame_idx, split_tag)

    gt_by_day = {}
    for f in gt_files:
        day, frame_idx, split_tag = parse_interpolation_filename(f)
        if day is not None:
            gt_by_day[day] = (f, frame_idx, split_tag)

    # Process all predicted days, matching with GT where available
    rows = []
    skip_stem = not args.include_stem

    days_to_process = sorted(pred_by_day.keys())
    for day in tqdm(days_to_process, desc=f"  {sequence_name}", leave=False):
        pred_file, pred_frame_idx, pred_split = pred_by_day[day]
        pred_path = os.path.join(pred_dir, pred_file)

        # Load predicted PLY
        pred_points, pred_labels, pred_normals = load_ply_with_organs(pred_path)
        if pred_labels is None:
            print(f"    Warning: No organ_label in {pred_file}, skipping")
            continue

        pred_organs = extract_organ_point_clouds(
            pred_points, pred_labels, pred_normals, skip_stem=skip_stem
        )

        # Load GT if available for this day
        gt_organs = {}
        if day in gt_by_day:
            gt_file, _, _ = gt_by_day[day]
            gt_path = os.path.join(gt_dir, gt_file)
            gt_points, gt_labels, gt_normals = load_ply_with_organs(gt_path)
            if gt_labels is not None:
                gt_organs = extract_organ_point_clouds(
                    gt_points, gt_labels, gt_normals, skip_stem=skip_stem
                )

        # Determine split tag
        split_tag = pred_split
        if split_tag is None and day in gt_by_day:
            split_tag = gt_by_day[day][2]

        # Union of all organ IDs from pred and GT
        all_organ_ids = sorted(set(pred_organs.keys()) | set(gt_organs.keys()))

        # Mesh output directories
        if args.save_meshes:
            mesh_pred_dir = os.path.join(
                seq_dir, "leaf_meshes", args.method, "predicted"
            )
            mesh_gt_dir = os.path.join(
                seq_dir, "leaf_meshes", args.method, "ground_truth"
            )
            os.makedirs(mesh_pred_dir, exist_ok=True)
            os.makedirs(mesh_gt_dir, exist_ok=True)

        frame_idx = pred_frame_idx

        for organ_id in all_organ_ids:
            pred_area = None
            gt_area = None
            pred_n_points = 0
            gt_n_points = 0

            # Process predicted
            if organ_id in pred_organs:
                pts, norms = pred_organs[organ_id]
                pred_n_points = len(pts)
                if pred_n_points >= args.min_points:
                    pred_area, pred_mesh = mesh_leaf_and_compute_area(
                        pts, norms,
                        method=args.method,
                        poisson_depth=args.poisson_depth,
                        density_percentile=args.density_percentile,
                        distance_multiplier=args.distance_multiplier,
                    )
                    if args.save_meshes and pred_mesh is not None:
                        frame_str = f"D{day:.2f}_F{frame_idx:04d}" if frame_idx is not None else f"D{day:.2f}"
                        mesh_path = os.path.join(
                            mesh_pred_dir,
                            f"{frame_str}_leaf{organ_id}.ply",
                        )
                        o3d.io.write_triangle_mesh(mesh_path, pred_mesh)

            # Process GT
            if organ_id in gt_organs:
                pts, norms = gt_organs[organ_id]
                gt_n_points = len(pts)
                if gt_n_points >= args.min_points:
                    gt_area, gt_mesh = mesh_leaf_and_compute_area(
                        pts, norms,
                        method=args.method,
                        poisson_depth=args.poisson_depth,
                        density_percentile=args.density_percentile,
                        distance_multiplier=args.distance_multiplier,
                    )
                    if args.save_meshes and gt_mesh is not None:
                        frame_str = f"D{day:.2f}_F{frame_idx:04d}" if frame_idx is not None else f"D{day:.2f}"
                        split_suffix = f"_{split_tag}" if split_tag else ""
                        mesh_path = os.path.join(
                            mesh_gt_dir,
                            f"{frame_str}_leaf{organ_id}{split_suffix}.ply",
                        )
                        o3d.io.write_triangle_mesh(mesh_path, gt_mesh)

            # Compute error metrics
            area_error = None
            area_ratio = None
            if pred_area is not None and gt_area is not None:
                area_error = abs(pred_area - gt_area)
                if gt_area > 0:
                    area_ratio = pred_area / gt_area

            rows.append({
                "sequence": sequence_name,
                "day": day,
                "frame_idx": frame_idx,
                "split": split_tag,
                "organ_id": organ_id,
                "pred_area": pred_area,
                "gt_area": gt_area,
                "pred_n_points": pred_n_points,
                "gt_n_points": gt_n_points,
                "area_error": area_error,
                "area_ratio": area_ratio,
            })

    return rows

def find_interpolation_dirs(results_dir, sequences=None):
    """Find all sequence interpolation directories.

    Checks these layouts in order:
    0. <results_dir> itself is a sequence dir (has predicted/ subdir)
    1. <results_dir>/<seq>/  (direct interpolation folder with sequence subdirs)
    2. <results_dir>/interpolation/<seq>/  (batch results dir layout)

    Returns:
        list[tuple]: (seq_dir, sequence_name) pairs
    """
    results_dir = Path(results_dir)
    found = []

    # Layout 0: results_dir itself is a sequence dir (e.g. interpolation/maize_control_plant2)
    if (results_dir / "predicted").is_dir():
        seq_name = results_dir.name
        if sequences is None or seq_name in sequences:
            found.append((str(results_dir), seq_name))
        return found

    # Layout 1: <results_dir>/<seq>/ (e.g. results_dir = interpolation/)
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / "predicted").is_dir():
            seq_name = d.name
            if sequences is None or seq_name in sequences:
                found.append((str(d), seq_name))
    if found:
        return found

    # Layout 2: <results_dir>/interpolation/<seq>/
    interp_base = results_dir / "interpolation"
    if interp_base.is_dir():
        for d in sorted(interp_base.iterdir()):
            if d.is_dir() and (d / "predicted").is_dir():
                seq_name = d.name
                if sequences is None or seq_name in sequences:
                    found.append((str(d), seq_name))

    return found

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-leaf surface area from interpolation PLY files"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Batch results directory containing interpolation output"
    )
    parser.add_argument(
        "--sequences", nargs="*", default=None,
        help="Limit to specific sequences (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: <results_dir>/interpolation/leaf_areas.csv)"
    )
    parser.add_argument(
        "--method", type=str, default="poisson",
        choices=["poisson", "ball_pivoting"],
        help="Meshing method (default: poisson)"
    )
    parser.add_argument(
        "--poisson_depth", type=int, default=8,
        help="Poisson reconstruction depth (default: 8)"
    )
    parser.add_argument(
        "--density_percentile", type=float, default=10,
        help="Density trim threshold percentile (default: 10)"
    )
    parser.add_argument(
        "--distance_multiplier", type=float, default=10.0,
        help="Distance trim multiplier (default: 10.0)"
    )
    parser.add_argument(
        "--min_points", type=int, default=20,
        help="Minimum points to attempt meshing (default: 20)"
    )
    parser.add_argument(
        "--save_meshes", action="store_true",
        help="Save individual leaf meshes"
    )
    parser.add_argument(
        "--include_stem", action="store_true",
        help="Include organ_label=0 (stem) — skipped by default"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Find interpolation directories
    seq_dirs = find_interpolation_dirs(args.results_dir, args.sequences)

    if not seq_dirs:
        print(f"No interpolation directories found in {args.results_dir}")
        return 1

    if verbose:
        print(f"Found {len(seq_dirs)} sequence(s) with interpolation data")
        print(f"Meshing method: {args.method}")
        if args.method == "poisson":
            print(f"Poisson depth: {args.poisson_depth}, density percentile: {args.density_percentile}")
            print(f"Distance multiplier: {args.distance_multiplier}")
        print(f"Min points: {args.min_points}")
        print()

    # Process all sequences
    all_rows = []
    for seq_dir, seq_name in seq_dirs:
        if verbose:
            print(f"Processing: {seq_name}")
        rows = process_sequence(seq_dir, seq_name, args)
        if rows:
            all_rows.extend(rows)
            if verbose:
                n_with_area = sum(1 for r in rows if r["pred_area"] is not None)
                print(f"  {len(rows)} organ entries, {n_with_area} with computed pred area")
        else:
            if verbose:
                print(f"  No results")

    if not all_rows:
        print("No leaf areas computed.")
        return 1

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Column order
    col_order = [
        "sequence", "day", "frame_idx", "split", "organ_id",
        "pred_area", "gt_area", "pred_n_points", "gt_n_points",
        "area_error", "area_ratio",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Determine output path
    if args.output is None:
        output_dir = os.path.join(args.results_dir, "interpolation")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"leaf_areas_{args.method}.csv")

    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # Summary statistics
    if verbose:
        print(f"\nSummary:")
        print(f"  Sequences: {df['sequence'].nunique()}")
        print(f"  Days: {df['day'].nunique()}")
        print(f"  Unique organs: {df['organ_id'].nunique()}")

        both_mask = df["pred_area"].notna() & df["gt_area"].notna()
        if both_mask.any():
            subset = df[both_mask]
            print(f"\n  Matched entries (both pred & GT): {both_mask.sum()}")
            print(f"  Mean area error: {subset['area_error'].mean():.6f} +/- {subset['area_error'].std():.6f}")
            print(f"  Mean area ratio (pred/GT): {subset['area_ratio'].mean():.4f} +/- {subset['area_ratio'].std():.4f}")
            print(f"  Mean pred area: {subset['pred_area'].mean():.6f}")
            print(f"  Mean GT area: {subset['gt_area'].mean():.6f}")

        pred_only = df["pred_area"].notna() & df["gt_area"].isna()
        gt_only = df["pred_area"].isna() & df["gt_area"].notna()
        if pred_only.any():
            print(f"  Pred-only entries (no GT match): {pred_only.sum()}")
        if gt_only.any():
            print(f"  GT-only entries (no pred match): {gt_only.sum()}")

    return 0

if __name__ == "__main__":
    exit(main())
