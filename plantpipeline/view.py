"""Look at processed scans.

    plantpipeline view M01            interactive window; keys: n / p = next / previous day, q = quit
    plantpipeline view M01 --png out  write one PNG per day instead (offscreen render)
    plantpipeline view maize_control_plant2 --out data/unified/trackplant3d   a unified sequence

Colours: stem grey, each leaf instance its own colour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

PALETTE = np.array([
    [0.55, 0.55, 0.55],  # 1 stem
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16], [0.58, 0.40, 0.74],
    [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.20, 0.20, 0.60],
])


def load_days(processed: Path, plant: str) -> list[tuple[str, o3d.geometry.PointCloud]]:
    out = []
    files = sorted((processed / plant).glob("*.npz"), key=lambda f: float(np.load(f)["day"]))  # processed or unified sequence folder
    for f in files:
        d = np.load(f)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(d["points"].astype(np.float64)))
        lab = d["labels"].astype(int)
        colours = PALETTE[np.where(lab <= 1, 0, 1 + (lab - 2) % (len(PALETTE) - 1))]
        pc.colors = o3d.utility.Vector3dVector(colours)
        out.append((f"{f.stem}  day {int(d['day'])}  {len(lab):,} pts", pc))
    return out


def show(processed: Path, plant: str) -> None:
    days = load_days(processed, plant)
    i = 0
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=days[i][0], width=1100, height=800)
    vis.add_geometry(days[i][1])
    vis.get_render_option().point_size = 2.0

    def step(delta):
        def cb(v):
            nonlocal i
            vis.remove_geometry(days[i][1], reset_bounding_box=False)
            i = (i + delta) % len(days)
            vis.add_geometry(days[i][1], reset_bounding_box=False)
            print(days[i][0])
            return False
        return cb

    vis.register_key_callback(ord("N"), step(1))
    vis.register_key_callback(ord("P"), step(-1))
    print(days[i][0], "| n = next day, p = previous day, q = quit")
    vis.run()
    vis.destroy_window()


def render_png(processed: Path, plant: str, out: Path) -> list[Path]:
    """One PNG per day (matplotlib), same axes for every day so growth is visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = load_days(processed, plant)
    out.mkdir(parents=True, exist_ok=True)
    all_pts = np.vstack([np.asarray(pc.points) for _, pc in days])
    lo, hi = all_pts.min(0), all_pts.max(0)
    half = float(np.max(hi - lo)) / 2
    centre = (lo + hi) / 2
    paths = []
    for title, pc in days:
        pts, col = np.asarray(pc.points), np.asarray(pc.colors)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=1.5, linewidths=0)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(lo[2], lo[2] + 2 * half)
        ax.set_title(title)
        ax.set_xlabel("x mm"), ax.set_ylabel("y mm"), ax.set_zlabel("z mm")
        ax.view_init(elev=20, azim=-60)
        p = out / f"{title.split()[0]}.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)
    return paths
