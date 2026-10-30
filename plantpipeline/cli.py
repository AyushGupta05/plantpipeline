"""Command line: plantpipeline {inventory,process,stats,view,unify} ..."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .pheno4d import list_scans
from .process import process


def main() -> None:
    p = argparse.ArgumentParser(prog="plantpipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inventory", help="list scans in Pheno4D.zip or an extracted folder")
    a.add_argument("raw", type=Path)

    b = sub.add_parser("process", help="raw -> soil-free, fixed-size npz + manifest.csv")
    b.add_argument("raw", type=Path)
    b.add_argument("--out", type=Path, default=Path("data/processed"))
    b.add_argument("--n-points", type=int, default=8192)
    b.add_argument("--plants", nargs="*", help="e.g. M01 T01 (default: all)")

    c = sub.add_parser("stats", help="summarise manifest.csv")
    c.add_argument("--out", type=Path, default=Path("data/processed"))

    d = sub.add_parser("view", help="open a plant in an Open3D window (n/p = next/previous day), or --png DIR to render images")
    d.add_argument("plant", help="e.g. M01 or T03; for unified data pass --out data/unified/<dataset> and a sequence_id")
    d.add_argument("--out", type=Path, default=Path("data/processed"), help="data/processed or data/unified/<dataset>")
    d.add_argument("--png", type=Path, help="write one PNG per day to this folder instead of opening a window")

    e = sub.add_parser("unify", help="Pheno4D (processed) + TrackPlant3D v2 + Sugar4D -> data/unified (one format, see sources/README.md)")
    e.add_argument("--raw", type=Path, default=Path("data/raw"), help="folder with TrackPlant3D_v2.zip and sugar4d.zip")
    e.add_argument("--processed", type=Path, default=Path("data/processed"), help="output of `process` (Pheno4D)")
    e.add_argument("--out", type=Path, default=Path("data/unified"))
    e.add_argument("--n-points", type=int, default=8192, help="FPS target for dense sources (Sugar4D)")
    e.add_argument("--datasets", nargs="*", default=["pheno4d", "trackplant3d", "sugar4d"])

    args = p.parse_args()
    if args.cmd == "unify":
        from .sources.unify import summary, unify
        w = unify(args.raw, args.processed, args.out, args.datasets, args.n_points)
        print(summary(w.rows, w.sequences))
        return
    if args.cmd == "view":
        from .view import render_png, show
        if args.png:
            for path in render_png(args.out, args.plant, args.png):
                print(path)
        else:
            show(args.out, args.plant)
        return
    if args.cmd == "inventory":
        scans = list_scans(args.raw)
        for s in scans:
            print(s.scan_id, s.species, "labelled" if s.labelled else "-", s.source)
        print(f"{len(scans)} scans, {sum(s.labelled for s in scans)} labelled, {len({s.plant for s in scans})} plants")
    elif args.cmd == "process":
        rows = process(args.raw, args.out, args.n_points, args.plants)
        print(f"wrote {len(rows)} scans to {args.out}")
    elif args.cmd == "stats":
        with open(args.out / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        print(f"{len(rows)} scans | splits {dict(Counter(r['split'] for r in rows))} | soil {dict(Counter(r['soil_method'] for r in rows))}")
        for r in rows:
            print(f"{r['scan_id']}  day {r['day']:>2}  {r['soil_method']:8s}  plant pts {int(r['n_plant']):>9,}  height {r['height_mm']:>6} mm  leaves {r['n_leaves']}")


if __name__ == "__main__":
    main()
