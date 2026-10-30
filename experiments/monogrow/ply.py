"""Binary little-endian PLY read/write with numpy, in the field layout GrowFields' `load_ply` expects:
x y z [nx ny nz] organ_label (int32) is_train (int32) is_leaf_tip (uint8) tip_organ_label (int32)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
          "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
          "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Returns {points (N,3) float64, normals (N,3)|absent, <scalar property> (N,) ...}."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    n, fields = 0, []
    for line in data[:end].decode().splitlines():
        tok = line.split()
        if tok[:2] == ["element", "vertex"]:
            n = int(tok[2])
        elif tok[0] == "property" and n:
            fields.append((tok[2], "<" + _TYPES[tok[1]]))
        elif tok[0] == "element" and n:
            break
    arr = np.frombuffer(data, dtype=np.dtype(fields), count=n, offset=end)
    out = {"points": np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)}
    if "nx" in arr.dtype.names:
        out["normals"] = np.stack([arr["nx"], arr["ny"], arr["nz"]], 1).astype(np.float32)
    for name in arr.dtype.names:
        if name not in ("x", "y", "z", "nx", "ny", "nz"):
            out[name] = np.array(arr[name])
    return out


def write_ply(path: str | Path, points: np.ndarray, organ_label: np.ndarray, is_train: np.ndarray | None = None,
              tips: np.ndarray | None = None, tip_labels: np.ndarray | None = None, normals: np.ndarray | None = None) -> None:
    """Same semantics as GrowFields `save_ply_with_organ_labels`: each tip position is snapped to the nearest
    not-yet-claimed point, which gets is_leaf_tip = 1 and tip_organ_label = its organ id."""
    n = len(points)
    cols = [("x", "f4", points[:, 0]), ("y", "f4", points[:, 1]), ("z", "f4", points[:, 2])]
    if normals is not None:
        cols += [("nx", "f4", normals[:, 0]), ("ny", "f4", normals[:, 1]), ("nz", "f4", normals[:, 2])]
    cols.append(("organ_label", "i4", organ_label))
    if is_train is not None:
        cols.append(("is_train", "i4", is_train))
    if tips is not None and len(tips) > 0:
        tip_mask = np.zeros(n, np.uint8); tip_lab = np.zeros(n, np.int32); claimed = set()
        for i, tip in enumerate(np.asarray(tips).reshape(-1, 3)):
            order = np.argsort(np.linalg.norm(points - tip, axis=1))
            j = next(int(k) for k in order if int(k) not in claimed)
            claimed.add(j); tip_mask[j] = 1
            if tip_labels is not None:
                tip_lab[j] = tip_labels[i]
        cols.append(("is_leaf_tip", "u1", tip_mask))
        if tip_labels is not None:
            cols.append(("tip_organ_label", "i4", tip_lab))
    arr = np.empty(n, dtype=np.dtype([(c[0], "<" + c[1]) for c in cols]))
    for name, _, val in cols:
        arr[name] = val
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {_NAMES[t]} {name}" for name, t, _ in cols]
    header.append("end_header")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode()); f.write(arr.tobytes())
