"""Minimal reader for the standard 3DGS point_cloud.ply (binary little-endian or ascii)."""
from __future__ import annotations

import numpy as np

from .format import SH_C0, cov_from_quat_scale

_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
          "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
          "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


def read_ply_vertices(path: str) -> np.ndarray:
    """Returns the vertex element as a structured array."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, count, props, in_vertex = None, None, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    count = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError(f"{path}: list properties on vertices are not supported")
                props.append((tok[2], _TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        if count is None:
            raise ValueError(f"{path}: no vertex element")
        if fmt == "binary_little_endian":
            return np.fromfile(f, dtype=np.dtype([(n, "<" + t) for n, t in props]), count=count)
        if fmt == "ascii":
            raw = np.loadtxt(f, max_rows=count, ndmin=2)
            out = np.empty(count, dtype=np.dtype(props))
            for i, (n, _) in enumerate(props):
                out[n] = raw[:, i]
            return out
        raise ValueError(f"{path}: unsupported PLY format {fmt}")


def gaussians_from_ply(path: str, with_sh: bool = False):
    """Standard 3DGS ply -> (means [N,3], covs [N,6], opacities [N], colors [N,3]).

    Applies the usual activations: exp on scale_*, sigmoid on opacity, SH degree 0 for colour.
    With with_sh=True a fifth item is returned: the SH coefficients [N,S,3] (f_dc_* then f_rest_*,
    which the 3DGS ply stores channel-major: f_rest_{c*(S-1)+j}), or None when the ply has no f_dc_*.
    """
    v = read_ply_vertices(path)
    names = v.dtype.names
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32))
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    if "f_dc_0" in names:
        dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1).astype(np.float32)
        colors = np.clip(SH_C0 * dc + 0.5, 0.0, 1.0)
    elif "red" in names:
        colors = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.float32) / 255.0
    else:
        colors = np.full((len(v), 3), 0.7, np.float32)
    covs = cov_from_quat_scale(quats, scales)
    if not with_sh:
        return means, covs, opac, colors
    sh = None
    if "f_dc_0" in names:
        n_rest = sum(1 for n in names if n.startswith("f_rest_"))
        S = n_rest // 3 + 1
        if n_rest % 3 or S not in (1, 4, 9, 16):
            raise ValueError(f"{path}: {n_rest} f_rest_* properties is not a full SH degree")
        sh = np.empty((len(v), S, 3), np.float32)
        sh[:, 0] = dc
        for c in range(3):
            for j in range(S - 1):
                sh[:, j + 1, c] = v[f"f_rest_{c * (S - 1) + j}"]
    return means, covs, opac, colors, sh
