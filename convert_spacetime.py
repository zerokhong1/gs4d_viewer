#!/usr/bin/env python3
"""Spacetime Gaussians (Li et al., oppo-us-research/SpacetimeGaussians) point_cloud.ply -> common .npz.

Formulas ported from thirdparty/gaussian_splatting/renderer/__init__.py (train_ours_full) and
scene/oursfull.py (save_ply), at time t (the repo's camera timestamp, normally frame/num_frames in [0,1]):
    dt        = t - trbf_center
    position  = xyz + m0*dt + m1*dt^2 + m2*dt^3            (motion_0..8 = m0(xyz) m1(xyz) m2(xyz))
    opacity   = sigmoid(opacity) * exp(-(dt / exp(trbf_scale))^2)
    rotation  = normalize(rot + dt * omega)                 (rot_0..3 and omega_0..3, wxyz, before normalising)
    scale     = exp(scale_0..2)                             (constant in time)
    colour    = f_dc_0..2 used directly as RGB (NOT SH0); clipped to [0,1]

Known limit: the real renderer feeds the rasterised features (f_dc_3..8, f_t_* * dt) through a small MLP
(`rgbdecoder`, the .pt next to the ply) that adds view- and time-dependent colour. That is not reproduced:
only the base RGB is used, so colours can look flatter or slightly off compared with the paper's renders.

Example:
    python convert_spacetime.py log/scene/point_cloud/iteration_25000/point_cloud.ply -o scene.npz --frames 50
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from gs4d import Sequence4D, cov_from_quat_scale
from gs4d.ply import read_ply_vertices


FULL_MODEL = "Spacetime Gaussians full model (rgbdecoder)"


def full_model_reason(path: str, names) -> str | None:
    """Why this ply belongs to the full model (colour from the `rgbdecoder` MLP), or None for a lite model."""
    if "f_dc_3" in names:
        return "the ply has f_dc_3.. (features the rgbdecoder MLP consumes)"
    here = os.path.dirname(os.path.abspath(path))
    for d in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if os.path.isdir(d) and any("rgbdecoder" in f.lower() for f in os.listdir(d)):
            return f"an rgbdecoder file is next to the ply ({d})"
    return None


def load_spacetime_ply(path: str, base_color_only: bool = False) -> dict:
    v = read_ply_vertices(path)
    names = v.dtype.names
    for n in ("x", "trbf_center", "trbf_scale", "opacity", "scale_0", "rot_0", "f_dc_0"):
        if n not in names:
            raise SystemExit(f"{path}: no '{n}' property; not a Spacetime Gaussians ply "
                             "(a plain 3DGS ply goes through convert_per_frame.py)")

    why = full_model_reason(path, names)
    if why and not base_color_only:
        raise SystemExit(f"{path}: {FULL_MODEL}: {why}. Its colour is view- and time-dependent and is not reproduced; "
                         "only the base colour f_dc_0..2 would be used. Run again with --base-color-only to convert anyway.")

    def cols(prefix, n, default=0.0):
        if f"{prefix}0" not in names:
            return np.full((len(v), n), default, np.float64)
        return np.stack([v[f"{prefix}{i}"] for i in range(n)], -1).astype(np.float64)

    return dict(xyz=np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64),
                center=v["trbf_center"].astype(np.float64), tscale=v["trbf_scale"].astype(np.float64),
                motion=cols("motion_", 9), opacity=v["opacity"].astype(np.float64),
                scale=cols("scale_", 3), rot=cols("rot_", 4), omega=cols("omega_", 4),
                rgb=cols("f_dc_", 3))


def slice_spacetime(g: dict, times: np.ndarray):
    """Gaussians at each time -> means [K,N,3], covs [K,N,6], opacities [K,N]."""
    K, N = len(times), len(g["xyz"])
    means = np.empty((K, N, 3), np.float32)
    covs = np.empty((K, N, 6), np.float32)
    opac = np.empty((K, N), np.float32)
    m = g["motion"]
    scales = np.exp(g["scale"])
    base = 1.0 / (1.0 + np.exp(-g["opacity"]))
    for k, t in enumerate(times):
        dt = t - g["center"]
        d = dt[:, None]
        means[k] = g["xyz"] + m[:, 0:3] * d + m[:, 3:6] * d**2 + m[:, 6:9] * d**3
        covs[k] = cov_from_quat_scale((g["rot"] + d * g["omega"]).astype(np.float32), scales.astype(np.float32))
        opac[k] = base * np.exp(-((dt / np.exp(g["tscale"])) ** 2))
    return means, covs, opac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--frames", type=int, default=50, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=0.0, help="start time (repo timestamps are frame/num_frames)")
    ap.add_argument("--t1", type=float, default=1.0)
    ap.add_argument("--min-peak-opacity", type=float, default=0.02,
                    help="drop Gaussians whose opacity never exceeds this at the keyframes")
    ap.add_argument("--base-color-only", action="store_true",
                    help=f"convert a {FULL_MODEL} with its base colour f_dc_0..2 only (without this flag it stops)")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (float16 offsets, about 2.2x smaller, lossy)")
    a = ap.parse_args()

    g = load_spacetime_ply(a.ply, a.base_color_only)
    times = np.linspace(a.t0, a.t1, a.frames, dtype=np.float64)
    means, covs, opac = slice_spacetime(g, times)
    keep = opac.max(0) > a.min_peak_opacity
    print(f"Keeping {keep.sum():,} of {len(keep):,} Gaussians")
    colors = np.clip(g["rgb"], 0, 1).astype(np.float32)
    seq = Sequence4D(times.astype(np.float32), means[:, keep], covs[:, keep], opac[:, keep], colors[keep],
                     source=f"Spacetime Gaussians (base colour only): {a.ply}")
    seq.save(a.output, compact=a.compact)
    print(f"Wrote {a.output}: {seq.summary()}")


if __name__ == "__main__":
    main()
