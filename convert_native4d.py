#!/usr/bin/env python3
"""Type B output (native 4D Gaussians, Yang et al. ICLR 2024) -> common .npz for the viewer.

Reads a `chkpnt*.pth` written by fudan-zvg/4d-gaussian-splatting and slices every 4D Gaussian
at K timestamps with the closed form used by that code base:

    mean(t)   = mu_xyz + Sigma_xt / Sigma_tt * (t - mu_t)
    cov3D     = Sigma_xx - Sigma_xt Sigma_tx / Sigma_tt          (constant in t)
    opacity(t)= sigmoid(o) * exp(-0.5 (t - mu_t)^2 / Sigma_tt)

Colour uses the SH degree-0 term only (no view- or time-dependent colour).
Needs torch only to unpickle the checkpoint; no GPU. The checkpoint is a pickle,
so only load files you produced yourself.

Example:
    python convert_native4d.py output/N3V/cook_spinach/chkpnt_best.pth -o spinach.npz --frames 60 --t0 0 --t1 10
"""
from __future__ import annotations

import argparse

import numpy as np

from gs4d import SH_C0, Sequence4D, cov_from_quat_scale, pack_cov


def rotation_4d(l: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Port of utils/general_utils.py::build_rotation_4d. l, r: [N,4] -> [N,4,4]."""
    l = l / np.linalg.norm(l, axis=-1, keepdims=True)
    r = r / np.linalg.norm(r, axis=-1, keepdims=True)
    a, b, c, d = l.T
    p, q, rr, s = r.T
    Ml = np.stack([a, -b, -c, -d,
                   b, a, -d, c,
                   c, d, a, -b,
                   d, -c, b, a], -1).reshape(-1, 4, 4)
    Mr = np.stack([p, q, rr, s,
                   -q, p, -s, rr,
                   -rr, s, p, -q,
                   -s, -rr, q, p], -1).reshape(-1, 4, 4)
    return (Ml @ Mr)[:, ::-1, ::-1]


def slice_native4d(xyz, t, scaling_xyz, scaling_t, rot_l, rot_r, opacity, times, rot_4d=True, prefilter_var=-1.0):
    """All inputs are activated (linear scales, opacity in [0,1]).

    Returns means [K,N,3], covs [N,6], opacities [K,N].
    """
    times = np.asarray(times, np.float64)
    xyz, t = xyz.astype(np.float64), t.astype(np.float64).reshape(-1)
    if rot_4d:
        s4 = np.concatenate([scaling_xyz, scaling_t.reshape(-1, 1)], 1).astype(np.float64)
        L = rotation_4d(rot_l.astype(np.float64), rot_r.astype(np.float64)) * s4[:, None, :]
        S = L @ L.transpose(0, 2, 1)
        cov_xt, var_t = S[:, :3, 3], S[:, 3, 3]
        cov3 = pack_cov(S[:, :3, :3] - cov_xt[:, :, None] * cov_xt[:, None, :] / var_t[:, None, None])
        vel = cov_xt / var_t[:, None]
    else:  # axis-aligned in time: no motion, scaling_t is used directly as the temporal variance
        cov3 = cov_from_quat_scale(rot_l, scaling_xyz).astype(np.float64)
        var_t = scaling_t.astype(np.float64).reshape(-1)
        vel = np.zeros_like(xyz)
    dt = times[:, None] - t[None, :]                                   # [K,N]
    means = xyz[None] + vel[None] * dt[..., None]
    sigma = var_t + (prefilter_var if prefilter_var > 0 else 0.0)
    opac = opacity.astype(np.float64).reshape(-1)[None] * np.exp(-0.5 * dt**2 / sigma[None])
    return means.astype(np.float32), cov3.astype(np.float32), opac.astype(np.float32)


def load_checkpoint(path: str) -> dict:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    cap = obj[0] if isinstance(obj, (tuple, list)) and isinstance(obj[0], (tuple, list)) else obj
    if len(cap) != 19:
        raise SystemExit(f"{path}: expected the 19-field capture() of a 4D model, got {len(cap)} fields "
                         "(12 fields means a plain 3D model; use convert_per_frame.py on its ply instead)")
    g = lambda i: cap[i].detach().float().cpu().numpy()  # noqa: E731
    return dict(xyz=g(1), features_dc=g(2), scaling=g(4), rotation=g(5), opacity=g(6),
                t=g(13), scaling_t=g(14), rotation_r=g(15), rot_4d=bool(cap[16]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--frames", type=int, default=60, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=None, help="start time (time_duration[0] of the training config)")
    ap.add_argument("--t1", type=float, default=None, help="end time (time_duration[1] of the training config)")
    ap.add_argument("--prefilter-var", type=float, default=-1.0, help="same value as in the training config")
    ap.add_argument("--min-peak-opacity", type=float, default=0.02,
                    help="drop Gaussians whose opacity never exceeds this in [t0,t1]")
    ap.add_argument("--max-gaussians", type=int, default=0, help="keep only the N most visible Gaussians (0 = all)")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (float16 offsets, about 2.2x smaller, lossy)")
    a = ap.parse_args()

    c = load_checkpoint(a.checkpoint)
    tt = c["t"].reshape(-1)
    if a.t0 is None or a.t1 is None:
        lo, hi = np.percentile(tt, [1, 99])
        a.t0 = float(lo) if a.t0 is None else a.t0
        a.t1 = float(hi) if a.t1 is None else a.t1
        print(f"Time range guessed from the Gaussians' temporal centres: [{a.t0:.3f}, {a.t1:.3f}]. "
              "Pass --t0/--t1 with the training time_duration for the exact range.")
    times = np.linspace(a.t0, a.t1, a.frames, dtype=np.float32)

    means, cov3, opac = slice_native4d(
        c["xyz"], c["t"], np.exp(c["scaling"]), np.exp(c["scaling_t"]), c["rotation"], c["rotation_r"],
        1.0 / (1.0 + np.exp(-c["opacity"])), times, rot_4d=c["rot_4d"], prefilter_var=a.prefilter_var)
    colors = np.clip(SH_C0 * c["features_dc"].reshape(len(tt), -1)[:, :3] + 0.5, 0, 1)

    peak = opac.max(0)
    keep = peak > a.min_peak_opacity
    if a.max_gaussians and keep.sum() > a.max_gaussians:
        thresh = np.partition(peak[keep], -a.max_gaussians)[-a.max_gaussians]
        keep &= peak >= thresh
    print(f"Keeping {keep.sum():,} of {len(keep):,} Gaussians")
    seq = Sequence4D(times, means[:, keep], cov3[keep], opac[:, keep], colors[keep],
                     source=f"native 4DGS: {a.checkpoint}")
    seq.save(a.output, compact=a.compact)
    print(f"Wrote {a.output}: {seq.summary()}")


if __name__ == "__main__":
    main()
