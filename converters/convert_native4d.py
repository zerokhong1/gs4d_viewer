#!/usr/bin/env python3
"""Type B output (native 4D Gaussians, Yang et al. ICLR 2024) -> common .npz for the viewer.

Reads a `chkpnt*.pth` written by fudan-zvg/4d-gaussian-splatting and slices every 4D Gaussian
at K timestamps with the closed form used by that code base:

    mean(t)   = mu_xyz + Sigma_xt / Sigma_tt * (t - mu_t)
    cov3D     = Sigma_xx - Sigma_xt Sigma_tx / Sigma_tt          (constant in t)
    opacity(t)= sigmoid(o) * exp(-0.5 (t - mu_t)^2 / Sigma_tt)

Colour: SH degree-0 term only, unless --sh is given. With --sh the view- and time-dependent colour of the
repo's `eval_shfs_4d` is kept: the checkpoint stores 3D SH coefficients s_j (j < (d+1)^2) plus, for temporal
degree m = 1..dt, extra coefficients s_{m(d+1)^2 + j}. At a keyframe time t the colour of Gaussian i is
    SH_j(t) = s_j + sum_m cos(2 pi m (t_i - t) / T) s_{m(d+1)^2 + j}       (T = t1 - t0)
which is an ordinary 3D SH (the temporal factor is a scalar), so `sh` [K,N,(d+1)^2,3] is written per keyframe
and `colors` holds the degree-0 part at that keyframe.
Needs torch only to unpickle the checkpoint; no GPU. The checkpoint is a pickle,
so only load files you produced yourself.

Example:
    python converters/convert_native4d.py output/N3V/cook_spinach/chkpnt_best.pth -o spinach.npz --frames 60 --t0 0 --t1 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, for `import gs4d`
from gs4d import SH_C0, Sequence4D, Sequence4DWriter, cov_from_quat_scale, pack_cov
from gs4d.checkpoint import torch_load


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


class Native4D:
    """4D Gaussians conditioned on time, one keyframe at a time. All inputs are activated (linear scales, opacity
    in [0,1]). `cov3` [N,6] is constant in t; `means(t)` [N,3] and `opacities(t)` [N] take an optional row selection."""

    def __init__(self, xyz, t, scaling_xyz, scaling_t, rot_l, rot_r, opacity, rot_4d=True, prefilter_var=-1.0):
        self.xyz, self.t = xyz.astype(np.float64), t.astype(np.float64).reshape(-1)
        if rot_4d:
            s4 = np.concatenate([scaling_xyz, scaling_t.reshape(-1, 1)], 1).astype(np.float64)
            L = rotation_4d(rot_l.astype(np.float64), rot_r.astype(np.float64)) * s4[:, None, :]
            S = L @ L.transpose(0, 2, 1)
            cov_xt, var_t = S[:, :3, 3], S[:, 3, 3]
            cov3 = pack_cov(S[:, :3, :3] - cov_xt[:, :, None] * cov_xt[:, None, :] / var_t[:, None, None])
            self.vel = cov_xt / var_t[:, None]
        else:  # axis-aligned in time: no motion, scaling_t is used directly as the temporal variance
            cov3 = cov_from_quat_scale(rot_l, scaling_xyz).astype(np.float64)
            var_t = scaling_t.astype(np.float64).reshape(-1)
            self.vel = np.zeros_like(self.xyz)
        self.cov3 = cov3.astype(np.float32)
        self.sigma = var_t + (prefilter_var if prefilter_var > 0 else 0.0)
        self.opacity = opacity.astype(np.float64).reshape(-1)

    def means(self, time, sel=slice(None)):
        dt = np.float64(time) - self.t[sel]
        return (self.xyz[sel] + self.vel[sel] * dt[:, None]).astype(np.float32)

    def opacities(self, time, sel=slice(None)):
        dt = np.float64(time) - self.t[sel]
        return (self.opacity[sel] * np.exp(-0.5 * dt**2 / self.sigma[sel])).astype(np.float32)


def slice_native4d(xyz, t, scaling_xyz, scaling_t, rot_l, rot_r, opacity, times, rot_4d=True, prefilter_var=-1.0):
    """All inputs are activated (linear scales, opacity in [0,1]).

    Returns means [K,N,3], covs [N,6], opacities [K,N] (the whole sequence in memory; main() goes keyframe by keyframe).
    """
    g = Native4D(xyz, t, scaling_xyz, scaling_t, rot_l, rot_r, opacity, rot_4d, prefilter_var)
    return np.stack([g.means(tk) for tk in times]), g.cov3, np.stack([g.opacities(tk) for tk in times])


def sh_channels(features_dc, features_rest, sh_degree, sh_degree_t):
    """features_dc [N,1,3] + features_rest [N,C-1,3] -> [N,C,3] float64, C = (sh_degree+1)^2 (sh_degree_t+1) checked."""
    f = np.concatenate([features_dc, features_rest], 1).astype(np.float64)
    S = (sh_degree + 1) ** 2
    if f.shape[1] < S * (sh_degree_t + 1):
        raise SystemExit(f"checkpoint has {f.shape[1]} SH channels, fewer than the {S * (sh_degree_t + 1)} "
                         f"needed for degrees ({sh_degree}, {sh_degree_t})")
    return f


def sh_at_time(f, t, time, duration, sh_degree, sh_degree_t):
    """Time-dependent SH folded into plain 3D SH at one keyframe time: [N,(sh_degree+1)^2,3] float32."""
    S = (sh_degree + 1) ** 2
    acc = f[:, :S].copy()
    for m in range(1, sh_degree_t + 1):
        acc += np.cos(2 * np.pi * m * (t - time) / duration)[:, None, None] * f[:, m * S:(m + 1) * S]
    return acc.astype(np.float32)


def sh_at_times(features_dc, features_rest, t, times, duration, sh_degree, sh_degree_t):
    """features_dc [N,1,3], features_rest [N,C-1,3], t [N] -> [K,N,(sh_degree+1)^2,3] float32 (whole sequence in memory)."""
    f = sh_channels(features_dc, features_rest, sh_degree, sh_degree_t)
    t = np.asarray(t, np.float64).reshape(-1)
    return np.stack([sh_at_time(f, t, tk, duration, sh_degree, sh_degree_t) for tk in times])


def load_checkpoint(path: str) -> dict:
    # the repo saves a tuple with Python objects in it, which weights_only=True refuses: only open checkpoints you trust
    obj = torch_load(path, "convert_native4d.py", weights_only=False)
    cap = obj[0] if isinstance(obj, (tuple, list)) and isinstance(obj[0], (tuple, list)) else obj
    if not isinstance(cap, (tuple, list)):
        raise SystemExit(f"{path}: not a native 4DGS checkpoint (expected the tuple torch.save((capture(), iteration)) writes)")
    if len(cap) != 19:
        raise SystemExit(f"{path}: expected the 19-field capture() of a 4D model, got {len(cap)} fields "
                         "(12 fields means a plain 3D model; use convert_per_frame.py on its ply instead)")
    g = lambda i: cap[i].detach().float().cpu().numpy()  # noqa: E731
    return dict(features_rest=g(3), sh_degree=int(cap[0]), sh_degree_t=int(cap[18]),
                xyz=g(1), features_dc=g(2), scaling=g(4), rotation=g(5), opacity=g(6),
                t=g(13), scaling_t=g(14), rotation_r=g(15), rot_4d=bool(cap[16]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--frames", type=int, default=60, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=None, help="start time (time_duration[0] of the training config)")
    ap.add_argument("--t1", type=float, default=None, help="end time (time_duration[1] of the training config)")
    ap.add_argument("--sh", action="store_true",
                    help="keep view- and time-dependent colour as SH per keyframe (gsplat renderer; bigger file)")
    ap.add_argument("--prefilter-var", type=float, default=-1.0, help="same value as in the training config")
    ap.add_argument("--min-peak-opacity", type=float, default=0.02,
                    help="drop Gaussians whose opacity never exceeds this in [t0,t1]")
    ap.add_argument("--max-gaussians", type=int, default=0, help="keep only the N most visible Gaussians (0 = all)")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (lossy, about 2-3x smaller; see docs/FORMAT.md)")
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

    g = Native4D(c["xyz"], c["t"], np.exp(c["scaling"]), np.exp(c["scaling_t"]), c["rotation"], c["rotation_r"],
                 1.0 / (1.0 + np.exp(-c["opacity"])), rot_4d=c["rot_4d"], prefilter_var=a.prefilter_var)
    peak = np.zeros(len(tt), np.float32)  # first pass: which Gaussians are ever visible
    for tk in times:
        np.maximum(peak, g.opacities(tk), out=peak)
    keep = peak > a.min_peak_opacity
    if a.max_gaussians and keep.sum() > a.max_gaussians:
        thresh = np.partition(peak[keep], -a.max_gaussians)[-a.max_gaussians]
        keep &= peak >= thresh
    print(f"Keeping {keep.sum():,} of {len(keep):,} Gaussians")
    const = dict(covs=g.cov3[keep])
    if a.sh:
        f = sh_channels(c["features_dc"][keep], c["features_rest"][keep], c["sh_degree"], c["sh_degree_t"])
        t_keep = tt[keep].astype(np.float64)
    else:
        const["colors"] = np.clip(SH_C0 * c["features_dc"].reshape(len(tt), -1)[:, :3] + 0.5, 0, 1)[keep]
    # second pass: one keyframe at a time to disk, so memory does not grow with --frames
    with Sequence4DWriter(a.output, compact=a.compact, source=f"native 4DGS: {a.checkpoint}", **const) as w:
        for tk in times:
            frame = dict(means=g.means(tk, keep), opacities=g.opacities(tk, keep))
            if a.sh:
                frame["sh"] = sh_at_time(f, t_keep, tk, a.t1 - a.t0, c["sh_degree"], c["sh_degree_t"])
                frame["colors"] = np.clip(SH_C0 * frame["sh"][:, 0] + 0.5, 0, 1)  # DC term changes with time too
            w.add_keyframe(tk, **frame)
    seq = Sequence4D.load(w.path, lazy=True)
    print(f"Wrote {w.path}: {seq.summary()}")


if __name__ == "__main__":
    main()
