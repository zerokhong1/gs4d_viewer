#!/usr/bin/env python3
"""Type A outputs (Gaussians already stored per frame) -> common .npz for the viewer.

Accepted inputs:
  * a folder of standard 3DGS .ply files, one per frame, e.g. the
    `gaussian_pertimestamp/time_00000.ply ...` folder written by
    4DGaussians' export_perframe_3DGS.py, or per-frame output of a feed-forward model;
  * a Dynamic 3D Gaussians (Luiten et al.) `params.npz`.

Examples:
    python convert_per_frame.py output/dnerf/bouncingballs/gaussian_pertimestamp -o balls.npz
    python convert_per_frame.py output/exp1/basketball/params.npz -o basketball.npz --fps 30
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from gs4d import Sequence4D, cov_from_quat_scale
from gs4d.ply import gaussians_from_ply


def _natural_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", os.path.basename(s))]


def from_ply_folder(folder: str, stride: int, fps: float) -> Sequence4D:
    files = sorted(glob.glob(os.path.join(folder, "*.ply")), key=_natural_key)[::stride]
    if not files:
        raise SystemExit(f"No .ply files in {folder}")
    frames = []
    for i, f in enumerate(files):
        frames.append(gaussians_from_ply(f))
        print(f"\r  read {i + 1}/{len(files)} ({frames[-1][0].shape[0]:,} Gaussians)", end="", flush=True)
    print()
    counts = [fr[0].shape[0] for fr in frames]
    same = len(set(counts)) == 1
    N = max(counts)
    if not same:
        print(f"  Gaussian count changes between frames ({min(counts):,}..{N:,}): "
              "padding with invisible Gaussians, interpolation off")

    def stack(idx, tail):
        out = np.zeros((len(frames), N) + tail, np.float32)
        for k, fr in enumerate(frames):
            out[k, : counts[k]] = fr[idx]
        return out

    times = np.arange(len(frames), dtype=np.float32) * stride / fps
    return Sequence4D(times, stack(0, (3,)), stack(1, (6,)), stack(2, ()), stack(3, (3,)),
                      interpolate=same, source=f"per-frame ply: {os.path.abspath(folder)}")


def from_dynamic3dgs(path: str, stride: int, fps: float, fg_only: bool) -> Sequence4D:
    with np.load(path) as z:
        p = {k: z[k] for k in z.files}
    means = p["means3D"][::stride].astype(np.float32)             # [T,N,3]
    quats = p["unnorm_rotations"][::stride].astype(np.float32)    # [T,N,4] wxyz
    rgb = p["rgb_colors"]
    rgb = (rgb[::stride] if rgb.ndim == 3 else rgb).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-p["logit_opacities"].astype(np.float32).reshape(-1)))
    scales = np.exp(p["log_scales"].astype(np.float32))           # [N,3] or [N,1]
    if scales.shape[-1] == 1:
        scales = np.repeat(scales, 3, -1)
    if fg_only and "seg_colors" in p:
        keep = p["seg_colors"][:, 0] > 0.5
        means, quats, opac, scales = means[:, keep], quats[:, keep], opac[keep], scales[keep]
        rgb = rgb[:, keep] if rgb.ndim == 3 else rgb[keep]
    covs = cov_from_quat_scale(quats, np.broadcast_to(scales, means.shape))
    times = np.arange(means.shape[0], dtype=np.float32) * stride / fps
    return Sequence4D(times, means, covs, opac, np.clip(rgb, 0, 1),
                      source=f"Dynamic 3D Gaussians: {os.path.abspath(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of per-frame .ply files, or a Dynamic 3D Gaussians params.npz")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the source frames (sets the time axis)")
    ap.add_argument("--fg-only", action="store_true", help="Dynamic 3D Gaussians: keep foreground Gaussians only")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        seq = from_ply_folder(a.input, a.stride, a.fps)
    elif a.input.endswith(".npz"):
        seq = from_dynamic3dgs(a.input, a.stride, a.fps, a.fg_only)
    else:
        raise SystemExit("input must be a folder of .ply files or a params.npz")
    seq.save(a.output)
    print(f"Wrote {a.output}: {seq.summary()}")


if __name__ == "__main__":
    main()
