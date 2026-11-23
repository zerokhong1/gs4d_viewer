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

from gs4d import Sequence4D, Sequence4DWriter, cov_from_quat_scale
from gs4d.ply import gaussians_from_ply


def _natural_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", os.path.basename(s))]


def from_ply_folder(folder: str, stride: int, fps: float, sh: bool = False) -> Sequence4D:
    """Every frame in memory; needed when the Gaussian count changes between frames (padded, interpolation off)."""
    files = _ply_files(folder, stride)
    frames = []
    for i, f in enumerate(files):
        frames.append(gaussians_from_ply(f, with_sh=sh))
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
    sh_all = None
    if sh:
        shapes = {None if fr[4] is None else fr[4].shape[1] for fr in frames}
        if len(shapes) == 1 and None not in shapes:
            sh_all = stack(4, (shapes.pop(), 3))
        else:
            print("  --sh ignored: the ply files have no f_dc_*/f_rest_* or differ in SH degree")
    return Sequence4D(times, stack(0, (3,)), stack(1, (6,)), stack(2, ()), stack(3, (3,)),
                      interpolate=same, source=f"per-frame ply: {os.path.abspath(folder)}", sh=sh_all)


def _ply_files(folder: str, stride: int) -> list:
    """Every `stride`-th .ply of the folder in natural order (time_2 before time_10)."""
    files = sorted(glob.glob(os.path.join(folder, "*.ply")), key=_natural_key)[::stride]
    if not files:
        raise SystemExit(f"No .ply files in {folder}")
    return files


def _ply_header(path: str):
    """(vertex count, property names) from the header only."""
    count, props, in_vertex = None, [], False
    with open(path, "rb") as f:
        for line in f:
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                count = int(tok[2]) if in_vertex else count
            elif tok[0] == "property" and in_vertex:
                props.append(tok[-1])
            elif tok[0] == "end_header":
                break
    return count, tuple(props)


def write_ply_folder(folder: str, output: str, stride: int, fps: float, sh: bool = False, compact: bool = False) -> bool:
    """Same file as `from_ply_folder(...).save(output, compact)`, written one frame at a time when every frame has the
    same Gaussian count. Returns False (nothing written) when the count changes: that mode keeps the old path."""
    files = _ply_files(folder, stride)
    heads = {_ply_header(f) for f in files}
    if len(heads) != 1:
        return False
    times = np.arange(len(files), dtype=np.float32) * stride / fps
    first = gaussians_from_ply(files[0], with_sh=sh)
    use_sh = sh and first[4] is not None
    if sh and not use_sh:
        print("  --sh ignored: the ply files have no f_dc_*/f_rest_* or differ in SH degree")
    with Sequence4DWriter(output, compact=compact, source=f"per-frame ply: {os.path.abspath(folder)}") as wr:
        for i, (t, f) in enumerate(zip(times, files)):
            fr = first if i == 0 else gaussians_from_ply(f, with_sh=use_sh)
            wr.add_keyframe(t, means=fr[0], covs=fr[1], opacities=fr[2], colors=fr[3], sh=fr[4] if use_sh else None)
            print(f"\r  read {i + 1}/{len(files)} ({len(fr[0]):,} Gaussians)", end="", flush=True)
    print()
    print(f"Wrote {wr.path}: {Sequence4D.load(wr.path, lazy=True).summary()}")
    return True


def _load_dynamic3dgs(path: str, stride: int, fg_only: bool):
    """Activated arrays of a Dynamic 3D Gaussians params.npz, every `stride`-th frame: means [T,N,3], quats [T,N,4] (wxyz),
    colours [T,N,3] or [N,3] clipped to [0,1], opacities [N], linear scales [N,3]; with fg_only the foreground only."""
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
    elif fg_only:
        print("  --fg-only ignored: params.npz has no seg_colors")
    return means, quats, np.clip(rgb, 0, 1), opac, scales


def from_dynamic3dgs(path: str, stride: int, fps: float, fg_only: bool) -> Sequence4D:
    means, quats, rgb, opac, scales = _load_dynamic3dgs(path, stride, fg_only)
    covs = cov_from_quat_scale(quats, np.broadcast_to(scales, means.shape))
    times = np.arange(means.shape[0], dtype=np.float32) * stride / fps
    return Sequence4D(times, means, covs, opac, rgb, source=f"Dynamic 3D Gaussians: {os.path.abspath(path)}")


def write_dynamic3dgs(path: str, output: str, stride: int, fps: float, fg_only: bool, compact: bool = False) -> None:
    """Same file as `from_dynamic3dgs(...).save(output, compact)`, covariances computed one frame at a time."""
    means, quats, rgb, opac, scales = _load_dynamic3dgs(path, stride, fg_only)
    times = np.arange(means.shape[0], dtype=np.float32) * stride / fps
    const = dict(opacities=opac, **({} if rgb.ndim == 3 else dict(colors=rgb)))
    with Sequence4DWriter(output, compact=compact, source=f"Dynamic 3D Gaussians: {os.path.abspath(path)}", **const) as wr:
        for k, t in enumerate(times):
            wr.add_keyframe(t, means=means[k], covs=cov_from_quat_scale(quats[k], np.broadcast_to(scales, means[k].shape)),
                            colors=rgb[k] if rgb.ndim == 3 else None)
    print(f"Wrote {wr.path}: {Sequence4D.load(wr.path, lazy=True).summary()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of per-frame .ply files, or a Dynamic 3D Gaussians params.npz")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the source frames (sets the time axis)")
    ap.add_argument("--fg-only", action="store_true", help="Dynamic 3D Gaussians: keep foreground Gaussians only")
    ap.add_argument("--sh", action="store_true",
                    help="ply folder: keep f_rest_* (view-dependent colour, gsplat renderer only); "
                         "adds N x K x 12 x (degree+1)^2 bytes")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (lossy, about 2-3x smaller; see docs/FORMAT.md)")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        if not write_ply_folder(a.input, a.output, a.stride, a.fps, a.sh, a.compact):
            # the Gaussian count changes between frames: padded with invisible Gaussians, kept on the in-memory path
            seq = from_ply_folder(a.input, a.stride, a.fps, a.sh)
            seq.save(a.output, compact=a.compact)
            print(f"Wrote {a.output}: {seq.summary()}")
    elif a.input.endswith(".npz"):
        write_dynamic3dgs(a.input, a.output, a.stride, a.fps, a.fg_only, a.compact)
    else:
        raise SystemExit("input must be a folder of .ply files or a params.npz")


if __name__ == "__main__":
    main()
