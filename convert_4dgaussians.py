#!/usr/bin/env python3
"""4D Gaussian Splatting (Wu et al., hustvl/4DGaussians) model folder -> common .npz.

Input: `point_cloud/iteration_N/point_cloud.ply` (canonical Gaussians, standard 3DGS layout) and
`point_cloud/iteration_N/deformation.pth` (state_dict of `deform_network`), plus `cfg_args` for the flags.
The network is re-implemented here from scene/deformation.py and scene/hexplane.py, evaluated on CPU at K times
(the repo's code is not imported), so `export_perframe_3DGS.py` need not be run in the repo's environment.

At time t (the repo's camera time, frame/(num_frames-1) in [0,1]), for each canonical Gaussian with raw (stored)
position x, log-scale s, rotation q, opacity logit o (all as in the ply):
    p      = (x - aabb[0]) * 2 / (aabb[1] - aabb[0]) - 1                 aabb = grid.aabb (max corner, min corner)
    f      = concat over resolution levels L of  prod over the 6 planes (a,b) of bilinear_sample(grid[L][ab], (p,t)[a,b])
             (planes in itertools.combinations(range(4), 2) order: xy xz xt yz yt zt; border padding, align_corners)
    h      = Linear(Linear-ReLU)^(D-1) applied to f   (feature_out)
    x'     = x + pos_deform(h)           pos_deform = ReLU-Linear-ReLU-Linear     (unless no_dx)
    s'     = s + scales_deform(h)                                                  (unless no_ds; log space)
    q'     = q + rotations_deform(h)     (apply_rotation: quaternion product q * dr)  (unless no_dr; not normalised)
    o'     = o + opacity_deform(h)                                                 (only when no_do is False)
and the ply is read like the repo's `export_perframe_3DGS.py` output (scale -> exp, opacity -> sigmoid, rotation
normalised when the covariance is built). The mask of the repo is 1 for the supported variants.

Not supported (the script stops and names the variant): no_grid, empty_voxel, static_mlp, grid_pe > 1,
and no_dshs=False (SH colours that change in time).
Colour is SH degree 0; --sh keeps all SH coefficients (constant in time) for the gsplat renderer.

Example:
    python convert_4dgaussians.py output/dnerf/bouncingballs -o balls.npz --frames 60
    python convert_4dgaussians.py point_cloud.ply --deform deformation.pth -o balls.npz
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import re

import numpy as np

from gs4d import Sequence4D, Sequence4DWriter, cov_from_quat_scale
from gs4d.ply import gaussians_from_ply, read_ply_vertices

_FLAGS = ("no_dx", "no_ds", "no_dr", "no_do", "no_dshs", "no_grid", "empty_voxel", "static_mlp", "apply_rotation")
_DEFAULT_FLAGS = dict(no_dx=False, no_ds=False, no_dr=False, no_do=True, no_dshs=True, no_grid=False,
                      empty_voxel=False, static_mlp=False, apply_rotation=False)


def find_files(model: str, deform: str | None, iteration: int):
    """Returns (ply, deformation.pth, cfg_args or None)."""
    if os.path.isfile(model):
        ply = model
    else:
        its = {}
        for p in glob.glob(os.path.join(model, "point_cloud", "iteration_*", "point_cloud.ply")):
            m = re.search(r"iteration_(\d+)", p)
            if m:
                its[int(m.group(1))] = p
        if not its:
            raise SystemExit(f"{model}: no point_cloud/iteration_*/point_cloud.ply")
        ply = its[max(its) if iteration < 0 else iteration]
    pth = deform or os.path.join(os.path.dirname(ply), "deformation.pth")
    if not os.path.exists(pth):
        raise SystemExit(f"{pth} not found; pass --deform")
    cfg = None
    d = os.path.dirname(ply)
    for _ in range(4):  # point_cloud/iteration_N -> model folder
        if os.path.exists(os.path.join(d, "cfg_args")):
            cfg = os.path.join(d, "cfg_args")
            break
        d = os.path.dirname(d)
    return ply, pth, cfg


def read_flags(cfg: str | None) -> dict:
    flags = dict(_DEFAULT_FLAGS)
    if cfg is None:
        print("  no cfg_args next to the model: assuming the default flags of the repo "
              "(no_do=True, no_dshs=True, the others False); pass --flag name=True/False to override")
        return flags
    text = open(cfg).read()
    for k in _FLAGS:
        m = re.search(rf"\b{k}=(True|False)", text)
        if m:
            flags[k] = m.group(1) == "True"
    return flags


class HexPlaneDeform:
    """Port of HexPlaneField + Deformation.forward_dynamic for the supported variants. sd: deformation.pth state_dict."""

    def __init__(self, sd: dict, flags: dict):
        import torch

        self.torch = torch
        self.flags = flags
        bad = [k for k in ("no_grid", "empty_voxel", "static_mlp") if flags[k]]
        if not flags["no_dshs"]:
            bad.append("no_dshs=False (time-varying SH colour)")
        if any(k.startswith("deformation_net.static_mlp") or k.startswith("deformation_net.empty_voxel") for k in sd):
            bad.append("static_mlp/empty_voxel weights in the checkpoint")
        if bad:
            raise SystemExit("4DGaussians variant not supported: " + ", ".join(bad))
        pre = "deformation_net."
        self.aabb = sd[pre + "grid.aabb"].float()
        self.grids = []  # [level][plane] -> [1,C,H,W]
        L = 0
        while any(k.startswith(f"{pre}grid.grids.{L}.") for k in sd):
            P = 0
            planes = []
            while f"{pre}grid.grids.{L}.{P}" in sd:
                planes.append(sd[f"{pre}grid.grids.{L}.{P}"].float())
                P += 1
            self.grids.append(planes)
            L += 1
        if not self.grids:
            raise SystemExit("deformation.pth has no hexplane grids: not a 4DGaussians deform_network")
        self.combos = list(itertools.combinations(range(4), 2))
        if len(self.grids[0]) != len(self.combos):
            raise SystemExit(f"expected {len(self.combos)} planes per level, found {len(self.grids[0])}")
        self.feature_out = []
        i = 0
        while f"{pre}feature_out.{i}.weight" in sd:
            self.feature_out.append((sd[f"{pre}feature_out.{i}.weight"].float(), sd[f"{pre}feature_out.{i}.bias"].float()))
            i += 2
        feat = sum(g[0].shape[1] for g in self.grids)
        if self.feature_out[0][0].shape[1] != feat:
            raise SystemExit(f"feature_out expects {self.feature_out[0][0].shape[1]} inputs but the grid gives {feat} "
                             "(grid_pe > 1 is not supported)")

        def head(name):
            return [(sd[f"{pre}{name}.{j}.weight"].float(), sd[f"{pre}{name}.{j}.bias"].float()) for j in (1, 3)]

        self.heads = {k: head(n) for k, n in (("dx", "pos_deform"), ("ds", "scales_deform"),
                                              ("dr", "rotations_deform"), ("do", "opacity_deform"))}
        self.depth, self.width = len(self.feature_out), self.feature_out[0][0].shape[0]

    def _features(self, x, t):
        F = self.torch.nn.functional
        torch = self.torch
        p = (x - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1.0
        pts = torch.cat([p, t], -1)  # [n,4]
        out = []
        for planes in self.grids:
            acc = 1.0
            for g, (a, b) in zip(planes, self.combos):
                coords = pts[:, [a, b]].view(1, 1, -1, 2)
                s = F.grid_sample(g, coords, mode="bilinear", padding_mode="border", align_corners=True)
                acc = acc * s.view(g.shape[1], -1).T
            out.append(acc)
        return torch.cat(out, -1)

    @staticmethod
    def _mlp2(h, layers):
        (w1, b1), (w2, b2) = layers
        h = (h.relu() @ w1.T + b1).relu()
        return h @ w2.T + b2

    def __call__(self, xyz, log_scale, rot, opac_logit, t: float):
        """Raw (stored) values [n,...] at time t -> deformed raw positions, log-scales, rotations, opacity logits."""
        torch = self.torch
        f = self.flags
        with torch.no_grad():
            x, s, q, o = (torch.as_tensor(a, dtype=torch.float32) for a in (xyz, log_scale, rot, opac_logit))
            h = self._features(x, torch.full((len(x), 1), float(t)))
            for i, (w, b) in enumerate(self.feature_out):
                h = (h if i == 0 else h.relu()) @ w.T + b
            if not f["no_dx"]:
                x = x + self._mlp2(h, self.heads["dx"])
            if not f["no_ds"]:
                s = s + self._mlp2(h, self.heads["ds"])
            if not f["no_dr"]:
                dr = self._mlp2(h, self.heads["dr"])
                if f["apply_rotation"]:
                    a, b = q, dr
                    q = torch.stack([a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3],
                                     a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2],
                                     a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1],
                                     a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]], -1)
                else:
                    q = q + dr
            if not f["no_do"]:
                o = o + self._mlp2(h, self.heads["do"])
        return x.numpy(), s.numpy(), q.numpy(), o.numpy()


def deform_frames(xyz, log_scale, rot, opac_logit, net, times, batch=65536, progress=False):
    """One time at a time -> (means [N,3], covs [N,6], opacities [N], sigmoid applied)."""
    N = len(xyz)
    for k, t in enumerate(times):
        means = np.empty((N, 3), np.float32)
        covs = np.empty((N, 6), np.float32)
        opac = np.empty(N, np.float32)
        for i in range(0, N, batch):
            sl = slice(i, i + batch)
            x, s, q, o = net(xyz[sl], log_scale[sl], rot[sl], opac_logit[sl], float(t))
            means[sl] = x
            covs[sl] = cov_from_quat_scale(q, np.exp(s))
            opac[sl] = 1.0 / (1.0 + np.exp(-o.reshape(-1)))
        if progress:
            print(f"\r  time {k + 1}/{len(times)}", end="", flush=True)
        yield means, covs, opac
    if progress:
        print()


def deform_sequence(xyz, log_scale, rot, opac_logit, net, times, batch=65536, progress=False):
    """Same as deform_frames, the whole sequence in memory: means [K,N,3], covs [K,N,6], opacities [K,N]."""
    fr = list(deform_frames(xyz, log_scale, rot, opac_logit, net, times, batch, progress))
    return np.stack([f[0] for f in fr]), np.stack([f[1] for f in fr]), np.stack([f[2] for f in fr])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model folder (with point_cloud/), or a point_cloud.ply")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--deform", default=None, help="deformation.pth, when it is not next to the ply")
    ap.add_argument("--iteration", type=int, default=-1, help="model folder: iteration to use (default: the last)")
    ap.add_argument("--frames", type=int, default=60, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=0.0, help="start time (the repo's times are frame/(num_frames-1) in [0,1])")
    ap.add_argument("--t1", type=float, default=1.0)
    ap.add_argument("--min-opacity", type=float, default=0.0, help="drop Gaussians with opacity at or below this")
    ap.add_argument("--flag", action="append", default=[], metavar="NAME=BOOL",
                    help="override a flag read from cfg_args, e.g. --flag no_do=False")
    ap.add_argument("--batch", type=int, default=65536, help="Gaussians per network batch (lower it if RAM runs out)")
    ap.add_argument("--sh", action="store_true", help="keep all SH coefficients for view-dependent colour (gsplat renderer)")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (float16 offsets, about 2.2x smaller, lossy)")
    a = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("convert_4dgaussians.py needs torch (CPU is enough): pip install torch")
    ply, pth, cfg = find_files(a.model, a.deform, a.iteration)
    flags = read_flags(cfg)
    for kv in a.flag:
        k, v = kv.split("=")
        if k not in _FLAGS:
            raise SystemExit(f"unknown flag {k}; known: {', '.join(_FLAGS)}")
        flags[k] = v.lower() in ("1", "true", "yes")
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    net = HexPlaneDeform(sd, flags)
    print(f"{ply}\n{pth}: {len(net.grids)} grid levels, MLP depth {net.depth} width {net.width}; flags "
          + " ".join(f"{k}={flags[k]}" for k in _FLAGS if flags[k] != _DEFAULT_FLAGS[k] or k in ("no_do",)))

    v = read_ply_vertices(ply)
    _, _, opac0, colors, sh = gaussians_from_ply(ply, with_sh=True)
    keep = opac0 > a.min_opacity
    g = lambda names: np.stack([v[n] for n in names], -1).astype(np.float32)[keep]  # noqa: E731
    xyz, log_scale = g(["x", "y", "z"]), g([f"scale_{i}" for i in range(3)])
    rot, op = g([f"rot_{i}" for i in range(4)]), v["opacity"].astype(np.float32)[keep].reshape(-1, 1)
    print(f"Keeping {keep.sum():,} of {len(keep):,} Gaussians")

    times = np.linspace(a.t0, a.t1, a.frames, dtype=np.float32)
    const = dict(colors=colors[keep], sh=sh[keep] if (a.sh and sh is not None) else None)
    frames = deform_frames(xyz, log_scale, rot, op, net, times, a.batch, progress=True)
    # one keyframe at a time to disk, so memory does not grow with --frames
    if flags["no_do"]:
        first = next(frames)
        const["opacities"] = first[2]
        frames = itertools.chain([first], frames)
    with Sequence4DWriter(a.output, compact=a.compact, source=f"4DGaussians: {ply}", **const) as wr:
        for t, (means, covs, opac) in zip(times, frames):
            wr.add_keyframe(t, means=means, covs=covs, opacities=None if flags["no_do"] else opac)
    print(f"Wrote {wr.path}: {Sequence4D.load(wr.path, lazy=True).summary()}")

if __name__ == "__main__":
    main()
