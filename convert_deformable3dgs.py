#!/usr/bin/env python3
"""Deformable 3D Gaussians (Yang et al., ingra14m/Deformable-3D-Gaussians) -> common .npz.

Input: the canonical Gaussians `point_cloud/iteration_N/point_cloud.ply` (standard 3DGS layout) and the
deformation MLP `deform/iteration_N/deform.pth` (a state_dict). The MLP is re-implemented here from
utils/time_utils.py (DeformNetwork) and evaluated on CPU at K times; the repo's code is not imported.

At time t (the repo's `fid`, frame / (num_frames - 1) in [0, 1]), for each canonical Gaussian:
    x_emb = [x, sin(2^j x), cos(2^j x)]  j = 0..9         (63 values)
    t_emb = [t, sin(2^j t), cos(2^j t)]  j = 0..9         (21 values; D-NeRF variant: j = 0..5, then
                                                           Linear(13,256)-ReLU-Linear(256,30) "timenet")
    h     = 8 x (Linear + ReLU), width 256; [x_emb, t_emb] is concatenated in front of h after layer 4
    d_xyz = gaussian_warp(h), d_rot = gaussian_rotation(h), d_scale = gaussian_scaling(h)
    6-DoF variant: w = branch_w(h), v = branch_v(h), theta = |w|, screw axis (w/theta + 1e-5, v/theta + 1e-5),
                   position = exp_se3(axis, theta) applied to xyz
and, from gaussian_renderer/__init__.py (render):
    position = xyz + d_xyz
    scale    = exp(scale_*) + d_scale            (added after the activation)
    rotation = normalize(rot_*) + d_rot          (wxyz; NOT normalised again: the rasteriser builds the
                                                  matrix from the quaternion as given, so does this script)
    opacity, colour: unchanged (sigmoid(opacity), SH)
Layer count, width, the D-NeRF ("is_blender") and 6-DoF variants are read from the state_dict.

Colour is SH degree 0; --sh keeps all SH coefficients (constant in time) for the gsplat renderer.

Example:
    python convert_deformable3dgs.py output/scene -o scene.npz --frames 60          # model folder, last iteration
    python convert_deformable3dgs.py point_cloud.ply --deform deform.pth -o scene.npz
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from gs4d import Sequence4D, Sequence4DWriter, pack_cov
from gs4d.checkpoint import import_torch, torch_load
from gs4d.ply import gaussians_from_ply, read_ply_vertices

XYZ_FREQS = 10  # `multires` of the position embedding; fixed in the repo


def find_files(path: str, deform: str | None, iteration: int):
    """(ply, deform.pth) from a model folder, or from an explicit ply (+ --deform)."""
    if os.path.isfile(path):
        if deform is None:  # <model>/point_cloud/iteration_N/point_cloud.ply -> <model>/deform/iteration_N/deform.pth
            it = os.path.basename(os.path.dirname(os.path.abspath(path)))
            deform = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "deform", it, "deform.pth")
        if not os.path.isfile(deform):
            raise SystemExit(f"deformation weights not found: {deform} (pass --deform)")
        return path, deform

    def pick(sub):
        root = os.path.join(path, sub)
        its = [int(m.group(1)) for d in (os.listdir(root) if os.path.isdir(root) else [])
               if (m := re.fullmatch(r"iteration_(\d+)", d))]
        if not its:
            raise SystemExit(f"{root}: no iteration_* folder; not a Deformable-3DGS model folder")
        if iteration >= 0 and iteration not in its:
            raise SystemExit(f"{root}: no iteration_{iteration} (have {sorted(its)})")
        return os.path.join(root, f"iteration_{iteration if iteration >= 0 else max(its)}")

    return os.path.join(pick("point_cloud"), "point_cloud.ply"), deform or os.path.join(pick("deform"), "deform.pth")


def embed(x, n_freqs: int):
    """[x, sin(x), cos(x), sin(2x), cos(2x), ..., sin(2^(n-1) x), cos(2^(n-1) x)] along the last axis."""
    import torch

    out = [x]
    for j in range(n_freqs):
        out += [torch.sin(x * 2.0 ** j), torch.cos(x * 2.0 ** j)]
    return torch.cat(out, -1)


class DeformMLP:
    """Forward pass of the repo's DeformNetwork, straight from its state_dict."""

    def __init__(self, sd: dict):
        self.sd = {k: v.float() for k, v in sd.items()}
        self.depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("linear."))
        self.skip = self.depth // 2
        self.blender = "timenet.0.weight" in sd
        self.six_dof = "branch_w.weight" in sd
        xyz_ch = 3 + 6 * XYZ_FREQS
        t_ch = sd["timenet.0.weight"].shape[1] if self.blender else sd["linear.0.weight"].shape[1] - xyz_ch
        if t_ch < 1 or (t_ch - 1) % 2:
            raise SystemExit(f"deform.pth: unexpected input width {tuple(sd['linear.0.weight'].shape)}; not a DeformNetwork")
        self.t_freqs = (t_ch - 1) // 2

    def _lin(self, name, h):
        import torch.nn.functional as F

        return F.linear(h, self.sd[f"{name}.weight"], self.sd[f"{name}.bias"])

    def __call__(self, xyz, t: float):
        """xyz [M,3] -> (d_xyz [M,3] or None, transform [M,4,4] or None, d_rot [M,4], d_scale [M,3])."""
        import torch

        t_emb = embed(torch.full((len(xyz), 1), float(t)), self.t_freqs)
        if self.blender:
            t_emb = self._lin("timenet.2", torch.relu(self._lin("timenet.0", t_emb)))
        inp = torch.cat([embed(xyz, XYZ_FREQS), t_emb], -1)
        h = inp
        for i in range(self.depth):
            h = torch.relu(self._lin(f"linear.{i}", h))
            if i == self.skip:
                h = torch.cat([inp, h], -1)
        d_rot, d_scale = self._lin("gaussian_rotation", h), self._lin("gaussian_scaling", h)
        if not self.six_dof:
            return self._lin("gaussian_warp", h), None, d_rot, d_scale
        w, v = self._lin("branch_w", h), self._lin("branch_v", h)
        theta = w.norm(dim=-1, keepdim=True)
        return None, exp_se3(w / theta + 1e-5, v / theta + 1e-5, theta), d_rot, d_scale


def exp_se3(w, v, theta):
    """Screw axis (w, v) [M,3] each and magnitude theta [M,1] -> homogeneous transforms [M,4,4]
    (Modern Robotics eq. 3.88; the repo's utils/rigid_utils.py)."""
    import torch

    M = len(w)
    W = torch.zeros(M, 3, 3)
    W[:, 0, 1], W[:, 0, 2], W[:, 1, 0] = -w[:, 2], w[:, 1], w[:, 2]
    W[:, 1, 2], W[:, 2, 0], W[:, 2, 1] = -w[:, 0], -w[:, 1], w[:, 0]
    W2, eye, th = W @ W, torch.eye(3).expand(M, 3, 3), theta.view(M, 1, 1)
    T = torch.zeros(M, 4, 4)
    T[:, :3, :3] = eye + torch.sin(th) * W + (1 - torch.cos(th)) * W2
    T[:, :3, 3] = ((th * eye + (1 - torch.cos(th)) * W + (th - torch.sin(th)) * W2) @ v[:, :, None])[:, :, 0]
    T[:, 3, 3] = 1.0
    return T


def cov_unnormalised(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Packed covariance [M,6] from wxyz quaternions used as given (no normalisation) and scales [M,3]."""
    w, x, y, z = (q[:, i] for i in range(4))
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    Mx = R * s[:, None, :]
    return pack_cov(Mx @ np.swapaxes(Mx, -1, -2))


def deform_frames(xyz, scales, quats, mlp: DeformMLP, times, batch: int = 65536, progress: bool = False):
    """Canonical xyz [N,3], activated scales [N,3], unit quaternions [N,4] -> one (means [N,3], covs [N,6]) per time."""
    import torch

    K, N = len(times), len(xyz)
    with torch.no_grad():
        for k, t in enumerate(times):
            means = np.empty((N, 3), np.float32)
            covs = np.empty((N, 6), np.float32)
            for a in range(0, N, batch):
                sl = slice(a, min(a + batch, N))
                x = torch.from_numpy(xyz[sl])
                d_xyz, T, d_rot, d_scale = mlp(x, float(t))
                if T is None:
                    means[sl] = (x + d_xyz).numpy()
                else:
                    hom = T @ torch.cat([x, torch.ones(len(x), 1)], -1)[:, :, None]
                    means[sl] = (hom[:, :3, 0] / hom[:, 3:, 0]).numpy()
                covs[sl] = cov_unnormalised(quats[sl] + d_rot.numpy(), scales[sl] + d_scale.numpy())
            if progress:
                print(f"\r  time {k + 1}/{K}", end="\n" if k + 1 == K else "", flush=True)
            yield means, covs


def deform_sequence(xyz, scales, quats, mlp: DeformMLP, times, batch: int = 65536, progress: bool = False):
    """Same as deform_frames, the whole sequence in memory: means [K,N,3], covs [K,N,6]."""
    frames = list(deform_frames(xyz, scales, quats, mlp, times, batch, progress))
    return np.stack([f[0] for f in frames]), np.stack([f[1] for f in frames])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model folder (with point_cloud/ and deform/), or a point_cloud.ply")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--deform", default=None, help="deform.pth, when it is not at its usual place in the model folder")
    ap.add_argument("--iteration", type=int, default=-1, help="model folder: iteration to use (default: the last)")
    ap.add_argument("--frames", type=int, default=60, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=0.0, help="start time (the repo's times are frame/(num_frames-1) in [0,1])")
    ap.add_argument("--t1", type=float, default=1.0)
    ap.add_argument("--min-opacity", type=float, default=0.0, help="drop Gaussians with opacity at or below this")
    ap.add_argument("--batch", type=int, default=65536, help="Gaussians per MLP batch (lower it if RAM runs out)")
    ap.add_argument("--sh", action="store_true", help="keep all SH coefficients for view-dependent colour (gsplat renderer)")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (lossy, about 2-3x smaller; see docs/FORMAT.md)")
    a = ap.parse_args()

    import_torch("convert_deformable3dgs.py")
    ply, pth = find_files(a.model, a.deform, a.iteration)
    sd = torch_load(pth, "convert_deformable3dgs.py")
    if not isinstance(sd, dict) or not any(k.startswith("linear.") for k in sd):
        raise SystemExit(f"{pth}: no linear.* weights; not a Deformable-3DGS deform.pth")
    mlp = DeformMLP(sd)
    print(f"{ply}\n{pth}: {mlp.depth} layers" + (", D-NeRF time net" if mlp.blender else "") + (", 6-DoF" if mlp.six_dof else ""))

    v = read_ply_vertices(ply)
    _, _, opac, colors, sh = gaussians_from_ply(ply, with_sh=True)
    keep = opac > a.min_opacity
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)[keep]
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float32))[keep]
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)[keep]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    print(f"Keeping {keep.sum():,} of {len(keep):,} Gaussians")

    times = np.linspace(a.t0, a.t1, a.frames, dtype=np.float32)
    # one keyframe at a time to disk, so memory does not grow with --frames
    with Sequence4DWriter(a.output, compact=a.compact, source=f"Deformable 3D Gaussians: {ply}", opacities=opac[keep],
                          colors=colors[keep], sh=sh[keep] if (a.sh and sh is not None) else None) as w:
        for t, (means, covs) in zip(times, deform_frames(xyz, scales, quats, mlp, times, a.batch, progress=True)):
            w.add_keyframe(t, means=means, covs=covs)
    print(f"Wrote {w.path}: {Sequence4D.load(w.path, lazy=True).summary()}")


if __name__ == "__main__":
    main()
