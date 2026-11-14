#!/usr/bin/env python3
"""SC-GS (Huang et al., yihua7/SC-GS) -> common .npz.

Input: a model folder with `point_cloud/iteration_N/point_cloud.ply` (3DGS layout plus per-Gaussian
`fea_*` columns) and `deform/iteration_N/deform.pth` (state_dict of ControlNodeWarp: control nodes + MLP).
The deformation is re-implemented here from utils/time_utils.py and gaussian_renderer/__init__.py and
run on CPU; the repo's code is not imported.

Control nodes: `nodes` [M, 3+H] (position + H "hyper" coordinates), `_node_radius` [M] (log),
`_node_weight` [M,1] (logit, optional). Per Gaussian, once (time independent):
    p      = [xyz, fea_0..H-1]                             (the Gaussian's own hyper coordinates)
    K nearest nodes by squared distance d_k = |p - node_k|^2            (K = 3 unless --knn)
    w_k    = exp(-d_k / (2 exp(_node_radius_k)^2)) * sigmoid(_node_weight_k) + 1e-7,  normalised over k
At time t (the repo's `fid` in [0, 1]) the MLP is evaluated at the M node positions (same network as
Deformable-3DGS: position/time embedding, 8 x 256 ReLU layers with one skip; D-NeRF variant with `timenet`)
and gives per node d_xyz, d_rotation, d_scaling (and local_rotation, d_opacity when trained with them):
    without --local_frame:  shift = sum_k w_k d_xyz_k
    with    --local_frame:  shift = sum_k w_k (R_k (xyz - node_k) + node_k + d_xyz_k) - xyz,
                            R_k = matrix of the quaternion local_rotation_k + (1,0,0,0)
    position = xyz + mask * shift
    rotation = normalize(rot + mask * sum_k w_k d_rotation_k)             (wxyz; d_rot_as_res, the default)
    scale    = exp(scale) + mask * sum_k w_k d_scaling_k
    opacity  = sigmoid(opacity) [+ mask * sum_k w_k d_opacity_k]
mask = sigmoid(last fea column) when the model was trained with --gs_with_motion_mask (the ply then has H+1
fea columns), otherwise 1. Colour is SH degree 0; --sh keeps all SH coefficients (constant in time).

Not supported (the script stops and says so): hash-grid networks (--use_hash), --skinning, and models
trained with d_rot_as_res off. A colour head (--pred_color) is ignored. --max_d_scale must be repeated here.

Example:
    python convert_scgs.py outputs/jumpingjacks_node -o scene.npz --frames 60
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from gs4d import Sequence4D, cov_from_quat_scale
from gs4d.ply import gaussians_from_ply, read_ply_vertices

XYZ_FREQS = 10  # `multires` of the position embedding; fixed in the repo


def find_files(path: str, deform: str | None, iteration: int):
    """(ply, deform.pth) from a model folder, or from an explicit ply (+ --deform)."""
    if os.path.isfile(path):
        if deform is None:
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
            raise SystemExit(f"{root}: no iteration_* folder; not an SC-GS model folder")
        if iteration >= 0 and iteration not in its:
            raise SystemExit(f"{root}: no iteration_{iteration} (have {sorted(its)})")
        return os.path.join(root, f"iteration_{iteration if iteration >= 0 else max(its)}")

    return os.path.join(pick("point_cloud"), "point_cloud.ply"), deform or os.path.join(pick("deform"), "deform.pth")


def embed(x, n_freqs: int, include_input: bool = True):
    """[x, sin(x), cos(x), sin(2x), cos(2x), ..., sin(2^(n-1) x), cos(2^(n-1) x)] along the last axis."""
    import torch

    out = [x] if include_input else []
    for j in range(n_freqs):
        out += [torch.sin(x * 2.0 ** j), torch.cos(x * 2.0 ** j)]
    return torch.cat(out, -1)


class NodeMLP:
    """Forward pass of the repo's DeformNetwork (the `network.*` entries of deform.pth) at the control nodes."""

    def __init__(self, sd: dict, max_d_scale: float = -1.0):
        if "network.linear.0.weight" not in sd:
            raise SystemExit("deform.pth has no network.linear.*: hash-grid (--use_hash) or static models are not supported")
        self.sd = {k[len("network."):]: v.detach().float() for k, v in sd.items() if k.startswith("network.") and hasattr(v, "detach")}
        self.depth = 1 + max(int(k.split(".")[1]) for k in self.sd if k.startswith("linear."))
        self.skip = self.depth // 2
        self.blender = "timenet.0.weight" in self.sd
        self.progressive = "embed_time_fn.cur_step" in self.sd  # ProgressiveBandFrequency: no raw t, mask is all ones after training
        self.local_frame = "local_rotation.weight" in self.sd
        self.pred_opacity = "gaussian_opacity.weight" in self.sd
        self.pred_color = any(k.startswith("gaussian_color") for k in self.sd)
        self.max_d_scale = max_d_scale
        xyz_ch = 3 + 6 * XYZ_FREQS
        t_ch = self.sd["timenet.0.weight"].shape[1] if self.blender else self.sd["linear.0.weight"].shape[1] - xyz_ch
        raw = 0 if self.progressive else 1
        if t_ch < 1 or (t_ch - raw) % 2:
            raise SystemExit(f"deform.pth: unexpected input width {tuple(self.sd['linear.0.weight'].shape)}; not an SC-GS DeformNetwork")
        self.t_freqs = (t_ch - raw) // 2

    def _lin(self, name, h):
        import torch.nn.functional as F

        return F.linear(h, self.sd[f"{name}.weight"], self.sd[f"{name}.bias"])

    def __call__(self, xyz, t: float) -> dict:
        import torch

        t_emb = embed(torch.full((len(xyz), 1), float(t)), self.t_freqs, include_input=not self.progressive)
        if self.blender:
            t_emb = self._lin("timenet.2", torch.relu(self._lin("timenet.0", t_emb)))
        inp = torch.cat([embed(xyz, XYZ_FREQS), t_emb], -1)
        h = inp
        for i in range(self.depth):
            h = torch.relu(self._lin(f"linear.{i}", h))
            if i == self.skip:
                h = torch.cat([inp, h], -1)
        d_scale = self._lin("gaussian_scaling", h)
        if self.max_d_scale > 0:
            d_scale = torch.tanh(d_scale) * float(np.log(self.max_d_scale))
        out = {"d_xyz": self._lin("gaussian_warp", h), "d_rot": self._lin("gaussian_rotation", h), "d_scale": d_scale}
        if self.local_frame:
            out["local_rot"] = self._lin("local_rotation", h)
        if self.pred_opacity:
            out["d_opacity"] = self._lin("gaussian_opacity", h)
        return {k: v.numpy().astype(np.float64) for k, v in out.items()}


def knn_weights(points: np.ndarray, sd: dict, K: int, batch: int = 20000):
    """Indices [N,K] and normalised weights [N,K] of each Gaussian's K nearest control nodes."""
    nodes = sd["nodes"].astype(np.float64)
    radius = np.exp(sd["_node_radius"].astype(np.float64))
    node_w = 1.0 / (1.0 + np.exp(-sd["_node_weight"].astype(np.float64).reshape(-1))) if "_node_weight" in sd else None
    K = min(K, len(nodes))
    idx = np.empty((len(points), K), np.int64)
    w = np.empty((len(points), K), np.float64)
    n2 = (nodes ** 2).sum(-1)
    for a in range(0, len(points), batch):
        p = points[a:a + batch].astype(np.float64)
        d = np.maximum((p ** 2).sum(-1)[:, None] - 2.0 * p @ nodes.T + n2[None], 0.0)  # squared distances [n,M]
        nn = np.argpartition(d, K - 1, axis=1)[:, :K]
        dk = np.take_along_axis(d, nn, 1)
        order = np.argsort(dk, axis=1)
        nn, dk = np.take_along_axis(nn, order, 1), np.take_along_axis(dk, order, 1)
        wk = np.exp(-dk / (2.0 * radius[nn] ** 2))
        if node_w is not None:
            wk = wk * node_w[nn]
        wk = wk + 1e-7
        idx[a:a + batch], w[a:a + batch] = nn, wk / wk.sum(-1, keepdims=True)
    return idx, w


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices [...,3,3] of wxyz quaternions of any length (the repo's quaternion_to_matrix)."""
    r, i, j, k = np.moveaxis(q, -1, 0)
    s = 2.0 / (q * q).sum(-1)
    return np.stack([1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r),
                     s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r),
                     s * (i * k - j * r), s * (j * k + i * r), 1 - s * (i * i + j * j)], -1).reshape(q.shape[:-1] + (3, 3))


def deform_sequence(xyz, scales, rots, opac, mask, idx, w, nodes_xyz, mlp: NodeMLP, times, progress: bool = False):
    """Canonical xyz [N,3], activated scales [N,3], raw quaternions [N,4], opacities [N], motion mask [N,1]
    -> means [K,N,3], covs [K,N,6], opacities [K,N] or None when the model has no opacity head."""
    import torch

    K, N = len(times), len(xyz)
    means = np.empty((K, N, 3), np.float32)
    covs = np.empty((K, N, 6), np.float32)
    opacs = np.empty((K, N), np.float32) if mlp.pred_opacity else None
    node_t = torch.from_numpy(nodes_xyz.astype(np.float32))
    x = xyz.astype(np.float64)
    blend = lambda v: (v[idx] * w[..., None]).sum(1)  # noqa: E731  node values [M,C] -> per Gaussian [N,C]
    with torch.no_grad():
        for k, t in enumerate(times):
            v = mlp(node_t, float(t))
            if mlp.local_frame:
                lr = v["local_rot"].copy()
                lr[:, 0] += 1.0
                R = quat_matrix(lr)[idx]                                        # [N,K,3,3]
                nn_nodes = nodes_xyz[idx].astype(np.float64)                    # [N,K,3]
                Ax = np.einsum("nkab,nkb->nka", R, x[:, None] - nn_nodes) + nn_nodes + v["d_xyz"][idx]
                shift = (Ax * w[..., None]).sum(1) - x
            else:
                shift = blend(v["d_xyz"])
            means[k] = x + shift * mask
            covs[k] = cov_from_quat_scale((rots + blend(v["d_rot"]) * mask).astype(np.float32),
                                          (scales + blend(v["d_scale"]) * mask).astype(np.float32))
            if opacs is not None:
                opacs[k] = np.clip(opac + (blend(v["d_opacity"]) * mask)[:, 0], 0.0, 1.0)
            if progress:
                print(f"\r  time {k + 1}/{K}", end="", flush=True)
    if progress:
        print()
    return means, covs, opacs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model folder (with point_cloud/ and deform/), or a point_cloud.ply")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--deform", default=None, help="deform.pth, when it is not at its usual place in the model folder")
    ap.add_argument("--iteration", type=int, default=-1, help="model folder: iteration to use (default: the last)")
    ap.add_argument("--frames", type=int, default=60, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=0.0, help="start time (the repo's times are in [0,1])")
    ap.add_argument("--t1", type=float, default=1.0)
    ap.add_argument("--knn", type=int, default=3, help="nodes per Gaussian; the repo's --K used for training")
    ap.add_argument("--max-d-scale", type=float, default=-1.0, help="the repo's --max_d_scale used for training, if any")
    ap.add_argument("--min-opacity", type=float, default=0.0, help="drop Gaussians with opacity at or below this")
    ap.add_argument("--sh", action="store_true", help="keep all SH coefficients for view-dependent colour (gsplat renderer)")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (float16 offsets, about 2.2x smaller, lossy)")
    a = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("convert_scgs.py needs torch (CPU is enough): pip install torch")
    ply, pth = find_files(a.model, a.deform, a.iteration)
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    if "nodes" not in sd:
        raise SystemExit(f"{pth}: no control nodes. An SC-GS model trained with --deform_type mlp is a Deformable-3DGS "
                         "network: use convert_deformable3dgs.py")
    if "_node_radius" not in sd:
        raise SystemExit(f"{pth}: no _node_radius: models trained with --skinning are not supported")
    mlp = NodeMLP(sd, a.max_d_scale)
    node_sd = {k: sd[k].detach().numpy() for k in ("nodes", "_node_radius", "_node_weight") if k in sd}
    H = node_sd["nodes"].shape[1] - 3

    v = read_ply_vertices(ply)
    _, _, opac, colors, sh = gaussians_from_ply(ply, with_sh=True)
    n_fea = sum(1 for n in v.dtype.names if n.startswith("fea_"))
    if n_fea not in (H, H + 1):
        raise SystemExit(f"{ply} has {n_fea} fea_* columns but the nodes have {H} hyper coordinates: ply and deform.pth do not match")
    fea = np.stack([v[f"fea_{i}"] for i in range(n_fea)], -1).astype(np.float64) if n_fea else np.zeros((len(v), 0))
    keep = opac > a.min_opacity
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)[keep]
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float64))[keep]
    rots = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)[keep]
    fea = fea[keep]
    mask = 1.0 / (1.0 + np.exp(-fea[:, -1:])) if n_fea == H + 1 else np.ones((len(xyz), 1))
    print(f"{ply}\n{pth}: {len(node_sd['nodes']):,} control nodes, {H} hyper coordinates"
          + (", D-NeRF time net" if mlp.blender else "") + (", local frame" if mlp.local_frame else "")
          + (", motion mask" if n_fea == H + 1 else "") + (", opacity head" if mlp.pred_opacity else ""))
    if mlp.pred_color:
        print("warning: the model has a colour head (--pred_color); it is ignored, colours stay constant in time")
    print(f"Keeping {keep.sum():,} of {len(keep):,} Gaussians")

    idx, w = knn_weights(np.concatenate([xyz, fea[:, :H]], 1), node_sd, a.knn)
    times = np.linspace(a.t0, a.t1, a.frames, dtype=np.float32)
    means, covs, opacs = deform_sequence(xyz, scales, rots, opac[keep], mask, idx, w, node_sd["nodes"][:, :3], mlp, times, progress=True)
    seq = Sequence4D(times, means, covs, opac[keep] if opacs is None else opacs, colors[keep],
                     sh=sh[keep] if (a.sh and sh is not None) else None, source=f"SC-GS: {ply}")
    seq.save(a.output, compact=a.compact)
    print(f"Wrote {a.output}: {seq.summary()}")


if __name__ == "__main__":
    main()
