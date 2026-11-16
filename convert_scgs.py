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
Trained with --pred_color, the MLP also gives per node d_color [3] (a 3-layer colour head on [input, hidden], or one
linear layer on the hidden state), added to the Gaussian's SH DC coefficient:
    f_dc(t) = f_dc + mask * sum_k w_k d_color_k
so colours (and, with --sh, the DC coefficient) are stored per keyframe.

Skinning models (--skinning, no `_node_radius`, nodes without hyper coordinates): the ply has one `fea_*` column per node
(plus the motion mask, when trained with one) and every Gaussian is blended over ALL nodes with
    w = softmax(fea_0..M-1)
instead of the K nearest; everything else is as above (the repo's cal_nn_weight with `skinning`).

Hash-grid models (--use_hash): the hash grid (12 levels, tiny-cuda-nn layout) is re-implemented in numpy (`hash_encode`) and
the MLP heads run in numpy; the repo's DeformNetwork output `d_rotation` already contains the (1,0,0,0) offset there.
--hash_time (4D grid) is read too; the fully fused tiny-cuda-nn MLP (tcnn_mlp) is not.

Not supported (the script stops and says so): models
trained with d_rot_as_res off, colour heads other than the default (hash colour, colour by view direction).
--ignore-pred-color keeps the colour of a --pred_color model constant in time (smaller file, wrong colours).
--max_d_scale must be repeated here.

Example:
    python convert_scgs.py outputs/jumpingjacks_node -o scene.npz --frames 60
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from gs4d import Sequence4D, Sequence4DWriter, cov_from_quat_scale
from gs4d.format import SH_C0
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
        # colour head: "resnet" = Linear, ReLU, Linear, ReLU, Linear on [position emb, time emb, hidden] (the repo's default),
        # "linear" = one Linear on the hidden state; anything else (hash colour, colour by view direction) is not supported
        self.color_kind = None
        if self.pred_color:
            w0 = self.sd.get("gaussian_color.0.weight")
            hidden = self.sd["gaussian_warp.weight"].shape[1]
            if w0 is not None and "gaussian_color.4.weight" in self.sd and w0.shape[1] == self.sd["linear.0.weight"].shape[1] + hidden:
                self.color_kind = "resnet"
            elif "gaussian_color.weight" in self.sd and self.sd["gaussian_color.weight"].shape[1] == hidden:
                self.color_kind = "linear"
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
        if self.color_kind == "resnet":
            c = torch.relu(self._lin("gaussian_color.0", torch.cat([inp, h], -1)))
            out["d_color"] = self._lin("gaussian_color.4", torch.relu(self._lin("gaussian_color.2", c)))
        elif self.color_kind == "linear":
            out["d_color"] = self._lin("gaussian_color", h)
        return {k: v.numpy().astype(np.float64) for k, v in out.items()}


# ------------------------------------------------------------------------------------------ hash-grid network (--use_hash)
HASH_LEVELS, HASH_FEATURES, HASH_LOG2_SIZE, HASH_BASE, HASH_SCALE = 12, 2, 19, 16, 2.0   # fixed in the repo (ProgressiveBandHashGridCosine)
HASH_START_LEVEL, HASH_UPDATE_STEPS = 6, 1000
HASH_PRIMES = (1, 2654435761, 805459861, 3674653429)   # tiny-cuda-nn hash factors (checked against the built library, tests/test_scgs_hash.py)


def hash_levels(dims: int):
    """Per level (scale, resolution, parameter offset, table size) as tiny-cuda-nn lays the grid out."""
    table, offset, out = 1 << HASH_LOG2_SIZE, 0, []
    for lv in range(HASH_LEVELS):
        scale = np.float32(np.exp2(np.float32(lv) * np.float32(np.log2(HASH_SCALE))) * HASH_BASE - 1.0)
        res = int(np.ceil(scale)) + 1
        n = res ** dims
        n = ((n + 7) // 8) * 8          # aligned memory accesses
        n = min(n, table)
        out.append((scale, res, offset, n))
        offset += n
    return out, offset


def hash_encode(x: np.ndarray, params: np.ndarray, dims: int) -> np.ndarray:
    """tiny-cuda-nn GridEncoding (type Hash, linear interpolation) of x [n, dims] in [0,1] -> [n, levels * features] float32."""
    x = np.ascontiguousarray(x, np.float32)
    n = len(x)
    levels, total = hash_levels(dims)
    if params.size != total * HASH_FEATURES:
        raise SystemExit(f"hash grid has {params.size} parameters, expected {total * HASH_FEATURES} for {HASH_LEVELS} levels")
    tab = params.reshape(-1, HASH_FEATURES).astype(np.float32)
    out = np.zeros((n, HASH_LEVELS * HASH_FEATURES), np.float32)
    for lv, (scale, res, off, size) in enumerate(levels):
        pos = x * scale + np.float32(0.5)
        grid = np.floor(pos)
        frac = (pos - grid).astype(np.float32)
        grid = grid.astype(np.int64)
        acc = np.zeros((n, HASH_FEATURES), np.float32)
        for corner in range(1 << dims):
            w = np.ones(n, np.float32)
            g = grid.copy()
            for d in range(dims):
                if (corner >> d) & 1:
                    w = w * frac[:, d]
                    g[:, d] += 1
                else:
                    w = w * (np.float32(1.0) - frac[:, d])
            stride, idx = 1, np.zeros(n, np.uint64)
            for d in range(dims):
                if stride > size:
                    break
                idx += g[:, d].astype(np.uint64) * np.uint64(stride)
                stride *= res
            if stride > size:   # the level does not fit its table: hash the integer position
                h = np.zeros(n, np.uint32)
                for d in range(dims):
                    h ^= (g[:, d].astype(np.uint32) * np.uint32(HASH_PRIMES[d]))
                idx = h.astype(np.uint64)
            idx = (idx % np.uint64(size)).astype(np.int64)
            acc += w[:, None] * tab[off + idx]
        out[:, lv * HASH_FEATURES:(lv + 1) * HASH_FEATURES] = acc
    return out


class HashNodeMLP:
    """Forward pass of the repo's HashDeformNetwork (`network.hashgrid.*`, `network.mlp.*`, `network.*_layer.*`) at the control nodes."""

    def __init__(self, sd: dict):
        self.sd = {k[len("network."):]: v.detach().float().numpy() for k, v in sd.items()
                   if k.startswith("network.") and hasattr(v, "detach")}
        if "mlp.layers.0.weight" not in self.sd:
            raise SystemExit("deform.pth: hash-grid network with the fully fused tiny-cuda-nn MLP (tcnn_mlp) is not supported")
        self.params = self.sd["hashgrid.encoding.params"]
        self.bbox = self.sd.get("bbox", np.array([-2.0, 2.0], np.float32))
        n_in = self.sd["mlp.layers.0.weight"].shape[1]
        out_dim = HASH_LEVELS * HASH_FEATURES
        self.hash_time = n_in == out_dim                       # 4D grid on (x, t), no separate time embedding
        self.t_freqs = 0 if self.hash_time else (n_in - out_dim - 1) // 2
        self.grid_dims = 4 if self.hash_time else 3
        self.blender = False
        self.progressive = False
        self.local_frame = "local_rotation_layer.layers.0.weight" in self.sd
        self.pred_opacity = "opacity_layer.layers.0.weight" in self.sd
        self.pred_color = "color_layer.layers.0.weight" in self.sd
        self.color_kind = "hash" if self.pred_color else None
        step = int(np.asarray(sd.get("network.hashgrid.current_step", 0)).reshape(-1)[0]) if "network.hashgrid.current_step" in sd else 10**9
        ratio = step / ((HASH_LEVELS - HASH_START_LEVEL) * HASH_UPDATE_STEPS)
        start = HASH_START_LEVEL * HASH_FEATURES
        band = out_dim - start
        self.mask = np.ones(out_dim, np.float32)
        self.mask[start:] = (1.0 - np.cos(np.pi * np.clip(ratio * band - np.arange(band), 0.0, 1.0))) / 2.0
        self.skip = 2 if not self.hash_time else None         # MLP skip=[2] re-injects the input before layer 6
        self.log_scale_range = float(np.log(2.0))

    def _mlp(self, name, x, skip_in=None):
        idx = sorted(int(k.split(".")[2]) for k in self.sd if k.startswith(name + ".layers.") and k.endswith(".weight"))
        h = x
        for n, i in enumerate(idx):                           # the repo's MLP: Linear at even indices, ReLU between
            if skip_in is not None and i == 6:                # skip=[2]: the input is concatenated before layer 6
                h = np.concatenate([h, skip_in], -1)
            h = h @ self.sd[f"{name}.layers.{i}.weight"].T + self.sd[f"{name}.layers.{i}.bias"]
            if n < len(idx) - 1:
                h = np.maximum(h, 0.0)
        return h

    def __call__(self, xyz, t: float) -> dict:
        x = (np.asarray(xyz, np.float32) - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
        if self.hash_time:
            embed = hash_encode(np.concatenate([x, np.full((len(x), 1), t, np.float32)], 1), self.params, 4) * self.mask
            skip_in = None
        else:
            te = np.concatenate([[t]] + [[np.sin(t * 2.0 ** j), np.cos(t * 2.0 ** j)] for j in range(self.t_freqs)]).astype(np.float32)
            te = np.tile(te / len(te) * HASH_LEVELS * HASH_FEATURES, (len(x), 1))
            embed = np.concatenate([hash_encode(x, self.params, 3) * self.mask, te], 1)
            skip_in = embed
        h = self._mlp("mlp", embed, skip_in)
        out = {"d_xyz": self._mlp("translate_layer", h),
               "d_rot": self._mlp("rotation_layer", h) + np.array([1.0, 0, 0, 0], np.float32),
               "d_scale": np.tanh(self._mlp("scaling_layer", h)) * self.log_scale_range}
        if self.local_frame:
            out["local_rot"] = self._mlp("local_rotation_layer", h)
        if self.pred_opacity:
            out["d_opacity"] = self._mlp("opacity_layer", h)
        if self.pred_color:
            out["d_color"] = self._mlp("color_layer", h)
        return {k: v.astype(np.float64) for k, v in out.items()}


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


SKIN_CHUNK = 20000


def deform_frames(xyz, scales, rots, opac, mask, idx, w, nodes_xyz, mlp: NodeMLP, times, progress: bool = False,
                  with_color: bool = False):
    """Canonical xyz [N,3], activated scales [N,3], raw quaternions [N,4], opacities [N], motion mask [N,1]
    -> per time one (means [N,3], covs [N,6], opacities [N] or None when the model has no opacity head).
    with_color: a fourth item, the offset [N,3] of the SH DC coefficient (None when the model has no colour head)."""
    import torch

    K = len(times)
    node_t = torch.from_numpy(nodes_xyz.astype(np.float32))
    x = xyz.astype(np.float64)
    blend = lambda v: (v[idx] * w[..., None]).sum(1)  # noqa: E731  node values [M,C] -> per Gaussian [N,C]
    skin = idx is None  # skinning: w is [N,M], softmax weights over all nodes
    if skin:
        def blend(v):  # noqa: F811
            return np.concatenate([w[a:a + SKIN_CHUNK] @ v for a in range(0, len(w), SKIN_CHUNK)])
    with torch.no_grad():
        for k, t in enumerate(times):
            v = mlp(node_t, float(t))
            if mlp.local_frame and skin:
                # sum_k w_k (R_k (x - n_k) + n_k + d_k) = (sum_k w_k R_k) x + sum_k w_k (n_k - R_k n_k + d_k)
                lr = v["local_rot"].copy()
                lr[:, 0] += 1.0
                R = quat_matrix(lr)                                             # [M,3,3]
                nodes64 = nodes_xyz.astype(np.float64)
                off = nodes64 - np.einsum("kab,kb->ka", R, nodes64) + v["d_xyz"]
                Rw = blend(R.reshape(len(R), 9)).reshape(-1, 3, 3)
                shift = np.einsum("nab,nb->na", Rw, x) + blend(off) - x
            elif mlp.local_frame:
                lr = v["local_rot"].copy()
                lr[:, 0] += 1.0
                R = quat_matrix(lr)[idx]                                        # [N,K,3,3]
                nn_nodes = nodes_xyz[idx].astype(np.float64)                    # [N,K,3]
                Ax = np.einsum("nkab,nkb->nka", R, x[:, None] - nn_nodes) + nn_nodes + v["d_xyz"][idx]
                shift = (Ax * w[..., None]).sum(1) - x
            else:
                shift = blend(v["d_xyz"])
            means = (x + shift * mask).astype(np.float32)
            covs = cov_from_quat_scale((rots + blend(v["d_rot"]) * mask).astype(np.float32),
                                       (scales + blend(v["d_scale"]) * mask).astype(np.float32))
            opacs = None
            if mlp.pred_opacity:
                opacs = np.clip(opac + (blend(v["d_opacity"]) * mask)[:, 0], 0.0, 1.0).astype(np.float32)
            if progress:
                print(f"\r  time {k + 1}/{K}", end="\n" if k + 1 == K else "", flush=True)
            if with_color:
                yield means, covs, opacs, (blend(v["d_color"]) * mask).astype(np.float32) if "d_color" in v else None
            else:
                yield means, covs, opacs


def deform_sequence(xyz, scales, rots, opac, mask, idx, w, nodes_xyz, mlp: NodeMLP, times, progress: bool = False):
    """Same as deform_frames, the whole sequence in memory: means [K,N,3], covs [K,N,6], opacities [K,N] or None."""
    frames = list(deform_frames(xyz, scales, rots, opac, mask, idx, w, nodes_xyz, mlp, times, progress))
    return (np.stack([f[0] for f in frames]), np.stack([f[1] for f in frames]),
            np.stack([f[2] for f in frames]) if mlp.pred_opacity else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model folder (with point_cloud/ and deform/), or a point_cloud.ply")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--deform", default=None, help="deform.pth, when it is not at its usual place in the model folder")
    ap.add_argument("--ignore-pred-color", action="store_true",
                    help="model trained with --pred_color: keep its colour constant in time (smaller file, colours are then wrong)")
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
    skinning = "_node_radius" not in sd
    if skinning and sd["nodes"].shape[1] != 3:
        raise SystemExit(f"{pth}: no _node_radius but the nodes have hyper coordinates: not an SC-GS control-node model")
    mlp = HashNodeMLP(sd) if "network.hashgrid.encoding.params" in sd else NodeMLP(sd, a.max_d_scale)
    node_sd = {k: sd[k].detach().numpy() for k in ("nodes", "_node_radius", "_node_weight") if k in sd}
    H = node_sd["nodes"].shape[1] - 3
    M = len(node_sd["nodes"])

    v = read_ply_vertices(ply)
    _, _, opac, colors, sh = gaussians_from_ply(ply, with_sh=True)
    n_fea = sum(1 for n in v.dtype.names if n.startswith("fea_"))
    if skinning:  # one fea column per node, then the motion mask if any
        H = M
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
          + (", skinning (softmax over all nodes)" if skinning else "") + (", motion mask" if n_fea == H + 1 else "") + (", opacity head" if mlp.pred_opacity else ""))
    time_color = mlp.pred_color and not a.ignore_pred_color
    if time_color and mlp.color_kind is None:
        raise SystemExit(f"{a.model}: SC-GS colour head (--pred_color) of a kind that is not supported (hash colour or colour by "
                         "view direction). Run again with --ignore-pred-color to convert anyway (colours then stay constant in time).")
    if mlp.pred_color:
        print("colour head (--pred_color): " + ("colours are written per keyframe" if time_color else
                                                 "ignored (--ignore-pred-color), colours stay constant in time"))
    print(f"Keeping {keep.sum():,} of {len(keep):,} Gaussians")

    if skinning:
        e = np.exp(fea[:, :M] - fea[:, :M].max(1, keepdims=True))
        idx, w = None, (e / e.sum(1, keepdims=True)).astype(np.float64)
    else:
        idx, w = knn_weights(np.concatenate([xyz, fea[:, :H]], 1), node_sd, a.knn)
    times = np.linspace(a.t0, a.t1, a.frames, dtype=np.float32)
    sh_keep = sh[keep] if (a.sh and sh is not None) else None
    const = dict(sh=sh_keep) if time_color else dict(colors=colors[keep], sh=sh_keep)
    if time_color:
        if sh is None:
            raise SystemExit(f"{ply}: no f_dc_* columns, the colour head cannot be applied")
        dc = sh[keep][:, 0].astype(np.float32)   # SH DC coefficient [N,3]; colour = 0.5 + SH_C0 * dc
        if sh_keep is not None:
            del const["sh"]                      # the DC coefficient changes: sh goes into every keyframe
    if not mlp.pred_opacity:
        const["opacities"] = opac[keep]
    frames = deform_frames(xyz, scales, rots, opac[keep], mask, idx, w, node_sd["nodes"][:, :3], mlp, times, progress=True,
                           with_color=True)
    # one keyframe at a time to disk, so memory does not grow with --frames
    with Sequence4DWriter(a.output, compact=a.compact, source=f"SC-GS: {ply}", **const) as wr:
        for t, (means, covs, opacs, d_color) in zip(times, frames):
            extra = {}
            if time_color:
                extra["colors"] = np.clip(0.5 + SH_C0 * (dc + d_color), 0.0, 1.0).astype(np.float32)
                if sh_keep is not None:
                    sh_t = sh_keep.copy()
                    sh_t[:, 0] = dc + d_color
                    extra["sh"] = sh_t
            wr.add_keyframe(t, means=means, covs=covs, opacities=opacs, **extra)
    print(f"Wrote {wr.path}: {Sequence4D.load(wr.path, lazy=True).summary()}")


if __name__ == "__main__":
    main()
