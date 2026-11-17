#!/usr/bin/env python3
"""MoSca (Lei et al., CVPR 2025; github.com/JiahuiLei/MoSca) -> common .npz, DYNAMIC Gaussians only.

Input: `photometric_d_model_<backend>.pth` of the log folder, the state_dict of `DynSCFGaussian` (it holds the scaffold under
`scf.*`). The formulas are ported from lib_mosca/mosca.py, dynamic_gs.py and scaffold_utils/dualquat_helper.py (read 17/11/2026);
the repo's code is not imported (it needs pytorch3d, torch_geometric, a CUDA rasteriser). torch (CPU) is used for the arithmetic.

Per Gaussian g (attached to node a_g, created at frame r_g, stored in that node's local frame when `leaf_local_flag`):
    x_live = R_wi(r_g, a_g) @ _xyz + node_xyz[r_g, a_g]        R_mtx = R_wi @ q2R(_rotation)
    K nearest nodes of a_g: sk = topo_knn_ind[a_g]; w = exp(-|x_live - node_xyz[r_g, sk]|^2 / (2 sigma[sk]^2)) + 1e-6, masked by
    topo_knn_mask (+ `_skinning_weight` when w_correction_flag), normalised; sigma = max_sigma * sigmoid(_node_sigma_logit)
    dyn_o = clamp(sum of the un-normalised w, 0, 1) if dyn_o_flag else 1 - 1e-3
    at frame t, per skinning node: R_ji = R_wj R_wi^T, t_ji = t_wj - R_ji t_wi (i = frame r_g, j = frame t) -> dual quaternion
    dq = sum_k w_k dq_k, blended with the identity by dyn_o, divided by its dual-number norm -> (R, t)
    mean = R x_live + t;  rotation = R @ R_mtx;  covariance = rot diag(s^2) rot^T,  s = min_s + sigmoid(_scaling) (max_s - min_s)
    opacity = sigmoid(_opacity) (zero for Gaussians whose ref frame is not among the nn_fusion nearest ones, when nn_fusion > 0)
    colour = SH0 of _features_dc (max_sph_order 0 only)
blending_method lbs is supported too. Time is the frame index: time = frame / --fps.

Not included: the static background (`photometric_s_model_*.pth`, StaticGaussian: key names not read). Quaternion signs follow
pytorch3d (real part >= 0), which dual-quaternion blending depends on. The scale is used as the repo's renderer is assumed to use it
(cov = R S S^T R^T); that renderer (lib_render, third party) was not read. NOT checked against a real model (none trained here),
only against a synthetic one: tests/test_mosca.py.

    python convert_mosca.py photometric_d_model_native_add3.pth -o scene.npz --fps 30 --stride 2 [--compact]
"""
from __future__ import annotations

import argparse

import numpy as np

from gs4d import Sequence4D, Sequence4DWriter
from gs4d.format import pack_cov

SH_C0 = 0.28209479177387814
DQ_EPS = 0.001   # mosca.py


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise SystemExit("convert_mosca.py needs torch (CPU is enough): pip install torch")


def quaternion_to_matrix(q):
    torch = _torch()
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack([1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r),
                     s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r),
                     s * (i * k - j * r), s * (j * k + i * r), 1 - s * (i * i + j * j)], -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(m):
    """pytorch3d's algorithm, result standardised to a non-negative real part."""
    torch = _torch()
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m.reshape(m.shape[:-2] + (9,)), -1)
    q_abs = torch.sqrt(torch.clamp(torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                                                1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], -1), min=0.0))
    cands = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1)], -2)
    cands = cands / (2.0 * q_abs[..., None].clamp(min=0.1))
    out = cands[torch.arange(len(q_abs)), q_abs.argmax(-1)] if q_abs.ndim == 2 else cands[q_abs.argmax(-1)]
    return torch.where(out[..., :1] < 0, -out, out)


def q2R(q):
    return quaternion_to_matrix(_torch().nn.functional.normalize(q, dim=-1))


def quaternion_multiply(p, q):
    torch = _torch()
    ps, pv, qs, qv = p[..., :1], p[..., 1:], q[..., :1], q[..., 1:]
    return torch.cat([ps * qs - (pv * qv).sum(-1, keepdim=True), ps * qv + qs * pv + torch.cross(pv, qv, dim=-1)], -1)


def Rt2dq(R, t):
    torch = _torch()
    p = matrix_to_quaternion(R)
    q = 0.5 * quaternion_multiply(torch.cat([torch.zeros_like(t[..., :1]), t], -1), p)
    return torch.cat([p, q], -1)


def dq2unitdq(dq):
    torch = _torch()
    bad = dq[..., :4].norm(dim=-1) < 1e-4
    a = dq[..., :4].norm(dim=-1).clamp(min=1e-8)
    b = (dq[..., :4] * dq[..., 4:]).sum(-1) / a
    inv_a, inv_b = 1 / a, -b / a ** 2
    unit = torch.cat([dq[..., :4] * inv_a[..., None], dq[..., :4] * inv_b[..., None] + dq[..., 4:] * inv_a[..., None]], -1)
    ident = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=dq.dtype)
    return torch.where(bad[..., None], ident, unit)


def dq2Rt(dq):
    p, q = dq[..., :4], dq[..., 4:]
    conj = p * _torch().tensor([1.0, -1.0, -1.0, -1.0], dtype=p.dtype)   # quaternion_invert of a unit quaternion
    return quaternion_to_matrix(p), 2 * quaternion_multiply(q, conj)[..., 1:]


class DynMoSca:
    def __init__(self, sd):
        torch = _torch()
        self.torch = torch
        g = lambda k: sd[k].detach().float() if sd[k].is_floating_point() else sd[k].detach()   # noqa: E731
        need = ["scf._node_xyz", "scf._node_rotation", "scf._node_sigma_logit", "scf.max_sigma", "scf.topo_knn_ind", "scf.topo_knn_mask",
                "_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "attach_ind", "ref_time", "max_scale", "min_scale"]
        missing = [k for k in need if k not in sd]
        if missing:
            raise SystemExit(f"missing {missing}; not a MoSca DynSCFGaussian state_dict (photometric_d_model_*.pth)")
        if int(sd.get("max_sph_order", torch.tensor(0))) != 0:
            raise SystemExit("MoSca with max_sph_order > 0 is not supported (layout of _features_rest not verified)")
        self.node_xyz, self.node_rot = g("scf._node_xyz"), g("scf._node_rotation")   # [T,M,3], [T,M,4]
        self.sigma = float(sd["scf.max_sigma"]) * torch.sigmoid(g("scf._node_sigma_logit")).squeeze(-1)   # [M]
        self.knn, self.knn_mask = g("scf.topo_knn_ind").long(), g("scf.topo_knn_mask").bool()
        self.lbs = int(sd["scf.blending_method"].reshape(-1)[0]) == 1 if "scf.blending_method" in sd else False
        self.xyz, self.rot, self.scaling = g("_xyz"), g("_rotation"), g("_scaling")
        self.opacity = torch.sigmoid(g("_opacity")).reshape(-1)
        self.dc = g("_features_dc")
        self.attach, self.ref_time = g("attach_ind").long(), g("ref_time").long()
        self.min_s, self.max_s = float(sd["min_scale"]), float(sd["max_scale"])
        self.local = bool(sd.get("leaf_local_flag", torch.tensor(True)))
        self.dyn_o_flag = bool(sd.get("dyn_o_flag", torch.tensor(False)))
        self.nn_fusion = int(sd.get("nn_fusion", torch.tensor(-1)))
        self.corr = g("_skinning_weight") if bool(sd.get("w_correction_flag", torch.tensor(False))) else None
        self.T, self.M = self.node_xyz.shape[:2]
        self.N = len(self.xyz)
        self.scale = self.min_s + torch.sigmoid(self.scaling) * (self.max_s - self.min_s)
        self.all_ref = self.ref_time

    def prepare(self, sl):
        """Everything that does not depend on the target frame, for the Gaussians in `sl`."""
        torch = self.torch
        a, r = self.attach[sl], self.ref_time[sl]
        if self.local:
            nq = self.node_rot[r, a]
            Rwi = q2R(nq)
            x = torch.einsum("nij,nj->ni", Rwi, self.xyz[sl]) + self.node_xyz[r, a]
            Rm = Rwi @ q2R(self.rot[sl])
        else:
            x, Rm = self.xyz[sl], q2R(self.rot[sl])
        sk, mask = self.knn[a], self.knn_mask[a]
        ref_xyz, ref_quat = self.node_xyz[r[:, None], sk], self.node_rot[r[:, None], sk]
        d2 = ((x[:, None] - ref_xyz) ** 2).sum(-1)
        w = (torch.exp(-d2 / (2 * self.sigma[sk] ** 2)) + 1e-6) * mask.float()
        if self.corr is not None:
            w = (w + self.corr[sl]).abs()
        wsum = w.sum(-1)
        w = w / wsum.clamp(min=1e-6)[:, None]
        dyn_o = wsum.clamp(0, 1) if self.dyn_o_flag else torch.ones_like(wsum) * (1.0 - DQ_EPS)
        return dict(x=x, Rm=Rm, sk=sk, w=w, dyn_o=dyn_o, Rwi=q2R(ref_quat), twi=ref_xyz)

    def warp(self, pre, t):
        torch = self.torch
        sk = pre["sk"]
        Rwj, twj = q2R(self.node_rot[t][sk]), self.node_xyz[t][sk]
        Rji = torch.einsum("nsij,nskj->nsik", Rwj, pre["Rwi"])
        tji = twj - torch.einsum("nsij,nsj->nsi", Rji, pre["twi"])
        w, dyn_o = pre["w"], pre["dyn_o"]
        if self.lbs:
            quat = torch.einsum("nki,nk->ni", matrix_to_quaternion(Rji.reshape(-1, 3, 3)).reshape(Rji.shape[:2] + (4,)), w)
            tt = torch.einsum("nki,nk->ni", tji, w)
            unit = torch.tensor([1.0, 0, 0, 0])
            quat = quat * dyn_o[:, None] + (1 - dyn_o)[:, None] * unit
            R, tr = q2R(quat), tt * dyn_o[:, None]
        else:
            dq = Rt2dq(Rji.reshape(-1, 3, 3), tji.reshape(-1, 3)).reshape(Rji.shape[:2] + (8,))
            dq = torch.einsum("nki,nk->ni", dq, w)
            unit = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
            dq = dq * dyn_o[:, None] + unit * (1 - dyn_o)[:, None]
            R, tr = dq2Rt(dq2unitdq(dq))
        return torch.einsum("nij,nj->ni", R, pre["x"]) + tr, torch.einsum("nij,njk->nik", R, pre["Rm"])

    def frame(self, t, pres, batch):
        torch = self.torch
        means, covs = [], []
        for sl, pre in pres:
            mu, fr = self.warp(pre, t)
            M = fr * self.scale[sl][:, None, :]
            means.append(mu.numpy()); covs.append(pack_cov((M @ M.transpose(1, 2)).numpy()))
        o = self.opacity.clone()
        if self.nn_fusion > 0:
            sup = torch.unique(self.all_ref)
            sup = sup[torch.argsort((sup - t).abs())[: self.nn_fusion]]
            o = o * (self.ref_time[:, None] == sup[None]).any(1).float()   # kept Gaussians only; supporting frames from all
        return np.concatenate(means).astype(np.float32), np.concatenate(covs).astype(np.float32), o.numpy().astype(np.float32)


def convert(path, out, fps=30.0, stride=1, compact=False, min_opacity=0.0, batch=65536):
    torch = _torch()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = DynMoSca(sd)
    keep = m.opacity.numpy() > min_opacity
    print(f"{path}: {m.N:,} dynamic Gaussians (keeping {keep.sum():,}), {m.M} nodes, {m.T} frames, "
          f"{'LBS' if m.lbs else 'DQB'}, nn_fusion {m.nn_fusion}")
    idx = torch.from_numpy(np.nonzero(keep)[0])
    pres = []
    with torch.no_grad():
        for a in range(0, len(idx), batch):
            sl = idx[a:a + batch]
            pres.append((sl, m.prepare(sl)))
        sub = DynMoSca.__new__(DynMoSca)
        sub.__dict__.update(m.__dict__)
        sub.opacity = m.opacity[idx]; sub.all_ref = m.ref_time; sub.ref_time = m.ref_time[idx]
        sub.scale = m.scale
        frames = list(range(0, m.T, stride))
        colors = np.clip(m.dc[idx].numpy() * SH_C0 + 0.5, 0, None).astype(np.float32)
        with Sequence4DWriter(out, compact=compact, source=f"MoSca: {path}", colors=colors) as w:
            for k, t in enumerate(frames):
                means, covs, op = sub.frame(t, pres, batch)
                w.add_keyframe(t / fps, means=means, covs=covs, opacities=op)
                print(f"\r  frame {k + 1}/{len(frames)}", end="\n" if k + 1 == len(frames) else "", flush=True)
    return w.path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="photometric_d_model_<backend>.pth")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="time = frame / fps")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th frame as a keyframe")
    ap.add_argument("--min-opacity", type=float, default=0.0)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--batch", type=int, default=65536)
    a = ap.parse_args()
    p = convert(a.checkpoint, a.output, a.fps, a.stride, a.compact, a.min_opacity, a.batch)
    print(f"Wrote {p}: {Sequence4D.load(p, lazy=True).summary()}")
