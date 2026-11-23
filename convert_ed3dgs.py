#!/usr/bin/env python3
"""E-D3DGS (Bae et al., ECCV 2024, github.com/JeongminB/E-D3DGS) -> common .npz.

Input: a model folder with `point_cloud/iteration_N/point_cloud.ply` (the canonical Gaussians, raw values: log scale, logit
opacity, unnormalised quaternion, SH, and `embedding_*` columns), `point_cloud/iteration_N/deformation.pth` (state_dict of
`deform_network`) and `cfg_args`. Formulas ported from scene/deformation.py, gaussian_renderer/__init__.py and render.py
(read 17/11/2026); the repo's code is not imported. torch (CPU) runs the small MLPs.

At time t (the repo's camera time: dynerf frame / 300, technicolor frame / 50), every Gaussian g with embedding z_g:
    t' = t + offset          offset = mean of the non-zero `offsets` (0 when there are none), as render.py does (no cam_no)
    temporal table `weight` [max_embeddings, D_t] -> bilinear resize to n rows (align_corners) -> grid_sample at (j/(D_t-1), t')
        n = min_embeddings (use_coarse_temporal_embedding), else max_embeddings (no_c2f_temporal_embedding),
            else int(min + (max - min) * min(max(iteration, 0), c2f_temporal_iter) / c2f_temporal_iter)   (coarse and fine use the same rule;
            render.py passes num_down_emb = min_embeddings)
    h = feature_out([temporal embedding, z_g])        (one MLP for the coarse stage, one for the fine stage)
    dx = pos(h), ds = scales(h), dr = rotations(h), do = opacity(h), dc = rgb(h)    (heads: ReLU, Linear, ReLU, Linear)
    xyz += dx * coef;  log_scale += ds * coef * coef_s;  quat += dr * coef;  logit += do * coef * coef_o;  sh += dc * coef_c
    (no_ds / no_dr / no_do / no_dc switch a head off; coef, coef_c, ... = 1, or the annealing ramps of `use_anneal`)
    coarse then fine, each adding to the result of the previous one; the stages do not feed the networks anything but h.
and then, as the renderer does: scale = exp(log_scale), rotation = normalize(quat), opacity = sigmoid(logit), SH as is
(degree 3). The deltas are added BEFORE the activations (unlike Deformable-3DGS).

    python convert_ed3dgs.py output/scene -o scene.npz --frames 60 --t0 0 --t1 0.9967 --sh --compact

`--set key=value` overrides a ModelHiddenParams entry that cfg_args does not have (or has wrong).
Not checked against a real model (none trained here): only against a synthetic one, see tests/test_ed3dgs.py.
HyperNeRF scenes (`loader = nerfies`) take their time from scene/hyper_loader.py, which was not read: check --t0 / --t1 there.
"""
from __future__ import annotations

import argparse
import ast
import os
import re

import numpy as np

from gs4d import SH_C0, Sequence4D, Sequence4DWriter, cov_from_quat_scale
from gs4d.checkpoint import import_torch, torch_load
from gs4d.ply import gaussians_from_ply, read_ply_vertices

# ModelHiddenParams defaults of arguments/__init__.py (what cfg_args holds unless the config overrides it)
DEFAULTS = dict(no_ds=False, no_dr=False, no_do=True, no_dc=False, min_embeddings=30, max_embeddings=150,
                use_coarse_temporal_embedding=False, no_c2f_temporal_embedding=False, no_coarse_deform=False,
                no_fine_deform=False, c2f_temporal_iter=20000, deform_from_iter=0, use_anneal=True)


def read_cfg(path):
    """cfg_args is the text of `Namespace(a=1, b='x', ...)`; parsed without eval."""
    with open(path) as f:
        tree = ast.parse(f.read().strip(), mode="eval").body
    if not (isinstance(tree, ast.Call) and getattr(tree.func, "id", "") == "Namespace"):
        raise SystemExit(f"{path}: not a Namespace(...) dump")
    out = {}
    for kw in tree.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            pass
    return out


def find_files(path, iteration):
    if not os.path.isdir(os.path.join(path, "point_cloud")):
        raise SystemExit(f"{path}: no point_cloud/ folder; not an E-D3DGS model folder")
    root = os.path.join(path, "point_cloud")
    its = sorted(int(m.group(1)) for d in os.listdir(root) if (m := re.fullmatch(r"iteration_(\d+)", d)))
    if not its:
        raise SystemExit(f"{root}: no iteration_* folder")
    if iteration >= 0 and iteration not in its:
        raise SystemExit(f"{root}: no iteration_{iteration} (have {its})")
    it = iteration if iteration >= 0 else its[-1]
    d = os.path.join(root, f"iteration_{it}")
    pth = os.path.join(d, "deformation.pth")
    if not os.path.isfile(pth):
        raise SystemExit(f"{pth} not found")
    return os.path.join(d, "point_cloud.ply"), pth, it


class Stage:
    """One of the two (coarse, fine) sets of networks."""

    def __init__(self, torch, sd, suffix):
        def seq(name):
            idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith(f"{name}_{suffix}.")})
            return [(sd[f"{name}_{suffix}.{i}.weight"].float(), sd[f"{name}_{suffix}.{i}.bias"].float()) for i in idx]
        self.torch = torch
        self.feature = seq("feature_out")
        self.heads = {n: seq(f"{n}_deform") for n in ("pos", "scales", "rotations", "opacity", "rgb")}

    def run(self, layers, x, relu_first):
        F = self.torch.nn.functional
        for n, (w, b) in enumerate(layers):
            if (n > 0) or relu_first:
                x = F.relu(x)
            x = F.linear(x, w, b)
        return x

    def hidden(self, h):
        return self.run(self.feature, h, relu_first=False)   # Linear, then (ReLU, Linear) x (D - 1): no ReLU after the last

    def head(self, name, hidden):
        return self.run(self.heads[name], hidden, relu_first=True)   # ReLU, Linear, ReLU, Linear


class EmbeddingDeformation:
    def __init__(self, torch, sd, cfg, iteration):
        self.torch, self.cfg, self.iteration = torch, cfg, iteration
        self.weight = sd["weight"].float()
        self.offsets = sd["offsets"].float() if "offsets" in sd else torch.zeros(30, 1)
        self.stages = [] if cfg["no_coarse_deform"] else [Stage(torch, sd, "c")]
        self.stages += [] if cfg["no_fine_deform"] else [Stage(torch, sd, "f")]
        nz = self.offsets[self.offsets != 0]
        self.offset = float(nz.mean()) if nz.numel() else 0.0
        it = float(iteration)
        ramp = (lambda x: float(np.clip(x, 0, 1))) if cfg["use_anneal"] else (lambda x: 1.0)
        self.coef, self.coef_c = ramp(it / 1000), ramp((it - cfg["deform_from_iter"]) / 1000)
        self.coef_o = self.coef_s = self.coef_c

    def temporal(self, t):
        """[D_t] temporal embedding at time t (the same for every Gaussian)."""
        torch, cfg = self.torch, self.cfg
        F = torch.nn.functional
        lo, hi, until = cfg["min_embeddings"], self.weight.shape[0], cfg["c2f_temporal_iter"]
        if cfg["use_coarse_temporal_embedding"]:
            n = lo
        elif cfg["no_c2f_temporal_embedding"]:
            n = hi
        else:
            n = int(lo + (hi - lo) * min(max(self.iteration, 0), until) / until)
        dim = self.weight.shape[1]
        emb = F.interpolate(self.weight[None, None], size=(n, dim), mode="bilinear", align_corners=True)
        grid = torch.cat([torch.arange(dim, dtype=torch.float32)[:, None] / (dim - 1), torch.ones(dim, 1) * (t + self.offset)], -1)[None, None]
        grid = (grid - 0.5) * 2
        return F.grid_sample(emb, grid, align_corners=True, mode="bilinear", padding_mode="reflection").reshape(dim)

    def __call__(self, t, z, xyz, log_scale, quat, logit, sh):
        """Batch of Gaussians (embedding z [B,E], canonical raw values) at time t -> deformed raw values."""
        torch, c = self.torch, self.cfg
        te = self.temporal(t)
        h_in = torch.cat([te[None].expand(len(z), -1), z], -1)
        for st in self.stages:
            h = st.hidden(h_in)
            xyz = xyz + st.head("pos", h) * self.coef
            if not c["no_ds"]:
                log_scale = log_scale + st.head("scales", h) * self.coef * self.coef_s
            if not c["no_dr"]:
                quat = quat + st.head("rotations", h) * self.coef
            if not c["no_do"]:
                logit = logit + st.head("opacity", h) * self.coef * self.coef_o
            if not c["no_dc"]:
                sh = sh + st.head("rgb", h).view(-1, 16, 3) * self.coef_c
        return xyz, log_scale, quat, logit, sh


def convert(model, out, frames=60, t0=0.0, t1=1.0, iteration=-1, keep_sh=True, compact=False, min_opacity=0.0, overrides=None, batch=65536):
    torch = import_torch("convert_ed3dgs.py")
    ply, pth, it = find_files(model, iteration)
    cfg = dict(DEFAULTS)
    cfg_path = os.path.join(model, "cfg_args")
    if os.path.isfile(cfg_path):
        cfg.update({k: v for k, v in read_cfg(cfg_path).items() if k in DEFAULTS})
    else:
        print(f"warning: {cfg_path} not found, using the ModelHiddenParams defaults of the repo (pass --set for the rest)")
    cfg.update(overrides or {})
    sd = torch_load(pth, "convert_ed3dgs.py")
    if not isinstance(sd, dict) or "weight" not in sd or not any(k.startswith("feature_out_c.") for k in sd):
        raise SystemExit(f"{pth}: no temporal table / feature_out_c; not an E-D3DGS deformation.pth")
    net = EmbeddingDeformation(torch, sd, cfg, it)
    v = read_ply_vertices(ply)
    emb_names = sorted((n for n in v.dtype.names if n.startswith("embedding")), key=lambda n: int(n.split("_")[-1]))
    if not emb_names:
        raise SystemExit(f"{ply}: no embedding_* columns; not an E-D3DGS point cloud")
    _, _, _, _, sh0 = gaussians_from_ply(ply, with_sh=True)
    if sh0 is None or sh0.shape[1] != 16:
        raise SystemExit(f"{ply}: E-D3DGS needs SH degree 3 (16 coefficients), got {None if sh0 is None else sh0.shape[1]}")
    opac0 = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    keep = opac0 > min_opacity
    f32 = lambda a: np.ascontiguousarray(a, np.float32)[keep]   # noqa: E731
    xyz = f32(np.stack([v["x"], v["y"], v["z"]], -1))
    lsc = f32(np.stack([v[f"scale_{i}"] for i in range(3)], -1))
    quat = f32(np.stack([v[f"rot_{i}"] for i in range(4)], -1))
    logit = f32(v["opacity"])[:, None]
    z = f32(np.stack([v[n] for n in emb_names], -1))
    sh0 = sh0[keep]
    print(f"{ply}\n{pth}: iteration {it}, {len(net.stages)} stage(s), embedding {z.shape[1]}, Gaussians {keep.sum():,} of {len(keep):,}")
    vary_o, vary_c = not cfg["no_do"], not cfg["no_dc"]
    const = {}
    if not vary_o:
        const["opacities"] = 1.0 / (1.0 + np.exp(-logit[:, 0]))
    if not vary_c:
        const["colors"] = np.clip(sh0[:, 0] * SH_C0 + 0.5, 0, None)
        if keep_sh:
            const["sh"] = sh0
    times = np.linspace(t0, t1, frames, dtype=np.float32)
    with Sequence4DWriter(out, compact=compact, source=f"E-D3DGS: {ply}", **const) as w:
        with torch.no_grad():
            for k, t in enumerate(times):
                M, C, O, SH = [], [], [], []
                for a in range(0, len(xyz), batch):
                    s = slice(a, a + batch)
                    p, ls, q, lg, shx = net(float(t), torch.from_numpy(z[s]), torch.from_numpy(xyz[s]), torch.from_numpy(lsc[s]),
                                            torch.from_numpy(quat[s]), torch.from_numpy(logit[s]), torch.from_numpy(sh0[s]))
                    q = torch.nn.functional.normalize(q, dim=-1)
                    M.append(p.numpy())
                    C.append(cov_from_quat_scale(q.numpy(), np.exp(ls.numpy())))
                    O.append(torch.sigmoid(lg[:, 0]).numpy())
                    SH.append(shx.numpy())
                fields = dict(means=np.concatenate(M), covs=np.concatenate(C))
                if vary_o:
                    fields["opacities"] = np.concatenate(O)
                if vary_c:
                    shk = np.concatenate(SH)
                    fields["colors"] = np.clip(shk[:, 0] * SH_C0 + 0.5, 0, None)   # the renderer clamps SH colour at 0
                    if keep_sh:
                        fields["sh"] = shk
                w.add_keyframe(float(t), **fields)
                print(f"\r  time {k + 1}/{frames}", end="\n" if k + 1 == frames else "", flush=True)
    return w.path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model folder (point_cloud/, cfg_args)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--iteration", type=int, default=-1, help="default: the last")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1.0, help="dynerf: (frames - 1) / 300, technicolor: (frames - 1) / 50")
    ap.add_argument("--min-opacity", type=float, default=0.0, help="drop Gaussians whose canonical opacity is at or below this")
    ap.add_argument("--sh", action="store_true", help="keep all SH coefficients (view-dependent colour, gsplat renderer)")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (lossy, about 2-3x smaller; see docs/FORMAT.md)")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a ModelHiddenParams entry (python literal)")
    a = ap.parse_args()
    ov = {}
    for s in a.set:
        k, sep, val = s.partition("=")
        if not sep or k not in DEFAULTS:
            raise SystemExit(f"--set {s}: expected KEY=VALUE with KEY one of {', '.join(DEFAULTS)}")
        try:
            ov[k] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            raise SystemExit(f"--set {s}: VALUE must be a Python literal (True, False, 30, ...)") from None
    path = convert(a.model, a.output, a.frames, a.t0, a.t1, a.iteration, a.sh, a.compact, a.min_opacity, ov, a.batch)
    print(f"Wrote {path}: {Sequence4D.load(path, lazy=True).summary()}")


if __name__ == "__main__":
    main()
