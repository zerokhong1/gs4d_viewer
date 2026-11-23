#!/usr/bin/env python3
"""Converter for 4D-Rotor Gaussian Splatting (Duan et al., SIGGRAPH 2024; github.com/weify627/4D-Rotor-Gaussians, Nerfstudio).

Input: the Nerfstudio checkpoint `outputs/<scene>/splatfacto/<run>/nerfstudio_models/step-*.ckpt` (needs torch to unpickle it).
Its `pipeline` state dict holds `gauss_params.{xyz, t, scales_xyz, scales_t, quats1, quats2, features_dc, features_rest, opacities}`.

Formulas, ported from `nerfstudio/utils/general_utils.py` and `SplatfactoModel.forward_1` of that repo (read 17/11/2026):
    rotation  R = build_rotation_from_rotor8(quats1, quats2)  (4x4, the two rotors are used raw, as the renderer does)
    Sigma = (R S)(R S)^T, S = diag(exp(scales_xyz), exp(scales_t));  w = 1 / Sigma[3,3];  speed = Sigma[:3,3] * w
    cov3 = Sigma[:3,:3] - Sigma[:3,3] Sigma[3,:3] * w                  (does not depend on time)
    mean(t)    = xyz + speed * (t - t_g)
    opacity(t) = sigmoid(opacities) * exp(-0.5 * w * (t - t_g)^2)      (the repo drops a Gaussian when that exponent >= 16)
    colour     = SH degree 3 of features_dc / features_rest, constant in time: written as `sh` (and `colors` = SH0)
So per keyframe only `means` and `opacities` change; covariance, colour and SH are written once.

    python converters/convert_4drotor.py step-000019999.ckpt -o scene.npz --frames 60 --t0 0 --t1 1 [--compact] [--min-peak-opacity 0.02]

Time is the raw `time` of the dataset's transforms json (D-NeRF: [0, 1]); pass --t0 / --t1 for other data.
NOT checked against a real checkpoint (none trained here); only against a synthetic one, see tests/test_4drotor.py.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, for `import gs4d`
from gs4d import SH_C0, Sequence4D, Sequence4DWriter
from gs4d.checkpoint import torch_load

FIELDS = ("xyz", "t", "scales_xyz", "scales_t", "quats1", "quats2", "features_dc", "features_rest", "opacities")


def load_params(path):
    # a Nerfstudio checkpoint pickles more than tensors, which weights_only=True refuses: only open checkpoints you trust
    ck = torch_load(path, "convert_4drotor.py", weights_only=False)
    sd = ck.get("pipeline", ck) if isinstance(ck, dict) else None
    if sd is None:
        raise SystemExit(f"{path}: not a Nerfstudio checkpoint (no 'pipeline' state dict)")
    out = {}
    for name in FIELDS:
        keys = [k for k in sd if k == name or k.endswith("gauss_params." + name)]
        if not keys:
            raise SystemExit(f"{path}: no gauss_params.{name}; not a 4D-Rotor Gaussians checkpoint")
        out[name] = sd[keys[0]].detach().float().cpu().numpy()
    return out


def rotation_from_rotor8(r1, r2):
    """[N,4] + [N,4] -> [N,4,4]; the repo's build_rotation_from_rotor8 (a, bxy, bxz, byz | bxw, byw, bzw, pxyzw)."""
    a, bxy, bxz, byz = (r1[:, i] for i in range(4))
    bxw, byw, bzw, p = (r2[:, i] for i in range(4))
    s2, bxy2, bxz2, bxw2, byz2, byw2, bzw2, p2 = a * a, bxy * bxy, bxz * bxz, bxw * bxw, byz * byz, byw * byw, bzw * bzw, p * p
    r = np.zeros((len(a), 4, 4), np.float64)
    r[:, 0, 0] = -bxy2 - bxz2 - bxw2 + byz2 + byw2 + bzw2 - p2 + s2
    r[:, 1, 0] = 2 * (bxy * a - bxz * byz - bxw * byw + bzw * p)
    r[:, 2, 0] = 2 * (bxy * byz + bxz * a - bxw * bzw - byw * p)
    r[:, 3, 0] = 2 * (bxy * byw + bxz * bzw + bxw * a + byz * p)
    r[:, 0, 1] = -2 * (bxy * a + bxz * byz + bxw * byw + bzw * p)
    r[:, 1, 1] = -bxy2 + bxz2 + bxw2 - byz2 - byw2 + bzw2 - p2 + s2
    r[:, 2, 1] = 2 * (-bxy * bxz + bxw * p + byz * a - byw * bzw)
    r[:, 3, 1] = 2 * (-bxy * bxw - bxz * p + byz * bzw + byw * a)
    r[:, 0, 2] = 2 * (bxy * byz - bxz * a - bxw * bzw + byw * p)
    r[:, 1, 2] = -2 * (bxy * bxz + bxw * p + byw * bzw + byz * a)
    r[:, 2, 2] = bxy2 - bxz2 + bxw2 - byz2 + byw2 - bzw2 - p2 + s2
    r[:, 3, 2] = 2 * (bxy * p - bxz * bxw - byw * byz + bzw * a)
    r[:, 0, 3] = 2 * (bxy * byw + bxz * bzw - bxw * a - byz * p)
    r[:, 1, 3] = 2 * (-bxy * bxw + bxz * p + byz * bzw - byw * a)
    r[:, 2, 3] = -2 * (bxy * p + bxz * bxw + byz * byw + bzw * a)
    r[:, 3, 3] = bxy2 + bxz2 - bxw2 + byz2 - byw2 - bzw2 - p2 + s2
    return r.transpose(0, 2, 1)


def slice_4d(scales, r1, r2):
    """scales [N,4] (linear), rotors -> (cov3 packed [N,6] as xx xy xz yy yz zz, speed [N,3], w [N])."""
    L = rotation_from_rotor8(r1.astype(np.float64), r2.astype(np.float64)) * scales.astype(np.float64)[:, None, :]   # R @ diag(s)
    sig = L @ L.transpose(0, 2, 1)
    w = 1.0 / sig[:, 3, 3]
    speed = sig[:, :3, 3] * w[:, None]
    cov = sig[:, :3, :3] - sig[:, :3, 3, None] * sig[:, None, 3, :3] * w[:, None, None]
    return np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1), speed, w


def means_opacities(p, cov_speed_w, t):
    cov, speed, w = cov_speed_w
    dt = t - p["t"][:, 0].astype(np.float64)
    mean = p["xyz"] + speed * dt[:, None]
    tshift = 0.5 * w * dt * dt
    op = 1.0 / (1.0 + np.exp(-p["opacities"][:, 0].astype(np.float64))) * np.where(tshift < 16, np.exp(-tshift), 0.0)
    return mean.astype(np.float32), op.astype(np.float32)


def convert(p, out, frames=60, t0=0.0, t1=1.0, compact=False, min_peak_opacity=0.0, source=""):
    cs = slice_4d(np.exp(np.concatenate([p["scales_xyz"], p["scales_t"]], 1)), p["quats1"], p["quats2"])
    times = np.linspace(t0, t1, frames)
    peak = np.zeros(len(p["xyz"]))
    for t in times:   # first pass: highest opacity over the keyframes (Gaussians that are never visible are dropped)
        peak = np.maximum(peak, means_opacities(p, cs, float(t))[1])
    keep = (peak > min_peak_opacity) & np.isfinite(cs[0]).all(1)
    p = {k: v[keep] for k, v in p.items()}
    cs = tuple(a[keep] for a in cs)
    sh = np.concatenate([p["features_dc"], p["features_rest"]], 1)   # [N, S, 3], 3DGS convention
    with Sequence4DWriter(out, compact=compact, source=source, covs=cs[0], colors=sh[:, 0] * SH_C0 + 0.5, sh=sh) as w:
        for t in times:
            m, o = means_opacities(p, cs, float(t))
            w.add_keyframe(float(t), means=m, opacities=o)
    return int(keep.sum()), len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="nerfstudio_models/step-*.ckpt")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--frames", type=int, default=60, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=0.0, help="start time (the dataset's raw time; D-NeRF: 0)")
    ap.add_argument("--t1", type=float, default=1.0)
    ap.add_argument("--min-peak-opacity", type=float, default=0.02, help="drop Gaussians whose opacity never exceeds this at a keyframe")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (lossy, about 2-3x smaller; see docs/FORMAT.md)")
    a = ap.parse_args()
    k, n = convert(load_params(a.checkpoint), a.output, a.frames, a.t0, a.t1, a.compact, a.min_peak_opacity, source=a.checkpoint)
    out = a.output if a.output.endswith(".npz") else a.output + ".npz"
    print(f"Kept {k:,} of {n:,} Gaussians\nWrote {out}: {Sequence4D.load(out, lazy=True).summary()}")


if __name__ == "__main__":
    main()
