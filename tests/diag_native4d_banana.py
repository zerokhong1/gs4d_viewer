#!/usr/bin/env python3
"""WP19c part 3: why is native 4DGS on HyperNeRF slice-banana 24.23 dB (ours) vs 26.14-26.18 dB (the repo's own eval)?
Diagnosis only. Each step changes one thing and prints PSNR on the 33 held-out frames. Needs CUDA, the banana split and
the self-trained checkpoint (same paths as tests/check_real.py).
    python tests/diag_native4d_banana.py [--steps a,b,...]
"""
import argparse, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_real as cr
import convert_native4d as cn
from gs4d import SH_C0, Sequence4D

TRAIN = "/home/cont/gs4d_train"
CKPT = os.path.join(TRAIN, "4dgs", "output", "banana", "chkpnt_best.pth")


def build(c, times, sh=True, sh_degree=None, sh_degree_t=None, mask=0.0, min_peak=0.02, ref_sh_dir=None):
    """Same maths as convert_native4d.main, with knobs. mask: drop (opacity 0) where the temporal factor is <= mask."""
    tt = c["t"].reshape(-1)
    sig = 1.0 / (1.0 + np.exp(-c["opacity"]))
    means, cov3, opac = cn.slice_native4d(c["xyz"], c["t"], np.exp(c["scaling"]), np.exp(c["scaling_t"]), c["rotation"],
                                          c["rotation_r"], sig, times, rot_4d=c["rot_4d"])
    if mask > 0:
        factor = opac / np.maximum(sig.reshape(1, -1), 1e-12)
        opac = np.where(factor > mask, opac, 0.0).astype(np.float32)
    keep = opac.max(0) > min_peak
    print(f"   {keep.sum():,} of {len(keep):,} Gaussians kept")
    dg = c["sh_degree"] if sh_degree is None else sh_degree
    dt = c["sh_degree_t"] if sh_degree_t is None else sh_degree_t
    if sh:
        sh_all = cn.sh_at_times(c["features_dc"][keep], c["features_rest"][keep], tt[keep], times, float(times_range[1] - times_range[0]),
                                c["sh_degree"], c["sh_degree_t"])
        if dg < c["sh_degree"]:
            sh_all = sh_all[:, :, :(dg + 1) ** 2]
        colors = np.clip(SH_C0 * sh_all[:, :, 0] + 0.5, 0, 1)
    else:
        sh_all = None
        colors = np.clip(SH_C0 * c["features_dc"].reshape(len(tt), -1)[keep][:, :3] + 0.5, 0, 1)
    build.xyz = c["xyz"][keep]
    return Sequence4D(times, means[:, keep], cov3[keep], opac[:, keep], colors, sh=sh_all, source="diag")


def render_psnr(seq, test, near=0.01, unshifted_dir=False, xyz=None, repo_focal=False):
    """Own gsplat loop (check_real.psnr_item with two knobs): near plane, and SH view direction from the time-independent
    position xyz as in the repo's CUDA computeColorFromSH_4D (it gets the un-offset point), not the sliced mean."""
    import torch
    from gsplat import rasterization, spherical_harmonics
    from gs4d.format import unpack_cov
    frames, K, W, H = test
    Kt = torch.from_numpy(K).cuda()[None]
    out = []
    for t, vm, gt in frames:
        means, covs, opac, _ = seq.at(t)
        sh = torch.from_numpy(np.ascontiguousarray(seq.sh_at(t), np.float32)).cuda()
        m = torch.from_numpy(np.ascontiguousarray(means, np.float32)).cuda()
        V = torch.from_numpy(vm).cuda()
        campos = torch.linalg.inv(V)[:3, 3]
        base = torch.from_numpy(xyz.astype(np.float32)).cuda() if unshifted_dir else m
        col = torch.clamp(spherical_harmonics(seq.sh_degree, base - campos, sh) + 0.5, min=0)
        cv = torch.from_numpy(unpack_cov(np.ascontiguousarray(covs, np.float32))).cuda()
        Vm, Kr = V, Kt
        if repo_focal:
            # The repo gives every frame fl_x/fl_y/cx/cy, which leaves FovX = FovY = -1 in CameraInfo, so the rasteriser's
            # 2D-covariance Jacobian uses focal = W / (2 tan(-0.5)) (sign cancels) while the projection matrix uses the true focal.
            # Emulation: same pixel positions (camera x,y scaled by f/f'), Jacobian focal f', covariance as in camera space.
            # The perspective column of the Jacobian is matched by a per-Gaussian shear of the covariance (exact up to gsplat's frustum clamp).
            fa, fb = W / (2 * np.tan(0.5)), H / (2 * np.tan(0.5))
            R = V[:3, :3]
            pc = m @ R.T + V[:3, 3]
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            # gsplat's Jacobian at the scaled point is Jg = [[fa/z,0,-fx x/z^2],[0,fb/z,-fy y/z^2]]; the repo's is
            # Jr = [[fa/z,0,-fa x/z^2],[0,fb/z,-fb y/z^2]]. Jg = Jr T with T = I + (t1, t2) in column 3, so use Sigma_g = T^-1 Sigma T^-T.
            Tinv = torch.eye(3, device="cuda").repeat(len(pc), 1, 1)
            # computeCov2D also clamps x/z to [-lim, lim] with lim = 1.3 tan(-0.5) < 0, i.e. clamp(v, +0.71, -0.71) = -0.71 for
            # every Gaussian, so the repo's perspective column is +0.71 f'/z (x and y) instead of -f' x/z^2.
            lim = 1.3 * float(np.tan(-0.5))
            Jr_c3x, Jr_c3y = -fa * lim / z, -fb * lim / z          # col 3 of Jr: -(f' * (clamped x/z) * z) / z^2 with clamped x/z = lim
            # Jr T = Jg  =>  t1 = (Jg_c3 - Jr_c3) / (f'/z)
            t1 = (-fx * x / z ** 2 - Jr_c3x) / (fa / z)
            t2 = (-fy * y / z ** 2 - Jr_c3y) / (fb / z)
            Tinv[:, 0, 2] = -t1
            Tinv[:, 1, 2] = -t2
            cv = R @ cv @ R.T
            cv = Tinv @ cv @ Tinv.transpose(1, 2)
            pc = pc * torch.tensor([fx / fa, fy / fb, 1.0], dtype=torch.float32, device="cuda")
            m, Vm = pc, torch.eye(4, device="cuda")
            Kr = torch.tensor([[fa, 0, K[0, 2]], [0, fb, K[1, 2]], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
        with torch.no_grad():
            img, _, _ = rasterization(m, None, None, torch.from_numpy(np.ascontiguousarray(opac, np.float32)).cuda(), col,
                                      Vm[None], Kr, W, H, near_plane=near, covars=cv)
        pred = img[0].clamp(0, 1).cpu().numpy()
        out.append(10 * np.log10(1.0 / max(float(np.mean((pred - gt) ** 2)), 1e-12)))
    return np.array(out)


times_range = (0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="base,a,a_plain,b,b_all,d_dc,d_deg1")
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()
    steps = a.steps.split(",")
    test = cr.load_nerfies_val_set(cr.banana_scene(TRAIN))
    test_times = sorted(set(float(t) for t, _, _ in test[0]))
    c = cn.load_checkpoint(a.ckpt)
    print(f"checkpoint: N={len(c['t']):,} sh_degree={c['sh_degree']} sh_degree_t={c['sh_degree_t']} rot_4d={c['rot_4d']}; "
          f"{len(test[0])} test frames at {len(test_times)} distinct times")
    lin = np.linspace(0, 1, 110, dtype=np.float32)
    tk = np.array(test_times, np.float32)
    V = {  # name: (times, compact, kwargs)
        "base":    (lin, True, {}),                       # what tests/check_real.py native4d_banana does
        "a":       (tk, True, {}),                        # (a) keyframes exactly at the test times
        "a_plain": (tk, False, {}),                       # (a) + no compact quantisation
        "b":       (tk, False, dict(mask=0.05)),          # (b) + the CUDA rasteriser's marginal_t <= 0.05 cut
        "b_all":   (tk, False, dict(min_peak=0.0)),       # (b') + keep Gaussians with peak opacity <= 0.02
        "d_dc":    (tk, False, dict(sh_degree=0)),        # (d) SH degree 0 (time-dependent DC only)
        "d_deg1":  (tk, False, dict(sh_degree=1)),
        "c_near":  (tk, False, {}),                       # (c) near plane 0.2 as in the repo's in_frustum
        "c_dir":   (tk, False, {}),                       # (c) SH direction from the un-offset position
        "c_both":  (tk, False, {}),
        "c_focal": (tk, False, {}),                       # (c) the repo's own focal bug emulated (see render_psnr)
    }
    print(f"{'step':10} {'min':>7} {'mean':>7} {'max':>7}")
    for s in steps:
        times, compact, kw = V[s]
        seq = build(c, times, **kw)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.npz")
            seq.save(p, compact=compact)
            if s.startswith("c_"):
                seq = Sequence4D.load(p)
                ps = render_psnr(seq, test, near=0.2 if s in ("c_near", "c_both") else 0.01,
                                 unshifted_dir=s in ("c_dir", "c_both"), xyz=build.xyz, repo_focal=s == "c_focal")
            else:
                ps, _ = cr.psnr_item("native4d_banana", p, test)
        print(f"{s:10} {ps.min():7.2f} {ps.mean():7.2f} {ps.max():7.2f}", flush=True)
        del seq


if __name__ == "__main__":
    main()
