#!/usr/bin/env python3
"""Self-checks. Run from the project root:  python tests/test_all.py [path/to/4d-gaussian-splatting]

1. interpolation / container round trip
2. per-frame ply folder -> common format
3. native 4D slicing: a sliced 3D Gaussian must equal the 4D density restricted to time t,
   and (if the reference repo path is given and torch is installed) match the reference code.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "converters"))

from convert_native4d import rotation_4d, slice_native4d  # noqa: E402
from gs4d import Sequence4D, cov_from_quat_scale, unpack_cov  # noqa: E402
from gs4d.ply import gaussians_from_ply  # noqa: E402


def test_container(tmp):
    rng = np.random.default_rng(0)
    K, N = 5, 100
    seq = Sequence4D(np.arange(K), rng.normal(size=(K, N, 3)), np.abs(rng.normal(size=(N, 6))),
                     rng.uniform(size=(K, N)), rng.uniform(size=(N, 3)))
    p = os.path.join(tmp, "a.npz")
    seq.save(p)
    s2 = Sequence4D.load(p)
    m, c, o, col = s2.at(1.25)
    assert np.allclose(m, 0.75 * seq.means[1] + 0.25 * seq.means[2], atol=1e-6)
    assert np.allclose(o, 0.75 * seq.opacities[1] + 0.25 * seq.opacities[2], atol=1e-6)
    assert c.shape == (N, 6) and col.shape == (N, 3)
    assert np.array_equal(s2.at(-5)[0], seq.means[0]) and np.array_equal(s2.at(99)[0], seq.means[-1])
    assert np.array_equal(s2.at(3.0)[0], seq.means[3])
    print("ok  container + interpolation")


def write_3dgs_ply(path, means, quats, log_scales, logit_opac, dc):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.concatenate([means, np.zeros_like(means), dc, logit_opac[:, None], log_scales, quats], 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(means)).encode())
        f.write("".join(f"property float {n}\n" for n in names).encode())
        f.write(b"end_header\n")
        arr.tofile(f)


def test_ply_folder(tmp):
    rng = np.random.default_rng(1)
    N = 200
    folder = os.path.join(tmp, "frames")
    os.makedirs(folder)
    ref = []
    for k in range(12):  # 12 frames: checks natural sort (time_2 before time_10)
        means, quats = rng.normal(size=(N, 3)), rng.normal(size=(N, 4))
        ls, lo, dc = rng.normal(-3, 0.3, (N, 3)), rng.normal(size=N), rng.normal(size=(N, 3))
        write_3dgs_ply(os.path.join(folder, f"time_{k}.ply"), means, quats, ls, lo, dc)
        ref.append((means, cov_from_quat_scale(quats, np.exp(ls)), 1 / (1 + np.exp(-lo))))
    m, c, o, _ = gaussians_from_ply(os.path.join(folder, "time_3.ply"))
    assert np.allclose(m, ref[3][0], atol=1e-5) and np.allclose(c, ref[3][1], atol=1e-5)
    out = os.path.join(tmp, "frames.npz")
    subprocess.run([sys.executable, os.path.join(ROOT, "converters", "convert_per_frame.py"), folder, "-o", out, "--fps", "10"],
                   check=True, stdout=subprocess.DEVNULL)
    seq = Sequence4D.load(out)
    assert seq.num_frames == 12 and seq.num_gaussians == N and seq.interpolate
    assert np.allclose(seq.means[10], ref[10][0], atol=1e-5) and np.allclose(seq.opacities[10], ref[10][2], atol=1e-5)
    assert np.isclose(seq.times[-1], 1.1)
    print("ok  per-frame ply folder -> common format")


def test_native4d(ref_repo):
    rng = np.random.default_rng(2)
    N = 50
    xyz, t = rng.normal(size=(N, 3)), rng.uniform(0, 1, (N, 1))
    s_xyz, s_t = rng.uniform(0.05, 0.5, (N, 3)), rng.uniform(0.05, 0.5, (N, 1))
    ql, qr = rng.normal(size=(N, 4)), rng.normal(size=(N, 4))
    opac = rng.uniform(0.2, 1, (N, 1))
    times = np.array([0.1, 0.5, 0.9])
    means, cov3, o = slice_native4d(xyz, t, s_xyz, s_t, ql, qr, opac, times)

    R = rotation_4d(ql, qr)
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(4), atol=1e-9), "4D rotation is not orthonormal"
    S4 = (R * np.concatenate([s_xyz, s_t], 1)[:, None, :])
    S4 = S4 @ S4.transpose(0, 2, 1)
    P = np.linalg.inv(S4)
    mu4 = np.concatenate([xyz, t], 1)
    # The 4D density at (x, t) must equal  marginal(t) * N3(x; mean(t), cov3)  for arbitrary x.
    C3 = unpack_cov(cov3).astype(np.float64)
    for k, tk in enumerate(times):
        x = means[k].astype(np.float64) + rng.normal(0, 0.1, (N, 3))
        d4 = np.concatenate([x, np.full((N, 1), tk)], 1) - mu4
        log4 = -0.5 * np.einsum("ni,nij,nj->n", d4, P, d4)
        d3 = x - means[k]
        log3 = -0.5 * np.einsum("ni,nij,nj->n", d3, np.linalg.inv(C3), d3)
        logm = np.log(o[k].astype(np.float64) / opac[:, 0])
        assert np.allclose(log4, log3 + logm, atol=2e-3, rtol=2e-3), np.abs(log4 - log3 - logm).max()
    print("ok  native 4D slicing matches the conditional of the 4D Gaussian")

    if ref_repo:
        import importlib.util

        import torch

        spec = importlib.util.spec_from_file_location("ref_general_utils", os.path.join(ref_repo, "utils", "general_utils.py"))
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        Rref = ref.build_rotation_4d(torch.tensor(ql), torch.tensor(qr)).numpy()
        assert np.allclose(R, Rref, atol=1e-9)
        L = torch.tensor(Rref) @ torch.diag_embed(torch.tensor(np.concatenate([s_xyz, s_t], 1)))
        cov = L @ L.transpose(1, 2)
        c12, ct = cov[:, 0:3, 3:4], cov[:, 3:4, 3:4]
        cur = (cov[:, :3, :3] - c12 @ c12.transpose(1, 2) / ct).numpy()
        assert np.allclose(unpack_cov(cov3), cur, atol=1e-5)
        for k, tk in enumerate(times):
            off = (c12.squeeze(-1) / ct.squeeze(-1) * (tk - torch.tensor(t))).numpy()
            assert np.allclose(means[k], xyz + off, atol=1e-5)
            marg = np.exp(-0.5 * (t - tk) ** 2 / ct.squeeze(-1).numpy())
            assert np.allclose(o[k], (opac * marg)[:, 0], atol=1e-6)
        print("ok  native 4D slicing matches the reference repo's functions")


def test_native4d_checkpoint(tmp):
    try:
        import torch
    except ImportError:
        print("skip  checkpoint round trip (torch not installed)")
        return
    g = torch.Generator().manual_seed(0)
    N = 400
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    cap = (3, r(N, 3), r(N, 1, 3), r(N, 15, 3), r(N, 3) * 0.3 - 2, r(N, 4), r(N, 1) + 2, torch.zeros(N),
           torch.zeros(N, 1), torch.zeros(N, 1), torch.zeros(N, 1), {}, 1.0,
           torch.rand(N, 1, generator=g), r(N, 1) * 0.3 - 1, r(N, 4), True, None, 0)
    ck, out = os.path.join(tmp, "chkpnt_best.pth"), os.path.join(tmp, "n4d.npz")
    torch.save((cap, 30000), ck)
    subprocess.run([sys.executable, os.path.join(ROOT, "converters", "convert_native4d.py"), ck, "-o", out, "--frames", "20",
                    "--t0", "0", "--t1", "1"], check=True, stdout=subprocess.DEVNULL)
    seq = Sequence4D.load(out)
    assert seq.num_frames == 20 and seq.means.ndim == 3 and seq.covs.ndim == 2 and seq.opacities.ndim == 2
    assert np.all(np.linalg.eigvalsh(unpack_cov(seq.covs).astype(np.float64)) > 0), "sliced covariance not PD"
    print(f"ok  native 4D checkpoint -> common format ({seq.num_gaussians} of {N} Gaussians kept)")


if __name__ == "__main__":
    ref_repo = sys.argv[1] if len(sys.argv) > 1 else None
    with tempfile.TemporaryDirectory() as tmp:
        test_container(tmp)
        test_ply_folder(tmp)
        test_native4d(ref_repo)
        test_native4d_checkpoint(tmp)
    print("all checks passed")
