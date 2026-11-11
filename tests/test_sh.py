#!/usr/bin/env python3
"""Checks for the optional `sh` field (view-dependent colour). CPU checks always run; the render check needs CUDA.

    python tests/test_sh.py
"""
import argparse, os, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D, SH_C0
from gs4d.ply import gaussians_from_ply
import convert_per_frame

SH_C1 = 0.4886025119029199
tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)


def seq_with(sh, K=3, N=5):
    return Sequence4D(np.arange(K, dtype=np.float32), rng.normal(size=(K, N, 3)), np.tile([1e-2, 0, 0, 1e-2, 0, 1e-2], (N, 1)),
                      np.full(N, 0.9), rng.uniform(size=(N, 3)), sh=sh)


def write_ply(path, N, degree, seed):
    """Standard 3DGS point_cloud.ply; returns the SH coefficients as [N,S,3]."""
    r = np.random.default_rng(seed)
    S = (degree + 1) ** 2
    sh = r.normal(0, 0.3, (N, S, 3)).astype(np.float32)
    rest = sh[:, 1:].transpose(0, 2, 1).reshape(N, -1)  # 3DGS: features_rest.transpose(1, 2).flatten()
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(3 * (S - 1))]
             + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    cols = [r.normal(size=(N, 3)), np.zeros((N, 3)), sh[:, 0], rest, r.normal(size=(N, 1)), r.normal(-3, 0.3, (N, 3)),
            r.normal(size=(N, 4))]
    data = np.concatenate(cols, 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {N}\n"
                 + "".join(f"property float {n}\n" for n in names) + "end_header\n").encode())
        f.write(data.tobytes())
    return sh


def test_roundtrip():
    for K_axis in (False, True):
        sh = rng.normal(size=((3, 5, 4, 3) if K_axis else (5, 4, 3))).astype(np.float32)
        s = seq_with(sh)
        p = os.path.join(tmp, f"s{int(K_axis)}.npz")
        s.save(p)
        back = Sequence4D.load(p)
        assert back.sh_degree == 1 and np.array_equal(back.sh, sh)
        assert np.array_equal(back.colors, s.colors) and np.array_equal(back.means, s.means)
        want = 0.75 * sh[1] + 0.25 * sh[2] if K_axis else sh
        assert np.allclose(back.sh_at(1.25), want, atol=1e-6)
    # a file without the field, as written before `sh` existed
    s = seq_with(None)
    p = os.path.join(tmp, "old.npz")
    np.savez(p, times=s.times, means=s.means, covs=s.covs, opacities=s.opacities, colors=s.colors,
             interpolate=np.bool_(True), source=np.str_("old"))
    back = Sequence4D.load(p)
    assert back.sh is None and back.sh_degree is None and back.sh_at(0.5) is None
    s.save(p)
    assert "sh" not in np.load(p).files
    for bad in (np.zeros((5, 5, 3)), np.zeros((4, 4, 3)), np.zeros((5, 4))):
        try:
            seq_with(bad)
        except ValueError:
            continue
        raise AssertionError(f"sh of shape {bad.shape} was accepted")
    print("ok  sh field: save/load round trip, interpolation, file without sh, shape validation")


def test_ply():
    folder = os.path.join(tmp, "frames")
    os.makedirs(folder)
    truth = [write_ply(os.path.join(folder, f"time_{k:05d}.ply"), 40, 3, k) for k in range(3)]
    *_, colors, sh = gaussians_from_ply(os.path.join(folder, "time_00001.ply"), with_sh=True)
    assert sh.shape == (40, 16, 3) and np.array_equal(sh, truth[1])
    assert np.allclose(colors, np.clip(SH_C0 * truth[1][:, 0] + 0.5, 0, 1), atol=1e-6)
    assert len(gaussians_from_ply(os.path.join(folder, "time_00001.ply"))) == 4
    with_sh = convert_per_frame.from_ply_folder(folder, 1, 30.0, sh=True)
    assert with_sh.sh.shape == (3, 40, 16, 3) and np.array_equal(with_sh.sh, np.stack(truth))
    assert convert_per_frame.from_ply_folder(folder, 1, 30.0).sh is None
    print("ok  3DGS ply f_rest_* -> sh, convert_per_frame --sh")


def test_render():
    try:
        import torch, gsplat  # noqa: F401
        assert torch.cuda.is_available()
    except Exception:
        print("skip gsplat render check (no CUDA)")
        return
    import viewer as V
    # two big Gaussians, 3 apart: one with degree-1 SH, one with DC only
    sh = np.zeros((2, 4, 3), np.float32)
    sh[0] = [[0.2, -0.4, 0.1], [0.5, -0.3, 0.2], [-0.4, 0.6, 0.3], [0.7, 0.2, -0.6]]
    sh[1, 0] = [0.9, -0.5, 0.3]
    centres = np.array([[0, 0, 0], [0, 3, 0]], np.float32)
    seq = Sequence4D([0.0, 1.0], centres, np.tile([0.09, 0, 0, 0.09, 0, 0.09], (2, 1)), np.full(2, 0.999),
                     np.clip(SH_C0 * sh[:, 0] + 0.5, 0, 1), sh=sh)
    vw = V.Viewer(seq, argparse.Namespace(renderer="gsplat", host="127.0.0.1", port=8145, up="+z", max_splats=300_000, max_res=1080))
    W = H = 200
    seen = []
    for target, coef in ((centres[0], sh[0]), (centres[1], sh[1])):
        for offset in ([4, 0, 0], [-4, 0, 0], [0, 0, 4], [2, -3, 1.5]):
            pos = target + np.array(offset, np.float32)
            fwd = (target - pos) / np.linalg.norm(target - pos)  # camera +z
            right = np.cross(fwd, [0.3, 0.5, 0.8]); right /= np.linalg.norm(right)
            R = np.stack([right, np.cross(fwd, right), fwd], 1)  # camera-to-world, columns x y z
            assert np.linalg.det(R) > 0.999
            # rotation matrix -> wxyz: eigenvector of the largest eigenvalue of the 4x4 K matrix (valid for any rotation)
            Kq = np.array([[R[0, 0] + R[1, 1] + R[2, 2], R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                           [R[2, 1] - R[1, 2], R[0, 0] - R[1, 1] - R[2, 2], R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]],
                           [R[0, 2] - R[2, 0], R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], R[1, 2] + R[2, 1]],
                           [R[1, 0] - R[0, 1], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1]]])
            wxyz = np.linalg.eigh(Kq)[1][:, -1]
            img = vw.render_gsplat(wxyz, pos, np.deg2rad(40), W, H)
            got = img[H // 2 - 2:H // 2 + 2, W // 2 - 2:W // 2 + 2].reshape(-1, 3).mean(0) / 255
            x, y, z = fwd  # direction camera -> Gaussian
            want = np.clip(SH_C0 * coef[0] - SH_C1 * y * coef[1] + SH_C1 * z * coef[2] - SH_C1 * x * coef[3] + 0.5, 0, 1)
            assert np.abs(got - want).max() < 0.02, f"from {offset}: rendered {got.round(3)}, formula {want.round(3)}"
            seen.append(got)
    vw.server.stop()
    assert np.abs(seen[0] - seen[1]).max() > 0.3, "SH Gaussian does not change colour with the view direction"
    assert np.abs(seen[4] - seen[5]).max() < 0.02, "DC-only Gaussian changed colour with the view direction"
    print("ok  gsplat renderer: SH colour matches the 3DGS formula from 4 directions, DC-only stays constant")


test_roundtrip()
test_ply()
test_render()
print("all checks passed")
