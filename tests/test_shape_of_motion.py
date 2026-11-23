"""convert_shape_of_motion.py against an independent formulation, on a fake checkpoint in the real layout.

The reference follows the repo's own route for the rotation: blended 6D rotation -> matrix -> unit quaternion,
Hamilton product with the Gaussian's quaternion, then covariance from that quaternion (the converter multiplies
matrices instead). Needs torch (CPU) to write the checkpoint; skipped without it.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D  # noqa: E402
from gs4d.format import unpack_cov  # noqa: E402


def quat_from_matrix(R):
    """Unit quaternion wxyz of a rotation matrix (largest-component branch)."""
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    w, v = np.linalg.eigh(K, UPLO="L")
    x, y, z, r = v[:, np.argmax(w)]
    return np.array([r, x, y, z])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def matrix_from_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                     [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                     [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])


def test_shape_of_motion():
    try:
        import torch
    except ImportError:
        print("skip  shape of motion (torch not installed)")
        return
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    G, Gb, B, T = 40, 15, 6, 9
    model = {
        "fg.params.means": r(G, 3), "fg.params.quats": r(G, 4), "fg.params.scales": r(G, 3) * 0.3 - 2,
        "fg.params.colors": r(G, 3), "fg.params.opacities": r(G) + 1, "fg.params.motion_coefs": r(G, B) * 2,
        "fg.scene_center": torch.zeros(3), "fg.scene_scale": torch.tensor(1.0),
        "motion_bases.params.rots": torch.tensor([1.0, 0, 0, 0, 1, 0]) + 0.6 * r(B, T, 6),
        "motion_bases.params.transls": r(B, T, 3),
        "bg.params.means": r(Gb, 3) * 3, "bg.params.quats": r(Gb, 4), "bg.params.scales": r(Gb, 3) * 0.3 - 2,
        "bg.params.colors": r(Gb, 3), "bg.params.opacities": r(Gb) + 1,
        "bg.scene_center": torch.zeros(3), "bg.scene_scale": torch.tensor(2.0), "bg_scene_scale": torch.tensor(2.0),
        "Ks": torch.eye(3).repeat(T, 1, 1), "w2cs": torch.eye(4).repeat(T, 1, 1),
    }
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "checkpoints"))
        ckpt, out = os.path.join(tmp, "checkpoints", "last.ckpt"), os.path.join(tmp, "som.npz")
        # real checkpoints carry numpy scalars in the optimizer / scheduler state
        torch.save({"model": model, "optimizers": {"lr": np.float64(1e-3)}, "schedulers": {"n": np.int64(7)},
                    "global_step": 123, "epoch": 4}, ckpt)
        subprocess.run([sys.executable, os.path.join(ROOT, "converters", "convert_shape_of_motion.py"), ckpt, "-o", out, "--fps", "10",
                        "--stride", "2"], check=True, stdout=subprocess.DEVNULL)
        seq = Sequence4D.load(out)
        subprocess.run([sys.executable, os.path.join(ROOT, "converters", "convert_shape_of_motion.py"), ckpt, "-o", out, "--fg-only"],
                       check=True, stdout=subprocess.DEVNULL)
        fg = Sequence4D.load(out)
    assert fg.num_gaussians == G and fg.num_frames == T
    assert seq.num_gaussians == G + Gb and seq.num_frames == 5
    assert np.allclose(seq.times, np.arange(0, T, 2) / 10.0)
    m = {k: v.numpy().astype(np.float64) for k, v in model.items()}
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    for k in (0, 2, 4):
        f = 2 * k
        for i in range(G):
            c = np.exp(m["fg.params.motion_coefs"][i])
            c /= c.sum()
            r6 = sum(c[b] * m["motion_bases.params.rots"][b, f] for b in range(B))
            tr = sum(c[b] * m["motion_bases.params.transls"][b, f] for b in range(B))
            x = r6[:3] / np.linalg.norm(r6[:3])
            y = r6[3:] - np.dot(r6[3:], x) * x
            y /= np.linalg.norm(y)
            Rf = np.stack([x, y, np.cross(x, y)], axis=1)
            p = Rf @ m["fg.params.means"][i] + tr
            q0 = m["fg.params.quats"][i] / np.linalg.norm(m["fg.params.quats"][i])
            R = matrix_from_quat(quat_mul(quat_from_matrix(Rf), q0))
            S = np.diag(np.exp(m["fg.params.scales"][i]))
            assert np.allclose(seq.means[k][i], p, atol=1e-4), (k, i)
            assert np.allclose(unpack_cov(seq.covs[k][i]), R @ S @ S @ R.T, atol=1e-5, rtol=1e-3), (k, i)
        for i in range(Gb):  # background does not move
            R = matrix_from_quat(m["bg.params.quats"][i])
            S = np.diag(np.exp(m["bg.params.scales"][i]))
            assert np.allclose(seq.means[k][G + i], m["bg.params.means"][i], atol=1e-6)
            assert np.allclose(unpack_cov(seq.covs[k][G + i]), R @ S @ S @ R.T, atol=1e-5, rtol=1e-3)
    assert np.allclose(seq.opacities, sig(np.concatenate([m["fg.params.opacities"], m["bg.params.opacities"]])), atol=1e-6)
    assert np.allclose(seq.colors, sig(np.concatenate([m["fg.params.colors"], m["bg.params.colors"]])), atol=1e-6)
    assert np.abs(seq.means[4][:G] - seq.means[0][:G]).max() > 0.1
    print("ok  shape of motion checkpoint -> common format matches independent formulas at 3 frames (fg moves, bg static)")


if __name__ == "__main__":
    test_shape_of_motion()
    print("all checks passed")
