"""convert_4drotor.py against an independent formulation on a synthetic Nerfstudio checkpoint (needs torch; CPU).

The rotation matrix of the repo's rotor code is checked by properties (orthogonal for a unit rotor with zero pseudo-scalar part;
identity rotor -> identity; a pure xy bivector rotates only the xy plane). The slicing (conditional mean, conditional covariance,
marginal opacity) is checked against the textbook conditioning of a 4D Gaussian, with numpy linear algebra on the full 4x4 matrix.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import convert_4drotor as C  # noqa: E402
from gs4d import Sequence4D  # noqa: E402
from gs4d.format import unpack_cov  # noqa: E402


def unit_rotor(rng, n):
    r = rng.normal(size=(n, 8))
    a, bxy, bxz, byz, bxw, byw, bzw, p = r.T
    # remove the pseudo-scalar coupling eps = p a - bxy bzw + bxz byw - bxw byz by setting p from it (p = (bxy bzw - bxz byw + bxw byz)/a)
    r[:, 7] = (bxy * bzw - bxz * byw + bxw * byz) / a
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def test_rotation():
    rng = np.random.default_rng(0)
    r = unit_rotor(rng, 20)
    R = C.rotation_from_rotor8(r[:, :4], r[:, 4:])
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(4), atol=1e-9), "not orthogonal"
    assert np.allclose(np.linalg.det(R), 1, atol=1e-9)
    ident = np.zeros((1, 8)); ident[0, 0] = 1
    assert np.allclose(C.rotation_from_rotor8(ident[:, :4], ident[:, 4:])[0], np.eye(4))
    th = 0.7
    xy = np.zeros((1, 8)); xy[0, 0], xy[0, 1] = np.cos(th / 2), np.sin(th / 2)
    M = C.rotation_from_rotor8(xy[:, :4], xy[:, 4:])[0]
    assert np.allclose(M[2:, 2:], np.eye(2)) and np.allclose(M[:2, 2:], 0) and np.allclose(abs(M[0, 0]), np.cos(th), atol=1e-9), M
    print("ok  rotor -> 4x4 rotation: orthogonal, identity, pure xy rotation")


def make_checkpoint(path, n=60, seed=1):
    import torch
    rng = np.random.default_rng(seed)
    r = unit_rotor(rng, n)
    p = dict(xyz=rng.normal(size=(n, 3)), t=rng.uniform(0, 1, (n, 1)), scales_xyz=rng.normal(-2.5, 0.3, (n, 3)),
             scales_t=rng.normal(-1.5, 0.3, (n, 1)), quats1=r[:, :4], quats2=r[:, 4:], features_dc=rng.normal(0, 0.5, (n, 1, 3)),
             features_rest=rng.normal(0, 0.1, (n, 15, 3)), opacities=rng.normal(1, 1, (n, 1)))
    p = {k: v.astype(np.float32) for k, v in p.items()}
    sd = {"_model.gauss_params." + k: torch.from_numpy(v) for k, v in p.items()}
    torch.save({"step": 1, "pipeline": sd, "optimizers": {}}, path)
    return p


def main():
    test_rotation()
    import torch  # noqa: F401
    with tempfile.TemporaryDirectory() as tmp:
        ck, out = os.path.join(tmp, "step-000000001.ckpt"), os.path.join(tmp, "s.npz")
        p = make_checkpoint(ck)
        subprocess.run([sys.executable, os.path.join(ROOT, "convert_4drotor.py"), ck, "-o", out, "--frames", "7", "--min-peak-opacity", "0"],
                       check=True, capture_output=True)
        seq = Sequence4D.load(out)
        bad = subprocess.run([sys.executable, os.path.join(ROOT, "convert_4drotor.py"), out, "-o", out + "x.npz"], capture_output=True, text=True)
    n = len(p["xyz"])
    assert seq.num_frames == 7 and seq.num_gaussians == n
    s = np.exp(np.concatenate([p["scales_xyz"], p["scales_t"]], 1)).astype(np.float64)
    R = C.rotation_from_rotor8(p["quats1"].astype(np.float64), p["quats2"].astype(np.float64))
    for t in seq.times:
        means, covs, opac, cols = seq.at(float(t))[:4]
        covs = unpack_cov(covs) if covs.shape[-1] == 6 else covs
        for g in range(0, n, 7):
            L = R[g] @ np.diag(s[g]); S = L @ L.T                        # full 4D covariance
            A, b, d = S[:3, :3], S[:3, 3], S[3, 3]
            mu_c = p["xyz"][g] + b / d * (t - p["t"][g, 0])                # conditional mean of xyz given time t
            cov_c = A - np.outer(b, b) / d                                  # conditional covariance
            op = 1 / (1 + np.exp(-p["opacities"][g, 0])) * np.exp(-0.5 * (t - p["t"][g, 0]) ** 2 / d)   # marginal density in t
            assert np.allclose(means[g], mu_c, atol=1e-4), ("mean", g)
            assert np.allclose(covs[g], cov_c, atol=1e-6), ("cov", g)
            assert np.allclose(opac[g], op, atol=1e-5), ("opacity", g)
    assert np.allclose(cols, p["features_dc"][:, 0] * C.SH_C0 + 0.5, atol=1e-6)
    assert seq.sh is not None and seq.sh.shape[-2] == 16
    assert bad.returncode != 0 and "gauss_params" in (bad.stderr + bad.stdout) or bad.returncode != 0
    print("ok  convert_4drotor.py: conditional mean, covariance and marginal opacity equal the textbook conditioning of a 4D Gaussian; SH kept; wrong input stops")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip  4drotor (torch not installed)")
        sys.exit(0)
    main()
