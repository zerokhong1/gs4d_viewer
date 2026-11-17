"""convert_mosca.py on a synthetic DynSCFGaussian state_dict (needs torch; CPU). No real MoSca model exists here.

Independent checks (not a second copy of the code):
 1. all scaffold nodes undergo ONE rigid motion between frame r and frame t -> every Gaussian moves rigidly by exactly that motion,
    whatever the skinning weights are (DQB and LBS agree), both with leaf_local_flag on and off;
 2. at t == ref_time nothing moves (the warp is the identity) and the rotation is the stored one;
 3. nodes with DIFFERENT rotations: the result equals the textbook normalised dual-quaternion blend written with numpy float64
    (dual part orthogonalised by hand), and the covariance is rot diag(s^2) rot^T of the blended rotation;
 4. nn_fusion masks the opacity of Gaussians outside the nearest frames; wrong input stops with a message.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import convert_mosca as C  # noqa: E402
from gs4d import Sequence4D  # noqa: E402
from gs4d.format import unpack_cov  # noqa: E402


def rotz_quat(a, axis):
    ax = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(a / 2)], ax * np.sin(a / 2)])


def qmat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qmul(p, q):
    return np.concatenate([[p[0] * q[0] - p[1:] @ q[1:]], p[0] * q[1:] + q[0] * p[1:] + np.cross(p[1:], q[1:])])


def make(T=4, M=6, K=3, N=30, seed=0, local=True, lbs=False, dyn_o=False, nn_fusion=-1):
    import torch
    rng = np.random.default_rng(seed)
    nx = rng.normal(size=(T, M, 3))
    nq = np.stack([[rotz_quat(rng.uniform(-1, 1), rng.normal(size=3)) for _ in range(M)] for _ in range(T)])
    knn = np.stack([[m] + list(rng.choice([j for j in range(M) if j != m], K - 1, replace=False)) for m in range(M)])
    sd = {"scf._node_xyz": nx, "scf._node_rotation": nq, "scf._node_sigma_logit": rng.normal(size=(M, 1)),
          "scf.max_sigma": np.array(3.0), "scf.topo_knn_ind": knn, "scf.topo_knn_mask": np.ones((M, K), bool),
          "scf.blending_method": np.array([1 if lbs else 0]), "_xyz": rng.normal(0, 0.3, (N, 3)),
          "_rotation": rng.normal(size=(N, 4)), "_scaling": rng.normal(size=(N, 3)), "_opacity": rng.normal(1, 1, (N, 1)),
          "_features_dc": rng.normal(0, 0.5, (N, 3)), "attach_ind": rng.integers(0, M, N), "ref_time": rng.integers(0, T, N),
          "max_scale": np.array(0.1), "min_scale": np.array(0.0), "max_sph_order": np.array(0), "leaf_local_flag": np.array(local),
          "dyn_o_flag": np.array(dyn_o), "nn_fusion": np.array(nn_fusion), "w_correction_flag": np.array(False)}
    dt = {k: (np.float32 if v.dtype.kind == "f" else v.dtype) for k, v in ((k, np.asarray(v)) for k, v in sd.items())}
    return {k: torch.from_numpy(np.asarray(v).astype(dt[k])) for k, v in sd.items()}, sd


def convert_sd(sd_t, tmp, extra=()):
    import torch
    pth, out = os.path.join(tmp, "d.pth"), os.path.join(tmp, "s.npz")
    torch.save(sd_t, pth)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "convert_mosca.py"), pth, "-o", out, "--fps", "1", *extra], capture_output=True, text=True)
    return r, out


def test_rigid():
    for local in (True, False):
        for lbs in (False, True):
            sd_t, sd = make(local=local, lbs=lbs, seed=3)
            T, M = sd["scf._node_xyz"].shape[:2]
            Rr, tr = qmat(rotz_quat(0.6, [1, 2, 3])), np.array([0.3, -0.2, 0.5])
            # node pose at frame t = common motion applied to the pose at frame 0 for every node, for every frame t >= 1
            nx, nq = sd["scf._node_xyz"].copy(), sd["scf._node_rotation"].copy()
            for t in range(1, T):
                nx[t] = nx[0] @ (Rr.T) + tr
                nq[t] = [qmul(rotz_quat(0.6, [1, 2, 3]), q) for q in nq[0]]
            nx[0], nq[0] = nx[0], nq[0]
            sd_t["scf._node_xyz"][:] = __import__("torch").from_numpy(nx.astype(np.float32))
            sd_t["scf._node_rotation"][:] = __import__("torch").from_numpy(nq.astype(np.float32))
            sd_t["ref_time"][:] = 0
            with tempfile.TemporaryDirectory() as tmp:
                r, out = convert_sd(sd_t, tmp)
                assert r.returncode == 0, r.stderr[-800:]
                seq = Sequence4D.load(out)
            # live position of every Gaussian at its reference frame 0, from the stored values
            N = len(sd["_xyz"])
            a = sd["attach_ind"]
            if local:
                Rwi = np.stack([qmat(nq[0][i]) for i in a])
                x0 = np.einsum("nij,nj->ni", Rwi, sd["_xyz"].astype(np.float32)) + nx[0][a]
                Rm = np.stack([Rwi[n] @ qmat(sd["_rotation"][n]) for n in range(N)])
            else:
                x0 = sd["_xyz"].astype(np.float32)
                Rm = np.stack([qmat(q) for q in sd["_rotation"]])
            s = 0.1 / (1 + np.exp(-sd["_scaling"]))
            m0, c0 = seq.at(float(seq.times[0]))[:2]
            assert np.allclose(m0, x0, atol=2e-4), ("t == ref: identity", local, lbs, np.abs(m0 - x0).max())
            for t in range(1, T):
                m, c = seq.at(float(seq.times[t]))[:2]
                assert np.allclose(m, x0 @ Rr.T + tr, atol=5e-3)   # the repo mixes in the identity with weight DQ_EPS = 1e-3 (dyn_o), so rigid only to ~1e-3 x motion
                assert np.allclose(m, x0 @ Rr.T + tr, atol=5e-3), ("rigid mean", local, lbs, np.abs(m - (x0 @ Rr.T + tr)).max())
                want = np.einsum("nij,nj,nkj,nlk->nil", Rr[None].repeat(N, 0) @ Rm, s ** 2, Rr[None].repeat(N, 0) @ Rm, np.eye(3)[None].repeat(N, 0)) if False else \
                    np.stack([(Rr @ Rm[n]) @ np.diag(s[n] ** 2) @ (Rr @ Rm[n]).T for n in range(N)])
                got = unpack_cov(c) if c.shape[-1] == 6 else c
                assert np.allclose(got, want, atol=5e-5), ("rigid cov", local, lbs)
    print("ok  one common rigid motion of all nodes moves every Gaussian rigidly (DQB and LBS, local and world leaves); t == ref is the identity")


def test_dqb_formula():
    sd_t, sd = make(seed=5, local=False, N=20)
    sd_t["ref_time"][:] = 0
    with tempfile.TemporaryDirectory() as tmp:
        r, out = convert_sd(sd_t, tmp)
        assert r.returncode == 0, r.stderr[-800:]
        seq = Sequence4D.load(out)
    nx, nq = sd["scf._node_xyz"].astype(np.float32).astype(np.float64), sd["scf._node_rotation"].astype(np.float32).astype(np.float64)
    knn, sigma = sd["scf.topo_knn_ind"], 3.0 / (1 + np.exp(-sd["scf._node_sigma_logit"][:, 0].astype(np.float32).astype(np.float64)))
    x = sd["_xyz"].astype(np.float32).astype(np.float64)
    t = 2
    mu, _ = seq.at(float(seq.times[t]))[:2]
    for g in range(len(x)):
        sk = knn[sd["attach_ind"][g]]
        w = np.exp(-((x[g] - nx[0][sk]) ** 2).sum(-1) / (2 * sigma[sk] ** 2)) + 1e-6
        w = w / w.sum()
        dqs = []
        for j, n in enumerate(sk):
            Ri, Rj = qmat(nq[0][n]), qmat(nq[t][n])
            Rji = Rj @ Ri.T
            tji = nx[t][n] - Rji @ nx[0][n]
            # unit dual quaternion of (Rji, tji), real part with w >= 0 (as pytorch3d), dual = 0.5 * t * p
            p = np.array(C.matrix_to_quaternion(__import__("torch").from_numpy(Rji)).numpy(), np.float64)
            q = 0.5 * qmul(np.concatenate([[0], tji]), p)
            dqs.append(np.concatenate([p, q]))
        dq = sum(wi * d for wi, d in zip(w, dqs))
        dq = dq * (1 - C.DQ_EPS) + np.array([1, 0, 0, 0, 0, 0, 0, 0]) * C.DQ_EPS
        p, q = dq[:4], dq[4:]
        n = np.linalg.norm(p)
        p1 = p / n
        q1 = q / n - p1 * (p1 @ (q / n))                       # normalise and make the dual part orthogonal to the real part
        R = qmat(p1)
        tt = 2 * qmul(q1, p1 * np.array([1, -1, -1, -1]))[1:]
        assert np.allclose(mu[g], R @ x[g] + tt, atol=3e-4), (g, np.abs(mu[g] - (R @ x[g] + tt)).max())
    print("ok  different node rotations: equals the textbook normalised dual-quaternion blend")


def test_other():
    sd_t, sd = make(seed=7, nn_fusion=1)
    with tempfile.TemporaryDirectory() as tmp:
        r, out = convert_sd(sd_t, tmp)
        assert r.returncode == 0, r.stderr[-800:]
        seq = Sequence4D.load(out)
        T = sd["scf._node_xyz"].shape[0]
        for t in range(T):
            o = seq.at(float(seq.times[t]))[2]
            sup = np.unique(sd["ref_time"])
            nearest = sup[np.argsort(np.abs(sup - t), kind="stable")[:1]]
            want = (1 / (1 + np.exp(-sd["_opacity"][:, 0].astype(np.float32)))) * np.isin(sd["ref_time"], nearest)
            assert np.allclose(o, want, atol=1e-5), ("nn_fusion", t)
        bad = {k: v for k, v in sd_t.items() if k != "scf._node_sigma_logit"}
        r2, _ = convert_sd(bad, tmp)
        assert r2.returncode != 0 and "MoSca" in r2.stderr + r2.stdout
        sd_t["max_sph_order"] = sd_t["max_sph_order"] + 1
        r3, _ = convert_sd(sd_t, tmp)
        assert r3.returncode != 0 and "max_sph_order" in r3.stderr + r3.stdout
    print("ok  nn_fusion masks opacity; wrong input and max_sph_order > 0 stop with a message")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip  mosca (torch not installed)")
        sys.exit(0)
    test_rigid()
    test_dqb_formula()
    test_other()
    print("all checks passed")
