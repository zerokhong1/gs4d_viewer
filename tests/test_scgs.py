"""convert_scgs.py against an independent formulation, on a fake model folder in the real SC-GS layout.

The node MLP reference is an nn.Module laid out like the repo's DeformNetwork (same parameter names), with
random weights; nearest nodes, blending weights, local-frame warp, rotation and covariance are then computed
per Gaussian in a plain numpy loop. Two variants:
  1. D-NeRF time net, local frame, 8 hyper coordinates, node weights          (the repo's recommended setup)
  2. plain time embedding, no local frame, no hyper coordinates, motion mask, opacity head, max_d_scale, K=4
Needs torch (CPU); skipped without it.
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


def write_ply(path, cols, attrs):
    N = len(cols["x"])
    arr = np.zeros(N, dtype=[(n, "<f4") for n in attrs])
    for n in attrs:
        arr[n] = cols[n]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N
                 + "".join(f"property float {n}\n" for n in attrs) + "end_header\n").encode())
        f.write(arr.tobytes())


def build_network(torch, blender, local_frame, pred_opacity, D=8, W=256):
    nn = torch.nn

    def posenc(x, n):
        freqs = 2.0 ** torch.linspace(0.0, n - 1, n)
        return torch.cat([x] + [fn(x * f) for f in freqs for fn in (torch.sin, torch.cos)], -1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.tm = 6 if blender else 10
            tch = 1 + 2 * self.tm
            if blender:
                self.timenet = nn.Sequential(nn.Linear(tch, 256), nn.ReLU(), nn.Linear(256, 30))
                tch = 30
            inch = 63 + tch
            self.linear = nn.ModuleList([nn.Linear(inch, W)] + [
                nn.Linear(W + inch, W) if i == D // 2 else nn.Linear(W, W) for i in range(D - 1)])
            self.gaussian_warp, self.gaussian_scaling, self.gaussian_rotation = nn.Linear(W, 3), nn.Linear(W, 3), nn.Linear(W, 4)
            if local_frame:
                self.local_rotation = nn.Linear(W, 4)
            if pred_opacity:
                self.gaussian_opacity = nn.Linear(W, 1)

        def forward(self, x, t):
            te = posenc(t, self.tm)
            if blender:
                te = self.timenet(te)
            xe = posenc(x, 10)
            h = torch.cat([xe, te], -1)
            for i, layer in enumerate(self.linear):
                h = torch.relu(layer(h))
                if i == D // 2:
                    h = torch.cat([xe, te, h], -1)
            out = {"d_xyz": self.gaussian_warp(h), "d_scale": self.gaussian_scaling(h), "d_rot": self.gaussian_rotation(h)}
            if local_frame:
                out["local_rot"] = self.local_rotation(h)
            if pred_opacity:
                out["d_opacity"] = self.gaussian_opacity(h)
            return out

    torch.manual_seed(3 + blender)
    net = Net()
    with torch.no_grad():  # make the heads clearly non-zero
        for name, p in net.named_parameters():
            if name.split(".")[0] in ("gaussian_warp", "gaussian_rotation", "gaussian_scaling", "local_rotation", "gaussian_opacity"):
                p.mul_(4.0)
    return net


def rot_from_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                     [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                     [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])


def run_variant(torch, blender, local_frame, H, motion_mask, pred_opacity, K, max_d_scale):
    rng = np.random.default_rng(11 + H)
    N, M, F = 60, 20, 4
    n_fea = H + (1 if motion_mask else 0)
    attrs = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(9)]
             + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)] + [f"fea_{i}" for i in range(n_fea)])
    c = {n: rng.normal(0, 0.5, N).astype(np.float32) for n in attrs}
    for i in range(3):
        c[f"scale_{i}"] = rng.normal(-2.5, 0.4, N).astype(np.float32)
    net = build_network(torch, blender, local_frame, pred_opacity)
    sd = {f"network.{k}": v for k, v in net.state_dict().items()}
    sd["nodes"] = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.5, (M, 3 + H)).astype(np.float32)))
    sd["_node_radius"] = torch.nn.Parameter(torch.from_numpy(rng.normal(-0.5, 0.3, M).astype(np.float32)))
    sd["_node_weight"] = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 1, (M, 1)).astype(np.float32)))
    sd["inited"] = torch.tensor(True)
    sd["gs__xyz"] = torch.nn.Parameter(torch.zeros(M, 3))  # the repo also stores its per-node Gaussians as gs_*
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("point_cloud", "deform"):
            os.makedirs(os.path.join(tmp, sub, "iteration_80000"))
        write_ply(os.path.join(tmp, "point_cloud", "iteration_80000", "point_cloud.ply"), c, attrs)
        torch.save(sd, os.path.join(tmp, "deform", "iteration_80000", "deform.pth"))
        out = os.path.join(tmp, "s.npz")
        cmd = [sys.executable, os.path.join(ROOT, "convert_scgs.py"), tmp, "-o", out, "--frames", str(F), "--knn", str(K)]
        if max_d_scale > 0:
            cmd += ["--max-d-scale", str(max_d_scale)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        seq = Sequence4D.load(out)
    assert seq.num_frames == F and seq.num_gaussians == N

    nodes = sd["nodes"].detach().numpy().astype(np.float64)
    radius = np.exp(sd["_node_radius"].detach().numpy().astype(np.float64))
    nweight = 1 / (1 + np.exp(-sd["_node_weight"].detach().numpy().astype(np.float64)[:, 0]))
    xyz = np.stack([c["x"], c["y"], c["z"]], -1).astype(np.float64)
    fea = np.stack([c[f"fea_{i}"] for i in range(n_fea)], -1).astype(np.float64) if n_fea else np.zeros((N, 0))
    quat = np.stack([c[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)
    scale = np.exp(np.stack([c[f"scale_{i}"] for i in range(3)], -1).astype(np.float64))
    opac0 = 1 / (1 + np.exp(-c["opacity"].astype(np.float64)))
    moved = 0.0
    for k in (0, 1, 3):
        t = float(seq.times[k])
        with torch.no_grad():
            v = {key: val.numpy().astype(np.float64) for key, val in
                 net(torch.from_numpy(nodes[:, :3].astype(np.float32)), torch.full((M, 1), t)).items()}
        if max_d_scale > 0:
            v["d_scale"] = np.tanh(v["d_scale"]) * np.log(max_d_scale)
        for i in range(N):
            p = np.concatenate([xyz[i], fea[i, :H]])
            d2 = ((nodes - p) ** 2).sum(-1)
            nn_ = np.argsort(d2)[:K]
            wk = np.exp(-d2[nn_] / (2 * radius[nn_] ** 2)) * nweight[nn_] + 1e-7
            wk /= wk.sum()
            m = 1 / (1 + np.exp(-fea[i, -1])) if motion_mask else 1.0
            if local_frame:
                acc = np.zeros(3)
                for j, wj in zip(nn_, wk):
                    Rj = rot_from_quat(v["local_rot"][j] + np.array([1.0, 0, 0, 0]))
                    acc += wj * (Rj @ (xyz[i] - nodes[j, :3]) + nodes[j, :3] + v["d_xyz"][j])
                shift = acc - xyz[i]
            else:
                shift = sum(wj * v["d_xyz"][j] for j, wj in zip(nn_, wk))
            pos = xyz[i] + m * shift
            R = rot_from_quat(quat[i] + m * sum(wj * v["d_rot"][j] for j, wj in zip(nn_, wk)))
            S = np.diag(scale[i] + m * sum(wj * v["d_scale"][j] for j, wj in zip(nn_, wk)))
            assert np.allclose(seq.means[k][i], pos, atol=2e-4), (k, i, seq.means[k][i], pos)
            assert np.allclose(unpack_cov(seq.covs[k][i]), R @ S @ S @ R.T, atol=1e-4, rtol=2e-3), (k, i)
            if pred_opacity:
                o = np.clip(opac0[i] + m * sum(wj * v["d_opacity"][j, 0] for j, wj in zip(nn_, wk)), 0, 1)
                assert abs(seq.opacities[k][i] - o) < 1e-4, (k, i)
            moved = max(moved, float(np.abs(pos - xyz[i]).max()))
    if not pred_opacity:
        assert seq.opacities.ndim == 1 and np.allclose(seq.opacities, opac0, atol=1e-6)
    assert moved > 0.01, "reference deformation is ~0, the test would not see a missing MLP"
    name = ", ".join(n for n, on in (("D-NeRF time net", blender), ("local frame", local_frame), (f"{H} hyper coordinates", H),
                                     ("motion mask", motion_mask), ("opacity head", pred_opacity), (f"K={K}", True)) if on)
    print(f"ok  sc-gs ({name}): positions and covariances match the independent formulas at 3 times (max shift {moved:.2f})")


def test_scgs():
    try:
        import torch
    except ImportError:
        print("skip  sc-gs (torch not installed)")
        return
    run_variant(torch, blender=True, local_frame=True, H=8, motion_mask=False, pred_opacity=False, K=3, max_d_scale=-1)
    run_variant(torch, blender=False, local_frame=False, H=0, motion_mask=True, pred_opacity=True, K=4, max_d_scale=2.0)


if __name__ == "__main__":
    test_scgs()
    print("all checks passed")
