"""convert_deformable3dgs.py against a reference forward pass, on a fake model folder in the real layout.

The reference is an nn.Module laid out like the repo's DeformNetwork (same parameter names, so its
state_dict is a real deform.pth) with random weights; the per-Gaussian position/covariance is then computed
in a plain numpy loop. Three variants: default, D-NeRF ("is_blender") time net, 6-DoF.
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

ATTRS = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(9)]
         + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])


def write_ply(path, cols):
    N = len(cols["x"])
    arr = np.zeros(N, dtype=[(n, "<f4") for n in ATTRS])
    for n, c in cols.items():
        arr[n] = c
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N
                 + "".join(f"property float {n}\n" for n in ATTRS) + "end_header\n").encode())
        f.write(arr.tobytes())


def build_reference(torch, blender, six_dof, D=8, W=256):
    nn = torch.nn

    def posenc(x, n):  # frequency-major, sin then cos, input first
        freqs = 2.0 ** torch.linspace(0.0, n - 1, n)
        parts = [x] + [fn(x * f) for f in freqs for fn in (torch.sin, torch.cos)]
        return torch.cat(parts, -1)

    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.tm = 6 if blender else 10
            xch, tch = 63, 1 + 2 * self.tm
            if blender:
                self.timenet = nn.Sequential(nn.Linear(tch, 256), nn.ReLU(), nn.Linear(256, 30))
                tch = 30
            self.inch = xch + tch
            self.linear = nn.ModuleList([nn.Linear(self.inch, W)] + [
                nn.Linear(W + self.inch, W) if i == D // 2 else nn.Linear(W, W) for i in range(D - 1)])
            if six_dof:
                self.branch_w, self.branch_v = nn.Linear(W, 3), nn.Linear(W, 3)
            else:
                self.gaussian_warp = nn.Linear(W, 3)
            self.gaussian_rotation, self.gaussian_scaling = nn.Linear(W, 4), nn.Linear(W, 3)

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
            head = (self.branch_w(h), self.branch_v(h)) if six_dof else self.gaussian_warp(h)
            return head, self.gaussian_rotation(h), self.gaussian_scaling(h)

    torch.manual_seed(1 + 2 * blender + 4 * six_dof)
    net = Ref()
    with torch.no_grad():  # default init gives tiny outputs; make the deformation clearly non-zero
        for name, p in net.named_parameters():
            if name.split(".")[0] in ("gaussian_warp", "gaussian_rotation", "gaussian_scaling", "branch_w", "branch_v"):
                p.mul_(4.0)
    return net


def rot_from_quat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                     [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                     [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])


def se3_point(w, v, p):
    """Point p moved by the repo's 6-DoF head: screw axis (w/|w| + 1e-5, v/|w| + 1e-5), magnitude |w|,
    closed form of the SE(3) exponential (Modern Robotics eq. 3.51 and 3.88) with plain 3x3 matrices."""
    theta = np.linalg.norm(w)
    w, v = w / theta + 1e-5, v / theta + 1e-5
    Wm = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + np.sin(theta) * Wm + (1 - np.cos(theta)) * Wm @ Wm
    G = theta * np.eye(3) + (1 - np.cos(theta)) * Wm + (theta - np.sin(theta)) * Wm @ Wm
    return R @ p + G @ v


def run_variant(torch, blender, six_dof):
    rng = np.random.default_rng(7)
    N, K = 50, 4
    c = {n: rng.normal(0, 0.5, N).astype(np.float32) for n in ATTRS}
    for i in range(3):
        c[f"scale_{i}"] = rng.normal(-2.5, 0.4, N).astype(np.float32)
    net = build_reference(torch, blender, six_dof)
    with tempfile.TemporaryDirectory() as tmp:
        for sub, it in (("point_cloud", 7000), ("point_cloud", 40000), ("deform", 7000), ("deform", 40000)):
            os.makedirs(os.path.join(tmp, sub, f"iteration_{it}"))
        write_ply(os.path.join(tmp, "point_cloud", "iteration_40000", "point_cloud.ply"), c)
        torch.save(net.state_dict(), os.path.join(tmp, "deform", "iteration_40000", "deform.pth"))
        out = os.path.join(tmp, "d.npz")
        subprocess.run([sys.executable, os.path.join(ROOT, "convert_deformable3dgs.py"), tmp, "-o", out, "--frames", str(K),
                        "--batch", "17", "--sh"], check=True, stdout=subprocess.DEVNULL)
        seq = Sequence4D.load(out)
    assert seq.num_frames == K and seq.num_gaussians == N and seq.sh.shape == (N, 4, 3)
    xyz = np.stack([c["x"], c["y"], c["z"]], -1)
    quat = np.stack([c[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    scale = np.exp(np.stack([c[f"scale_{i}"] for i in range(3)], -1).astype(np.float64))
    moved = 0.0
    for k in (0, 1, 3):
        t = float(seq.times[k])
        with torch.no_grad():
            head, d_rot, d_scale = net(torch.from_numpy(xyz), torch.full((N, 1), t))
        for i in range(N):
            if six_dof:
                p = se3_point(head[0][i].numpy().astype(np.float64), head[1][i].numpy().astype(np.float64), xyz[i].astype(np.float64))
            else:
                p = xyz[i] + head[i].numpy()
            R = rot_from_quat(quat[i] + d_rot[i].numpy())
            S = np.diag(scale[i] + d_scale[i].numpy())
            cov = R @ S @ S @ R.T
            assert np.allclose(seq.means[k][i], p, atol=2e-4), (k, i, seq.means[k][i], p)
            assert np.allclose(unpack_cov(seq.covs[k][i]), cov, atol=1e-4, rtol=1e-3), (k, i)
            moved = max(moved, float(np.abs(p - xyz[i]).max()))
    assert moved > 0.01, "reference deformation is ~0, the test would not see a missing MLP"
    assert np.allclose(seq.opacities, 1 / (1 + np.exp(-c["opacity"])), atol=1e-6)
    assert np.allclose(seq.colors, np.clip(0.28209479177387814 * np.stack([c[f"f_dc_{i}"] for i in range(3)], -1) + 0.5, 0, 1), atol=1e-6)
    name = ("D-NeRF time net" if blender else "default") + (" + 6-DoF" if six_dof else "")
    print(f"ok  deformable-3dgs ({name}): positions and covariances match the reference MLP at 3 times (max shift {moved:.2f})")


def test_deformable3dgs():
    try:
        import torch
    except ImportError:
        print("skip  deformable-3dgs (torch not installed)")
        return
    for blender, six_dof in ((False, False), (True, False), (False, True)):
        run_variant(torch, blender, six_dof)


if __name__ == "__main__":
    test_deformable3dgs()
    print("all checks passed")
