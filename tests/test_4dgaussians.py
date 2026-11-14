"""convert_4dgaussians.py on a fake model folder in the real hustvl/4DGaussians layout (needs torch).

The reference below is written in numpy from the formulas (manual bilinear sampling, per-Gaussian loops), not
with torch's grid_sample, and the state_dict uses the key names of scene/deformation.py + scene/hexplane.py.
"""
import itertools
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D  # noqa: E402
from gs4d.format import unpack_cov  # noqa: E402

COMBOS = list(itertools.combinations(range(4), 2))


def make_state(rng, C=4, res=(5, 6, 7, 4), mults=(1, 2), W=8, D=2, bounds=1.6):
    sd = {}
    for L, m in enumerate(mults):
        reso = [r * m for r in res[:3]] + [res[3]]
        for P, comb in enumerate(COMBOS):
            sd[f"deformation_net.grid.grids.{L}.{P}"] = rng.normal(0, 0.7, (1, C, reso[comb[1]], reso[comb[0]])).astype(np.float32)
    sd["deformation_net.grid.aabb"] = np.array([[bounds] * 3, [-bounds] * 3], np.float32) * np.array(1.0, np.float32)
    feat = C * len(mults)
    dims = [feat] + [W] * D
    for i in range(D):
        sd[f"deformation_net.feature_out.{2 * i}.weight"] = rng.normal(0, 0.5, (dims[i + 1], dims[i])).astype(np.float32)
        sd[f"deformation_net.feature_out.{2 * i}.bias"] = rng.normal(0, 0.2, dims[i + 1]).astype(np.float32)
    for name, out in (("pos_deform", 3), ("scales_deform", 3), ("rotations_deform", 4), ("opacity_deform", 1),
                      ("shs_deform", 48)):
        sd[f"deformation_net.{name}.1.weight"] = rng.normal(0, 0.4, (W, W)).astype(np.float32)
        sd[f"deformation_net.{name}.1.bias"] = rng.normal(0, 0.1, W).astype(np.float32)
        sd[f"deformation_net.{name}.3.weight"] = rng.normal(0, 0.3, (out, W)).astype(np.float32)
        sd[f"deformation_net.{name}.3.bias"] = rng.normal(0, 0.1, out).astype(np.float32)
    return sd


def bilinear(grid, u, v):
    """grid [C,H,W]; u indexes W, v indexes H; both in [-1,1]; align_corners=True, border padding."""
    C, H, W = grid.shape
    fx = np.clip((u + 1) / 2 * (W - 1), 0, W - 1)
    fy = np.clip((v + 1) / 2 * (H - 1), 0, H - 1)
    x0, y0 = int(np.floor(fx)), int(np.floor(fy))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    ax, ay = fx - x0, fy - y0
    return (grid[:, y0, x0] * (1 - ax) * (1 - ay) + grid[:, y0, x1] * ax * (1 - ay)
            + grid[:, y1, x0] * (1 - ax) * ay + grid[:, y1, x1] * ax * ay)


def ref_forward(sd, flags, x, s, q, o, t, nlevels, D):
    pre = "deformation_net."
    aabb = sd[pre + "grid.aabb"].astype(np.float64)
    p = (x - aabb[0]) * 2.0 / (aabb[1] - aabb[0]) - 1.0
    pt = np.append(p, t)
    feats = []
    for L in range(nlevels):
        prod = 1.0
        for P, (a, b) in enumerate(COMBOS):
            prod = prod * bilinear(sd[f"{pre}grid.grids.{L}.{P}"][0].astype(np.float64), pt[a], pt[b])
        feats.append(prod)
    h = np.concatenate(feats)
    for i in range(D):
        h = sd[f"{pre}feature_out.{2 * i}.weight"].astype(np.float64) @ (np.maximum(h, 0) if i else h) + sd[f"{pre}feature_out.{2 * i}.bias"]

    def head(n):
        r = np.maximum(h, 0)
        r = np.maximum(sd[f"{pre}{n}.1.weight"] @ r + sd[f"{pre}{n}.1.bias"], 0)
        return sd[f"{pre}{n}.3.weight"] @ r + sd[f"{pre}{n}.3.bias"]

    if not flags["no_dx"]:
        x = x + head("pos_deform")
    if not flags["no_ds"]:
        s = s + head("scales_deform")
    if not flags["no_dr"]:
        dr = head("rotations_deform")
        if flags["apply_rotation"]:
            w1, x1, y1, z1 = q
            w2, x2, y2, z2 = dr
            q = np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                          w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])
        else:
            q = q + dr
    if not flags["no_do"]:
        o = o + head("opacity_deform")[0]
    return x, s, q, o


def cov_ref(q, s):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                  [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                  [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])
    M = R * np.exp(s)[None, :]
    return M @ M.T


def write_ply(path, cols):
    names = list(cols)
    N = len(cols["x"])
    arr = np.zeros(N, dtype=[(n, "<f4") for n in names])
    for n in names:
        arr[n] = cols[n]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N
                 + "".join(f"property float {n}\n" for n in names) + "end_header\n").encode())
        f.write(arr.tobytes())


def run_case(flags, tmp, name):
    import torch

    rng = np.random.default_rng(7)
    nlevels, D, N = 2, 2, 40
    sd = make_state(rng, D=D, mults=(1, 2)[:nlevels])
    root = os.path.join(tmp, name, "point_cloud", "iteration_20000")
    os.makedirs(root)
    cols = {"x": rng.uniform(-1, 1, N), "y": rng.uniform(-1, 1, N), "z": rng.uniform(-1, 1, N)}
    for n in ("nx", "ny", "nz"):
        cols[n] = np.zeros(N)
    for i in range(3):
        cols[f"f_dc_{i}"] = rng.normal(0, 0.5, N)
    for i in range(45):
        cols[f"f_rest_{i}"] = rng.normal(0, 0.1, N)
    cols["opacity"] = rng.normal(0.5, 1.0, N)
    for i in range(3):
        cols[f"scale_{i}"] = rng.normal(-2.5, 0.3, N)
    for i in range(4):
        cols[f"rot_{i}"] = rng.normal(size=N)
    cols = {k: np.asarray(v, np.float32) for k, v in cols.items()}
    write_ply(os.path.join(root, "point_cloud.ply"), cols)
    torch.save({k: torch.tensor(v) for k, v in sd.items()}, os.path.join(root, "deformation.pth"))
    with open(os.path.join(tmp, name, "cfg_args"), "w") as f:
        f.write("Namespace(" + ", ".join(f"{k}={v}" for k, v in flags.items()) + ", net_width=8)")
    out = os.path.join(tmp, name + ".npz")
    subprocess.run([sys.executable, os.path.join(ROOT, "convert_4dgaussians.py"), os.path.join(tmp, name), "-o", out,
                    "--frames", "5"], check=True, stdout=subprocess.DEVNULL)
    seq = Sequence4D.load(out)
    assert seq.num_frames == 5 and seq.num_gaussians == N
    worst = 0.0
    for k in (0, 2, 4):  # t = 0, 0.5, 1
        t = float(seq.times[k])
        for i in range(N):
            x = np.array([cols["x"][i], cols["y"][i], cols["z"][i]], np.float64)
            s = np.array([cols[f"scale_{j}"][i] for j in range(3)], np.float64)
            q = np.array([cols[f"rot_{j}"][i] for j in range(4)], np.float64)
            xr, sr, qr, orr = ref_forward(sd, flags, x, s, q, float(cols["opacity"][i]), t, nlevels, D)
            op = 1 / (1 + np.exp(-orr))
            worst = max(worst, np.abs(seq.means[k][i] - xr).max(),
                        np.abs(unpack_cov(seq.covs[k][i]) - cov_ref(qr, sr)).max(), abs((seq.opacities[k] if seq.opacities.ndim == 2 else seq.opacities)[i] - op))
    assert worst < 2e-4, (name, worst)
    return worst


def test_4dgaussians():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip  4DGaussians converter (torch not installed)")
        return
    base = dict(no_dx=False, no_ds=False, no_dr=False, no_do=True, no_dshs=True, no_grid=False,
                empty_voxel=False, static_mlp=False, apply_rotation=False)
    with tempfile.TemporaryDirectory() as tmp:
        w1 = run_case(base, tmp, "default")
        w2 = run_case(dict(base, no_do=False, apply_rotation=True), tmp, "opacity_rotmul")
        w3 = run_case(dict(base, no_ds=True, no_dr=True), tmp, "position_only")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "convert_4dgaussians.py"), os.path.join(tmp, "default"),
                            "-o", os.path.join(tmp, "x.npz"), "--flag", "static_mlp=True"], capture_output=True, text=True)
        assert r.returncode != 0 and "static_mlp" in (r.stderr + r.stdout)
    print(f"ok  4DGaussians hexplane + MLP matches the independent formulas at 3 times "
          f"(max error {max(w1, w2, w3):.1e}; default, opacity+quaternion-product, position-only variants; unsupported variant stops)")


if __name__ == "__main__":
    test_4dgaussians()
