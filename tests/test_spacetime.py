"""convert_spacetime.py against an independent formulation, on a fake ply in the real Spacetime layout."""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D  # noqa: E402
from gs4d.format import unpack_cov  # noqa: E402

# real lite layout (as in the two public lite models); the full model adds f_dc_3..8 and f_t_*
ATTRS = (["x", "y", "z", "trbf_center", "trbf_scale", "nx", "ny", "nz"] + [f"motion_{i}" for i in range(9)]
         + [f"f_dc_{i}" for i in range(3)] + ["opacity"] + [f"scale_{i}" for i in range(3)]
         + [f"rot_{i}" for i in range(4)] + [f"omega_{i}" for i in range(4)])
FULL_EXTRA = [f"f_dc_{i}" for i in range(3, 9)] + [f"f_t_{i}" for i in range(3)]


def write_ply(path, cols, full=False):
    attrs = ATTRS + FULL_EXTRA if full else ATTRS
    N = len(cols["x"])
    arr = np.zeros(N, dtype=[(n, "<f4") for n in attrs])
    for n, c in cols.items():
        arr[n] = c
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N
                 + "".join(f"property float {n}\n" for n in attrs) + "end_header\n").encode())
        f.write(arr.tobytes())


def reference(c, t):
    """Per-Gaussian python loop straight from the formulas in the repo's renderer."""
    out = []
    for i in range(len(c["x"])):
        dt = t - c["trbf_center"][i]
        p = np.array([c["x"][i], c["y"][i], c["z"][i]], np.float64)
        mo = np.array([c[f"motion_{k}"][i] for k in range(9)], np.float64)
        p = p + mo[0:3] * dt + mo[3:6] * dt**2 + mo[6:9] * dt**3
        q = np.array([c[f"rot_{k}"][i] + dt * c[f"omega_{k}"][i] for k in range(4)], np.float64)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                      [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                      [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])
        S = np.diag(np.exp([c[f"scale_{k}"][i] for k in range(3)]))
        cov = R @ S @ S @ R.T
        o = 1 / (1 + np.exp(-c["opacity"][i])) * np.exp(-(dt / np.exp(c["trbf_scale"][i])) ** 2)
        out.append((p, cov, o))
    return out


def test_spacetime():
    rng = np.random.default_rng(0)
    N = 60
    c = {n: rng.normal(0, 0.3, N).astype(np.float32) for n in ATTRS}
    c["trbf_center"] = rng.uniform(0, 1, N).astype(np.float32)
    c["trbf_scale"] = rng.normal(-0.5, 0.2, N).astype(np.float32)
    c["opacity"] = rng.normal(1, 1, N).astype(np.float32)
    for k in range(4):
        c[f"rot_{k}"] = rng.normal(size=N).astype(np.float32)
    c["f_dc_0"] = np.full(N, 0.25, np.float32)
    c["f_dc_1"] = np.full(N, 0.5, np.float32)
    c["f_dc_2"] = np.full(N, 1.5, np.float32)  # out of range: kept, the repo blends raw colours and clamps the pixel
    with tempfile.TemporaryDirectory() as tmp:
        ply, out = os.path.join(tmp, "point_cloud.ply"), os.path.join(tmp, "st.npz")
        write_ply(ply, c)
        subprocess.run([sys.executable, os.path.join(ROOT, "convert_spacetime.py"), ply, "-o", out, "--frames", "5",
                        "--min-peak-opacity", "-1"], check=True, stdout=subprocess.DEVNULL)
        seq = Sequence4D.load(out)
    assert seq.num_frames == 5 and seq.num_gaussians == N
    for k in (0, 2, 4):  # t = 0, 0.5, 1
        ref = reference(c, float(seq.times[k]))
        for i in range(N):
            p, cov, o = ref[i]
            assert np.allclose(seq.means[k][i], p, atol=1e-4), (k, i)
            assert np.allclose(unpack_cov(seq.covs[k][i]), cov, atol=1e-4), (k, i)
            assert abs(seq.opacities[k][i] - o) < 1e-5, (k, i)
    assert np.allclose(seq.colors[0], [0.25, 0.5, 1.5])
    print("ok  spacetime ply -> common format matches independent formulas at 3 times")
    test_full_model_stops(c)


def test_full_model_stops(c):
    """The full model (f_dc_3.. in the ply, or an rgbdecoder file beside it) is refused unless --base-color-only."""
    script = os.path.join(ROOT, "convert_spacetime.py")
    for how in ("f_dc_3", "rgbdecoder"):
        with tempfile.TemporaryDirectory() as tmp:
            ply, out = os.path.join(tmp, "point_cloud.ply"), os.path.join(tmp, "st.npz")
            cc = dict(c)
            if how == "f_dc_3":
                write_ply(ply, cc, full=True)
            else:
                write_ply(ply, cc)
                open(os.path.join(tmp, "rgbdecoder.pt"), "wb").write(b"x")
            cmd = [sys.executable, script, ply, "-o", out, "--frames", "2"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            assert r.returncode != 0 and "full model (rgbdecoder)" in r.stderr and "--base-color-only" in r.stderr, (how, r.stderr)
            assert not os.path.exists(out)
            subprocess.run(cmd + ["--base-color-only"], check=True, capture_output=True)
            assert Sequence4D.load(out).num_frames == 2
    print("ok  spacetime full model stops, --base-color-only converts")


if __name__ == "__main__":
    test_spacetime()
