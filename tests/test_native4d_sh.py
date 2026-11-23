"""convert_native4d.py --sh against the repo's eval_shfs_4d, written out term by term (needs torch for the pickle)."""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import SH_C0, Sequence4D  # noqa: E402

C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]


def basis16(d):
    """The 16 spatial terms of utils/sh_utils.py::eval_shfs_4d, in its order (l0m0 ... l3p3). d: unit [3]."""
    x, y, z = d
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return np.array([SH_C0, -C1 * y, C1 * z, -C1 * x,
                     C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy),
                     C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
                     C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                     C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)])


def reference_rgb_minus_half(sh48, d, dt, T):
    """eval_shfs_4d(deg=3, deg_t=2): sh48 [48,3] -> [3]. Temporal terms sit at channels 16..31 (cos 2pi dt/T) and 32..47."""
    B = basis16(d)
    out = B @ sh48[0:16]
    out = out + np.cos(2 * np.pi * dt / T) * (B @ sh48[16:32])
    out = out + np.cos(2 * np.pi * 2 * dt / T) * (B @ sh48[32:48])
    return out


def test_native4d_sh():
    import torch

    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    N, T = 120, 1.0
    f_dc, f_rest = r(N, 1, 3), r(N, 47, 3) * 0.3
    t_c = torch.rand(N, 1, generator=g)
    cap = (3, r(N, 3), f_dc, f_rest, r(N, 3) * 0.3 - 2, r(N, 4), r(N, 1) + 2, torch.zeros(N),
           torch.zeros(N, 1), torch.zeros(N, 1), torch.zeros(N, 1), {}, 1.0,
           t_c, r(N, 1) * 0.3 - 1, r(N, 4), True, None, 2)
    with tempfile.TemporaryDirectory() as tmp:
        ck, out = os.path.join(tmp, "chkpnt_best.pth"), os.path.join(tmp, "n4d_sh.npz")
        torch.save((cap, 20000), ck)
        subprocess.run([sys.executable, os.path.join(ROOT, "converters", "convert_native4d.py"), ck, "-o", out, "--frames", "5",
                        "--t0", "0", "--t1", "1", "--sh", "--min-peak-opacity", "-1"], check=True,
                       stdout=subprocess.DEVNULL)
        seq = Sequence4D.load(out)
    assert seq.sh is not None and seq.sh.shape == (5, N, 16, 3), seq.sh.shape
    full = np.concatenate([f_dc.numpy(), f_rest.numpy()], 1).astype(np.float64)  # [N,48,3]
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(3, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    worst = 0.0
    for k in (0, 2, 4):  # t = 0, 0.5, 1
        tk = float(seq.times[k])
        for d in dirs:
            B = basis16(d)
            for i in range(N):
                ref = reference_rgb_minus_half(full[i], d, t_c[i, 0].item() - tk, T)
                got = B @ np.asarray(seq.sh[k][i], np.float64)  # plain 3D SH evaluation of the converted file
                worst = max(worst, np.abs(ref - got).max())
        # colors = degree-0 part at that keyframe
        dc_t = (full[:, 0] + np.cos(2 * np.pi * (t_c.numpy()[:, 0] - tk) / T)[:, None] * full[:, 16]
                + np.cos(4 * np.pi * (t_c.numpy()[:, 0] - tk) / T)[:, None] * full[:, 32])
        assert np.allclose(seq.colors[k], np.clip(SH_C0 * dc_t + 0.5, 0, 1), atol=1e-5)
    assert worst < 1e-4, worst
    print(f"ok  native 4DGS --sh matches eval_shfs_4d at 3 times x 3 directions (max error {worst:.1e})")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip  native 4DGS --sh (torch not installed)")
        sys.exit(0)
    test_native4d_sh()
