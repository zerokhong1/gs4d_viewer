"""convert_template.py against an independent formulation (full 3x3 covariance from R S S R^T), on a fake model, CPU only."""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D  # noqa: E402
from gs4d.format import unpack_cov  # noqa: E402


def rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    rng = np.random.default_rng(0)
    N = 50
    m = dict(xyz=rng.normal(size=(N, 3)), vel=rng.normal(size=(N, 3)), quat=rng.normal(size=(N, 4)),
             log_scale=rng.normal(-2, 0.3, (N, 3)), logit_opacity=rng.normal(size=N), rgb=rng.uniform(size=(N, 3)))
    with tempfile.TemporaryDirectory() as tmp:
        model, out = os.path.join(tmp, "m.npz"), os.path.join(tmp, "s.npz")
        np.savez(model, **m)
        subprocess.run([sys.executable, os.path.join(ROOT, "convert_template.py"), model, "-o", out, "--frames", "5",
                        "--t0", "0", "--t1", "2"], check=True, capture_output=True)
        seq = Sequence4D.load(out)
        bad = subprocess.run([sys.executable, os.path.join(ROOT, "convert_template.py"), out, "-o", out + "x.npz"],
                             capture_output=True, text=True)
    assert seq.num_frames == 5 and seq.num_gaussians == N and np.allclose(seq.times, np.linspace(0, 2, 5))
    for k, t in enumerate(seq.times):
        means, covs, opac, cols = seq.at(float(t))[:4]
        assert np.allclose(means, m["xyz"] + m["vel"] * t, atol=1e-5), "position"
        want = np.stack([rotmat(q) @ np.diag(np.exp(s)) @ np.diag(np.exp(s)) @ rotmat(q).T for q, s in zip(m["quat"], m["log_scale"])])
        assert np.allclose(unpack_cov(covs) if covs.shape[-1] == 6 else covs, want, atol=1e-6), "covariance"
        assert np.allclose(opac, 1 / (1 + np.exp(-m["logit_opacity"])), atol=1e-6), "opacity"
        assert np.allclose(cols, m["rgb"], atol=1e-6), "colour"
    assert bad.returncode != 0 and "missing" in (bad.stderr + bad.stdout), "unsupported input must stop with a message"
    print("ok  convert_template.py matches the independent formulation; wrong input stops with a message")


if __name__ == "__main__":
    main()
