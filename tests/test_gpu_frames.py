#!/usr/bin/env python3
"""gs4d.GpuFrames (the decoder of the gsplat renderer) against Sequence4D.at / sh_at, run on torch's CPU device, so it
needs torch but not CUDA. The renderer uses the same code on the GPU.

    python tests/test_gpu_frames.py

Covers every layout GpuFrames reads: plain float32 and compact (float16 offsets, int16 Cholesky factors, uint8, per-keyframe
sh as uint8), the static group (one block at either end, or scattered), fields without a keyframe axis, files whose
keyframes do not share Gaussians (interpolate=False), and the bound on the keyframe cache.
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import GpuFrames, Sequence4D, cov_from_quat_scale  # noqa: E402

K, N = 6, 500


def scene(rng, static=(), sh="keyframe", colours="constant", moving_covs=True):
    """Random sequence; the Gaussians in `static` never change; sh: None / "constant" / "keyframe"."""
    times = np.cumsum(rng.uniform(0.2, 0.7, K)).astype(np.float32)
    means = np.cumsum(rng.normal(0, 0.1, (K, N, 3)), 0).astype(np.float32)
    q = rng.normal(size=(K if moving_covs else 1, N, 4)).astype(np.float32)
    s = np.exp(rng.uniform(np.log(0.01), np.log(0.3), (K if moving_covs else 1, N, 3))).astype(np.float32)
    covs = cov_from_quat_scale(q, s)
    covs = covs if moving_covs else covs[0]
    opac = rng.uniform(0.05, 1, (K, N)).astype(np.float32)
    cols = rng.uniform(0, 1, (K, N, 3) if colours == "keyframe" else (N, 3)).astype(np.float32)
    shs = None if sh is None else rng.normal(0, 0.5, (K, N, 4, 3) if sh == "keyframe" else (N, 4, 3)).astype(np.float32)
    idx = list(static)
    for a in (means, opac) + ((covs,) if moving_covs else ()) + ((cols,) if colours == "keyframe" else ()) + \
            ((shs,) if sh == "keyframe" else ()):
        a[:, idx] = a[0, idx]
    return times, means, covs, opac, cols, shs


def check(name, seq, cache=2):
    g = GpuFrames(seq, device="cpu", cache=cache)
    ts = seq.times
    probes = [float(ts[0]) - 1, float(ts[0]), float(ts[2]), 0.3 * float(ts[1]) + 0.7 * float(ts[2]), float(ts[-1]),
              float(ts[-1]) + 5, 0.5 * float(ts[3] + ts[4]), float(ts[2]), 0.9 * float(ts[0]) + 0.1 * float(ts[1])]
    worst = 0.0
    for t in probes:
        got = [x.numpy().copy() for x in g.at(t)]   # copies: the tensors are buffers that the next call reuses
        m, c, o, col = seq.at(t)
        want = [m, c, o, seq.sh_at(t) if seq.sh is not None else col]
        for field, a, b in zip(("means", "covs", "opacities", "sh" if seq.sh is not None else "colours"), got, want):
            b = np.asarray(b, np.float32)
            assert a.shape == b.shape, (name, field, a.shape, b.shape)
            err = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))
            assert err < 2e-6, (name, field, t, err)
            worst = max(worst, err)
        assert len(g._cache) <= g.cache_size * g.n_dyn, (name, len(g._cache))
    print(f"ok  {name}: GpuFrames == Sequence4D.at at {len(probes)} times (largest relative difference {worst:.1e})")


def main():
    rng = np.random.default_rng(7)
    tmp = tempfile.mkdtemp()
    scattered = sorted(rng.choice(N, 120, replace=False).tolist())
    cases = [
        ("plain, per-keyframe sh", dict(), False, True),
        ("compact, per-keyframe sh, static group scattered", dict(static=scattered), True, True),
        ("compact, static block at the start", dict(static=range(0, 150)), True, True),
        ("plain, static block at the end", dict(static=range(N - 150, N)), False, True),
        ("compact, colours per keyframe, no sh", dict(sh=None, colours="keyframe", static=scattered), True, True),
        ("compact, constant sh and covariance (4D-Rotor-like)", dict(sh="constant", moving_covs=False), True, True),
        ("plain, nearest keyframe (interpolate=False)", dict(sh=None, colours="keyframe"), False, False),
    ]
    for name, kw, compact, interpolate in cases:
        times, means, covs, opac, cols, sh = scene(rng, **kw)
        path = os.path.join(tmp, "s.npz")
        Sequence4D(times, means, covs, opac, cols, sh=sh, interpolate=interpolate).save(path, compact=compact)
        seq = Sequence4D.load(path, lazy=True)
        assert seq.is_compact == compact
        if "static" in kw:
            assert seq.num_static == len(list(kw["static"])), (name, seq.num_static)
        check(name, seq, cache=2 if "block" in name else 3)
        del seq
    print("all checks passed")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip  GpuFrames on the CPU device (torch not installed)")
        sys.exit(0)
    main()
