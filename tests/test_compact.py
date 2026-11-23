#!/usr/bin/env python3
"""Checks for compact storage and lazy loading in gs4d/format.py (CPU only).

    python tests/test_compact.py
"""
import os
import subprocess
import sys
import tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D, cov_from_quat_scale, unpack_cov

tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
K, N = 6, 3000


def make(sh=False, static=False):
    base = rng.uniform(-5, 5, (N, 3)).astype(np.float32)
    means = base + rng.normal(0, 0.5, (K, N, 3)).astype(np.float32)
    scales = np.exp(rng.uniform(np.log(1e-3), np.log(0.5), (N, 3))).astype(np.float32)
    scales[:50] = [0.5, 5e-4, 5e-4]  # needles, 1000:1
    scales[50:100] = [0.3, 0.3, 1e-4]  # flat discs
    covs = cov_from_quat_scale(rng.normal(size=(K, N, 4)).astype(np.float32), np.broadcast_to(scales, (K, N, 3)))
    opac = rng.uniform(0, 1, (K, N)).astype(np.float32)
    covs[:, -20:], opac[:, -20:] = 0.0, 0.0  # padding as written by convert_per_frame for a changing Gaussian count
    cols = rng.uniform(0, 1, (K, N, 3)).astype(np.float32)
    if static:
        means, covs, opac, cols = means[0], covs[0], opac[0], cols[0]
    return Sequence4D(np.linspace(0, 1, K), means, covs, opac, cols, source="test",
                      sh=rng.normal(size=(N, 4, 3)).astype(np.float32) if sh else None)


def test_roundtrip():
    s = make(sh=True)
    plain, small = os.path.join(tmp, "plain.npz"), os.path.join(tmp, "small.npz")
    s.save(plain)
    s.save(small, compact=True)
    assert not s.is_compact, "save(compact=True) must not change the sequence in memory"
    ratio = os.path.getsize(plain) / os.path.getsize(small)
    keys = set(np.load(small).files)
    assert {"means_ref", "means_d16", "covs_q", "covs_scale", "opacities_u8", "colors_u8", "sh"} <= keys
    assert not {"means", "covs", "opacities", "colors"} & keys
    for lazy in (False, True):
        c = Sequence4D.load(small, lazy=lazy)
        assert c.is_compact and c.means.shape == (K, N, 3) and c.covs.shape == (K, N, 6) and c.opacities.shape == (K, N)
        err_m = np.abs(np.asarray(c.means) - s.means).max()
        off = np.abs(s.means - s.means[0]).max()
        assert err_m <= off * 2.0 ** -11 + 1e-6, err_m  # float16 has 11 significant bits
        assert np.abs(np.asarray(c.opacities) - s.opacities).max() <= 0.5 / 255 + 1e-6
        assert np.abs(np.asarray(c.colors) - s.colors).max() <= 0.5 / 255 + 1e-6
        full, ref = unpack_cov(np.asarray(c.covs)).astype(np.float64), unpack_cov(s.covs).astype(np.float64)
        lam, lam_ref = np.linalg.eigvalsh(full), np.linalg.eigvalsh(ref)
        assert lam.min() >= -1e-7 * lam_ref.max(), f"decoded covariance is not PSD: {lam.min()}"
        big = lam_ref[..., 2:3]
        rel = np.abs(full - ref).max((-1, -2)) / np.maximum(big[..., 0], 1e-30)
        assert rel[:, :-20].max() < 2e-3, rel.max()  # error relative to the largest eigenvalue
        sig, sig_ref = np.sqrt(np.maximum(lam, 0)), np.sqrt(np.maximum(lam_ref, 0))
        assert (np.abs(sig - sig_ref) / sig_ref[..., 2:3].clip(1e-30))[:, :-20].max() < 2e-3  # axis lengths vs the longest axis
        assert np.array_equal(np.asarray(c.covs)[:, -20:], np.zeros((K, 20, 6)))
        assert np.array_equal(c.sh, s.sh) and c.source == "test" and np.array_equal(c.times, s.times)
        # interpolation and the accessors the viewer uses
        m, cv, o, col = c.at(0.5 * (s.times[2] + s.times[3]))
        assert np.allclose(m, 0.5 * (c.means[2] + c.means[3]), atol=1e-6) and m.dtype == np.float32
        assert np.allclose(c.opacities.max(0), np.asarray(c.opacities).max(0)) and c.means[0].shape == (N, 3)
        assert np.array_equal(c.at(-1)[0], c.means[0]) and np.array_equal(c.at(9)[1], c.covs[K - 1])
    # compact -> plain float32 again
    back = os.path.join(tmp, "back.npz")
    Sequence4D.load(small, lazy=True).save(back)
    b = Sequence4D.load(back)
    assert not b.is_compact and np.array_equal(b.means, np.asarray(c.means)) and np.array_equal(b.covs, np.asarray(c.covs))
    # compact -> compact keeps the stored bits
    again = os.path.join(tmp, "again.npz")
    Sequence4D.load(small).save(again, compact=True)
    assert all(np.array_equal(np.load(again)[k], np.load(small)[k]) for k in keys)
    print(f"ok  compact round trip ({ratio:.2f}x smaller): max position error {err_m:.1e} for offsets up to {off:.1f}, "
          f"covariances stay PSD, max axis-length error {(np.abs(sig - sig_ref) / sig_ref[..., 2:3].clip(1e-30))[:, :-20].max():.1e} of the longest axis")


def mapped(a):
    """True when the array's memory is a file mapping rather than a copy in RAM."""
    import mmap
    while a is not None and not isinstance(a, mmap.mmap):
        a = getattr(a, "base", None)
    return a is not None


def test_plain_and_lazy():
    s = make(sh=True)
    p = os.path.join(tmp, "p.npz")
    s.save(p)
    eager, lazy = Sequence4D.load(p), Sequence4D.load(p, lazy=True)
    assert mapped(lazy.means) and mapped(lazy.covs) and mapped(lazy.opacities) and not mapped(eager.means)
    assert mapped(Sequence4D.load(os.path.join(tmp, 'small.npz'), lazy=True).covs.raw)
    for name in ("means", "covs", "opacities", "colors", "sh", "times"):
        assert np.array_equal(getattr(eager, name), getattr(s, name)) and np.array_equal(getattr(lazy, name), getattr(s, name)), name
    for t in (0.0, 0.33, 1.0):
        assert all(np.array_equal(a, b) for a, b in zip(eager.at(t), lazy.at(t)))
    # compressed npz (not written by this code) cannot be mapped but must still load
    q = os.path.join(tmp, "z.npz")
    np.savez_compressed(q, times=s.times, means=s.means, covs=s.covs, opacities=s.opacities, colors=s.colors)
    z = Sequence4D.load(q, lazy=True)
    assert np.array_equal(z.means, s.means) and z.interpolate and z.source == ""
    print("ok  plain float32 files: exact round trip, lazy (memory-mapped) load equals eager load")


def test_edge_cases():
    st = make(static=True)
    p = os.path.join(tmp, "static.npz")
    st.save(p, compact=True)
    b = Sequence4D.load(p)
    assert not b.is_compact and np.array_equal(b.means, st.means) and np.array_equal(b.covs, st.covs)
    far = make()
    far.means = far.means.copy()
    far.means[3] += 1e6  # offset outside the float16 range -> means stay float32, the rest is still compact
    far.save(p, compact=True)
    b = Sequence4D.load(p)
    assert "means" in np.load(p).files and np.array_equal(b.means, far.means) and b.is_compact
    inp, out = os.path.join(tmp, "cli_in.npz"), os.path.join(tmp, "cli_out.npz")
    make().save(inp)
    subprocess.run([sys.executable, "-m", "gs4d", inp, out], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    assert Sequence4D.load(out).is_compact and os.path.getsize(out) < 0.5 * os.path.getsize(inp)
    assert subprocess.run([sys.executable, "-m", "gs4d", inp, inp], cwd=ROOT, capture_output=True).returncode != 0
    print("ok  static fields untouched, float16 overflow falls back to float32, `python -m gs4d` re-saves")


def test_sh_per_keyframe():
    """Per-keyframe sh: uint8 over each coefficient's own range in time."""
    s = make()
    amp = np.exp(rng.uniform(np.log(1e-3), np.log(2.0), (N, 16, 3))).astype(np.float32)  # ranges from 0.001 to 2
    sh = (rng.normal(0, 1, (N, 16, 3)) + amp * rng.uniform(-0.5, 0.5, (K, N, 16, 3))).astype(np.float32)
    sh[:, :, 5] = sh[0, :, 5]          # a coefficient that is constant in time
    s = Sequence4D(s.times, s.means, s.covs, s.opacities, s.colors, sh=sh, source="test")
    plain, small = os.path.join(tmp, "shp.npz"), os.path.join(tmp, "shs.npz")
    s.save(plain)
    s.save(small, compact=True)
    keys = set(np.load(small).files)
    assert {"sh_lo", "sh_step", "sh_u8"} <= keys and "sh" not in keys and np.load(small)["sh_u8"].dtype == np.uint8
    span = sh.max(0) - sh.min(0)
    for lazy in (False, True):
        c = Sequence4D.load(small, lazy=lazy)
        assert c.is_compact and c.sh.shape == sh.shape and c.sh_degree == 3 and "per keyframe" in c.summary()
        assert not lazy or mapped(c.sh.raw)
        err = np.abs(np.asarray(c.sh) - sh)
        assert (err <= span / 510 * 1.001 + 1e-6).all(), err.max()   # half a step of that coefficient's own range
        assert np.array_equal(np.asarray(c.sh)[:, :, 5], sh[:, :, 5])
        t = 0.5 * (s.times[2] + s.times[3])
        assert np.allclose(c.sh_at(t), 0.5 * (c.sh[2] + c.sh[3]), atol=1e-6) and c.sh_at(t).dtype == np.float32
        assert np.array_equal(c.sh_at(s.times[4]), c.sh[4])
    again, back = os.path.join(tmp, "sha.npz"), os.path.join(tmp, "shb.npz")
    Sequence4D.load(small, lazy=True).save(again, compact=True)        # compact -> compact keeps the stored bits
    assert all(np.array_equal(np.load(again)[k], np.load(small)[k]) for k in keys)
    Sequence4D.load(small, lazy=True).save(back)                        # compact -> plain float32
    b = Sequence4D.load(back)
    assert not b.is_compact and b.sh.dtype == np.float32 and np.array_equal(b.sh, np.asarray(c.sh))
    # file written before sh was part of the compact layout: compact fields + float32 sh [K,N,S,3]
    old = os.path.join(tmp, "sho.npz")
    z = dict(np.load(small))
    for k in ("sh_lo", "sh_step", "sh_u8"):
        del z[k]
    np.savez(old, sh=sh, **z)
    o = Sequence4D.load(old, lazy=True)
    assert o.is_compact and np.array_equal(o.sh, sh) and np.array_equal(o.sh_at(0.3), s.sh_at(0.3))
    # static group: the static Gaussians keep their float32 sh, the others are quantised
    st = Sequence4D(s.times, s.means.copy(), s.covs.copy(), s.opacities.copy(), s.colors.copy(), sh=sh.copy())
    for a in (st.means, st.covs, st.opacities, st.colors, st.sh):
        a[:, 200:900] = a[0, 200:900]
    st.save(small, compact=True)
    c = Sequence4D.load(small, lazy=True)
    assert c.num_static == 700 and np.load(small)["sh_u8"].shape == (K, N - 700, 16, 3)
    assert np.array_equal(np.asarray(c.sh)[:, 200:900], st.sh[:, 200:900])
    assert (np.abs(np.asarray(c.sh) - st.sh) <= span / 510 * 1.001 + 1e-6).all()
    # a value that is not finite: sh stays float32, the rest is still compact
    bad = sh.copy()
    bad[2, 7, 3, 1] = np.inf
    Sequence4D(s.times, s.means, s.covs, s.opacities, s.colors, sh=bad).save(small, compact=True)
    assert "sh" in np.load(small).files and "sh_u8" not in np.load(small).files and "covs_q" in np.load(small).files
    ratio = os.path.getsize(plain) / os.path.getsize(again)
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    gpu = "GPU decode not checked (no CUDA)"
    if cuda:
        from gs4d import GpuFrames
        for path in (again, small):  # without and with a static group
            c = Sequence4D.load(path, lazy=True)
            g = GpuFrames(c)
            d = max(float(np.abs(g.at(t)[3].cpu().numpy() - c.sh_at(t)).max()) for t in (0.0, 0.13, 0.5, 1.0))
            assert d < 1e-5, d
        gpu = f"GpuFrames differs from the CPU decode by {d:.0e}"
    print(f"ok  per-keyframe sh ({ratio:.2f}x smaller file): error at most half a step of each coefficient's own range "
          f"(largest {err.max():.1e} for ranges up to {span.max():.1f}), old files load, {gpu}")


test_roundtrip()
test_plain_and_lazy()
test_edge_cases()
test_sh_per_keyframe()
print("all checks passed")
