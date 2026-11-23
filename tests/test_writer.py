#!/usr/bin/env python3
"""Checks for gs4d.Sequence4DWriter: a file written one keyframe at a time holds the same arrays, bit for bit,
as `Sequence4D.save` of the whole sequence (CPU only).

    python tests/test_writer.py
"""
import os
import sys
import tempfile
import tracemalloc
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D, Sequence4DWriter, cov_from_quat_scale

tmp = tempfile.mkdtemp()
rng = np.random.default_rng(3)
K, N = 9, 3000
KFIELDS = ("means", "covs", "opacities", "colors", "sh")


def make(static_idx=(), const=("colors", "sh"), far=False, k=K, bad_sh=False):
    """Random scene; the fields named in `const` have no keyframe axis, Gaussians in static_idx never change."""
    f = dict(means=rng.uniform(-5, 5, (k, N, 3)).astype(np.float32),
             covs=cov_from_quat_scale(rng.normal(size=(k, N, 4)).astype(np.float32),
                                      np.exp(rng.uniform(np.log(1e-2), np.log(0.5), (k, N, 3))).astype(np.float32)),
             opacities=rng.uniform(0.05, 1, (k, N)).astype(np.float32),
             colors=rng.uniform(0, 1, (k, N, 3)).astype(np.float32),
             sh=rng.normal(size=(k, N, 4, 3)).astype(np.float32))
    if far:
        f["means"][k // 2, 7] = 1e6  # offset beyond float16: means must stay float32 in the compact file
    if bad_sh:
        f["sh"][k // 2, 11, 2, 0] = np.inf  # sh must stay float32 in the compact file
    for n in KFIELDS:
        f[n][:, list(static_idx)] = f[n][0, list(static_idx)]
        if n in const:
            f[n] = f[n][0]
    return np.linspace(0, 2, k).astype(np.float32), f


def write_both(times, f, compact, split_static, with_sh=True, interpolate=True):
    """(one-shot file, incremental file)."""
    a, b = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
    f = {n: v for n, v in f.items() if with_sh or n != "sh"}
    Sequence4D(times, f["means"], f["covs"], f["opacities"], f["colors"], sh=f.get("sh"), source="test",
               interpolate=interpolate).save(a, compact=compact, split_static=split_static)
    const = {n: v for n, v in f.items() if v.ndim == (2 if n == "opacities" else 3 if n != "sh" else 4) - 1}
    with Sequence4DWriter(b, compact=compact, split_static=split_static, source="test", interpolate=interpolate, **const) as w:
        for k, t in enumerate(times):
            w.add_keyframe(t, **{n: v[k] for n, v in f.items() if n not in const})
    assert not [d for d in os.listdir(tmp) if d.startswith(".gs4d_writer_")], "scratch folder left behind"
    return a, b


def same_arrays(a, b):
    """Member names, order, dtypes, shapes and bytes of two .npz files."""
    with np.load(a) as za, np.load(b) as zb:
        assert za.files == zb.files, (za.files, zb.files)
        for n in za.files:
            x, y = za[n], zb[n]
            assert x.dtype == y.dtype and x.shape == y.shape, (n, x.dtype, y.dtype, x.shape, y.shape)
            assert x.tobytes() == y.tobytes(), f"{n} differs"
        return za.files


def test_bit_identical():
    some = rng.choice(N, 400, replace=False)
    cases = [("nothing static", dict()),
             ("scattered static Gaussians", dict(static_idx=some)),
             ("static block at the end", dict(static_idx=range(N - 500, N))),
             ("every field per keyframe", dict(static_idx=some, const=())),
             ("covs and opacities constant", dict(const=("covs", "opacities", "colors", "sh"))),
             ("all Gaussians static", dict(static_idx=range(N))),
             ("one keyframe", dict(k=1)),
             ("offset beyond float16", dict(static_idx=some, far=True)),
             ("sh value not finite", dict(static_idx=some, const=(), bad_sh=True))]
    for label, kw in cases:
        times, f = make(**kw)
        for compact in (False, True):
            for split in (True, False):
                names = same_arrays(*write_both(times, f, compact, split))
                if label == "scattered static Gaussians":
                    assert ("static_index" in names) == split and ("means_d16" in names) == compact
                if label == "offset beyond float16" and compact:
                    assert "means" in names and "means_d16" not in names and "covs_q" in names
                if label == "every field per keyframe":
                    assert ("sh_u8" in names) == compact and ("sh" in names) != compact and ("sh_static" in names) == split
                if label == "sh value not finite" and compact:
                    assert "sh" in names and "sh_u8" not in names and "covs_q" in names
                if label == "all Gaussians static" and split:
                    assert "static_index" not in names and "means_d16" not in names
        same_arrays(*write_both(times, f, True, True, with_sh=False, interpolate=False))
    print(f"ok  incremental file == one-shot file, bit for bit (plain and compact, with and without the static group): "
          f"{len(cases)} scenes")


def test_load_and_errors():
    times, f = make(static_idx=range(100, 900))
    a, b = write_both(times, f, True, True)
    sa, sb = Sequence4D.load(a), Sequence4D.load(b, lazy=True)
    assert sb.num_static == 800 and sb.is_compact and sb.source == "test" and sb.summary() == sa.summary()
    for t in (0.0, 0.33, 1.0, 2.0):
        for x, y in zip(sa.at(t) + (sa.sh_at(t),), sb.at(t) + (sb.sh_at(t),)):
            assert np.array_equal(x, y)

    p = os.path.join(tmp, "noext")                                   # like np.savez: ".npz" is appended
    with Sequence4DWriter(p, colors=f["colors"]) as w:
        w.add_keyframe(0.0, means=f["means"][0], covs=f["covs"][0], opacities=f["opacities"][0])
    assert Sequence4D.load(p + ".npz").num_frames == 1

    def fails(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    kf = lambda k: dict(means=f["means"][k], covs=f["covs"][k], opacities=f["opacities"][k])  # noqa: E731
    bad = os.path.join(tmp, "bad.npz")
    w = Sequence4DWriter(bad, colors=f["colors"])
    w.add_keyframe(0.5, **kf(0))
    assert fails(lambda: w.add_keyframe(0.5, **kf(1)))                                        # time not increasing
    assert fails(lambda: w.add_keyframe(0.7, means=f["means"][1], covs=f["covs"][1]))         # field set changed
    assert fails(lambda: w.add_keyframe(0.7, **dict(kf(1), means=f["means"][1][:10])))        # wrong N
    w.abort()
    assert fails(lambda: Sequence4DWriter(bad, colors=f["colors"]).add_keyframe(0.0, means=f["means"][0]))  # covs missing
    assert fails(lambda: Sequence4DWriter(bad, colors=f["colors"]).add_keyframe(0.0, colors=f["colors"], **kf(0)))  # twice
    assert fails(lambda: Sequence4DWriter(bad, colors=f["colors"]).close())                   # no keyframe
    try:
        with Sequence4DWriter(bad, colors=f["colors"]) as w:
            w.add_keyframe(0.0, **kf(0))
            raise RuntimeError("converter failed")
    except RuntimeError:
        pass
    assert not os.path.exists(bad), "a failed run must not leave an output file"
    for d in [d for d in os.listdir(tmp) if d.startswith(".gs4d_writer_")]:  # writers above that were never closed
        os.rmdir(os.path.join(tmp, d))
    print("ok  Sequence4D.load (eager and lazy) reads the file; wrong input raises; a failed run leaves no file")


def test_memory():
    """Peak memory of the writer does not grow with K (numpy allocations, measured with tracemalloc)."""
    n = 20000
    m0 = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    c0 = cov_from_quat_scale(rng.normal(size=(n, 4)).astype(np.float32), np.full((n, 3), 0.1, np.float32))
    o, col = np.full(n, 0.5, np.float32), np.full((n, 3), 0.5, np.float32)
    peaks = {}
    for k in (10, 200):
        tracemalloc.start()
        with Sequence4DWriter(os.path.join(tmp, "m.npz"), compact=True, opacities=o, colors=col) as w:
            for i in range(k):
                w.add_keyframe(i, means=m0 + np.float32(i * 0.01), covs=c0 * np.float32(1 + i * 0.01))
        peaks[k] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    full = 200 * n * 9 * 4
    assert peaks[200] < 1.2 * peaks[10] + 1e5 and peaks[200] < full / 5, (peaks, full)
    assert Sequence4D.load(os.path.join(tmp, "m.npz"), lazy=True).num_frames == 200
    print(f"ok  peak memory {peaks[10] / 1e6:.1f} MB at 10 keyframes, {peaks[200] / 1e6:.1f} MB at 200 "
          f"(the float32 sequence alone would be {full / 1e6:.0f} MB)")


def test_converters():
    """The three converters that use the writer still give what their in-memory functions give (toy input)."""
    import convert_native4d as c
    n, k = 500, 6
    xyz, t = rng.normal(size=(n, 3)).astype(np.float32), rng.uniform(0, 1, n).astype(np.float32)
    s, st = np.exp(rng.normal(-2, 0.3, (n, 3))).astype(np.float32), np.exp(rng.normal(-1, 0.3, (n, 1))).astype(np.float32)
    ql, qr = rng.normal(size=(n, 4)).astype(np.float32), rng.normal(size=(n, 4)).astype(np.float32)
    op = rng.uniform(0.1, 1, n).astype(np.float32)
    times = np.linspace(0, 1, k, dtype=np.float32)
    means, cov3, opac = c.slice_native4d(xyz, t, s, st, ql, qr, op, times)
    g = c.Native4D(xyz, t, s, st, ql, qr, op)
    assert np.array_equal(g.cov3, cov3)
    for i, tk in enumerate(times):
        assert np.array_equal(g.means(tk), means[i]) and np.array_equal(g.opacities(tk), opac[i])
    sub = rng.choice(n, 100, replace=False)
    assert np.array_equal(g.means(times[2], sub), means[2][sub]) and np.array_equal(g.opacities(times[2], sub), opac[2][sub])
    dc, rest = rng.normal(size=(n, 1, 3)).astype(np.float32), rng.normal(size=(n, 7, 3)).astype(np.float32)
    whole = c.sh_at_times(dc, rest, t, times, 1.0, 1, 1)
    f = c.sh_channels(dc, rest, 1, 1)
    for i, tk in enumerate(times):
        assert np.array_equal(c.sh_at_time(f, t.astype(np.float64), tk, 1.0, 1, 1), whole[i])
    print("ok  convert_native4d: per-keyframe slicing == slice_native4d / sh_at_times")


def test_colors_out_of_range():
    """Per-keyframe colours outside [0,1] (Spacetime Gaussians keeps raw colours) are not clipped by the compact layout:
    they stay float32, in `save` and in the writer alike; colours inside [0,1] are still packed as uint8."""
    rng = np.random.default_rng(4)
    K, N = 4, 50
    times = np.arange(K, dtype=np.float32)
    means = rng.normal(size=(K, N, 3)).astype(np.float32)
    covs = np.tile(np.array([1, 0, 0, 1, 0, 1], np.float32) * 0.01, (K, N, 1))
    opac = rng.uniform(0.1, 1, (K, N)).astype(np.float32)
    wild = rng.uniform(-2, 3, (K, N, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
        for cols, packed in ((wild, False), (np.clip(wild, 0, 1), True)):
            Sequence4D(times, means, covs, opac, cols).save(a, compact=True)
            with Sequence4DWriter(b, compact=True) as w:
                for k in range(K):
                    w.add_keyframe(times[k], means=means[k], covs=covs[k], opacities=opac[k], colors=cols[k])
            za, zb = np.load(a), np.load(b)
            assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[f], zb[f]) for f in za.files)
            assert ("colors_u8" in za.files) == packed and ("colors" in za.files) == (not packed), za.files
            if not packed:
                assert np.array_equal(Sequence4D.load(a, lazy=True).at(1.0)[3], cols[1])
        # a STATIC Gaussian with a colour outside [0,1]: it is stored once in float32 (colors_static), so the per-keyframe
        # colours of the others are still packed as uint8, by `save` and by the writer alike
        cols, still = np.clip(wild, 0, 1), 3
        means[:, still], opac[:, still], cols[:, still] = means[0, still], opac[0, still], 1.7
        Sequence4D(times, means, covs, opac, cols).save(a, compact=True)
        with Sequence4DWriter(b, compact=True) as w:
            for k in range(K):
                w.add_keyframe(times[k], means=means[k], covs=covs[k], opacities=opac[k], colors=cols[k])
        files = same_arrays(a, b)
        assert "colors_u8" in files and "colors_static" in files, files
        assert np.allclose(Sequence4D.load(b).at(2.0)[3][still], 1.7)
    print("ok  compact: colours outside [0,1] stay float32 (save == writer), colours inside are uint8, "
          "also next to a static Gaussian whose colour is outside")


test_bit_identical()
test_colors_out_of_range()
test_load_and_errors()
test_memory()
test_converters()
print("all checks passed")
