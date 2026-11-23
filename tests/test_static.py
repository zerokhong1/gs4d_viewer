#!/usr/bin/env python3
"""Checks for the static group in gs4d/format.py: Gaussians that never change are stored once (CPU only).

    python tests/test_static.py
"""
import os
import subprocess
import sys
import tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D, cov_from_quat_scale

tmp = tempfile.mkdtemp()
rng = np.random.default_rng(1)
K, N = 7, 2000


def make(static_idx, sh_per_keyframe=False, colours_static=True):
    """Scene where the Gaussians in static_idx are identical at every keyframe and all others change in every field."""
    means = rng.uniform(-5, 5, (K, N, 3)).astype(np.float32)
    scales = np.exp(rng.uniform(np.log(1e-2), np.log(0.5), (K, N, 3))).astype(np.float32)
    covs = cov_from_quat_scale(rng.normal(size=(K, N, 4)).astype(np.float32), scales)
    opac = rng.uniform(0.05, 1, (K, N)).astype(np.float32)
    cols = rng.uniform(0, 1, (N, 3) if colours_static else (K, N, 3)).astype(np.float32)
    sh = rng.normal(size=(K, N, 4, 3) if sh_per_keyframe else (N, 4, 3)).astype(np.float32)
    for a in (means, covs, opac) + (() if colours_static else (cols,)) + ((sh,) if sh_per_keyframe else ()):
        a[:, static_idx] = a[0, static_idx]
    return Sequence4D(np.linspace(0, 2, K), means, covs, opac, cols, source="test", sh=sh)


def frames(seq, ts):
    return [tuple(np.array(x) for x in seq.at(t)) + (np.array(seq.sh_at(t)),) for t in ts]


TS = (0.0, 0.31, 1.0, 1.77, 2.0)


def test_roundtrip():
    block = np.arange(600, N)                                       # one block at the end, as Shape of Motion writes it
    scattered = np.sort(rng.choice(N, 700, replace=False))          # anywhere
    for name, idx, kw in (("block", block, {}), ("scattered", scattered, dict(sh_per_keyframe=True, colours_static=False)),
                          ("block at the start", np.arange(0, 900), dict(sh_per_keyframe=True))):
        s = make(idx, **kw)
        want = frames(s, TS)
        p_old, p_new = os.path.join(tmp, "old.npz"), os.path.join(tmp, "new.npz")
        s.save(p_old, split_static=False)
        s.save(p_new)
        assert s.num_static == 0 and not hasattr(s.means, "dyn"), "save must not change the sequence in memory"
        z = np.load(p_new)
        assert np.array_equal(z["static_index"], idx) and z["means"].shape == (K, N - len(idx), 3)
        assert z["means_static"].shape == (len(idx), 3) and z["opacities_static"].shape == (len(idx),)
        assert ("sh_static" in z.files) == kw.get("sh_per_keyframe", False) and ("colors_static" in z.files) == (not kw.get("colours_static", True))
        assert "static_index" not in np.load(p_old).files and np.load(p_old)["means"].shape == (K, N, 3)
        for lazy in (False, True):
            b = Sequence4D.load(p_new, lazy=lazy)
            assert b.num_static == len(idx) and b.num_gaussians == N and b.means.shape == (K, N, 3) and b.sh_degree == 1
            got = frames(b, TS)
            for w, g in zip(want, got):   # same Gaussians, same order, same values: only the float32 blend order differs
                assert all(x.shape == y.shape and x.dtype == np.float32 and np.allclose(x, y, rtol=0, atol=1e-6) for x, y in zip(w, g))
            for k in (0, 3, K - 1):       # at a keyframe nothing is blended: exact
                assert np.array_equal(b.means[k], s.means[k]) and np.array_equal(b.covs[k], s.covs[k])
                assert np.array_equal(b.at(float(s.times[k]))[2], s.opacities[k])
            assert np.array_equal(np.asarray(b.means), s.means) and np.array_equal(b.opacities.max(0), s.opacities.max(0))
            old = Sequence4D.load(p_old, lazy=lazy)   # file written the old way still loads, and gives the same frames
            assert old.num_static == 0 and all(np.array_equal(x, y) for w, g in zip(want, frames(old, TS)) for x, y in zip(w, g))
        ratio = os.path.getsize(p_old) / os.path.getsize(p_new)
        assert ratio > 1.25, ratio
        print(f"ok  static group, {name} ({len(idx)} of {N} static): same frames in the same order, file {ratio:.2f}x smaller")


def test_compact_and_cli():
    idx = np.arange(500, N)
    s = make(idx)
    s.means[:, :500] = s.means[0, :500] + rng.normal(0, 0.3, (K, 500, 3)).astype(np.float32)   # offsets small enough for float16
    s.means[:, idx] += 100.0    # far away, but static: stays exact float32 instead of a float16 offset
    p_plain, p_c, p_cs = (os.path.join(tmp, n) for n in ("plain.npz", "compact.npz", "compact_split.npz"))
    s.save(p_plain, split_static=False)
    s.save(p_c, compact=True, split_static=False)
    s.save(p_cs, compact=True)
    c, cs = Sequence4D.load(p_c, lazy=True), Sequence4D.load(p_cs, lazy=True)
    assert cs.is_compact and cs.num_static == len(idx) and c.num_static == 0 and "means_d16" in np.load(p_cs).files
    assert np.load(p_cs)["means_d16"].shape == (K, 500, 3) and np.load(p_cs)["covs_q"].shape == (K, 500, 6)
    for t in TS:
        a, b, ref = c.at(t), cs.at(t), s.at(t)
        assert all(np.array_equal(x[:500], y[:500]) for x, y in zip(a, b)), "dynamic Gaussians must decode identically"
        assert all(np.allclose(y[500:], r[500:], rtol=1e-6, atol=1e-6) for y, r in zip(b, ref)), "static Gaussians must be exact"
    assert np.array_equal(cs.means[2][500:], s.means[2][500:]) and np.array_equal(cs.covs[4][500:], s.covs[4][500:])
    assert os.path.getsize(p_cs) < os.path.getsize(p_c) < os.path.getsize(p_plain)

    out, out_ns, out_f = (os.path.join(tmp, n) for n in ("cli.npz", "cli_nosplit.npz", "cli_f32.npz"))
    run = lambda *args: subprocess.run([sys.executable, "-m", "gs4d", *args], check=True, cwd=ROOT, capture_output=True, text=True).stdout
    assert f"{len(idx):,} static" in run(p_plain, out)                      # old file -> new layout
    assert "static" not in run(p_plain, out_ns, "--no-split") and "static_index" not in np.load(out_ns).files
    run(out, out_f, "--float32")                                            # split compact -> split float32
    f = Sequence4D.load(out_f)
    assert f.num_static == len(idx) and not f.is_compact and Sequence4D.load(out).is_compact
    assert np.array_equal(np.asarray(f.means)[:, idx], s.means[:, idx])
    print(f"ok  static group with compact storage: {os.path.getsize(p_plain) / os.path.getsize(p_cs):.1f}x smaller than float32 "
          f"({os.path.getsize(p_plain) / os.path.getsize(p_c):.1f}x without the static group), `python -m gs4d` converts old files")


def test_edge_cases():
    p = os.path.join(tmp, "edge.npz")
    none = make(np.arange(0))
    none.save(p)
    assert "static_index" not in np.load(p).files and np.array_equal(Sequence4D.load(p).means, none.means)

    every = make(np.arange(N))                       # nothing moves: the fields lose the keyframe axis
    every.save(p)
    b = Sequence4D.load(p)
    assert b.means.shape == (N, 3) and b.num_static == 0 and np.array_equal(b.at(1.0)[0], every.means[0])

    part = make(np.arange(100, N))                   # static in position but not in opacity -> must stay dynamic
    part.opacities[3, 150] += 0.01
    part.save(p)
    b = Sequence4D.load(p)
    assert b.num_static == N - 101 and 150 not in np.load(p)["static_index"]
    assert np.array_equal(np.asarray(b.opacities), part.opacities)

    tiny = make(np.arange(100, N))                   # a change far above float32 rounding is kept, rounding noise is not
    tiny.means[2, 200, 0] *= 1 + 1e-4
    tiny.means[2, 300, 0] = np.nextafter(tiny.means[2, 300, 0], np.float32(np.inf))
    tiny.save(p)
    si = np.load(p)["static_index"]
    assert 200 not in si and 300 in si
    assert np.array_equal(Sequence4D.load(p).means[2][200], tiny.means[2][200])

    nointerp = make(np.arange(50, N))
    nointerp.interpolate = False
    nointerp.save(p)
    b = Sequence4D.load(p)
    assert not b.interpolate and np.array_equal(b.at(0.2)[0], nointerp.at(0.2)[0])

    b.split_static()
    assert b.num_static == N - 50                    # already split: unchanged
    b.merge_static()
    assert b.num_static == 0 and isinstance(b.means, np.ndarray) and np.array_equal(b.means, nointerp.means)
    print("ok  no static Gaussians, all static, static in only some fields, tolerance, nearest-keyframe mode, merge back")


test_roundtrip()
test_compact_and_cli()
test_edge_cases()
print("all checks passed")
