#!/usr/bin/env python3
"""Size, load time, memory and error of plain vs compact storage on a large random sequence.

    python tests/bench_large.py [--gaussians 500000] [--frames 100] [--dir /tmp]

Needs about 2x the plain file size in free RAM and 1.5x on disk (files are deleted at the end).
"""
import argparse, os, subprocess, sys, tempfile, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gs4d import Sequence4D, cov_from_quat_scale

ap = argparse.ArgumentParser()
ap.add_argument("--gaussians", type=int, default=500_000)
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--dir", default=None)
a = ap.parse_args()
N, K = a.gaussians, a.frames
d = tempfile.mkdtemp(dir=a.dir)
plain, small = os.path.join(d, "plain.npz"), os.path.join(d, "compact.npz")

if os.environ.get("GS4D_BENCH_CHILD"):  # measure one load in a fresh process: time, peak memory, first frame
    path, lazy = os.environ["GS4D_BENCH_CHILD"], os.environ["GS4D_BENCH_LAZY"] == "1"
    t = time.perf_counter(); s = Sequence4D.load(path, lazy=lazy); t_load = time.perf_counter() - t
    t = time.perf_counter(); s.at(0.5 * (s.times[3] + s.times[4])); t_at = time.perf_counter() - t
    rss = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS")) / 1024  # includes mapped pages in use
    print(t_load, t_at, rss)
    sys.exit(0)

rng = np.random.default_rng(0)
means = np.empty((K, N, 3), np.float32); covs = np.empty((K, N, 6), np.float32); opac = np.empty((K, N), np.float32)
base = rng.uniform(-10, 10, (N, 3)).astype(np.float32)
vel = rng.normal(0, 1.0, (N, 3)).astype(np.float32)
scales = np.exp(rng.uniform(np.log(2e-3), np.log(0.2), (N, 3))).astype(np.float32)
q0, q1 = rng.normal(size=(2, N, 4)).astype(np.float32)
for k in range(K):
    u = k / (K - 1)
    means[k] = base + vel * u + 0.2 * np.sin(6.28 * u + base)   # drift of up to a few units plus a wobble
    covs[k] = cov_from_quat_scale(q0 + u * q1, scales)
    opac[k] = rng.uniform(0, 1, N)
seq = Sequence4D(np.linspace(0, 1, K), means, covs, opac, rng.uniform(0, 1, (N, 3)).astype(np.float32))
print(seq.summary())
t = time.perf_counter(); seq.save(plain); t_plain = time.perf_counter() - t
t = time.perf_counter(); seq.save(small, compact=True); t_small = time.perf_counter() - t

c = Sequence4D.load(small, lazy=True)
err_pos = err_op = err_cov = 0.0
for k in range(K):
    err_pos = max(err_pos, float(np.abs(c.means[k] - means[k]).max()))
    err_op = max(err_op, float(np.abs(c.opacities[k] - opac[k]).max()))
    err_cov = max(err_cov, float((np.abs(c.covs[k] - covs[k]).max(-1) / covs[k][:, [0, 3, 5]].max(-1)).max()))
max_off = float(np.abs(means - means[0]).max())
del seq, means, covs, opac, c

def child(path, lazy):
    subprocess.run(["sync"])
    env = dict(os.environ, GS4D_BENCH_CHILD=path, GS4D_BENCH_LAZY=str(int(lazy)))
    return [float(x) for x in subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True).stdout.split()]

print(f"\n{'layout':<10}{'load':<8}{'file MB':>9}{'save s':>8}{'load s':>8}{'1st frame s':>12}{'RAM MB':>12}")
for name, path, t_save in (("float32", plain, t_plain), ("compact", small, t_small)):
    for lazy in (False, True):
        tl, ta, rss = child(path, lazy)
        print(f"{name:<10}{'lazy' if lazy else 'eager':<8}{os.path.getsize(path) / 1e6:>9.0f}{t_save:>8.1f}{tl:>8.2f}{ta:>12.3f}{rss:>12.0f}")
print(f"\ncompact errors: position {err_pos:.2e} (largest offset from keyframe 0: {max_off:.2f}), opacity {err_op:.2e}, "
      f"covariance {err_cov:.2e} of the largest variance")
os.remove(plain); os.remove(small); os.rmdir(d)
