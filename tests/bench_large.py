#!/usr/bin/env python3
"""Size, load time, memory and error of plain vs compact storage on a large random sequence.

    python tests/bench_large.py [--gaussians 500000] [--frames 100] [--dir /tmp]

Needs about 2x the plain file size in free RAM and 1.5x on disk (files are deleted at the end).

    python tests/bench_large.py --file ../data_real/som/paper-windmill.npz     # play-frame timing of an existing file only

Play-frame timing (needs CUDA): where the time of one frame goes on the CPU path (read from disk / decode /
interpolate / upload), and the time to get the gsplat inputs for one frame while playing, CPU path vs GPU path
(viewer.frame_cpu vs viewer.frame_gpu), plus the largest difference between the two renders.
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
ap.add_argument("--file", default=None, help="only time playing this existing sequence, then exit")
ap.add_argument("--no-play", action="store_true", help="skip the play-frame timing")
a = ap.parse_args()
N, K = a.gaussians, a.frames


def sync():
    import torch
    torch.cuda.synchronize()


def ms(f, *args, reps=1):
    sync(); t = time.perf_counter()
    for _ in range(reps):
        r = f(*args)
    sync()
    return (time.perf_counter() - t) / reps * 1e3, r


def breakdown(seq, k=3, w=0.4):
    """Milliseconds spent in each stage of Sequence4D.at for one frame between keyframes k and k+1, CPU path."""
    import torch
    from gs4d.format import _Split
    rows = {"read": 0.0, "decode": 0.0, "interpolate": 0.0, "upload": 0.0}
    for name in ("means", "covs", "opacities", "colors"):
        f = getattr(seq, name)
        if f.ndim <= (1 if name == "opacities" else 2):
            continue
        dyn = f.dyn if isinstance(f, _Split) else f
        raw = getattr(dyn, "raw", None)
        frames = []
        for kk in (k, k + 1):
            if raw is not None:  # touch the stored bytes so that the disk / page cache cost is measured on its own
                t = time.perf_counter(); r = np.array(raw[kk]); rows["read"] += (time.perf_counter() - t) * 1e3
                mem = type(dyn).__new__(type(dyn)); mem.__dict__.update(dyn.__dict__); mem.raw = r[None]
                t = time.perf_counter(); frames.append(mem.frame(0)); rows["decode"] += (time.perf_counter() - t) * 1e3
            else:
                t = time.perf_counter(); frames.append(np.array(dyn[kk])); rows["read"] += (time.perf_counter() - t) * 1e3
        t = time.perf_counter(); out = (1.0 - w) * frames[0] + w * frames[1]; rows["interpolate"] += (time.perf_counter() - t) * 1e3
        if isinstance(f, _Split):
            t = time.perf_counter(); out = f.layout.merge(out, f.static); rows["interpolate"] += (time.perf_counter() - t) * 1e3
        t = time.perf_counter(); torch.from_numpy(np.ascontiguousarray(out)).cuda(); torch.cuda.synchronize()
        rows["upload"] += (time.perf_counter() - t) * 1e3
    return rows


def play_bench(path):
    """Per-frame time to get the gsplat inputs while playing: 3 keyframe intervals, 10 frames each, CPU vs GPU path."""
    import torch
    from gsplat import rasterization
    from gs4d import GpuFrames
    from viewer import frame_cpu, frame_gpu, camera_matrices
    dev = torch.device("cuda")
    torch.zeros(1, device=dev); sync()  # CUDA context start-up is not part of any frame
    seq = Sequence4D.load(path, lazy=True)
    print(f"\n{os.path.basename(path)}: {seq.summary()}, file {os.path.getsize(path) / 1e6:.0f} MB")
    for k in range(0, 4):  # warm the page cache for what the CPU path will read
        seq.at(float(seq.times[k]))
    b = breakdown(seq)
    print("CPU at(t) by stage (ms): " + ", ".join(f"{n} {v:.1f}" for n, v in b.items()) + f"; total {sum(b.values()):.1f}")
    t0 = float(seq.times[3])
    ts = [t0 + (seq.times[6] - seq.times[3]) * j / 30 for j in range(31)]  # crosses 2 keyframes, 30 frames
    res = {}
    for name, fn in (("CPU path", lambda t: frame_cpu(seq, t, 0.02, 1.0, dev)),
                     ("GPU path", None)):
        if fn is None:
            g = GpuFrames(seq, dev)
            fn = lambda t: frame_gpu(g, t, 0.02, 1.0)
        fn(ts[0]); sync()
        per = []
        for t in ts:
            sync(); tt = time.perf_counter(); fn(t); sync(); per.append((time.perf_counter() - tt) * 1e3)
        res[name] = per
        if name == "CPU path":
            at = []
            for t in ts:
                tt = time.perf_counter(); seq.at(t); at.append((time.perf_counter() - tt) * 1e3)
            print(f"  of which Sequence4D.at(t) alone: mean {np.mean(at):.1f} ms (the rest: 3x3 covariances, opacity mask, upload)")
        print(f"{name}: mean {np.mean(per):.1f} ms/frame over {len(per)} frames, median {np.median(per):.1f}, worst {max(per):.1f}")
    g = GpuFrames(seq, dev)
    worst = 0
    W, H = 640, 360
    means = np.asarray(seq.means[0])
    c = np.median(means, 0); r = float(np.percentile(np.linalg.norm(means - c, axis=1), 90)) or 1.0
    for t in ts[::10]:
        imgs = []
        for tens in (frame_cpu(seq, t, 0.02, 1.0, dev), frame_gpu(g, t, 0.02, 1.0)):
            vm, K = camera_matrices((0.0, 0.0, 0.0, 1.0), c + np.array([0, 0, -2.5 * r]), 0.9, W, H)
            m, cv, o, co = tens
            with torch.no_grad():
                img, al, _ = rasterization(m, None, None, o, co, torch.from_numpy(vm).to(dev)[None], torch.from_numpy(K).to(dev)[None],
                                           W, H, covars=cv, sh_degree=seq.sh_degree)
            imgs.append(((img[0] + 1 - al[0]).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().astype(int))
        worst = max(worst, int(np.abs(imgs[0] - imgs[1]).max()))
    print(f"largest render difference, CPU vs GPU path, {len(ts[::10])} times: {worst}/255 per channel")


if a.file:
    play_bench(a.file)
    sys.exit(0)
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
if not a.no_play:
    try:
        import torch
        ok = torch.cuda.is_available()
    except ImportError:
        ok = False
    if ok:
        for p in (plain, small):
            play_bench(p)
    else:
        print("\nno CUDA: play-frame timing skipped")
os.remove(plain); os.remove(small); os.rmdir(d)
