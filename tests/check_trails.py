#!/usr/bin/env python3
"""WP35: "Colour by" speed / displacement and trails (viewer.py "Motion" folder, Sequence4D.speed_at / displacement_at, TrailData).

1. Sequence4D.speed_at / displacement_at on a hand-made sequence (uneven keyframe times, one static, one low-opacity Gaussian)
   against numbers worked out here from the keyframe positions; same through the compact + static-group file layout.
2. TrailData: only moving, opaque Gaussians are chosen, the first m of a longer ranking are the m trails, positions at the
   sample times equal np.interp of the keyframe positions.
3. The viewer on the toy scene, both renderers: the colours handed to the renderer are the colour scale at speed / displacement
   divided by the scale maximum; trails have m x 23 segments and cost X ms per frame; screenshots in shots/trails/.

    python tests/check_trails.py [--shots DIR]
    python tests/check_trails.py --file ../data_real/scgs/bouncingballs_sh.npz --name scgs --up=+z    # screenshots only
"""
import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import viewer as V
from gs4d import Sequence4D
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--chrome", default="/usr/bin/google-chrome")
ap.add_argument("--port", type=int, default=8191)
ap.add_argument("--shots", default=os.path.join(ROOT, "shots", "trails"))
ap.add_argument("--file", default=None, help="screenshots (and trail timing) of this file instead of the checks")
ap.add_argument("--name", default="scene")
ap.add_argument("--up", default="+z")
ap.add_argument("--renderer", default="auto")
ap.add_argument("--cameras", default=None, help="--file only: cameras file (viewer --cameras); the first camera is used for the screenshots")
opt = ap.parse_args()
os.makedirs(opt.shots, exist_ok=True)
ok = True
FLAGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def check(good, text):
    global ok
    ok &= bool(good)
    print(f"{'ok ' if good else 'FAIL'} {text}")


def make_seq():
    """6 Gaussians, 5 keyframes at t = 0, 1, 2.5, 3, 4. 0 static; 1 moves 1 unit per time unit along x; 2 changes speed
    per interval; 3 moves like 1 but has opacity 0.05; 4 moves 0.5 per time unit along y; 5 jumps in the last interval only."""
    t = np.array([0, 1, 2.5, 3, 4], np.float32)
    K, N = len(t), 6
    m = np.zeros((K, N, 3), np.float32)
    m[:, 1, 0] = t
    m[:, 2, 2] = [0, 2, 2.0, 5, 5]
    m[:, 3, 0] = t
    m[:, 4, 1] = 0.5 * t
    m[:, 5, 1] = [0, 0, 0, 0, 3]
    m += np.arange(N)[None, :, None] * np.array([10, 0, 0], np.float32)  # keep them apart
    covs = np.tile(np.array([0.01, 0, 0, 0.01, 0, 0.01], np.float32), (N, 1))
    op = np.ones(N, np.float32)
    op[3] = 0.05
    return Sequence4D(times=t, means=m, covs=covs, opacities=op, colors=np.tile(np.array([0.5, 0.5, 0.5], np.float32), (N, 1))), m, t


def part1_2():
    seq, m, t = make_seq()
    # speeds by hand: (G1) 1 everywhere; (G2) 2, 0, 6; wait: m2 z = 0, 2, 2, 5, 5 over t 0, 1, 2.5, 3, 4: 2/1, 0/1.5, 3/0.5, 0/1
    want = {0.5: [0, 1, 2, 1, .5, 0], 1.0: [0, 1, 0, 1, .5, 0], 2.0: [0, 1, 0, 1, .5, 0], 2.75: [0, 1, 6, 1, .5, 0], 4.0: [0, 1, 0, 1, .5, 3]}
    for tt, w in want.items():
        got = seq.speed_at(tt)
        check(np.allclose(got, w, atol=1e-6), f"speed_at({tt}) = {np.round(got, 3).tolist()} (worked out by hand {w})")
    d = seq.displacement_at(1.7)  # interpolated positions at 1.7: G1 x = 1.7, G2 z = 2, G4 y = .85, G5 y = 0
    check(np.allclose(d, [0, 1.7, 2, 1.7, 0.85, 0], atol=1e-6), f"displacement_at(1.7) = {np.round(d, 3).tolist()}")
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "s.npz")
        seq.save(f, compact=True, split_static=True)
        c = Sequence4D.load(f, lazy=True)
        err = max(np.abs(c.speed_at(tt) - seq.speed_at(tt)).max() for tt in want)
        check(err < 2e-2 and c.num_static >= 1, f"speed through the compact file with a static group matches (max |diff| {err:.1e}, {c.num_static} static)")
    td = V.TrailData(seq, max_m=10, min_peak=0.3)
    check(sorted(td.idx.tolist()) == [1, 2, 4, 5], f"trails are the moving, opaque Gaussians only: {sorted(td.idx.tolist())} (not 0 static, not 3 transparent)")
    check(np.allclose(td.traj, m[:, td.idx], atol=1e-6), "trail data = keyframe positions of the chosen Gaussians")
    pos, frac = td.at(2.0, 1.5, 4, samples=7)
    ts = np.clip(2.0 + np.linspace(-1.5, 1.5, 7), 0, 4)
    ref = np.stack([[np.interp(ts, t, m[:, g, c]) for c in range(3)] for g in td.idx[:4]], 0).transpose(2, 0, 1)
    check(np.abs(pos - ref).max() < 1e-5 and np.isclose(frac[0], 0) and np.isclose(frac[-1], 1),
          f"trail positions over the window equal np.interp of the keyframes (max |diff| {np.abs(pos - ref).max():.1e})")
    check(np.array_equal(td.at(2.0, 1.5, 2, 7)[0], pos[:, :2]), "asking for fewer trails keeps the first ones")
    td2 = V.TrailData(seq, max_m=2)
    check(np.array_equal(td2.idx, td.idx[:2]), "the ranking is the same whatever max_m is (fixed seed)")
    clip = td.at(0.2, 1.5, 4, 5)[0]  # window reaching before the start is clipped to t0
    check(np.allclose(clip[0], m[0, td.idx[:4]]), "a window reaching past the start is clipped to the first keyframe")


def content(img):
    return np.abs(img - 255.0).max(-1) > 30


def drive(renderer, port, path, name, up, shots, expect_colours):
    from PIL import Image
    args = argparse.Namespace(renderer=renderer, host="127.0.0.1", port=port, up=up, max_splats=300_000, max_res=480, file=path, compare=None,
                              cameras=opt.cameras if opt.file else None, gt_background="black")
    vw = V.Viewer(Sequence4D.load(path, lazy=True), args)
    threading.Thread(target=vw.run, daemon=True).start()
    t_mid = vw.t0 + 0.4 * vw.duration
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=FLAGS)
        pg = b.new_page(viewport={"width": 1100, "height": 700})
        pg.goto(f"http://localhost:{port}")
        time.sleep(7)
        wait = 3.0 if renderer == "gsplat" else 6.0
        shot = {}
        if vw.cameras:
            vw._select_camera(0, next(iter(vw.server.get_clients().values())))

        def grab(key):
            time.sleep(wait)
            f = os.path.join(shots, f"{name}_{renderer}_{key}.png")
            pg.screenshot(path=f)
            shot[key] = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)[:, :770]

        vw._set_time(t_mid)
        grab("colour")
        for kind in (V.SPEED, V.DISPLACEMENT):
            vw.colour_by.value = kind
            grab(kind)
            if expect_colours:
                vmax = vw._vmax_eff()
                seq = vw.seqs[0]
                v = seq.speed_at(vw.time.value) if kind == V.SPEED else seq.displacement_at(vw.time.value)
                rgb = V.colour_lut()[(np.clip(v / vmax, 0, 1) * 255 + 0.5).astype(int)]
                if renderer == "client":
                    cols = vw._current(0, full_cov=False)[3]
                    sub = vw.subsets[0]
                    ref = rgb if sub is None else rgb[sub]
                    check(np.abs(cols - ref).max() < 1e-6, f"client: colours sent to the browser are the colour scale at the {kind} (max |diff| {np.abs(cols - ref).max():.1e})")
                else:
                    means, covs, opac, cols = vw._frame(0, vw.min_opacity.value, 1.0)
                    keep = seq.at(vw.time.value)[2] >= vw.min_opacity.value
                    ref = rgb[keep]
                    check(cols.shape == ref.shape and np.abs(cols.cpu().numpy() - ref).max() < 1e-5,
                          f"gsplat: colours handed to the rasteriser are the colour scale at the {kind} (max |diff| {np.abs(cols.cpu().numpy() - ref).max():.1e})")
        d = np.abs(shot[V.SPEED] - shot["colour"]).mean()
        check(d > 0.5, f"{renderer}: 'Colour by speed' changes the picture (mean {d:.1f}/255)")
        d2 = np.abs(shot[V.DISPLACEMENT] - shot[V.SPEED]).mean()
        check(d2 > 0.2 or not expect_colours, f"{renderer}: speed and displacement pictures differ (mean {d2:.1f}/255)")
        vw.colour_by.value = V.COLOUR
        vw.trails_on.value = True
        grab("trails")
        h = vw.trail_h.get(0)
        n_seg = len(h.points) if h is not None else 0
        m = min(int(vw.trail_m.value), len(vw.trail_data[0].idx))
        check(h is not None and n_seg == m * 23, f"{renderer}: trails drawn, {m} trails x 23 segments = {n_seg}")
        d3 = np.abs(shot["trails"] - shot["colour"]).mean()
        check(d3 > 0.05, f"{renderer}: trails change the picture (mean {d3:.2f}/255)")
        # cost of one frame with M = 500 (or all there are): trail update only, and the whole drawn frame
        vw.trail_m.value = 500
        vw._draw_trails()
        a = time.perf_counter()
        for k in range(20):
            vw._set_time(vw.t0 + (0.3 + 0.01 * k) * vw.duration)
            vw._draw_trails()
        trail_ms = (time.perf_counter() - a) / 20 * 1000
        a = time.perf_counter()
        for k in range(20):
            vw._set_time(vw.t0 + (0.3 + 0.01 * k) * vw.duration)
            if renderer == "gsplat":
                vw._load_gsplat()
                vw.render_gsplat((1.0, 0.0, 0.0, 0.0), tuple(vw.centre - np.array([0, 0, 2.5 * vw.radius])), 1.0, 1280, 720)
                vw.torch.cuda.synchronize()
            vw._draw_trails()
        total_ms = (time.perf_counter() - a) / 20 * 1000
        print(f"    {renderer}: trails M={min(500, len(vw.trail_data[0].idx))}: {trail_ms:.1f} ms per frame for the trails"
              + (f", {total_ms:.1f} ms for decode + render 1280x720 + trails" if renderer == "gsplat" else ""))
        b.close()
    vw.server.stop()


if opt.file:
    drive(V.pick_renderer(opt.renderer), opt.port, opt.file, opt.name, opt.up, opt.shots, expect_colours=False)
    sys.exit(0)
part1_2()
tmp = tempfile.mkdtemp()
toy = os.path.join(tmp, "toy.npz")
subprocess.run([sys.executable, os.path.join(ROOT, "make_toy.py"), "-o", toy], check=True, stdout=subprocess.DEVNULL)
for k, r in enumerate(["client"] + (["gsplat"] if V.pick_renderer("auto") == "gsplat" else [])):
    drive(r, opt.port + k, toy, "toy", "+z", opt.shots, expect_colours=True)
print("motion colour and trails verified" if ok else "motion colour and trails MISMATCH")
sys.exit(0 if ok else 1)
