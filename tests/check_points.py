#!/usr/bin/env python3
"""Checks "Draw as: points" (one dot at each Gaussian's centre) in headless Chrome, for both renderers.

For each file and renderer: screenshots at three times in points mode and one in splats mode, same camera. Checks:
points are drawn; the points picture differs from the splats picture; it changes with time (only required for the
toy scene and with --moving); a larger "Point size" changes the picture; a high "Hide opacity below" removes points.
On the toy scene also `--compare` side by side in points mode. The gsplat part needs CUDA and also prints the server
side time per frame while playing (decode + rasterise at 1280x720), points against splats.

    python tests/check_points.py                       # toy scene
    python tests/check_points.py --file ../data_real/scgs/bouncingballs_sh.npz --name scgs --moving
    python tests/check_points.py --file ../data_real/som/paper-windmill.npz --name windmill --up=-y
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
from PIL import Image
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=None, help="sequence to use instead of a fresh toy scene")
ap.add_argument("--name", default="toy", help="prefix of the screenshot files")
ap.add_argument("--up", default="+z")
ap.add_argument("--moving", action="store_true", help="require the picture to change with time")
ap.add_argument("--out", default=os.path.join(ROOT, "shots", "points"))
ap.add_argument("--chrome", default="/usr/bin/google-chrome")
ap.add_argument("--port", type=int, default=8161)
opt = ap.parse_args()
os.makedirs(opt.out, exist_ok=True)
toy = opt.file is None
tmp = tempfile.mkdtemp()
path = opt.file
if toy:
    path = os.path.join(tmp, "toy.npz")
    subprocess.run([sys.executable, os.path.join(ROOT, "make_toy.py"), "-o", path], check=True, stdout=subprocess.DEVNULL)
W, H = 1280, 760
SCENE_W = int(W * 0.7)  # the GUI panel covers the right edge
FLAGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
ok = True


def check(good, text):
    global ok
    ok &= bool(good)
    print(f"{'ok ' if good else 'FAIL'} {text}")


def content(img):
    return np.abs(img - 255.0).max(-1) > 30  # not the white background


def make(renderer, port, file, compare=None):
    args = argparse.Namespace(renderer=renderer, host="127.0.0.1", port=port, up=opt.up, max_splats=300_000,
                              max_res=1080, file=file, compare=compare)
    vw = V.Viewer(Sequence4D.load(file, lazy=True), args, Sequence4D.load(compare, lazy=True) if compare else None)
    threading.Thread(target=vw.run, daemon=True).start()
    return vw


def run(renderer, port):
    vw = make(renderer, port, path)
    t0, t1 = vw.t0, vw.t1
    times = [t0 + f * (t1 - t0) for f in (0.1, 0.45, 0.8)]
    wait = 3.0 if renderer == "gsplat" else 6.0
    shots = {}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=FLAGS)
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto(f"http://localhost:{port}")
        time.sleep(8)

        def grab(key, save=True):
            time.sleep(wait)
            f = os.path.join(opt.out if save else tmp, f"{opt.name}_{renderer}_{key}.png")
            pg.screenshot(path=f)
            shots[key] = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)[:, :SCENE_W]

        vw._set_time(times[0])
        grab("splats_t0")
        vw.draw_as.value = V.POINTS
        for k, t in enumerate(times):
            vw._set_time(t)
            grab(f"points_t{k}")
        size = vw.point_size.value
        vw.point_size.value = size * 3
        grab("points_big")
        vw.point_size.value = size
        op = vw.seqs[0].at(times[2])[2]
        op = op if vw.subsets[0] is None else op[vw.subsets[0]]  # the client renderer may send only the most opaque ones
        kept = [int((op >= th).sum()) for th in (0.02, 0.5)]
        vw.min_opacity.value = 0.5
        grab("points_opaque_only")
        vw.min_opacity.value = 0.02
        # play in points mode: the time must advance and the picture must follow
        vw._set_time(t0)
        vw.playing = True
        time.sleep(min(2.0, 0.5 * (t1 - t0)))  # stop before the loop wraps around
        vw.playing = False
        check(vw.time.value > t0 + 0.2 * (t1 - t0), f"{renderer}: playing in points mode advances the time ({t0:.3g} -> {vw.time.value:.3g})")
        b.close()

    cov = {k: content(v).mean() for k, v in shots.items()}
    check(cov["points_t0"] > 0.002, f"{renderer}: points are drawn ({cov['points_t0']:.2%} of the scene area)")
    d = np.abs(shots["points_t0"] - shots["splats_t0"]).mean()
    check(d > 0.2, f"{renderer}: points picture differs from splats picture (mean {d:.2f}/255)")
    dt = max(np.abs(shots["points_t0"] - shots[f"points_t{k}"]).mean() for k in (1, 2))
    if toy or opt.moving:
        check(dt > 0.05, f"{renderer}: points move with time (mean change {dt:.2f}/255)")
    else:
        print(f"    {renderer}: change with time {dt:.3f}/255 (not required for this file)")
    # (coverage is no measure here: dense clouds already cover the object, and bright Gaussians sit near the white background)
    d = np.abs(shots["points_big"] - shots["points_t2"]).mean()
    check(d > 0.2, f"{renderer}: 3x point size changes the picture (mean {d:.2f}/255)")
    d = np.abs(shots["points_opaque_only"] - shots["points_t2"]).mean()
    if kept[1] < kept[0]:
        check(d > 0.05, f"{renderer}: 'Hide opacity below' 0.5 removes points "
              f"({kept[0]:,} -> {kept[1]:,} drawn, picture changes by {d:.2f}/255)")
    else:
        check(d < 0.05, f"{renderer}: all {kept[0]:,} drawn Gaussians have opacity >= 0.5, the threshold changes nothing ({d:.2f}/255)")

    if renderer == "gsplat":  # server-side cost of one played frame at 1280x720
        cam = next(iter(vw.server.get_clients().values()), None)
        wxyz, pos, fov = ((1.0, 0.0, 0.0, 0.0), tuple(vw.centre - np.array([0, 0, 2.5 * vw.radius])), 1.0) if cam is None \
            else (np.array(cam.camera.wxyz), np.array(cam.camera.position), float(cam.camera.fov))
        for mode in (V.SPLATS, V.POINTS):
            vw.draw_as.value = mode
            ms = []
            for k in range(40):
                vw.time.value = float(t0 + (t1 - t0) * (k + 0.5) / 40)
                vw.torch.cuda.synchronize()
                a = time.perf_counter()
                vw._load_gsplat()
                vw.render_gsplat(wxyz, pos, fov, 1280, 720)
                ms.append((time.perf_counter() - a) * 1000)
            ms = np.sort(ms[5:])
            print(f"    gsplat {mode:6s}: {np.median(ms):.1f} ms per played frame at 1280x720 (decode + rasterise + copy), "
                  f"{1000 / np.median(ms):.0f} frames/s possible")
    vw.server.stop()


def run_compare(renderer, port):
    pb = os.path.join(tmp, "toy_b.npz")
    subprocess.run([sys.executable, os.path.join(ROOT, "make_toy.py"), "-o", pb, "--seed", "3", "--points", "600"],
                   check=True, stdout=subprocess.DEVNULL)
    vw = make(renderer, port, path, pb)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=FLAGS)
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto(f"http://localhost:{port}")
        time.sleep(8)
        vw.draw_as.value = V.POINTS
        vw._set_time(0.3)
        time.sleep(4)
        f = os.path.join(opt.out, f"{opt.name}_{renderer}_compare_points.png")
        pg.screenshot(path=f)
        b.close()
    vw.server.stop()
    cs = content(np.asarray(Image.open(f).convert("RGB")).astype(np.float32)[:, :SCENE_W])
    half = SCENE_W // 2 if renderer == "client" else W // 2
    left, right = cs[:, :half].mean(), cs[:, half:].mean()
    check(left > 0.002 and right > 0.002, f"{renderer}: --compare side by side in points mode has both halves ({left:.2%}, {right:.2%})")


renderers = ["client"] + (["gsplat"] if V.pick_renderer("auto") == "gsplat" else [])
if len(renderers) == 1:
    print("skip gsplat (no CUDA)")
for k, r in enumerate(renderers):
    run(r, opt.port + k)
    if toy:
        run_compare(r, opt.port + 10 + k)
print(f"screenshots in {opt.out}")
print("points mode verified" if ok else "points mode MISMATCH")
sys.exit(0 if ok else 1)
