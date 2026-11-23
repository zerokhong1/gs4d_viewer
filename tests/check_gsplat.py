#!/usr/bin/env python3
"""Checks the gsplat renderer against the browser (client) renderer on the toy scene. Needs CUDA.

For each renderer: start the viewer, open it in headless Chrome, take a screenshot at three times with
the same camera. Then: (1) gsplat and client screenshots must agree, (2) the gsplat image must change
when the time changes and when the camera moves. Screenshots are written to --out for a visual check.

    python tests/check_gsplat.py [--out shots] [--chrome /usr/bin/google-chrome]
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
ap.add_argument("--out", default=os.path.join(ROOT, "shots"))
ap.add_argument("--chrome", default="/usr/bin/google-chrome")
ap.add_argument("--file", default=None, help="sequence to use instead of a fresh toy scene")
opt = ap.parse_args()
os.makedirs(opt.out, exist_ok=True)
path = opt.file
if path is None:
    path = os.path.join(tempfile.mkdtemp(), "toy.npz")
    subprocess.run([sys.executable, os.path.join(ROOT, "make_toy.py"), "-o", path], check=True, stdout=subprocess.DEVNULL)
seq = Sequence4D.load(path)
t0, t1 = float(seq.times[0]), float(seq.times[-1])
times = [t0, t0 + 0.37 * (t1 - t0), t0 + 0.8 * (t1 - t0)]
W, H = 1280, 760
FLAGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def grab(pg, name):
    p = os.path.join(opt.out, name)
    pg.screenshot(path=p)
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)


def capture(renderer, port):
    args = argparse.Namespace(renderer=renderer, host="127.0.0.1", port=port, up="+z", max_splats=300_000, max_res=1080)
    vw = V.Viewer(seq, args)
    threading.Thread(target=vw.run, daemon=True).start()
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=FLAGS)
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto(f"http://localhost:{port}")
        time.sleep(8)
        for i, t in enumerate(times):
            vw._set_time(t)
            time.sleep(2.5)
            out[i] = grab(pg, f"{renderer}_t{i}.png")
        if renderer == "gsplat":  # orbit a quarter turn around the scene centre at the last time
            cam = next(iter(vw.server.get_clients().values())).camera
            d = np.asarray(cam.position) - vw.centre
            cam.position = tuple(vw.centre + np.array([-d[1], d[0], d[2]]))
            cam.look_at = tuple(vw.centre)
            time.sleep(2.5)
            out["orbit"] = grab(pg, "gsplat_orbit.png")
        b.close()
    vw.server.stop()
    return out


g = capture("gsplat", 8141)
c = capture("client", 8142)
ok = True
scene = slice(0, int(W * 0.7))  # the GUI panel covers the right edge of both screenshots
for i, t in enumerate(times):
    a, b_ = g[i][:, scene], c[i][:, scene]
    diff = np.abs(a - b_).mean()
    content = np.abs(a - a[0, 0]).max(-1) > 30  # pixels that are not background in the gsplat render
    content_c = np.abs(b_ - b_[0, 0]).max(-1) > 30
    iou = (content & content_c).sum() / max((content | content_c).sum(), 1)
    good = diff < 8.0 and iou > 0.8 and content.mean() > 0.01
    ok &= bool(good)
    print(f"{'ok ' if good else 'FAIL'} t={t:.3f}: mean |gsplat - client| = {diff:.2f}/255, silhouette IoU = {iou:.3f}, "
          f"covered = {content.mean():.1%}")
for name, a, b_ in (("time change", g[0], g[1]), ("time change", g[1], g[2]), ("camera move", g[2], g["orbit"])):
    d = np.abs(a[:, scene] - b_[:, scene]).mean()
    ok &= bool(d > 0.5)
    print(f"{'ok ' if d > 0.5 else 'FAIL'} image updates on {name}: mean change {d:.2f}/255")
print(f"screenshots in {opt.out}")
print("gsplat renderer verified" if ok else "gsplat renderer MISMATCH")
sys.exit(0 if ok else 1)
