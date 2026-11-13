#!/usr/bin/env python3
"""Checks the two-sequence comparison (viewer.py --compare) in headless Chrome, for both renderers.

A = toy scene, B = a sparser toy scene with another seed. For each renderer the script takes screenshots
in the modes "A", "B" and "A | B" and at a second time, then checks: A and B alone differ; B alone sits
where A alone sits; side by side shows content in both halves of the scene area; both halves change with time.
The gsplat part is skipped without CUDA. Screenshots go to --out.

    python tests/check_compare.py [--out shots/compare] [--chrome /usr/bin/google-chrome]
"""
import argparse, os, subprocess, sys, tempfile, threading, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import viewer as V
from gs4d import Sequence4D
from PIL import Image
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "shots", "compare"))
ap.add_argument("--chrome", default="/usr/bin/google-chrome")
opt = ap.parse_args()
os.makedirs(opt.out, exist_ok=True)
tmp = tempfile.mkdtemp()
pa, pb = os.path.join(tmp, "toy_a.npz"), os.path.join(tmp, "toy_b.npz")
for p, extra in ((pa, []), (pb, ["--seed", "3", "--points", "600"])):
    subprocess.run([sys.executable, os.path.join(ROOT, "make_toy.py"), "-o", p, *extra], check=True, stdout=subprocess.DEVNULL)
W, H = 1280, 760
SCENE_W = int(W * 0.7)  # the GUI panel covers the right edge
FLAGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
ok = True


def check(good, text):
    global ok
    ok &= bool(good)
    print(f"{'ok ' if good else 'FAIL'} {text}")


def content(img):
    return np.abs(img - img[2, 2]).max(-1) > 30  # not background


def run(renderer, port):
    args = argparse.Namespace(renderer=renderer, host="127.0.0.1", port=port, up="+z", max_splats=300_000,
                              max_res=1080, file=pa, compare=pb)
    vw = V.Viewer(Sequence4D.load(pa), args, Sequence4D.load(pb))
    threading.Thread(target=vw.run, daemon=True).start()
    shots = {}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=FLAGS)
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto(f"http://localhost:{port}")
        time.sleep(8)
        for name, mode, t in (("side_t0", V.SIDE_BY_SIDE, 0.3), ("side_t1", V.SIDE_BY_SIDE, 1.1), ("a", "A", 0.3), ("b", "B", 0.3)):
            vw.mode.value = mode
            vw._set_time(t)
            time.sleep(3.0)
            path = os.path.join(opt.out, f"{renderer}_{name}.png")
            pg.screenshot(path=path)
            shots[name] = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)[:, :SCENE_W]
        b.close()
    vw.server.stop()

    a, b_, s0, s1 = shots["a"], shots["b"], shots["side_t0"], shots["side_t1"]
    ca, cb = content(a), content(b_)
    check(ca.mean() > 0.01 and cb.mean() > 0.01, f"{renderer}: A alone and B alone are drawn ({ca.mean():.1%}, {cb.mean():.1%} covered)")
    check(np.abs(a - b_).mean() > 0.5, f"{renderer}: A alone and B alone differ (mean {np.abs(a - b_).mean():.2f}/255)")
    xa, xb = np.nonzero(ca)[1].mean(), np.nonzero(cb)[1].mean()
    check(abs(xa - xb) < 0.08 * SCENE_W, f"{renderer}: B alone sits where A alone sits (centroid x {xa:.0f} vs {xb:.0f})")
    half = SCENE_W // 2 if renderer == "client" else W // 2  # gsplat splits the whole window, the panel hides part of B
    cs = content(s0)
    left, right = cs[:, :half].mean(), cs[:, half:].mean()
    check(left > 0.01 and right > 0.01, f"{renderer}: side by side has content in both halves ({left:.1%}, {right:.1%})")
    dl, dr = np.abs(s0 - s1)[:, :half].mean(), np.abs(s0 - s1)[:, half:].mean()
    check(dl > 0.3 and dr > 0.3, f"{renderer}: both halves change with time (mean change {dl:.2f}, {dr:.2f}/255)")


run("client", 8151)
if V.pick_renderer("auto") == "gsplat":
    run("gsplat", 8152)
else:
    print("skip gsplat (no CUDA)")
print(f"screenshots in {opt.out}")
print("comparison verified" if ok else "comparison MISMATCH")
sys.exit(0 if ok else 1)
