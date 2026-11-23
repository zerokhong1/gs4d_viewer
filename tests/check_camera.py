#!/usr/bin/env python3
"""Checks viewer.camera_matrices against what the browser actually draws (no GPU needed).

Starts the viewer with the browser renderer on the toy scene, reads the connected client's camera,
projects the centres of the red and blue balls with camera_matrices(), and checks that the screenshot
has the expected colour at those pixels. Needs: playwright + a Chrome/Chromium binary.

    python tests/check_camera.py [/usr/bin/google-chrome]
"""
import argparse
import os
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

chrome = sys.argv[1] if len(sys.argv) > 1 else "/usr/bin/google-chrome"
tmp = tempfile.mkdtemp()
toy = os.path.join(tmp, "toy.npz")
os.system(f'"{sys.executable}" "{os.path.join(ROOT, "tools", "make_toy.py")}" -o "{toy}" > /dev/null')
args = argparse.Namespace(renderer="client", host="127.0.0.1", port=8131, up="+z", max_splats=300_000, max_res=1080)
vw = V.Viewer(Sequence4D.load(toy), args)
threading.Thread(target=vw.run, daemon=True).start()
W, H = 1280, 760
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=chrome, args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
                                                        "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto("http://localhost:8131")
    time.sleep(8)
    shot = os.path.join(tmp, "s.png")
    pg.screenshot(path=shot)
    cam = next(iter(vw.server.get_clients().values())).camera
    wxyz, pos, fov, aspect = cam.wxyz, cam.position, cam.fov, cam.aspect
    b.close()
img = np.asarray(Image.open(shot).convert("RGB")).astype(int)
assert abs(aspect - W / H) < 0.02, f"canvas aspect {aspect} != viewport {W / H}"
viewmat, K = V.camera_matrices(wxyz, pos, fov, W, H)
ok = True
for name, centre, chan in (("red ball", (1.5, 0.0, 0.45), 0), ("blue ball", (0.0, 0.0, 0.3), 2)):
    pc = viewmat @ np.array([*centre, 1.0], np.float32)
    u, v = (K @ pc[:3])[:2] / pc[2]
    rgb = img[int(round(v)) - 3:int(round(v)) + 4, int(round(u)) - 3:int(round(u)) + 4].reshape(-1, 3).mean(0)
    good = pc[2] > 0 and rgb[chan] > 150 and rgb[chan] - rgb[[c for c in range(3) if c != chan]].max() > 60
    ok &= bool(good)
    print(f"{'ok ' if good else 'FAIL'} {name}: projects to ({u:.0f}, {v:.0f}), depth {pc[2]:.2f}, screenshot colour {rgb.round()}")
print("camera convention verified" if ok else "camera convention MISMATCH")
sys.exit(0 if ok else 1)
