#!/usr/bin/env python3
"""WP18: --cameras, camera list and ground-truth comparison in the viewer (needs CUDA, ../data_real, and Chrome for the screenshots).

1. load_cameras on D-NeRF transforms_test.json / transforms_train.json (counts, OpenGL -> OpenCV, time, image found).
2. For SC-GS and Deformable-3DGS `bouncingballs` (converted with --sh, as in check_real.py): the PSNR shown by the viewer
   (Viewer.gt_compare, default Hide-opacity 0.02) vs check_real.psnr_item on the 20 test cameras: max difference must be <= 0.1 dB.
3. Headless Chrome: select a camera, switch the comparison on, save a screenshot in --out.

    python tests/check_cameras.py [--out shots/cameras] [--no-browser]
"""
import argparse, os, sys, tempfile, threading, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import check_real as CR
import viewer as V
from gs4d import Sequence4D

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=CR.default_data_root())
ap.add_argument("--train", default="/home/cont/gs4d_train")
ap.add_argument("--out", default=os.path.join(ROOT, "shots", "cameras"))
ap.add_argument("--chrome", default="/usr/bin/google-chrome")
ap.add_argument("--no-browser", action="store_true")
opt = ap.parse_args()
os.makedirs(opt.out, exist_ok=True)
scene = os.path.join(opt.data, "dnerf", "data", "bouncingballs")
ok = True


def check(good, text):
    global ok
    ok &= bool(good)
    print(f"{'ok ' if good else 'FAIL'} {text}")


tr, te = V.load_cameras(os.path.join(scene, "transforms_train.json")), V.load_cameras(os.path.join(scene, "transforms_test.json"))
check(len(te) == 20 and all(c["image"] and c["time"] is not None and c["w"] == 800 for c in te), f"test cameras: {len(te)}, images and times found")
check(len(tr) > 20 and all(c["image"] for c in tr), f"train cameras: {len(tr)}")
check(all(abs(np.linalg.det(c["c2w"][:3, :3]) - 1) < 1e-4 for c in te), "rotations are proper (det 1) after the OpenGL flip")

work = tempfile.mkdtemp(prefix="check_cameras_")
models = {"scgs": ("convert_scgs.py", os.path.join(opt.train, "out", "scgs_bouncingballs_node"), 400),
          "deformable3dgs": ("convert_deformable3dgs.py", os.path.join(opt.train, "out", "d3dgs_bouncingballs"), 800)}
npz = {}
port = 8161
for name, (script, model, size) in models.items():
    out = os.path.join(work, f"{name}.npz")
    CR.convert(script, [CR.need(model, name), "--frames", "60", "--sh"], out)
    npz[name] = out
    test = CR.load_test_set(scene, size)
    ref, _ = CR.psnr_item(name, out, test)
    args = argparse.Namespace(renderer="gsplat", host="127.0.0.1", port=port, up="+z", max_splats=1, max_res=size, file=out,
                              compare=None, cameras=os.path.join(scene, "transforms_test.json"), gt_background="black")
    vw = V.Viewer(Sequence4D.load(out, lazy=True), args)
    shown = []
    for k, c in enumerate(te):
        vw._set_time(c["time"])
        vw._load_gsplat()
        shown.append(vw.gt_compare(0, k, False, size)[2])
    shown = np.array(shown)
    d = np.abs(shown - ref).max()
    check(d <= 0.1, f"{name} at {size}px: viewer PSNR vs check_real, max |diff| {d:.4f} dB (mean {shown.mean():.2f} vs {ref.mean():.2f})")
    vw.server.stop()
    port += 1

if not opt.no_browser:
    from PIL import Image
    from playwright.sync_api import sync_playwright
    FLAGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
    name = "scgs"
    args = argparse.Namespace(renderer="gsplat", host="127.0.0.1", port=8170, up="+z", max_splats=1, max_res=400, file=npz[name],
                              compare=None, cameras=os.path.join(scene, "transforms_test.json"), gt_background="black")
    vw = V.Viewer(Sequence4D.load(npz[name], lazy=True), args)
    threading.Thread(target=vw.run, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=FLAGS)
        pg = b.new_page(viewport={"width": 1400, "height": 800})
        pg.goto("http://localhost:8170")
        time.sleep(6)
        pg.screenshot(path=os.path.join(opt.out, "free_camera.png"))
        client = next(iter(vw.server.get_clients().values()))
        vw._select_camera(7, client)          # what a click in the list does
        time.sleep(3)
        pg.screenshot(path=os.path.join(opt.out, "camera_7.png"))
        check(abs(vw.time.value - te[7]["time"]) < 1e-6, f"selecting a camera moves the slider to the image time ({vw.time.value:.3f})")
        cam = client.camera
        d = np.abs(np.array(cam.position) - te[7]["c2w"][:3, 3]).max()
        check(d < 1e-3, f"client camera is at the training camera position (|diff| {d:.1e})")
        vw.gt_on.value = True
        time.sleep(3)
        pg.screenshot(path=os.path.join(opt.out, "ground_truth.png"))
        check("PSNR" in vw.gt_info.content, f"panel text: {vw.gt_info.content!r}")
        vw._select_camera(13, client)
        time.sleep(3)
        pg.screenshot(path=os.path.join(opt.out, "ground_truth_13.png"))
        b.close()
    vw.server.stop()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
