#!/usr/bin/env python3
"""WP34: camera path (viewer.py "Camera path" folder, --camera-path, mp4 along the path).

1. path_pose against an independent computation (Hermite form of Catmull-Rom with explicit tangents, slerp by exp/log of
   viser.transforms.SO3) at 5 values of u, open and closed; the key poses are reproduced exactly; save/load round trip.
2. In headless Chrome, for each renderer: a 3-key path on the toy scene is put on the client's camera at 5 values of u and the
   camera the server reads back must match the interpolation; then mp4 export along the path (frame count = fps x duration,
   frames differ, closed path loops) and the "run through the sequence" mode moves the time slider.

    python tests/check_path.py
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
from viser.transforms import SO3
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--chrome", default="/usr/bin/google-chrome")
ap.add_argument("--port", type=int, default=8171)
ap.add_argument("--orbit-mp4", metavar="FILE", default=None,
                help="instead of the checks: export an mp4 of FILE along an orbit (closed, runs through the sequence) into shots/path/")
ap.add_argument("--name", default="orbit")
ap.add_argument("--up", default="+z")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--max-res", type=int, default=720)
opt = ap.parse_args()
ok = True


def orbit_mp4():
    """Manual run of WP34: orbit around the scene while the sequence plays, gsplat renderer, written to shots/path/<name>.mp4."""
    out = os.path.join(ROOT, "shots", "path")
    os.makedirs(out, exist_ok=True)
    args = argparse.Namespace(renderer="gsplat", host="127.0.0.1", port=opt.port, up=opt.up, max_splats=300_000, max_res=opt.max_res,
                              file=opt.orbit_mp4, compare=None, video_dir=out)
    vw = V.Viewer(Sequence4D.load(opt.orbit_mp4, lazy=True), args)
    threading.Thread(target=vw.run, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
                                                                  "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.goto(f"http://localhost:{opt.port}")
        time.sleep(8)
        client = next(iter(vw.server.get_clients().values()))
        centre, dist = vw._home()
        vw.path_keys = V.orbit_keys(centre, dist, V.VIEW_DIR[opt.up], vw.up_vec, float(client.camera.fov))
        vw.path_closed.value, vw.path_time.value, vw.path_duration.value = True, V.PATH_RUN, opt.seconds
        vw.video_cam.value, vw.video_fps.value = V.VIDEO_PATH, 30
        t0 = time.time()
        vw._export_video(client)
        print(f"orbit mp4: {opt.seconds:.0f} s at 30 fps rendered in {time.time() - t0:.1f} s")
        b.close()
    vw.server.stop()
    mp4 = max((f for f in os.listdir(out) if f.endswith(".mp4") and "_path_" in f), key=lambda f: os.path.getmtime(os.path.join(out, f)))
    os.replace(os.path.join(out, mp4), os.path.join(out, f"{opt.name}.mp4"))
    print(f"wrote {os.path.join(out, opt.name + '.mp4')}")


if opt.orbit_mp4:
    orbit_mp4()
    sys.exit(0)


def check(good, text):
    global ok
    ok &= bool(good)
    print(f"{'ok ' if good else 'FAIL'} {text}")


def key(pos, rpy, fov):
    return dict(position=np.array(pos, float), wxyz=np.asarray(SO3.from_rpy_radians(*rpy).wxyz, float), fov=fov)


def reference(keys, u, closed):
    """Independent: Hermite segment with tangents m_i = (p_{i+1} - p_{i-1}) / 2 (end points: one-sided), slerp via exp/log."""
    n = len(keys)
    P = np.array([k["position"] for k in keys])
    nseg = n if closed else n - 1
    s = u * nseg
    i = min(int(s), nseg - 1)
    f = s - i
    nb = (lambda a: P[a % n]) if closed else (lambda a: P[min(max(a, 0), n - 1)])
    m0, m1 = (nb(i + 1) - nb(i - 1)) / 2, (nb(i + 2) - nb(i)) / 2
    h00, h10, h01, h11 = 2 * f ** 3 - 3 * f ** 2 + 1, f ** 3 - 2 * f ** 2 + f, -2 * f ** 3 + 3 * f ** 2, f ** 3 - f ** 2
    pos = h00 * nb(i) + h10 * m0 + h01 * nb(i + 1) + h11 * m1
    qa, qb = SO3(keys[i]["wxyz"]), SO3(keys[(i + 1) % n]["wxyz"])
    q = qa @ SO3.exp(f * (qa.inverse() @ qb).log())
    fov = keys[i]["fov"] + f * (keys[(i + 1) % n]["fov"] - keys[i]["fov"])
    return pos, np.asarray(q.wxyz), fov


def same_rot(a, b):
    return min(np.abs(np.asarray(a) - np.asarray(b)).max(), np.abs(np.asarray(a) + np.asarray(b)).max())


# ---------------------------------------------------------------- 1. interpolation
keys = [key((0, 0, 0), (0.0, 0.0, 0.0), 0.8), key((2, 1, 0.5), (0.2, 0.4, 1.0), 1.0), key((3, 3, 1.5), (-0.3, 0.9, 2.2), 0.6)]
for closed in (False, True):
    worst = [0, 0, 0]
    for u in (0.0, 0.2, 0.5, 0.8, 1.0):
        pos, q, fov = V.path_pose(keys, u, closed)
        rp, rq, rf = reference(keys, u, closed)
        worst = [max(worst[0], np.abs(pos - rp).max()), max(worst[1], same_rot(q, rq)), max(worst[2], abs(fov - rf))]
    check(max(worst) < 1e-9, f"{'closed' if closed else 'open'} path at u = 0, .2, .5, .8, 1 equals the independent interpolation "
                             f"(position {worst[0]:.1e}, orientation {worst[1]:.1e}, fov {worst[2]:.1e})")
for closed, us in ((False, (0.0, 0.5, 1.0)), (True, (0.0, 1 / 3, 2 / 3))):
    for u, k in zip(us, keys):
        pos, q, fov = V.path_pose(keys, u, closed)
        check(np.abs(pos - k["position"]).max() < 1e-9 and same_rot(q, k["wxyz"]) < 1e-9 and abs(fov - k["fov"]) < 1e-12,
              f"{'closed' if closed else 'open'}: u = {u:.3f} is exactly key {us.index(u) + 1}")
orb = V.orbit_keys(np.array([1.0, 2.0, 3.0]), 4.0, (1, -1, 0.6), np.array([0, 0, 1.0]), 0.9, n=12)
cs = np.array([k["position"] for k in orb])
d = np.linalg.norm(cs - [1, 2, 3], axis=1)
fwd = np.array([SO3(k["wxyz"]).as_matrix()[:, 2] for k in orb])
check(np.ptp(d) < 1e-9 and abs(d[0] - 4.0) < 1e-9 and np.ptp(cs[:, 2]) < 1e-9, "orbit keys: constant distance and height around the centre")
check(np.abs(fwd - (np.array([1, 2, 3]) - cs) / d[:, None]).max() < 1e-9, "orbit keys: every camera looks at the centre")
tmp = tempfile.mkdtemp()
V.save_path(os.path.join(tmp, "p.json"), keys, 7.5, True, True)
back = V.load_path(os.path.join(tmp, "p.json"))
check(back["duration"] == 7.5 and back["closed"] and back["run"] and len(back["keys"]) == 3 and
      all(np.abs(a["position"] - b["position"]).max() < 1e-12 and same_rot(a["wxyz"], b["wxyz"]) < 1e-12 for a, b in zip(keys, back["keys"])),
      "save_path / load_path round trip")

# ---------------------------------------------------------------- 2. viewer in the browser
toy = os.path.join(tmp, "toy.npz")
subprocess.run([sys.executable, os.path.join(ROOT, "make_toy.py"), "-o", toy], check=True, stdout=subprocess.DEVNULL)
FLAGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def run(renderer, port):
    args = argparse.Namespace(renderer=renderer, host="127.0.0.1", port=port, up="+z", max_splats=300_000, max_res=360,
                              file=toy, compare=None, video_dir=os.path.join(tmp, renderer), camera_path=os.path.join(tmp, "p.json"))
    vw = V.Viewer(Sequence4D.load(toy, lazy=True), args)
    threading.Thread(target=vw.run, daemon=True).start()
    check(len(vw.path_keys) == 3 and vw.path_duration.value == 7.5 and vw.path_closed.value and vw.path_time.value == V.PATH_RUN,
          f"{renderer}: --camera-path loads keys, duration, closed and time mode into the panel")
    vw.path_closed.value, vw.path_time.value, vw.path_duration.value = False, V.PATH_HOLD, 2.0
    vw.path_keys = [dict(k) for k in keys]
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=opt.chrome, args=FLAGS)
        pg = b.new_page(viewport={"width": 640, "height": 400})
        pg.goto(f"http://localhost:{port}")
        time.sleep(6)
        client = next(iter(vw.server.get_clients().values()))
        worst = [0, 0, 0]
        for u in (0.0, 0.25, 0.5, 0.8, 1.0):
            vw._apply_path(u)
            time.sleep(1.0)
            rp, rq, rf = reference(keys, u, False)
            c = client.camera
            worst = [max(worst[0], np.abs(np.array(c.position) - rp).max()), max(worst[1], same_rot(c.wxyz, rq)), max(worst[2], abs(c.fov - rf))]
        check(max(worst[0], worst[1]) < 1e-4 and worst[2] < 1e-5,
              f"{renderer}: the browser's camera follows the path at 5 values of u (position {worst[0]:.1e}, orientation {worst[1]:.1e}, fov {worst[2]:.1e})")
        # run through the sequence: the time slider follows u
        vw.path_time.value = V.PATH_RUN
        vw._apply_path(0.5)
        check(abs(vw.time.value - (vw.t0 + 0.5 * vw.duration)) < 1e-6, f"{renderer}: 'run through the sequence' puts the time at t0 + u x duration")
        # playing: the main loop moves the camera and stops at the end
        vw.path_time.value, vw.loop.value = V.PATH_HOLD, False
        vw.path_u, vw.path_playing = 0.0, True
        time.sleep(1.0)
        mid = vw.path_u
        time.sleep(2.5)
        check(0.1 < mid < 0.9 and not vw.path_playing and vw.path_u == 1.0, f"{renderer}: Play advances u ({mid:.2f} after 1 s of a 2 s path) and stops at the end")
        # mp4 along the path
        import imageio.v2 as imageio
        vw.video_fps.value, vw.video_cam.value = 10, V.VIDEO_PATH
        for closed in (False, True):
            vw.path_closed.value = closed
            t0 = time.time()
            vw._export_video(client)
            took = time.time() - t0
            mp4 = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))[-1]
            frames = [np.asarray(f, np.float32) for f in imageio.get_reader(os.path.join(args.video_dir, mp4))]
            check(len(frames) == 20, f"{renderer}: mp4 along the {'closed' if closed else 'open'} path has {len(frames)} frames (10 fps x 2 s), {took:.1f} s")
            diff = np.abs(frames[0] - frames[len(frames) // 2]).mean()
            check(diff > 0.5, f"{renderer}: first and middle frame differ (mean {diff:.1f}/255)")
            if closed:  # the last frame is one step before the first
                step = np.abs(frames[-1] - frames[0]).mean()
                check(step < diff, f"{renderer}: closed path loops (last-first {step:.1f} < first-middle {diff:.1f})")
            time.sleep(1.5)
        b.close()
    vw.server.stop()


renderers = ["client"] + (["gsplat"] if V.pick_renderer("auto") == "gsplat" else [])
for k, r in enumerate(renderers):
    run(r, opt.port + k)
print("camera path verified" if ok else "camera path MISMATCH")
sys.exit(0 if ok else 1)
