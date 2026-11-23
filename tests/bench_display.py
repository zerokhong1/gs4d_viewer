#!/usr/bin/env python3
"""Display speed of the gsplat renderer in a real (GPU) Chrome on this machine, while playing (WP19b).

Starts the viewer in this process, opens a Chrome WINDOW on the local display (not headless, no SwiftShader),
presses play and counts, per "Update rate":
  sent       frames the server rendered and sent per second (calls of Viewer.render_gsplat)
  shown      browser frames per second that drew a new picture: requestAnimationFrame intervals in which the page
             uploaded a new full-size image to a WebGL texture (texImage2D / texSubImage2D are wrapped to count;
             the wrapper does no work besides counting)
  uploads    full-size texture uploads per second (more than `shown` when two pictures arrive within one browser frame)
  browser    requestAnimationFrame rate of the page (the compositor rate, limited by the monitor)
The first pass runs without the wrapper and reports `sent` and `browser` only, as a control.
(Copying the canvas in every frame to compare pixels was tried first: it halves the browser rate, so it is not used.)

    python tests/bench_display.py [file.npz] [--seconds 10] [--rates 20 60] [--size 1280 720] [--draw-as points]

Needs CUDA, gsplat, an X display and Chrome. Not part of tests/run_all.py (opens a window, takes about a minute).
"""
import argparse
import os
import sys
import threading
import time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import viewer as V
from gs4d import Sequence4D
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("file", nargs="?", default=os.path.join(ROOT, "..", "data_real", "som", "paper-windmill_split_compact.npz"))
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--rates", type=int, nargs="+", default=[20, 60], help='values of the "Update rate (Hz)" slider')
ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
ap.add_argument("--draw-as", choices=[V.SPLATS, V.POINTS], default=V.SPLATS, help='"Draw as" in the Display folder (WP24)')
ap.add_argument("--up", default="+z")
ap.add_argument("--port", type=int, default=8161)
ap.add_argument("--chrome", default="/usr/bin/google-chrome")
opt = ap.parse_args()
W, H = opt.size

HOOK = """
window.__uploads = 0; window.__sizes = {};
for (const name of ['texImage2D', 'texSubImage2D']) {
  const orig = WebGL2RenderingContext.prototype[name];
  WebGL2RenderingContext.prototype[name] = function (...a) {
    const src = a[a.length - 1];
    if (src && typeof src === 'object' && src.width * src.height >= MINPIX) {
      window.__uploads++;
      const k = src.width + 'x' + src.height; window.__sizes[k] = (window.__sizes[k] || 0) + 1;
    }
    return orig.apply(this, a);
  };
}
"""
GPU_NAME = """() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  const e = gl && gl.getExtension('WEBGL_debug_renderer_info');
  return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'no WebGL';
}"""
COUNT = """async (ms) => {
  const src = [...document.querySelectorAll('canvas')].sort((a, b) => b.width * b.height - a.width * a.height)[0];
  const hook = window.__uploads !== undefined;
  let frames = 0, changed = 0, last = hook ? window.__uploads : 0;
  const u0 = last, t0 = performance.now();
  window.__sizes = {};
  await new Promise(done => {
    const tick = () => {
      frames++;
      if (hook && window.__uploads !== last) { changed++; last = window.__uploads; }
      if (performance.now() - t0 < ms) requestAnimationFrame(tick); else done();
    };
    requestAnimationFrame(tick);
  });
  const s = (performance.now() - t0) / 1000;
  return {visible: document.visibilityState === 'visible' && document.hasFocus(), browser: frames / s, shown: changed / s, uploads: (last - u0) / s, sizes: window.__sizes, canvas: [src.width, src.height]};
}"""

args = argparse.Namespace(renderer="gsplat", host="127.0.0.1", port=opt.port, up=opt.up, max_splats=300_000, max_res=H,
                          file=opt.file, compare=None, cameras=None, draw_as=opt.draw_as, gt_background="black", cpu_decode=False, video_dir=".")
seq = Sequence4D.load(opt.file, lazy=True)
vw = V.Viewer(seq, args, None)
sent = {"n": 0, "size": None}
render = vw.render_gsplat


def counted(wxyz, position, fov, w, h):
    sent["n"] += 1
    sent["size"] = (w, h)
    return render(wxyz, position, fov, w, h)


vw.render_gsplat = counted
threading.Thread(target=vw.run, daemon=True).start()

rows = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=opt.chrome, headless=False, args=[
        "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding", "--disable-background-timer-throttling",
        "--disable-features=CalculateNativeWinOcclusion", f"--window-size={W + 40},{H + 160}"])
    version = b.version
    for copy in (False, True):
        pg = b.new_page(viewport={"width": W, "height": H})
        if copy:
            pg.add_init_script(HOOK.replace("MINPIX", str(W * H // 2)))
        pg.goto(f"http://localhost:{opt.port}")
        time.sleep(8)
        gpu = pg.evaluate(GPU_NAME)
        vw.loop.value = True
        vw.playing = True
        for rate in opt.rates:
            vw.target_fps.value = rate
            pg.bring_to_front()
            time.sleep(2)
            n0, t0 = sent["n"], time.time()
            r = pg.evaluate(COUNT, opt.seconds * 1000)
            r["sent"] = (sent["n"] - n0) / (time.time() - t0)
            rows.append((copy, rate, r))
        vw.playing = False
        pg.close()
    b.close()
vw.server.stop()

print(f"\n{os.path.basename(opt.file)}: {seq.num_gaussians:,} Gaussians x {seq.num_frames} keyframes")
print(f"Chrome {version}, WebGL renderer: {gpu}, draw as {opt.draw_as}")
print(f"render size {sent['size'][0]}x{sent['size'][1]}, browser canvas {rows[-1][2]['canvas'][0]}x{rows[-1][2]['canvas'][1]}, {opt.seconds:g} s per row")
print(f"{'update rate':>12} {'sent/s':>8} {'shown/s':>8} {'uploads/s':>10} {'browser/s':>10}  upload sizes")
for copy, rate, r in rows:
    shown = f"{r['shown']:8.1f} {r['uploads']:10.1f}" if copy else f"{'-':>8} {'-':>10}"
    print(f"{rate:>9} Hz {r['sent']:8.1f} {shown} {r['browser']:10.1f}  {r['sizes'] if copy else ''}{'' if r['browser'] > 30 else '  INVALID: window hidden or throttled'}")
