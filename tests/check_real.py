#!/usr/bin/env python3
"""Repeats the hand checks of 14/11/2026 on real method outputs, in one command. Needs CUDA and the real data.

PSNR items (D-NeRF `bouncingballs`, ground-truth images exist): run the converter, render the .npz with gsplat at
the camera and time of EVERY image in transforms_test.json (black background), print min/mean/max PSNR,
fail when the minimum is below the item's threshold.
  Deformable-3DGS --sh, SC-GS --sh, native 4DGS.
  Evaluation size = the resolution each model was TRAINED at (a 3DGS model rendered at another size loses several
  dB): Deformable-3DGS 800 px (resolution=-1), SC-GS 400 px (--resolution 2), native 4DGS 400 px (resolution: 2).
  The 800 px ground truth is resized with LANCZOS on the RGBA image, then composited on black; fx scales with W.
Smoke items (no original images in the data folder, so images are NOT compared): run the converter and check
the Gaussian count, keyframe count and file size against the README.
  Spacetime Gaussians (lite model), Shape of Motion (paper-windmill).

An item whose data or CUDA is missing is SKIPPED with the reason; it is never reported as passed.
Exit code: 1 if any item failed (or, with --strict, was skipped), else 0.

    python tests/check_real.py [--data DIR] [--train DIR] [--work DIR] [--only NAME[,NAME]] [--strict]

--data  : folder with dnerf/, spacetime/, som/   (env GS4D_DATA, default: `data_real` next to the main checkout)
--train : folder with out/ and 4dgs/             (env GS4D_TRAIN, default /home/cont/gs4d_train)
--work  : where the converted .npz files go (default: a temporary folder, deleted at the end)
Model paths can also be set one by one with env GS4D_D3DGS, GS4D_SCGS, GS4D_NATIVE4D, GS4D_SPACETIME, GS4D_SOM.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Minimum PSNR (dB) per method = the lowest value measured over all 20 test images on 14/11/2026 (RTX 3060),
# minus 0.5 dB. See the table printed by this script and the README.
MEASURED_ON = "14/11/2026"
MIN_PSNR = {"deformable3dgs": 36.70, "scgs": 34.85, "native4d": 25.54}  # measured minima 37.20, 35.35, 26.04, minus 0.5
EVAL_SIZE = {"deformable3dgs": 800, "scgs": 400, "native4d": 400}  # training resolution of each model

SIZE_TOL = 0.05  # file size must be within 5 % of the README figure


def default_data_root():
    here = ROOT
    if f"{os.sep}.claude{os.sep}worktrees{os.sep}" in here:  # inside a worktree of the main checkout
        here = here.split(f"{os.sep}.claude{os.sep}worktrees{os.sep}")[0]
    return os.path.join(os.path.dirname(here), "data_real")


class Skip(Exception):
    pass


def need(path, what):
    if not os.path.exists(path):
        raise Skip(f"{what} not found: {path}")
    return path


def convert(script, args, out):
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, script)] + args + ["-o", out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{r.stdout[-800:]}\n{r.stderr[-1500:]}")
    return time.time() - t


# ------------------------------------------------------------------------------------------ PSNR items
def load_test_set(scene, size):
    from PIL import Image
    meta = json.load(open(need(os.path.join(scene, "transforms_test.json"), "transforms_test.json")))
    fx_scale = float(meta["camera_angle_x"])
    out = []
    for fr in meta["frames"]:
        p = need(os.path.join(scene, fr["file_path"].lstrip("./") + ".png"), "test image")
        pil = Image.open(p).convert("RGBA")
        if pil.size[0] != size:
            pil = pil.resize((size, size * pil.size[1] // pil.size[0]), Image.LANCZOS)
        im = np.asarray(pil).astype(np.float32) / 255.0
        gt = im[..., :3] * im[..., 3:4]  # composite on black
        c2w = np.array(fr["transform_matrix"], np.float64)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV
        out.append((float(fr["time"]), np.linalg.inv(c2w).astype(np.float32), gt))
    H, W = out[0][2].shape[:2]
    f = 0.5 * W / np.tan(0.5 * fx_scale)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    return out, K, W, H


def psnr_item(name, npz, test):
    import torch
    from gsplat import rasterization
    from gs4d import Sequence4D
    from gs4d.format import unpack_cov
    frames, K, W, H = test
    seq = Sequence4D.load(npz)
    dev = "cuda"
    Kt = torch.from_numpy(K).to(dev)[None]
    ps = []
    for t, viewmat, gt in frames:
        means, covs, opac, cols = seq.at(t)
        sh = seq.sh_at(t)
        kw = {}
        if sh is not None:
            colors, kw["sh_degree"] = torch.from_numpy(np.ascontiguousarray(sh, np.float32)).to(dev), seq.sh_degree
        else:
            colors = torch.from_numpy(np.ascontiguousarray(cols, np.float32)).to(dev)
        with torch.no_grad():
            img, _, _ = rasterization(
                torch.from_numpy(np.ascontiguousarray(means, np.float32)).to(dev), None, None,
                torch.from_numpy(np.ascontiguousarray(opac, np.float32)).to(dev), colors,
                torch.from_numpy(viewmat).to(dev)[None], Kt, W, H,
                covars=torch.from_numpy(unpack_cov(np.ascontiguousarray(covs, np.float32))).to(dev), **kw)
        pred = img[0].clamp(0, 1).cpu().numpy()
        mse = float(np.mean((pred - gt) ** 2))
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        del img
    return np.array(ps), seq


# ------------------------------------------------------------------------------------------ items
def run_psnr(name, label, script, model, extra, work, data, results):
    test = load_test_set(os.path.join(data, "dnerf", "data", "bouncingballs"), EVAL_SIZE[name])
    need(model, f"{label} model")
    out = os.path.join(work, f"{name}.npz")
    secs = convert(script, [model] + extra, out)
    ps, seq = psnr_item(name, out, test)
    thr = MIN_PSNR[name]
    ok = ps.min() >= thr
    results.append(dict(name=label, size=EVAL_SIZE[name], kind="psnr", ok=ok, ps=ps, thr=thr, n=seq.num_gaussians, k=seq.num_frames,
                        mb=os.path.getsize(out) / 1e6, secs=secs))


def run_smoke(name, label, script, model, extra, work, expect, results):
    need(model, f"{label} model")
    out = os.path.join(work, f"{name}.npz")
    secs = convert(script, [model] + extra, out)
    from gs4d import Sequence4D
    seq = Sequence4D.load(out)
    mb = os.path.getsize(out) / 1e6
    bad = []
    if seq.num_gaussians != expect["n"]:
        bad.append(f"Gaussians {seq.num_gaussians} != README {expect['n']}")
    if seq.num_frames != expect["k"]:
        bad.append(f"keyframes {seq.num_frames} != README {expect['k']}")
    if abs(mb - expect["mb"]) > SIZE_TOL * expect["mb"]:
        bad.append(f"size {mb:.0f} MB differs more than {SIZE_TOL:.0%} from README {expect['mb']} MB")
    results.append(dict(name=label, kind="smoke", ok=not bad, bad=bad, n=seq.num_gaussians, k=seq.num_frames, mb=mb, secs=secs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("GS4D_DATA") or default_data_root())
    ap.add_argument("--train", default=os.environ.get("GS4D_TRAIN") or "/home/cont/gs4d_train")
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default="", help="comma list of: deformable3dgs,scgs,native4d,spacetime,som")
    ap.add_argument("--per-image", action="store_true", help="also print the PSNR of every test image")
    ap.add_argument("--strict", action="store_true", help="a skipped item also gives exit code 1")
    opt = ap.parse_args()

    E = os.environ.get
    paths = dict(
        deformable3dgs=E("GS4D_D3DGS") or os.path.join(opt.train, "out", "d3dgs_bouncingballs"),
        scgs=E("GS4D_SCGS") or os.path.join(opt.train, "out", "scgs_bouncingballs_node"),
        native4d=E("GS4D_NATIVE4D") or os.path.join(opt.train, "4dgs", "output", "dnerf", "bouncingballs", "chkpnt_best.pth"),
        spacetime=E("GS4D_SPACETIME") or os.path.join(opt.data, "spacetime", "n3d_sear_steak_lite_allcam", "point_cloud"),
        som=E("GS4D_SOM") or os.path.join(opt.data, "som", "paper-windmill", "checkpoints", "last.ckpt"),
    )
    # README figures (14/11/2026)
    spacetime_expect = dict(n=98993, k=50, mb=199)
    som_expect = dict(n=71867 + 924075, k=35, mb=147)   # 1270 MB before the static group (WP11)

    try:
        import torch
        cuda = torch.cuda.is_available()
        import gsplat  # noqa: F401
    except ImportError as e:
        torch, cuda = None, False
        cuda_why = f"torch/gsplat missing ({e})"
    else:
        cuda_why = "no CUDA device"

    work = opt.work or tempfile.mkdtemp(prefix="check_real_")
    os.makedirs(work, exist_ok=True)
    only = {s for s in opt.only.split(",") if s}
    results, skipped = [], []

    def spacetime_ply():
        p = paths["spacetime"]
        if os.path.isdir(p):  # model folder: pick the last iteration
            plys = sorted(os.path.join(d, "point_cloud.ply") for d, _, fs in os.walk(p) if "point_cloud.ply" in fs)
            if not plys:
                raise Skip(f"no point_cloud.ply under {p}")
            return plys[-1]
        return p

    items = [
        ("deformable3dgs", "Deformable-3DGS --sh", True,
         lambda: run_psnr("deformable3dgs", "Deformable-3DGS --sh", "convert_deformable3dgs.py", paths["deformable3dgs"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("scgs", "SC-GS --sh", True,
         lambda: run_psnr("scgs", "SC-GS --sh", "convert_scgs.py", paths["scgs"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("native4d", "native 4DGS", True,
         lambda: run_psnr("native4d", "native 4DGS", "convert_native4d.py", paths["native4d"],
                          ["--frames", "60", "--t0", "0", "--t1", "1"], work, opt.data, results)),
        ("spacetime", "Spacetime Gaussians (smoke)", False,
         lambda: run_smoke("spacetime", "Spacetime Gaussians (smoke)", "convert_spacetime.py", spacetime_ply(),
                           ["--frames", "50", "--t0", "0", "--t1", "1"], work, spacetime_expect, results)),
        ("som", "Shape of Motion (smoke)", False,
         lambda: run_smoke("som", "Shape of Motion (smoke)", "convert_shape_of_motion.py", paths["som"],
                           ["--stride", "8", "--fps", "30"], work, som_expect, results)),
    ]
    for key, label, needs_cuda, fn in items:
        if only and key not in only:
            continue
        try:
            if needs_cuda and not cuda:
                raise Skip(cuda_why)
            print(f"[{key}] running ...", flush=True)
            fn()
        except Skip as e:
            skipped.append((label, str(e)))
        finally:  # free the big .npz before the next item
            f = os.path.join(work, f"{key}.npz")
            if opt.work is None and os.path.exists(f):
                os.remove(f)
    if opt.work is None:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\nPSNR of the gsplat render against all test images (black background); thresholds from {MEASURED_ON}")
    print(f"{'item':<30}{'px':>5}{'imgs':>5}{'min':>8}{'mean':>8}{'max':>8}{'thr':>8}  result")
    for r in results:
        if r["kind"] == "psnr":
            ps = r["ps"]
            print(f"{r['name']:<30}{r['size']:>5}{len(ps):>5}{ps.min():>8.2f}{ps.mean():>8.2f}{ps.max():>8.2f}{r['thr']:>8.2f}  "
                  + ("PASS" if r["ok"] else "FAIL"))
            if opt.per_image:
                print("    per image:", " ".join(f"{v:.1f}" for v in ps))
    smoke = [r for r in results if r["kind"] == "smoke"]
    if smoke:
        print("\nConverter smoke checks (NO original images in the data folder: images were NOT compared, no PSNR)")
        for r in smoke:
            print(f"{r['name']:<30} {r['n']} Gaussians x {r['k']} keyframes, {r['mb']:.0f} MB, {r['secs']:.0f} s  "
                  + ("PASS (count/size match README)" if r["ok"] else "FAIL: " + "; ".join(r["bad"])))
    for label, why in skipped:
        print(f"SKIPPED {label}: {why}")
    failed = [r["name"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed or (opt.strict and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
