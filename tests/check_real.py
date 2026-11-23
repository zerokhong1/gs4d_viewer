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
the Gaussian count, keyframe count and file size against docs/SUPPORTED.md.
  Spacetime Gaussians (lite model), Shape of Motion (paper-windmill).

An item whose data or CUDA is missing is SKIPPED with the reason; it is never reported as passed.
Exit code: 1 if any item failed (or, with --strict, was skipped), else 0.

    python tests/check_real.py [--data DIR] [--train DIR] [--work DIR] [--only NAME[,NAME]] [--strict]

--data  : folder with dnerf/, spacetime/, som/   (env GS4D_DATA, default: `data_real` next to the main checkout)
--train : folder with out/ and 4dgs/             (env GS4D_TRAIN, default /home/cont/gs4d_train)
--work  : where the converted .npz files go (default: a temporary folder, deleted at the end)
Model paths can also be set one by one with env GS4D_D3DGS, GS4D_SCGS, GS4D_NATIVE4D, GS4D_SPACETIME, GS4D_SOM.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "converters"))

# Minimum PSNR (dB) per method = the lowest value measured over all 20 test images on 14/11/2026 (RTX 3060),
# minus 0.5 dB. See the table printed by this script and docs/SUPPORTED.md.
MEASURED_ON = "14/11/2026"
MIN_PSNR = {"deformable3dgs": 36.70, "scgs": 34.85, "native4d": 26.78, "4dgaussians": 36.17, "d3dgs_default": 34.85, "d3dgs_6dof": 36.49, "scgs_mask": 34.98, "scgs_predcolor": 26.81, "scgs_skin": 0.0, "scgs_hash": 0.0, "d3dgs_banana": 18.78, "scgs_banana": 19.71, "native4d_banana": 20.91, "4dgaussians_banana": 21.55, "spacetime_psnr": 27.26, "som_psnr": 21.11}  # measured minima 37.20, 35.35, 27.28 (native4d with --sh; 26.04 without), 36.67 (4DGaussians --sh, white background), 35.35 (d3dgs_default, mean 39.62), 36.99 (d3dgs_6dof, mean 40.34), 35.48 (scgs_mask, mean 40.14), minus 0.5
EVAL_SIZE = {"deformable3dgs": 800, "scgs": 400, "native4d": 400, "4dgaussians": 800, "d3dgs_default": 800, "d3dgs_6dof": 800, "scgs_mask": 400, "scgs_predcolor": 400, "scgs_skin": 400, "scgs_hash": 400, "d3dgs_banana": 268, "scgs_banana": 268, "native4d_banana": 268, "4dgaussians_banana": 268, "spacetime_psnr": 1352}  # training resolution of each model

SIZE_TOL = 0.05  # file size must be within 5 % of the figure in docs/SUPPORTED.md


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
    r = subprocess.run([sys.executable, os.path.join(ROOT, "converters", script)] + args + ["-o", out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{r.stdout[-800:]}\n{r.stderr[-1500:]}")
    return time.time() - t


# ------------------------------------------------------------------------------------------ PSNR items
WHITE_BG = {"4dgaussians", "4dgaussians_banana"}  # trained with white_background=True: compare on white, not black


def load_test_set(scene, size, white=False):
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
        gt = im[..., :3] * im[..., 3:4] + ((1.0 - im[..., 3:4]) if white else 0.0)  # composite on black (white)
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
            img, alpha, _ = rasterization(
                torch.from_numpy(np.ascontiguousarray(means, np.float32)).to(dev), None, None,
                torch.from_numpy(np.ascontiguousarray(opac, np.float32)).to(dev), colors,
                torch.from_numpy(viewmat).to(dev)[None], Kt, W, H,
                covars=torch.from_numpy(unpack_cov(np.ascontiguousarray(covs, np.float32))).to(dev), **kw)
        pred = (img[0] + ((1.0 - alpha[0]) if name in WHITE_BG else 0.0)).clamp(0, 1).cpu().numpy()
        mse = float(np.mean((pred - gt) ** 2))
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        del img
    return np.array(ps), seq


# ------------------------------------------------------------------------------------------ items
def run_psnr(name, label, script, model, extra, work, data, results, scene="bouncingballs"):
    test = load_test_set(os.path.join(data, "dnerf", "data", scene), EVAL_SIZE[name], name in WHITE_BG)
    need(model, f"{label} model")
    out = os.path.join(work, f"{name}.npz")
    secs = convert(script, [model] + extra, out)
    ps, seq = psnr_item(name, out, test)
    thr = MIN_PSNR[name]
    ok = ps.min() >= thr
    results.append(dict(name=label, size=EVAL_SIZE[name], kind="psnr", ok=ok, ps=ps, thr=thr, n=seq.num_gaussians, k=seq.num_frames,
                        mb=os.path.getsize(out) / 1e6, secs=secs))


# ---------------------------------------------------------------- HyperNeRF scene (slice-banana, real, monocular video)
def banana_scene(train):
    """Folder made by tools/make_hypernerf_split.py: vrig_slice-banana/slice-banana (4x images = 268 x 480 px, portrait)."""
    return os.environ.get("GS4D_BANANA") or os.path.join(train, "data", "vrig_slice-banana", "slice-banana")


def load_nerfies_val_set(scene, raw=False):
    """Held-out frames (dataset.json val_ids) of a HyperNeRF scene at 4x. Same camera maths as the repos'
    readNerfiesCameras: position = (position - center) * scale, w2c rotation = orientation, f = focal / 4, principal
    point at the image centre, no lens distortion; time = time_id / max(time_id) (in [0, 1]).
    raw=True: hustvl 4DGaussians (scene/hyper_loader.py) keeps the camera positions as stored, without centre and scale,
    and uses warp_id / max(warp_id) as time."""
    from PIL import Image
    need(os.path.join(scene, "dataset.json"), "HyperNeRF dataset.json")
    sc = json.load(open(os.path.join(scene, "scene.json")))
    meta = json.load(open(os.path.join(scene, "metadata.json")))
    ids = json.load(open(os.path.join(scene, "dataset.json")))["val_ids"]
    tkey = "warp_id" if raw else "time_id"
    tmax = max(m[tkey] for m in meta.values())
    out = []
    for i in ids:
        cam = json.load(open(os.path.join(scene, "camera", f"{i}.json")))
        pos = np.array(cam["position"]) if raw else (np.array(cam["position"]) - np.array(sc["center"])) * sc["scale"]
        R = np.array(cam["orientation"])
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = R, -R @ pos
        gt = np.asarray(Image.open(need(os.path.join(scene, "rgb", "4x", f"{i}.png"), "4x image")).convert("RGB"), np.float32) / 255.0
        H, W = gt.shape[:2]
        f = cam["focal_length"] / 4
        out.append((meta[i][tkey] / tmax, vm.astype(np.float32), gt, f))
    H, W = out[0][2].shape[:2]
    f = out[0][3]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    return [o[:3] for o in out], K, W, H


def run_banana(name, label, script, model, extra, work, train, results, thr_key=None, raw=False):
    test = load_nerfies_val_set(need(banana_scene(train), "HyperNeRF slice-banana split (run tools/make_hypernerf_split.py)"), raw)
    need(model, f"{label} model")
    out = os.path.join(work, f"{name}.npz")
    secs = convert(script, [model] + extra, out)
    ps, seq = psnr_item(name, out, test)
    thr = MIN_PSNR[name]
    ok = thr is None or ps.min() >= thr
    results.append(dict(name=label, size=EVAL_SIZE[name], kind="psnr", ok=ok, ps=ps, thr=thr if thr is not None else float("nan"),
                        n=seq.num_gaussians, k=seq.num_frames, mb=os.path.getsize(out) / 1e6, secs=secs))


def run_vs_export(label, model, data, results):
    """convert_4dgaussians.py against the per-frame ply written by the repo's export_perframe_3DGS.py
    (<model>/gaussian_pertimestamp, one ply per test camera). The export is run once by hand, see docs/SUPPORTED.md (4DGaussians)."""
    import torch
    sys.path.insert(0, ROOT)
    import convert_4dgaussians as c4
    from gs4d.ply import gaussians_from_ply, read_ply_vertices
    ref = os.path.join(model, "gaussian_pertimestamp")
    need(ref, "gaussian_pertimestamp (run export_perframe_3DGS.py once)")
    meta = json.load(open(need(os.path.join(data, "dnerf", "data", "bouncingballs", "transforms_test.json"), "transforms_test.json")))
    times = [float(f["time"]) for f in meta["frames"]]
    ply, pth, cfg = c4.find_files(model, None, -1)
    net = c4.HexPlaneDeform(torch.load(pth, map_location="cpu", weights_only=True), c4.read_flags(cfg))
    v = read_ply_vertices(ply)
    g = lambda names: np.stack([v[k] for k in names], -1).astype(np.float32)  # noqa: E731
    t0 = time.time()
    means, covs, opac = c4.deform_sequence(g(["x", "y", "z"]), g([f"scale_{i}" for i in range(3)]),
                                           g([f"rot_{i}" for i in range(4)]), v["opacity"].astype(np.float32)[:, None], net, times)
    secs = time.time() - t0
    em = ec = eo = 0.0
    for k in range(len(times)):
        m, cv, op, _ = gaussians_from_ply(os.path.join(ref, f"time_{k:05d}.ply"))
        em = max(em, float(np.abs(means[k] - m).max()))
        ec = max(ec, float((np.abs(covs[k] - cv) / (np.abs(cv).max(1, keepdims=True) + 1e-12)).max()))
        eo = max(eo, float(np.abs((opac[k] if opac.ndim == 2 else opac) - op).max()))
    bad = [f"{n} {e:.1e} > {t:.0e}" for n, e, t in (("mean", em, 1e-4), ("cov(rel)", ec, 1e-4), ("opacity", eo, 1e-5)) if e > t]
    results.append(dict(name=label, kind="export", ok=not bad, bad=bad, n=len(v), k=len(times), secs=secs,
                        note=f"max error vs export: mean {em:.1e}, cov(rel) {ec:.1e}, opacity {eo:.1e}"))


# ---------------------------------------------------------------- Spacetime Gaussians, Neural 3D Video (WP19c part 2)
N3D_FRAMES = (0, 12, 25, 37, 49)   # of the 50 frames the model was trained on


def load_n3d_frame(model_dir, video_dir, frame, duration=50, down=2):
    """Frame `frame` of every camNN.mp4 in video_dir with the camera of the same name from the model's cameras.json
    (3DGS layout: position = camera centre, rotation = camera-to-world, fx at full width). Images are resized by 1/down
    (the model was trained with resolution=2); time = frame / duration as in the repo's loader."""
    import imageio.v2 as imageio
    from PIL import Image
    cams = {}
    for c in json.load(open(need(os.path.join(model_dir, "cameras.json"), "Spacetime cameras.json"))):
        cams.setdefault(c["img_name"], c)   # one entry per camera and frame; the pose does not change
    out, K, size = [], None, None
    for name in sorted(cams):
        mp4 = os.path.join(video_dir, name + ".mp4")
        if not os.path.exists(mp4):
            continue
        c = cams[name]
        with imageio.get_reader(mp4) as rd:
            pil = Image.fromarray(rd.get_data(frame))
        assert pil.size == (c["width"], c["height"]), (name, pil.size)
        W, H = c["width"] // down, c["height"] // down
        gt = np.asarray(pil.resize((W, H), Image.LANCZOS), np.float32) / 255.0
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = np.array(c["rotation"]), np.array(c["position"])
        f = c["fx"] / down
        k = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
        assert K is None or (np.allclose(K, k) and size == (W, H)), "cameras differ in size or focal length"
        K, size = k, (W, H)
        out.append((frame / duration, np.linalg.inv(c2w).astype(np.float32), gt))
    if not out:
        raise Skip(f"no camNN.mp4 in {video_dir}")
    return out, K, size[0], size[1]


def run_spacetime_psnr(name, label, ply, video_dir, work, results):
    need(video_dir, "Neural 3D Video sear_steak videos (cam00.mp4 ...)")
    out = os.path.join(work, f"{name}.npz")
    secs = convert("convert_spacetime.py", [ply, "--frames", "50", "--t0", "0", "--t1", "1"], out)
    ps, seq = [], None
    for fr in N3D_FRAMES:   # one frame index at a time: 21 images of 1352 x 1014 are 350 MB
        p, seq = psnr_item(name, out, load_n3d_frame(os.path.dirname(os.path.dirname(os.path.dirname(ply))), video_dir, fr))
        ps.append(p)
    ps = np.concatenate(ps)
    thr = MIN_PSNR[name]
    results.append(dict(name=label, size=1352, kind="psnr", ok=ps.min() >= thr, ps=ps, thr=thr, n=seq.num_gaussians,
                        k=seq.num_frames, mb=os.path.getsize(out) / 1e6, secs=secs))


# ---------------------------------------------------------------- Shape of Motion, DyCheck iPhone scene (WP19c part 1)
def load_som_test(ckpt, scene, step=4, fps=30.0):
    """Every step-th training frame of a DyCheck iPhone scene (rgb/1x/<splits/train.json frame_names[i]>.png) with the
    camera of that frame from the checkpoint (`Ks`, `w2cs`, OpenCV); time = frame / fps as in the converter."""
    from PIL import Image
    sys.path.insert(0, ROOT)
    from convert_shape_of_motion import load_checkpoint
    sd = load_checkpoint(ckpt)
    sd = sd.get("model", sd)
    Ks, w2cs = np.asarray(sd["Ks"], np.float32), np.asarray(sd["w2cs"], np.float32)
    names = json.load(open(need(os.path.join(scene, "splits", "train.json"), "DyCheck splits/train.json")))["frame_names"]
    assert len(names) == len(Ks), (len(names), len(Ks))
    assert np.abs(Ks - Ks[0]).max() < 1e-3, "intrinsics change between frames"
    out = []
    for i in range(0, len(names), step):
        pil = Image.open(need(os.path.join(scene, "rgb", "1x", names[i] + ".png"), "DyCheck 1x image")).convert("RGB")
        out.append((i / fps, w2cs[i], np.asarray(pil, np.float32) / 255.0))
    H, W = out[0][2].shape[:2]
    return out, Ks[0], W, H


def run_som_psnr(name, label, ckpt, scene, work, results, stride=8):
    need(ckpt, f"{label} checkpoint")
    need(scene, "DyCheck iPhone paper-windmill (rgb/1x, splits)")
    test = load_som_test(ckpt, scene, step=stride // 2)   # keyframes and the frames half way between them
    out = os.path.join(work, f"{name}.npz")
    secs = convert("convert_shape_of_motion.py", [ckpt, "--stride", str(stride), "--fps", "30"], out)
    ps, seq = psnr_item(name, out, test)
    thr = MIN_PSNR[name]
    at_key, between = ps[0::2], ps[1::2]
    results.append(dict(name=label, size=test[2], kind="psnr", ok=ps.min() >= thr, ps=ps, thr=thr, n=seq.num_gaussians,
                        k=seq.num_frames, mb=os.path.getsize(out) / 1e6, secs=secs,
                        note=f"at keyframes mean {at_key.mean():.2f} (min {at_key.min():.2f}), half way between keyframes mean "
                             f"{between.mean():.2f} (min {between.min():.2f})"))


def run_smoke(name, label, script, model, extra, work, expect, results):
    need(model, f"{label} model")
    out = os.path.join(work, f"{name}.npz")
    secs = convert(script, [model] + extra, out)
    from gs4d import Sequence4D
    seq = Sequence4D.load(out)
    mb = os.path.getsize(out) / 1e6
    bad = []
    if seq.num_gaussians != expect["n"]:
        bad.append(f"Gaussians {seq.num_gaussians} != docs {expect['n']}")
    if seq.num_frames != expect["k"]:
        bad.append(f"keyframes {seq.num_frames} != docs {expect['k']}")
    if abs(mb - expect["mb"]) > SIZE_TOL * expect["mb"]:
        bad.append(f"size {mb:.0f} MB differs more than {SIZE_TOL:.0%} from docs {expect['mb']} MB")
    results.append(dict(name=label, kind="smoke", ok=not bad, bad=bad, n=seq.num_gaussians, k=seq.num_frames, mb=mb, secs=secs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("GS4D_DATA") or default_data_root())
    ap.add_argument("--train", default=os.environ.get("GS4D_TRAIN") or "/home/cont/gs4d_train")
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default="", help="comma list of: native4d_banana,d3dgs_banana,scgs_banana,4dgaussians_banana,deformable3dgs,d3dgs_default,d3dgs_6dof,scgs,scgs_mask,scgs_predcolor,scgs_skin,scgs_hash,native4d,4dgaussians,4dgaussians_export,spacetime,spacetime_psnr,spacetime_tb,som,som_psnr")
    ap.add_argument("--per-image", action="store_true", help="also print the PSNR of every test image")
    ap.add_argument("--strict", action="store_true", help="a skipped item also gives exit code 1")
    opt = ap.parse_args()

    E = os.environ.get
    paths = dict(
        deformable3dgs=E("GS4D_D3DGS") or os.path.join(opt.train, "out", "d3dgs_bouncingballs"),
        d3dgs_default=E("GS4D_D3DGS_DEFAULT") or os.path.join(opt.train, "out", "d3dgs_default"),
        d3dgs_6dof=E("GS4D_D3DGS_6DOF") or os.path.join(opt.train, "out", "d3dgs_6dof"),
        d3dgs_banana=E("GS4D_D3DGS_BANANA") or os.path.join(opt.train, "out", "d3dgs_banana"),
        scgs_banana=E("GS4D_SCGS_BANANA") or os.path.join(opt.train, "out", "scgs_banana_node"),
        native4d_banana=E("GS4D_NATIVE4D_BANANA") or os.path.join(opt.train, "4dgs", "output", "banana_fov", "chkpnt_best.pth"),
        scgs_mask=E("GS4D_SCGS_MASK") or os.path.join(opt.train, "out", "scgs_mask_bouncingballs_node"),
        scgs_predcolor=E("GS4D_SCGS_PREDCOLOR") or os.path.join(opt.train, "out", "scgs_tint2_bouncingballs_node"),
        scgs_skin=E("GS4D_SCGS_SKIN") or os.path.join(opt.train, "out", "scgs_skin_bouncingballs_node"),
        scgs_hash=E("GS4D_SCGS_HASH") or os.path.join(opt.train, "out", "scgs_hash2_bouncingballs_node"),
        scgs=E("GS4D_SCGS") or os.path.join(opt.train, "out", "scgs_bouncingballs_node"),
        fourdg_banana=E("GS4D_4DGAUSSIANS_BANANA") or os.path.join(opt.train, "4dgaussians", "output", "hypernerf", "banana"),
        fourdg=E("GS4D_4DGAUSSIANS") or os.path.join(opt.train, "4dgaussians", "output", "dnerf", "bouncingballs"),
        native4d=E("GS4D_NATIVE4D") or os.path.join(opt.train, "4dgs", "output", "dnerf", "bouncingballs", "chkpnt_best.pth"),
        spacetime=E("GS4D_SPACETIME") or os.path.join(opt.data, "spacetime", "n3d_sear_steak_lite_allcam", "point_cloud"),
        spacetime_tb=E("GS4D_SPACETIME_TB") or os.path.join(opt.data, "spacetime", "techni_Birthday_dl", "techni_Birthday_allcam_allpoints",
                                                             "point_cloud", "iteration_20000", "point_cloud.ply"),
        spacetime_video=E("GS4D_SPACETIME_VIDEO") or os.path.join(opt.data, "spacetime", "n3d_sear_steak_video", "sear_steak"),
        som_scene=E("GS4D_SOM_SCENE") or os.path.join(opt.data, "som", "paper-windmill_data", "paper-windmill"),
        som=E("GS4D_SOM") or os.path.join(opt.data, "som", "paper-windmill", "checkpoints", "last.ckpt"),
    )
    # figures of docs/SUPPORTED.md (measured 14/11/2026)
    spacetime_expect = dict(n=98993, k=50, mb=199)
    spacetime_tb_expect = dict(n=596601, k=50, mb=583)   # --compact; 658,999 in the ply, 62,398 dropped by --min-peak-opacity
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
        ("d3dgs_default", "Deformable-3DGS default --sh", True,
         lambda: run_psnr("d3dgs_default", "Deformable-3DGS default --sh", "convert_deformable3dgs.py", paths["d3dgs_default"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("d3dgs_6dof", "Deformable-3DGS 6-DoF --sh", True,
         lambda: run_psnr("d3dgs_6dof", "Deformable-3DGS 6-DoF --sh", "convert_deformable3dgs.py", paths["d3dgs_6dof"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("d3dgs_banana", "Deformable-3DGS HyperNeRF banana", True,
         lambda: run_banana("d3dgs_banana", "Deformable-3DGS HyperNeRF banana", "convert_deformable3dgs.py", paths["d3dgs_banana"],
                            ["--frames", "330", "--sh", "--compact"], work, opt.train, results)),
        ("scgs_banana", "SC-GS HyperNeRF banana", True,
         lambda: run_banana("scgs_banana", "SC-GS HyperNeRF banana", "convert_scgs.py", paths["scgs_banana"],
                            ["--frames", "110", "--sh", "--compact"], work, opt.train, results)),
        ("native4d_banana", "native 4DGS HyperNeRF banana", True,
         lambda: run_banana("native4d_banana", "native 4DGS HyperNeRF banana", "convert_native4d.py", paths["native4d_banana"],
                            ["--frames", "110", "--t0", "0", "--t1", "1", "--sh", "--compact"], work, opt.train, results)),
        ("4dgaussians_banana", "4DGaussians HyperNeRF banana", True,
         lambda: run_banana("4dgaussians_banana", "4DGaussians HyperNeRF banana", "convert_4dgaussians.py", paths["fourdg_banana"],
                            ["--frames", "330", "--sh", "--compact"], work, opt.train, results, raw=True)),
        ("scgs", "SC-GS --sh", True,
         lambda: run_psnr("scgs", "SC-GS --sh", "convert_scgs.py", paths["scgs"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("scgs_mask", "SC-GS motion mask --sh", True,
         lambda: run_psnr("scgs_mask", "SC-GS motion mask --sh", "convert_scgs.py", paths["scgs_mask"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("scgs_predcolor", "SC-GS --pred_color --sh (time-tinted scene)", True,
         lambda: run_psnr("scgs_predcolor", "SC-GS --pred_color --sh (time-tinted scene)", "convert_scgs.py", paths["scgs_predcolor"],
                          ["--frames", "60", "--sh"], work, opt.data, results, scene="bouncingballs_tint")),
        ("scgs_skin", "SC-GS --skinning --sh", True,
         lambda: run_psnr("scgs_skin", "SC-GS --skinning --sh", "convert_scgs.py", paths["scgs_skin"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("scgs_hash", "SC-GS --use_hash --sh", True,
         lambda: run_psnr("scgs_hash", "SC-GS --use_hash --sh", "convert_scgs.py", paths["scgs_hash"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("native4d", "native 4DGS --sh", True,
         lambda: run_psnr("native4d", "native 4DGS --sh", "convert_native4d.py", paths["native4d"],
                          ["--frames", "60", "--t0", "0", "--t1", "1", "--sh"], work, opt.data, results)),
        ("4dgaussians", "4DGaussians --sh", True,
         lambda: run_psnr("4dgaussians", "4DGaussians --sh", "convert_4dgaussians.py", paths["fourdg"],
                          ["--frames", "60", "--sh"], work, opt.data, results)),
        ("4dgaussians_export", "4DGaussians vs export_perframe_3DGS", False,
         lambda: run_vs_export("4DGaussians vs export_perframe_3DGS", paths["fourdg"], opt.data, results)),
        ("spacetime", "Spacetime Gaussians (smoke)", False,
         lambda: run_smoke("spacetime", "Spacetime Gaussians (smoke)", "convert_spacetime.py", spacetime_ply(),
                           ["--frames", "50", "--t0", "0", "--t1", "1"], work, spacetime_expect, results)),
        ("spacetime_psnr", "Spacetime lite sear_steak", True,
         lambda: run_spacetime_psnr("spacetime_psnr", "Spacetime lite sear_steak", spacetime_ply(), paths["spacetime_video"], work, results)),
        ("spacetime_tb", "Spacetime Technicolor Birthday lite (smoke)", False,
         lambda: run_smoke("spacetime_tb", "Spacetime Technicolor Birthday lite (smoke)", "convert_spacetime.py", paths["spacetime_tb"],
                           ["--frames", "50", "--t0", "0", "--t1", "1", "--compact"], work, spacetime_tb_expect, results)),
        ("som", "Shape of Motion (smoke)", False,
         lambda: run_smoke("som", "Shape of Motion (smoke)", "convert_shape_of_motion.py", paths["som"],
                           ["--stride", "8", "--fps", "30"], work, som_expect, results)),
        ("som_psnr", "Shape of Motion paper-windmill", True,
         lambda: run_som_psnr("som_psnr", "Shape of Motion paper-windmill", paths["som"], paths["som_scene"], work, results)),
    ]
    for key, label, needs_cuda, fn in items:
        if only and key not in only:
            continue
        if key in ("scgs_hash", "scgs_skin") and key not in only:
            continue   # opt-in: the repo's own --use_hash training did not converge (best 6.8 dB) and --skinning training crashes after densification: no usable model
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
            if r.get("note"):
                print("    " + r["note"])
            if opt.per_image:
                print("    per image:", " ".join(f"{v:.1f}" for v in ps))
    smoke = [r for r in results if r["kind"] == "smoke"]
    if smoke:
        print("\nConverter smoke checks (NO original images in the data folder: images were NOT compared, no PSNR)")
        for r in smoke:
            print(f"{r['name']:<30} {r['n']} Gaussians x {r['k']} keyframes, {r['mb']:.0f} MB, {r['secs']:.0f} s  "
                  + ("PASS (count/size match docs/SUPPORTED.md)" if r["ok"] else "FAIL: " + "; ".join(r["bad"])))
    for r in [r for r in results if r["kind"] == "export"]:
        print(f"\n{r['name']}: {r['n']} Gaussians x {r['k']} test times, {r['note']}  "
              + ("PASS" if r["ok"] else "FAIL: " + "; ".join(r["bad"])))
    for label, why in skipped:
        print(f"SKIPPED {label}: {why}")
    failed = [r["name"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed or (opt.strict and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
