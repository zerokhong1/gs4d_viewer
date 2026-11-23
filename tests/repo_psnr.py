#!/usr/bin/env python3
"""WP31: PSNR of the method's OWN renderer on the images check_real.py uses, next to the PSNR of our converted file.

This is the one script in the repo that executes a method's code (the converters only port formulas). It needs a clone
of the method's repo and is not run by tests/run_all.py.

    python tests/repo_psnr.py som        [--repo /home/cont/gs4d_train/som] [--site /home/cont/gs4d_train/site_som]
    /home/cont/gs4d_train/env/bin/python tests/repo_psnr.py spacetime   [--repo /home/cont/gs4d_train/stg]   (training env, CUDA toolkit libs)
    python tests/repo_psnr.py spacetime_ours [--frames 50]              (this repo's .venv; compares with the npy of the line above)

som:        vye16/shape-of-motion `flow3d.scene_model.SceneModel.render` (gsplat) on the `paper-windmill` checkpoint, the
            training cameras stored in the checkpoint, every 4th DyCheck 1x image (item `som_psnr` of check_real.py).
            Needs `roma` (pip install --no-deps --target <site> roma), torch and gsplat from this repo's .venv.
spacetime:  oppo-us-research/SpacetimeGaussians `test_ours_lite` with its `forward_lite` CUDA rasteriser on the lite
            `n3d_sear_steak` model, 21 cameras x 5 frames of the Neural 3D Video videos (item `spacetime_psnr`).
            Needs forward_lite built into --site (pip install --no-deps --target <site> <repo>/thirdparty/gaussian_splatting/
            submodules/forward_lite, after adding `#include <cstdint>` to cuda_rasterizer/rasterizer_impl.h) and an empty
            `mmcv/ops.py` stub with a `knn` name there. Writes the per-image PSNR to --out (npy).
"""
import argparse
import math
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))


def report(label, repo, ours):
    d = ours - repo
    print(f"{label}: {len(repo)} images")
    print(f"  repo's renderer : {repo.min():.2f}-{repo.max():.2f} dB, mean {repo.mean():.2f}")
    print(f"  our file, gsplat: {ours.min():.2f}-{ours.max():.2f} dB, mean {ours.mean():.2f}")
    print(f"  ours - repo     : mean {d.mean():+.3f} dB, per image {d.min():+.3f} .. {d.max():+.3f}")


def run_som(opt):
    import tempfile
    import torch
    import check_real as CR
    for p in (opt.repo, opt.site):
        sys.path.insert(0, p)
    from flow3d.scene_model import SceneModel
    from convert_shape_of_motion import load_checkpoint

    data = CR.default_data_root()
    ckpt = os.path.join(data, "som", "paper-windmill", "checkpoints", "last.ckpt")
    scene = os.path.join(data, "som", "paper-windmill_data", "paper-windmill")
    step = 4
    frames, K, W, H = CR.load_som_test(ckpt, scene, step=step)
    sd = load_checkpoint(ckpt)
    sd = sd.get("model", sd)
    sd = {k: torch.as_tensor(np.asarray(v)) if not torch.is_tensor(v) else v for k, v in sd.items()}  # our loader returns numpy
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    model = SceneModel.init_from_state_dict(sd).cuda().eval()
    Kt = torch.from_numpy(K).cuda()[None]
    repo = {}
    for bg in (0.0, 1.0):  # our check composites over black; the repo's default background is white
        ps = []
        with torch.no_grad():
            for n, (_t, w2c, gt) in enumerate(frames):
                img = model.render(n * step, torch.from_numpy(w2c).cuda()[None], Kt, (W, H), bg_color=bg)["img"][0]
                ps.append(psnr(img.clamp(0, 1).cpu().numpy(), gt))
        repo[bg] = np.array(ps)
    print(f"repo render on black vs white background: max |diff| {np.abs(repo[0.0] - repo[1.0]).max():.4f} dB")
    work = tempfile.mkdtemp(prefix="repo_psnr_")
    for stride in (1, 8):
        out = os.path.join(work, f"som_{stride}.npz")
        CR.convert("convert_shape_of_motion.py", [ckpt, "--stride", str(stride), "--fps", "30"], out)
        ours, _ = CR.psnr_item("som_psnr", out, (frames, K, W, H))
        if stride == 1:
            report("Shape of Motion paper-windmill, our file with --stride 1 (every frame is a keyframe)", repo[0.0], ours)
        else:
            report("  --stride 8, images at keyframes", repo[0.0][0::2], ours[0::2])
            report("  --stride 8, images half way between keyframes", repo[0.0][1::2], ours[1::2])
        os.remove(out)


def run_spacetime(opt):
    """The repo's lite renderer on the frames of check_real.load_n3d_frame (same cameras, same images, same order)."""
    import glob
    import torch
    for p in (opt.repo, os.path.join(opt.repo, "thirdparty", "gaussian_splatting"), opt.site):
        sys.path.insert(0, p)
    import types
    gs = os.path.join(opt.repo, "thirdparty", "gaussian_splatting")
    for name in ("scene", "thirdparty.gaussian_splatting.scene"):   # skip scene/__init__.py: it pulls in the dataset readers
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(gs, "scene")]
        sys.modules[name] = mod
    from thirdparty.gaussian_splatting.scene.ourslite import GaussianModel
    from thirdparty.gaussian_splatting.renderer import test_ours_lite
    from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix
    from forward_lite import GaussianRasterizationSettings, GaussianRasterizer
    import check_real as CR

    data = CR.default_data_root()
    model_dir = os.path.join(data, "spacetime", "n3d_sear_steak_lite_allcam")
    video = os.path.join(data, "spacetime", "n3d_sear_steak_video", "sear_steak")
    ply = sorted(glob.glob(os.path.join(model_dir, "**", "point_cloud.ply"), recursive=True))[-1]
    pc = GaussianModel(3, None)
    pc.load_ply(ply)
    bg = torch.zeros(3, device="cuda")
    ps = []
    for frame in CR.N3D_FRAMES:
        frames, K, W, H = CR.load_n3d_frame(model_dir, video, frame)
        fovx, fovy = 2 * math.atan(W / (2 * K[0, 0])), 2 * math.atan(H / (2 * K[1, 1]))
        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()   # as scene/cameras.py
        for t, w2c, gt in frames:
            view = torch.tensor(np.asarray(w2c, np.float32)).transpose(0, 1).cuda()
            cam = argparse.Namespace(FoVx=fovx, FoVy=fovy, image_width=W, image_height=H, world_view_transform=view,
                                     full_proj_transform=view @ proj, camera_center=view.inverse()[3, :3], timestamp=float(t))
            with torch.no_grad():
                img = test_ours_lite(cam, pc, None, bg, GRsetting=GaussianRasterizationSettings, GRzer=GaussianRasterizer)["render"]
            ps.append(psnr(img.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), gt))
    ps = np.array(ps)
    np.save(opt.out, ps)
    print(f"repo's renderer ({ply}): {len(ps)} images at {W}x{H}: {ps.min():.2f}-{ps.max():.2f} dB, mean {ps.mean():.2f}; saved {opt.out}")


def run_spacetime_ours(opt):
    """Our converted file on the same frames (this repo's .venv), compared with the npy written by `spacetime`."""
    import glob
    import tempfile
    import check_real as CR
    data = CR.default_data_root()
    model_dir = os.path.join(data, "spacetime", "n3d_sear_steak_lite_allcam")
    video = os.path.join(data, "spacetime", "n3d_sear_steak_video", "sear_steak")
    ply = sorted(glob.glob(os.path.join(model_dir, "**", "point_cloud.ply"), recursive=True))[-1]
    out = os.path.join(tempfile.mkdtemp(prefix="repo_psnr_"), "st.npz")
    CR.convert("convert_spacetime.py", [ply, "--frames", str(opt.frames), "--t0", "0", "--t1", "1"], out)
    ours = np.concatenate([CR.psnr_item("spacetime_psnr", out, CR.load_n3d_frame(model_dir, video, fr))[0] for fr in CR.N3D_FRAMES])
    os.remove(out)
    np.save(opt.out.replace(".npy", f"_ours{opt.frames}.npy"), ours)
    report(f"Spacetime Gaussians lite n3d_sear_steak, our file with {opt.frames} keyframes", np.load(opt.out), ours)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=("som", "spacetime", "spacetime_ours"))
    ap.add_argument("--repo", default=None)
    ap.add_argument("--site", default=None, help="folder with the extra packages (roma / forward_lite + mmcv stub)")
    ap.add_argument("--frames", type=int, default=50, help="spacetime_ours: keyframes of our file")
    ap.add_argument("--out", default="/home/cont/gs4d_train/spacetime_repo_psnr.npy")
    opt = ap.parse_args()
    opt.repo = opt.repo or ("/home/cont/gs4d_train/som" if opt.method == "som" else "/home/cont/gs4d_train/stg")
    opt.site = opt.site or ("/home/cont/gs4d_train/site_som" if opt.method == "som" else "/home/cont/gs4d_train/site_stg")
    {"som": run_som, "spacetime": run_spacetime, "spacetime_ours": run_spacetime_ours}[opt.method](opt)
