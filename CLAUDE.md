# gs4d_viewer — context for Claude Code sessions

Read this instead of re-deriving it. Work packages and ownership rules are in `PLAN.md`.

## What this is
Browser viewer (viser) for 4D Gaussian Splatting results with a time slider. The viewer reads ONE
common `.npz` ("Gaussians per keyframe", `gs4d/format.py`, class `Sequence4D`). Every 4DGS method gets
its own `convert_<method>.py` that writes that format. The viewer never imports a method's code.

## Layout
- `gs4d/format.py`  container, interpolation (`Sequence4D.at(t)`), `cov_from_quat_scale`, `pack_cov/unpack_cov`;
  compact storage and the static group (Gaussians that never change are stored once, `save(split_static=True)`)
- `gs4d/ply.py`     standard 3DGS ply reader (exp on scale, sigmoid on opacity, SH0 colour)
- `viewer.py`       GUI + two renderers: `client` (viser WebGL splats) and `gsplat` (CUDA, image streamed);
  `--compare b.npz` shows a second sequence (A, B, or side by side) on the same time slider
- `convert_per_frame.py`  type A: folder of per-frame ply, Dynamic 3D Gaussians `params.npz`
- `convert_native4d.py`   type B: Yang et al. `chkpnt*.pth` (`--sh`: view- and time-dependent SH per keyframe)
- `convert_spacetime.py`  type C: Spacetime Gaussians `point_cloud.ply` (base colour only)
- `convert_shape_of_motion.py`  type C: Shape of Motion `checkpoints/last.ckpt` (SE(3) motion bases)
- `convert_deformable3dgs.py`  type D: Deformable-3DGS model folder (deformation MLP re-run on CPU)
- `convert_scgs.py`       type D: SC-GS model folder (control nodes + MLP + K-nearest-node blending)
- `convert_4dgaussians.py`  type D: hustvl 4DGaussians model folder (HexPlane + MLP re-run on CPU); checked on a self-trained `bouncingballs`
- `gs4d/__main__.py`  `python -m gs4d in.npz out.npz [--float32] [--no-split]` re-saves compact / plain
- `gs4d/open.py`     `python -m gs4d.open <output> [viewer args]` recognises the output, converts it next to it
  (`<name>.gs4d.npz`, reused while the output is unchanged) and opens the viewer
- `tests/run_all.py` runs every `test_*.py`/`check_*.py` one at a time and prints pass / skipped (reason) / failed; exit 1 on failure
- `make_toy.py`     synthetic scene (`--sh` adds fake SH); `tests/test_all.py` self-checks (CPU only)
- `tests/`          `test_sh.py`, `test_compact.py`, `test_static.py`, `test_spacetime.py`, `test_shape_of_motion.py`,
  `test_deformable3dgs.py`, `test_scgs.py` (the last three need torch), `check_camera.py`, `check_gsplat.py`
  (needs CUDA), `check_compare.py` (needs Chrome), `test_open.py`,
  `check_real.py` (needs CUDA + `../data_real` + `/home/cont/gs4d_train`: PSNR on real outputs with thresholds), `bench_large.py`; screenshots and the demo mp4 are in `shots/`
- Environment: `.venv` (Python 3.10, torch 2.4.1+cu124, gsplat 1.5.3); system `python3` has no torch.
  Install commands are in the README.

## Verified facts (do not re-research)
- Deformable-3DGS and SC-GS add the MLP's `d_scaling` to `exp(scale)` (after the activation). Deformable-3DGS adds
  `d_rotation` to the normalised quaternion and the rasteriser does not normalise again; SC-GS normalises
  `rot + d_rotation`. SC-GS ply has extra `fea_*` columns (hyper coordinates, then the motion mask if trained with one).
- Shape of Motion's official repo is `vye16/shape-of-motion`; the checkpoint is `{"model": state_dict, ...}` with
  keys `fg.params.*`, `bg.params.*`, `motion_bases.params.rots [B,T,6]` / `transls [B,T,3]`; time is a frame index.
- viser 1.1.1: `scene.add_gaussian_splats(name, centers[N,3], covariances[N,3,3], rgbs[N,3], opacities[N,1])`,
  handle has `set_gaussians(...)`. Cameras are OpenCV convention (+z forward, +y down), `camera.fov` is
  vertical, radians. `client.scene.set_background_image(img, format="jpeg")` streams a rendered frame.
- gsplat 1.5.3: `rasterization(means, quats, scales, opacities, colors, viewmats[C,4,4], Ks[C,3,3], W, H, covars=...)`;
  with `covars` [N,3,3] given, quats/scales are passed as None. Returns `(colors[C,H,W,3], alphas, meta)`.
- Native 4DGS (fudan-zvg/4d-gaussian-splatting): checkpoint = `torch.save((capture(), iteration))`; the 4D
  `capture()` tuple has 19 fields: 1 xyz, 2 features_dc, 4 scaling(log), 5 rotation, 6 opacity(logit),
  13 t, 14 scaling_t(log), 15 rotation_r, 16 rot_4d. Renderer drops Gaussians with marginal_t <= 0.05.
- 4DGaussians (hustvl): `export_perframe_3DGS.py` writes `<model_path>/gaussian_pertimestamp/time_00000.ply`
  in standard 3DGS ply layout -> feed that folder to `convert_per_frame.py`.
- Real Shape of Motion checkpoints pickle numpy scalars next to the weights; `load_checkpoint` allow-lists them
  for `torch.load(weights_only=True)`.
- D-NeRF scenes (Deformable-3DGS `--is_blender`, SC-GS, native 4DGS `configs/dnerf/*`) use time in [0, 1]; cameras
  in `transforms_test.json` are OpenGL (flip c2w columns 1 and 2 for viser/gsplat).
- Dynamic 3D Gaussians: `params.npz` keys `means3D[T,N,3]`, `unnorm_rotations[T,N,4]` (wxyz),
  `rgb_colors[T,N,3]`, `logit_opacities[N,1]`, `log_scales[N,3]`, `seg_colors[N,3]`.

## Status
Updated 14/11/2026. Phase 1 (WP1-WP8) and phase 2 wave 1 (WP10, WP11, WP12) are merged into `main`; see `PLAN.md`.
- WP9 is the product goal: every 4DGS output opens with `python -m gs4d.open <output>`. Coverage table (file type, what is
  missing, which WP) is in `PLAN.md`; open items are WP13, WP15, WP16.
- Tested: client renderer in headless Chrome on the toy scene; `tests/test_all.py` (4 checks);
  `viewer.camera_matrices` (viser camera -> viewmat, K) against the browser render, `tests/check_camera.py`.
- Tested on CUDA (RTX 3060, WP1): the gsplat `rasterization(...)` call and image streaming in `_draw_gsplat`,
  `tests/check_gsplat.py`; FPS table in the README.
- Tested (WP6): optional `sh` field, view-dependent colour in the gsplat renderer, `tests/test_sh.py`.
  The client renderer always draws `colors`; `--sh` in `convert_per_frame.py`, `convert_deformable3dgs.py` and
  `convert_scgs.py` writes `sh`.
- Tested (WP7b): mp4 export from the current camera over the whole time range, both renderers.
- Tested (WP8): compact save (float16 offsets, quantised Cholesky, uint8) and lazy memory-mapped load,
  `tests/test_compact.py`, `tests/bench_large.py`. Do not overwrite a file the viewer has open.
- Tested (WP11, 14/11/2026): static group in the file format, `tests/test_static.py`; `at(t)` still returns all N
  Gaussians in the original order, `viewer.py` unchanged. Real files: Shape of Motion `paper-windmill` 1271 -> 147 MB,
  `at(t)` 34 -> 8.7 ms; Deformable-3DGS `bouncingballs` has no static Gaussians (unchanged). Renders before/after are
  identical at keyframes; between keyframes up to 7 of 702,845 pixels differ on `paper-windmill` because the old path
  blended static Gaussians with themselves (float32 rounding). Files with a static group cannot be read by older code.
- Tested (WP14, 14/11/2026): `gs4d.GpuFrames` (format.py) decodes/interpolates on the GPU for the gsplat renderer; `viewer.frame_gpu`
  vs `frame_cpu` (`--cpu-decode`). Play frame 100 -> 2.2 ms (500k x 100 compact), 214 -> 3.1 ms (`paper-windmill` compact); renders differ
  <= 1/255; `check_real.py` PSNR unchanged (also run with `at` patched to GpuFrames). `tests/bench_large.py [--file f.npz]` prints the table.
  NOT measured: display FPS in a real GPU Chrome (built-in browser has no WebGL, Chrome extension not connected).
- Tested (WP12): `tests/check_real.py`, 5 items pass on this machine. Evaluate a model at the resolution it was
  trained at (Deformable-3DGS 800 px, SC-GS and native 4DGS 400 px); away from it PSNR drops by several dB.
- WP10 (`gs4d/open.py`, `tests/test_open.py`): synthetic recognition test passes; its session reported all 6 real
  outputs open with one command, re-checked here only for the SC-GS folder.
- Tested with real output (WP2): `params.npz` reader on Dynamic 3D Gaussians `basketball`; up axis is `-y`.
- Tested (WP7a): `--compare` in headless Chrome with both renderers, `tests/check_compare.py`. gsplat: split
  screen, same camera; client: B moved sideways in the scene (viser cannot split the viewport).
- Tested with real output (14/11/2026), gsplat render of the converted `.npz` against the method's own images
  (numbers in the README table, sheets in `shots/real/`, top = ours, bottom = ground truth):
  - `convert_spacetime.py`: authors' `n3d_sear_steak_lite_allcam` (lite model: `f_dc` only, no `rgbdecoder`);
    checked by eye at training cameras, image not in the repo (shows a person).
  - `convert_shape_of_motion.py`: authors' `paper-windmill` checkpoint, checked by eye.
  - `convert_deformable3dgs.py --sh`: self-trained D-NeRF `bouncingballs` (20k iterations, trained at 800 px), 37.2-43.6 dB
    on all 20 test frames at 800 px (29.3-33.2 dB when rendered at 400 px: evaluate at the training resolution).
  - `convert_scgs.py --sh`: self-trained `bouncingballs` (512 nodes, `--local_frame`, hyper_dim 8, trained at 400 px), 35.4-43.0 dB
    on all 20 test frames at 400 px.
  - `convert_native4d.py`: self-trained `bouncingballs` `chkpnt_best.pth` (trained at 400 px), 26.0-31.9 dB on 20 frames, mean 29.0; with `--sh` (WP13) 27.3-39.6 dB, mean 32.9 (the repo's own eval:
    33.4 dB); SH degree 0 only, so colours are mottled. The 19-field tuple above matches the real checkpoint.
- Each of those converters also has a synthetic test, `tests/test_<name>.py`.
- Tested with real output (WP15, 14/11/2026): `convert_4dgaussians.py --sh` on self-trained hustvl 4DGaussians `bouncingballs`
  (20k iterations, white background, net_width 64, defor_depth 0): 36.7-45.9 dB, mean 40.8 on 20 test frames at 800 px (white
  composite); matches the repo's `export_perframe_3DGS.py` per-frame ply to 1e-6 (position), 7e-6 (relative covariance).
  `tests/check_real.py` items `4dgaussians` and `4dgaussians_export` (the latter needs `gaussian_pertimestamp/`, run the export once).
  Training needed a patched clone in `/home/cont/gs4d_train/4dgaussians` (`np.byte` -> `np.uint8`), a built
  ingra14m depth-diff-gaussian-rasterization (+ `#include <cstdint>`), and shims for mmcv/open3d in `site_4dgaussians`.
- Tested (WP16, 14/11/2026): `convert_deformable3dgs.py --sh` on the Deformable-3DGS default variant (no `--is_blender`, 20k iterations,
  800 px): 35.35-43.14 dB, mean 39.62 on 20 test frames, no converter change; `tests/check_real.py` item `d3dgs_default`.
- Tested (WP16): Deformable-3DGS `--is_blender --is_6dof` (20k iterations, 800 px): 36.99-43.75 dB, mean 40.34 on 20 test frames, no converter
  change; `tests/check_real.py` item `d3dgs_6dof`.
- Tested (WP16): SC-GS with motion mask (`--gs_with_motion_mask --gt_alpha_mask_as_dynamic_mask`, 512 nodes, hyper_dim 8, 20k iterations, 400 px): 35.48-43.14 dB,
  mean 40.14 on 20 test frames; ply has 9 `fea_*` columns, converter recognises the mask, no change; item `scgs_mask`.
- NOT tested: the full Spacetime Gaussians model (with `rgbdecoder`); and any non-D-NeRF scene for Deformable-3DGS / SC-GS / native 4DGS; the client render of
  `basketball` against the original `visualize.py`; display FPS in a real browser (README numbers are server side).
- Local training setup for those methods (not in the repo): `/home/cont/gs4d_train` (env, CUDA toolkit, clones,
  outputs), data in `../data_real`. Build or train one job at a time: the machine has 15 GB RAM.
- Not done: SC-GS `--use_hash`, `--skinning`, `--pred_color`, `d_rot_as_res` off.
- Tested (WP17): `--compact` in every `convert_*.py` (only `save(compact=True)`); `gs4d.open` adds it when the estimate
  (`estimate_bytes`, rough: ply size / 248 B x frames x 40 B) is over 1 GB, `tests/test_open.py`. Deformable-3DGS `bouncingballs --sh`
  153 -> 84 MB, `at(t)` differs from the plain file by <2e-4. `python tests/run_all.py` all pass on this machine (CUDA, Chrome, data present).
  `requirements.txt` now lists numpy<2, viser, pillow, imageio, imageio-ffmpeg, playwright (torch, gsplat as comments).

## Conventions
- A converter is a new file `convert_<method>.py` + a new `tests/test_<method>.py` + one row in the README table.
  It must not edit `viewer.py`, `gs4d/format.py` or other converters.
- Converters run on CPU with numpy (torch only to unpickle weights or run a small MLP). Read reference
  repos to port formulas; do not execute their code and do not add them as dependencies.
- Every converter test builds synthetic inputs in the method's real file layout and checks the maths
  against an independent formulation, as `tests/test_all.py::test_native4d` does.
- Run before finishing: `python tests/test_all.py` and your own test file.
- Headless screenshot check: start `python viewer.py toy.npz --port 8123`, open with Playwright + Chrome flags
  `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`.
- Finish with a 5-line report: what changed, what was run, what is untested, files touched.
