# gs4d_viewer — context for Claude Code sessions

Read this instead of re-deriving it. Work packages and ownership rules are in `PLAN.md`.

## What this is
Browser viewer (viser) for 4D Gaussian Splatting results with a time slider. The viewer reads ONE
common `.npz` ("Gaussians per keyframe", `gs4d/format.py`, class `Sequence4D`). Every 4DGS method gets
its own `convert_<method>.py` that writes that format. The viewer never imports a method's code.

## Layout
- `gs4d/format.py`  container, interpolation (`Sequence4D.at(t)`), `cov_from_quat_scale`, `pack_cov/unpack_cov`;
  compact storage (also per-keyframe `sh`, WP21) and the static group (Gaussians that never change are stored once, `save(split_static=True)`);
  `Sequence4DWriter` writes the same file one keyframe at a time (used by the SC-GS, native 4DGS and Deformable-3DGS converters)
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
  `test_deformable3dgs.py`, `test_scgs.py` (the last three need torch), `check_camera.py`, `check_cameras.py` (needs CUDA, data, Chrome), `check_gsplat.py`
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
- Tested (WP18, 14/11/2026): `viewer.py --cameras <transforms_*.json | cameras.json>` (`load_cameras`): Cameras folder in the panel,
  selecting one moves the client camera and the slider to the image time; gsplat renderer: "Ground truth beside render" shows
  render | image with PSNR (`Viewer.gt_compare`, ignores the opacity/size sliders, black or white background, capped by Max render
  height). `tests/check_cameras.py`: PSNR matches `check_real.psnr_item` to 0.0002 dB on the 20 test cameras (SC-GS 400 px,
  Deformable-3DGS 800 px); `check_real.py` unchanged (7 passed). Screenshots `shots/cameras/`. `gs4d/open.py` passes `--cameras` since WP19a; NOT done: GT view needs one sequence (not with `--compare`); display FPS in a real browser.
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
    33.4 dB); the colour is SH degree 0 only without `--sh` (then mottled). The 19-field tuple above matches the real checkpoint.
- Tested (WP19a, 15/11/2026): `gs4d/open.py` recognises a hustvl 4DGaussians model folder (`deformation.pth` beside the ply, no `deform/`) and passes `--cameras`
  (a readable `cameras.json` with `c2w`/`transform_matrix`, else `transforms_test.json` of `cfg_args` `source_path`; the 3DGS-style `cameras.json` with `position`/`rotation`
  is not readable by `load_cameras` and is skipped; none with `--compare` or when the user gives `--cameras`). Unsupported variants stop and name the flag that converts anyway:
  SC-GS `--pred_color` (`--ignore-pred-color`), Spacetime full model with `rgbdecoder` (`f_dc_3..` in the ply or an rgbdecoder file beside it; `--base-color-only`).
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
- Tested (WP16, smoke only, no ground truth, no PSNR): `convert_spacetime.py --compact` on a second LITE model, Technicolor `Birthday` (`ours_lite`, 658,999 -> 596,601 Gaussians x 50, 583 MB, 20 s);
  gsplat render at t = 0, 0.5, 1 from `cameras.json` camera 0 looks right by eye (image not in the repo, shows a person); item `spacetime_tb` (skips if data missing).
  Still NOT tested: the full model with `rgbdecoder` (none public; needs training with the original repo on Technicolor / Neural 3D data, tens of GB, beyond 15 GB RAM).
- Tested (WP16, 15/11/2026): Deformable-3DGS on a REAL non-D-NeRF scene, HyperNeRF `slice-banana` (monocular video, 4x images 268x480, 297 train / 33 held-out, 20k iterations,
  73 min, 392k Gaussians, trained with the repo's own Nerfies loader through `tests/make_hypernerf_split.py`): `convert_deformable3dgs.py --frames 330 --sh --compact`, PSNR 19.28-29.78 dB, mean 24.72 on the 33 held-out frames
  at 268 px (lens distortion ignored, as the repo does). Time already in [0, 1] (time_id / max), no converter change. `tests/check_real.py` item `d3dgs_banana`. `tests/make_hypernerf_blender.py` writes the D-NeRF-style layout native 4DGS needs
  (it has no Nerfies reader); SC-GS reads the same split as Deformable-3DGS.
- Tested (WP16, 15/11/2026): SC-GS on the same real HyperNeRF `slice-banana` split (512 nodes, hyper_dim 8, 20k iterations, 52 min, 603k Gaussians, no `--is_blender`; the clone needed `np.float` -> `np.float64`):
  `convert_scgs.py --frames 110 --sh --compact` (330 keyframes did not fit in 15 GB before WP19d), PSNR 20.21-30.22 dB, mean 26.69 on 33 held-out frames at 268 px; held-out times fall between keyframes. No converter change; item `scgs_banana`.
- Tested (WP16, 15/11/2026): native 4DGS on the same real `slice-banana` (via `tests/make_hypernerf_blender.py`; config `batch_size: 1`, `densify_until_num_points: 250_000` because a Sekiro game holds ~3 GB of the 12 GB GPU
  and the first three runs hit CUDA OOM; 20k iterations, 26 min): `convert_native4d.py --frames 110 --sh --compact`, PSNR 21.14-26.19 dB, mean 24.23 on 33 held-out frames at 268 px; the repo's own eval 26.14 dB (gap 1.9 dB not isolated).
  330 keyframes got the process OOM-killed before WP19d. No converter change; item `native4d_banana`.
- Tested (WP19d, 15/11/2026): `gs4d.Sequence4DWriter` (`add_keyframe(t, ...)`, constant fields in the constructor; raw float32 scratch files in
  `.gs4d_writer_*` beside the output, about K x N x 40 B, removed on close; static group and compact encoding kept), `tests/test_writer.py`: arrays bit for bit
  equal to `save` (plain, compact, with/without static group). `convert_scgs.py`, `convert_native4d.py`, `convert_deformable3dgs.py` use it; their outputs on
  `slice-banana` are identical to the old code in every array (SC-GS 110, native 110, Deformable-3DGS 330 keyframes). Peak RAM: SC-GS 110 kf 4.5 -> 1.4 GB,
  native 4DGS 110 kf 8.3 -> 1.3 GB, Deformable-3DGS 330 kf 7.2 -> 1.0 GB. 330 keyframes now convert: SC-GS 3.5 min, 3.7 GB file, PSNR 21.27-30.05, mean 26.87
  (110 kf: 26.69); native 4DGS 5 min, 19.9 GB file (per-keyframe `sh` is float32), PSNR 21.22-26.12, mean 24.27 (110 kf: 24.23), so interpolation between
  keyframes does not explain the 1.9 dB gap to the repo's eval. `check_real.py` items still use 110 keyframes; `psnr_item` loads eagerly, so the 19.9 GB
  file needs `load(lazy=True)`. The other converters still build the whole sequence in RAM.
- Tested (WP21, 15/11/2026): compact layout now also packs per-keyframe `sh`: `sh_lo [N,S,3]` + `sh_step [N,S,3]` float32 + `sh_u8 [K,N,S,3]`
  (`_PackedSH` in format.py; every coefficient of every Gaussian quantised over its own range in time, `sh = sh_lo + sh_u8 * sh_step`); constant `sh` stays
  float32; non-finite `sh` falls back to float32. `save`, `Sequence4DWriter` (bit for bit equal), lazy load, `GpuFrames`, static group all handle it;
  old compact files (float32 `sh`) still load; files with `sh_u8` cannot be read by older code. `tests/test_compact.py::test_sh_per_keyframe`, `tests/test_writer.py`.
  Native 4DGS `--sh --compact`: `slice-banana` 330 kf 19.86 -> 5.82 GB, PSNR mean 24.2687 both (worst image -0.0002 dB); `bouncingballs` 60 kf 1412 -> 453 MB,
  mean 32.9117 both (worst -0.0004 dB). Play frame, GPU path 2.2 -> 1.4 ms (banana), CPU path (`--cpu-decode`) 60 -> 128 ms (numpy decode of `sh`). Step 0: a
  coefficient's range in time (median 0.03) is far below its magnitude (DC up to 6.8), so a global int8 scale is ~300x worse (rms 1.6e-2 vs 5e-5).
  The writer's scratch files are still float32 (25 GB for banana 330 kf). `gs4d/open.py::estimate_bytes` does not know about per-keyframe `sh` (CC-B's file, not changed).
- Tested (WP19c part 3, 15/11/2026): the 1.9 dB gap of native 4DGS on `slice-banana` (ours 24.23, repo 26.14) is a bug in the repo's camera path for OUR Blender layout, not the converter: per-frame `fl_x/fl_y/cx/cy` leave `FovX = FovY = -1`, so the CUDA rasteriser's 2D covariance uses focal `W/(2 tan 0.5)` and clamps x/z to -0.71 (min > max), and the model learned to compensate. Emulating that in gsplat (`tests/diag_native4d_banana.py --steps c_focal`) gives mean 26.13 dB. No effect: keyframes at the 33 test times 24.27, no compact 24.27, `marginal_t` cut 24.29, near plane 0.2 24.27, SH direction from un-offset xyz 24.27; SH degree 0 / 1 only 22.38 / 23.05. `slice_native4d` matches the CUDA `computeCov3D_conditional` to 1e-15. No converter change. The diag script needs one process per step (110 keyframes with `--sh` get OOM-killed).
- NOT tested: the full Spacetime Gaussians model (with `rgbdecoder`); the client render of
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
