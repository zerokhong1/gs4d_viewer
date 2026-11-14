# gs4d_viewer — context for Claude Code sessions

Read this instead of re-deriving it. Work packages and ownership rules are in `PLAN.md`.

## What this is
Browser viewer (viser) for 4D Gaussian Splatting results with a time slider. The viewer reads ONE
common `.npz` ("Gaussians per keyframe", `gs4d/format.py`, class `Sequence4D`). Every 4DGS method gets
its own `convert_<method>.py` that writes that format. The viewer never imports a method's code.

## Layout
- `gs4d/format.py`  container, interpolation (`Sequence4D.at(t)`), `cov_from_quat_scale`, `pack_cov/unpack_cov`
- `gs4d/ply.py`     standard 3DGS ply reader (exp on scale, sigmoid on opacity, SH0 colour)
- `viewer.py`       GUI + two renderers: `client` (viser WebGL splats) and `gsplat` (CUDA, image streamed);
  `--compare b.npz` shows a second sequence (A, B, or side by side) on the same time slider
- `convert_per_frame.py`  type A: folder of per-frame ply, Dynamic 3D Gaussians `params.npz`
- `convert_native4d.py`   type B: Yang et al. `chkpnt*.pth`
- `convert_spacetime.py`  type C: Spacetime Gaussians `point_cloud.ply` (base colour only)
- `convert_shape_of_motion.py`  type C: Shape of Motion `checkpoints/last.ckpt` (SE(3) motion bases)
- `convert_deformable3dgs.py`  type D: Deformable-3DGS model folder (deformation MLP re-run on CPU)
- `convert_scgs.py`       type D: SC-GS model folder (control nodes + MLP + K-nearest-node blending)
- `gs4d/__main__.py`  `python -m gs4d in.npz out.npz [--float32]` re-saves compact / plain
- `make_toy.py`     synthetic scene (`--sh` adds fake SH); `tests/test_all.py` self-checks (CPU only)
- `tests/`          `test_sh.py`, `test_compact.py`, `test_spacetime.py`, `test_shape_of_motion.py`,
  `test_deformable3dgs.py`, `test_scgs.py` (the last three need torch), `check_camera.py`, `check_gsplat.py`
  (needs CUDA), `check_compare.py` (needs Chrome), `bench_large.py`; screenshots and the demo mp4 are in `shots/`
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
Updated 14/11/2026. All eight work packages (WP1-WP8, WP7 both options, WP5 both methods) are merged into `main`.
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
- Tested with real output (WP2): `params.npz` reader on Dynamic 3D Gaussians `basketball`; up axis is `-y`.
- Tested (WP7a): `--compare` in headless Chrome with both renderers, `tests/check_compare.py`. gsplat: split
  screen, same camera; client: B moved sideways in the scene (viser cannot split the viewport).
- Tested with real output (14/11/2026), gsplat render of the converted `.npz` against the method's own images
  (numbers in the README table, sheets in `shots/real/`, top = ours, bottom = ground truth):
  - `convert_spacetime.py`: authors' `n3d_sear_steak_lite_allcam` (lite model: `f_dc` only, no `rgbdecoder`);
    checked by eye at training cameras, image not in the repo (shows a person).
  - `convert_shape_of_motion.py`: authors' `paper-windmill` checkpoint, checked by eye.
  - `convert_deformable3dgs.py --sh`: self-trained D-NeRF `bouncingballs` (20k iterations), 29.3-33.2 dB on 5 test frames.
  - `convert_scgs.py --sh`: self-trained `bouncingballs` (512 nodes, `--local_frame`, hyper_dim 8), 36.2-41.3 dB.
  - `convert_native4d.py`: self-trained `bouncingballs` `chkpnt_best.pth`, 26.0-31.1 dB (the repo's own eval:
    33.4 dB); SH degree 0 only, so colours are mottled. The 19-field tuple above matches the real checkpoint.
- Each of those converters also has a synthetic test, `tests/test_<name>.py`.
- NOT tested: the full Spacetime Gaussians model (with `rgbdecoder`); Deformable-3DGS default and 6-DoF variants,
  SC-GS with a motion mask, and any non-D-NeRF scene for Deformable-3DGS / SC-GS / native 4DGS; the client render of
  `basketball` against the original `visualize.py`; display FPS in a real browser (README numbers are server side).
- Local training setup for those methods (not in the repo): `/home/cont/gs4d_train` (env, CUDA toolkit, clones,
  outputs), data in `../data_real`. Build or train one job at a time: the machine has 15 GB RAM.
- Not done: a compact flag in the converters; SC-GS `--use_hash`, `--skinning`, `--pred_color`, `d_rot_as_res` off.

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
