# gs4d_viewer — context for Claude Code sessions

Read this instead of re-deriving it. Work packages and ownership rules are in `PLAN.md`.

## What this is
Browser viewer (viser) for 4D Gaussian Splatting results with a time slider. The viewer reads ONE
common `.npz` ("Gaussians per keyframe", `gs4d/format.py`, class `Sequence4D`). Every 4DGS method gets
its own `convert_<method>.py` that writes that format. The viewer never imports a method's code.

## Layout
- `gs4d/format.py`  container, interpolation (`Sequence4D.at(t)`), `cov_from_quat_scale`, `pack_cov/unpack_cov`
- `gs4d/ply.py`     standard 3DGS ply reader (exp on scale, sigmoid on opacity, SH0 colour)
- `viewer.py`       GUI + two renderers: `client` (viser WebGL splats) and `gsplat` (CUDA, image streamed)
- `convert_per_frame.py`  type A: folder of per-frame ply, Dynamic 3D Gaussians `params.npz`
- `convert_native4d.py`   type B: Yang et al. `chkpnt*.pth`
- `convert_spacetime.py`  type C: Spacetime Gaussians `point_cloud.ply` (base colour only)
- `gs4d/__main__.py`  `python -m gs4d in.npz out.npz [--float32]` re-saves compact / plain
- `make_toy.py`     synthetic scene (`--sh` adds fake SH); `tests/test_all.py` self-checks (CPU only)
- `tests/`          `test_sh.py`, `test_compact.py`, `test_spacetime.py`, `check_camera.py`, `check_gsplat.py`
  (needs CUDA), `bench_large.py`; screenshots and the demo mp4 are in `shots/`
- Environment: `.venv` (Python 3.10, torch 2.4.1+cu124, gsplat 1.5.3); system `python3` has no torch.
  Install commands are in the README.

## Verified facts (do not re-research)
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
- Dynamic 3D Gaussians: `params.npz` keys `means3D[T,N,3]`, `unnorm_rotations[T,N,4]` (wxyz),
  `rgb_colors[T,N,3]`, `logit_opacities[N,1]`, `log_scales[N,3]`, `seg_colors[N,3]`.

## Status
Updated 13/11/2026. WP1, WP2, WP3, WP6, WP7b, WP8 are merged into `main`.
- Tested: client renderer in headless Chrome on the toy scene; `tests/test_all.py` (4 checks);
  `viewer.camera_matrices` (viser camera -> viewmat, K) against the browser render, `tests/check_camera.py`.
- Tested on CUDA (RTX 3060, WP1): the gsplat `rasterization(...)` call and image streaming in `_draw_gsplat`,
  `tests/check_gsplat.py`; FPS table in the README.
- Tested (WP6): optional `sh` field, view-dependent colour in the gsplat renderer, `tests/test_sh.py`.
  The client renderer always draws `colors`; only `convert_per_frame.py --sh` writes `sh`.
- Tested (WP7b): mp4 export from the current camera over the whole time range, both renderers.
- Tested (WP8): compact save (float16 offsets, quantised Cholesky, uint8) and lazy memory-mapped load,
  `tests/test_compact.py`, `tests/bench_large.py`. Do not overwrite a file the viewer has open.
- Tested with real output (WP2): `params.npz` reader on Dynamic 3D Gaussians `basketball`; up axis is `-y`.
- Synthetic test only (WP3): `convert_spacetime.py`, `tests/test_spacetime.py`.
- NOT tested: any real native-4DGS `chkpnt*.pth`; Spacetime Gaussians with real output; the client render of
  `basketball` against the original `visualize.py`; display FPS in a real browser (README numbers are server side).
- Not done: WP7(a) two sequences side by side; WP4, WP5 converters; a compact flag in the converters.

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
