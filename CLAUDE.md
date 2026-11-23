# gs4d_viewer — context for Claude Code sessions

Read this instead of re-deriving it. Work packages and ownership rules are in `PLAN.md`; the dated log of every check and
measurement is `docs/HISTORY.md` (read the part you need, add new entries at its end).

## What this is
Browser viewer (viser) for 4D Gaussian Splatting results with a time slider. The viewer reads ONE
common `.npz` ("Gaussians per keyframe", `gs4d/format.py`, class `Sequence4D`). Every 4DGS method gets
its own `convert_<method>.py` that writes that format. The viewer never imports a method's code.

## Layout
- `gs4d/format.py`  container, interpolation (`Sequence4D.at(t)`), `cov_from_quat_scale`, `pack_cov/unpack_cov`;
  compact storage (also per-keyframe `sh`) and the static group (Gaussians that never change are stored once, `save(split_static=True)`);
  `Sequence4DWriter` writes the same file one keyframe at a time (every converter uses it); `GpuFrames` decodes and
  interpolates with torch for the gsplat renderer (any torch device; pinned memory only on CUDA)
- `gs4d/ply.py`     standard 3DGS ply reader (exp on scale, sigmoid on opacity, SH0 colour)
- `gs4d/checkpoint.py`  `import_torch(script)`, `torch_load(path, script, weights_only)`: torch on demand, one-line errors
- `gs4d/open.py`     `python -m gs4d.open <output> [viewer args]` recognises the output, converts it next to it
  (`<name>.gs4d.npz`, reused while the output and the converter are unchanged) and opens the viewer
- `gs4d/__main__.py`  `python -m gs4d in.npz out.npz [--float32] [--no-split]` re-saves compact / plain
- `viewer.py`       GUI + two renderers: `client` (viser WebGL splats) and `gsplat` (CUDA, image streamed);
  `--compare b.npz` (second sequence on the same time slider), `--cameras` (training cameras, ground truth + PSNR),
  `--camera-path`, Draw as points, Motion (colour by speed / displacement, trails), mp4 export
- `convert_per_frame.py`  type A: folder of per-frame ply, Dynamic 3D Gaussians `params.npz`
- `convert_native4d.py`   type B: Yang et al. `chkpnt*.pth` (`--sh`: view- and time-dependent SH per keyframe)
- `convert_4drotor.py`    type B: 4D-Rotor GS Nerfstudio `step-*.ckpt`; synthetic test only
- `convert_spacetime.py`  type C: Spacetime Gaussians `point_cloud.ply` (lite model; base colour only)
- `convert_shape_of_motion.py`  type C: Shape of Motion `checkpoints/last.ckpt` (SE(3) motion bases)
- `convert_mosca.py`      type C/D: MoSca `photometric_d_model_*.pth`, dynamic Gaussians only; synthetic test only
- `convert_deformable3dgs.py`  type D: Deformable-3DGS model folder (deformation MLP re-run on CPU)
- `convert_scgs.py`       type D: SC-GS model folder (control nodes + MLP + K-nearest-node blending; hash grid, skinning, colour head)
- `convert_4dgaussians.py`  type D: hustvl 4DGaussians model folder (HexPlane + MLP re-run on CPU)
- `convert_ed3dgs.py`     type D: E-D3DGS model folder (per-Gaussian `embedding_*` + MLP re-run on CPU); synthetic test only
- `convert_template.py`   working template for a new converter (`docs/ADD_A_CONVERTER.md`, `tests/test_template.py`)
- `make_toy.py`     synthetic scene (`--sh` adds fake SH)
- `tests/run_all.py` runs every `test_*.py` / `check_*.py` one at a time: pass / skipped (reason) / failed, exit 1 on failure;
  `NEEDS` says what a file needs (torch, cuda, chrome, data); screenshots go to a temp folder unless `--keep-shots`
- `tests/test_*.py` CPU checks (converters on synthetic files in the real layout, format, writer, GpuFrames on the CPU device,
  `gs4d.open`, CLI errors); the converter ones need torch. `tests/check_*.py` drive the viewer in headless Chrome;
  `check_gsplat.py` and `check_cameras.py` need CUDA, `check_real.py` and `check_cameras.py` the real data
  (`../data_real`, `/home/cont/gs4d_train`). `bench_large.py`, `bench_display.py` measure; `repo_psnr.py` runs the methods'
  own renderers; `make_hypernerf_*.py`, `diag_native4d_banana.py` are one-off tools of WP16 / WP19c.
- `docs/` `SUPPORTED.md` (methods, real outputs checked, PSNR), `FORMAT.md` (file format, compact layout, static group,
  writer), `PERFORMANCE.md` (speed and memory), `ADD_A_CONVERTER.md`, `HISTORY.md` (the log); `shots/` screenshots and mp4s
- `ruff.toml` lint rules (CI runs `ruff check .`); `.github/workflows/test.yml` lint + tests without / with CPU torch
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

## Status (23/11/2026)
- Product goal (WP9) reached for every known output type: `python -m gs4d.open <output>` recognises, converts and opens it.
  Per method (real output checked or synthetic test only, PSNR, limits): `docs/SUPPORTED.md`.
- Every viewer feature has a headless-Chrome check script and was tried on real outputs (details in `docs/HISTORY.md`).
- Checks with real data run only on Thai's machine (RTX 3060 12 GB, 15 GB RAM): `tests/check_real.py`, `tests/check_cameras.py`,
  `tests/bench_display.py`. GitHub Actions runs lint, the tests without and with CPU torch, and the client-renderer browser checks.
- Synthetic test only (no real model): 4D-Rotor GS, E-D3DGS, MoSca, SC-GS `--use_hash` / `--skinning`. The SC-GS colour head
  (`--pred_color`) is checked on a self-trained time-tinted `bouncingballs` (37.83 dB with the head, 24.48 without).
  Not supported: Spacetime full model (`rgbdecoder`), view-direction colour heads, SC-GS with `d_rot_as_res` off.
- Not measured: a browser on another machine over SSH (WP38). `GpuFrames` changes of 23/11 checked on the CPU device only.
- Open work packages: WP32 (demo scenes), WP33 (`setup.sh`), WP36, WP37, WP38; see `PLAN.md`.

## Conventions
- A converter is a new file `convert_<method>.py` + a new `tests/test_<method>.py` + one row in `docs/SUPPORTED.md`
  (and the short table of the README). It must not edit `viewer.py`, `gs4d/format.py` or other converters.
- Converters run on CPU with numpy (torch only to unpickle weights or run a small MLP, imported through `gs4d.checkpoint`).
  Read reference repos to port formulas; do not execute their code and do not add them as dependencies.
- Every converter test builds synthetic inputs in the method's real file layout and checks the maths
  against an independent formulation, as `tests/test_all.py::test_native4d` does. A test that needs torch goes into `NEEDS`.
- Unsupported variants stop with a one-line `SystemExit` that names the variant (and the flag that converts anyway, if any).
- Run before finishing: `python tests/run_all.py` (at least `tests/test_all.py` and your own test file) and `ruff check .`.
- Headless screenshot check: start `python viewer.py toy.npz --port 8123`, open with Playwright + Chrome flags
  `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`.
- Measurements go to `docs/` (per-method results: `SUPPORTED.md`, speed and memory: `PERFORMANCE.md`, file format: `FORMAT.md`)
  plus a dated entry at the end of `docs/HISTORY.md`; the README stays a user guide.
- Everything in the repository is written in English (docs, `PLAN.md`, comments, messages), with English number separators.
- Finish with a 5-line report: what changed, what was run, what is untested, files touched.
