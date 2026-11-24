# Supported methods and validation results

The viewer reads one common `.npz` format ([FORMAT.md](FORMAT.md)); every 4DGS method has a converter that writes it.
The table below is the summary; the "Details" section gives the commands, measurements and images of each method.

- PSNR is measured with the viewer's gsplat renderer on the test / held-out images, at the resolution the model was trained at
  (at another resolution the PSNR drops by several dB), with `tests/check_real.py`, unless stated otherwise.
- "Real output": a trained model (published by the authors or trained by us) run through the converter. "Synthetic": only a test on
  fake data in the real file layout (`tests/test_<name>.py`), no real model yet.
- Output types: **A** Gaussians per frame · **B** 4D Gaussians, sliced in time · **C** parametric motion functions ·
  **D** deformation networks, re-run on the CPU.

| Type | Method (converter) | Real outputs checked | PSNR | Limits |
|---|---|---|---|---|
| A | Folder of per-frame ply files, including the `gaussian_pertimestamp/` of 4DGaussians (`convert_per_frame.py`) | ply files exported from 4DGaussians `bouncingballs`, `slice-banana` | matches the 4DGaussians converter | Gaussian counts that change between frames are not interpolated (`interpolate=False`) |
| A | Dynamic 3D Gaussians `params.npz` (`convert_per_frame.py`) | `basketball` (375,870 Gaussians × 150 frames) | no original images | up axis is `-y`; ~14% of the original colours lie outside [0,1] and are clipped; the client renderer was not compared with `visualize.py` |
| B | Native 4DGS, Yang et al. (`convert_native4d.py`, `--sh`) | self-trained `bouncingballs` (400 px); HyperNeRF `slice-banana` | `bouncingballs` 32.9 dB with `--sh` (repo 33.4); `slice-banana` 26.31 dB (repo 26.30) | without `--sh` only SH degree 0, mottled colour (29.0 dB); `--sh` makes the file big (`--compact` brings it to about 1/3) |
| B | 4D-Rotor GS, SIGGRAPH 2024 (`convert_4drotor.py`) | none yet | — | synthetic test only; colour is SH degree 3, constant in time (written to `sh`); default time range [0, 1] as in D-NeRF, not checked with real Nerfstudio cameras and data |
| C | Spacetime Gaussians, lite model (`convert_spacetime.py`) | `n3d_sear_steak` (published by the authors); Technicolor `Birthday` (no original images) | 34.37 dB; the repo's renderer 34.48 | full model with `rgbdecoder`: the converter stops, `--base-color-only` converts with the base colour; the full model was never tried |
| C | Shape of Motion (`convert_shape_of_motion.py`) | `paper-windmill` (published by the authors) | 25.86 dB at keyframes; `--stride 1` matches the repo's renderer to 0.000 dB | with a sparse `--stride`, lower between two keyframes (24.67 dB at stride 8) |
| C/D | MoSca, CVPR 2025 (`convert_mosca.py`) | none yet (preprocessing too heavy) | — | synthetic test only; DYNAMIC Gaussians only, the file of the static background is not read; stops on `max_sph_order > 0`; how the third-party renderer uses the scale was not read |
| D | E-D3DGS, ECCV 2024 (`convert_ed3dgs.py`) | none yet (py3.7/torch 1.13 environment not built) | — | synthetic test only; needs `cfg_args` (or `--set`); time N3V = frame/300, Technicolor = frame/50, HyperNeRF: `hyper_loader.py` not read; research-only license |
| D | 4DGaussians, hustvl (`convert_4dgaussians.py`) | self-trained `bouncingballs`; HyperNeRF `slice-banana` | 40.8 dB (repo 39.7); `slice-banana` 27.13 dB, same as the repo | stops on `no_grid`, `empty_voxel`, `static_mlp`, `grid_pe > 1`, `no_dshs=False` |
| D | Deformable-3DGS (`convert_deformable3dgs.py`) | self-trained: D-NeRF variant, default, `--is_6dof`; HyperNeRF `slice-banana` | 40.3 / 39.62 / 40.34 dB; `slice-banana` 24.72 dB | lens distortion ignored, as in the repo |
| D | SC-GS (`convert_scgs.py`) | self-trained `bouncingballs` (plain, with motion mask, with colour head); `slice-banana` | 40.3 dB; mask 40.14 dB; colour head 37.83 dB; `slice-banana` 26.69 dB | `--use_hash` and `--skinning` synthetic test only (the repo's own training did not converge / crashed); hash or view-direction colour heads stop |

Every converter needs torch except `convert_per_frame.py` and `convert_spacetime.py` (the CPU build is enough: torch only opens the
checkpoint or re-runs a small MLP). `python -m gs4d.open <output>` recognises all of the above (see the [user guide](USAGE.md#opening-an-output-directly-python--m-gs4dopen)).

## Details

### A — Folder of per-frame ply files (`convert_per_frame.py`)

```bash
python converters/convert_per_frame.py <ply_folder> -o scene.npz --fps 30 \
    [--stride 2] [--sh] [--compact]
```

- Input: a folder of standard 3DGS `.ply` files, one per frame, in natural order (`time_2` before `time_10`): per-frame output of a
  feed-forward model, or the `gaussian_pertimestamp/time_00000.ply ...` written by 4DGaussians' `export_perframe_3DGS.py`.
- `--sh` keeps `f_rest_*` (view-dependent colour, used by the gsplat renderer only); costs N × K × 12 × (degree+1)² more bytes, i.e.
  192 bytes per Gaussian per keyframe at SH degree 3.
- Gaussian counts that change between frames: padded with hidden Gaussians, the viewer jumps to the nearest keyframe instead of
  interpolating.
- Checked: the ply files exported from 4DGaussians `bouncingballs` and `slice-banana` give the file of `convert_4dgaussians.py` (see the
  4DGaussians section).

### A — Dynamic 3D Gaussians (`convert_per_frame.py`, `params.npz`)

```bash
python converters/convert_per_frame.py params.npz -o scene.npz \
    --stride 5 --fps 30 --fg-only
python viewer.py scene.npz --up -y
```

- Real output `basketball` (375,870 Gaussians × 150 original frames): `--fg-only` gives 112,592 Gaussians × 30 keyframes, 163 MB; without
  `--fg-only`: 375,870 Gaussians, 543 MB, but the scene is hidden behind background and floaters. The up axis is `-y`; screenshots at three times:
  [t0](images/dynamic3dgs/basketball_fg_up-y_t0.png), [t1](images/dynamic3dgs/basketball_fg_up-y_t1.png), [t2](images/dynamic3dgs/basketball_fg_up-y_t2.png).
- ~14% of the original colours lie outside [0,1] and are clipped; the client renderer was not compared with the original `visualize.py`.

### B — Native 4DGS, Yang et al. (`convert_native4d.py`)

```bash
python converters/convert_native4d.py chkpnt_best.pth -o scene.npz \
    --frames 60 --t0 0 --t1 1 [--sh] [--compact]
```

- `--t0/--t1` are the `time_duration` of the training config (D-NeRF scenes: [0, 1]); without them the script guesses the range from the
  Gaussians' temporal centres and prints the guess.
- Without `--sh` the colour is SH degree 0 only. With `--sh` the view- and time-dependent colour (`eval_shfs_4d`) is folded into the SH
  degree-3 coefficients of every keyframe.
- D-NeRF `bouncingballs`, self-trained for 20,000 iterations (38 minutes on an RTX 3060, 400 px): 134,260 Gaussians in the checkpoint,
  116,193 kept × 60 keyframes, 116 MB, converted in 2 seconds. 20 test images at 400 px: 26.0–31.9 dB, mean 29.0 dB (slightly mottled
  colour, the higher SH degrees are dropped); with `--sh` 27.3–39.6 dB, mean 32.9 dB, file 1.4 GB (453 MB with `--compact`), 8 seconds.
  The repo's own evaluation gives 33.4 dB; the remaining 0.5 dB has not been traced. Image: [native4d_bouncingballs.png](images/real/native4d_bouncingballs.png).
- HyperNeRF `slice-banana` (297 training images, 33 held-out images, 4x images = 268×480; the repo has no Nerfies loader, so the data is
  written in the D-NeRF layout by `tools/make_hypernerf_blender.py`; config `resolution: 1`, `batch_size: 1`,
  `densify_until_num_points: 250_000`, 20,000 iterations, 20 minutes): `--frames 110 --sh --compact` gives 21.41–29.17 dB,
  **mean 26.31 dB**; the repo's own evaluation 26.30 dB.
- Lesson from the first training run (WP19c, WP20): the old layout wrote `fl_x/fl_y/cx/cy` for every frame, which makes the repo set
  `FovX = FovY = -1`; the CUDA rasteriser then uses a wrong focal length and the model learned to compensate. That model gives 24.23 dB in
  the viewer (repo 26.14 dB); reproducing the wrong rendering with gsplat (`tools/diag_native4d_banana.py`) gives 26.13 dB, so the
  converter was not at fault. The new layout writes `camera_angle_x`; the 26.31 dB above is the retrained model.

### B — 4D-Rotor Gaussian Splatting (`convert_4drotor.py`)

```bash
python converters/convert_4drotor.py nerfstudio_models/step-000019999.ckpt -o scene.npz \
    --frames 60 --t0 0 --t1 1
```

- Input: a Nerfstudio checkpoint `step-*.ckpt` (`pipeline` state dict, keys `gauss_params.*`); needs torch.
- Positions and opacities change in time; covariance and SH degree 3 are written once. Time is the raw `time` of the transforms file
  (D-NeRF: [0, 1]).
- No real output yet: checked only with a synthetic checkpoint (`tests/test_4drotor.py`, against the conditioning of a 4D Gaussian).

### C — Spacetime Gaussians, lite model (`convert_spacetime.py`)

```bash
python converters/convert_spacetime.py point_cloud/iteration_25000/point_cloud.ply \
    -o scene.npz --frames 50 --t0 0 --t1 1
```

- Base colour `f_dc_0..2` only; the `rgbdecoder` MLP of the full model is not reproduced. For a full model (a ply with `f_dc_3..` or an
  `rgbdecoder` file next to the ply) the converter stops; `--base-color-only` converts it with the base colour.
- Colours are kept raw, not clipped to [0,1]: the model stores colours outside that range for 30% of the Gaussians (−5 to 6.9), and the
  repo blends the raw colours and clamps only the final pixel. Before 16/11/2026 the converter clipped the colours and reached only
  31.43 dB; files converted before that date must be converted again (`gs4d.open` reconverts when the converter is newer than the cached
  file). The client renderer still clips colours.
- `n3d_sear_steak_lite_allcam` (published by the authors): 108,317 Gaussians in the ply, 98,993 kept × 50 keyframes, 199 MB, 3 seconds.
  Against the original `sear_steak` videos of Neural 3D Video (21 cameras × frames 0, 12, 25, 37, 49 = 105 images, 1352×1014,
  time = frame/50): 27.76–37.81 dB, **mean 34.37 dB**. The repo's own renderer (`test_ours_lite` + the `forward_lite` rasteriser,
  `tools/repo_psnr.py spacetime`) on the same 105 images: 27.79–38.20 dB, mean 34.48 dB; the 0.11 dB gap comes from
  `--min-peak-opacity 0.02` (with 0 the match is within 0.01 dB).
- Technicolor `Birthday` (`ours_lite`, 658,999 Gaussians): `--frames 50 --compact` keeps 596,601 Gaussians, 583 MB, 20 seconds; renders from
  camera 0 of `cameras.json` at t = 0, 0.5, 1 show the right scene with the person moving; no original images, so no PSNR.

### C — Shape of Motion (`convert_shape_of_motion.py`)

```bash
python converters/convert_shape_of_motion.py <work_dir>/checkpoints/last.ckpt \
    -o scene.npz --fps 30 [--stride 8] [--fg-only]
```

- SE(3) motion bases + per-Gaussian coefficients, static background (stored only once in the file); needs torch to read the checkpoint.
  Time = frame / `--fps`.
- `paper-windmill` (published by the authors), `--stride 8`: 71,867 foreground + 924,075 background Gaussians × 35 keyframes (277
  original frames), 147 MB (1.27 GB before the static group), 10 seconds. Renders from the training cameras (`Ks`, `w2cs` in the
  checkpoint) match the scene; image: [shape_of_motion_paper-windmill.png](images/real/shape_of_motion_paper-windmill.png).
- Against the 1x images (720×960) of the DyCheck iPhone set at the training cameras, every 4th frame (70 images): at keyframes
  **25.86 dB** (min 24.28); at frames half way between two `--stride 8` keyframes 24.67 dB (min 21.61). With `--stride 1` (every frame a
  keyframe, 416 MB with `--compact`) 25.85 dB everywhere: the 1.2 dB drop is the linear interpolation between sparse keyframes of the
  spinning windmill.
- The repo's own renderer (`SceneModel.render`, `tools/repo_psnr.py som`) on the same 70 images: 24.28–27.66 dB, mean 25.85 dB; the
  `--stride 1` file differs by 0.000 dB on every image.

### C/D — MoSca (`convert_mosca.py`)

```bash
python converters/convert_mosca.py photometric_d_model_native_add3.pth -o scene.npz \
    --fps 30 --stride 2
```

- Scaffold nodes + skinning over the K nearest nodes, dual-quaternion blending (or LBS); needs torch. Time = frame / `--fps`.
- Dynamic Gaussians only; the static background (`photometric_s_model_*.pth`) is not read. Stops on `max_sph_order > 0`.
- No real output yet: checked only with a synthetic state dict (`tests/test_mosca.py`, against the textbook normalised dual-quaternion
  blend).

### D — E-D3DGS (`convert_ed3dgs.py`)

```bash
python converters/convert_ed3dgs.py <model_folder> -o scene.npz \
    --frames 60 --t0 0 --t1 0.9967 --sh
```

- A model folder with `point_cloud/iteration_*/{point_cloud.ply,deformation.pth}` and `cfg_args`. Per-Gaussian embeddings + a temporal
  table; the coarse and fine MLPs are re-run on the CPU; needs torch.
- Time: N3V = frame/300 (`--t1 0.9967` for 300 frames), Technicolor = frame/50. `cfg_args` decides which heads are on (`no_do`, `no_dc`,
  `use_anneal`, ...); `--set KEY=VALUE` overrides one value. Colour and opacity are per keyframe when `no_dc` / `no_do` are False.
- No real output yet: checked only with a synthetic model against a torch module written after the repo (`tests/test_ed3dgs.py`).

### D — 4DGaussians, hustvl (`convert_4dgaussians.py`)

```bash
python converters/convert_4dgaussians.py <model_folder> -o scene.npz --frames 60 [--sh]
```

- A model folder with `point_cloud/iteration_*/{point_cloud.ply,deformation.pth}` and `cfg_args`. The HexPlane + deformation MLP are
  re-run on the CPU (needs torch), without the repo's environment. The old route still works: run `export_perframe_3DGS.py` in the repo,
  then use type A.
- Stops (naming the variant) on `no_grid`, `empty_voxel`, `static_mlp`, `grid_pe > 1`, `no_dshs=False` (SH that change in time).
- D-NeRF `bouncingballs`, self-trained (20,000 iterations, 692 s on an RTX 3060, `arguments/dnerf/bouncingballs.py`, white background):
  27,043 Gaussians, `--frames 60 --sh` gives a 64 MB file in 4 seconds. 20 test images at 800 px (composited on white): 36.7–45.9 dB,
  mean 40.8 dB (the repo's own evaluation 39.7 dB at iteration 14,000). Against `export_perframe_3DGS.py` on the same model, at the 20 test
  times: positions differ by at most 1.2e-6, covariances by 7e-6 relative, opacities are identical.
- HyperNeRF `slice-banana` (the repo's own HyperNeRF reader, `arguments/hypernerf/banana.py`, 14,000 iterations, 18 minutes, 4x images
  268×480, 297 training / 33 held-out images): 52,741 Gaussians, `--frames 330 --sh --compact` gives a 325 MB file in 35 seconds, peak RAM
  1.5 GB. 33 held-out images at 268 px: 22.05–30.39 dB, mean 27.13 dB; the repo's own renders of the same 33 images give the same numbers
  (at most 0.003 dB apart per image). This repo keeps HyperNeRF cameras and points in their original coordinates, time = warp_id / max.

### D — Deformable-3DGS (`convert_deformable3dgs.py`)

```bash
python converters/convert_deformable3dgs.py <model_folder> -o scene.npz --frames 60 [--sh]
```

- A model folder with `point_cloud/` and `deform/`; the deformation MLP is re-run on the CPU (needs torch). The D-NeRF (timenet), default
  and 6-DoF variants are read from the state dict.
- `bouncingballs` (D-NeRF variant, self-trained for 20,000 iterations, 34 minutes, 800 px): `--sh` gives 64,571 Gaussians × 60 keyframes,
  SH degree 3, 153 MB (84 MB with `--compact`), 49 seconds. 20 test images at 800 px: 37.2–43.6 dB, mean 40.3 dB (at 400 px only
  29.3–33.2 dB, because the model was trained at 800 px). Image (400 px):
  [deformable3dgs_bouncingballs.png](images/real/deformable3dgs_bouncingballs.png).
- Default variant (no `--is_blender`): 35.35–43.14 dB, mean 39.62 dB. 6-DoF variant (`--is_6dof`, 27 minutes): 36.99–43.75 dB,
  mean 40.34 dB. Neither needed a converter change.
- HyperNeRF `slice-banana` (single-camera video, 297 training / 33 held-out images, 4x images 268×480, 20,000 iterations, 73 minutes,
  392,000 Gaussians; data built with `tools/make_hypernerf_split.py`): `--frames 330 --sh --compact` (exactly the 330 times of the video),
  33 held-out images at 268 px: 19.28–29.78 dB, mean 24.72 dB (lens distortion ignored, as in the repo's loader).

### D — SC-GS (`convert_scgs.py`)

```bash
python converters/convert_scgs.py <model_folder>_node -o scene.npz \
    --frames 60 [--knn 3] [--sh]
```

- A model folder with `point_cloud/` and `deform/`; control nodes + MLP + blending over the K nearest nodes, run on the CPU (needs torch).
  `--knn` = the `--K` used for training when it was not 3; `--max-d-scale` = the training `--max_d_scale`, if any.
- `bouncingballs` (512 nodes, `--local_frame`, 20,000 iterations after 10,000 node-initialisation steps, 19 minutes, 400 px): 165,900
  Gaussians × 60 keyframes, 10 seconds, 361 MB (393 MB with `--sh`). 20 test images at 400 px: 35.4–43.0 dB, mean 40.3 dB with `--sh`;
  27.8–33.5 dB (5 images) with the base colour only. Image: [scgs_bouncingballs.png](images/real/scgs_bouncingballs.png).
- With motion mask (`--gs_with_motion_mask --gt_alpha_mask_as_dynamic_mask`, ply with `fea_0..8`, 20,000 iterations, 20 minutes, 170,091
  Gaussians): 35.48–43.14 dB, mean 40.14 dB; the converter recognises the mask by itself.
- Colour head `--pred_color`: the per-node d_color is re-run and added to the SH DC coefficient, so the colours (and `sh` with `--sh`) are
  written per keyframe; `--ignore-pred-color` keeps the colour constant (smaller file, wrong colours). Checked on a self-trained
  `bouncingballs` tinted over time: 37.83 dB with the colour head, 24.48 dB without it (item `scgs_predcolor`). Hash or view-direction
  colour heads: the converter stops.
- HyperNeRF `slice-banana` (512 nodes, hyper_dim 8, 20,000 iterations, 52 minutes, 603,000 Gaussians; the SC-GS clone needed
  `np.float` → `np.float64`): `--frames 110 --sh --compact` 20.21–30.22 dB, mean 26.69 dB; `--frames 330` (every held-out image on a
  keyframe; 3.5 minutes, 3.7 GB file) 21.27–30.05 dB, mean 26.87 dB.
- `--use_hash` (the hash grid rewritten in numpy, matches tinycudann to 2e-4) and `--skinning`: synthetic test only; the repo's own
  training did not converge (hash, 6.8 dB) or crashed after densification (skinning).

## Stops with a message (the converters never guess)

- SC-GS: `--deform_type mlp` (that is Deformable-3DGS), hash with `tcnn_mlp`, hash / view-direction colour heads, ply and `deform.pth`
  with mismatching numbers of `fea_*` columns.
- Spacetime: the full model with `rgbdecoder` (use `--base-color-only` to convert without it).
- 4DGaussians: `no_grid`, `empty_voxel`, `static_mlp`, `grid_pe > 1`, `no_dshs=False`.
- E-D3DGS: a ply without SH degree 3 or without `embedding_*` columns; `--set` with a key that is not in ModelHiddenParams.
- MoSca: `max_sph_order > 0`; the static background file `photometric_s_model_*.pth` (`gs4d.open` says the dynamic file is needed).
- Native 4DGS: a checkpoint without exactly the 19 fields of `capture()` (not a 4D model), or fewer SH channels than required.
- Every converter: torch missing when needed (with the install command), a torch file that cannot be read, a model folder without
  `iteration_*` (or without the chosen `--iteration`), a file in the wrong layout. The message is one line, not a traceback.

`python -m gs4d.open <output>` recognises all of the above and converts; unsupported variants stop and name the flag that converts anyway.
To add a method: [ADD_A_CONVERTER.md](ADD_A_CONVERTER.md).

## Repository notes taken before writing the three newest converters (17/11/2026)

Kept for reference: these are the notes from reading the repositories before `convert_4drotor.py`, `convert_ed3dgs.py` and
`convert_mosca.py` were written. Points marked "not verified" were read in full while writing the converters (see each converter's
docstring).

Gathered from the official repositories by reading the README and the source code (nothing was run). An item marked "not verified" is
something the reading did not cover. The effort estimates for a converter are the writer's own, based on similar existing converters; the
repositories publish no GPU, RAM or training-time figures.

### 4D-Rotor Gaussian Splatting (4DRotorGS, SIGGRAPH 2024) — `convert_4drotor.py` exists (17/11/2026)
Position formula of `slice_4d_to_3d`, now read: `mean(t) = xyz + speed·(t − t_g)`, `speed = Σ[:3,3]/Σ[3,3]`; the rotors are not normalised
when rendering. The checkpoint is a `pipeline` state dict whose keys end in `gauss_params.<name>`; the `_model.` prefix was not verified on
a real file, so the converter matches on the end of the key.
- Repo: https://github.com/weify627/4D-Rotor-Gaussians, Apache-2.0 license, built on Nerfstudio. Paper: arXiv 2402.03307.
- Output: `step-<N>.ckpt` (`torch.save` with `step, pipeline, optimizers, ...`); the Gaussian parameters are in `gauss_params`:
  `xyz, t, scales_xyz, scales_t, quats1, quats2, features_dc, features_rest, opacities`. No per-Gaussian ply export was found.
- Formulas (`splatfacto.py`, `general_utils.py::slice_4d`): the 8-number rotor → a 4D rotation matrix, `scale = exp(cat(scales_xyz,
  scales_t))`, Σ = L·Lᵀ (4×4). `w = 1/Σ[3,3]`, conditional 3D covariance `cov3 = Σ[:3,:3] − Σ[:3,3]·Σ[3,:3]·w` (independent of t); opacity
  `sigmoid(o)·exp(−0.5·w·(t_g − t)²)`, a Gaussian is dropped when the exponent is ≥ 16; colour is SH degree 3, constant in time. Position
  formula of `slice_4d_to_3d` and time range: not verified.
- Type: B (same family as native 4DGS; `convert_native4d.py` is the model, including how Gaussians are dropped and opacity is baked per
  keyframe).
- Data: D-NeRF (`splatfacto`, 20,000 iterations) and N3V (`splatfacto-big`, needs `scripts/n3v2blender.py`).
- Installation: Python 3.8, torch 2.1.2+cu118, tiny-cuda-nn, PyTorch3D, three separate CUDA submodules
  (`diff-gaussian-rasterization-confidence`, `knn`, `knn_ops_3_fwd_bwd_mask`) and `simple_knn`. Custom rasteriser; the README says the
  fast-FPS version is not released.
- Converter effort: small to medium (about 1 day): numpy only + torch to open the ckpt; the formulas are clear except the position slicing,
  which needs `slice_4d_to_3d` read in full. Use a native-4DGS-style `--sh`, since the colour is SH.
- Training on a 12 GB GPU / 15 GB RAM machine: D-NeRF looks feasible (small scenes, as native 4DGS was trained), but needs a
  py3.8/cu118 environment and the three submodules built; N3V unknown, no memory figures.

### E-D3DGS (ECCV 2024, Per-Gaussian Embedding-Based Deformation) — `convert_ed3dgs.py` exists (17/11/2026)
`scene/deformation.py`, the renderer and `render.py` were read in full: the MLP input is only [temporal embedding, Gaussian embedding], the
coarse and fine branches only add up (the coarse branch's output is not fed into the fine branch, contrary to the first note); the deltas
are added BEFORE the activations (log scale, logit opacity, quaternion, then normalised).
- Repo: https://github.com/JeongminB/E-D3DGS (project page https://jeongminb.github.io/e-d3dgs/). The license is the Inria/MPII Gaussian
  Splatting license: research only, no commercial use.
- Output: `point_cloud/iteration_N/point_cloud.ply` (x, y, z, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`, plus `embedding_0..E-1`;
  values before activation) and `deformation.pth` (state_dict of `deform_network`), with `cfg_args`.
- Deformation (`scene/deformation.py`): every Gaussian has its own embedding; time is a learned latent table (`weight`, bilinear lookup at
  t); coarse and fine MLP branches, the fine one fed by the coarse one. The outputs are added to: position, log-scale, quaternion, logit
  opacity, and the SH coefficients (view-dependent colour included). Time `t = frame / duration` (dynerf 300, Technicolor 50). The number
  of Gaussians is fixed over time.
- Type: D (like Deformable-3DGS: the MLP is re-run on the CPU), plus opacity and SH deltas, so per-keyframe `sh` is needed, as in
  `convert_scgs.py --pred_color`. Two settings not verified: the embedding size (`arguments/` files) and the iteration used for rendering.
- Data: N3V (multi-camera), Technicolor, HyperNeRF (single camera); needs COLMAP.
- Old setup: Python 3.7, torch 1.13.1+cu116, mmcv 1.6.0, open3d, submodules `diff-gaussian-rasterization` and `simple-knn`.
- Converter effort: medium (1–2 days): coarse + fine MLPs and the temporal-table lookup in numpy, checked against `deformation.pth` run
  with torch, as in `test_deformable3dgs.py`.
- Training on this machine: uncertain. N3V is many cameras × 300 frames (tens of GB, like the full Spacetime model that could not be
  done); HyperNeRF `slice-banana` (already in `../data_real`) is possible, but the py3.7/torch 1.13 environment has to be built separately.

### MoSca (CVPR 2025, Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds) — `convert_mosca.py` exists (17/11/2026)
Read: `mosca.py`, `dynamic_gs.py`, `dualquat_helper.py`, `photo_recon.py`, `mosca_evaluate.py`. File needed:
`photometric_d_model_<backend>.pth` (state_dict of `DynSCFGaussian`, scaffold under `scf.*`). Node sigma = `max_sigma·sigmoid(logit)`; DQB
also blends with the identity through `dyn_o` (default `1 − 1e-3`), so motion is never perfectly rigid. The static background is in a
separate file that the converter does not read.
- Repo: https://github.com/JiahuiLei/MoSca, MIT for `lib_moca` and `lib_mosca`; the third-party rasterisers and models have their own
  licenses.
- Output: `mosca/mosca.pth` (scaffold: `_node_xyz [T,M,3]`, per-frame `_node_rotation`, `_node_sigma_logit`, `topo_knn_ind`, ...); the
  dynamic Gaussian model `DynSCFGaussian` stores `_xyz` (local coordinates of the attached node), `_rotation`, `_scaling`, `_opacity`,
  `_features_*`, `_skinning_weight`, `attach_ind`, `ref_time`. The file that saves the Gaussian model (`photo_recon.py`) was not read, so
  its name was not verified.
- Motion: position = the rotation/translation of the attached node at `ref_time`, then `warp` to frame t by skinning over the K nearest
  nodes (K = 16 during reconstruction) with weights `exp(−|x−node|²/(2σ²))`; dual-quaternion blending (`dqb`, default) or `lbs`. Scale,
  opacity and colour are constant in time (except `nn_fusion`, which sets opacity to 0 outside the supporting frames). Time is the frame
  index. The DQB part (`scaffold_utils/dualquat_helper.py`) was not read.
- Type: D or hybrid C (close to SC-GS: control nodes + K nearest nodes, plus DQB); `convert_scgs.py` and `convert_shape_of_motion.py` are
  the models.
- Data: casual videos (iPhone/DyCheck, NVIDIA, TUM, Sintel, in the wild); needs heavy preprocessing: trackers (BootsTAPIR, CoTracker,
  SpaTracker), depth (DepthCrafter, Metric3D-v2, UniDepth), RAFT, then camera bundle adjustment; weights downloaded by hand.
- Installation: Python 3.10, torch 2.1.0+cu118, pytorch3d, torch_geometric; three custom rasterisers (the repo's TODO: move to gsplat).
  `cuda:0` is hard-coded.
- Converter effort: bigger than the two above (several days): DQB and `get_skinning_weights` must be copied exactly, `resample_time` goes
  from sparse keyframes (every 4 frames) to every frame, and Gaussians have a `ref_time`, so the set of live Gaussians changes per frame.
- Training on this machine: unlikely. Several heavy preprocessing models must run; the repo publishes no memory requirements. The
  `paper-windmill` scene is DyCheck iPhone data, which MoSca supports, so it could be compared with Shape of Motion if training worked.

### Suggested order (by effort / benefit, not a decision)
4D-Rotor GS (clearest formulas, same family as native 4DGS) → E-D3DGS (needs an old environment) → MoSca (heavy preprocessing and
converter).
