# Verification log

What was checked, how, and with which numbers, in the order it was done (14/11/2026 onward). This was the "Status"
section of `CLAUDE.md` until 23/11/2026; it moved here unchanged so that `CLAUDE.md` stays short. Newer entries go at the
end. Per-method results in table form: [SUPPORTED.md](SUPPORTED.md); speed and memory: [PERFORMANCE.md](PERFORMANCE.md).
WPnn are the work packages of `PLAN.md`.

## Log up to 20/11/2026

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
  Display FPS in a real GPU Chrome: measured in WP19b (below).
- Tested (WP18, 14/11/2026): `viewer.py --cameras <transforms_*.json | cameras.json>` (`load_cameras`): Cameras folder in the panel,
  selecting one moves the client camera and the slider to the image time; gsplat renderer: "Ground truth beside render" shows
  render | image with PSNR (`Viewer.gt_compare`, ignores the opacity/size sliders, black or white background, capped by Max render
  height). `tests/check_cameras.py`: PSNR matches `check_real.psnr_item` to 0.0002 dB on the 20 test cameras (SC-GS 400 px,
  Deformable-3DGS 800 px); `check_real.py` unchanged (7 passed). Screenshots `shots/cameras/`. `gs4d/open.py` passes `--cameras` since WP19a; With `--compare` since WP29.
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
- Tested (WP22, 15/11/2026): `convert_4dgaussians.py`, `convert_shape_of_motion.py`, `convert_spacetime.py`, `convert_per_frame.py` write through `Sequence4DWriter` (generators `deform_frames`, `foreground_frames`,
  `slice_frames` + `peak_opacity` (Spacetime: first pass computes the peak opacity for `--min-peak-opacity`), `write_ply_folder`, `write_dynamic3dgs`; the in-memory functions stay). Output on real data is
  identical to the old code in every array (np.load byte compare): 4DGaussians `bouncingballs` and `slice-banana` 330 kf, Shape of Motion `paper-windmill --stride 8`, Spacetime `n3d_sear_steak` and Technicolor `Birthday`,
  per-frame ply folder (with/without `--sh --compact`); Dynamic 3D Gaussians (no real params.npz here) identical on synthetic data, 8 combinations. Peak RAM before -> after: 4DGaussians banana 1.49 -> 0.60 GB, SoM stride 8 3.2 -> 0.88 GB,
  Birthday 3.28 -> 0.44 GB, steak 590 -> 102 MB. SoM with all 277 frames now converts (0.88 GB, 69 s, 416 MB); the old code needs > 11 GB (it OOM-killed the whole app, exit 137). per-frame ply folders whose Gaussian count changes
  (`interpolate=False`) keep the old in-memory path (Writer needs fixed N). `tests/run_all.py`: 17 passed. Writer lacked nothing.
- Tested (WP21, 15/11/2026): compact layout now also packs per-keyframe `sh`: `sh_lo [N,S,3]` + `sh_step [N,S,3]` float32 + `sh_u8 [K,N,S,3]`
  (`_PackedSH` in format.py; every coefficient of every Gaussian quantised over its own range in time, `sh = sh_lo + sh_u8 * sh_step`); constant `sh` stays
  float32; non-finite `sh` falls back to float32. `save`, `Sequence4DWriter` (bit for bit equal), lazy load, `GpuFrames`, static group all handle it;
  old compact files (float32 `sh`) still load; files with `sh_u8` cannot be read by older code. `tests/test_compact.py::test_sh_per_keyframe`, `tests/test_writer.py`.
  Native 4DGS `--sh --compact`: `slice-banana` 330 kf 19.86 -> 5.82 GB, PSNR mean 24.2687 both (worst image -0.0002 dB); `bouncingballs` 60 kf 1412 -> 453 MB,
  mean 32.9117 both (worst -0.0004 dB). Play frame, GPU path 2.2 -> 1.4 ms (banana), CPU path (`--cpu-decode`) 60 -> 128 ms (numpy decode of `sh`). Step 0: a
  coefficient's range in time (median 0.03) is far below its magnitude (DC up to 6.8), so a global int8 scale is ~300x worse (rms 1.6e-2 vs 5e-5).
  The writer's scratch files are still float32 (25 GB for banana 330 kf). `gs4d/open.py::estimate_bytes` does not know about per-keyframe `sh` (CC-B's file, not changed).
- Tested (WP19c part 3, 15/11/2026): the 1.9 dB gap of native 4DGS on `slice-banana` (ours 24.23, repo 26.14) is a bug in the repo's camera path for OUR Blender layout, not the converter: per-frame `fl_x/fl_y/cx/cy` leave `FovX = FovY = -1`, so the CUDA rasteriser's 2D covariance uses focal `W/(2 tan 0.5)` and clamps x/z to -0.71 (min > max), and the model learned to compensate. Emulating that in gsplat (`tests/diag_native4d_banana.py --steps c_focal`) gives mean 26.13 dB. No effect: keyframes at the 33 test times 24.27, no compact 24.27, `marginal_t` cut 24.29, near plane 0.2 24.27, SH direction from un-offset xyz 24.27; SH degree 0 / 1 only 22.38 / 23.05. `slice_native4d` matches the CUDA `computeCov3D_conditional` to 1e-15. No converter change. The diag script needs one process per step (110 keyframes with `--sh` get OOM-killed).
- Tested (WP20, 15/11/2026): native 4DGS `slice-banana` retrained with correct cameras: `tests/make_hypernerf_blender.py` now writes top-level `camera_angle_x` and no per-frame `fl_x/fl_y/cx/cy` (those left FovX = FovY = -1 in the repo). Same config (20k iterations, 20 min, `configs/banana_fov.yaml`, data `/home/cont/gs4d_train/data/banana_blender_fov`, output `4dgs/output/banana_fov`): repo eval 26.30 dB; `convert_native4d.py --frames 110 --sh --compact` + gsplat 21.41-29.17 dB, mean 26.31 on 33 held-out frames at 268 px (old model 24.23). Gap closed; no converter change; `check_real.py` item `native4d_banana` uses the new checkpoint (threshold min 20.91).
- Tested (WP23, 15/11/2026): hustvl 4DGaussians on the real HyperNeRF `slice-banana` (repo's own HyperNeRF reader, `arguments/hypernerf/banana.py`, 14k iterations, 18 min,
  52,741 Gaussians; data folder `/home/cont/gs4d_train/data/banana_4dgaussians` = symlinks + the 297/33 `dataset.json` + `points3D_downsample2.ply` written from the RAW
  `points.npy`; clone patched `Load_hyper_data(datadir, 0.5 -> 0.25)` to read the 4x images). `convert_4dgaussians.py --frames 330 --sh --compact`: 325 MB, 35 s, peak RAM 1.5 GB;
  PSNR 22.05-30.39 dB, mean 27.13 on the 33 held-out frames at 268 px; the repo's own renders of the same frames 27.13 (per image within 0.003 dB). No converter change;
  `check_real.py` item `4dgaussians_banana` (threshold min 21.55). This repo keeps HyperNeRF cameras and points in RAW coordinates (no centre/scale, unlike Deformable-3DGS / SC-GS)
  and time = warp_id / max; its FovX/FovY come from the focal length and are correct. `gs4d.open` writes the camera list for this layout since WP27.
- Tested (WP19b, 15/11/2026): display speed in a real GPU Chrome (157, headed, ANGLE on the RTX 3060, 120 Hz screen), `tests/bench_display.py`, `paper-windmill` compact,
  1280x720, playing: update rate 20 / 30 / 40 / 50 / 60 Hz -> server sends 20.0 / 29.9 / 39.9 / 49.5 / 54.3 frames/s, browser shows 20.0 / 30.0 / 39.9 / 49.3 / 52.9 new frames/s
  (counted as full-size WebGL texture uploads), requestAnimationFrame 117-120/s. The browser shows every frame; the server is the limit. A locked screen or a training job on
  the GPU invalidates the run (rAF drops to 1/s; sent rate to ~31). Copying the canvas per frame to compare pixels halves the browser rate: do not measure that way.
  NOT measured: 1920x1080, a browser on another machine over SSH.
- Tested (WP24, 16/11/2026): Display folder has "Draw as" `splats` / `points` (`--draw-as`) and "Point size"; one dot per Gaussian centre in the Gaussian's
  colour, "Hide opacity below" applies, works while playing, with `--compare` and mp4 export. gsplat: `_raster` rasterises opaque isotropic Gaussians with
  sigma = size_px / 2.355 * depth / focal (constant size on screen, SH colour kept); client: `scene.add_point_cloud` (size in scene radius / 1000, base colour,
  re-added when the count changes). The ground-truth view always draws splats. `tests/check_points.py [--file f.npz --name n --up=-y --moving]`: toy, SC-GS
  `bouncingballs_sh`, `paper-windmill` pass in headless Chrome; server time per played frame at 1280x720: SC-GS 10-11 ms (splats 12-15), windmill 24-27 ms
  (splats 27-40). NOT measured: shown frames per second in a GPU Chrome in points mode; client point cloud on a real GPU. `tests/run_all.py` has no `NEEDS`
  entry for `check_points.py` (needs Chrome; CC-B's file, not changed).
- viser 1.1.1 camera controls (tried in Chrome): left drag orbits, right drag pans, wheel / middle drag zooms, W/S forward/back (horizontal), A/D left/right,
  Q/E down/up along the up axis, arrow keys orbit.
- Phase 3 (16/11/2026, PLAN.md "Phase 3"), all done on branch `cc-a/wp24-points`, one session:
  - WP25: `tests/run_all.py` `NEEDS["check_points.py"] = ("chrome",)`.
  - WP26: `gs4d/open.py::estimate_bytes` counts `--sh` (native 4DGS: 192 B per Gaussian per keyframe; others: 192 B per Gaussian once), `tests/test_open.py`.
  - WP27: `gs4d/open.py::nerfies_scene` / `write_nerfies_cameras`: for a 4DGaussians, Deformable-3DGS or SC-GS folder whose `cfg_args` `source_path` has the
    Nerfies / HyperNeRF layout, the held-out cameras (`val_ids`, else all) are written to `<name>.gs4d.cameras.json` beside the output and passed as `--cameras`
    (4DGaussians: raw positions, time = warp_id / max; the other two: `(position - center) * scale`, time_id / max; smallest `rgb/<N>x`, no lens distortion).
    Real check on `slice-banana`: `Viewer.gt_compare` over the generated list for 4DGaussians gives 22.05-30.39, mean 27.13 dB (= `check_real.py`); cameras of
    all three models equal `check_real.load_nerfies_val_set` to 3e-7. `tests/test_open.py` has a synthetic layout test.
  - WP19c part 2: Spacetime lite `n3d_sear_steak` against the Neural 3D Video `sear_steak` videos (`../data_real/spacetime/n3d_sear_steak_video/sear_steak/camNN.mp4`,
    1.11 GB from the GitHub release): 21 cameras x frames 0, 12, 25, 37, 49 at 1352x1014, time = frame / 50: 26.18-33.53 dB, mean 31.43; `check_real.py` item
    `spacetime_psnr` (these numbers are from before the colour fix of WP31, see below). The model's `cameras.json` is the 3DGS layout (position, camera-to-world rotation, fx at full width; one entry per camera and frame).
  - WP19c part 1: Shape of Motion `paper-windmill` against DyCheck iPhone 1x images (720x960; `../data_real/som/paper-windmill_data/paper-windmill/rgb/1x/0_*.png`
    + `splits/train.json`, extracted from a 6.61 GB zip of the HF mirror `sibicatleychandar/iphonedataset-shapeofmotion`; the zip is kept beside it), training
    cameras from the checkpoint, every 4th frame: at keyframes mean 25.86 dB (min 24.28), half way between `--stride 8` keyframes 24.67 (min 21.61); with
    `--stride 1` 25.85 everywhere, so the drop is the linear interpolation. Item `som_psnr` (threshold 21.11). No number from the repos' own renderers for either.
  - WP29 (16/11/2026): ground-truth view works with `--compare`: `_gt_view` shows render A | image | render B in mode `A | B` (two panels in mode A or B),
    PSNR of both in the panel; `gs4d/open.py` passes `--cameras` also with `--compare`. `tests/check_cameras.py` checks it (PSNR equal to the single-file viewer).
  - WP30 (16/11/2026): `convert_scgs.py` reproduces the SC-GS colour head (`--pred_color`): d_color per node (default head: Linear-ReLU-Linear-ReLU-Linear on
    [position emb, time emb, hidden]; or one Linear on hidden), blended with the node weights, times the motion mask, ADDED TO THE SH DC coefficient; `colors` (and `sh`
    with `--sh`) are then per keyframe. `--ignore-pred-color` keeps them constant. Synthetic test only proves the maths: the self-trained `bouncingballs` model
    (`/home/cont/gs4d_train/out/scgs_predcolor_bouncingballs_node`, 20k iterations) has a head still at its initialisation (|d_color| <= 2.3e-4), PSNR 35.62-43.75,
    mean 40.70 with or without the head; item `scgs_predcolor`.
  - WP31 (16/11/2026): `tests/repo_psnr.py` runs the methods' OWN renderers on check_real's images (clones `/home/cont/gs4d_train/som`, `stg`; extra packages in
    `site_som` (roma, installed with `--no-deps`: plain `pip install roma` downloads torch again, 8 GB) and `site_stg` (forward_lite built after adding
    `#include <cstdint>` to `rasterizer_impl.h`, plus an `mmcv/ops.py` stub)). Shape of Motion: repo 25.85 dB = ours with `--stride 1` to 0.000 dB per image.
    Spacetime lite `n3d_sear_steak`: repo 27.79-38.20, mean 34.48; ours was 31.43 because the converter clipped per-Gaussian colours to [0, 1].
  - Spacetime Gaussians stores RAW colours: 30 % of the Gaussians of the lite model have f_dc outside [0, 1] (-5 .. 6.9); the repo blends them and clamps the pixel.
    `convert_spacetime.py` no longer clips (now 27.76-37.81, mean 34.37; the remaining 0.11 dB is `--min-peak-opacity 0.02`, with 0 the match is 0.01 dB).
    `colors` in the format may therefore lie outside [0, 1]: the gsplat path passes them on, the client renderer clips, compact keeps per-keyframe colours
    float32 when any is out of range. `gs4d.open` reconverts when the converter script is newer than the cache. `spacetime_psnr` threshold is now 27.26.
  - WP28 done 17/11/2026 (`tests/bench_display.py`, GPU Chrome, `paper-windmill` compact, playing, points mode): 1280x720, update rate 20/30/40/50/60 Hz -> sent 20.0/29.9/39.9/48.7/52.3, shown 20.0/30.0/39.8/48.2/49.5 per s. 1920x1080: 20 Hz -> sent 20.1, shown 20.0; 60 Hz -> sent 38.8, shown 34.2 (server is the limit). Not measured: a browser on another machine, trails on.
  - Converting Deformable-3DGS `slice-banana` with `--frames 330` takes more than 10 minutes on this machine.
- NOT tested: the full Spacetime Gaussians model (with `rgbdecoder`); the client render of
  `basketball` against the original `visualize.py`.
- Local training setup for those methods (not in the repo): `/home/cont/gs4d_train` (env, CUDA toolkit, clones,
  outputs), data in `../data_real`. Build or train one job at a time: the machine has 15 GB RAM.
- Done 16/11/2026 (synthetic only): SC-GS `--use_hash` (numpy tiny-cuda-nn hash grid, matches tinycudann to 2e-4 with `PYTHONPATH=/home/cont/gs4d_train/site_tcnn LD_LIBRARY_PATH=/home/cont/gs4d_train/cuda/targets/x86_64-linux/lib`; `tests/test_scgs_hash.py`) and `--skinning` (`tests/test_scgs.py`). No real model: the repo's own hash training stays at 6.8 dB on `bouncingballs`, its skinning training crashes after densification (`motion_mask` size). `check_real.py` items `scgs_hash`, `scgs_skin` run only with `--only`. `--pred_color` verified on a time-tinted `bouncingballs` (`scgs_predcolor`: 37.83 dB with the head, 24.48 ignoring it).
- Tested (WP34, 16/11/2026, branch `cc-a/wp34-path`): camera path in `viewer.py` ("Camera path" folder, `--camera-path`; `path_pose`, `orbit_keys`, `load_path`/`save_path`): keys from the clicking client's camera,
  Catmull-Rom positions + slerp + linear fov (closed or open), Play / scrub, "Scene time on the path" hold or run, mp4 along the path (Export video -> Camera). `tests/check_path.py`: equals an independent
  Hermite / exp-log computation to 4e-16, browser camera follows it, mp4 frame count; `--orbit-mp4 f.npz --name n` wrote `shots/path/*.mp4` (SC-GS 300 frames in 5 s, windmill 10 s). `python -m gs4d.open <output> --camera-path p.json` passes the flag on like any viewer argument; it does not look for a saved path by itself.
- Tested (WP35, 16/11/2026): "Motion" folder: Colour by speed / displacement (`Sequence4D.speed_at`, `segment_speed`, `displacement_at`, `keyframe_means`; colour scale in `viewer.colour_lut`, replaces SH colours; `_frame(..., plain=True)` for ground truth) and trails (`viewer.TrailData`:
  weighted random ranking by path length among opaque moving Gaussians, positions at 24 sample times; viser line segments over the image, no depth test). Only for `interpolate=True`. `tests/check_trails.py` (hand-worked numbers, compact file with static group, both renderers,
  screenshots in `shots/trails/`). 500 trails: 0.5 ms server side per frame; shown FPS in a GPU Chrome NOT measured.
- Not done: view-direction colour heads, `d_rot_as_res` off, Spacetime full model.
- Tested (WP17): `--compact` in every `convert_*.py` (only `save(compact=True)`); `gs4d.open` adds it when the estimate
  (`estimate_bytes`, rough: ply size / 248 B x frames x 40 B) is over 1 GB, `tests/test_open.py`. Deformable-3DGS `bouncingballs --sh`
  153 -> 84 MB, `at(t)` differs from the plain file by <2e-4. `python tests/run_all.py` all pass on this machine (CUDA, Chrome, data present).
  `requirements.txt` now lists numpy<2, viser, pillow, imageio, imageio-ffmpeg, playwright (torch, gsplat as comments).
- Tested (20/11/2026, branch `cc-b/open-new-converters`): `gs4d/open.py` recognises the outputs of the three converters of 17/11.
  Before, an E-D3DGS folder went to `convert_4dgaussians.py` (KeyError `deformation_net.grid.aabb`), a 4D-Rotor `step-*.ckpt` to
  `convert_shape_of_motion.py`, and MoSca was not recognised. Now a `.ckpt` / `.pth` (other than `chkpnt*.pth`) is classified by the
  state-dict key names in its pickle, read without torch (`data.pkl` of the zip, or the start of a legacy file): `quats1` + `quats2` +
  `scales_t` = 4D-Rotor, `fg.params.means` + `motion_bases.params` = Shape of Motion, `scf._node_xyz` + `attach_ind` = MoSca; when the keys
  cannot be read, by the name (`step-*.ckpt` 4D-Rotor, other `*.ckpt` Shape of Motion, `photometric_d_model_*.pth` MoSca). Folders:
  `nerfstudio_models/step-*.ckpt` (the last step), a MoSca log folder (the newest `photometric_d_model_*.pth`; the static
  `photometric_s_model_*.pth` is refused with a hint), and a 4DGaussians layout whose ply has `embedding_*` columns = E-D3DGS (no time
  range or cameras passed). `estimate_bytes` has rough rules for the three. `tests/test_open.py`: fake checkpoints without torch, plus
  files written by `torch.save` (zip and legacy) when torch is installed. `run_all.py`: 15 passed / 10 skipped without torch, 22 / 3 with
  CPU torch (no CUDA here); `python -m gs4d.open` converted and opened synthetic E-D3DGS, 4D-Rotor (Nerfstudio run folder) and MoSca
  (log folder) outputs built with the converters' test fixtures. Still no real model of any of the three.

## 23/11/2026 — review and polish (branch `polish/code-and-docs`)
- Fixed: `Sequence4DWriter` decided uint8 colours on all Gaussians while `save` decides on the per-keyframe ones, so a
  static Gaussian with a colour outside [0,1] made the two files differ; now tracked per Gaussian
  (`tests/test_writer.py::test_colors_out_of_range`). `convert_ed3dgs.py` clamps per-keyframe colours at 0 like the renderer
  (`tests/test_ed3dgs.py` now has negative colours). `python -m gs4d.open` exits quietly on Ctrl+C (code 130) and removes
  a half-written `.partial.npz`. `viewer.py`, `python -m gs4d` and the converters stop with one line instead of a traceback
  on a missing / wrong file, missing torch or a malformed flag (`tests/test_cli.py`, `gs4d/checkpoint.py`).
- `tests/run_all.py`: `test_4dgaussians.py` and `test_native4d_sh.py` are skipped (not "passed") without torch; the
  check scripts write their screenshots to a temporary folder unless `--keep-shots`.
- `GpuFrames` runs on any torch device (pinned memory only on CUDA): `tests/test_gpu_frames.py` compares it with
  `Sequence4D.at` on the CPU for every layout (largest relative difference 1.1e-7). Still not re-run on the GPU here.
- Every converter output of the synthetic tests is bit for bit the one of the code before the polish (33 files compared).
- CI: `ruff check .` (rules in `ruff.toml`) and a second test job with CPU torch, so the converter tests run on GitHub.
- README rewritten as a user guide; measurements moved to `docs/SUPPORTED.md`, `docs/FORMAT.md`, `docs/PERFORMANCE.md`.

## 23/11/2026 — documentation in English (branch `docs/english`)
- PR #2 (the polish above, plus Thai's commit that makes the tests run on Windows) is merged into `main`.
- `README.md`, `PLAN.md` and `docs/` (`SUPPORTED.md`, `FORMAT.md`, `PERFORMANCE.md`, `ADD_A_CONVERTER.md`) translated from
  Vietnamese; numbers now use English separators (98,993 Gaussians; 26.31 dB). Checked by script: the digits of every file
  are the same as in the Vietnamese version (`PLAN.md` only gains the new dates and notes below), no Vietnamese letters left
  in any tracked file, all relative links and `#anchors` resolve, every table row has as many cells as its header.
- `PLAN.md`: the WP28 row says done 17/11/2026 (it was measured then, see above), finished "Do now" items are struck
  through, and the prompts' "write the numbers in the README" now points to `docs/`.
- `shots/readme_hero.png`: the right-hand panel is cut from `shots/cameras/ground_truth_compare.png`, so the browser's
  "Software WebGL rendering detected" toast and the control panel no longer cover the original image.
- How GitHub shows the docs: rendered with cmark-gfm, the parser GitHub uses (for the files on `main` it gives the same
  structure and text as GitHub's own pages; the 103 heading ids there follow the anchor rule of the link check), then shown
  inside the real GitHub pages at a 1280 px window (README column 838 px, other files 861 px). Fixed: three tables wider
  than the page (README "Supported methods": converter name only, full commands in `docs/SUPPORTED.md`; the summary table
  of `docs/SUPPORTED.md`; the WP9 table of `PLAN.md`) and long code comments that were cut off. Still scrolling sideways:
  the gsplat install line, a few long converter commands and the prompt blocks of `PLAN.md` (kept on one line to copy).
