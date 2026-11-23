# gs4d_viewer — view 4D Gaussian Splatting results over time

[![tests](https://github.com/zerokhong1/gs4d_viewer/actions/workflows/test.yml/badge.svg)](https://github.com/zerokhong1/gs4d_viewer/actions/workflows/test.yml)

A browser viewer (built on [viser](https://viser.studio)) for the results of 4D Gaussian Splatting (4DGS) methods after training: drag a
time slider, play/pause, orbit the camera around the scene, draw the Gaussians as splats or as points, compare two results, check a
render against the original image with its PSNR, colour Gaussians by speed, and record a video along a camera path.

The viewer reads **one common `.npz` format** ("Gaussians per keyframe"). Every 4DGS method has a converter that writes this format;
`python -m gs4d.open <output>` recognises the kind of output, converts it and opens the viewer in one command. The viewer never imports a
method's code.

![Toy scene over time, motion trails, and an SC-GS render next to the original image with its PSNR](shots/readme_hero.png)

_Left to right: the synthetic scene (`make_toy.py`) with the gsplat renderer; motion trails of the Gaussians; a self-trained SC-GS model
of the D-NeRF `bouncingballs` scene, rendered next to the original image of a test camera with its PSNR._

## Features

- Time slider, play/pause, speed, loop, step by keyframe; linear interpolation between keyframes.
- Two renderers: **gsplat** (the NVIDIA GPU of the machine that runs the viewer, for scenes with millions of Gaussians) and **client**
  (WebGL in the browser, no CUDA needed).
- Draw Gaussians as **splats** or as **points** (one point at each Gaussian's centre), hide them by opacity, scale them.
- **Compare** two results on one time slider: show A, B or both side by side, with the same camera.
- **Training cameras**: jump to the camera and the time of an original image; the gsplat renderer shows the render next to the image
  with its **PSNR**.
- **Motion**: colour by speed or displacement, draw the paths (trails) of the Gaussians.
- **Camera path**: place keys, play them back, save / load the path; **export mp4** from a fixed camera or along the path.
- Large files: compact storage (2–3x smaller), lazy loading (only the keyframes on screen are read), decoding on the GPU while playing.
- 11 4DGS methods: 8 have been run on real outputs (6 of them with PSNR against the original images, matching the method's own renderer
  wherever that comparison could be made); 3 have synthetic tests only ([docs/SUPPORTED.md](docs/SUPPORTED.md)).

## Installation

Client renderer only (runs on any machine, no GPU needed):

```bash
pip install -r requirements.txt
```

gsplat renderer (NVIDIA GPU; no CUDA toolkit needed; the prebuilt gsplat wheels exist for Python 3.10 only):

```bash
uv venv --python 3.10 --seed .venv
uv pip install --python .venv/bin/python torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python "gsplat==1.5.3+pt24cu124" --find-links https://docs.gsplat.studio/whl/gsplat/ --index-strategy unsafe-best-match
```

The converters of methods that save torch checkpoints also need torch (the CPU build is enough): `pip install torch`.

## Quick start

```bash
# synthetic scene: static disc, orbiting ball, bouncing ball, cloud that fades in and out
python make_toy.py -o toy.npz
python viewer.py toy.npz             # then open http://localhost:8080
```

With the real output of a method:

```bash
# a model folder, checkpoint, ply, ... (see the table below)
python -m gs4d.open output/dnerf/bouncingballs
python -m gs4d.open output/dnerf/bouncingballs --renderer gsplat --port 8081 --draw-as points
```

## Opening an output directly: `python -m gs4d.open`

```bash
python -m gs4d.open <output_path> [--reconvert] [--dry-run] [--convert-args "--stride 4"]
                    [viewer.py arguments]
```

- Recognised (checked in this order, the first match wins, nothing is guessed): an `.npz` already in the common format (opened as it
  is); `params.npz` or a folder of per-frame ply files; `chkpnt*.pth`; other `.ckpt` / `.pth` files by the key names in the checkpoint
  (the pickle is read without torch; when it cannot be read, by the file name): `step-*.ckpt` of 4D-Rotor GS, `last.ckpt` of Shape of
  Motion, `photometric_d_model_*.pth` of MoSca; a folder with `checkpoints/last.ckpt`, `nerfstudio_models/` (the last step) or a MoSca
  log; a folder with `point_cloud/` + `deform/` (SC-GS when the ply has `fea_*` columns, else Deformable-3DGS); a folder with
  `point_cloud/iteration_*/deformation.pth` (E-D3DGS when the ply has `embedding_*` columns, else 4DGaussians); the `point_cloud.ply`
  of Spacetime Gaussians.
- The converted file is `<name>.gs4d.npz` next to the output and is reused while the output and the converter are unchanged;
  `--reconvert` converts again, `--dry-run` only prints the commands. When the estimated file is larger than 1 GB the conversion uses
  `--compact`.
- D-NeRF scenes (`cfg_args` says dnerf / is_blender) get the time range [0, 1].
- Cameras: unless you pass `--cameras` yourself, a readable `cameras.json` (next to the output or up to 3 folders above it) or the
  `transforms_test.json` of the dataset named in `cfg_args` is passed to the viewer. For Nerfies / HyperNeRF datasets (4DGaussians,
  Deformable-3DGS, SC-GS) the held-out cameras are written to `<name>.gs4d.cameras.json` next to the output.
- An output that is not recognised prints the checks that were made and exits with code 2. Ctrl+C stops the viewer and a running
  converter.

## Supported methods

| Type | Method | Input | Converter | Checked on a real output |
|---|---|---|---|---|
| A | Gaussians per frame (feed-forward models; the per-frame export of 4DGaussians) | folder of `*.ply` | `convert_per_frame.py` | yes |
| A | Dynamic 3D Gaussians | `params.npz` | `convert_per_frame.py` | yes (no original images) |
| B | Native 4DGS (Yang et al., ICLR 2024) | `chkpnt*.pth` | `convert_native4d.py` | yes |
| B | 4D-Rotor GS (SIGGRAPH 2024) | `step-*.ckpt` (Nerfstudio) | `convert_4drotor.py` | no, synthetic test only |
| C | Spacetime Gaussians (lite model) | `point_cloud.ply` | `convert_spacetime.py` | yes |
| C | Shape of Motion | `checkpoints/last.ckpt` | `convert_shape_of_motion.py` | yes |
| C/D | MoSca (CVPR 2025), dynamic Gaussians only | `photometric_d_model_*.pth` | `convert_mosca.py` | no, synthetic test only |
| D | 4DGaussians (hustvl) | model folder | `convert_4dgaussians.py` | yes |
| D | Deformable-3DGS | model folder | `convert_deformable3dgs.py` | yes |
| D | SC-GS | model folder `<name>_node` | `convert_scgs.py` | yes |
| D | E-D3DGS (ECCV 2024) | model folder | `convert_ed3dgs.py` | no, synthetic test only |

`python -m gs4d.open` runs the right converter for you. By hand: `python <converter> <input> -o scene.npz [flags]`, for example
`python convert_native4d.py chkpnt_best.pth -o scene.npz --frames 60 --t0 0 --t1 1 --sh`; the command of every method is in
[docs/SUPPORTED.md](docs/SUPPORTED.md).

Types: **A** Gaussians per frame · **B** 4D Gaussians sliced in time · **C** parametric motion functions · **D** deformation networks,
re-run on the CPU. PSNR per method, the variants that were tried, the variants a converter refuses and why:
[docs/SUPPORTED.md](docs/SUPPORTED.md). Adding a method: [docs/ADD_A_CONVERTER.md](docs/ADD_A_CONVERTER.md) (with a working
`convert_template.py`).

Flags shared by the converters: `--frames` number of keyframes (types B, D); `--t0 --t1` time range in the method's unit (D-NeRF
scenes: [0, 1]); `--fps` and `--stride` for per-frame outputs; `--sh` keeps view-dependent colour (used by the gsplat renderer only,
bigger file); `--compact` writes the compact layout. `python <converter> --help` lists every flag.

## Using the viewer

```bash
python viewer.py scene.npz [--renderer auto|gsplat|client] [--up +z] [--port 8080]
                 [--compare b.npz] [--cameras transforms_test.json]
```

Scenes from COLMAP usually need `--up -y`; Blender / D-NeRF scenes use `+z` (the default).

### Renderers

- `--renderer gsplat`: rasterises with gsplat on the NVIDIA GPU of the machine that runs the viewer and streams the image to the browser.
  For real scenes (hundreds of thousands to millions of Gaussians). View-dependent colour (`sh`) exists only in this renderer.
- `--renderer client`: sends the Gaussians to the browser, which draws them with WebGL. No CUDA needed; suits a laptop and small scenes;
  limited to the 300,000 most opaque Gaussians by default (`--max-splats`).
- `--renderer auto` (default): gsplat when CUDA is available, else client.

Measured speed: [docs/PERFORMANCE.md](docs/PERFORMANCE.md) (gsplat at 1280×720 about 14 ms per frame for a small scene; fetching one
frame of a scene with 1 million Gaussians while playing takes 3 ms).

### Moving the camera

| Action | Result |
|---|---|
| Left drag | orbit around the point you look at |
| Right drag | pan the camera sideways / up and down |
| Mouse wheel, or middle drag | move closer to / away from the point you look at |
| `W` / `S` | forward / backward, horizontally |
| `A` / `D` | left / right |
| `Q` / `E` | down / up along the scene's up axis (`--up`) |
| Arrow keys | orbit around the point you look at |

Keys work only while the pointer is over the scene (not over the control panel).

### Comparing two results: `--compare`

```bash
python viewer.py ours.npz --compare baseline.npz
```

Both files share the time slider, the play button and the camera. **Show** selects `A`, `B` or `A | B` (side by side, the default);
switch between `A` and `B` from the same viewpoint to spot differences. The gsplat renderer splits the screen, each half drawing one file
from the same camera; the client renderer places B to the right of A in the same scene, with a file-name label. The time slider spans
from the earliest to the latest keyframe of both files; both files must share the coordinate frame and the up axis. Screenshots:
`shots/compare/`.

### Training cameras and original images: `--cameras`

```bash
python viewer.py bouncingballs.npz --cameras ../data/dnerf/bouncingballs/transforms_test.json
```

`--cameras` takes a `transforms_*.json` (NeRF / D-NeRF / instant-ngp, OpenGL cameras) or a `cameras.json` (`{"convention":
"opencv"|"opengl", "frames": [{"c2w": [[4x4]], "fx" or "fov_x"/"fov_y", "w", "h", "time", "image", "name"}]}`). The **Cameras** folder in the
control panel lists the cameras; selecting one moves the view to it and the time slider to the time of its image.

With the gsplat renderer, **Ground truth beside render** shows the render next to the original image of the selected camera, with its
**PSNR** (same recipe as `tests/check_real.py`; the two Display sliders do not apply, so the number is the model's own). Set **Max render
height** to the resolution the model was trained at (for example 400 for SC-GS and native 4DGS on D-NeRF), and **Image background** to
white for models trained on a white background (4DGaussians): a different resolution costs several dB of PSNR. With `--compare` it shows
three panels, **render A | original image | render B**, with the PSNR of both. Screenshots: `shots/cameras/`.

### Points

**Display** → **Draw as**: `splats` (the Gaussians as usual) or `points` (one point at each Gaussian's centre, in the Gaussian's colour),
and the **Point size** slider. Open directly in points mode: `--draw-as points`. Works while playing, with `--compare`, during video
export, in both renderers; **Hide opacity below** still applies. The gsplat renderer draws every point as an opaque round Gaussian whose
diameter is **Point size** pixels at any depth (near points hide far ones in the right order, view-dependent colour stays correct); the
client renderer uses viser's point cloud (point size in scene units). Screenshots: `shots/points/`.

### Speed and motion trails

The **Motion** folder (only when the keyframes share Gaussian identities):

- **Colour by**: `colour` (the Gaussian's own colour), `speed` (magnitude of the velocity between the two neighbouring keyframes) or
  `displacement` (distance from the position at the first keyframe). Scale blue → green → yellow → red; the value shown as red is written
  in the panel (99th percentile over the whole sequence × "Scale maximum (%)").
- **Trails**: the paths of "Number of trails" Gaussians over a window of ± "Window" around the current time (blue = past, red = future),
  drawn at random with a weight that grows with path length, among Gaussians that are opaque enough. Trails are lines drawn over the
  picture: with the gsplat renderer they are not hidden by depth. Screenshots: `shots/trails/`.

### Camera path

The **Camera path** folder: move the camera, then press **Add key from my camera** to place a key (drawn as an orange camera frustum, with
the path in the scene), or **Replace by an orbit around the scene** to get a ready-made loop of 12 keys around the scene. Positions are
interpolated with a Catmull-Rom spline, orientations with slerp, the field of view linearly.

- **Play / stop path** runs the path over "Path duration (s)" and drives the camera of every connected browser; drag "Path position" to
  look at any point of it. **Scene time on the path**: stay at the current time, or run through the sequence while the camera moves.
- **Save path (json)** writes to `--camera-path` (or `<video-dir>/camera_path.json`) and downloads the file; load it again with
  `python viewer.py scene.npz --camera-path path.json`. Same convention as `--cameras`: `{"convention", "duration", "closed",
  "time": "hold"|"run", "frames": [{"c2w": [[4x4]], "fov_y": radians}]}`.

### Video export

The **Export video** folder: choose "Frames per second", "Video length (s)" and "Camera" (`this camera (fixed)` or `camera path`), then
press **Export mp4**. The file `<name>_<date_time>.mp4` is written to `--video-dir` (default: the current folder) on the machine that runs
the viewer, and downloaded in the browser. The gsplat renderer renders directly on the GPU (60 frames at 1280×720 in about 1 second); the
client renderer has to wait for the browser to draw every frame, which is much slower. While exporting, the viewer stops updating every
browser.

### Viewing from another machine

Viewer on the GPU machine, browser on another one: `ssh -L 8080:localhost:8080 <gpu_machine>`, then open `http://localhost:8080`.
Or run the viewer with `--host 0.0.0.0` to accept connections from the network (trusted networks only).

## File format

One `.npz` file with `times [K]`, `means [K,N,3]`, `covs [K,N,6]` (upper triangle of the covariance), `opacities [K,N]`, `colors [K,N,3]`
and optionally `sh`; a field without the K axis is constant over time. There is a compact layout (`--compact`, or
`python -m gs4d in.npz out.npz`), Gaussians that never change are stored once, and `Sequence4DWriter` writes one keyframe at a time when
the scene does not fit in memory. Details and measurements: [docs/FORMAT.md](docs/FORMAT.md).

```python
from gs4d import Sequence4D
# lazy: memory-maps the file, only the keyframes used are read
seq = Sequence4D.load("scene.npz", lazy=True)
# Gaussians at t = 0.5 (interpolated between two keyframes)
means, covs, opacities, colors = seq.at(0.5)
```

## Tests

```bash
# every tests/test_*.py and tests/check_*.py, one at a time;
# prints pass / skipped (with the reason) / failed
python tests/run_all.py
python tests/run_all.py open sh    # only the files whose name contains one of the words
ruff check .                       # lint, as in CI
```

- `tests/test_*.py`: CPU checks. Every converter runs on synthetic files in the method's real layout and is compared with an independent
  formulation (needs torch); the format, compact storage, the static group, `Sequence4DWriter`, `GpuFrames`, `gs4d.open`, the error
  messages of the commands.
- `tests/check_*.py`: open the viewer in headless Chrome and check screenshots (needs Playwright + Chrome; the gsplat parts need CUDA). The
  screenshots of a `run_all.py` run go to a temporary folder; run a script on its own (or `run_all.py --keep-shots`) to refresh `shots/`.
- `tests/check_real.py`: PSNR with thresholds on real outputs of every method (needs CUDA and the real data, see the top of the file).
- A file that lacks what it needs (torch, CUDA, Chrome, data) is reported as "skipped" with the reason, not as failed. GitHub Actions runs
  the lint, the tests without torch and the tests with CPU torch on every push.

## Repository layout

```
viewer.py                  the viewer (GUI, two renderers)
gs4d/format.py             the common format: Sequence4D, Sequence4DWriter, GpuFrames
gs4d/open.py               python -m gs4d.open: recognise, convert and open an output
gs4d/ply.py, checkpoint.py 3DGS ply reader; loading torch / checkpoints for the converters
convert_<method>.py        one converter per method; convert_template.py is the template
make_toy.py                synthetic scene to try the viewer
tests/                     tests (run_all.py runs them all) and benchmarks
docs/                      SUPPORTED.md, FORMAT.md, PERFORMANCE.md,
                           ADD_A_CONVERTER.md, HISTORY.md
shots/                     screenshots and videos, the evidence for each feature
CLAUDE.md, PLAN.md         context and plan for the Claude Code sessions
```

## Current limitations

- View-dependent colour needs the `sh` field (the converters' `--sh` flag) and the gsplat renderer; the client renderer always draws
  `colors`.
- Size ≈ N × K × 40 bytes for the time-varying part (≈ 19 bytes in the compact layout): lower `--frames` / `--stride`, or use `--compact`.
- Gaussian counts that change between frames (some feed-forward models): the viewer jumps to the nearest keyframe instead of
  interpolating, and the Motion folder is off.
- Unsupported variants (the full Spacetime model with `rgbdecoder`, SC-GS colour heads that depend on the view direction, ...) make the
  converter stop and say why; the list is in [docs/SUPPORTED.md](docs/SUPPORTED.md).

## License and data sources

The code in this repository is under the MIT license (see `LICENSE`).

The three images `shots/wp2/basketball_fg_up-y_t*.png` are **not** under the MIT license. They are rendered from the trained
`basketball` model of Dynamic 3D Gaussians, which is built from CMU Panoptic Studio recordings. The CMU Panoptic Studio dataset may be
used for non-commercial research only; these three images may only be used under the same terms.

- J. Luiten, G. Kopanas, B. Leibe, D. Ramanan. *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View
  Synthesis.* 3DV 2024. <https://github.com/JonathonLuiten/Dynamic3DGaussians>
- H. Joo, H. Liu, L. Tan, L. Gui, B. Nabbe, I. Matthews, T. Kanade, S. Nobuhara, Y. Sheikh. *Panoptic Studio:
  A Massively Multiview System for Social Motion Capture.* ICCV 2015.
  <https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox>

The images below combine renders of the viewer with images of the original datasets (the four images in `shots/real/`: top row
render, bottom row original image), so they are **not** under this repository's MIT license either:

- `shots/real/shape_of_motion_paper-windmill.png`: the `paper-windmill` scene of the iPhone set in DyCheck. The DyCheck repository
  releases both code and dataset under Apache 2.0. The checkpoint was published by the Shape of Motion authors (MIT repository).
- `shots/real/deformable3dgs_bouncingballs.png`, `scgs_bouncingballs.png`, `native4d_bouncingballs.png`: the synthetic `bouncingballs`
  scene of D-NeRF. The D-NeRF repository is MIT-licensed and states no separate terms for the dataset. We trained the models ourselves
  with the code of Deformable-3DGS, SC-GS and native 4DGS (all three repositories are MIT).
- The original-image panels of `shots/cameras/ground_truth*.png` and the right-hand panel of `shots/readme_hero.png`: also test images of
  the D-NeRF `bouncingballs` scene, same terms as above.

- H. Gao, R. Li, S. Tulsiani, B. Russell, A. Kanazawa. *Monocular Dynamic View Synthesis: A Reality Check.*
  NeurIPS 2022. <https://github.com/KAIR-BAIR/dycheck>
- A. Pumarola, E. Corona, G. Pons-Moll, F. Moreno-Noguer. *D-NeRF: Neural Radiance Fields for Dynamic Scenes.*
  CVPR 2021. <https://github.com/albertpumarola/D-NeRF>
- Q. Wang, V. Ye, H. Gao, W. Zeng, J. Austin, Z. Li, A. Kanazawa. *Shape of Motion: 4D Reconstruction from a Single
  Video.* ICCV 2025. <https://github.com/vye16/shape-of-motion>
- Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, X. Jin. *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic
  Scene Reconstruction.* CVPR 2024. <https://github.com/ingra14m/Deformable-3D-Gaussians>
- Y. Huang, Y. Sun, Z. Yang, X. Lyu, Y. Cao, X. Qi. *SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic
  Scenes.* CVPR 2024. <https://github.com/yihua7/SC-GS>
- Z. Yang, H. Yang, Z. Pan, L. Zhang. *Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D
  Gaussian Splatting.* ICLR 2024. <https://github.com/fudan-zvg/4d-gaussian-splatting>
