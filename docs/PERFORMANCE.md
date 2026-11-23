# Speed and memory

All measurements on the development machine: RTX 3060 12 GB, 15 GB RAM, NVMe SSD, torch 2.4.1+cu124, gsplat 1.5.3, browser on the same
machine (unless stated otherwise). The machine was shared with other sessions, so CPU-side numbers vary by about ±20%. The command used is
given next to each table.

## gsplat renderer: one frame

`toy.npz` (15,000 Gaussians × 60 keyframes), playing:

| Render resolution | Time per frame (rasterise + JPEG + send) | Frames/s sent with "Update rate" = 60 Hz |
|---|---|---|
| 1280×720 | 14 ms | 57 |
| 1920×1080 | 23 ms | 42 |

The default "Update rate" is 20 Hz, so by default the viewer sends 20 frames/s at both resolutions.

## Decoding and interpolation on the GPU while playing

Before, every frame while playing: `at(t)` decoded and interpolated with numpy, the covariances were unpacked to 3×3, opacity was
thresholded, and everything was copied to the GPU. Step 0 (`python benchmarks/bench_large.py`, 500,000 Gaussians × 100 keyframes, compact):
copying from the disk mapping 3 ms, decoding 41, interpolation 10, copy to the GPU 4; the whole frame 100 ms. The time goes into CPU
decoding and the numpy steps after `at(t)`, not into reading the disk.

Now `gs4d.GpuFrames` keeps the keyframes around the current time on the GPU (a cache of 3 keyframes per field, float32): the raw arrays
(float16 / int16 / uint8) are copied up one keyframe at a time when the slider enters a new keyframe interval, then decoded and
interpolated with torch. Constant fields and the static group are uploaded once at start-up. `viewer.py --cpu-decode` uses the old CPU path
for comparison.

Time to get the gsplat inputs of one frame while playing (`viewer.frame_cpu` / `viewer.frame_gpu`, mean of 31 frames across 2 keyframes):

| | CPU, before | GPU, after |
|---|---|---|
| 500,000 × 100, float32 | 63 ms | 2.3 ms |
| 500,000 × 100, compact | 100 ms (`at(t)` 48 ms) | 2.2 ms |
| `paper-windmill` (995,942 × 35), compact + static split | 214 ms (`at(t)` 11 ms) | 3.1 ms |
| `paper-windmill`, float32 + static split | 204 ms | 3.2 ms |
| `paper-windmill`, original 1271 MB file | 225 ms | 4.6 ms |

Renders differ from the CPU path by at most 1/255 per channel (0/255 on the compact `paper-windmill`; 1/255 on the random scene, from
float32 rounding). `tests/check_real.py` was run again, and also with `Sequence4D.at` replaced by `GpuFrames`: both give exactly the same
PSNR. Since 23/11/2026 `tests/test_gpu_frames.py` compares `GpuFrames` with `Sequence4D.at` on the CPU for every file layout (largest
relative difference 1.1·10⁻⁷).

Per-keyframe `sh` (native 4DGS `--sh --compact`), fetching a frame while playing (`python benchmarks/bench_large.py --file`):

| | `slice-banana` 330 keyframes, `sh` float32 | `sh` uint8 | `bouncingballs` 60 keyframes, `sh` float32 | `sh` uint8 |
|---|---|---|---|---|
| GPU path (`GpuFrames`) | 2.2 ms | 1.4 ms | 1.1 ms | 0.8 ms |
| CPU path (`--cpu-decode`) | 60 ms | 128 ms | 29 ms | 49 ms |
| Conversion time / peak RAM | 4 min 58 s / 1.3 GB | 4 min 04 s / 1.5 GB | 10 s / 0.8 GB | 14 s / 0.8 GB |

The CPU path is slower because `sh` has to be decoded with numpy every frame; the gsplat renderer uses the GPU path by default, and the
client renderer does not use `sh`.

## Display in a GPU-accelerated Chrome

`python benchmarks/bench_display.py --rates 20 30 40 50 60` (needs an unlocked screen and a GPU that is not training): `paper-windmill` compact +
static split (995,942 Gaussians × 35 keyframes), playing, Chrome 157 on the same machine (WebGL through ANGLE/OpenGL), 120 Hz screen,
10 seconds per row.

`splats` mode, 1280×720 (15/11/2026):

| "Update rate" | Server sends (frames/s) | Browser shows new frames (frames/s) | Browser repaints (frames/s) |
|---|---|---|---|
| 20 Hz (default) | 20.0 | 20.0 | 120 |
| 30 Hz | 29.9 | 30.0 | 120 |
| 40 Hz | 39.9 | 39.9 | 119 |
| 50 Hz | 49.5 | 49.3 | 119 |
| 60 Hz | 54.3 | 52.9 | 117 |

`points` mode (17/11/2026, with `--draw-as points`, and `--size 1920 1080`):

| | 20 Hz | 30 Hz | 40 Hz | 50 Hz | 60 Hz |
|---|---|---|---|---|---|
| 1280×720: server sends / browser shows | 20.0 / 20.0 | 29.9 / 30.0 | 39.9 / 39.8 | 48.7 / 48.2 | 52.3 / 49.5 |
| 1920×1080: server sends / browser shows | 20.1 / 20.0 | | | | 38.8 / 34.2 |

The browser shows almost every frame the server sends; the server is the limit. "New frames" counts how often the page loads a full-size
image into a WebGL texture; the last column is the `requestAnimationFrame` rate. A locked screen or a training job on the GPU invalidates
the measurement (rAF drops to 1/s). Copying the canvas every frame to compare pixels halves the browser's rate: do not measure that way.
Not measured: a browser on another machine over SSH, and trails switched on.

## Points mode (Draw as: points)

Server-side time per frame while playing, gsplat renderer, 1280×720 (decode + rasterise + copy the image back to the CPU; printed by
`tests/check_points.py`; 2–3 runs, 16/11/2026):

| File | Gaussians | `splats` | `points` (3 px) |
|---|---|---|---|
| SC-GS `bouncingballs_sh.npz` | 165,900 × 60 keyframes | 12–15 ms | 10–11 ms (about 90 frames/s) |
| Shape of Motion `paper-windmill.npz` | 995,942 × 35 keyframes | 27–40 ms | 24–27 ms (about 40 frames/s) |

Points mode is not slower than splats. The client renderer has not been measured on a real GPU (the tests run software WebGL).

## Motion trails

500 trails, gsplat renderer: updating the trails takes 0.5 ms per frame (network transfer not included); decode + render 1280×720 + trails
17 ms (SC-GS `bouncingballs_sh`), 7 ms (4DGaussians `slice-banana`, 52,741 Gaussians). Client renderer on the toy scene: 0.5 ms for the
trails. Switching trails on for the first time reads every keyframe twice (`paper-windmill`, 277 keyframes: a few seconds).

## Video export

- Fixed camera, gsplat renderer: 60 frames of `toy.npz` at 1280×720 in about 1 second.
- Fixed camera, client renderer: every frame waits for the browser to draw and send the image back; 30 frames take 24 seconds with headless
  Chrome drawing in software.
- Along a camera path, gsplat, 1280×720, 300 frames: SC-GS `bouncingballs_sh` 5.2 s, `paper-windmill` 10.0 s; client renderer on the toy
  scene, 480×300, 20 frames: 8 s.

## Compact layout and lazy loading

`python benchmarks/bench_large.py`: 500,000 random Gaussians × 100 keyframes (means, covs, opacities change over time); the file was just
written, so it is still in the operating system's cache:

| | Float32, full load | Float32, lazy | Compact, full load | Compact, lazy |
|---|---|---|---|---|
| File size | 2006 MB | 2006 MB | 964 MB | 964 MB |
| Load time | 1.04 s | under 0.01 s | 0.50 s | under 0.01 s |
| RAM after loading and fetching 1 frame | 1963 MB | 89 MB | 980 MB | 94 MB |
| Fetching one interpolated frame (`at(t)`, CPU) | 14 ms | 17 ms | 77 ms | 77 ms |
| Save time | 2.2 s | 2.2 s | 13.5 s | 13.5 s |
| Largest position error | 0 | 0 | 0.002 | 0.002 |

The static group on real files (size, RAM, `at(t)`): [FORMAT.md](FORMAT.md#static-gaussians-stored-once).

## Converters: peak RAM and time

The converters write one keyframe at a time to disk through `Sequence4DWriter`, so peak RAM does not grow with the number of keyframes.
Measured with `/usr/bin/time -v`; the output equals the old code's (which built the whole sequence in memory) bit for bit in every array.

| Real input | Command | Peak RAM before → after |
|---|---|---|
| SC-GS `slice-banana` | `--frames 110 --sh --compact` | 4.5 → 1.4 GB |
| Native 4DGS `slice-banana` | `--frames 110 --sh --compact` | 8.3 → 1.3 GB |
| Deformable-3DGS `slice-banana` | `--frames 330 --sh --compact` | 7.2 → 1.0 GB |
| 4DGaussians `bouncingballs` | `--frames 60 --sh --compact` | 538 → 473 MB |
| 4DGaussians `slice-banana` | `--frames 330 --sh --compact` | 1.49 → 0.60 GB |
| Shape of Motion `paper-windmill` | `--stride 8 --compact` (35 keyframes) | 3.2 → 0.88 GB |
| Shape of Motion `paper-windmill` | `--compact`, all 277 frames | did not run (needs > 11 GB, crashed the app) → 0.88 GB, 69 seconds, 416 MB file |
| Spacetime `n3d_sear_steak` lite | `--compact` | 590 → 102 MB |
| Spacetime Technicolor `Birthday` lite | `--compact` | 3.28 → 0.44 GB |
| per-frame ply (4DGaussians `gaussian_pertimestamp`) | `--sh --compact` / no flags | 300 → 97 MB / 97 → 53 MB |

330 keyframes of `slice-banana`: SC-GS 3.5 minutes, 3.7 GB file; native 4DGS 5 minutes, 19.9 GB file (`sh` as float32; 5.8 GB with
compact `sh`); Deformable-3DGS more than 10 minutes. Spacetime reads the opacities twice (the first pass only computes the peak opacity for
`--min-peak-opacity`). Ply folders whose Gaussian count changes between frames are still built in memory (the writer needs a fixed N); the
Dynamic 3D Gaussians `params.npz` is loaded whole, but the covariances are written one frame at a time.
