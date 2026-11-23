# The common `.npz` format

The viewer reads one kind of file: "Gaussians per keyframe", class `Sequence4D` in `gs4d/format.py` (the docstring at the top of that file
is the most precise reference). Every converter writes this file; the viewer never imports a method's code.

## Fields

| Key | Shape | Meaning |
|---|---|---|
| `times` | `[K]` | keyframe times, increasing (unit depends on the method) |
| `means` | `[K,N,3]` or `[N,3]` | centres |
| `covs` | `[K,N,6]` or `[N,6]` | upper triangle of the 3×3 covariance: xx xy xz yy yz zz |
| `opacities` | `[K,N]` or `[N]` | in [0,1] |
| `colors` | `[K,N,3]` or `[N,3]` | view-independent RGB, normally in [0,1]. Values outside are kept and reach the gsplat renderer unchanged (Spacetime Gaussians blends raw colours and clamps only the final pixel); the client renderer clips them to [0,1] |
| `sh` (optional) | `[K,N,S,3]` or `[N,S,3]`, S = (degree+1)² | SH coefficients in the 3DGS convention: colour = max(SH(view direction) + 0.5, 0), `sh[...,0,:]` is `f_dc`. Used by the gsplat renderer only; `colors` must still be there |
| `interpolate` | bool | `False` when the keyframes do not share Gaussian identities (the viewer jumps to the nearest keyframe) |
| `source` | str | free text, shown in the viewer |

A field without the K axis is constant over time. Between two keyframes the viewer interpolates linearly (`Sequence4D.at(t)`).
The time-varying part takes ≈ N × K × 40 bytes as float32 (≈ 19 bytes in the compact layout).

## Writing a file

- The whole sequence fits in memory: build `Sequence4D(times, means, covs, opacities, colors, sh=...)`, then `.save(path)`.
- Long scenes: `Sequence4DWriter` writes one keyframe at a time, so peak memory does not depend on K. Constant fields go to the
  constructor, the others to `add_keyframe(t, ...)`; the file is bit for bit the one `save` writes, compact layout and static group
  included (`tests/test_writer.py`). Writing needs extra scratch disk space of ≈ N × K × 40 bytes next to the output (a `.gs4d_writer_*`
  folder, removed at the end; left behind if the process is killed), more with per-keyframe `sh`.

```python
from gs4d import Sequence4DWriter

# constant fields go to the constructor
with Sequence4DWriter("scene.npz", compact=True, colors=colors, source="my method") as w:
    for t in times:
        # fields that change with t
        w.add_keyframe(t, means=means_at(t), covs=covs_at(t), opacities=opacities_at(t))
```

Every converter writes through `Sequence4DWriter` (except ply folders whose Gaussian count changes between frames, since the writer
needs a fixed N). Peak RAM before and after the switch to the writer:
[PERFORMANCE.md](PERFORMANCE.md#converters-peak-ram-and-time).

## Compact layout and lazy loading

```bash
# re-save in the compact layout (slightly lossy), and back to plain float32
python -m gs4d scene.npz scene_small.npz
python -m gs4d scene_small.npz scene.npz --float32
```

In code: `seq.save(path, compact=True)`; the default is `compact=False` (float32, lossless). Every `convert_*.py` has a `--compact` flag;
`python -m gs4d.open` adds it when the estimated file is larger than 1 GB. `Sequence4D.load` reads both layouts and every older file.

The compact layout only changes the time-varying fields:

| Field | Stored as |
|---|---|
| `means` | `means_ref [N,3]` float32 (first keyframe) + `means_d16 [K,N,3]` float16 (offset from it) |
| `covs` | `covs_scale [N]` float32 + `covs_q [K,N,6]` int16: the Cholesky factor L (cov = L Lᵀ) divided by `covs_scale`, times 32767; always decodes to a positive semi-definite matrix |
| `opacities` | `opacities_u8 [K,N]` uint8 |
| `colors` | `colors_u8 [K,N,3]` uint8 (kept float32 when a value lies outside [0,1]) |
| `sh` | `sh_lo [N,S,3]` + `sh_step [N,S,3]` float32 + `sh_u8 [K,N,S,3]` uint8: every coefficient of every Gaussian is quantised over its own range in time, `sh = sh_lo + sh_u8 · sh_step` |

Fields without the K axis stay float32. The viewer loads lazily (`load(path, lazy=True)`, the file is memory-mapped): only the keyframes
on screen are read from disk, so **do not overwrite a file while the viewer has it open**.

Error, measured with `python tests/bench_large.py` on 500,000 random Gaussians × 100 keyframes: largest position error 0.002 for Gaussians
that move up to 5.35 units from the first keyframe (the error grows with the offset: float16 has 11 significant bits); opacity 0.002;
covariance 7·10⁻⁵ times the Gaussian's largest variance. File size 2006 → 964 MB; lazy loading takes under 0.01 s and 94 MB of RAM instead
of 1.04 s and 1963 MB when loading everything. In exchange, decoding a frame on the CPU is slower (77 ms instead of 14 ms); the gsplat
renderer decodes on the GPU and is not affected. Full table:
[PERFORMANCE.md](PERFORMANCE.md#compact-layout-and-lazy-loading).

### Per-keyframe `sh` in the compact layout

Native 4DGS with `--sh` writes `sh` per keyframe (`[K,N,16,3]`), so the file is almost all `sh`. A coefficient changes over time much less
than its magnitude, so quantising over each coefficient's own range in time is far more accurate than one shared scale. Distribution
measured on real files (12 evenly spread keyframes; "range in time" is the largest minus the smallest value of one coefficient of one
Gaussian):

| | Degree 0: value range | \|x\| at the 99th percentile | range in time, median / largest | Degrees 1–3: value range | \|x\| at the 99th percentile | range in time, median / largest |
|---|---|---|---|---|---|---|
| `bouncingballs` | −2.10 … 4.40 | 1.88 | 0.026 / 0.36 | −0.72 … 0.64 | 0.14–0.21 | 0.035–0.040 / 0.62 |
| `slice-banana` | −2.09 … 6.79 | 2.76 | 0.028 / 0.47 | −0.94 … 0.94 | 0.16–0.18 | 0.032–0.038 / 0.95 |

RMS error of the degree-0 coefficients on `slice-banana`: int8 with one scale per degree 1.6·10⁻²; int8 with one scale per coefficient of
every Gaussian 2.3·10⁻³; float16 2.3·10⁻⁴ (the file only halves); uint8 over each coefficient's own range in time 5.3·10⁻⁵ (degrees 1–3:
6–8·10⁻⁵, largest 1.8·10⁻³). `sh` with non-finite values stays float32.

On real native 4DGS outputs (`convert_native4d.py --sh --compact`, same checkpoint, before = `sh` as float32, after = uint8; PSNR with the
code of `tests/check_real.py`, lazy loading):

| | `slice-banana` 330 keyframes, before | after | `bouncingballs` 60 keyframes, before | after |
|---|---|---|---|---|
| Gaussians | 297,745 | 297,745 | 116,193 | 116,193 |
| File size | 19,858 MB | 5,824 MB (0.29×) | 1,412 MB | 453 MB (0.32×) |
| Mean PSNR (min – max) | 24.2687 dB (21.220–26.119), 33 images | 24.2687 dB (21.220–26.119) | 32.9117 dB (27.281–39.594), 20 images | 32.9117 dB (27.281–39.593) |
| Largest PSNR drop on one image | — | 0.0002 dB | — | 0.0004 dB |

(The `slice-banana` numbers here are those of the first native 4DGS model, trained before the camera fix; see
[SUPPORTED.md](SUPPORTED.md).) Frame fetch times of both layouts:
[PERFORMANCE.md](PERFORMANCE.md#decoding-and-interpolation-on-the-gpu-while-playing). While writing, `Sequence4DWriter` still needs
float32 scratch space for `sh` (25 GB for `slice-banana` with 330 keyframes).

## Static Gaussians stored once

When saving, `Sequence4D.save` finds the Gaussians that do not change over the keyframes in any field (they differ from the first keyframe
by at most 10⁻⁶ times the value's magnitude) and stores them once instead of K times; `save(path, split_static=False)` turns this off.
Callers do not have to know: `at(t)` still returns all N Gaussians in their original order. The file gains `static_index [Ns]` (positions of
the static Gaussians) and `<field>_static [Ns,...]`; the per-keyframe fields hold only the Nd = N − Ns other Gaussians.
`python -m gs4d old.npz new.npz --float32` converts an old file to the new layout without loss (drop `--float32` to compact it as well, add
`--no-split` to write the old layout).

Measured on real outputs (lazy loading, `at(t)` is the mean over 39 times between keyframes):

| | Shape of Motion `paper-windmill`, before | after | after + compact | Deformable-3DGS `bouncingballs`, before | after |
|---|---|---|---|---|---|
| Static Gaussians / total | — | 924,075 / 995,942 | 924,075 / 995,942 | — | 0 / 64,571 |
| File size | 1271 MB | 147 MB | 103 MB | 153 MB | 153 MB |
| RAM after a full (not lazy) load | 1248 MB | 177 MB | 135 MB | 181 MB | 181 MB |
| RAM after playing the whole time range (lazy) | 1248 MB | 205 MB | 161 MB | 166 MB | 167 MB |
| Fetching one interpolated frame (`at(t)`) | 34 ms | 8.7 ms | 10.9 ms | 1.7 ms | 1.6 ms |

`bouncingballs` has no static Gaussians, because the Deformable-3DGS deformation MLP moves every Gaussian a little. gsplat renders of the
file before and after: `bouncingballs` identical at all 3 times; `paper-windmill` (715 × 983, training camera) identical at keyframes,
while between two keyframes 0–7 of 702,845 pixels differ by more than 1/255 (largest 8.6/255): the old code interpolated the static
Gaussians with themselves, so their values were off by up to 5·10⁻⁷ from rounding; replacing those values with the originals makes the old
image equal the new one pixel for pixel.

## Compatibility between versions

`Sequence4D.load` reads every file earlier versions wrote. The reverse does not hold: older viewers cannot read files with a static group
(`static_index`, since 14/11/2026) or compact files with `sh_u8` (since 15/11/2026); convert with
`python -m gs4d old.npz new.npz --float32 --no-split` when an older version has to open them.
