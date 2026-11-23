# Method and validation

How gs4d_viewer converts the outputs of 4DGS methods, how the conversion is validated, and its limits. Installation and
usage: [README](../README.md) and [user guide](USAGE.md).

## Summary

Methods for 4D Gaussian Splatting (4DGS) describe a dynamic scene in incompatible ways: Gaussians stored per frame, 4D Gaussians
that are sliced in time, parametric motion functions, or canonical Gaussians displaced by a neural deformation field. Each method
ships its own renderer, so inspecting or comparing trained results requires the software environment of every method. gs4d_viewer
converts the trained output of each method into one representation, the Gaussians at a set of keyframes with linear interpolation
in between, by re-implementing the method's evaluation at time $`t`$ on the CPU from its published code. One renderer (gsplat [2],
or WebGL in the browser through viser [3]) then displays every method under the same camera and time controls. The conversion is
validated against the methods' own renderers on the same images: the mean PSNR agrees to within 0.01 dB for Shape of Motion,
4DGaussians and 4DGS (Yang et al.) on real scenes, and to within 0.11 dB for Spacetime Gaussians (0.01 dB when no Gaussian is culled
by opacity); one comparison, 4DGS on D-NeRF `bouncingballs`, differs by 0.5 dB and is not explained yet. Eleven methods are
supported: eight have been run on real outputs, three are covered by synthetic tests only.

## 1. Method

### 1.1 A common representation: Gaussians per keyframe

A converted result is a sequence of $`K`$ keyframes at times $`t_1 < \dots < t_K`$. At keyframe $`k`$, Gaussian $`i`$ has a mean
$`\boldsymbol{\mu}_i^k \in \mathbb{R}^3`$, a covariance $`\boldsymbol{\Sigma}_i^k`$ (symmetric positive semi-definite, stored as its
six upper-triangle entries), an opacity $`\alpha_i^k \in [0, 1]`$, an RGB colour $`\mathbf{c}_i^k`$ and, optionally, the
spherical-harmonics (SH) coefficients of a view-dependent colour. Between two keyframes every quantity is interpolated linearly:

```math
\theta_i(t) = (1-\lambda)\,\theta_i^{k} + \lambda\,\theta_i^{k+1},
\qquad \lambda = \frac{t-t_k}{t_{k+1}-t_k},
\qquad \theta \in \{\boldsymbol{\mu}, \boldsymbol{\Sigma}, \alpha, \mathbf{c}\}.
```

A convex combination of positive semi-definite matrices is positive semi-definite, so $`\boldsymbol{\Sigma}_i(t)`$ is always a
valid covariance. A quantity that does not change over time is stored once, and so is every Gaussian that does not change at all
(the static group). When the keyframes do not share Gaussian identities (some feed-forward models), the viewer shows the nearest
keyframe instead of interpolating. The file format is specified in [FORMAT.md](FORMAT.md).

### 1.2 Converting four types of 4DGS output

Each converter re-implements in NumPy (with PyTorch on the CPU only to unpickle weights or to run a small network) how its method
evaluates the scene at time $`t`$, and writes the Gaussians at $`K`$ chosen times. The formulas are ported from the method's
published code, which is read but never executed by the converters. The four types of output differ in what has to be evaluated:

- **A. Gaussians per frame.** The frames are the keyframes; nothing is evaluated.
- **B. 4D Gaussians.** A Gaussian over $`(\mathbf{x}, t)`$ with mean $`(\boldsymbol{\mu}_x, \mu_t)`$ and covariance blocks
  $`\boldsymbol{\Sigma}_{xx}, \boldsymbol{\Sigma}_{xt}, \Sigma_{tt}`$ is conditioned on time:

```math
\boldsymbol{\mu}(t) = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xt}\,\Sigma_{tt}^{-1}\,(t-\mu_t),
\qquad
\boldsymbol{\Sigma}_{3 \mid t} = \boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xt}\,\Sigma_{tt}^{-1}\,\boldsymbol{\Sigma}_{tx},
\qquad
\alpha(t) = \alpha\,\exp\!\Big(-\tfrac{1}{2}\,\Sigma_{tt}^{-1}\,(t-\mu_t)^2\Big).
```

- **C. Parametric motion.** Closed-form functions of time. In Spacetime Gaussians [7], with $`\tau`$ the temporal centre and
  $`s`$ the temporal scale of a Gaussian,
  $`\boldsymbol{\mu}(t) = \boldsymbol{\mu} + \sum_{j=1}^{3} \mathbf{m}_j\,(t-\tau)^j`$ and
  $`\alpha(t) = \alpha\,\exp\!\big(-((t-\tau)/s)^2\big)`$; in Shape of Motion [8], every Gaussian follows a softmax-weighted blend of
  $`B`$ shared SE(3) trajectories.
- **D. Deformation fields.** Canonical Gaussians are displaced by a network evaluated at time $`t`$,
  $`(\delta\boldsymbol{\mu}, \delta\mathbf{q}, \delta\mathbf{s}) = F_\theta(\boldsymbol{\mu}, t)`$. The converter loads the trained
  weights and re-runs $`F_\theta`$ on the CPU in batches. Whether the offsets are added before or after the activations (exponential
  scale, normalised quaternion) differs between methods and follows each method's renderer exactly.

**Table 1.** Supported methods. The method name links to its converter, whose docstring gives the formulas that were ported.

| Type | Method | Evaluated by the converter at time $`t`$ | Checked on |
|---|---|---|---|
| A | [Per-frame 3DGS ply files](../converters/convert_per_frame.py), e.g. the per-frame export of 4DGaussians [12] | nothing (frames are keyframes) | real output |
| A | [Dynamic 3D Gaussians](../converters/convert_per_frame.py) [4] | nothing (frames are keyframes) | real output, no ground truth |
| B | [4DGS, Yang et al.](../converters/convert_native4d.py) [5] | slice of a 4D Gaussian; SH with a cosine series in time | real output |
| B | [4D-Rotor GS](../converters/convert_4drotor.py) [6] | slice of a 4D Gaussian whose rotation is given by two rotors | synthetic test only |
| C | [Spacetime Gaussians](../converters/convert_spacetime.py), lite model [7] | cubic polynomial motion, linear quaternion motion, temporal RBF opacity | real output |
| C | [Shape of Motion](../converters/convert_shape_of_motion.py) [8] | blend of shared SE(3) motion bases | real output |
| C/D | [MoSca](../converters/convert_mosca.py), dynamic Gaussians [9] | dual-quaternion blending on a 4D motion scaffold | synthetic test only |
| D | [Deformable-3DGS](../converters/convert_deformable3dgs.py) [10] | MLP on positional encodings of position and time (default, D-NeRF and 6-DoF variants) | real output |
| D | [SC-GS](../converters/convert_scgs.py) [11] | MLP on sparse control nodes, blended over the K nearest nodes; optional colour head | real output |
| D | [4DGaussians](../converters/convert_4dgaussians.py) [12] | multi-resolution HexPlane features and MLP heads | real output |
| D | [E-D3DGS](../converters/convert_ed3dgs.py) [13] | per-Gaussian and temporal embeddings, coarse and fine MLPs | synthetic test only |

A variant whose appearance or motion model is not reproduced makes the converter stop with a message that names it, instead of
producing a wrong scene ([list](SUPPORTED.md#stops-with-a-message-the-converters-never-guess)).
`python -m gs4d.open <output>` recognises each output, runs the right converter and opens the viewer.

### 1.3 Rendering

The viewer is built on viser [3]. Two renderers draw the same interpolated Gaussians. **gsplat** [2] rasterises them on the GPU of
the machine that runs the viewer, with their full covariances as in 3DGS [1], evaluates view-dependent colour from the SH
coefficients, and streams the image to the browser. The **client** renderer sends the Gaussians to the browser, which splats them
with WebGL (RGB colour only). While playing, the keyframes around the current time are kept on the GPU and decoded and interpolated
with PyTorch.

### 1.4 Storage

The time-varying fields cost about $`40\,NK`$ bytes in float32 for $`N`$ Gaussians and $`K`$ keyframes. A compact layout stores means
as float16 offsets from the first keyframe, covariances as int16 Cholesky factors (which always decode to a positive semi-definite
matrix), opacities and colours as uint8, and per-keyframe SH quantised over each coefficient's own range in time: about 19 bytes per
Gaussian and keyframe. Files are memory-mapped, so only the keyframes on screen are read, and converters write one keyframe at a time,
so their peak memory does not grow with $`K`$.

## 2. Validation

### 2.1 Protocol

The question is whether a converted file renders like the original model. For each trained model, the converted file is rendered
with gsplat at the camera and time of every evaluation image, at the resolution the model was trained at (a 3DGS model rendered at
another resolution loses several dB), and $`\mathrm{PSNR} = 10\log_{10}(1/\mathrm{MSE})`$ is computed per image on RGB values in
$`[0, 1]`$; tables give the mean over the images, and the range where useful. D-NeRF ground truth is resized with Lanczos filtering
and composited on black (on white for 4DGaussians, which was trained on a white background). Where the method's own renderer could
be run on the same images (`tools/repo_psnr.py`), or the method's own evaluation or renders were available, the two numbers are
compared directly (Table 2). Data: D-NeRF `bouncingballs` [14] (20 test views), HyperNeRF `slice-banana` [15] (297 training and 33
held-out views), Neural 3D Video `sear_steak` [16] and DyCheck iPhone `paper-windmill` [17]; Technicolor `Birthday` [18] and CMU
Panoptic `basketball` [19] (through Dynamic 3D Gaussians) have no ground truth here and were checked visually. Hardware: one NVIDIA
RTX 3060 (12 GB), 15 GB of RAM; PyTorch 2.4.1+cu124, gsplat 1.5.3.

### 2.2 Agreement with the methods' own rendering

**Table 2.** Mean PSNR (dB) of the same model rendered from the converted file and by the method itself, on the same images.

| Method | Scene | Images | Converted file | Method's own | Difference |
|---|---|---|---|---|---|
| Shape of Motion [8] | `paper-windmill` | 70, 720×960 | 25.85 | 25.85 (renderer) | 0.000 dB on every image |
| 4DGaussians [12] | `slice-banana` | 33, 268×480 | 27.13 | 27.13 (renders) | at most 0.003 dB per image |
| 4DGS, Yang et al. [5] | `slice-banana` | 33, 268×480 | 26.31 | 26.30 (evaluation) | +0.01 dB |
| Spacetime Gaussians, lite [7] | `sear_steak` | 105, 1352×1014 | 34.37 | 34.48 (renderer) | −0.11 dB; within 0.01 dB with `--min-peak-opacity 0` |
| 4DGS, Yang et al. [5] | `bouncingballs` | 20, 400×400 | 32.9 | 33.4 (evaluation) | −0.5 dB, not explained yet |

Shape of Motion is converted with every frame as a keyframe (`--stride 1`). The Spacetime converter culls Gaussians whose opacity
never exceeds 0.02 by default; without culling the two renderers agree to within 0.01 dB.

### 2.3 Numerical checks of the conversion

**Table 3.** Checks that compare numbers rather than images.

| Check | Result |
|---|---|
| 4DGaussians converter against the method's own per-frame export (`export_perframe_3DGS.py`), 20 times | positions within 1.2·10⁻⁶, covariances within 7·10⁻⁶ (relative), opacities identical |
| Every converter against an independent formulation, on synthetic inputs in the method's file layout (`tests/test_<method>.py`) | pass; run by CI on every push |
| GPU decoding and interpolation against the CPU reference (`Sequence4D.at`) | largest relative difference 1.1·10⁻⁷ |
| Images of the GPU path against the CPU path | at most 1/255 per channel; PSNR of every real output unchanged |
| Streaming writer against writing the whole sequence at once | byte-identical files |
| Compact layout, 500,000 random Gaussians × 100 keyframes | largest error: position 0.002 (for offsets up to 5.35 units), opacity 0.002, covariance 7·10⁻⁵ of the largest variance |
| Compact per-keyframe SH on two real 4DGS models | PSNR change at most 0.0004 dB per image |

### 2.4 Image quality of the converted models

These numbers characterise the trained models as seen through the converter; the self-trained models were trained once, on one
GPU, and are not comparable with the results published by the methods' authors.

**Table 4.** PSNR (dB) of the converted file against ground truth: mean (min–max). $`K`$: number of keyframes.

| Method | Scene | Model | Images | $`K`$ | PSNR |
|---|---|---|---|---|---|
| Deformable-3DGS [10], D-NeRF variant | `bouncingballs` | self-trained | 20 test, 800×800 | 60 | 40.3 (37.2–43.6) |
| Deformable-3DGS, default variant | `bouncingballs` | self-trained | 20 test, 800×800 | 60 | 39.62 (35.35–43.14) |
| Deformable-3DGS, 6-DoF variant | `bouncingballs` | self-trained | 20 test, 800×800 | 60 | 40.34 (36.99–43.75) |
| SC-GS [11] | `bouncingballs` | self-trained | 20 test, 400×400 | 60 | 40.3 (35.4–43.0) |
| SC-GS with motion mask | `bouncingballs` | self-trained | 20 test, 400×400 | 60 | 40.14 (35.48–43.14) |
| 4DGaussians [12] | `bouncingballs` | self-trained | 20 test, 800×800 | 60 | 40.8 (36.7–45.9) |
| 4DGS, Yang et al. [5] | `bouncingballs` | self-trained | 20 test, 400×400 | 60 | 32.9 (27.3–39.6) |
| Deformable-3DGS | `slice-banana` | self-trained | 33 held-out, 268×480 | 330 | 24.72 (19.28–29.78) |
| SC-GS | `slice-banana` | self-trained | 33 held-out, 268×480 | 110 | 26.69 (20.21–30.22) |
| 4DGaussians | `slice-banana` | self-trained | 33 held-out, 268×480 | 330 | 27.13 (22.05–30.39) |
| 4DGS, Yang et al. | `slice-banana` | self-trained | 33 held-out, 268×480 | 110 | 26.31 (21.41–29.17) |
| Spacetime Gaussians, lite [7] | `sear_steak` | authors' | 105 (all 21 cameras × 5 frames), 1352×1014 | 50 | 34.37 (27.76–37.81) |
| Shape of Motion [8] | `paper-windmill` | authors' | 70 training views (every 4th frame), 720×960 | 277 | 25.85 (24.28–27.66) |

Shape of Motion is evaluated at the training cameras stored in its checkpoint, and the published Spacetime model (`allcam`) was
presumably trained on all 21 cameras, so these two rows measure the fit to training views rather than novel-view quality. All 4DGS
rows use the SH colour (`--sh`); without it, 4DGS on `bouncingballs` gives 29.0 dB. The SC-GS colour head was checked on a
self-trained `bouncingballs` whose colours change over time: 37.83 dB with the head, 24.48 dB when it is ignored. Every row is a
check with a threshold in `tests/check_real.py`; per-method details are in [SUPPORTED.md](SUPPORTED.md).

### 2.5 Speed and memory

**Table 5.** Measured on the hardware of Section 2.1; full tables in [PERFORMANCE.md](PERFORMANCE.md) and
[FORMAT.md](FORMAT.md).

| Quantity | Setting | Result |
|---|---|---|
| Time to fetch one interpolated frame | 500,000 Gaussians × 100 keyframes, compact | 100 ms (CPU decoding) → 2.2 ms (GPU decoding) |
| Time to fetch one interpolated frame | `paper-windmill`, 995,942 Gaussians × 35 keyframes | 214 ms → 3.1 ms |
| Render, encode and send one frame | synthetic scene, 15,000 Gaussians, 1280×720 / 1920×1080 | 14 ms / 23 ms |
| Frames shown by Chrome with GPU | `paper-windmill`, 1280×720, update rate 20 → 60 Hz | 20.0 → 52.9 frames/s |
| File size, static group stored once | `paper-windmill` | 1271 MB → 147 MB (103 MB compact) |
| File size, compact per-keyframe SH | 4DGS `slice-banana`, 330 keyframes | 19,858 MB → 5,824 MB |
| Peak RAM of a converter, streaming writer | 4DGS `slice-banana`, 110 keyframes | 8.3 GB → 1.3 GB |

## 3. Limitations

- **Keyframe sampling.** Linear interpolation cannot follow motion that is non-linear within one keyframe interval. On
  `paper-windmill`, keyframes every 8 frames give 25.86 dB at the keyframes and 24.67 dB half way between them; with every frame a
  keyframe the PSNR is 25.85 dB everywhere, at the cost of a larger file (416 MB compact). Storage grows linearly with $`K`$.
- **Appearance models that are not reproduced.** The neural colour decoder of the full Spacetime model and view-dependent colour
  heads (SC-GS) are not ported; the converters stop on them, or convert with the base colour when asked to.
- **Untested on real data.** 4D-Rotor GS, MoSca and E-D3DGS, and the hash-grid and skinning variants of SC-GS, are checked on
  synthetic inputs only; the static background of MoSca is not converted.
- **Open discrepancy.** 4DGS (Yang et al.) on `bouncingballs` is 0.5 dB below the method's own evaluation (Table 2).
- **Measurement scope.** One machine and one GPU; the viewer was not measured with a browser on another machine; lens distortion of
  HyperNeRF images is ignored, as in the methods' own loaders.
- **Client renderer.** WebGL rendering has no view-dependent colour, clips colours to $`[0, 1]`$ and draws the 300,000 most opaque
  Gaussians by default.

## 4. Reproducing the results

```bash
python tests/run_all.py        # every test and browser check (pass / skipped / failed)
python tests/check_real.py     # Tables 2 and 4: PSNR of every real output (CUDA + data)
python tools/repo_psnr.py som  # Table 2: the method's own renderer (needs its repository)
python benchmarks/bench_large.py    # Tables 3 and 5: compact layout, decoding speed
python benchmarks/bench_display.py --rates 20 30 40 50 60    # Table 5: frames shown by Chrome
```

The data paths of `tests/check_real.py` are set at the top of the file.

GitHub Actions runs the lint and the tests without and with CPU PyTorch on every push. The checks with real data need the trained
models and datasets of Table 4 and run on the development machine; how each model was trained, and the conversion command for each
method, are in [SUPPORTED.md](SUPPORTED.md). Every measurement is logged with its date in
[HISTORY.md](HISTORY.md).

## References

1. B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM
   Transactions on Graphics 42(4), 2023 (SIGGRAPH 2023).
2. V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye, J. Hu, M. Tancik, A. Kanazawa. *gsplat: An
   Open-Source Library for Gaussian Splatting.* Journal of Machine Learning Research 26(34), 2025.
   <https://github.com/nerfstudio-project/gsplat>
3. B. Yi, C. M. Kim, J. Kerr, G. Wu, R. Feng, A. Zhang, J. Kulhanek, H. Choi, Y. Ma, M. Tancik, A. Kanazawa. *Viser: Imperative,
   Web-based 3D Visualization in Python.* arXiv:2507.22885, 2025. <https://viser.studio>
4. J. Luiten, G. Kopanas, B. Leibe, D. Ramanan. *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis.* 3DV 2024.
   <https://github.com/JonathonLuiten/Dynamic3DGaussians>
5. Z. Yang, H. Yang, Z. Pan, L. Zhang. *Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian
   Splatting.* ICLR 2024. <https://github.com/fudan-zvg/4d-gaussian-splatting>
6. Y. Duan, F. Wei, Q. Dai, Y. He, W. Chen, B. Chen. *4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for
   Dynamic Scenes.* SIGGRAPH 2024 Conference Papers. <https://github.com/weify627/4D-Rotor-Gaussians>
7. Z. Li, Z. Chen, Z. Li, Y. Xu. *Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis.* CVPR 2024.
   <https://github.com/oppo-us-research/SpacetimeGaussians>
8. Q. Wang, V. Ye, H. Gao, W. Zeng, J. Austin, Z. Li, A. Kanazawa. *Shape of Motion: 4D Reconstruction from a Single Video.*
   ICCV 2025. <https://github.com/vye16/shape-of-motion>
9. J. Lei, Y. Weng, A. W. Harley, L. Guibas, K. Daniilidis. *MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion
   Scaffolds.* CVPR 2025. <https://github.com/JiahuiLei/MoSca>
10. Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, X. Jin. *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene
    Reconstruction.* CVPR 2024. <https://github.com/ingra14m/Deformable-3D-Gaussians>
11. Y. Huang, Y. Sun, Z. Yang, X. Lyu, Y. Cao, X. Qi. *SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes.*
    CVPR 2024. <https://github.com/yihua7/SC-GS>
12. G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, X. Wang. *4D Gaussian Splatting for Real-Time Dynamic Scene
    Rendering.* CVPR 2024. <https://github.com/hustvl/4DGaussians>
13. J. Bae, S. Kim, Y. Yun, H. Lee, G. Bang, Y. Uh. *Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian
    Splatting.* ECCV 2024. <https://github.com/JeongminB/E-D3DGS>
14. A. Pumarola, E. Corona, G. Pons-Moll, F. Moreno-Noguer. *D-NeRF: Neural Radiance Fields for Dynamic Scenes.* CVPR 2021.
    <https://github.com/albertpumarola/D-NeRF>
15. K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, S. M. Seitz. *HyperNeRF: A
    Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields.* ACM Transactions on Graphics 40(6), 2021
    (SIGGRAPH Asia 2021). <https://hypernerf.github.io>
16. T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe, Z. Lv.
    *Neural 3D Video Synthesis from Multi-view Video.* CVPR 2022. <https://neural-3d-video.github.io>
17. H. Gao, R. Li, S. Tulsiani, B. Russell, A. Kanazawa. *Monocular Dynamic View Synthesis: A Reality Check.* NeurIPS 2022.
    <https://github.com/KAIR-BAIR/dycheck>
18. N. Sabater, G. Boisson, B. Vandame, P. Kerbiriou, F. Babon, M. Hog, R. Gendrot, T. Langlois, O. Bureller, A. Schubert,
    V. Allié. *Dataset and Pipeline for Multi-View Light-Field Video.* CVPR Workshops 2017.
19. H. Joo, H. Liu, L. Tan, L. Gui, B. Nabbe, I. Matthews, T. Kanade, S. Nobuhara, Y. Sheikh. *Panoptic Studio: A Massively
    Multiview System for Social Motion Capture.* ICCV 2015. <https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox>

## License and data sources

The code in this repository is under the MIT license (see `LICENSE`). Some images in `shots/` contain data of other parties and are
**not** under the MIT license:

- `shots/dynamic3dgs/basketball_fg_up-y_t*.png` are rendered from the `basketball` model of Dynamic 3D Gaussians [4], built from CMU Panoptic
  Studio recordings [19]. The CMU Panoptic Studio dataset may be used for non-commercial research only; these three images may only be
  used under the same terms.
- `shots/real/shape_of_motion_paper-windmill.png` combines renders with the `paper-windmill` scene of the DyCheck iPhone set [17]. The
  DyCheck repository releases both code and dataset under Apache 2.0; the checkpoint was published by the Shape of Motion authors [8]
  (MIT repository).
- `shots/real/deformable3dgs_bouncingballs.png`, `scgs_bouncingballs.png` and `native4d_bouncingballs.png` (top row render, bottom row
  original image), the original-image panels of `shots/cameras/ground_truth*.png` and the right-hand panel of `shots/readme_hero.png`
  contain test images of the synthetic D-NeRF `bouncingballs` scene [14]. The D-NeRF repository is MIT-licensed and states no separate
  terms for the dataset. The models were trained by us with the code of Deformable-3DGS [10], SC-GS [11] and 4DGS [5] (all three
  repositories are MIT).
