# gs4d_viewer

One representation and one viewer for the outputs of 4D Gaussian Splatting (4DGS) methods.

[![tests](https://github.com/zerokhong1/gs4d_viewer/actions/workflows/test.yml/badge.svg)](https://github.com/zerokhong1/gs4d_viewer/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](#installation)

![Toy scene over time, motion trails, and an SC-GS render next to the ground-truth image](shots/readme_hero.png)

gs4d_viewer converts the trained output of a 4DGS method into one format, the Gaussians at a set of keyframes with linear
interpolation in between, and plays it in the browser with a time slider. Each converter re-implements its method's evaluation at
time *t* from the published code, so the method's own environment is not needed. How it works and how it is validated:
[docs/METHOD.md](docs/METHOD.md).

## Installation

Client renderer only (any machine, no GPU needed):

```bash
pip install -r requirements.txt
```

gsplat renderer (NVIDIA GPU, Python 3.10):

```bash
uv venv --python 3.10 --seed .venv
uv pip install --python .venv/bin/python torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python "gsplat==1.5.3+pt24cu124" --find-links https://docs.gsplat.studio/whl/gsplat/ --index-strategy unsafe-best-match
```

Converters for methods that save PyTorch checkpoints also need PyTorch (the CPU build is enough): `pip install torch`.

## Quick start

```bash
python tools/make_toy.py -o toy.npz             # synthetic dynamic scene
python viewer.py toy.npz                        # then open http://localhost:8080
python -m gs4d.open output/dnerf/bouncingballs  # open the output of a method
```

Comparing two results, ground truth with PSNR, points, motion trails, camera paths and mp4 export: [user guide](docs/USAGE.md).

## Supported methods

| Method | Input | Tested on |
|---|---|---|
| Per-frame 3DGS `.ply` (e.g. the export of 4DGaussians) | folder of `*.ply` | real output |
| [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians) | `params.npz` | real output (no ground truth) |
| [4DGS, Yang et al.](https://github.com/fudan-zvg/4d-gaussian-splatting) | `chkpnt*.pth` | real output |
| [4D-Rotor GS](https://github.com/weify627/4D-Rotor-Gaussians) | `step-*.ckpt` | synthetic data only |
| [Spacetime Gaussians](https://github.com/oppo-us-research/SpacetimeGaussians) (lite) | `point_cloud.ply` | real output |
| [Shape of Motion](https://github.com/vye16/shape-of-motion) | `checkpoints/last.ckpt` | real output |
| [MoSca](https://github.com/JiahuiLei/MoSca) (dynamic part) | `photometric_d_model_*.pth` | synthetic data only |
| [Deformable-3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians) | model folder | real output |
| [SC-GS](https://github.com/yihua7/SC-GS) | model folder | real output |
| [4DGaussians](https://github.com/hustvl/4DGaussians) | model folder | real output |
| [E-D3DGS](https://github.com/JeongminB/E-D3DGS) | model folder | synthetic data only |

`python -m gs4d.open` picks the converter. Commands and notes for each method: [docs/SUPPORTED.md](docs/SUPPORTED.md); adding a
method: [docs/ADD_A_CONVERTER.md](docs/ADD_A_CONVERTER.md).

## Results

Mean PSNR (dB) of the same model rendered by the method's own code and from the converted file, on the same images:

| Method | Scene | Own code | Converted |
|---|---|---|---|
| Shape of Motion | `paper-windmill` | 25.85 | 25.85 |
| 4DGaussians | `slice-banana` | 27.13 | 27.13 |
| 4DGS, Yang et al. | `slice-banana` | 26.30 | 26.31 |
| Spacetime Gaussians (lite) | `sear_steak` | 34.48 | 34.37 |
| 4DGS, Yang et al. | `bouncingballs` | 33.4 | 32.9 |

The 0.5 dB gap on `bouncingballs` is not explained yet. Protocol, more results and limitations: [docs/METHOD.md](docs/METHOD.md);
speed and memory: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Repository layout

```
gs4d/         library: the common format, readers, python -m gs4d.open
converters/   one converter per 4DGS method, and a template for new ones
viewer.py     the viewer
tools/        toy scene, data preparation, the methods' own renderers for validation
tests/        unit tests and browser checks (python tests/run_all.py)
benchmarks/   speed and memory measurements
docs/         user guide, method and validation, results, file format, history
shots/        screenshots and videos used as evidence in the docs
```

## Contributing

Contributions are welcome: support for a new 4DGS method, a check of a converter on a real model (4D-Rotor GS, MoSca and E-D3DGS
are tested on synthetic data only), viewer features, fixes and documentation. Read [CONTRIBUTING.md](CONTRIBUTING.md) first; to
add a method, follow [docs/ADD_A_CONVERTER.md](docs/ADD_A_CONVERTER.md). Bugs and questions:
[issues](https://github.com/zerokhong1/gs4d_viewer/issues).

## Citation

```bibtex
@misc{luu2026gs4dviewer,
  author       = {Luu, Thai Cong},
  title        = {gs4d\_viewer: One Representation and One Viewer for the Outputs
                  of 4D Gaussian Splatting Methods},
  year         = {2026},
  howpublished = {\url{https://github.com/zerokhong1/gs4d_viewer}}
}
```

Please also cite the methods whose outputs you view.

## License and data sources

Code: MIT (see `LICENSE`). Some images in `shots/` contain images of the CMU Panoptic, DyCheck and D-NeRF datasets and are not
under the MIT license; their terms are listed in [docs/METHOD.md](docs/METHOD.md#license-and-data-sources).
