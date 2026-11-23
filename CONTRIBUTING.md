# Contributing to gs4d_viewer

Thank you for helping. gs4d_viewer is only useful if its conversions are correct, so every contribution comes with the evidence
that it works: a test, a measurement, or a screenshot.

## Ways to contribute

- **Report a bug**, or a scene that renders wrongly: open an issue with the command, the full message and, if you can, a screenshot.
- **Ask for a method**: open an issue with the paper, the official repository and the files its training writes.
- **Add a method**: write a converter, following [docs/ADD_A_CONVERTER.md](docs/ADD_A_CONVERTER.md) and its working template.
- **Validate a converter on real data**: 4D-Rotor GS, MoSca and E-D3DGS (and SC-GS with `--use_hash` or `--skinning`) are tested
  on synthetic data only. A trained model with the PSNR of the converted file and of the method's own renderer on the same images
  (protocol: [docs/METHOD.md](docs/METHOD.md#21-protocol)) is a valuable contribution.
- **Improve the viewer or the documentation.**

## Set up

```bash
git clone https://github.com/zerokhong1/gs4d_viewer.git
cd gs4d_viewer
pip install -r requirements.txt   # viewer, mp4 export and Playwright (browser checks)
pip install torch ruff==0.16.9    # converter tests (CPU PyTorch is enough), CI linter
```

The gsplat (GPU) renderer is installed as described in the [README](README.md#installation).

## Before opening a pull request

```bash
python tests/run_all.py   # every test and browser check; prints why any is skipped
ruff check .              # lint, as in CI
```

The browser checks (`tests/check_*.py`) start Google Chrome from `/usr/bin/google-chrome`; files that need Chrome, CUDA or the
real data are skipped when these are missing. GitHub Actions runs the lint and all tests that need no GPU or data, browser
checks included, on every push and pull request.

## Rules

- **Where files go** ([README](README.md#repository-layout)): library code in `gs4d/`, converters in `converters/`, tests in
  `tests/`, other scripts in `tools/`, measurements in `benchmarks/`, documents in `docs/`, screenshots in `shots/<feature>/`.
  Nothing new at the root.
- **Converters port, they do not import.** A converter re-implements the method's formulas from its published code and runs on
  the CPU (NumPy; PyTorch only to unpickle weights or to run a small network). It never imports or runs the method's code, and its
  docstring states the formulas and the files they were ported from.
- **Every converter has a test** on synthetic inputs in the method's real file layout, checked against an independent
  formulation (`tests/test_<method>.py`).
- **No silent approximations.** A variant that a converter does not support stops with a one-line message that names it.
- **Results are reproducible.** A measurement goes to `docs/` with how it was made (data, views, resolution, hardware, command),
  plus a dated entry at the end of `docs/HISTORY.md`. A number in the README must also be in `docs/`.
- **English** everywhere, with English number separators (98,993; 26.31 dB).
- **Data licenses.** Do not commit datasets, trained models, or images whose terms do not allow it; note the terms of any dataset
  image added to `shots/`.
- **One topic per pull request.** Describe what changed, how it was checked, and what is not tested.

## Code of conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are released under the MIT license of this repository.
