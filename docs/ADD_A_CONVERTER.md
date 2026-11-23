# Adding a 4DGS method

The viewer reads one common `.npz` file (`gs4d/format.py`, class `Sequence4D`). Adding a method = writing a converter that writes that file;
do not change `viewer.py`, `gs4d/format.py` or other converters.

## Steps

1. `cp convert_template.py convert_<name>.py` and `cp tests/test_template.py tests/test_<name>.py`. The template is a made-up method
   (position = `xyz + vel * t`) that already runs.
2. **Read the original repository, do not run it.** Find: the output files (checkpoint, ply, npz), how position / quaternion / scale /
   opacity / colour are computed at time `t`, the activations (`exp` for scale, `sigmoid` for opacity), and the range of time. Port the
   formulas to numpy. Torch only opens the checkpoint or re-runs a small MLP on the CPU, and is loaded through `gs4d.checkpoint`
   (`import_torch`, `torch_load`: a missing torch or a broken file gives a one-line message). Do not add the original repository as a
   dependency.
3. Put the method in a type ([SUPPORTED.md](SUPPORTED.md)): A Gaussians per frame; B 4D Gaussians; C parametric motion functions;
   D deformation networks re-run on the CPU.
4. Replace `load_model` and `keyframe` in the template. What the format needs: `means [N,3]`, `covs [N,6]` (upper triangle of the 3×3;
   use `cov_from_quat_scale(quat_wxyz, linear_scale)`), `opacities [N]` in [0,1], `colors [N,3]` (may lie outside [0,1] when the method
   stores raw colours, like Spacetime), optional `sh [N,S,3]`. Fields that do not change over time go to the constructor of
   `Sequence4DWriter`, the others to every `add_keyframe(t, ...)`.
5. Write with `Sequence4DWriter` (peak memory of a few keyframes; the static group and `--compact` come for free), never build the whole
   sequence in memory.
6. An unsupported variant: `raise SystemExit("... name the variant ...")`; never convert it wrongly without a word.
7. Test: fake data in the real file layout, compared with an independent formulation (like `tests/test_template.py`: the full covariance
   matrix `R S Sᵀ Rᵀ`). `tests/run_all.py` picks up every `tests/test_*.py`; a test that needs torch goes into `NEEDS` in
   `tests/run_all.py`.
8. Add a row to the methods table of the README, to Table 1 of `docs/METHOD.md` (with the reference in its References list) and to
   the inputs table of `docs/USAGE.md`, and a row plus a "Details" section to `docs/SUPPORTED.md`. Making `gs4d.open` recognise the
   output is a separate job in `gs4d/open.py` (and `tests/test_open.py`).
9. Run `python tests/run_all.py` and `ruff check .`. Check on a real output (see below) before writing down a PSNR, with a dated entry at
   the end of `docs/HISTORY.md`.

## Lessons learned

- **Check the original repository's cameras before trusting its evaluation.** Native 4DGS on `slice-banana` was 1.9 dB below the repo;
  the cause was that the cameras built from our layout left `FovX = FovY = -1`, so the CUDA rasteriser used a fake focal length and the
  model learned to compensate. The converter was not at fault. After the camera fix the two numbers agree within 0.01 dB.
- **Measure at the training resolution.** Deformable-3DGS trained at 800 px gives 37.2–43.6 dB; rendered at 400 px only 29.3–33.2 dB.
- **Compare with the repository's own renderer** when possible (`tests/repo_psnr.py`): Shape of Motion matches to 0.000 dB; doing this is
  how we found that Spacetime stores raw colours outside [0,1], which the converter used to clip (31.43 → 34.37 dB).
- Add in the right order: Deformable-3DGS and SC-GS add `d_scaling` to `exp(scale)` (after the activation), not to the log-scale. Check
  every such detail in the repository, do not infer it.
- Time: D-NeRF uses [0, 1], video scenes use the frame index or frame/fps. Choose `--t0 --t1` to match and say so in `--help`.
- The machine has 15 GB of RAM: for long sequences, check the peak RAM; compare a file written with `Sequence4DWriter` with
  `Sequence4D.save` on small data.
