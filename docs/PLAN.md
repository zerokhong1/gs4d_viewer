# Plan — open work packages of gs4d_viewer

The work packages still open, with the prompts to start them. The finished ones (WP1–WP31, WP34, WP35, WP39 and tasks 1–2 of
WP40) are recorded in [HISTORY.md](HISTORY.md); their plans and prompts are in the git history (this file at commit `a455a04`).

## Status

| WP | Task | Status | Session | Notes |
|---|---|---|---|---|
| 32 | Demo scene set and a 10-minute demo script (`docs/DEMO.md`) | Not started | CC-C | Preparation for the demo with the professor |
| 33 | Install on a clean machine with one command (`tools/setup.sh`) | Not started | CC-B | Needs about 8 GB of free disk to try on a fresh clone |
| 36 | Pick the scene in the GUI; compare up to four sequences | Not started | CC-A | Its precondition WP35 is done |
| 37 | Depth and opacity view modes | Not started | CC-A | Small; after WP36, or earlier if the professor needs it |
| 38 | Viewing from another machine over SSH | Not started | human | Needs a second machine |
| 40 | Task 3: assess the methods the professor asks for | Waits for the professor's feedback | CC-C | Tasks 1–2 (`docs/SUPPORTED.md`, the converter template) done 17/11/2026 |

Order: WP32 and WP33 in parallel; the demo with the professor as soon as WP32 is done; then WP36, WP37 and WP38, reordered after the
demo feedback, and task 3 of WP40 once the professor has named methods.

## By hand

- **Demo for the professor** as soon as WP32 is done; fill in the feedback table below.
- **Free disk space** (about 15 GB were left on 16/11/2026): delete `../data_real/som/paper-windmill_data/paper-windmill.zip` (6.6 GB)
  and `sear_steak.zip` (1.1 GB) after checking that `tests/check_real.py` still passes; both are fully extracted.
- **Clean up worktrees**: `git worktree list`; `git branch --no-merged main` must be empty; then `git worktree remove <path>` for each
  merged worktree and `git worktree prune`.
- Not measured yet: the client renderer in points mode on a real GPU.

## Professor's feedback after the demo

| Question | Answer | Date |
|---|---|---|
| What is still missing when looking at a result? | | |
| Is points mode (WP24) what "observe the 4DGS points" meant? Are motion trails (WP35) needed? | | |
| Are videos along a camera path (WP34), or several methods side by side (WP36), needed? | | |
| Besides the methods already supported, which methods does the lab need? (e.g. 4D-Rotor GS, E-D3DGS, MoSca) | | |
| Who will use it, on which machine? (decides WP33, WP38) | | |

## Conventions

Drawn from what went wrong in phases 2 to 4:

- **Every session works in its own worktree** (`git worktree add ../gs4d-wp<N> -b cc-<session>/wp<N>-<name>`), and never checks out a
  branch in the main folder.
- **`docs/PLAN.md` is edited only in the main folder, and a session edits only its own package's row** in the status table (replace that
  row, never rewrite the whole file). Never reuse a WP number; the next new package is WP41.
- Files each session may edit: CC-A: `viewer.py`, `gs4d/format.py`, the package's new `tests/check_*.py`; CC-B: `gs4d/open.py`,
  `gs4d/doctor.py`, `tools/setup.sh`, `requirements.txt`, `tests/run_all.py`, `.github/`; CC-C: `docs/`, `tools/make_demo_scenes.py`,
  `converters/convert_template.py`. In `docs/USAGE.md` every session edits only its own package's paragraph. A change needed in another
  session's file goes into the report, not into the file.
- The README stays short: a new feature gets its paragraph in `docs/USAGE.md`, a measurement goes into `docs/` (see
  `.claude/CLAUDE.md`).
- A package that adds a GUI feature comes with a `tests/check_*.py` script that takes headless screenshots as evidence, and is tried on at
  least one real scene. The scripts write to `shots/<feature>/`, which git ignores; an image a document shows is copied to
  `docs/images/<feature>/`.
- A finished package is merged into `main` through a pull request with green CI; `python tests/run_all.py` runs before that session's
  next package.
- One shared GPU (RTX 3060 12 GB) and 15 GB of RAM: only one session at a time may train, run `check_real.py` or `bench_display.py`; ask
  the user first.

## Done criteria for every package

1. A package that touches a converter or the format is done only when it has run on **at least one real output** in `../data_real`
   and its measurements (PSNR against the original images, size, time) are written down in `docs/` (`SUPPORTED.md`, `PERFORMANCE.md`).
2. A package that changes the format or the renderer reruns `tests/check_real.py` and must not lower any PSNR by more than 0.2 dB.
3. Every old test still passes; old `.npz` files can still be read.
4. Update the package's row in the status table above, and add a dated entry at the end of `docs/HISTORY.md`.
5. Report in 5 lines, then stop; do not widen the scope. Missing data or GPU: say so in the first turn, do not guess.

## Prompts

Paste the whole block into a new Claude Code session opened in `gs4d_viewer`.

#### WP32 — demo scene set and demo script
```
Read .claude/CLAUDE.md and the "Conventions" and "Done criteria for every package"
sections of docs/PLAN.md. Work in your own worktree.
Task 1: write tools/make_demo_scenes.py, which builds the folder `../demo_scenes/`
(outside the repo) from the real outputs already on this machine: one compact `.npz` per
method on `bouncingballs` (or the scene that method already has), plus the four
`slice-banana` files of Deformable-3DGS, SC-GS, native 4DGS and 4DGaussians. Each file
comes with its camera file if there is one. Print a table: name, method, number of
Gaussians, number of keyframes, size.
Train nothing new; when an output is missing, skip it and print why.
Task 2: write docs/DEMO.md, one page, for a 10-minute demo with the professor: 5–6
commands in order, each with one line "what it shows" (open an output with one command;
drag the time slider and orbit the camera; points mode; compare two methods on the same
scene with --compare; original image next to the render with PSNR; mp4 export), and one
line on the known limits. Run every command for real before writing it down.
Task 3: note in docs/DEMO.md which scenes have their own terms of use (data showing
people, research-only licenses: see the "License and data sources" section of
docs/METHOD.md) so the presenter knows which scenes must not be shown publicly. Upload
nothing.
Only add: tools/make_demo_scenes.py, docs/DEMO.md, one line in the README pointing to
docs/DEMO.md.
Done when: the demo folder has every file, every command in docs/DEMO.md runs on this
machine, and the total size of the folder is written down.
Report in 5 lines, then stop.
```

#### WP33 — install on a clean machine with one command
```
Read .claude/CLAUDE.md and the "Conventions" and "Done criteria for every package"
sections of docs/PLAN.md. Work in your own worktree.
Task: write tools/setup.sh, which creates `.venv` with the checked versions (Python 3.10,
torch 2.4.1+cu124, gsplat 1.5.3, numpy<2, viser, and the packages in requirements.txt),
picks the non-CUDA branch by itself when the machine has no NVIDIA GPU (then only the
client renderer is available), and runs a smoke test: build toy.npz, `python
tests/run_all.py`, and print clearly which renderers work. Add `python -m gs4d.doctor`,
which prints the package versions, CUDA, Chrome, and what to do when something is missing.
Rewrite the installation section of the README as: clone, `bash tools/setup.sh`, `python
-m gs4d.open <output>`.
Step 0: `df -h .`; about 8 GB of free space is needed for a second `.venv` (the disk has
about 15 GB left); if not, report and stop.
Check with a fresh clone in a temporary folder, without reusing the existing `.venv`;
write down the install time and download size.
Only add or edit: tools/setup.sh, gs4d/doctor.py, requirements.txt, the installation
section of the README.
Done when: on the fresh clone the three commands above run from start to end and open
toy.npz with the gsplat renderer. Report in 5 lines, then stop.
```

#### WP36 — pick the scene in the GUI; compare more than two sequences
```
Read .claude/CLAUDE.md and the "Conventions" and "Done criteria for every package"
sections of docs/PLAN.md. Work in your own worktree.
Task 1: `python viewer.py <folder>` (a folder with several `.npz` files in the common
format, e.g. ../demo_scenes) shows a "Scene" dropdown that switches scenes without
restarting; lazy loading, the old scene is released; the camera stays if both scenes share
a coordinate system (same name prefix), otherwise it is reset.
Task 2: `--compare` takes several files (up to 4): the gsplat renderer splits the view
into a 2x2 grid with the same camera, the file name in the corner of each cell; the client
renderer puts them side by side. The original image (`--cameras`) is one cell of the grid
when switched on. mp4 export and the camera path (WP34) work with the grid.
Only edit viewer.py, tests/check_compare.py, one paragraph in docs/USAGE.md; making
`gs4d/open.py` accept a folder of several raw outputs is CC-B's job.
Done when: tests/check_compare.py passes with 3 and 4 sequences; a screenshot of a grid of
four methods on `slice-banana` in shots/compare/; docs/PERFORMANCE.md gives the time per
frame of the 2x2 grid at 1280x720 and the RAM / GPU memory used. Report in 5 lines, then
stop.
```

#### WP37 — depth and opacity view modes
```
Read .claude/CLAUDE.md and the "Conventions" and "Done criteria for every package"
sections of docs/PLAN.md. Work in your own worktree.
Task: a "Render" dropdown for the gsplat renderer: `colour`, `depth` (expected depth,
colour scale, near/far taken from the 2–98% percentiles of the frame and shown in the
control panel), `alpha` (accumulated opacity). Read the documentation of the installed
gsplat 1.5.3 to use `render_mode` correctly; do not recompute depth yourself.
Works while playing, with `--compare`, the camera path and mp4 export. Client renderer:
hide this control.
Only edit viewer.py, add tests/check_depth.py, one paragraph in docs/USAGE.md.
Done when: the test compares the depth at the image centre of toy with an independently
computed distance; screenshots in shots/depth/ for `paper-windmill` and one `slice-banana`
scene. Report in 5 lines, then stop.
```

#### WP38 — viewing from another machine over SSH (human)

On a second machine: `ssh -L 8080:localhost:8080 <this machine>`, open `http://localhost:8080` in Chrome, and repeat the WP19b measurement
steps at 1280×720 (`benchmarks/bench_display.py`). Done when `docs/PERFORMANCE.md` has the frames/s shown over SSH next to the
same-machine numbers, with the kind of network (LAN, Wi-Fi, VPN) and the bitrate if it can be read. If it is clearly slower, open a new
package for CC-A: lower the JPEG quality or the resolution while the camera is moving.

#### WP40, task 3 — methods the professor asks for
```
Read .claude/CLAUDE.md and the "Conventions" and "Done criteria for every package"
sections of docs/PLAN.md. Work in your own worktree.
Task: for the methods the professor names in the table "Professor's feedback after the
demo" (if the table is empty, stop and say so): for each method one short paragraph:
official repository, output format, which type A/B/C/D it is, estimated effort to write
the converter, whether it can be trained on a machine with a 12 GB GPU / 15 GB RAM. Look
at the real repository, do not write from memory; do not write a converter in this
package.
Only edit files in docs/.
Done when: every method named in the table has its paragraph. Report in 5 lines, then
stop.
```
