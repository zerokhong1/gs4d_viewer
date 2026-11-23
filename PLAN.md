# Plan for phases 2, 3 and 4 — gs4d_viewer

Update 23/11/2026: the whole repository was reviewed and polished (PR #2, branch `polish/code-and-docs`: bug fixes, new tests, the
README rewritten as a user guide, measurements moved to `docs/`, the `Status` log of `CLAUDE.md` moved to `docs/HISTORY.md`); the same day
all documentation was translated into English (branch `docs/english`). Every old `cc-*` branch is already in `main`. In the work-package tables below only the WP28 and WP39 rows
were updated, and the finished items of the "Do now" lists are struck through. Where a prompt below says a measurement goes into the
README, it now goes into `docs/` (see the conventions in `CLAUDE.md`).

Update 16/11/2026 (20:20): phase 4 added right below (WP32–WP40, no package started yet); phase 3 is done except the FPS measurement that
needs a human.

Update 16/11/2026: phase 2 is done (WP24 was the last package); phase 3 is in the "Phase 3" section below.

Update 15/11/2026 (15:15). Phase 1 (WP1–WP8) is done; its original plan is in the appendix at the end of this file. Phase 2: WP10–WP18,
WP19a, WP19d and part 3 of WP19c are done, committed and merged into `main` (13:45). The professor has answered (table below): no package
has to wait any more. Every Claude Code session reads `CLAUDE.md` by itself, so the context does not need to be explained again.

## Phase 4 — usable by other people, and showing the motion (16/11/2026)

Phases 2 and 3 produced a product that reads every known kind of output, with PSNR numbers to compare against. Phase 4 is the features and
packaging work that appear nowhere else in this file. Numbering starts at WP32 to avoid clashing with WP25–WP31 of phase 3.

### Do now

1. ~~**Bring branch `cc-a/wp24-points` into `main`** (human).~~ (Done: merged.) The two commits of 16/11 (WP24 + phase 3; SC-GS
   `--use_hash`, `--skinning`) were not in `main` yet.
   ```bash
   git status --short
   git add -A && git commit -m "PLAN.md: phase 4"        # if there are uncommitted changes
   git checkout main && git merge cc-a/wp24-points
   python tests/run_all.py && python tests/check_real.py
   ```
2. ~~**Measure FPS in points mode and at 1920×1080**~~ (done 17/11/2026; human, unlocked screen, GPU not training; WP28 of phase 3):
   `python tests/bench_display.py --draw-as points --rates 20 30 40 50 60`, then again with `--size 1920 1080`; the table is in
   `docs/PERFORMANCE.md`.
3. **Free disk space** (about 15 GB left): delete `../data_real/som/paper-windmill_data/paper-windmill.zip` (6.6 GB) and `sear_steak.zip`
   (1.1 GB) after checking that `check_real.py` still passes; both are fully extracted.
4. **Open three sessions** (after step 1): CC-C does WP32, CC-B WP33, CC-A WP34. The three packages touch no file in common.
5. **Demo for the professor** (human) as soon as WP32 is done, and fill in the feedback table below. The feedback reorders WP35–WP37 and
   WP40.

### Phase 4 status

| WP | Task | Status | Session | Notes |
|---|---|---|---|---|
| 32 | Demo scene set and a 10-minute demo script (`DEMO.md`) | Not started | CC-C | Preparation for the demo with the professor |
| 33 | Install on a clean machine with one command (`setup.sh`) | Not started | CC-B | Needs about 8 GB of free disk to try on a fresh clone |
| 34 | Camera path: set waypoints, play back, export an mp4 along the path | Done 16/11/2026 (branch `cc-a/wp34-path`) | CC-A | Closest to the request "move the camera to observe" |
| 35 | Motion trails and colouring by speed | Done 16/11/2026 (branch `cc-a/wp34-path`) | CC-A | After WP34 |
| 36 | Pick the scene in the GUI; compare up to four sequences | Not started | CC-A | After WP35 |
| 37 | Depth and opacity view modes | Not started | CC-A | Small; after WP36, or earlier if the professor needs it |
| 38 | Viewing from another machine over SSH | Not started | human | Needs a second machine |
| 39 | Run the tests automatically on push | Done 17/11/2026; runs on GitHub since 17/11, green; 23/11 added lint (ruff) and a job with CPU torch | CC-B | After WP33; needs the repository on GitHub |
| 40 | List of supported methods + guide and template for adding a converter | Done 17/11/2026 (tasks 1-2; task 3 waits for the professor's feedback) | CC-C | After WP32; which new method comes next: waiting for the feedback after the demo |

Order: round 1 is WP32, WP33, WP34 in parallel; the demo as soon as WP32 is done; round 2 is WP35 (CC-A), WP39 (CC-B), WP40 (CC-C);
round 3 is WP36, WP37 and WP38, reordered after the demo feedback.

### Professor's feedback after the demo

| Question | Answer | Date |
|---|---|---|
| What is still missing when looking at a result? | | |
| Is points mode (WP24) what "observe the 4DGS points" meant? Are motion trails (WP35) needed? | | |
| Are videos along a camera path (WP34), or several methods side by side (WP36), needed? | | |
| Besides the methods already supported, which methods does the lab need? (e.g. 4D-Rotor GS, E-D3DGS, MoSca) | | |
| Who will use it, on which machine? (decides WP33, WP38) | | |

### Phase 4 conventions

Drawn from what went wrong in phases 2 and 3:

- **Every session works in its own worktree** (`git worktree add ../gs4d-wp<N> -b cc-<session>/wp<N>-<name>`), and never checks out a
  branch in the main folder. Phases 2–3 piled WP22, WP24 and all of phase 3 onto one branch `cc-a/wp24-points`, while `main` had not moved
  since the morning of 16/11.
- **`PLAN.md` is edited only in the main folder, and a session edits only its own package's row** in the status table (replace that row,
  never rewrite the whole file). Twice a session has overwritten someone else's part of this file. Never reuse a WP number.
- Files each session may edit: CC-A: `viewer.py`, `gs4d/format.py`, the package's new `tests/check_*.py`; CC-B: `gs4d/open.py`,
  `gs4d/doctor.py`, `setup.sh`, `requirements.txt`, `tests/run_all.py`, `.github/`; CC-C: `DEMO.md`, `docs/`, `tests/make_demo_scenes.py`,
  `convert_template.py`. README: every session edits only its own package's paragraph. A change needed in another session's file goes into
  the report, not into the file.
- A package that adds a GUI feature comes with a `tests/check_*.py` script that takes headless screenshots as evidence, and is tried on at
  least one real scene.
- When a package is done, the human merges it into `main` and runs `python tests/run_all.py` before opening that session's next package.
- One shared GPU (RTX 3060 12 GB) and 15 GB of RAM: only one session at a time may train, run `check_real.py` or `bench_display.py`; ask
  the user first.

### Phase 4 prompts

Paste the whole block into a new Claude Code session opened in `gs4d_viewer`.


#### WP32 — demo scene set and demo script
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Task 1: write tests/make_demo_scenes.py, which builds the folder `../demo_scenes/` (outside the repo) from the real outputs already on this
machine: one compact `.npz` per method on `bouncingballs` (or the scene that method already has), plus the four `slice-banana` files of
Deformable-3DGS, SC-GS, native 4DGS and 4DGaussians. Each file comes with its camera file if there is one. Print a table: name, method,
number of Gaussians, number of keyframes, size.
Train nothing new; when an output is missing, skip it and print why.
Task 2: write DEMO.md, one page, for a 10-minute demo with the professor: 5–6 commands in order, each with one line "what it shows"
(open an output with one command; drag the time slider and orbit the camera; points mode; compare two methods on the same scene with
--compare; original image next to the render with PSNR; mp4 export), and one line on the known limits. Run every command for real before
writing it down.
Task 3: note in DEMO.md which scenes have their own terms of use (data showing people, research-only licenses: see the "License and data
sources" section of the README) so the presenter knows which scenes must not be shown publicly. Upload nothing.
Only add: tests/make_demo_scenes.py, DEMO.md, one line in the README pointing to DEMO.md.
Done when: the demo folder has every file, every command in DEMO.md runs on this machine, and the total size of the folder is written down.
Report in 5 lines, then stop.
```

#### WP33 — install on a clean machine with one command
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Task: write setup.sh, which creates `.venv` with the checked versions (Python 3.10, torch 2.4.1+cu124, gsplat 1.5.3, numpy<2, viser, and
the packages in requirements.txt), picks the non-CUDA branch by itself when the machine has no NVIDIA GPU (then only the client renderer is
available), and runs a smoke test: build toy.npz, `python tests/run_all.py`, and print clearly which renderers work. Add
`python -m gs4d.doctor`, which prints the package versions, CUDA, Chrome, and what to do when something is missing. Rewrite the
installation section of the README as: clone, `./setup.sh`, `python -m gs4d.open <output>`.
Step 0: `df -h .`; about 8 GB of free space is needed for a second `.venv` (the disk has about 15 GB left); if not, report and stop.
Check with a fresh clone in a temporary folder, without reusing the existing `.venv`; write down the install time and download size.
Only add or edit: setup.sh, gs4d/doctor.py, requirements.txt, the installation section of the README.
Done when: on the fresh clone the three commands above run from start to end and open toy.npz with the gsplat renderer. Report in 5 lines,
then stop.
```

#### WP34 — camera path
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Problem: the professor wants to move the camera to observe dynamic scenes; at present the camera is controlled only with the mouse and
videos are recorded only from a fixed camera.
Task: add a "Camera path" folder to the control panel of viewer.py: a button that adds a waypoint from the current camera, the list of
waypoints (jump to, delete, delete all), a button that plays the path, a "Path duration (s)" slider, and a choice of scene time while
following the path: frozen at the current time, or running from the start to the end of the sequence. Interpolate positions smoothly
through the waypoints, view directions with slerp, fov linearly. Save and load paths as JSON (`--camera-path file.json`), in the same
camera convention as `--cameras`. The mp4 export button gets an extra option "along the path". Works with both renderers, with `--compare`
and with points mode. Add one built-in path: one orbit around the scene centre.
Only edit viewer.py, add tests/check_path.py, one paragraph in the README; passing `--camera-path` through `gs4d/open.py` is CC-B's job
(note it in the report).
Done when: tests/check_path.py builds a 3-waypoint path on toy, checks the camera at 5 times against an independently computed
interpolation, and exports an mp4; run by hand on SC-GS `bouncingballs_sh` and `paper-windmill`, with the two mp4 files in shots/path/;
the README gives the export times. Report in 5 lines, then stop.
```

#### WP35 — motion trails and colouring by speed
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Precondition: WP34 is merged.
Problem: points mode (WP24) shows the positions at each time, but not how the scene moves.
Task 1: a "Colour by" dropdown: `colour` (as now), `speed` (magnitude of the velocity at the time on screen, from the two neighbouring
keyframes), `displacement` (offset from the first keyframe); with a colour scale and the maximum value shown in the control panel. Works in
both splats and points modes.
Task 2: a "Trails" checkbox: draw the paths of M Gaussians (slider for M; random choice with a fixed seed, preferring Gaussians that move a
lot and are opaque enough) in a time window ±Δ around the time on screen (slider for Δ), updated while playing. With the gsplat renderer
the image is the background, so the trails are drawn as viser line objects on top of it, without depth occlusion: state this limit
clearly.
Only for sequences with `interpolate=True` (the keyframes share Gaussian identities); for the others hide both controls and say why.
Only edit viewer.py (and gs4d/format.py if a velocity function is needed), add tests/check_trails.py, one paragraph in the README.
Done when: the test checks speed and trails on toy against independently computed values; screenshots in shots/trails/ for SC-GS
`bouncingballs_sh` and 4DGaussians `slice-banana`; the README gives the time per frame with trails on and M = 500. Report in 5 lines, then
stop.
```

#### WP36 — pick the scene in the GUI; compare more than two sequences
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Precondition: WP35 is merged.
Task 1: `python viewer.py <folder>` (a folder with several `.npz` files in the common format, e.g. ../demo_scenes) shows a "Scene"
dropdown that switches scenes without restarting; lazy loading, the old scene is released; the camera stays if both scenes share a
coordinate system (same name prefix), otherwise it is reset.
Task 2: `--compare` takes several files (up to 4): the gsplat renderer splits the view into a 2x2 grid with the same camera, the file name
in the corner of each cell; the client renderer puts them side by side. The original image (`--cameras`) is one cell of the grid when
switched on. mp4 export and the camera path (WP34) work with the grid.
Only edit viewer.py, tests/check_compare.py, one paragraph in the README; making `gs4d/open.py` accept a folder of several raw outputs is
CC-B's job.
Done when: tests/check_compare.py passes with 3 and 4 sequences; a screenshot of a grid of four methods on `slice-banana` in
shots/compare/; the README gives the time per frame of the 2x2 grid at 1280x720 and the RAM / GPU memory used. Report in 5 lines, then
stop.
```

#### WP37 — depth and opacity view modes
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Task: a "Render" dropdown for the gsplat renderer: `colour`, `depth` (expected depth, colour scale, near/far taken from the 2–98%
percentiles of the frame and shown in the control panel), `alpha` (accumulated opacity). Read the documentation of the installed gsplat
1.5.3 to use `render_mode` correctly; do not recompute depth yourself.
Works while playing, with `--compare`, the camera path and mp4 export. Client renderer: hide this control.
Only edit viewer.py, add tests/check_depth.py, one paragraph in the README.
Done when: the test compares the depth at the image centre of toy with an independently computed distance; screenshots in shots/depth/
for `paper-windmill` and one `slice-banana` scene. Report in 5 lines, then stop.
```

#### WP38 — viewing from another machine over SSH (human)

On a second machine: `ssh -L 8080:localhost:8080 <this machine>`, open `http://localhost:8080` in Chrome, and repeat the WP19b measurement
steps at 1280×720. Done when the README has the frames/s shown over SSH next to the same-machine numbers, with the kind of network (LAN,
Wi-Fi, VPN) and the bitrate if it can be read. If it is clearly slower, open a new package for CC-A: lower the JPEG quality or the
resolution while the camera is moving.

#### WP39 — run the tests automatically on push
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Precondition: WP33 is merged. Task: add .github/workflows/test.yml, running on every push and pull request: install the non-CUDA build
with setup.sh (or the matching part of it), run `python tests/run_all.py` (items that need CUDA, a GPU-accelerated Chrome or
../data_real skip themselves with a reason), Python 3.10. Add a status badge to the README. Do not push, do not create a repository, do
not change GitHub settings: only add the file; pushing is the user's job.
Done when: the equivalent command runs in a fresh clone without CUDA on this machine, and the list of skipped items is in the report.
Report in 5 lines, then stop.
```

#### WP40 — list of supported methods and a guide to adding a converter
```
Read CLAUDE.md, the "Phase 4" section (conventions) and the "Done criteria for every package" section of PLAN.md. Work in your own worktree.
Task 1: write docs/SUPPORTED.md: a table of every supported method and variant (which real output it was checked on, PSNR, limits), and
the variants the converters stop on with a message. Take the numbers from the README and the Status section of CLAUDE.md, do not measure
again.
Task 2: write docs/ADD_A_CONVERTER.md (the steps, the conventions written in CLAUDE.md, the lessons: check the original repository's
cameras before trusting its evaluation numbers, measure at the training resolution) and convert_template.py + tests/test_template.py,
which run on fake data and use `Sequence4DWriter`.
Task 3: for the methods the professor names in the table "Professor's feedback after the demo" (if the table is empty, stop after task 2
and say so): for each method one short paragraph: official repository, output format, which type A/B/C/D it is, estimated effort to write
the converter, whether it can be trained on a machine with a 12 GB GPU / 15 GB RAM. Look at the real repository, do not write from
memory; do not write a converter in this package.
Only add: docs/, convert_template.py, tests/test_template.py, one line in the README.
Done when: the template's test passes and is part of `python tests/run_all.py`. Report in 5 lines, then stop.
```

## Phase 3 — paying back the leftover work (16/11/2026)

Phase 3 adds no new features: it collects the items phase 2 marked "not done", "not measured" or "needs another session to fix".
One session did all of it, on branch `cc-a/wp24-points`, not committed. The convention "every session edits only its own files" does not
apply to this phase, since there was only one session.

| WP | Task | Status | Notes |
|---|---|---|---|
| 25 | `check_points.py` in `tests/run_all.py` | Done | Added to `NEEDS` (needs Chrome): a machine without Chrome reports skipped instead of failed |
| 26 | `estimate_bytes` counts `sh` | Done | `gs4d/open.py`: `--sh` adds 192 B per Gaussian (native 4DGS: per Gaussian per keyframe); `tests/test_open.py` |
| 27 | Camera list for HyperNeRF datasets in `gs4d.open` | Done | 4DGaussians, Deformable-3DGS and SC-GS folders with a Nerfies-style `source_path`: writes `<name>.gs4d.cameras.json` (held-out cameras) and passes `--cameras`. `slice-banana`: PSNR through this list for 4DGaussians 27.13 dB, the same as `check_real.py`; the cameras of the three models match the loader of `check_real.py` to 3e-7 |
| 19c part 2 | PSNR of Spacetime lite `sear_steak` | Done | Downloaded `sear_steak.zip`, 1.11 GB (the GitHub release of Neural 3D Video). 105 images at 1352×1014: 26.18–33.53 dB, mean 31.43. Item `spacetime_psnr`, min threshold 25.68 |
| 19c part 1 | PSNR of Shape of Motion `paper-windmill` | Done | Downloaded `paper-windmill.zip`, 6.61 GB, from a Hugging Face mirror (over the 5 GB limit of the original prompt; the user agreed). 70 images of 720×960 at the training cameras: keyframes 25.86 dB; between keyframes of `--stride 8` 24.67 dB; `--stride 1` 25.85 dB everywhere. Item `som_psnr`, min threshold 21.11 |
| 28 | Display FPS in points mode and at 1920×1080 | Done 17/11/2026 (table in `docs/PERFORMANCE.md`) | Needed a human: with the screen locked, Chrome was throttled to 1 frame/s. Measured with the screen unlocked: `python tests/bench_display.py --draw-as points --rates 20 30 40 50 60`, then again with `--size 1920 1080` |

### Phase 3, round 2 — the three remaining limits (16/11/2026, same session, same branch, not committed)

| WP | Task | Status | Notes |
|---|---|---|---|
| 29 | Original-image view usable with `--compare` | Done | `viewer.py::_gt_view`: the `A \| B` mode shows render A \| original image \| render B, with the PSNR of both in the panel; `gs4d.open` passes `--cameras` even with `--compare`. `tests/check_cameras.py`: the PSNR of A and B equals the single-file viewer's (≤ 1e-3 dB, 20 images); image `shots/cameras/ground_truth_compare.png` |
| 30 | SC-GS `--pred_color` | Done for the formula; the real model cannot tell the difference | `convert_scgs.py` re-runs the colour head (the default "resnet" type and the single linear-layer type), adds it to the SH DC coefficient, and writes `colors` (and `sh` with `--sh`) per keyframe; `--ignore-pred-color` keeps the colour fixed; hash / view-direction types stop. Synthetic test against an independent formula. Real model `bouncingballs` (20k iterations, 20 minutes): 35.62–43.75 dB, mean 40.70, but its colour head is still at its initial values (\|d_color\| ≤ 2.3e-4), so dropping the colour head also gives 40.70. Item `scgs_predcolor` (threshold 35.12) |
| 31 | PSNR from the repositories' own renderers | Done, and found a bug | `tests/repo_psnr.py` (not in `run_all.py`; clones in `/home/cont/gs4d_train/som`, `stg`). **Shape of Motion**: repo 25.85 dB, our `--stride 1` file differs by 0.000 dB on every image. **Spacetime lite**: repo 34.48 dB, ours 31.43 dB → cause: the converter clipped the Gaussian colours to [0, 1] (30% of the Gaussians have colours outside that range; the repo blends raw colours). Fixed `convert_spacetime.py` (keeps raw colours), `gs4d/format.py` (per-keyframe colours outside [0, 1] stay float32 when compact), `viewer.py` (the client clips colours), `gs4d/open.py` (a cache older than the converter is converted again): now 34.37 dB; the remaining 0.11 dB comes from `--min-peak-opacity 0.02` |

Left after phase 3 (not Claude Code's job, or not possible yet):

1. ~~**WP28** above~~ (done 17/11/2026; 5 minutes, needs an unlocked screen and a GPU that is not training).
2. ~~**Commit**: branch `cc-a/wp24-points` holds WP24 and all of phase 3, not committed.~~ (Done: merged into `main`.)
3. **Clean up worktrees** (item 5 of "Do now (phase 2)") and the disk: of `../data_real/som/paper-windmill_data/paper-windmill.zip`
   (6.6 GB) only 384 MB of extracted files is needed (`rgb/1x/0_*`, `splits/`); `sear_steak.zip` (1.1 GB) is fully extracted. About 15 GB
   of disk left.
4. **Known limits**: the full Spacetime model (`rgbdecoder`); SC-GS: `--use_hash` and `--skinning` are ported but only checked on
   synthetic data (the repo's own training: hash does not converge, 6.8 dB; skinning crashes after densification because of
   `motion_mask`), view-direction colour heads; the `--pred_color` colour head was checked on a scene tinted over time (37.83 dB against
   24.48 when the colour head is ignored).
5. Not measured: viewing over SSH or on another machine; the client renderer in points mode on a real GPU.

## Do now (phase 2)

1. ~~**WP20**~~ (done 15/11/2026) (one session, needs a free GPU for about 30 minutes): retrain native 4DGS `slice-banana` with the right
   cameras. The current 24.23 dB is measured on a model trained with wrong cameras. This is the only package still blocking the native 4DGS
   row of WP9.
2. **WP19b: done 15/11/2026.** `python tests/bench_display.py --rates 20 30 40 50 60` (a GPU-accelerated Chrome window; needs an unlocked
   screen and a GPU that is not training). `paper-windmill` 1280×720: the browser shows every frame the server sends, from 20.0 at 20 Hz up
   to 52.9 frames/s at 60 Hz (the server sends 54.3). Table in `docs/PERFORMANCE.md`.
3. **Open in parallel with WP20** (no GPU use while it trains): ~~WP22~~ (CC-B, four converters, done 15/11/2026). WP21 (CC-A) was done
   15/11/2026, waiting for branch `cc-a/wp21-compact-sh` to be merged.
4. **After WP20, one package at a time because of the shared GPU:** ~~WP23~~ (done 15/11/2026) → ~~WP19c part 1~~ → ~~WP19c part 2~~
   (done 16/11/2026, phase 3).
5. **Clean up worktrees** (human). Git still has 10 worktrees registered, including `gs4d-wp19c` (merged) and `gs4d-wp2` from phase 1.
   ```bash
   git worktree list
   git branch --no-merged main        # must be empty before removing
   git worktree remove ../gs4d-wp19c && git worktree remove ../gs4d-wp2
   git worktree remove <path of each merged claude/* worktree> && git worktree prune
   ```
6. ~~**Confirm one of the professor's points** before opening WP24~~ (confirmed, WP24 done 16/11/2026): whether "observe the 4DGS points"
   means that a mode drawing the Gaussian centres as points is needed (see the note under the answers table).

## WP9 — the product: read and view every 4DGS file

Changed on 14/11/2026: WP9 used to be "demo + settle the questions with the professor"; it is now the product goal.

**Goal.** Give any 4DGS output to `python -m gs4d.open <output>` and view it with a time slider, without installing the method's
environment or running the original repository's code. WP9 has no branch of its own: it is done when every row of the table below is
"Done".

"Every file" means the kinds in this table. A new method gets a row, a `convert_<method>.py` and a detection branch in
`gs4d/open.py`.

| File kind | Converter | Now | Missing | Package |
|---|---|---|---|---|
| Folder of per-frame 3DGS ply files (any method can export one) | `convert_per_frame.py` | Done: the `gaussian_pertimestamp/` of 4DGaussians `bouncingballs`, item `4dgaussians_export` | — | — |
| Dynamic 3D Gaussians `params.npz` | `convert_per_frame.py` | Done (`basketball`) | — | — |
| Shape of Motion `last.ckpt` | `convert_shape_of_motion.py` | Done (`paper-windmill`): PSNR at the training cameras against the DyCheck iPhone images 25.86 dB at keyframes, 24.67 dB between keyframes of `--stride 8` (WP19c part 1, 16/11/2026) | — | — |
| Native 4DGS `chkpnt*.pth` | `convert_native4d.py` | `--sh`: `bouncingballs` 27.3–39.6 dB, mean 32.9 (original repo 33.4). Real scene `slice-banana` (retrained with the right cameras, WP20): 26.31 dB (110 keyframes), the original repo's own evaluation 26.30 | Done. The old 1.9 dB gap came from the old layout of `make_hypernerf_blender.py` (`FovX = FovY = -1`), fixed and retrained. 330-keyframe file: 19.9 → 5.8 GB since WP21 (`sh` compacted to uint8) | WP20 |
| 4DGaussians (hustvl), model folder | `convert_4dgaussians.py` | Converter done: `bouncingballs` 36.7–45.9 dB, mean 40.8; matches `export_perframe_3DGS.py` to 1e-6 | Done (WP19a): `gs4d.open` recognises this folder and opens it with a camera list. Real scene `slice-banana` (WP23): 22.05–30.39 dB, mean 27.13, matches the repo; `gs4d.open` writes the camera list for the HyperNeRF layout by itself (WP27) | — |
| Spacetime Gaussians `point_cloud.ply` | `convert_spacetime.py` | Lite model: `n3d_sear_steak` 27.76–37.81 dB, mean 34.37 on 105 images of the original videos (the repo's renderer: 34.48; WP31, 16/11/2026; before the colour-clipping fix: 31.43); Technicolor `Birthday` checked by eye. Full model (`rgbdecoder`): the converter stops and names it, `--base-color-only` converts anyway (WP19a) | The full model is a known limit: no public model, training needs more than 15 GB of RAM | — |
| Deformable-3DGS, model folder | `convert_deformable3dgs.py` | Done: D-NeRF 37.2–43.6 dB; default 35.35–43.14; 6-DoF 36.99–43.75; real scene `slice-banana` 19.28–29.78 (mean 24.72) | — | — |
| SC-GS, model folder | `convert_scgs.py` | Done: D-NeRF 35.4–43.0 dB; motion mask 35.48–43.14; real scene `slice-banana` mean 26.69 dB (110 keyframes), 26.87 dB (330 keyframes, peak RAM 1.4 GB since WP19d). `--use_hash`, `--skinning`, `d_rot_as_res` off, `--pred_color` (`--ignore-pred-color` converts anyway): stop with a message | — | — |
| A file of none of the kinds above | `gs4d/open.py` | Reports which detections were tried | — | — |

**WP9 done criteria** (in addition to "Done criteria for every package"):

1. Every row opens with one command `python -m gs4d.open <output>` on a real output, and has an item in `tests/check_real.py`.
2. A variant the converter does not support yet stops with a message that names the variant; it must never give a wrong image without a
   word.
3. The "Use with real outputs" table of the README (since 23/11/2026: `docs/SUPPORTED.md`) lists every row above, with measurements.

When a row has no real output, train one on `bouncingballs` in `/home/cont/gs4d_train` (one job at a time, 15 GB RAM), as was done for
Deformable-3DGS, SC-GS and native 4DGS.

### The professor's answers (15/11/2026)

| Question | Answer | Date | Consequence for the plan |
|---|---|---|---|
| Which 4DGS methods does the lab use? | All of them | 15/11/2026 | No method of the WP9 table is dropped. The packages that used to say "when the lab uses it" are all open: WP19c parts 1–2, WP21, WP22 |
| Where are the output files, which scenes, how many Gaussians? | The lab has no outputs yet: run the methods ourselves and use those outputs | 15/11/2026 | Keep training in `/home/cont/gs4d_train` and use the checkpoints the authors published. WP23 added so that 4DGaussians also has a real scene, like the three others. The full Spacetime model stays a limit (training needs more than 15 GB of RAM) |
| On which machine will it be viewed? | This machine for now (RTX 3060 12 GB, 15 GB RAM) | 15/11/2026 | The `gsplat` renderer is the main path; WP19b measures in this machine's Chrome. Viewing over SSH or on a laptop does not need checking yet |
| What else is needed when viewing? | Moving the camera to observe the 4DGS points | 15/11/2026 | See the note below |

Note on the last answer. The viewer already lets the camera move freely with the mouse (viser's orbit, pan and zoom) even while playing,
with both renderers; if that is all, nothing more is needed. The viewer had no mode that draws the Gaussian centres as points: if "the 4DGS
points" means seeing every Gaussian as a point cloud, that is WP24 (done 16/11/2026: the **Draw as** control in the Display folder).

For the demo: `python -m gs4d.open <output folder>` (picks the `gsplat` renderer on this machine by itself), the `--compare` mode,
and the "Use with real outputs" table of the README (since 23/11/2026: `docs/SUPPORTED.md`).

## Phase 2 status

| WP | Task | Status | Branch | Notes |
|---|---|---|---|---|
| 9 | Product: read and view every 4DGS file | Done 16/11/2026, except the known limits | — | Every row of the WP9 table has a PSNR against the original images and an item in `check_real.py`. Limit: the full Spacetime model (`rgbdecoder`) has no public model, training needs more than 15 GB of RAM |
| 10 | Command that opens an output directly | Done | `claude/practical-wiles-00a752` | `python -m gs4d.open <output>`; the WP10 session reported opening all 6 real outputs |
| 11 | Split static and dynamic Gaussians in the format | Done | `cc-a/wp11-static` | `paper-windmill` 1271 → 147 MB; `viewer.py` unchanged. Before/after images differ by more than 1/255 on at most 7 pixels between two keyframes (from the old code's rounding) |
| 12 | Automatic checks with real outputs (PSNR with thresholds) | Done | `claude/friendly-feynman-9aa271` | Every model measured at its training resolution; 20 test images; Spacetime and Shape of Motion only check the Gaussian count and the file size |
| 13 | Native 4DGS: per-keyframe SH colour | Done (merged) | `cc-c/wp13-native4d-sh` | `--sh`: PSNR `bouncingballs` min 26.04 → 27.28, mean 29.0 → 32.9 dB; file 1.4 GB (116k x 60 keyframes) |
| 14 | GPU decoding and interpolation while playing | Done | `claude/wp14-gpu-decode-559de8` (merged) | Frame fetch while playing 100 → 2.2 ms (500,000 × 100, compact), 214 → 3.1 ms (`paper-windmill`); images differ by at most 1/255; check_real PSNR unchanged. FPS in a GPU-accelerated Chrome: measured in WP19b |
| 15 | Direct converter for 4DGaussians (hustvl) | Done (merged) | `cc-c/wp15-4dgaussians` | Self-trained `bouncingballs`: 36.7–45.9 dB (mean 40.8), matches `export_perframe_3DGS.py` to 1e-6. `gs4d/open.py` recognises 4DGaussians folders since WP19a |
| 16 | Untried variants (full Spacetime, Deformable-3DGS, SC-GS) | Done, except full Spacetime (merged) | `claude/wp16-variants-8fe2f3`, `claude/wp16-real-scene-5cd4d8` | `bouncingballs`: Deformable-3DGS default 35.35–43.14 dB, 6-DoF 36.99–43.75, SC-GS with motion mask 35.48–43.14. Real HyperNeRF scene `slice-banana` (15/11/2026, 33 held-out images, 268 px): Deformable-3DGS mean 24.72 dB, SC-GS 26.69, native 4DGS 24.23 (model trained with wrong cameras, see WP20). Spacetime: added the lite model `Birthday`, no original images yet; the full model (`rgbdecoder`) is not possible on this machine |
| 17 | `--compact` flag for the converters, one command that runs every test | Done | `claude/wp17-housekeeping-56ac95` | `python tests/run_all.py`: 13 files pass (check_real run separately, after fixing the data lookup in worktrees); `--compact` in all 6 converters, `gs4d.open` uses it when the estimate is > 1 GB |
| 18 | Training cameras and original images in the viewer | Done 14/11/2026 (merged) | `claude/wp18-cameras-398403` | `--cameras`, camera list, original image + PSNR (gsplat); PSNR matches check_real to 0.0002 dB; not usable with `--compare`; `gs4d/open.py` passes `--cameras` by itself since WP19a |
| 19a | `open.py` recognises 4DGaussians and `--cameras`; converters stop on unsupported variants | Done 15/11/2026 (committed) | directly on `main` | `gs4d.open` recognises 4DGaussians and passes `--cameras` (skipping 3DGS-style `cameras.json`, which `load_cameras` cannot read); SC-GS `--pred_color` and full Spacetime stop, with a flag to convert anyway |
| 19b | Measure display FPS in a GPU-accelerated Chrome | Done 15/11/2026 (committed 086a80b) | directly on `main` | `tests/bench_display.py`; `paper-windmill` 1280×720, Chrome 157, RTX 3060: shows 20.0 / 30.0 / 39.9 / 49.3 / 52.9 frames/s at 20 / 30 / 40 / 50 / 60 Hz, the same as the server sends; 1920×1080 not measured |
| 19c | PSNR for Spacetime and Shape of Motion; cause of the 1.9 dB gap of native 4DGS on a real scene | Done: part 3 15/11/2026 (merged); parts 1, 2 16/11/2026 (not committed) | parts 1, 2 done on `cc-a/wp24-points` | Part 1: Shape of Motion `paper-windmill` against the DyCheck iPhone 1x images, 70 images: at keyframes 25.86 dB, between keyframes of `--stride 8` 24.67 dB (`--stride 1`: 25.85 everywhere); item `som_psnr`. Part 2: Spacetime lite `sear_steak` against the Neural 3D Video videos, 105 images at 1352×1014: 26.18–33.53, mean 31.43 dB; item `spacetime_psnr`. Part 3: the gap comes from a model trained with `FovX = FovY = -1`, not from the converter |
| 19d | Converters write one keyframe at a time to disk instead of holding the whole sequence in memory | Done 15/11/2026 (committed) | directly on `main` | `gs4d.Sequence4DWriter`, `tests/test_writer.py` (bit-identical to `save`); SC-GS, native 4DGS and Deformable-3DGS use it. Peak RAM: SC-GS 4.5 → 1.4 GB, native 4DGS 8.3 → 1.3 GB, Deformable-3DGS 7.2 → 1.0 GB; 330 keyframes of `slice-banana` can be converted. `check_real.py` still measures at 110 keyframes |
| 20 | Retrain native 4DGS `slice-banana` with the right cameras, measure again | Done 15/11/2026 (committed f24bd6d) | `cc-c/wp20-native4d-banana` | Retrained with `camera_angle_x` (no per-frame `fl_x`), 20,000 iterations, 20 minutes: the repo's own evaluation 26.30 dB; converter + gsplat 21.41–29.17, mean **26.31 dB** (before: 24.23): the 1.9 dB gap is gone. `make_hypernerf_blender.py` fixed; `check_real.py` points to `output/banana_fov`, min threshold 20.91 |
| 21 | Compact the per-keyframe `sh` field | Done 15/11/2026 (committed 5841d10) | `cc-a/wp21-compact-sh` | Per-keyframe `sh` becomes uint8 over each coefficient's own range in time (`sh_lo`, `sh_step`, `sh_u8`). Native 4DGS `slice-banana` 330 keyframes: 19.86 → 5.82 GB (0.29×), PSNR 24.2687 dB unchanged (worst image −0.0002 dB); `bouncingballs`: 1412 → 453 MB, 32.9117 dB unchanged. Frame fetch on the GPU path 2.2 → 1.4 ms; CPU path 60 → 128 ms. Left for CC-B: `estimate_bytes` in `gs4d/open.py` does not count per-keyframe `sh` yet |
| 22 | `Sequence4DWriter` for the four remaining converters | Done 15/11/2026 (merged 08b0512) | `claude/wp22-writer-rest-f7db21` | 4DGaussians, Shape of Motion, Spacetime and per-frame write one keyframe at a time to disk; details in `docs/PERFORMANCE.md` |
| 23 | 4DGaussians on the real scene `slice-banana` | Done 15/11/2026 (committed f24bd6d) | directly on `main` | Trained for 14,000 iterations (the repo's hypernerf config), 18 minutes, 52,741 Gaussians. PSNR on the 33 held-out images: ours 22.05–30.39, mean 27.13; the repo's own renders 27.13 (≤ 0.003 dB apart per image). No converter change; item `4dgaussians_banana` in `check_real.py`. The repo's cameras were checked: FovX/FovY computed from the focal length, correct |
| 24 | Draw Gaussian centres as points | Done 16/11/2026 (not committed) | `cc-a/wp24-points` | **Draw as** control `splats` / `points` + **Point size** slider in the Display folder (`--draw-as`); works while playing, with `--compare`, mp4 export, both renderers. gsplat: rasterises round opaque Gaussians of a fixed pixel size; client: viser's point cloud. `tests/check_points.py` passes on toy, SC-GS `bouncingballs_sh`, `paper-windmill`; one frame while playing at 1280×720: SC-GS 10–11 ms, `paper-windmill` 24–27 ms. The README has a table of camera controls (tried for real). Display FPS in points mode: WP28 |

## Assignment: 3 Claude Code sessions

| Session | Role | Model | Tasks, in this order | Files it may edit (this session only) |
|---|---|---|---|---|
| **CC-A** | Format and rendering | strong (Opus/Fable) | WP11 → WP14 → (WP18) | `viewer.py`, `gs4d/format.py`, `gs4d/__main__.py`, `tests/test_compact.py`, `tests/bench_large.py`, the body of the README |
| **CC-B** | Everyday tools | Sonnet | WP10 → WP17 | new file `gs4d/open.py`, `tests/test_open.py`, `tests/run_all.py`; in WP17: the command-line arguments of the `convert_*.py` |
| **CC-C** | Validation and converters | Sonnet | WP12 → WP15 → WP13 → WP16 (one variant at a time) | new file `tests/check_real.py`; `convert_native4d.py` (WP13); new files `convert_4dgaussians.py`, `tests/test_4dgaussians.py` (WP15) |

Shared conventions (kept from phase 1, corrected where they had drifted):

- A session edits only the files in its own column. A change needed in another session's file goes into the report, not into the file.
- Branches are named `cc-<session>/wp<N>-<name>`. CC-B and CC-C work in their own worktrees (`git worktree add ../gs4d-wp<N> -b <branch>`),
  and remove the worktree right after the merge.
- All three sessions share one GPU (RTX 3060) and 15 GB of RAM: no training; render or PSNR commands run one at a time.
- WP17 edits the `convert_*.py`, so it runs only when CC-C has no converter package open.
- The WP9 packages (WP13, WP15, WP16) can start right away, one at a time because of the shared GPU. Exception to the "no training" line
  above: a package without a real output may train a `bouncingballs` model in `/home/cont/gs4d_train`, when no other session uses the GPU.

## Order

- **Round 1 (in parallel, without waiting for the professor):** WP10 + WP11 + WP12. The three packages touch no file in common and help
  every method.
- **Round 2 (WP9, without waiting for the professor):** CC-A does WP14; CC-B has finished WP17; CC-C does WP15 → WP13 → WP16, one package at
  a time. WP15 goes first because 4DGaussians is the only method that cannot be read directly yet; the professor's answers may change this
  order.
- **Round 3 (when asked):** WP18 (done 14/11/2026).
- **Round 4 (15/11/2026):** WP19a, WP19d and part 3 of WP19c done. WP19b done 15/11/2026 (`tests/bench_display.py`).
- **Round 5 (15/11/2026, after the professor's answers):** WP20 first. In parallel with it, without using the GPU while it trains: WP21
  (`gs4d/format.py`) and WP22 (four converters); the two packages touch no file in common. After WP20, one after another because of the
  shared GPU: WP23 → WP19c part 1 → WP19c part 2. WP21 and WP22 run `check_real.py` at the end, when no package is training. WP24 done
  16/11/2026.

## Done criteria for every package

Phase 1 asked only for synthetic tests, so running on real outputs had to be caught up later. From now on:

1. A package that touches a converter or the format is done only when it has run on **at least one real output** in `../data_real`
   and its measurements (PSNR against the original images, size, time) are written down in `docs/` (`SUPPORTED.md`, `PERFORMANCE.md`;
   before 23/11/2026: the README).
2. A package that changes the format or the renderer reruns `tests/check_real.py` (once WP12 exists) and must not lower any PSNR by more
   than 0.2 dB.
3. Every old test still passes; old `.npz` files can still be read.
4. Update the package's row in the status table above, and add a dated entry at the end of `docs/HISTORY.md` (before 23/11/2026: the
   Status section of `CLAUDE.md`).
5. Report in 5 lines, then stop; do not widen the scope. Missing data or GPU: say so in the first turn, do not guess.

## Prompts for each session

Paste the whole block into a new Claude Code session opened in `gs4d_viewer`.

### WP10 — command that opens an output directly
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md.
Task: write gs4d/open.py so that `python -m gs4d.open <output_path> [viewer arguments]` recognises the kind of output by itself,
calls the right converter, then opens the viewer. Do not edit the converters or viewer.py: call them through subprocess.
Detection: `params.npz` or a folder of per-frame ply files -> convert_per_frame; `chkpnt*.pth` -> convert_native4d;
`last.ckpt` or a folder with `checkpoints/last.ckpt` -> convert_shape_of_motion; a folder with `point_cloud/` and `deform/`
-> convert_scgs if the ply has `fea_*` columns, otherwise convert_deformable3dgs; a `point_cloud.ply` with the motion attributes
of Spacetime Gaussians -> convert_spacetime; an `.npz` file already in the common format -> open it directly.
The converted file is saved next to the output (`<name>.gs4d.npz`) and reused while the output is unchanged (compare mtimes). If nothing
is recognised, print the signs that were checked and exit with an error code; do not guess. D-NeRF scenes: set the time range [0, 1].
Only add: gs4d/open.py, tests/test_open.py, an "Opening an output directly" paragraph at the top of the README.
Done when: the test builds a fake layout of every kind and checks the detection; and all 6 real outputs in ../data_real
open with exactly one command (list the commands and results in the report). Report in 5 lines, then stop.
```

### WP11 — split static and dynamic Gaussians
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md.
Problem: when a field has the K axis, every Gaussian is stored at every keyframe. Shape of Motion `paper-windmill` has
71,867 dynamic foreground Gaussians and 924,075 static background Gaussians, file 1.27 GB.
Task: let Sequence4D store a group of static Gaussians (without the K axis) next to the dynamic group, in the same file;
`at(t)` returns both groups joined, so viewer.py does not have to know. When saving, find the Gaussians that do not change over the
keyframes (every field, difference within float32 precision) and put them in the static group; with a flag to turn it off. The compact
layout and lazy loading keep working. Old files can still be read; `python -m gs4d in.npz out.npz` converts an old file to the new layout.
Only edit gs4d/format.py, gs4d/__main__.py, tests/test_compact.py (or add tests/test_static.py); edit viewer.py only
if it cannot be avoided, and say why.
Done when: a round-trip test with a fake scene that has both groups; the README has a before/after table for ../data_real/som/paper-windmill.npz
and ../data_real/d3dgs/bouncingballs.npz (size, RAM after loading, `at(t)` time); gsplat renders of the file
before and after at 3 different times differ by at most 1/255 per channel. Report in 5 lines, then stop.
```

### WP12 — automatic checks with real outputs
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md.
Task: write tests/check_real.py, which turns the comparisons done by hand on 14/11/2026 into one command. For every real output
with original images in ../data_real (Deformable-3DGS, SC-GS, native 4DGS on `bouncingballs`): run the converter, render with
gsplat at the camera and time of EVERY test image (not just 5), compute the PSNR of each image, print a table (min, mean,
max), and return an error code if the minimum PSNR is below a threshold. Initial threshold = the minimum measured this time minus 0.5 dB,
written as a constant at the top of the file with the date of the measurement. For Spacetime Gaussians and Shape of Motion (no PSNR
comparison yet): use the training cameras in `cameras.json` / the checkpoint and the original images if they are in ../data_real; without
original images only check that the converter runs and that the Gaussian count and the file size match the README, and say clearly that
no images were compared.
Without CUDA or without a data folder, skip that item and print why; never report it as passed.
Only add: tests/check_real.py, one line in the "Tests" section of the README. Do not edit the converters; if a converter is wrong, say so
in the report.
Done when: the command runs through on this machine, the PSNR table is pasted into the report, and the numbers in the README match the
table (if they differ, say which, and do not edit the README beyond your own line). Report in 5 lines, then stop.
```

### WP13 — native 4DGS: per-keyframe SH colour (part of WP9)
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md. Precondition: WP12 is merged.
Problem: convert_native4d.py writes only the SH degree-0 colour; the PSNR on `bouncingballs` is 26.0–31.1 dB, the original repo's own
evaluation gives 33.4 dB.
Task: add a --sh flag. Read the fudan-zvg/4d-gaussian-splatting repository to port the view- and time-dependent colour;
at every keyframe, precompute the time-dependent part and write the remaining 3D SH coefficients into the `sh` field [K,N,S,3].
Do not import the repository's code. Only edit convert_native4d.py, add tests/test_native4d_sh.py, edit the native 4DGS row
of the README and the matching threshold in tests/check_real.py.
Done when: the test compares the colour at 3 times and 3 view directions with an independently written formula; tests/check_real.py gives
a higher minimum PSNR on `bouncingballs` than before. Target mean >= 32 dB; if it is not reached, report the remaining cause
(e.g. the marginal_t filter, interpolation between keyframes) instead of tuning until it matches. Report in 5 lines, then stop.
```

### WP14 — GPU decoding and interpolation while playing
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md. Precondition: WP11 is merged.
Problem: `at(t)` on the compact layout takes 77 ms with 500,000 Gaussians (float32: 14 ms), so playing a large scene reaches only about
13 frames/s even before rasterisation.
Step 0: measure where the time goes (disk reads, decoding, interpolation, copy to the GPU) with tests/bench_large.py; write the numbers down.
Task: for the gsplat renderer, keep the two neighbouring keyframes as tensors on the GPU, decode and interpolate with torch, and load a
keyframe only when the time slider crosses into a new interval; copy the static group (WP11) to the GPU once. The client renderer and the
CPU path stay as they are.
Only edit viewer.py, gs4d/format.py, tests/bench_large.py.
Also measure the display rate in a real browser (Chrome with a GPU, not headless swiftshader) at 1280x720 and write it
next to the server-side numbers in the README.
Done when: the README has a before/after table of the time per frame while playing (500,000 Gaussians x 100 keyframes, float32 and compact)
and with ../data_real/som/paper-windmill.npz; target for fetching a frame <= 20 ms on the compact layout; renders differ by at most
1/255 per channel from the CPU path. Report in 5 lines, then stop.
```

### WP15 — direct converter for 4DGaussians (part of WP9)
```
Read CLAUDE.md and the "WP9" and "Done criteria for every package" sections of PLAN.md.
The real output is at: <PATH>. If there is none: train hustvl/4DGaussians on D-NeRF `bouncingballs` in /home/cont/gs4d_train
(one job, nothing else on the GPU), then run `export_perframe_3DGS.py` once to get per-frame ply files as a reference. Task: write
convert_4dgaussians.py, which reads a 4DGaussians model folder (hustvl/4DGaussians: `point_cloud/iteration_*/point_cloud.ply` + the
weights of the deformation network) and writes a Sequence4D at K times, so that `export_perframe_3DGS.py` no longer has to run in the
original repository's environment. Same constraints as the Deformable-3DGS converter: read the repository to port the formulas
(multi-resolution HexPlane + MLP deforming position/rotation/scale), rewrite the forward pass with torch on the CPU, run in batches, do not
import the repository's code. Only add: convert_4dgaussians.py, tests/test_4dgaussians.py, one README row,
two items in tests/check_real.py (the new converter, and the per-frame ply folder through convert_per_frame.py). The detection branch in
gs4d/open.py is CC-B's file: note it in the report.
Done when: a synthetic test against an independently written formula; and with the real output above, the result matches the per-frame
ply files written by `export_perframe_3DGS.py` (if available), or the PSNR against the test images is written into the README. Report in 5
lines, then stop.
```

### WP16 — untried variants (part of WP9; one variant per session)
```
Read CLAUDE.md and the "WP9" and "Done criteria for every package" sections of PLAN.md.
Method and variant: <ONE of: full Spacetime Gaussians with rgbdecoder | Deformable-3DGS default | Deformable-3DGS
6-DoF | SC-GS with motion mask | SC-GS --use_hash | SC-GS --skinning | SC-GS --pred_color | SC-GS with d_rot_as_res off |
a real (non-D-NeRF) scene for <name>>. The real output is at: <PATH>; if there is none, train that variant in
/home/cont/gs4d_train (one job).
Task: run the matching converter on this output, compare with the original images through tests/check_real.py (add an item), and fix the
converter and its test if the real format differs from what was assumed. A variant that cannot be supported must make the converter stop
with a message naming the variant. Do not edit viewer.py or gs4d/format.py.
Done when: the README and the Status section of CLAUDE.md give the measured PSNR and move this variant from "NOT tested" to "Tested".
Report in 5 lines, then stop.
```

### WP17 — the `--compact` flag and one command that runs every test
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md. Precondition: WP10 is merged; no converter package
of CC-C is open.
Task: (1) add a --compact flag to every convert_*.py that only passes `compact=True` to `Sequence4D.save`; change nothing else
in the converters. gs4d/open.py uses --compact when the estimated file is larger than 1 GB. (2) write tests/run_all.py, which runs every
test_*.py and check_*.py and prints a table of passed / skipped (with the reason: no CUDA, no Chrome, no data) / failed,
and returns an error code if anything failed. Replace the list of commands in the "Tests" section of the README with this command, and
keep the old list below it.
(3) Compare requirements.txt with the packages the README says to install (viser, numpy<2, imageio, imageio-ffmpeg, pillow, playwright)
and make them match.
Done when: `python tests/run_all.py` passes on this machine, and a converter run with --compact gives a file that can be read.
Report in 5 lines, then stop.
```

### WP18 — training cameras and original images in the viewer (only when asked)
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md. Precondition: WP14 is merged.
Task: add a `--cameras <cameras.json | transforms_*.json>` argument to viewer.py: a camera list in the control panel,
click to jump to that camera (and to the image's time if the file has one), and with the gsplat renderer a checkbox that shows the
original image next to the render with the PSNR of the frame on screen. The camera conventions are in CLAUDE.md (D-NeRF is OpenGL: flip
columns 1 and 2 of c2w).
Only edit viewer.py; gs4d/open.py passes --cameras by itself when it finds the file (note it in the report for CC-B if it needs a change).
Done when: run with ../data_real/dnerf/data/bouncingballs and a converted model, with screenshots in shots/, the PSNR shown matches
tests/check_real.py within 0.1 dB, and the README has a paragraph of instructions. Report in 5 lines, then stop.
```

### WP19a — `open.py` recognises 4DGaussians and `--cameras`; stop on unsupported variants
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md.
Task 1: in gs4d/open.py, add a detection branch for 4DGaussians (hustvl) model folders that calls convert_4dgaussians.py.
Read the top of convert_4dgaussians.py to get exactly the files it needs; tell it apart from Deformable-3DGS / SC-GS (which have `deform/`)
and Spacetime Gaussians by signs inside the files, not by folder names. Add this case to tests/test_open.py.
Task 2: gs4d/open.py passes `--cameras` to the viewer by itself when it finds `cameras.json` in the model folder, or the
dataset's `transforms_test.json` (read the dataset path from `cfg_args` if there is one); when the user passes it, keep the user's.
When nothing is found, pass nothing and report no error.
Task 3 (criterion 2 of WP9): an unsupported variant must stop and name the variant.
 - convert_scgs.py: a model with a colour head (`--pred_color`) currently only prints a warning. Make it stop, unless the flag
   `--ignore-pred-color` is given.
 - convert_spacetime.py: the ply of the full model (with `f_dc_3..`, or with an `rgbdecoder` file next to the ply) is currently converted
   with the base colour without a word. Make it stop and say "Spacetime Gaussians full model (rgbdecoder)", unless the flag
   `--base-color-only` is given. The two lite models in ../data_real must still run as before, without a flag.
 The stop message must give the flag that converts anyway. Update the tests of both converters for these two cases.
Task 4: fix the stale sentence in the Status section of CLAUDE.md ("SH degree 0 only, so colours are mottled" after the `--sh` numbers).
Only edit: gs4d/open.py, tests/test_open.py, convert_scgs.py, convert_spacetime.py, tests/test_scgs.py, tests/test_spacetime.py,
the README (the matching rows), CLAUDE.md, and the status table in PLAN.md.
Done when: `python -m gs4d.open <the 4DGaussians bouncingballs folder in /home/cont/gs4d_train>` opens the viewer with a camera list;
`python tests/run_all.py` and `python tests/check_real.py` pass, with unchanged PSNR; the WP9 table in PLAN.md is updated.
Report in 5 lines, then stop.
```

### WP19b — measure display FPS in a GPU-accelerated Chrome (done 15/11/2026)

The steps are in "Do now (phase 2)", item 2. Done when the README has the browser FPS at 1280×720 next to the server-side numbers, with the
GPU name and the Chrome version; remove the line "display FPS in a real browser" from the NOT tested part of `CLAUDE.md`.

### WP19c — PSNR for Spacetime and Shape of Motion; the 1.9 dB gap of native 4DGS (part 3 done; parts 1, 2 opened)
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md. One part per session; say which part when pasting.
Part 1 — Shape of Motion `paper-windmill`: get the scene's original images from the DyCheck iPhone set (this scene only), render at the
training cameras (`Ks`, `w2cs` in the checkpoint) and add a PSNR with a threshold to the matching item of tests/check_real.py.
Report the download size in the first turn and stop if it exceeds 5 GB.
Part 2 — Spacetime Gaussians `n3d_sear_steak` (lite): the same with the `sear_steak` video of Neural 3D Video and `cameras.json`.
Report the download size in the first turn and stop if it exceeds 5 GB. The images show people: do not commit them to the repo, only the
numbers.
Part 3 — native 4DGS on `slice-banana`: ours 24.23 dB, the original repo's own evaluation 26.14 dB. Diagnose only, do not tune until it
matches. Try in turn and write down the PSNR after every step: (a) keyframes placed exactly at the 33 test times instead of
interpolating; (b) the marginal_t threshold and how opacity is multiplied over time, compared with the original renderer; (c) the
background and resolution used for the measurement compared with the repo's evaluation; (d) SH: whether the degree and the time-dependent
part are fully accounted for. Stop at the step that explains most of the gap.
Only edit: tests/check_real.py; convert_native4d.py and tests/test_native4d_sh.py if part 3 finds a bug in the converter;
the matching rows in the README, CLAUDE.md and PLAN.md.
Done when: the README has the PSNR (parts 1, 2) or a table of PSNR per step and a conclusion (part 3). Report in 5 lines, then stop.
```

### WP19d — converters write one keyframe at a time to disk (when long scenes are needed) — DONE 15/11/2026
Result: `Sequence4DWriter` in `gs4d/format.py` (keeps the static group and the compact layout, file bit-identical to `save`),
`tests/test_writer.py`; `slice-banana` with 330 keyframes: SC-GS peak RAM 1.4 GB, 26.87 dB; native 4DGS 1.3 GB, 24.27 dB;
Deformable-3DGS 7.2 → 1.0 GB.
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md.
Problem: the converters build the whole sequence [K,N,...] in memory before saving. On `slice-banana` (392k–603k Gaussians), SC-GS and
native 4DGS can only convert 110 keyframes; with 330 keyframes the 15 GB of RAM run out and the process is killed.
Step 0: measure the peak RAM of `convert_scgs.py --frames 110 --sh --compact` on `slice-banana` and find which array takes the space
(usually `sh` [K,N,16,3]).
Task: add an incremental writer to gs4d/format.py (e.g. `Sequence4DWriter`: open the file, `add_keyframe(t, means, covs, ...)`,
`close()`) that writes both the plain and the compact layout, with peak RAM independent of K; the output must be readable by the current
`Sequence4D.load`. Detecting the static group (WP11) may be skipped on this path if it needs two passes; say so if it is skipped. Then move
convert_scgs.py, convert_native4d.py and convert_deformable3dgs.py to it; the other converters stay as they are.
Only edit: gs4d/format.py, the three converters above, add tests/test_writer.py, the matching rows in the README.
Done when: a test compares the incrementally written file with the file written at once (bit-identical in the plain layout); SC-GS and
native 4DGS convert the 330 keyframes of `slice-banana` with peak RAM below 8 GB; tests/check_real.py passes, and the PSNR of
`scgs_banana` and `native4d_banana` at 330 keyframes is written next to the 110-keyframe numbers. Report in 5 lines, then stop.
```

### WP20 — retrain native 4DGS `slice-banana` with the right cameras
```
Read CLAUDE.md (the WP19c part 3 entry) and the "Done criteria for every package" section of PLAN.md. Precondition: branch
cc-c/wp19c-psnr is merged.
Problem: tests/make_hypernerf_blender.py writes `fl_x/fl_y/cx/cy` per frame; the Blender reader of fudan-zvg/4d-gaussian-splatting
then sets `FovX = FovY = -1`, so the `slice-banana` model was trained with wrong cameras, and our 24.23 dB (the repo's own evaluation 26.14)
does not reflect the converter's quality.
Step 0: `nvidia-smi`; if the GPU has less than 10 GB free, report and stop (last time a game held ~3 GB and three training runs hit CUDA
OOM).
Task: read the repo's reader to find which field it takes the FoV from, fix tests/make_hypernerf_blender.py to write exactly that field
(do not patch the original repo), rebuild the layout, retrain in /home/cont/gs4d_train with the same config (20k iterations, batch_size 1,
densify_until_num_points 250_000), then rerun the repo's evaluation and `convert_native4d.py --frames 110 --sh --compact`.
Only edit: tests/make_hypernerf_blender.py, the `native4d_banana` item and its threshold in tests/check_real.py,
tests/diag_native4d_banana.py (drop the emulation step if it is no longer needed), the matching rows in the README, CLAUDE.md, PLAN.md.
Do not edit convert_native4d.py; if after retraining the gap to the repo's evaluation is still more than 0.5 dB, report the numbers of each
step of the diag script and stop; do not tune until it matches.
Done when: the README gives the new PSNR of the repo and ours on the 33 held-out images; the two differ by at most 0.5 dB; check_real.py
passes. Report in 5 lines, then stop.
```

### WP21 — compact the per-keyframe `sh` field
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md.
Problem: the compact layout (WP8) leaves `sh` alone. Native 4DGS writes `sh` per keyframe [K,N,S,3] as float32:
`slice-banana` with 330 keyframes gives 19.9 GB, and check_real.py cannot load that file without `load(lazy=True)`.
Step 0: measure the distribution of the `sh` values in real files (degree 0 against the higher degrees) to choose the quantisation;
write the numbers down.
Task: in the compact layout, store the time-varying `sh` quantised (e.g. float16, or int8 with a scale per degree); `sh` that is constant
over time stays as it is; `Sequence4DWriter`, lazy loading and `GpuFrames` can all read it; old files can still be read.
Only edit: gs4d/format.py, tests/test_compact.py, tests/test_writer.py, tests/bench_large.py, the "Large files" paragraph of the README.
Done when: the README has a before/after table for native 4DGS `slice-banana` with 330 keyframes and `bouncingballs --sh` (size, frame
fetch time while playing, PSNR); target size at most 1/3 of the float32 file and PSNR drop at most 0.1 dB; check_real.py passes.
Report in 5 lines, then stop.
```

### WP22 — `Sequence4DWriter` for the four remaining converters — DONE 15/11/2026

Result: the four converters write through the Writer, and their output equals the old code's bit for bit in every array on real outputs
(peak RAM now in `docs/PERFORMANCE.md`). `convert_per_frame.py` keeps the old path when the Gaussian count changes between frames (the
Writer needs a fixed N). Shape of Motion with all 277 frames can now be converted (peak RAM 0.88 GB).

Original prompt:
```
Read CLAUDE.md (the WP19d entry) and the "Done criteria for every package" section of PLAN.md.
Task: move convert_4dgaussians.py, convert_shape_of_motion.py, convert_spacetime.py and convert_per_frame.py to writing through
`gs4d.Sequence4DWriter`, exactly as convert_scgs.py does; change no formula. For convert_per_frame.py in the mode where the
Gaussian count changes between frames (`interpolate=False`): if the Writer does not support it, keep the old path for that mode and say so.
Only edit the four converters above and the matching rows in the README, CLAUDE.md, PLAN.md.
Done when: for every converter, the output on a real output in ../data_real equals the old code's output in every array (compared with a
script, results pasted into the report); the README has the peak RAM before/after; `python tests/run_all.py` and check_real.py pass.
Report in 5 lines, then stop.
```

### WP23 — 4DGaussians on the real scene `slice-banana`
```
Read CLAUDE.md (the WP15 and WP16 entries of 15/11/2026) and the "Done criteria for every package" section of PLAN.md.
Precondition: WP20 is done (GPU free). Step 0: `nvidia-smi`; if the GPU has less than 10 GB free, report and stop.
Task: train hustvl 4DGaussians on HyperNeRF `slice-banana` in /home/cont/gs4d_train/4dgaussians (a clone that is already patched), with
the repo's own HyperNeRF reader and the same split of 297 training / 33 held-out images as the three other methods (see
tests/make_hypernerf_split.py), 268x480 images, 20k iterations, or the repo's default hypernerf config if it fits in a 12 GB GPU and 15 GB
of RAM. Then `convert_4dgaussians.py --sh --compact`, open with `python -m gs4d.open`, and measure the PSNR on the 33 held-out images next
to the repo's evaluation numbers.
Check before trusting the repo's numbers: are the cameras the repo uses for training right (the lesson of WP19c part 3: FovX/FovY)?
Only edit: convert_4dgaussians.py and tests/test_4dgaussians.py if the real format differs from what was assumed; add a
`4dgaussians_banana` item to tests/check_real.py; the matching rows in the README, CLAUDE.md, PLAN.md.
Done when: the README gives our PSNR and the repo's on the 33 images, the Gaussian count, the size and the conversion time; if they differ
by more than 0.5 dB, report the causes tried, do not tune until it matches. Report in 5 lines, then stop.
```

### WP24 — draw Gaussian centres as points (done 16/11/2026)
```
Read CLAUDE.md and the "Done criteria for every package" section of PLAN.md.
Task: add a "Display" choice to the control panel of viewer.py: `splats` (as now) or `points` (every Gaussian is a point at its
centre, in the Gaussian's colour, with a "Point size" slider), usable while playing, with `--compare`, and with both renderers
(client: viser's point cloud; gsplat: rasterise with a small fixed pixel scale, or send a point cloud to the client if that is simpler;
pick one and say why). The "Hide opacity below" slider still applies. Write in the README how to move the camera (mouse and keys) for the
installed viser version, after trying it for real.
Only edit viewer.py, README, CLAUDE.md, PLAN.md; add tests/check_points.py (headless screenshots at 3 times).
Done when: run with `../data_real/scgs/bouncingballs_sh.npz` and `../data_real/som/paper-windmill.npz`, with screenshots in
shots/points/, and the playback speed in points mode written in the README. Report in 5 lines, then stop.
```

---

# Appendix: the phase 1 plan (done, kept for reference)

Where it drifted from what happened: rule 2 names branches `wp<N>-<name>`, but the real branches are `cc-<session>/wp<N>-<name>`; the
convention "CC-C opens only after the professor answers" was not kept; WP7 says "pick one", but both were done.

### Status (14/11/2026)

All 8 packages met their done criteria and are merged into `main`. The rest of the file is the original plan, kept for reference.

| WP | Status | Branch | Notes |
|---|---|---|---|
| 1 | Done | `cc-a/wp1-gsplat` | FPS table in the README, now in `docs/PERFORMANCE.md` (RTX 3060) |
| 2 | Done | `cc-b/wp2-real-data` | Real Dynamic 3D Gaussians output `basketball`; images in `shots/wp2/` |
| 3 | Done | `cc-c/wp3-spacetime` | Run with the real model `n3d_sear_steak_lite_allcam` (lite model, no `rgbdecoder`) |
| 4 | Done | `cc-c/wp4-deformable3dgs` | 3 variants (default, D-NeRF, 6-DoF); the D-NeRF variant run with a real self-trained model (`bouncingballs`) |
| 5 | Done | `cc-c/wp5-shape-of-motion`, `cc-c/wp5-scgs` | Both methods done; run with the real checkpoint `paper-windmill` (Shape of Motion) and a self-trained model `bouncingballs` (SC-GS) |
| 6 | Done | `cc-a/wp6-sh` | |
| 7 | Done | `cc-a/wp7-video`, `cc-a/wp7-compare` | Both (a) comparing two sequences and (b) mp4 export were done |
| 8 | Done | `cc-a/wp8-large` | Before/after table in the README, now in `docs/PERFORMANCE.md` |

Still open, outside the packages' done criteria:

- Run with real outputs (14/11/2026): all four converters of WP3–5 and native 4DGS; measurements in the "Use with real outputs" table of
  the README (now `docs/SUPPORTED.md`), images compared with the originals in `shots/real/`. Not tried: the full Spacetime Gaussians model
  (with `rgbdecoder`), the default and 6-DoF variants of Deformable-3DGS, real (non-D-NeRF) scenes
  for Deformable-3DGS, SC-GS and native 4DGS.
- The question for the professor below has no recorded answer; WP3–5 were done without waiting.
- Display rate in a real browser: measured in WP19b (15/11/2026), table in the README (now `docs/PERFORMANCE.md`).
- The full list of what has and has not been checked is in the Status section of `CLAUDE.md` (now `docs/SUPPORTED.md` and
  `docs/HISTORY.md`).

### Assignment: 3 Claude Code sessions

| Session | Role | Machine | Model | Tasks, in this order | Files it may edit (this session only) |
|---|---|---|---|---|---|
| **CC-A** | Rendering and GUI | machine with an NVIDIA GPU | strong (Opus/Fable) | WP1 → WP6 → WP7 → WP8 | `viewer.py`, `gs4d/format.py`, `gs4d/ply.py`, `make_toy.py`, the body of the README |
| **CC-B** | Real data | machine with the output files | Sonnet | WP2 | `convert_per_frame.py`, `convert_native4d.py`, `tests/test_all.py` |
| **CC-C** | New converters | any, a CPU is enough | Sonnet | WP3 → WP4 → WP5, only the packages the lab needs | new files only: `convert_<name>.py`, `tests/test_<name>.py` |

Shared conventions:

- A session edits only the files in its own column. A change needed in another session's file goes into the report, not into the file.
- CC-B and CC-C may add exactly one row per converter to the "Use with real outputs" table of the README.
- CC-A and CC-B start at the same time. CC-C opens only after the professor answers the question below.
- CC-A keeps one session for all 4 packages so it does not have to reload the context about `viewer.py`; between two packages, merge the
  branch before going on.
- If CC-B finds that a real output needs a change to the common format (e.g. a missing field), CC-B stops and reports; that change belongs
  to CC-A.
- Git branches: `cc-a/wp1-gsplat`, `cc-b/wp2-real-data`, `cc-c/wp3-spacetime`, ... CC-B and CC-C work in their own worktrees.

### Before opening CC-C: one question for the professor

**Which 4DGS methods does the lab use, and where are the output files?**

This is where the most can be saved. Each of WP3–WP5 serves one method; writing a converter for a method
the lab does not use is wasted work. Only WP1 and WP2 are certainly needed.

### Work packages

| WP | Task | Files edited | Needs | Depends on | When |
|---|---|---|---|---|---|
| 1 | Run and fix the gsplat renderer | `viewer.py` (gsplat part only) | NVIDIA GPU + CUDA | none | Now |
| 2 | Run end to end with one real output | the matching converter | a real output file | the professor's answer | As soon as there is a file |
| 3 | Spacetime Gaussians converter (type C) | `convert_spacetime.py`, new test | CPU | none | Only if the lab uses it |
| 4 | Deformable-3DGS converter (type D) | `convert_deformable3dgs.py`, new test | CPU + torch | none | Only if the lab uses it |
| 5 | Shape of Motion / SC-GS converter | new converter file, new test | CPU + torch | none | Only if the lab uses it |
| 6 | View-dependent SH colour | `gs4d/format.py`, `viewer.py` | GPU | WP1 done | After the first demo |
| 7 | Demo features: two sequences side by side, video export | `viewer.py` | GPU if gsplat is used | WP1 done | When the professor asks |
| 8 | Large files: float16/quantisation, lazy loading | `gs4d/format.py` | CPU | WP6 done or WP6 dropped | When files exceed a few GB |

### Order

- **Round 1 (in parallel, at most 3 sessions):** WP1 + WP2 + at most one of WP3/4/5.
  These three packages touch no file in common.
- **Round 2 (in sequence, one session at a time):** WP6 → WP7 → WP8. All three edit `viewer.py` or `format.py`;
  running them in parallel would conflict and have to be redone.

### Rules against wasted work

1. **One file, one owner.** Only one session at a time may edit `viewer.py` and `gs4d/format.py`. Converters only
   add new files.
2. **Git first.** Run `git init && git add -A && git commit -m "baseline"` once. One branch per WP,
   `wp<N>-<name>`, merged when done. Two sessions never share a working tree: use `git worktree add ../gs4d-wp3 -b wp3-spacetime`.
3. **No re-research.** The viser and gsplat APIs and the checkpoint layouts are written in `CLAUDE.md`.
4. **Pick the model for the job.** Converters (WP3–5) are formula ports with clear tests: use a cheaper model
   (Sonnet). WP1, WP6, WP7 touch rendering and the GUI: use a stronger model.
5. **Stop at the right time.** Every prompt below has done criteria. A session that meets them reports in 5 lines and stops,
   without widening the scope.
6. **Missing data: stop, do not guess.** Without a GPU or a real output file, the session says so
   in the first turn instead of writing more code that cannot be checked.

### Prompts for each session

Paste the whole block into a new Claude Code session opened in `gs4d_viewer`.

#### WP1 — gsplat renderer
```
Read CLAUDE.md. Task: make `python viewer.py toy.npz --renderer gsplat` work correctly on this machine.
Step 0: check `nvidia-smi` and `python -c "import torch, gsplat; print(torch.cuda.is_available())"`.
Without CUDA, report and stop.
Only edit the gsplat part of viewer.py (_draw_gsplat and the related set-up). Do not edit the GUI, the format or the converters.
Done when: (1) toy.npz renders correctly, compared by eye with the client renderer through headless screenshots at the same time t;
(2) dragging the time slider and orbiting the camera both update the image; (3) the README gives the FPS measured with toy.npz
and the resolution it was measured at. Report in 5 lines, then stop.
```

#### WP2 — run with a real output
```
Read CLAUDE.md. The real output is at: <PATH>. Method: <NAME>.
Task: convert this output to the common format with the existing converter and open it in the viewer.
If the converter fails because the real format differs from what the code assumes, fix that converter (and its test);
do not edit viewer.py or gs4d/format.py. If the method has no converter yet, report and stop (that is WP3–5).
Done when: there is an .npz file, the viewer opens it, there are screenshots at 3 times, and the README gives the command used with the
file size and the number of Gaussians. Say clearly what looks wrong (colours, up axis, scale). Report in 5 lines, then stop.
```

#### WP3 — Spacetime Gaussians
```
Read CLAUDE.md. Task: write convert_spacetime.py, which reads the point_cloud.ply of Spacetime Gaussians
(oppo-us-research/SpacetimeGaussians) and writes a Sequence4D.
Clone the repo to READ the formulas (polynomial motion over time, rotation over time, opacity as a temporal RBF)
and the attribute names in the ply; do not run the repo's code. Only add: convert_spacetime.py, tests/test_spacetime.py,
one row in the README table. Colour: use the base colour component, drop the view-dependent part and state this limit clearly.
Done when: the test builds a fake ply in the right layout, runs the converter, and compares positions/opacities at 3 times with a formula
written independently in the test; `python tests/test_all.py` still passes. Report in 5 lines, then stop.
```

#### WP4 — Deformable-3DGS
```
Read CLAUDE.md. Task: write convert_deformable3dgs.py, which reads the point_cloud.ply + deform.pth of Deformable-3DGS
(ingra14m/Deformable-3D-Gaussians) and writes a Sequence4D at K times.
Clone the repo to READ the architecture of the deformation MLP (positional encoding, number of layers, skip, outputs
d_xyz/d_rotation/d_scaling), then write an independent forward pass with torch on the CPU, loading the weights from the state_dict; do
not import the repo's code. Run in batches so the RAM does not run out. Only add: convert_deformable3dgs.py,
tests/test_deformable3dgs.py, one README row.
Done when: the test builds a random MLP with the same architecture, saves the state_dict + a fake ply, runs the converter, and compares
with a reference forward pass written in the test; `python tests/test_all.py` still passes. Report in 5 lines, then stop.
```

#### WP5 — Shape of Motion or SC-GS
```
Read CLAUDE.md. Method: <Shape of Motion | SC-GS>. Task and constraints as in WP4: read the repo to port the formulas
(SE(3) motion bases + coefficients, or control nodes + LBS), write convert_<name>.py + tests/test_<name>.py + one README row,
do not import the repo's code, do not edit other files. Done when the test matches an independently written formula and
`python tests/test_all.py` still passes. Report in 5 lines, then stop.
```

#### WP6 — SH colour
```
Read CLAUDE.md. Precondition: WP1 is merged. Task: add an optional field `sh` [N,(d+1)^2,3] to Sequence4D
(old files without this field can still be read), and let the gsplat renderer use sh_degree when it is there; the client renderer keeps
using `colors`. Update gs4d/ply.py and convert_per_frame.py to keep f_rest_* when the --sh flag is given. Add a round-trip test.
Done when: toy with fake SH changes colour with the view direction in the gsplat renderer, and every old test still passes. Report in 5
lines, then stop.
```

#### WP7 — demo features
```
Read CLAUDE.md. Precondition: WP1 is merged. Task: <PICK ONE: (a) open 2 .npz files and toggle between them or put them side by side,
with a shared time slider; (b) a button that exports an mp4 video along the current camera path over the whole time range>.
Only edit viewer.py. Done when the feature works with toy.npz, with a screenshot or a video file as evidence,
and the README has a paragraph of instructions. Report in 5 lines, then stop.
```

#### WP8 — large files
```
Read CLAUDE.md. Task: reduce the size and RAM of large sequences. Measure first: the size and load time of a random
sequence of 500k Gaussians x 100 keyframes. Then add storing means as float16 offsets from the first keyframe and
quantised covs/opacities, with a flag to turn it off; old files can still be read. Only edit gs4d/format.py and add tests.
Done when the README has a before/after table (size, load time, largest position error) and every test passes.
Report in 5 lines, then stop.
```
