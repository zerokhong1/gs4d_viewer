#!/usr/bin/env python3
"""Template for a new converter: copy to convert_<method>.py and replace the parts marked TODO.

This one reads a made-up "method" with the simplest possible motion (a type C parametric function): a .npz with
    xyz [N,3]  position at t = 0       vel [N,3]  velocity       quat [N,4] wxyz   log_scale [N,3]
    logit_opacity [N]                  rgb [N,3]  colour in [0, 1]
and writes the common format (see docs/ADD_A_CONVERTER.md). Position at time t is xyz + vel * t.

    python convert_template.py model.npz -o scene.npz --frames 30 --t0 0 --t1 1 [--compact]

Rules (CLAUDE.md): run on CPU with numpy, port formulas from the method's repo instead of importing it, stop with a
message (SystemExit) on variants that are not supported, write through Sequence4DWriter so that memory stays small.
"""
from __future__ import annotations

import argparse

import numpy as np

from gs4d import Sequence4D, Sequence4DWriter, cov_from_quat_scale

REQUIRED = ("xyz", "vel", "quat", "log_scale", "logit_opacity", "rgb")   # TODO: the method's own names


def load_model(path):
    with np.load(path) as d:
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            raise SystemExit(f"{path}: missing {missing}; not a model of this method")   # say what is wrong, do not guess
        return {k: np.asarray(d[k], np.float32) for k in REQUIRED}


def keyframe(m, t):
    """Everything that depends on time, for one time t. TODO: the method's formulas (activations, deformation, ...)."""
    q = m["quat"] / np.linalg.norm(m["quat"], axis=1, keepdims=True)   # the format wants normalised wxyz
    return dict(means=m["xyz"] + m["vel"] * t, covs=cov_from_quat_scale(q, np.exp(m["log_scale"])))


def convert(m, out, frames=30, t0=0.0, t1=1.0, compact=False, source=""):
    opacities = 1.0 / (1.0 + np.exp(-m["logit_opacity"]))   # sigmoid: the format stores opacity in [0, 1]
    # fields that never change go to the constructor; the others go to every add_keyframe
    with Sequence4DWriter(out, compact=compact, source=source, opacities=opacities, colors=np.clip(m["rgb"], 0, 1)) as w:
        for t in np.linspace(t0, t1, frames):
            w.add_keyframe(float(t), **keyframe(m, float(t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--frames", type=int, default=30, help="number of keyframes K")
    ap.add_argument("--t0", type=float, default=0.0, help="start time (TODO: the method's time unit)")
    ap.add_argument("--t1", type=float, default=1.0)
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (lossy, about 2-3x smaller; see docs/FORMAT.md)")
    a = ap.parse_args()
    convert(load_model(a.model), a.output, a.frames, a.t0, a.t1, a.compact, source=a.model)
    out = a.output if a.output.endswith(".npz") else a.output + ".npz"
    print(f"Wrote {out}: {Sequence4D.load(out, lazy=True).summary()}")


if __name__ == "__main__":
    main()
