"""Reading training checkpoints in the converters.

torch is not needed by the viewer, so the converters import it on demand through these two helpers: a missing torch, or a
file torch cannot read, stops with one line that names the problem instead of a traceback.
"""
from __future__ import annotations


def import_torch(script: str):
    """The torch module, or SystemExit saying how to install it (the CPU build is enough for every converter)."""
    try:
        import torch
    except ImportError:
        raise SystemExit(f"{script} needs torch (the CPU build is enough): pip install torch") from None
    return torch


def torch_load(path: str, script: str, weights_only: bool = True):
    """torch.load onto the CPU. weights_only=False runs arbitrary pickle code: use it only for files you trust."""
    torch = import_torch(script)
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except Exception as e:  # not a checkpoint, truncated, or a pickle that weights_only refuses
        raise SystemExit(f"{path}: cannot read it as a torch checkpoint ({type(e).__name__}: {e})") from None
