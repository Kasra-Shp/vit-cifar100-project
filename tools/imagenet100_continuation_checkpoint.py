"""Small atomic checkpoint primitives used by the ImageNet continuation job."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_torch_payload_atomic(path: str | os.PathLike[str], payload: Any) -> None:
    """Write a torch payload with replace-on-success semantics."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def write_done_marker_atomic(path: str | os.PathLike[str]) -> None:
    """Create a completion marker only after final evaluation succeeds."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text("done\n", encoding="utf-8")
    os.replace(tmp, target)


def load_torch_payload(path: str | os.PathLike[str]) -> Any:
    """Load a continuation payload onto CPU for safe reconstruction."""
    return torch.load(path, map_location="cpu")
