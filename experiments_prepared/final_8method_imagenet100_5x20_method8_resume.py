#!/usr/bin/env python
"""Dedicated, fail-closed resume launcher for ImageNet-100 Method 8 only."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHOD = "rank_extension_factor_orth_lam50_fullkd_T2_protect30"
CHECKPOINT = ROOT / "results" / (
    "imagenet100_5x20_final_8method_canonical_continuation_methods6to8_seed42_ep9_checkpoints"
) / f"{METHOD}__rankext_state.pt"
SOURCE = ROOT / "experiments_prepared" / "final_8method_imagenet100_5x20_continuation_methods6to8_seed42_ep9.py"

if not CHECKPOINT.is_file():
    raise SystemExit(f"FATAL: required completed-task checkpoint is missing: {CHECKPOINT}")
if not SOURCE.is_file():
    raise SystemExit(f"FATAL: audited continuation source is missing: {SOURCE}")

os.chdir(ROOT)

# The source script reads these before constructing its method map and uses
# the same deterministic checkpoint directory.  Do not permit a caller to
# redirect recovery to a different method/checkpoint.
os.environ["IMAGENET100_METHOD8_ONLY_RESUME"] = "1"
os.environ["IMAGENET100_RESUME_CHECKPOINT"] = str(CHECKPOINT)
os.environ["REPLICATION_SEED"] = "42"
os.environ.pop("FAST_RUN_DEBUG", None)

runpy.run_path(str(SOURCE), run_name="__main__")
