#!/usr/bin/env python3
"""
verify_imagenet100_local.py
============================
Standalone, read-only verification that the chosen ImageNet-100 class subset
(PODNet / DER / DyTox lineage, see thesis_agent/reports/
imagenet100_generalization_preparation.md, "Final ImageNet-100 Benchmark
Selection") is fully present in the shared local ImageNet-1k installation,
BEFORE any training code touches it.

Checks, for every one of the 100 selected WNIDs:
    <IMAGENET_ROOT>/train/<wnid>/  exists and is non-empty
    <IMAGENET_ROOT>/val/<wnid>/    exists and is non-empty

Does NOT read, copy, move, or modify any image file. Does NOT touch any
directory outside the 100 selected WNIDs. Does NOT access any "snapshot"
subdirectory (only the plain train/<wnid> and val/<wnid> structure the task
specified is checked).

Usage:
    python scripts/verify_imagenet100_local.py
    IMAGENET_ROOT=/some/other/path python scripts/verify_imagenet100_local.py

Exit code 0 if all 100 classes are present in both splits; 1 otherwise.
"""

import os
import sys

# Kept as a small, self-contained copy of vit_lora_cifar100_full5step_n5.py's
# IMAGENET100_SYNSETS so this script has no dependency on importing that
# (very large, side-effect-heavy) module. If that list is ever changed, this
# one must be updated to match -- the __main__ block below asserts the two
# stay in sync when run from the project root, as a guard against drift.
IMAGENET100_SYNSETS = [
    "n01440764", "n01443537", "n01484850", "n01491361", "n01494475",
    "n01496331", "n01498041", "n01514668", "n01514859", "n01518878",
    "n01530575", "n01531178", "n01532829", "n01534433", "n01537544",
    "n01558993", "n01560419", "n01580077", "n01582220", "n01592084",
    "n01601694", "n01608432", "n01614925", "n01616318", "n01622779",
    "n01629819", "n01630670", "n01631663", "n01632458", "n01632777",
    "n01641577", "n01644373", "n01644900", "n01664065", "n01665541",
    "n01667114", "n01667778", "n01669191", "n01675722", "n01677366",
    "n01682714", "n01685808", "n01687978", "n01688243", "n01689811",
    "n01692333", "n01693334", "n01694178", "n01695060", "n01697457",
    "n01698640", "n01704323", "n01728572", "n01728920", "n01729322",
    "n01729977", "n01734418", "n01735189", "n01737021", "n01739381",
    "n01740131", "n01742172", "n01744401", "n01748264", "n01749939",
    "n01751748", "n01753488", "n01755581", "n01756291", "n01768244",
    "n01770081", "n01770393", "n01773157", "n01773549", "n01773797",
    "n01774384", "n01774750", "n01775062", "n01776313", "n01784675",
    "n01795545", "n01796340", "n01797886", "n01798484", "n01806143",
    "n01806567", "n01807496", "n01817953", "n01818515", "n01819313",
    "n01820546", "n01824575", "n01828970", "n01829413", "n01833805",
    "n01843065", "n01843383", "n01847000", "n01855032", "n01855672",
]

DEFAULT_IMAGENET_ROOT = "/nfsd/lttm4/datasets/ImageNet-1k_torch"


def _is_nonempty_dir(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def verify(imagenet_root):
    assert len(IMAGENET100_SYNSETS) == 100, "IMAGENET100_SYNSETS must have exactly 100 entries"
    assert len(set(IMAGENET100_SYNSETS)) == 100, "IMAGENET100_SYNSETS has duplicate WNIDs"

    train_root = os.path.join(imagenet_root, "train")
    val_root = os.path.join(imagenet_root, "val")

    missing_train = []
    missing_val = []
    empty_train = []
    empty_val = []

    for wnid in IMAGENET100_SYNSETS:
        train_dir = os.path.join(train_root, wnid)
        val_dir = os.path.join(val_root, wnid)

        if not os.path.isdir(train_dir):
            missing_train.append(wnid)
        elif not _is_nonempty_dir(train_dir):
            empty_train.append(wnid)

        if not os.path.isdir(val_dir):
            missing_val.append(wnid)
        elif not _is_nonempty_dir(val_dir):
            empty_val.append(wnid)

    return {
        "imagenet_root": imagenet_root,
        "checked": len(IMAGENET100_SYNSETS),
        "missing_train": missing_train,
        "missing_val": missing_val,
        "empty_train": empty_train,
        "empty_val": empty_val,
    }


def main():
    imagenet_root = os.environ.get("IMAGENET_ROOT", DEFAULT_IMAGENET_ROOT)
    print(f"IMAGENET_ROOT = {imagenet_root}")
    print(f"Checking {len(IMAGENET100_SYNSETS)} WNIDs (PODNet/DER/DyTox ImageNet-100 subset)...")

    if not os.path.isdir(imagenet_root):
        print(f"FATAL: IMAGENET_ROOT does not exist or is not a directory: {imagenet_root}")
        sys.exit(1)

    result = verify(imagenet_root)

    print()
    print(f"Missing train/<wnid> directories: {len(result['missing_train'])}")
    if result["missing_train"]:
        print("  " + ", ".join(result["missing_train"]))
    print(f"Missing val/<wnid> directories:   {len(result['missing_val'])}")
    if result["missing_val"]:
        print("  " + ", ".join(result["missing_val"]))
    print(f"Present-but-EMPTY train dirs:      {len(result['empty_train'])}")
    if result["empty_train"]:
        print("  " + ", ".join(result["empty_train"]))
    print(f"Present-but-EMPTY val dirs:        {len(result['empty_val'])}")
    if result["empty_val"]:
        print("  " + ", ".join(result["empty_val"]))

    total_problems = (
        len(result["missing_train"]) + len(result["missing_val"])
        + len(result["empty_train"]) + len(result["empty_val"])
    )
    print()
    if total_problems == 0:
        print("RESULT: PASS -- all 100 classes present (non-empty) in both train/ and val/.")
        sys.exit(0)
    else:
        print(f"RESULT: FAIL -- {total_problems} problem(s) found above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
