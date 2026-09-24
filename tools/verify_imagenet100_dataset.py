#!/usr/bin/env python3
"""Offline verifier for a prepared ImageNet-100 transfer directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

try:
    from .imagenet100_common import CLASSES_PER_TASK, IMAGENET100_SYNSETS, NUM_CLASSES, NUM_TASKS, SEED, task_split
    from .imagenet100_shared import split_validation_paths, verify_shared_root
except ImportError:  # direct `python tools/verify_imagenet100_dataset.py`
    from imagenet100_common import CLASSES_PER_TASK, IMAGENET100_SYNSETS, NUM_CLASSES, NUM_TASKS, SEED, task_split
    from imagenet100_shared import split_validation_paths, verify_shared_root  # type: ignore

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--source-mode", choices=("auto", "prepared", "shared_imagenet1k"), default="auto")
    p.add_argument("--max-readable-checks", type=int, default=0, help="0 checks every image; otherwise checks a deterministic prefix")
    p.add_argument("--eval-per-class", type=int, default=25, help="shared-root calibration images per class; remainder is frozen test")
    return p.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def verify_prepared(args: argparse.Namespace) -> int:
    root = args.data_root.resolve()
    errors: list[str] = []
    metadata = root / "metadata"
    required = [metadata / "classes.json", metadata / "task_split.json", metadata / "dataset_manifest.json", metadata / "source.json"]
    for path in required:
        if not path.is_file():
            fail(errors, f"missing metadata file: {path}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    try:
        classes = json.loads((metadata / "classes.json").read_text(encoding="utf-8"))
        split = json.loads((metadata / "task_split.json").read_text(encoding="utf-8"))
        manifest = json.loads((metadata / "dataset_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL: invalid metadata JSON: {exc}")
        return 1

    class_rows = classes.get("classes", [])
    synsets = [row.get("synset") for row in class_rows]
    if len(class_rows) != NUM_CLASSES or synsets != IMAGENET100_SYNSETS:
        fail(errors, "classes.json does not contain the exact canonical 100 synsets in deterministic order")
    if classes.get("seed") != SEED:
        fail(errors, f"classes.json seed must be {SEED}")
    expected_split = task_split(SEED)
    if split != expected_split:
        fail(errors, "task_split.json is not the exact seed-42 canonical 5x20 partition")
    if split.get("num_tasks") != NUM_TASKS or split.get("classes_per_task") != CLASSES_PER_TASK:
        fail(errors, "task_split.json has wrong task dimensions")

    split_names = ("train", "calibration", "test")
    all_paths: dict[str, list[Path]] = {}
    counts_by_class: dict[str, Counter[int]] = {}
    checked_readable = 0
    for split_name in split_names:
        split_root = root / split_name
        if not split_root.is_dir():
            fail(errors, f"missing split directory: {split_root}")
            continue
        paths: list[Path] = []
        counts = Counter()
        for class_id, synset in enumerate(IMAGENET100_SYNSETS):
            class_root = split_root / synset
            if not class_root.is_dir():
                fail(errors, f"missing {split_name}/{synset}")
                continue
            images = sorted(p for p in class_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                fail(errors, f"empty class: {split_name}/{synset}")
            counts[class_id] = len(images)
            paths.extend(images)
        all_paths[split_name] = paths
        counts_by_class[split_name] = counts
        if args.max_readable_checks:
            paths_to_check = paths[: args.max_readable_checks]
        else:
            paths_to_check = paths
        for path in paths_to_check:
            try:
                with Image.open(path) as image:
                    image.verify()
                checked_readable += 1
            except Exception as exc:
                fail(errors, f"unreadable image {path}: {exc}")

    resolved_sets = {name: {p.resolve() for p in paths} for name, paths in all_paths.items()}
    for left in split_names:
        for right in split_names:
            if left < right and resolved_sets.get(left, set()) & resolved_sets.get(right, set()):
                fail(errors, f"path overlap between {left} and {right}")

    manifest_rows = manifest.get("samples", [])
    manifest_ids = defaultdict(set)
    manifest_paths = defaultdict(set)
    for row in manifest_rows:
        manifest_ids[row.get("split")].add(row.get("sample_id"))
        manifest_paths[row.get("split")].add(row.get("relative_path"))
        if row.get("synset") not in IMAGENET100_SYNSETS:
            fail(errors, f"manifest contains unknown synset: {row.get('synset')}")
    for left in split_names:
        for right in split_names:
            if left < right and manifest_ids[left] & manifest_ids[right]:
                fail(errors, f"sample-id overlap between {left} and {right}")
            if left < right and manifest_paths[left] & manifest_paths[right]:
                fail(errors, f"manifest path overlap between {left} and {right}")
    if any(set(counts_by_class.get(name, {})) != set(range(NUM_CLASSES)) for name in split_names):
        fail(errors, "one or more splits do not cover all 100 class IDs")
    if len(split.get("tasks", [])) != NUM_TASKS or any(len(task.get("new_class_ids", [])) != CLASSES_PER_TASK for task in split.get("tasks", [])):
        fail(errors, "one or more tasks do not contain exactly 20 classes")

    total_counts = {name: sum(counter.values()) for name, counter in counts_by_class.items()}
    per_class_values = [counts_by_class[name][class_id] for name in split_names for class_id in range(NUM_CLASSES)]
    print(f"DATA_ROOT: {root}")
    print(f"classes: {NUM_CLASSES}; tasks: {NUM_TASKS}x{CLASSES_PER_TASK}; seed: {SEED}")
    print(f"train images: {total_counts.get('train', 0)}")
    print(f"calibration images: {total_counts.get('calibration', 0)}")
    print(f"test images: {total_counts.get('test', 0)}")
    print(f"images/class min/max across splits: {min(per_class_values) if per_class_values else 0}/{max(per_class_values) if per_class_values else 0}")
    print(f"readable images checked: {checked_readable}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("PASS: prepared ImageNet-100 dataset is offline-ready, class/task-complete, readable, and split-disjoint.")
    return 0


def verify_shared(args: argparse.Namespace) -> int:
    root = args.data_root.resolve()
    try:
        report = verify_shared_root(root)
    except (FileNotFoundError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1

    missing_train = report["missing_train"]
    missing_val = report["missing_val"]
    train_found = NUM_CLASSES - len(missing_train)
    val_found = NUM_CLASSES - len(missing_val)
    print("SOURCE MODE: SHARED_IMAGENET1K_FILTERED")
    print(f"SOURCE CLASSES: {report['source_class_count']}")
    print(f"SELECTED PROJECT CLASSES: {NUM_CLASSES}")
    print(f"TASKS: {NUM_TASKS}x{CLASSES_PER_TASK}")
    print(f"EXPECTED PROJECT WNIDS: {NUM_CLASSES}")
    print(f"TRAIN WNIDS FOUND: {train_found}/{NUM_CLASSES}")
    print(f"VAL WNIDS FOUND: {val_found}/{NUM_CLASSES}")
    print(f"MISSING TRAIN: {', '.join(missing_train) if missing_train else 'NONE'}")
    print(f"MISSING VAL: {', '.join(missing_val) if missing_val else 'NONE'}")
    subset_pass = not missing_train and not missing_val and report["source_class_count"] == 1000
    print(f"PROJECT SUBSET MATCH: {'PASS' if subset_pass else 'FAIL'}")
    print("TASK WNIDS (seed 42):")
    for task in task_split(SEED)["tasks"]:
        print(f"  Task {task['task_number']}: {' '.join(task['synsets'])}")
    if not subset_pass:
        return 1

    empty_train = [synset for synset in IMAGENET100_SYNSETS if not report["train_paths"].get(synset)]
    empty_val = [synset for synset in IMAGENET100_SYNSETS if not report["val_paths"].get(synset)]
    if empty_train or empty_val:
        print(f"FAIL: empty selected classes: train={empty_train or 'NONE'}, val={empty_val or 'NONE'}")
        return 1
    try:
        calibration, test = split_validation_paths(
            report["val_paths"], eval_per_class=args.eval_per_class, seed=SEED
        )
    except ValueError as exc:
        print(f"FAIL: invalid shared validation split: {exc}")
        return 1
    calibration_paths = {path.resolve() for paths in calibration.values() for path in paths}
    test_paths = {path.resolve() for paths in test.values() for path in paths}
    disjoint = not calibration_paths.intersection(test_paths)
    print(f"VALIDATION/TEST DISJOINT: {'PASS' if disjoint else 'FAIL'}")
    if not disjoint:
        return 1

    train_counts = [len(report["train_paths"][synset]) for synset in IMAGENET100_SYNSETS]
    print(f"SELECTED TRAIN IMAGES: {sum(train_counts)}")
    print(f"SELECTED TRAIN IMAGES/CLASS MIN/MAX/MEAN: {min(train_counts)}/{max(train_counts)}/{sum(train_counts) / NUM_CLASSES:.2f}")
    print(f"CALIBRATION IMAGES: {sum(len(paths) for paths in calibration.values())}")
    print(f"TEST IMAGES: {sum(len(paths) for paths in test.values())}")

    checked_readable = 0
    errors: list[str] = []
    for paths_by_synset in (report["train_paths"], calibration, test):
        paths = [path for synset in IMAGENET100_SYNSETS for path in paths_by_synset[synset]]
        paths_to_check = paths if not args.max_readable_checks else paths[:args.max_readable_checks]
        for path in paths_to_check:
            try:
                with Image.open(path) as image:
                    image.verify()
                checked_readable += 1
            except Exception as exc:
                errors.append(f"unreadable image {path}: {exc}")
    print(f"READABLE IMAGES CHECKED: {checked_readable}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("CLUSTER NETWORK DATASET ACCESS: NONE")
    print("PASS: shared ImageNet-1K root is project-subset-complete and split-disjoint.")
    return 0


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    if not root.is_dir():
        print(f"FAIL: dataset root does not exist: {root}")
        return 1
    if args.source_mode == "shared_imagenet1k" or (
        args.source_mode == "auto" and (root / "train").is_dir() and (root / "val").is_dir()
    ):
        return verify_shared(args)
    return verify_prepared(args)


if __name__ == "__main__":
    raise SystemExit(main())
