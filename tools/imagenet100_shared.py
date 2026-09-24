"""Offline shared-ImageNet-1K filtering for the project ImageNet-100 subset.

This module deliberately enumerates only the 100 WNID directories selected by
``imagenet100_common.py``.  The other ImageNet-1K class directories are used
only for the cheap source-class-count sanity check.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    from .imagenet100_common import (
        CLASSES_PER_TASK,
        IMAGENET100_SYNSETS,
        NUM_CLASSES,
        NUM_TASKS,
        SEED,
        task_split,
    )
except ImportError:  # direct invocation from the tools directory
    from imagenet100_common import (  # type: ignore
        CLASSES_PER_TASK,
        IMAGENET100_SYNSETS,
        NUM_CLASSES,
        NUM_TASKS,
        SEED,
        task_split,
    )


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _image_paths(class_root: Path) -> list[Path]:
    return sorted(
        path
        for path in class_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _selected_paths(root: Path, split: str) -> tuple[dict[str, list[Path]], list[str]]:
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing ImageNet split directory: {split_root}")
    missing: list[str] = []
    paths_by_synset: dict[str, list[Path]] = {}
    for synset in IMAGENET100_SYNSETS:
        class_root = split_root / synset
        if not class_root.is_dir():
            missing.append(synset)
            continue
        paths_by_synset[synset] = _image_paths(class_root)
    return paths_by_synset, missing


def source_class_count(root: Path) -> int:
    """Count immediate WNID directories without traversing their image files."""
    train_root = root / "train"
    return sum(1 for path in train_root.iterdir() if path.is_dir()) if train_root.is_dir() else 0


def verify_shared_root(data_root: str | Path) -> dict[str, Any]:
    """Return path-only verification data for a shared ImageNet-1K root."""
    root = Path(data_root).resolve()
    train_paths, missing_train = _selected_paths(root, "train")
    val_paths, missing_val = _selected_paths(root, "val")
    return {
        "root": root,
        "source_class_count": source_class_count(root),
        "train_paths": train_paths,
        "val_paths": val_paths,
        "missing_train": missing_train,
        "missing_val": missing_val,
    }


def split_validation_paths(
    paths_by_synset: dict[str, list[Path]],
    *,
    eval_per_class: int = 25,
    seed: int = SEED,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Deterministically split each selected val class into calibration and test."""
    if eval_per_class < 1:
        raise ValueError("eval_per_class must be positive")
    calibration: dict[str, list[Path]] = {}
    test: dict[str, list[Path]] = {}
    for class_id, synset in enumerate(IMAGENET100_SYNSETS):
        paths = list(paths_by_synset.get(synset, []))
        if len(paths) <= eval_per_class:
            raise ValueError(
                f"{synset} has {len(paths)} validation images; need more than {eval_per_class}"
            )
        order = list(range(len(paths)))
        random.Random(seed + class_id).shuffle(order)
        calibration[synset] = [paths[index] for index in order[:eval_per_class]]
        test[synset] = [paths[index] for index in order[eval_per_class:]]
    return calibration, test


def _rows(paths_by_synset: dict[str, list[Path]], split_name: str) -> Iterable[dict[str, Any]]:
    for class_id, synset in enumerate(IMAGENET100_SYNSETS):
        for path in paths_by_synset.get(synset, []):
            yield {
                "image": str(path),
                "label": class_id,
                "sample_id": f"{split_name}:{synset}:{path.as_posix()}",
            }


def _to_dataset(paths_by_synset: dict[str, list[Path]], split_name: str) -> Any:
    from datasets import Dataset, Image as HFImage

    rows = list(_rows(paths_by_synset, split_name))
    dataset = Dataset.from_dict({
        "image": [row["image"] for row in rows],
        "label": [row["label"] for row in rows],
        "sample_id": [row["sample_id"] for row in rows],
    })
    return dataset.cast_column("image", HFImage())


def load_shared_imagenet1k_datasets(
    data_root: str | Path,
    *,
    eval_per_class: int = 25,
    seed: int = SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only project WNIDs from a shared ImageNet-1K ImageFolder tree."""
    report = verify_shared_root(data_root)
    if report["source_class_count"] != 1000:
        raise ValueError(
            "shared ImageNet-1K root must expose exactly 1000 train class directories; "
            f"found {report['source_class_count']}"
        )
    if report["missing_train"] or report["missing_val"]:
        raise FileNotFoundError(
            "project ImageNet-100 WNID directories are missing: "
            f"train={report['missing_train'] or 'NONE'}, "
            f"val={report['missing_val'] or 'NONE'}"
        )
    if any(not report["train_paths"].get(synset) for synset in IMAGENET100_SYNSETS):
        empty = [synset for synset in IMAGENET100_SYNSETS if not report["train_paths"].get(synset)]
        raise ValueError(f"selected train classes have no images: {empty}")
    calibration, test = split_validation_paths(
        report["val_paths"], eval_per_class=eval_per_class, seed=seed
    )
    datasets = {
        "train": _to_dataset(report["train_paths"], "train"),
        "calibration": _to_dataset(calibration, "calibration"),
        "test": _to_dataset(test, "test"),
    }
    report["calibration_paths"] = calibration
    report["test_paths"] = test
    report["eval_per_class"] = eval_per_class
    report["seed"] = seed
    report["task_split"] = task_split(seed)
    return datasets, report


def selected_train_stats(report: dict[str, Any]) -> dict[str, float | int]:
    counts = [len(report["train_paths"].get(synset, [])) for synset in IMAGENET100_SYNSETS]
    return {
        "total": sum(counts),
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "mean": mean(counts) if counts else 0.0,
    }


def write_source_manifest(path: str | Path, report: dict[str, Any]) -> None:
    """Persist exact source paths and stable IDs without copying image data."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": report["train_paths"],
        "calibration": report["calibration_paths"],
        "test": report["test_paths"],
    }
    with output.open("w", encoding="utf-8") as handle:
        for split_name, paths_by_synset in split_paths.items():
            for row in _rows(paths_by_synset, split_name):
                handle.write(json.dumps({"split": split_name, **row}, sort_keys=True) + "\n")


def task_wnids(seed: int = SEED) -> list[list[str]]:
    return [task["synsets"] for task in task_split(seed)["tasks"]]


__all__ = [
    "CLASSES_PER_TASK",
    "IMAGENET100_SYNSETS",
    "NUM_CLASSES",
    "NUM_TASKS",
    "SEED",
    "load_shared_imagenet1k_datasets",
    "selected_train_stats",
    "source_class_count",
    "split_validation_paths",
    "task_wnids",
    "verify_shared_root",
    "write_source_manifest",
]
