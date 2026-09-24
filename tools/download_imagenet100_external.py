#!/usr/bin/env python3
"""Prepare a transfer-ready ImageNet-100 directory outside the cluster.

Supported sources:
  * ``--source hf``: authenticated Hugging Face ILSVRC/imagenet-1k.
  * ``--source local``: an existing ImageNet-1k-style directory with train/val.

This is intentionally the only new component allowed to perform network I/O.
The output contains ordinary JPEG files and JSON metadata; cluster training
does not need ``datasets`` or Hugging Face access to read it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

try:
    from .imagenet100_common import IMAGENET100_SYNSETS, NUM_CLASSES, SEED, classes_metadata, task_split, write_json
except ImportError:  # direct `python tools/download_imagenet100_external.py`
    from imagenet100_common import IMAGENET100_SYNSETS, NUM_CLASSES, SEED, classes_metadata, task_split, write_json


DEFAULT_HF_DATASET = "ILSVRC/imagenet-1k"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("hf", "local"), default="hf")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-name", default=DEFAULT_HF_DATASET)
    p.add_argument("--config", default=None)
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--token", default=None, help="HF token; prefer HF_TOKEN or huggingface-cli login")
    p.add_argument("--source-dir", type=Path, default=None, help="Existing ImageNet root for --source local")
    p.add_argument("--train-subdir", default="train")
    p.add_argument("--eval-subdir", default="val")
    p.add_argument("--eval-per-class", type=int, default=25, help="Calibration images/class; remaining held-out images become test")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ensure_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is non-empty: {root}. Use --overwrite only after checking it.")
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for split in ("train", "calibration", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def save_rgb(source: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, Image.Image):
        image = source
    else:
        image = Image.open(source)
    image.convert("RGB").save(destination, format="JPEG", quality=95)


def split_eval_samples(samples: dict[int, list[tuple[str, Any]]], eval_per_class: int, seed: int) -> tuple[dict[int, list[tuple[str, Any]]], dict[int, list[tuple[str, Any]]]]:
    calibration: dict[int, list[tuple[str, Any]]] = defaultdict(list)
    test: dict[int, list[tuple[str, Any]]] = defaultdict(list)
    for class_id in range(NUM_CLASSES):
        items = sorted(samples[class_id], key=lambda x: x[0])
        if len(items) <= eval_per_class:
            raise RuntimeError(f"Class {class_id} has {len(items)} held-out images; need > {eval_per_class} for calibration and test")
        rng = random.Random(seed + class_id)
        indices = list(range(len(items)))
        rng.shuffle(indices)
        calibration[class_id] = [items[i] for i in indices[:eval_per_class]]
        test[class_id] = [items[i] for i in indices[eval_per_class:]]
    return calibration, test


def copy_local_split(source_root: Path, split_dir_name: str, output_root: Path, split_name: str, manifest: list[dict[str, Any]]) -> dict[int, list[tuple[str, Path]]]:
    source_split = source_root / split_dir_name
    if not source_split.is_dir():
        raise RuntimeError(f"Missing local source split: {source_split}")
    collected: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for class_id, synset in enumerate(IMAGENET100_SYNSETS):
        class_dir = source_split / synset
        if not class_dir.is_dir():
            raise RuntimeError(f"Missing selected synset directory: {class_dir}")
        for path in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            collected[class_id].append((path.as_posix(), path))
    return collected


def write_collected(collected: dict[int, list[tuple[str, Any]]], split_name: str, output_root: Path, manifest: list[dict[str, Any]], source_split: str) -> None:
    for class_id in range(NUM_CLASSES):
        synset = IMAGENET100_SYNSETS[class_id]
        for ordinal, (source_id, image_source) in enumerate(sorted(collected[class_id], key=lambda x: x[0])):
            filename = f"{source_split}_{class_id:03d}_{ordinal:06d}.JPEG"
            destination = output_root / split_name / synset / filename
            save_rgb(image_source, destination)
            manifest.append({
                "split": split_name,
                "benchmark_class_id": class_id,
                "synset": synset,
                "sample_id": f"{source_split}:{source_id}",
                "relative_path": destination.relative_to(output_root).as_posix(),
            })


def hf_split(dataset_name: str, config: str | None, split: str, revision: str, cache_dir: Path | None, token: str | None, selected_only: bool = True) -> tuple[dict[int, list[tuple[str, Any]]], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HF source requires `pip install datasets pillow`; run this only on the external preparation machine") from exc
    kwargs: dict[str, Any] = {"split": split, "revision": revision}
    if config:
        kwargs["name"] = config
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    auth = token or os.environ.get("HF_TOKEN")
    if auth:
        kwargs["token"] = auth
    print(f"Loading HF dataset {dataset_name!r}, split={split!r}; this is the external/off-cluster network step.", flush=True)
    try:
        ds = load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not access {dataset_name!r}. If this is ImageNet, accept its terms and run `huggingface-cli login` or set HF_TOKEN locally; never put the token in this repository. Original error: {exc}"
        ) from exc
    label_column = "label" if "label" in ds.column_names else "labels"
    image_column = "image" if "image" in ds.column_names else "img"
    if label_column not in ds.column_names or image_column not in ds.column_names:
        raise RuntimeError(f"Expected image/label columns; got {ds.column_names}")
    label_names = list(getattr(getattr(ds.features[label_column], "names", None), "__iter__", lambda: [])()) if getattr(ds.features[label_column], "names", None) else []
    selected: dict[int, list[tuple[str, Any]]] = defaultdict(list)
    for index, row in enumerate(ds):
        label = int(row[label_column])
        if selected_only and not 0 <= label < NUM_CLASSES:
            continue
        source_id = f"{split}:{index:09d}"
        selected[label].append((source_id, row[image_column]))
        if index and index % 10000 == 0:
            print(f"  scanned {index} source records; selected {sum(map(len, selected.values()))}", flush=True)
    return selected, label_names


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    if args.seed != SEED:
        raise SystemExit(f"This benchmark preparation is pinned to seed {SEED}; got {args.seed}")
    if args.eval_per_class < 1:
        raise SystemExit("--eval-per-class must be positive")
    root = args.output_dir.resolve()
    ensure_output(root, args.overwrite)
    manifest: list[dict[str, Any]] = []
    source_info: dict[str, Any] = {
        "provider": "Hugging Face" if args.source == "hf" else "user-provided local ImageNet directory",
        "dataset_identifier": args.dataset_name if args.source == "hf" else None,
        "config": args.config,
        "revision": args.revision if args.source == "hf" else None,
        "authenticated": bool(args.source == "hf" and (args.token or os.environ.get("HF_TOKEN"))),
        "selected_synsets": IMAGENET100_SYNSETS,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_splits": {"train": "train", "held_out": "validation" if args.source == "hf" else args.eval_subdir},
        "license_note": "ImageNet access terms apply; this tool does not bypass gating or terms.",
    }

    if args.source == "hf":
        train_samples, label_names = hf_split(args.dataset_name, args.config, "train", args.revision, args.cache_dir, args.token)
        eval_samples, _ = hf_split(args.dataset_name, args.config, "validation", args.revision, args.cache_dir, args.token)
        display_names = [label_names[i] if i < len(label_names) else IMAGENET100_SYNSETS[i] for i in range(NUM_CLASSES)]
    else:
        if args.source_dir is None:
            raise SystemExit("--source-dir is required with --source local")
        train_samples = copy_local_split(args.source_dir.resolve(), args.train_subdir, root, "train", manifest)
        eval_samples = copy_local_split(args.source_dir.resolve(), args.eval_subdir, root, "eval", manifest)
        display_names = IMAGENET100_SYNSETS

    for split_name, samples in (("train", train_samples), ("held_out", eval_samples)):
        missing = [i for i in range(NUM_CLASSES) if not samples.get(i)]
        if missing:
            raise RuntimeError(f"Selected split {split_name} is missing classes: {missing}")
    calibration, test = split_eval_samples(eval_samples, args.eval_per_class, args.seed)
    write_collected(train_samples, "train", root, manifest, "train")
    write_collected(calibration, "calibration", root, manifest, "calibration")
    write_collected(test, "test", root, manifest, "test")

    split_payload = task_split(args.seed)
    write_json(root / "metadata" / "classes.json", classes_metadata(display_names))
    write_json(root / "metadata" / "task_split.json", split_payload)
    source_info["final_layout"] = {"train": "train", "calibration": "calibration", "test": "test"}
    write_json(root / "metadata" / "source.json", source_info)
    counts = defaultdict(int)
    for row in manifest:
        counts[row["split"]] += 1
    manifest_payload = {
        "schema_version": 2,
        "num_classes": NUM_CLASSES,
        "selected_synsets": IMAGENET100_SYNSETS,
        "counts": dict(counts),
        "seed": args.seed,
        "samples": manifest,
    }
    write_json(root / "metadata" / "dataset_manifest.json", manifest_payload)
    print(json.dumps({"output_dir": str(root), "counts": dict(counts), "metadata": str(root / "metadata")}, indent=2))
    print("Next: run tools/verify_imagenet100_dataset.py before packaging or transfer.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
