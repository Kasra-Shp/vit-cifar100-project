#!/usr/bin/env python3
"""Prepare a transfer-ready ImageNet-100 directory outside the cluster.

Supported sources:
  * ``--source hf``: Hugging Face datasets, including the public
    ``clane9/imagenet-100`` mirror for schema/class-set inspection.
  * ``--source local``: an existing ImageNet-1k-style directory with train/val.

This is intentionally the only new component allowed to perform network I/O.
The output contains ordinary JPEG files and JSON metadata; cluster training
does not need ``datasets`` or Hugging Face access to read it.

The repository's canonical ImageNet-100 definition is deliberately checked
before any full HF split is materialized.  ``clane9/imagenet-100`` is public
and no-login, but it is the CMC random-100 subset; it must not be silently
substituted for the thesis PODNet/DER/DyTox first-100 subset.
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
    from .imagenet100_common import (
        CLANE9_IMAGENET100_SYNSETS,
        IMAGENET100_SYNSETS,
        NUM_CLASSES,
        SEED,
        classes_metadata,
        task_split,
        write_json,
    )
except ImportError:  # direct `python tools/download_imagenet100_external.py`
    from imagenet100_common import (
        CLANE9_IMAGENET100_SYNSETS,
        IMAGENET100_SYNSETS,
        NUM_CLASSES,
        SEED,
        classes_metadata,
        task_split,
        write_json,
    )


DEFAULT_HF_DATASET = "clane9/imagenet-100"
PUBLIC_HF_DATASET = "clane9/imagenet-100"
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
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect HF schema/class compatibility with streaming and do not materialize images",
    )
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


def hf_kwargs(config: str | None, revision: str, cache_dir: Path | None, token: str | None, *, split: str | None = None, streaming: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"revision": revision}
    if split is not None:
        kwargs["split"] = split
    if config:
        kwargs["name"] = config
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if streaming:
        kwargs["streaming"] = True
    auth = token or os.environ.get("HF_TOKEN")
    if auth:
        kwargs["token"] = auth
    return kwargs


def hf_load(dataset_name: str, config: str | None, revision: str, cache_dir: Path | None, token: str | None, *, split: str | None = None, streaming: bool = False) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HF source requires `pip install datasets pillow`; run this only on the external preparation machine") from exc
    kwargs = hf_kwargs(config, revision, cache_dir, token, split=split, streaming=streaming)
    split_text = "all splits" if split is None else f"split={split!r}"
    print(f"Loading HF dataset {dataset_name!r}, {split_text}; this is the external/off-cluster network step.", flush=True)
    try:
        return load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not access {dataset_name!r}. If this is ImageNet, accept its terms and run `huggingface-cli login` or set HF_TOKEN locally; never put the token in this repository. Original error: {exc}"
        ) from exc


def public_synsets_for_dataset(dataset_name: str) -> list[str] | None:
    if dataset_name == PUBLIC_HF_DATASET:
        return list(CLANE9_IMAGENET100_SYNSETS)
    if dataset_name == "ILSVRC/imagenet-1k":
        # The canonical source uses the ImageNet-1k label ordering; the
        # benchmark selects its first 100 sorted WNIDs.  Keep this explicit
        # rather than inferring a WNID mapping from arbitrary HF label names.
        return list(IMAGENET100_SYNSETS)
    return None


def class_set_report(public_synsets: list[str]) -> dict[str, Any]:
    expected_set = set(IMAGENET100_SYNSETS)
    public_set = set(public_synsets)
    missing = [synset for synset in IMAGENET100_SYNSETS if synset not in public_set]
    extra = [synset for synset in public_synsets if synset not in expected_set]
    return {
        "expected_count": len(IMAGENET100_SYNSETS),
        "public_count": len(public_synsets),
        "intersection_count": len(expected_set & public_set),
        "missing_from_public": missing,
        "extra_in_public": extra,
        "match": not missing and not extra and len(public_synsets) == NUM_CLASSES,
    }


def print_class_set_report(report: dict[str, Any]) -> None:
    print(f"EXPECTED WNIDS: {report['expected_count']}")
    print(f"PUBLIC DATASET WNIDS: {report['public_count']}")
    print(f"INTERSECTION: {report['intersection_count']}")
    print("MISSING FROM PUBLIC: " + (", ".join(report["missing_from_public"]) if report["missing_from_public"] else "NONE"))
    print("EXTRA IN PUBLIC: " + (", ".join(report["extra_in_public"]) if report["extra_in_public"] else "NONE"))
    print("CLASS SET MATCH: " + ("PASS" if report["match"] else "FAIL"))


def inspect_hf_source(dataset_name: str, config: str | None, revision: str, cache_dir: Path | None, token: str | None) -> dict[str, Any]:
    """Inspect public/source schema without downloading image shards."""
    ds_dict = hf_load(dataset_name, config, revision, cache_dir, token, streaming=True)
    if not hasattr(ds_dict, "keys"):
        raise RuntimeError(f"Expected a dataset with named splits; got {type(ds_dict).__name__}")
    split_names = list(ds_dict.keys())
    if not split_names:
        raise RuntimeError("HF dataset has no named splits")
    split_details: dict[str, Any] = {}
    label_names: list[str] = []
    for split_name in split_names:
        ds = ds_dict[split_name]
        columns = list(ds.features.keys()) if getattr(ds, "features", None) is not None else list(getattr(ds, "column_names", []))
        label_column = "label" if "label" in columns else "labels" if "labels" in columns else None
        image_column = "image" if "image" in columns else "img" if "img" in columns else None
        if label_column is None or image_column is None:
            raise RuntimeError(f"{dataset_name!r} split {split_name!r} must expose image/label fields; got {columns}")
        feature = ds.features[label_column]
        names = list(feature.names) if getattr(feature, "names", None) else []
        if not label_names:
            label_names = names
        elif names != label_names:
            raise RuntimeError(f"HF label mapping differs between splits: {split_name!r}")
        first_row = next(iter(ds), None)
        split_details[split_name] = {
            "columns": columns,
            "image_column": image_column,
            "label_column": label_column,
            "label_feature": type(feature).__name__,
            "num_labels": len(names),
            "first_label": None if first_row is None else int(first_row[label_column]),
        }
    public_synsets = public_synsets_for_dataset(dataset_name)
    if public_synsets is None:
        public_synsets = []
    if len(label_names) != NUM_CLASSES and dataset_name == PUBLIC_HF_DATASET:
        raise RuntimeError(f"{PUBLIC_HF_DATASET} must expose exactly 100 ClassLabel names; got {len(label_names)}")
    report = class_set_report(public_synsets) if public_synsets else None
    return {
        "split_names": split_names,
        "splits": split_details,
        "label_names": label_names,
        "public_synsets": public_synsets,
        "class_set_report": report,
    }


def hf_split(dataset_name: str, config: str | None, split: str, revision: str, cache_dir: Path | None, token: str | None, selected_only: bool = True) -> tuple[dict[int, list[tuple[str, Any]]], list[str]]:
    ds = hf_load(dataset_name, config, revision, cache_dir, token, split=split)
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
    manifest: list[dict[str, Any]] = []

    # Inspect HF metadata and the source-specific label/WNID mapping before
    # creating or overwriting the output directory.  In particular, this
    # prevents clane9's public but incompatible subset from becoming a
    # misleading canonical prepared dataset.
    hf_inspection: dict[str, Any] | None = None
    if args.source == "hf":
        hf_inspection = inspect_hf_source(args.dataset_name, args.config, args.revision, args.cache_dir, args.token)
        report = hf_inspection["class_set_report"]
        if report is not None:
            print_class_set_report(report)
        print(json.dumps({"splits": hf_inspection["splits"], "label_names": hf_inspection["label_names"]}, indent=2))
        if args.inspect_only:
            if report is not None and not report["match"]:
                print("INSPECTION RESULT: FAIL (public source is not the canonical thesis class set)")
                return 2
            print("INSPECTION RESULT: PASS")
            return 0
        if report is not None and not report["match"]:
            raise RuntimeError(
                f"Refusing to materialize {args.dataset_name!r}: its class set is not the canonical ImageNet-100 definition. "
                "Use the canonical licensed/local source or change the benchmark definition explicitly in a separate study."
            )
        if "train" not in hf_inspection["split_names"] or "validation" not in hf_inspection["split_names"]:
            raise RuntimeError(
                f"HF source must provide train and validation splits for this workflow; got {hf_inspection['split_names']}"
            )
    elif args.inspect_only:
        raise SystemExit("--inspect-only is supported only with --source hf")

    ensure_output(root, args.overwrite)
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
        assert hf_inspection is not None
        train_samples, label_names = hf_split(args.dataset_name, args.config, "train", args.revision, args.cache_dir, args.token)
        eval_samples, _ = hf_split(args.dataset_name, args.config, "validation", args.revision, args.cache_dir, args.token)
        display_names = [label_names[i] if i < len(label_names) else IMAGENET100_SYNSETS[i] for i in range(NUM_CLASSES)]
        source_info.update({
            "public_no_auth_source": args.dataset_name == PUBLIC_HF_DATASET,
            "schema": hf_inspection["splits"],
            "original_label_mapping": [
                {
                    "label_id": i,
                    "class_name": name,
                    "wnid": hf_inspection["public_synsets"][i] if i < len(hf_inspection["public_synsets"]) else None,
                }
                for i, name in enumerate(label_names)
            ],
            "class_set_compatibility": hf_inspection["class_set_report"],
            "calibration_test_split_rule": {
                "held_out_source_split": "validation",
                "calibration_images_per_class": args.eval_per_class,
                "test_images_per_class": "remaining held-out images",
                "seed": args.seed,
                "overlap": "none",
            },
        })
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
