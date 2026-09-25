#!/usr/bin/env python
"""Cheap ImageNet-100 startup preflight: one real batch, one CUDA forward.

This intentionally performs no optimizer step, backward pass, epoch, or
training update.  It uses the same shared ImageNet-1K root, canonical 100
WNIDs, seed-42 label remap, CLIP preprocessing, and 100-way model shape as the
production benchmark.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from datasets import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.imagenet100_common import IMAGENET100_SYNSETS  # noqa: E402
from tools.imagenet100_index_cache import ImageNetTaskIndexCache, select_dataset  # noqa: E402
from tools.imagenet100_shared import load_shared_imagenet1k_datasets  # noqa: E402


DATA_ROOT = os.environ.get("IMAGENET100_ROOT", "/nfsd/lttm4/datasets/ImageNet-1k_torch")
CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "openai/clip-vit-base-patch16")
ARTIFACT = REPO_ROOT / "experiments_prepared" / "splits" / "imagenet100_class_order_seed42.json"
BATCH_SIZE = int(os.environ.get("IMAGENET100_PREFLIGHT_BATCH_SIZE", "8"))


def to_pil(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if "array" in value:
            value = value["array"]
        elif "bytes" in value:
            import io
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, list):
        value = np.asarray(value, dtype=np.uint8)
    if isinstance(value, np.ndarray):
        array = np.squeeze(value).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array).convert("RGB")
    return value


def main() -> int:
    print(f"PREFLIGHT DATA ROOT: {DATA_ROOT}")
    print(f"CANONICAL 100 CLASSES: {len(IMAGENET100_SYNSETS)}")
    if not Path(DATA_ROOT).is_dir():
        print(f"FATAL: dataset root missing: {DATA_ROOT}", file=sys.stderr)
        return 1

    with ARTIFACT.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert artifact["seed"] == 42
    assert artifact["num_classes"] == 100
    assert artifact["num_tasks"] == 5
    assert artifact["classes_per_task"] == 20
    task_splits = [task["new_class_ids"] for task in artifact["tasks"]]
    original_to_new = {int(key): int(value) for key, value in artifact["original_id_to_new_id"].items()}

    load_started = perf_counter()
    datasets, _ = load_shared_imagenet1k_datasets(DATA_ROOT, seed=42)
    print(f"DATASET LOAD: {perf_counter() - load_started:.3f}s")

    remapped_labels = {
        split_name: np.asarray([original_to_new[int(label)] for label in split_ds["label"]], dtype=np.int64)
        for split_name, split_ds in datasets.items()
    }
    index_started = perf_counter()
    caches = {
        split_name: ImageNetTaskIndexCache.build(labels, task_splits)
        for split_name, labels in remapped_labels.items()
    }
    print(f"TRAIN INDEX BUILD: {perf_counter() - index_started:.3f}s")

    select_started = perf_counter()
    task1 = select_dataset(datasets["train"], caches["train"].task_indices[0])
    print(f"TASK1 SELECT: {perf_counter() - select_started:.3f}s ({len(task1)} samples)")

    processor = CLIPImageProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
    if processor.crop_size is not None:
        height = int(processor.crop_size.get("height", 224))
        width = int(processor.crop_size.get("width", 224))
    else:
        height = width = 224
    shortest_edge = int(getattr(processor, "size", {}).get("shortest_edge", max(height, width)))
    train_transform = transforms.Compose([
        transforms.Resize(shortest_edge, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    def preprocess(example):
        return {
            "pixel_values": train_transform(to_pil(example["image"])),
            "labels": int(original_to_new[int(example["label"])]),
        }

    task1 = task1.with_transform(preprocess)

    def collate(examples):
        return {
            "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
            "labels": torch.tensor([example["labels"] for example in examples], dtype=torch.long),
        }

    cuda_available = torch.cuda.is_available()
    print(f"CUDA AVAILABLE: {cuda_available}")
    if not cuda_available:
        print("FATAL: CUDA is required for the GPU preflight", file=sys.stderr)
        return 1
    print(f"CUDA DEVICE: {torch.cuda.get_device_name(0)}")

    loader_started = perf_counter()
    loader = torch.utils.data.DataLoader(
        task1,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=int(os.environ.get("IMAGENET100_PREFLIGHT_WORKERS", "0")),
        collate_fn=collate,
    )
    print(f"DATALOADER CONSTRUCTION: {perf_counter() - loader_started:.3f}s")

    batch_started = perf_counter()
    batch = next(iter(loader))
    print(f"FIRST BATCH LOAD: PASS ({perf_counter() - batch_started:.3f}s)")

    model_started = perf_counter()
    vision_model = CLIPVisionModel.from_pretrained(CHECKPOINT, use_safetensors=True, local_files_only=True)
    classifier = torch.nn.Linear(vision_model.config.hidden_size, 100)
    vision_model.eval().cuda()
    classifier.eval().cuda()
    print(f"MODEL CONSTRUCTION: {perf_counter() - model_started:.3f}s")

    pixel_values = batch["pixel_values"].cuda(non_blocking=True)
    forward_started = perf_counter()
    torch.cuda.synchronize()
    with torch.no_grad():
        pooled = vision_model(pixel_values=pixel_values, return_dict=True).pooler_output
        logits = classifier(pooled)
    torch.cuda.synchronize()
    print(f"FIRST GPU FORWARD: PASS ({perf_counter() - forward_started:.3f}s)")
    print(f"GPU ALLOCATED MB: {torch.cuda.memory_allocated() / (1024 ** 2):.1f}")
    assert tuple(logits.shape) == (min(BATCH_SIZE, len(task1)), 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
