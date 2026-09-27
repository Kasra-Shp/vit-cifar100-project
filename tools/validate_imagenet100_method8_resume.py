"""Validate the exact ImageNet-100 Method-8 task-boundary checkpoint.

This utility is intentionally independent of the training launcher so it can
inspect a checkpoint before importing the large experiment module.  It never
trains, downloads data, or mutates the requested checkpoint.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
from pathlib import Path

import torch

try:
    from tools.imagenet100_continuation_checkpoint import (
        load_torch_payload,
        save_torch_payload_atomic,
    )
except ModuleNotFoundError:  # direct ``python tools/<script>.py`` invocation
    from imagenet100_continuation_checkpoint import load_torch_payload, save_torch_payload_atomic


METHOD = "rank_extension_factor_orth_lam50_fullkd_T2_protect30"
DEFAULT_CHECKPOINT = Path(
    "results/imagenet100_5x20_final_8method_canonical_continuation_methods6to8_seed42_ep9_checkpoints"
) / f"{METHOD}__rankext_state.pt"
DEFAULT_SOURCE = Path(
    "experiments_prepared/final_8method_imagenet100_5x20_continuation_methods6to8_seed42_ep9.py"
)
DEFAULT_DATASET_ROOT = Path("/nfsd/lttm4/datasets/ImageNet-1k_torch")
CLASS_ORDER = Path("experiments_prepared/splits/imagenet100_class_order_seed42.json")
MANIFEST = Path("experiments_prepared/imagenet100_5x20_canonical_protocol_manifest.json")


def _literal_assignment(tree: ast.AST, name: str):
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
    if not values:
        raise AssertionError(f"source does not define literal {name}")
    return values[-1]


def validate_launcher_and_protocol(
    source_path: Path, dataset_root: Path, require_dataset: bool = True
) -> dict[str, object]:
    if not source_path.is_file():
        raise FileNotFoundError(f"resume launcher is missing: {source_path.resolve()}")
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    required_guards = (
        "IMAGENET100_METHOD8_ONLY_RESUME",
        "METHOD8_NAME",
        "METHOD8_RESUME_TASK = 3",
        "METHOD8_ONLY_RESUME",
        "rank_extension_execution_order == [METHOD8_NAME]",
        "Task 4 restarts at epoch 0",
    )
    for guard in required_guards:
        if guard not in source_text:
            raise AssertionError(f"resume source is missing required guard: {guard}")

    methods_to_run = _literal_assignment(tree, "METHODS_TO_RUN")
    expected_base_methods = {
        "rank_extension_fullkd_T2_protect30",
        "rank_extension_factor_orth_lam50_new",
        METHOD,
    }
    source_enabled = {name for name, enabled in methods_to_run.items() if enabled}
    if source_enabled != expected_base_methods:
        raise AssertionError(f"unexpected canonical continuation flags: {sorted(source_enabled)}")
    simulated_method8_only = {METHOD}

    schedule = _literal_assignment(tree, "RANKEXT_RANK_SCHEDULE")
    num_steps = _literal_assignment(tree, "NUM_STEPS")
    classes_per_step = _literal_assignment(tree, "CLASSES_PER_STEP")
    rankext_epochs = _literal_assignment(tree, "RANKEXT_EPOCHS")
    if (schedule, num_steps, classes_per_step, rankext_epochs) != ([16, 32, 48, 64, 80], 5, 20, 9):
        raise AssertionError("resume source protocol constants do not match canonical 5x20/9-epoch protocol")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    class_order_hash = hashlib.sha256(CLASS_ORDER.read_bytes()).hexdigest()
    expected_manifest = {
        "seed": 42,
        "num_classes": 100,
        "num_tasks": 5,
        "classes_per_task": 20,
        "epochs": 9,
        "rank_schedule": [16, 32, 48, 64, 80],
        "wnid_split_sha256": class_order_hash,
        "dataset_identifier": "ImageNet-100/PODNet-DER-DyTox-first100-sorted-wnids",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise AssertionError(f"protocol manifest mismatch for {key}: {manifest.get(key)!r} != {expected!r}")
    if dataset_root != DEFAULT_DATASET_ROOT:
        raise AssertionError(f"dataset root must be exactly {DEFAULT_DATASET_ROOT}, got {dataset_root}")
    if require_dataset and (
        not dataset_root.is_dir()
        or not (dataset_root / "train").is_dir()
        or not (dataset_root / "val").is_dir()
    ):
        raise AssertionError(f"shared ImageNet-1K dataset root/train/val is not available: {dataset_root}")

    return {
        "methods_active_in_recovery": sorted(simulated_method8_only),
        "methods_1_to_7_train": "NO",
        "seed": 42,
        "protocol": "5 tasks x 20 classes, 9 epochs/task",
        "rank_schedule": schedule,
        "wnid_split_sha256": class_order_hash,
        "dataset_root": str(dataset_root),
        "task4_behavior": "restart from epoch 0; no mid-epoch cursor/optimizer state accepted",
        "done_marker": "absent before recovery",
    }


def validate_payload(payload: dict, path: Path) -> dict[str, object]:
    done_marker = path.parent / f"{METHOD}__DONE.marker"
    if done_marker.exists():
        raise AssertionError(f"incomplete-resume validation refuses an existing DONE.marker: {done_marker}")
    required = {
        "method_name", "completed_task_index", "previous_rank_state",
        "stepwise_task_accuracies", "forward_transfer_probe", "rng_state",
        "rankext_state_fingerprint", "config_fingerprint",
    }
    missing = required - set(payload)
    if missing:
        raise AssertionError(f"missing checkpoint keys: {sorted(missing)}")
    if payload["method_name"] != METHOD:
        raise AssertionError(f"wrong method_name: {payload['method_name']!r}")
    if int(payload["completed_task_index"]) != 3:
        raise AssertionError(
            "safe resume requires completed_task_index=3; refusing any mid-task or other boundary"
        )
    for key in ("current_task_index", "current_epoch", "epoch", "batch_index", "optimizer_state"):
        if key in payload:
            raise AssertionError(f"unverified mid-task metadata present: {key}")

    state = payload["previous_rank_state"]
    for key in ("lora", "classifier_weight", "classifier_bias"):
        if key not in state:
            raise AssertionError(f"persistent RankExt state missing {key!r}")
    weight = state["classifier_weight"]
    bias = state["classifier_bias"]
    if tuple(weight.shape) != (100, 768) or tuple(bias.shape) != (100,):
        raise AssertionError(f"classifier/head state shapes are {tuple(weight.shape)} / {tuple(bias.shape)}")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise AssertionError("classifier/head state contains non-finite values")

    lora = state["lora"]
    if len(lora) != 24:
        raise AssertionError(f"expected 24 q/v LoRA modules, found {len(lora)}")
    for module_name, entry in lora.items():
        if not module_name.endswith(("q_proj", "v_proj")):
            raise AssertionError(f"unexpected persistent module: {module_name}")
        for key in ("A", "B", "frozen_rank", "new_rank", "total_rank"):
            if key not in entry:
                raise AssertionError(f"{module_name}: missing {key}")
        ranks = (int(entry["total_rank"]), int(entry["frozen_rank"]), int(entry["new_rank"]))
        if ranks != (48, 32, 16):
            raise AssertionError(f"{module_name}: expected Task-3 ranks (48,32,16), got {ranks}")
        if tuple(entry["A"].shape) != (48, 768) or tuple(entry["B"].shape) != (768, 48):
            raise AssertionError(f"{module_name}: unexpected A/B shapes")
        if not torch.isfinite(entry["A"]).all() or not torch.isfinite(entry["B"]).all():
            raise AssertionError(f"{module_name}: non-finite persistent state")

    if set(payload["stepwise_task_accuracies"]) != {0, 1, 2}:
        raise AssertionError("stepwise accuracy state must contain Tasks 1-3 only")
    if set(payload["forward_transfer_probe"]) != {1, 2}:
        raise AssertionError("forward-transfer state must contain Tasks 2-3 only")
    if not {"python", "numpy", "torch"}.issubset(payload["rng_state"]):
        raise AssertionError("deterministic RNG state is incomplete")
    if not payload["rankext_state_fingerprint"] or not payload["config_fingerprint"]:
        raise AssertionError("checkpoint fingerprints are missing")

    return {
        "checkpoint": str(path),
        "method": METHOD,
        "completed_tasks": "1,2,3",
        "resume_start_task": 4,
        "rank_structure": "total=48, frozen=32, new=16 per q_proj/v_proj module",
        "classifier_head": "weight=(100,768), bias=(100,)",
        "persistent_state": "24 CLIP q/v LoRA modules present",
        "determinism": "python/numpy/torch RNG state present; KD/protection pinned by method config",
    }


def synthetic_payload() -> dict:
    lora = {}
    for layer in range(12):
        for projection in ("q_proj", "v_proj"):
            name = f"vision_model.vision_model.encoder.layers.{layer}.self_attn.{projection}"
            lora[name] = {
                "A": torch.zeros(48, 768), "B": torch.zeros(768, 48),
                "frozen_rank": 32, "new_rank": 16, "total_rank": 48,
            }
    return {
        "method_name": METHOD, "completed_task_index": 3,
        "previous_rank_state": {
            "lora": lora,
            "classifier_weight": torch.zeros(100, 768),
            "classifier_bias": torch.zeros(100),
        },
        "stepwise_task_accuracies": {0: {}, 1: {}, 2: {}},
        "forward_transfer_probe": {1: 0.0, 2: 0.0},
        "rng_state": {"python": None, "numpy": None, "torch": torch.zeros(1, dtype=torch.uint8)},
        "rankext_state_fingerprint": "synthetic",
        "config_fingerprint": "synthetic",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--synthetic", action="store_true", help="validate an in-memory task-boundary payload")
    args = parser.parse_args()
    report = validate_launcher_and_protocol(
        args.source, args.dataset_root, require_dataset=not args.synthetic
    )
    if args.synthetic:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.pt"
            payload = synthetic_payload()
            save_torch_payload_atomic(path, payload)
            report.update(validate_payload(load_torch_payload(path), path))
    else:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"required checkpoint is missing: {args.checkpoint.resolve()}")
        report.update(validate_payload(load_torch_payload(args.checkpoint), args.checkpoint))
    for key, value in report.items():
        print(f"{key}: {value}")
    print("CHECKPOINT VALID: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
