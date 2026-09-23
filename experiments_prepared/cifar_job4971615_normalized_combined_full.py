#!/usr/bin/env python
"""Full canonical job4971615 CIFAR run for normalized Combined RankExt.

The reference file is loaded read-only and its definitions are executed after
top-level training/reporting statements have been removed from an in-memory
AST.  The reference source itself is never imported as a module and is never
modified.  Exactly one new RankExt method is trained here.
"""

from __future__ import annotations

import ast
import csv
import gc
import hashlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "experiments_prepared" / "final_9method_5x20_performance_recovery.py"
OUTPUT_DIR = REPO_ROOT / "R8" / "performance_improvement_research"
SEED = 42
NUM_STEPS = 5
CLASSES_PER_TASK = 20
EPOCHS = 9
RANK_SCHEDULE = [16, 32, 48, 64, 80]
BATCH_SIZE = 16
METHOD_NAME = "rank_extension_normalized_factor_orth_lam50_fullkd_T2_protect30"
CANONICAL_COMBINED = "rank_extension_factor_orth_lam50_fullkd_T2_protect30"
RUN_NAME = "cifar_job4971615_normalized_combined_full_seed42_9ep"


def _name_of_target(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.Tuple, ast.List)):
        names = []
        for item in node.elts:
            name = _name_of_target(item)
            if name:
                names.append(name)
        return names
    return None


def _constant_node(value):
    return ast.parse(repr(value), mode="eval").body


def load_canonical_definitions():
    """Execute definitions/configuration from the real canonical CIFAR file.

    This deliberately keeps function/class bodies byte-for-byte from the
    reference source while omitting its top-level dataset/training/reporting
    execution blocks.  Assertions are omitted because this probe changes only
    protocol size, not the RankExt mechanisms under test.
    """
    source = CANONICAL.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CANONICAL))
    replacement_values = {
        "SEED": 42,
        "FAST_RUN": False,
        "NUM_STEPS": NUM_STEPS,
        "CLASSES_PER_STEP": CLASSES_PER_TASK,
        "FULL_FT_EPOCHS": EPOCHS,
        "FULL_LORA_EPOCHS": EPOCHS,
        "FULL_JOINT_EPOCHS": EPOCHS,
        "FULL_ORTH_EPOCHS": EPOCHS,
        "FULL_RANKEXT_EPOCHS": EPOCHS,
        "SCRATCH_EPOCHS": EPOCHS,
        "FT_EPOCHS": EPOCHS,
        "LORA_EPOCHS": EPOCHS,
        "JOINT_EPOCHS": EPOCHS,
        "ORTH_EPOCHS": EPOCHS,
        "RANKEXT_EPOCHS": EPOCHS,
        "BATCH_LORA": BATCH_SIZE,
        "RANKEXT_RANK_SCHEDULE": RANK_SCHEDULE,
        "RANKEXT_RANK_SCHEDULE_WIDE": RANK_SCHEDULE,
        "USE_RANKEXT_RANK_SCHEDULE_WIDE": False,
        "RUN_NAME_BASE": RUN_NAME,
    }
    single_method = {
        "rank_extension_factor_orth_lam50_fullkd_T2_protect30": True,
    }
    skip_assignments = {
        "dataset", "LABEL_COL", "IMAGE_COL", "image_processor", "H", "W", "train_transform", "val_transform", "class_splits",
        "first_step_classes", "later_step_classes", "all_classes", "train_source",
        "val_source", "train_val_split_df", "validation_split_path", "eval_first",
        "eval_later", "eval_all_seen", "ROOT_RESULTS_DIR", "BASE_OUTPUT_DIR",
        "TABLES_DIR", "PLOTS_DIR", "REPORTS_DIR", "LOGS_DIR", "CONFIGS_DIR",
        "MODELS_DIR", "CHECKPOINTS_DIR", "training_merge_summary_csv_path",
        "historical_r7_reference_csv_path",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assert):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body.append(node)
            continue
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.AsyncWith, ast.Try)):
            # All executable notebook cells are top-level control-flow nodes;
            # function/class internals are retained untouched above.
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if int(getattr(node, "lineno", 0)) >= 9300:
                continue
            targets = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = _name_of_target(target)
                    targets.extend(name if isinstance(name, list) else [name])
            else:
                name = _name_of_target(node.target)
                targets.extend(name if isinstance(name, list) else [name])
            targets = {x for x in targets if x}
            if targets & skip_assignments or any(str(x).startswith("final9_") for x in targets):
                continue
            if "METHODS_TO_RUN" in targets:
                body.append(ast.Assign(
                    targets=[ast.Name(id="METHODS_TO_RUN", ctx=ast.Store())],
                    value=_constant_node(single_method),
                ))
                continue
            if len(targets) == 1 and next(iter(targets)) in replacement_values:
                name = next(iter(targets))
                if isinstance(node, ast.Assign):
                    body.append(ast.Assign(
                        targets=node.targets,
                        value=_constant_node(replacement_values[name]),
                    ))
                else:
                    body.append(ast.AnnAssign(
                        target=node.target, annotation=node.annotation,
                        value=_constant_node(replacement_values[name]), simple=node.simple,
                    ))
                continue
            body.append(node)
            continue
        # Top-level expressions are prints, file writes, and other notebook
        # execution.  None are needed for the reusable definitions.
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    env = {
        "__name__": "__canonical_job4971615_probe_core__",
        "__file__": str(CANONICAL),
    }
    exec(compile(module, str(CANONICAL), "exec"), env, env)
    return env


def initialize_full_canonical_data(env):
    """Initialize the exact canonical CIFAR-100 5x20 data path in new dirs."""
    dataset = env["load_dataset"]("cifar100")
    label_col = "fine_label"
    image_col = "img"
    assert dataset["train"].num_rows == 50000 and dataset["test"].num_rows == 10000
    class_splits = [list(range(i * 20, (i + 1) * 20)) for i in range(5)]
    assert class_splits == [list(range(i * 20, (i + 1) * 20)) for i in range(5)]
    selected_classes = [c for block in class_splits for c in block]
    env.update({
        "dataset": dataset, "LABEL_COL": label_col, "IMAGE_COL": image_col,
        "class_splits": class_splits, "first_step_classes": class_splits[0],
        "later_step_classes": [c for b in class_splits[1:] for c in b],
        "all_classes": selected_classes,
    })

    processor = env["CLIPImageProcessor"].from_pretrained(env["MODEL_CHECKPOINT"])
    crop = processor.crop_size if getattr(processor, "crop_size", None) is not None else {"height": 224, "width": 224}
    height = int(crop.get("height", 224))
    width = int(crop.get("width", 224))
    transforms = env["transforms"]
    train_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomCrop((height, width), padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    env.update({
        "image_processor": processor, "H": height, "W": width,
        "train_transform": train_transform, "val_transform": val_transform,
    })
    train_source, val_source, split_df = env["build_classwise_train_val_splits"](
        dataset["train"], val_per_class=25,
    )
    env.update({"train_source": train_source, "val_source": val_source, "train_val_split_df": split_df})
    return class_splits


def add_normalized_factororth(env):
    original = env["compute_delta_orth_components"]
    env["PROBE_CURRENT_METHOD"] = METHOD_NAME
    env["PROBE_CURRENT_STEP"] = 0
    env["PROBE_TRAINER_STATE"] = None
    env["FACTOR_TRACE"] = []

    def normalized_components(model, eps=1e-12):
        raw = original(model, eps=eps)
        active_modules = [
            module for module in model.modules()
            if isinstance(module, env["GrowingRankLoRALinear"])
        ]
        total_rank = int(active_modules[0].total_rank) if active_modules else 0
        schedule = list(env.get("RANKEXT_RANK_SCHEDULE", RANK_SCHEDULE))
        trace_step = schedule.index(total_rank) + 1 if total_rank in schedule else env["PROBE_CURRENT_STEP"]
        trainer_state = env.get("PROBE_TRAINER_STATE")
        epoch = float(getattr(trainer_state, "epoch", np.nan)) if trainer_state is not None else np.nan

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        a_terms, b_terms, pair_terms, a_abs, b_abs = [], [], [], [], []
        for name, module in model.named_modules():
            if not isinstance(module, env["GrowingRankLoRALinear"]):
                continue
            if module.frozen_rank <= 0 or module.new_rank <= 0:
                continue
            A_old = module.A_frozen.to(device=device, dtype=dtype)
            B_old = module.B_frozen.to(device=device, dtype=dtype)
            A_new, B_new = module.A_new, module.B_new
            A_old_hat = A_old / A_old.norm(dim=1, keepdim=True).clamp_min(eps)
            A_new_hat = A_new / A_new.norm(dim=1, keepdim=True).clamp_min(eps)
            B_old_hat = B_old / B_old.norm(dim=0, keepdim=True).clamp_min(eps)
            B_new_hat = B_new / B_new.norm(dim=0, keepdim=True).clamp_min(eps)
            denom = float(max(1, int(module.frozen_rank) * int(module.new_rank)))
            norm_a = (A_old_hat @ A_new_hat.T).pow(2).sum() / denom
            norm_b = (B_old_hat.T @ B_new_hat).pow(2).sum() / denom
            pair = 0.5 * (norm_a + norm_b)
            a_terms.append(norm_a)
            b_terms.append(norm_b)
            pair_terms.append(pair)
            a_abs.append((A_old_hat @ A_new_hat.T).abs().mean())
            b_abs.append((B_old_hat.T @ B_new_hat).abs().mean())
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        norm_a = torch.stack(a_terms).mean() if a_terms else zero
        norm_b = torch.stack(b_terms).mean() if b_terms else zero
        norm_total = torch.stack(pair_terms).mean() if pair_terms else zero
        env["FACTOR_TRACE"].append({
            "method": env["PROBE_CURRENT_METHOD"], "step": trace_step,
            "epoch": epoch, "training": bool(model.training),
            "raw_factororth": float(raw["factor_total_mean"].detach().cpu().item()),
            "normalized_factororth": float(norm_total.detach().cpu().item()),
        })
        raw.update({
            "factor_A_mean": norm_a, "factor_B_mean": norm_b,
            "factor_total_mean": norm_total,
            "mean_A_overlap": torch.stack(a_abs).mean() if a_abs else zero,
            "mean_B_overlap": torch.stack(b_abs).mean() if b_abs else zero,
        })
        return raw

    env["compute_delta_orth_components"] = normalized_components
    trainer_cls = env["DeltaOrthRankExtensionTrainer"]
    original_compute_loss = trainer_cls.compute_loss

    def tracked_compute_loss(self, *args, **kwargs):
        env["PROBE_TRAINER_STATE"] = self.state
        return original_compute_loss(self, *args, **kwargs)

    trainer_cls.compute_loss = tracked_compute_loss


def predict_logits(env, model, ds, device):
    model.eval()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=env["collate_fn"], num_workers=0,
    )
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels.append(batch["labels"].cpu().numpy())
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits.append(model(**batch).logits.detach().cpu().numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def restricted_accuracy(logits, labels, class_splits):
    task_for_label = {int(c): i for i, block in enumerate(class_splits) for c in block}
    preds = []
    for row, label in zip(logits, labels):
        block = class_splits[task_for_label[int(label)]]
        masked = np.full_like(row, -np.inf, dtype=np.float64)
        masked[block] = row[block]
        preds.append(int(np.argmax(masked)))
    return float(np.mean(np.asarray(preds) == labels))


def task_ids_for_labels(labels, class_splits):
    task_for_label = {int(c): i for i, block in enumerate(class_splits) for c in block}
    return np.asarray([task_for_label[int(v)] for v in labels], dtype=np.int64)


def save_logits(path, logits, labels, class_splits):
    np.savez_compressed(
        path, logits=np.asarray(logits, dtype=np.float32), labels=np.asarray(labels, dtype=np.int64),
        task_ids=task_ids_for_labels(labels, class_splits),
        sample_ids=np.arange(len(labels), dtype=np.int64),
    )


def aggregate_factor_trajectory(env, loss_rows, class_splits, output_dir):
    traces = [x for x in env["FACTOR_TRACE"] if x.get("training") and x["method"] == METHOD_NAME]
    rows = []
    for step in range(1, NUM_STEPS + 1):
        step_traces = [x for x in traces if int(x["step"]) == step]
        step_losses = [x for x in loss_rows if int(x.get("step", -1)) == step]
        epoch_keys = sorted({int(math.floor(float(x.get("epoch", 0.0)))) for x in step_traces})
        if not epoch_keys:
            epoch_keys = list(range(EPOCHS))
        for epoch in epoch_keys:
            tr = [x for x in step_traces if int(math.floor(float(x.get("epoch", 0.0)))) == epoch]
            lr = [x for x in step_losses if int(math.floor(float(x.get("epoch", 0.0)))) == epoch]
            if not tr:
                continue
            raw = float(np.mean([x["raw_factororth"] for x in tr]))
            norm = float(np.mean([x["normalized_factororth"] for x in tr]))
            ce_vals = [float(x["ce_loss"]) for x in lr if math.isfinite(float(x.get("ce_loss", np.nan)))]
            kd_vals = [float(x.get("weighted_kd_loss", 0.0)) for x in lr]
            protect_vals = [float(x.get("weighted_protect_loss", 0.0)) for x in lr]
            ce = float(np.mean(ce_vals)) if ce_vals else np.nan
            weighted = 50.0 * norm
            active = ce + (float(np.mean(kd_vals)) if kd_vals else 0.0) + (float(np.mean(protect_vals)) if protect_vals else 0.0)
            rows.append({
                "method": METHOD_NAME, "task_step": step, "epoch": epoch,
                "raw_factororth": raw, "normalized_factororth": norm,
                "weighted_normalized_factororth": weighted,
                "ce": ce,
                "weighted_factororth_over_ce": weighted / max(ce, 1e-12) if math.isfinite(ce) else np.nan,
                "weighted_factororth_over_total_non_orthogonal_loss": weighted / max(active, 1e-12) if active > 0 else np.nan,
                "rank": RANK_SCHEDULE[step - 1], "new_rank": RANK_SCHEDULE[step - 1] - (RANK_SCHEDULE[step - 2] if step > 1 else 0),
            })
    path = output_dir / "normalized_factororth_trajectory.csv"
    env["pd"].DataFrame(rows).to_csv(path, index=False)
    return rows


def write_rows(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_pending_report(output_path, source_hash):
    output_path.write_text(f"""# CIFAR-100 job4971615 normalized Combined full experiment

STATUS: PENDING — this source/configuration is intentionally not submitted by the build step.

Canonical source SHA256 before/after execution must equal: `{source_hash}`

| Variant | All-seen | Restricted | Δ vs canonical |
| --- | ---: | ---: | ---: |
| Canonical Combined | 70.76% | 94.78% | 0.00 pp |
| Normalized Combined | pending | pending | pending |
| Normalized Combined + Calibration | pending | pending | pending |

Method: `{METHOD_NAME}`
Protocol: CIFAR-100, 5×20, seed 42, 9 epochs/task, rank schedule `[16,32,48,64,80]`, canonical preprocessing/optimizer, RankExt head LR 1e-4, full KD T=2 weight 1, Protect30, normalized FactorOrth λ=50.

The offline calibration script fills this report after the Slurm run using validation-only fitting and saved logits only.
""", encoding="utf-8")


def main():
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full experiment; refusing CPU fallback")
    device = torch.device("cuda")
    output_dir = OUTPUT_DIR / RUN_NAME
    for name in ("tables", "logs", "models", "checkpoints", "plots", "reports", "configs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    source_hash_before = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    env = load_canonical_definitions()
    env.update({
        "NUM_STEPS": NUM_STEPS, "CLASSES_PER_STEP": CLASSES_PER_TASK,
        "RANKEXT_RANK_SCHEDULE": RANK_SCHEDULE, "RANKEXT_RANK_SCHEDULE_WIDE": RANK_SCHEDULE,
        "USE_RANKEXT_RANK_SCHEDULE_WIDE": False, "RANKEXT_EPOCHS": EPOCHS,
        "BATCH_LORA": BATCH_SIZE, "FAST_RUN": False,
        "ROOT_RESULTS_DIR": str(output_dir), "RUN_NAME_BASE": RUN_NAME,
        "BASE_OUTPUT_DIR": str(output_dir),
        "TABLES_DIR": str(output_dir / "tables"), "LOGS_DIR": str(output_dir / "logs"),
        "MODELS_DIR": str(output_dir / "models"), "CHECKPOINTS_DIR": str(output_dir / "checkpoints"),
        "PLOTS_DIR": str(output_dir / "plots"), "REPORTS_DIR": str(output_dir / "reports"),
        "CONFIGS_DIR": str(output_dir / "configs"),
    })
    class_splits = initialize_full_canonical_data(env)
    assert float(env["LR_RANKEXT"]) == 1e-4, "Canonical RankExt base LR changed"
    assert float(env["family_head_lr_multiplier"]("rank_extension")) == 1.0, "Canonical RankExt head LR multiplier changed"
    assert int(env["NUM_CLASSES"]) == 100 and int(env["NUM_STEPS"]) == 5 and int(env["CLASSES_PER_STEP"]) == 20
    assert int(env["RANKEXT_EPOCHS"]) == 9 and list(env["RANKEXT_RANK_SCHEDULE"]) == RANK_SCHEDULE
    env["METHOD_DISPLAY_NAME_MAP"][METHOD_NAME] = "RankExt + normalized FactorOrth50 + fullKD T2 + Protect30"
    env["ACTIVE_METHOD_MAP"][METHOD_NAME] = dict(env["ACTIVE_METHOD_MAP"][CANONICAL_COMBINED])
    env["ACTIVE_METHOD_MAP"][METHOD_NAME]["method"] = METHOD_NAME
    env["RANKEXT_PROJECTED_PROTECT_METHODS"].add(METHOD_NAME)
    env["RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS"].add(METHOD_NAME)
    add_normalized_factororth(env)
    env["PROBE_CURRENT_METHOD"] = METHOD_NAME
    env["FACTOR_TRACE"].clear()

    write_pending_report(OUTPUT_DIR / "cifar_job4971615_normalized_combined_full_report.md", source_hash_before)
    env["pd"].DataFrame(env["train_val_split_df"]).to_csv(output_dir / "tables" / "validation_split_summary.csv", index=False)
    loss_rows, trajectory_rows = [], []
    previous_rank_state, _, _ = env["train_rank_extension_arm"](
        method_name=METHOD_NAME, replay_per_class=0, use_orth=True, orth_mode="factor_orth",
        lambda_orth=50.0, zero_old_merge=False, use_kd=True, kd_weight=1.0,
        kd_temperature=2.0, kd_class_scope="full", kd_warmup_epochs=0.0,
        orth_train_records=loss_rows, classifier_restoration_records=None,
        cl_trajectory_records=trajectory_rows,
    )
    final_model = env["build_rank_extension_model"](
        previous_rank_state=previous_rank_state, step_idx=NUM_STEPS - 1, old_active_in_forward=True,
    ).to(device)
    val_ds = env["val_source"].with_transform(env["preprocess_val"])
    test_ds = env["dataset"]["test"].with_transform(env["preprocess_val"])
    val_logits, val_labels = predict_logits(env, final_model, val_ds, device)
    test_logits, test_labels = predict_logits(env, final_model, test_ds, device)
    save_logits(output_dir / "validation_logits.npz", val_logits, val_labels, class_splits)
    save_logits(output_dir / "test_logits.npz", test_logits, test_labels, class_splits)
    checkpoint = {
        "method": METHOD_NAME, "canonical_source_sha256": source_hash_before,
        "seed": SEED, "class_splits": class_splits, "rank_schedule": RANK_SCHEDULE,
        "previous_rank_state": previous_rank_state,
        "model_state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
    }
    torch.save(checkpoint, output_dir / "checkpoints" / f"{METHOD_NAME}_final.pt")
    env["pd"].DataFrame(loss_rows).to_csv(output_dir / "tables" / "training_loss_rows.csv", index=False)
    write_rows(output_dir / "tables" / "cl_trajectory.csv", trajectory_rows)
    aggregate_factor_trajectory(env, loss_rows, class_splits, output_dir / "tables")
    source_hash_after = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    if source_hash_before != source_hash_after:
        raise RuntimeError("Canonical source SHA256 changed during the run")
    (output_dir / "run_metadata.json").write_text(__import__("json").dumps({
        "method": METHOD_NAME, "device": str(device), "runtime_seconds": time.time() - started,
        "canonical_source_sha256_before": source_hash_before,
        "canonical_source_sha256_after": source_hash_after,
        "class_splits": class_splits, "seed": SEED, "epochs": EPOCHS,
        "rank_schedule": RANK_SCHEDULE, "head_lr": 1e-4, "kd_temperature": 2.0,
        "kd_weight": 1.0, "protect_weight": 30.0, "factororth_lambda": 50.0,
        "validation_logits": "validation_logits.npz", "test_logits": "test_logits.npz",
    }, indent=2), encoding="utf-8")
    print(f"FULL RUN COMPLETE: {output_dir}")
    print(f"CANONICAL SHA256 BEFORE: {source_hash_before}")
    print(f"CANONICAL SHA256 AFTER: {source_hash_after}")


if __name__ == "__main__":
    main()
