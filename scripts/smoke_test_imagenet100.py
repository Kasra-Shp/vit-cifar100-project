#!/usr/bin/env python3
"""
scripts/smoke_test_imagenet100.py
===================================
Lightweight, CLUSTER-ONLY smoke test for the ImageNet-100 generalization
study (see thesis_agent/reports/imagenet100_generalization_preparation.md).

This script does NOT duplicate SimpleAvg / RankExt / KD / FactorOrth /
calibration / model logic. It imports the canonical implementation directly
from `vit_lora_cifar100_full5step_n5.py` (the SAME source used for every real
training run) and calls its actual functions/classes:
    fresh_pretrained_model, add_lora, family_target_modules,
    build_rank_extension_model, get_rank_extension_rank_triplet,
    get_training_args, IndependentLoraOrthTrainer, DeltaOrthRankExtensionTrainer,
    collate_fn, compute_metrics, class_splits, make_train_dataset,
    build_classwise_train_val_splits (already applied at import time),
    calibrate_classifier_row_norms(_confidence_weighted), LAMBDA_ORTH,
    KD_WEIGHT, KD_TEMPERATURE, CALIBRATION_MODE_BY_FAMILY.

It does NOT:
  - modify any scientific methodology, hyperparameter, or config value.
  - launch training (no multi-epoch loop, no result-directory creation,
    no checkpoints, no experiment/Agent result record).
  - run more than ONE optimizer step per model family (enforced via
    `TrainingArguments.max_steps = 1`, not a hand-rolled training loop).

How the canonical module is imported without running its own training:
`vit_lora_cifar100_full5step_n5.py` defines every dataset/model/Trainer/
calibration function and class, then (much later in the file) actually
TRAINS every active method. This script sets `N5_SKIP_TRAINING_DRIVER=1`
before importing, which makes that module raise `N5StopAfterSetup` at the
exact point (right before its own training-driver loop) where all setup is
complete but no training has started -- this script catches that specific
exception and proceeds with everything already defined. Running
`python vit_lora_cifar100_full5step_n5.py` directly (the existing, unchanged
cluster usage) never sets that variable and is completely unaffected.

Usage (on the cluster, with full ImageNet-1k mounted):
    python scripts/smoke_test_imagenet100.py
    IMAGENET_ROOT=/some/other/path python scripts/smoke_test_imagenet100.py

Exit code 0 if every check passes, 1 otherwise. Safe to re-run any number of
times -- no state is written to disk (all Trainer output goes to a temporary
directory that is removed at the end).
"""

import os
import sys
import shutil
import tempfile
import traceback
from collections import Counter

# ---------------------------------------------------------------------------
# Environment MUST be set before importing the canonical module.
# ---------------------------------------------------------------------------
os.environ.setdefault("N5_DATASET_NAME", "imagenet100")
os.environ.setdefault("N5_EXPERIMENT_LABEL", "smoke_test")
os.environ["N5_SKIP_TRAINING_DRIVER"] = "1"  # always force this, regardless of caller's env
os.environ.setdefault("IMAGENET_ROOT", "/nfsd/lttm4/datasets/ImageNet-1k_torch")
os.environ.setdefault("REPLICATION_SEED", "42")

import importlib.util

RESULTS = {}   # check_name -> (bool, message)
_TMP_DIR = tempfile.mkdtemp(prefix="n5_smoke_test_")


def record(name, ok, msg=""):
    RESULTS[name] = (bool(ok), str(msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def fail_fast(name, msg):
    record(name, False, msg)
    print_summary()
    cleanup_and_exit(1)


def cleanup_and_exit(code):
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    sys.exit(code)


def print_summary():
    print()
    print("=" * 80)
    print("PASS/FAIL MATRIX")
    print("=" * 80)
    n_pass = sum(1 for ok, _ in RESULTS.values() if ok)
    n_total = len(RESULTS)
    width = max((len(k) for k in RESULTS), default=10)
    for name, (ok, msg) in RESULTS.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status:4s}  {name.ljust(width)}  {msg}")
    print("-" * 80)
    print(f"  {n_pass}/{n_total} checks passed")
    print()
    all_pass = (n_pass == n_total) and n_total > 0
    print(f"IMAGE NET-100 PIPELINE RUNTIME-VERIFIED: {'YES' if all_pass else 'NO'}")
    print(f"SAFE TO SUBMIT FULL 8-METHOD RUN: {'YES' if all_pass else 'NO'}")


print("=" * 80)
print("ImageNet-100 lightweight cluster smoke test (NOT a training run)")
print("=" * 80)
print(f"IMAGENET_ROOT       = {os.environ['IMAGENET_ROOT']}")
print(f"N5_DATASET_NAME     = {os.environ['N5_DATASET_NAME']}")
print(f"REPLICATION_SEED    = {os.environ['REPLICATION_SEED']}")
print(f"Temp scratch dir    = {_TMP_DIR} (removed at the end; not a result directory)")
print()

N5_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vit_lora_cifar100_full5step_n5.py",
)

# ---------------------------------------------------------------------------
# Import the canonical module up to (not including) the training driver.
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("n5_canonical", N5_PATH)
n5 = importlib.util.module_from_spec(spec)
sys.modules["n5_canonical"] = n5
try:
    spec.loader.exec_module(n5)
    fail_fast(
        "00_import_setup",
        "module ran to completion WITHOUT stopping at the training-driver guard "
        "-- N5_SKIP_TRAINING_DRIVER did not work as expected. ABORTING (this would "
        "otherwise mean real training just started).",
    )
except Exception as e:
    if type(e).__name__ == "N5StopAfterSetup":
        record("00_import_setup", True, "canonical module imported through full setup, stopped before any training driver ran")
    else:
        print(f"UNEXPECTED EXCEPTION during canonical-module setup (before any check could run): {type(e).__name__}: {e}")
        traceback.print_exc()
        cleanup_and_exit(1)

import torch  # noqa: E402  (after n5 import so we reuse whatever torch n5 already loaded)


# ---------------------------------------------------------------------------
# 1 & 2. Verify the 100 selected train/val WNIDs exist locally.
# ---------------------------------------------------------------------------
try:
    _verify_spec = importlib.util.spec_from_file_location(
        "verify_imagenet100_local",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_imagenet100_local.py"),
    )
    _verify_mod = importlib.util.module_from_spec(_verify_spec)
    _verify_spec.loader.exec_module(_verify_mod)
    _verify_result = _verify_mod.verify(os.environ["IMAGENET_ROOT"])
    n_missing_train = len(_verify_result["missing_train"]) + len(_verify_result["empty_train"])
    n_missing_val = len(_verify_result["missing_val"]) + len(_verify_result["empty_val"])
    record("01_train_wnids_present", n_missing_train == 0,
           f"{100 - n_missing_train}/100 train WNID dirs present+non-empty"
           + ("" if n_missing_train == 0 else f"; missing/empty: {_verify_result['missing_train'] + _verify_result['empty_train']}"))
    record("02_val_wnids_present", n_missing_val == 0,
           f"{100 - n_missing_val}/100 val WNID dirs present+non-empty"
           + ("" if n_missing_val == 0 else f"; missing/empty: {_verify_result['missing_val'] + _verify_result['empty_val']}"))
except Exception as e:
    record("01_train_wnids_present", False, f"exception: {e}")
    record("02_val_wnids_present", False, f"exception: {e}")

# ---------------------------------------------------------------------------
# 3. Verify remapped targets are exactly 0..99.
# ---------------------------------------------------------------------------
try:
    label_counts = Counter(n5.dataset["train"][n5.LABEL_COL])
    distinct_labels = sorted(label_counts.keys())
    ok = distinct_labels == list(range(100))
    record("03_remapped_targets_0_99", ok,
           f"{len(distinct_labels)} distinct labels, range [{min(distinct_labels)}, {max(distinct_labels)}]"
           if distinct_labels else "no labels found")
except Exception as e:
    record("03_remapped_targets_0_99", False, f"exception: {e}")

# ---------------------------------------------------------------------------
# 4. Verify four disjoint 25-class groups.
# ---------------------------------------------------------------------------
try:
    splits = n5.class_splits
    ok = (
        len(splits) == 4
        and all(len(s) == 25 for s in splits)
        and len(set().union(*splits)) == 100
        and sorted(set().union(*splits)) == list(range(100))
    )
    record("04_four_disjoint_25class_groups", ok, f"group sizes={[len(s) for s in splits]}, union size={len(set().union(*splits))}")
except Exception as e:
    record("04_four_disjoint_25class_groups", False, f"exception: {e}")

# ---------------------------------------------------------------------------
# 5. Verify train/internal-validation split (carved out of ImageNet train).
# ---------------------------------------------------------------------------
try:
    n_train = len(n5.train_source)
    n_val = len(n5.val_source)
    expected_val = 100 * n5.VALIDATION_PER_CLASS
    ok = n_val == expected_val and n_train > n_val
    record("05_train_internal_val_split", ok,
           f"train_source={n_train}, val_source={n_val} (expected {expected_val} = 100*{n5.VALIDATION_PER_CLASS})")
except Exception as e:
    record("05_train_internal_val_split", False, f"exception: {e}")

# ---------------------------------------------------------------------------
# 6. Verify ImageNet val (the official split) is final-evaluation only.
# ---------------------------------------------------------------------------
try:
    final_split_name = n5.DATASET_REGISTRY["final_eval_split"]
    ok = final_split_name == "validation" and final_split_name not in ("train",)
    # make_eval_dataset must read from this split, never from train_source/val_source.
    import inspect as _inspect
    src = _inspect.getsource(n5.make_eval_dataset)
    ok = ok and "DATASET_REGISTRY" in src and "final_eval_split" in src
    record("06_imagenet_val_is_final_eval_only", ok,
           f"DATASET_REGISTRY['final_eval_split']={final_split_name!r}; make_eval_dataset reads it, not train_source/val_source")
except Exception as e:
    record("06_imagenet_val_is_final_eval_only", False, f"exception: {e}")

# ---------------------------------------------------------------------------
# 7 & 8. Load ONE real training batch; print shape and labels.
# ---------------------------------------------------------------------------
train_ds_step0 = None
one_batch = None
try:
    train_ds_step0 = n5.make_train_dataset(step_idx=0, replay_per_class=0)
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds_step0, batch_size=8, shuffle=False, collate_fn=n5.collate_fn)
    one_batch = next(iter(dl))
    shape_ok = tuple(one_batch["pixel_values"].shape[1:]) == (3, 224, 224)
    record("07_load_one_training_batch", True, f"batch loaded from step-0 (classes {n5.classes_for_step(0)[:3]}...) train dataset ({len(train_ds_step0)} examples)")
    record("08_batch_shape_and_labels", shape_ok,
           f"pixel_values.shape={tuple(one_batch['pixel_values'].shape)}, labels={one_batch['labels'].tolist()}")
except Exception as e:
    record("07_load_one_training_batch", False, f"exception: {e}")
    record("08_batch_shape_and_labels", False, "skipped (batch load failed)")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 9 & 10. Instantiate CLIP ViT-B/16; verify classifier output dim = 100.
# ---------------------------------------------------------------------------
base_model = None
try:
    base_model = n5.fresh_pretrained_model()
    out_dim = base_model.classifier.out_features
    record("09_instantiate_clip_vit_b16", True, f"MODEL_CHECKPOINT={n5.MODEL_CHECKPOINT!r}")
    record("10_classifier_output_dim_100", out_dim == 100, f"classifier.out_features={out_dim}")
except Exception as e:
    record("09_instantiate_clip_vit_b16", False, f"exception: {e}")
    record("10_classifier_output_dim_100", False, "skipped")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 11. Instantiate canonical SimpleAvg (rank80/alpha160/scaling2).
# ---------------------------------------------------------------------------
simple_model = None
try:
    simple_target_modules = n5.family_target_modules("simple_avg")
    simple_model = n5.add_lora(n5.fresh_pretrained_model(), target_modules=simple_target_modules)
    peft_cfg = simple_model.peft_config["default"]
    r, alpha = int(peft_cfg.r), float(peft_cfg.lora_alpha)
    scaling = alpha / r
    ok = (r == 80 and alpha == 160.0 and abs(scaling - 2.0) < 1e-9 and set(peft_cfg.target_modules) == set(simple_target_modules))
    record("11_instantiate_simpleavg", ok,
           f"r={r}, alpha={alpha}, scaling={scaling}, target_modules={sorted(peft_cfg.target_modules)}")
except Exception as e:
    record("11_instantiate_simpleavg", False, f"exception: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 12 & 17 (part 1). One SimpleAvg forward pass; finite loss/logits.
# ---------------------------------------------------------------------------
simple_forward_ok = False
try:
    simple_model.eval()
    with torch.no_grad():
        out = simple_model(pixel_values=one_batch["pixel_values"], labels=one_batch["labels"])
    finite = torch.isfinite(out.loss).item() and torch.isfinite(out.logits).all().item()
    simple_forward_ok = finite
    record("12_simpleavg_forward_pass", True, f"loss={out.loss.item():.4f}, logits.shape={tuple(out.logits.shape)}")
    record("17a_simpleavg_finite", finite, f"loss finite={torch.isfinite(out.loss).item()}, logits all finite={torch.isfinite(out.logits).all().item()}")
    simple_model.train()
except Exception as e:
    record("12_simpleavg_forward_pass", False, f"exception: {e}")
    record("17a_simpleavg_finite", False, "skipped")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 13. Run ONE SimpleAvg optimizer step via the canonical Trainer + TrainingArguments
#     (max_steps=1 -- NOT a hand-rolled training loop).
# ---------------------------------------------------------------------------
try:
    simple_out_dir = os.path.join(_TMP_DIR, "simple_avg_smoke")
    args = n5.get_training_args(
        output_dir=simple_out_dir,
        epochs=1,
        lr=n5.LR_LORA,
        batch_size=4,          # reduced from canonical BATCH_LORA for smoke-test speed/memory only
        accum_steps=1,
        train_dataset_len=len(train_ds_step0),
        eval_strategy="no",    # no eval needed for a 1-step mechanics check
    )
    args.max_steps = 1  # the ONLY non-canonical override in this script: caps training at exactly one optimizer step
    trainer = n5.IndependentLoraOrthTrainer(
        model=simple_model,
        args=args,
        train_dataset=train_ds_step0,
        eval_dataset=None,
        data_collator=n5.collate_fn,
        compute_metrics=n5.compute_metrics,
        method_name="simple_avg",
        step_idx=0,
    )
    trainer.train()
    ok = trainer.state.global_step == 1
    record("13_simpleavg_one_optimizer_step", ok, f"trainer.state.global_step={trainer.state.global_step}")
except Exception as e:
    record("13_simpleavg_one_optimizer_step", False, f"exception: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 18a. Verify SimpleAvg's frozen backbone got no unintended gradients.
# ---------------------------------------------------------------------------
try:
    bad = []
    for name, p in simple_model.named_parameters():
        if not p.requires_grad and p.grad is not None and torch.any(p.grad != 0):
            bad.append(name)
    record("18a_simpleavg_frozen_backbone_no_grad", len(bad) == 0,
           "no frozen parameter has a nonzero gradient" if not bad else f"UNEXPECTED gradients on: {bad[:5]}")
except Exception as e:
    record("18a_simpleavg_frozen_backbone_no_grad", False, f"exception: {e}")

del simple_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 14. Instantiate canonical RankExt step 1 (rank20, schedule [20,40,60,80], scaling2).
# ---------------------------------------------------------------------------
rankext_model = None
try:
    total_rank, frozen_rank, new_rank = n5.get_rank_extension_rank_triplet(0)
    rankext_model = n5.build_rank_extension_model(previous_rank_state=None, step_idx=0)
    growing_layers = [m for m in rankext_model.modules() if isinstance(m, n5.GrowingRankLoRALinear)]
    scalings = {float(m.scaling) for m in growing_layers}
    ok = (
        total_rank == 20
        and n5.active_rankext_rank_schedule() == [20, 40, 60, 80]
        and len(growing_layers) > 0
        and scalings == {2.0}
    )
    record("14_instantiate_rankext_step1", ok,
           f"step1 total_rank={total_rank} (expected 20), schedule={n5.active_rankext_rank_schedule()}, "
           f"{len(growing_layers)} GrowingRankLoRALinear layers, scaling(s)={scalings}")
except Exception as e:
    record("14_instantiate_rankext_step1", False, f"exception: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 15 & 17 (part 2). One RankExt forward pass; finite loss/logits.
# ---------------------------------------------------------------------------
try:
    rankext_model.eval()
    with torch.no_grad():
        out = rankext_model(pixel_values=one_batch["pixel_values"], labels=one_batch["labels"])
    finite = torch.isfinite(out.loss).item() and torch.isfinite(out.logits).all().item()
    record("15_rankext_forward_pass", True, f"loss={out.loss.item():.4f}, logits.shape={tuple(out.logits.shape)}")
    record("17b_rankext_finite", finite, f"loss finite={torch.isfinite(out.loss).item()}, logits all finite={torch.isfinite(out.logits).all().item()}")
    rankext_model.train()
except Exception as e:
    record("15_rankext_forward_pass", False, f"exception: {e}")
    record("17b_rankext_finite", False, "skipped")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 16. Run ONE RankExt optimizer step via the canonical Trainer (max_steps=1).
# ---------------------------------------------------------------------------
try:
    rankext_out_dir = os.path.join(_TMP_DIR, "rank_extension_smoke")
    args = n5.get_training_args(
        output_dir=rankext_out_dir,
        epochs=1,
        lr=n5.LR_RANKEXT,
        batch_size=4,          # reduced from canonical BATCH_LORA for smoke-test speed/memory only
        accum_steps=1,
        train_dataset_len=len(train_ds_step0),
        eval_strategy="no",
    )
    args.max_steps = 1
    trainer = n5.DeltaOrthRankExtensionTrainer(
        model=rankext_model,
        args=args,
        train_dataset=train_ds_step0,
        eval_dataset=None,
        data_collator=n5.collate_fn,
        compute_metrics=n5.compute_metrics,
        method_name="rank_extension",
        step_idx=0,
        lambda_orth=0.0,      # plain rank_extension: no FactorOrth, no KD (step 1 has neither anyway)
        orth_mode="none",     # explicitly the no-op value the canonical compute_loss() checks for
        kd_weight=0.0,
    )
    trainer.train()
    ok = trainer.state.global_step == 1
    record("16_rankext_one_optimizer_step", ok, f"trainer.state.global_step={trainer.state.global_step}")
except Exception as e:
    record("16_rankext_one_optimizer_step", False, f"exception: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 18b. Verify RankExt's frozen backbone got no unintended gradients.
# ---------------------------------------------------------------------------
try:
    bad = []
    for name, p in rankext_model.named_parameters():
        if not p.requires_grad and p.grad is not None and torch.any(p.grad != 0):
            bad.append(name)
    record("18b_rankext_frozen_backbone_no_grad", len(bad) == 0,
           "no frozen parameter has a nonzero gradient" if not bad else f"UNEXPECTED gradients on: {bad[:5]}")
except Exception as e:
    record("18b_rankext_frozen_backbone_no_grad", False, f"exception: {e}")

del rankext_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 19. Statically verify KD, FactorOrth and calibration paths are compatible.
#     (No real KD/FactorOrth training run here -- existence, callability, and
#     canonical-config-value checks only.)
# ---------------------------------------------------------------------------
try:
    checks = {
        "build_simple_avg_teacher_model (KD, simple_avg family)": callable(getattr(n5, "build_simple_avg_teacher_model", None)),
        "compute_independent_lora_factor_orth_components (FactorOrth, simple_avg family)": callable(getattr(n5, "compute_independent_lora_factor_orth_components", None)),
        "compute_independent_lora_orth_components (DeltaOrth, simple_avg family)": callable(getattr(n5, "compute_independent_lora_orth_components", None)),
        "calibrate_classifier_row_norms": callable(getattr(n5, "calibrate_classifier_row_norms", None)),
        "calibrate_classifier_row_norms_confidence_weighted": callable(getattr(n5, "calibrate_classifier_row_norms_confidence_weighted", None)),
        "KD_WEIGHT == 1.0": float(n5.KD_WEIGHT) == 1.0,
        "KD_TEMPERATURE == 2.0": float(n5.KD_TEMPERATURE) == 2.0,
        "LAMBDA_ORTH == 50.0": float(n5.LAMBDA_ORTH) == 50.0,
        "CALIBRATION_MODE_BY_FAMILY has both families": set(["simple_avg", "rank_extension"]).issubset(n5.CALIBRATION_MODE_BY_FAMILY.keys()),
        "confidence_weighted_regime_grouped is a configured mode": "confidence_weighted_regime_grouped" in n5.CALIBRATION_MODE_BY_FAMILY.values(),
    }
    failed = [k for k, v in checks.items() if not v]
    record("19_kd_factororth_calibration_static", len(failed) == 0,
           "all present/callable/canonical-valued" if not failed else f"FAILED: {failed}")
except Exception as e:
    record("19_kd_factororth_calibration_static", False, f"exception: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 20. Print CUDA GPU name and peak allocated memory.
# ---------------------------------------------------------------------------
try:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        record("20_cuda_gpu_info", True, f"GPU={gpu_name}, peak_allocated={peak_mb:.1f} MB")
    else:
        record("20_cuda_gpu_info", False, "CUDA not available in this process (expected on a GPU cluster node -- if this fires on the real cluster job, investigate CUDA visibility)")
except Exception as e:
    record("20_cuda_gpu_info", False, f"exception: {e}")

print_summary()
cleanup_and_exit(0 if all(ok for ok, _ in RESULTS.values()) else 1)
