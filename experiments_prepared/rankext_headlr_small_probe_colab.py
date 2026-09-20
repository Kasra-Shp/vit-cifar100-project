#!/usr/bin/env python
# coding: utf-8

#
# ==============================================================================
# FINAL 9-METHOD CIFAR-100 5x20 PERFORMANCE-RECOVERY EXPERIMENT.
#
# Derived by copying experiments_prepared/final_8arm_family_specific_5x20.py
# (job 4970580's script, itself derived from job 4969059's script and the
# VERIFIED R7/Exp1 harness, supervisor_exp1_cifar100_5x20_fixed_rankext.py)
# and applying a precisely-documented set of changes, listed below. All three
# source scripts, and R7/vit_lora_cifar100_full5step_n5.py, are left
# completely UNTOUCHED at their own paths -- this is a new, separate,
# standalone file. Full rationale: R7/final_5x20_performance_recovery_plan.md.
#
# WHY THIS SCRIPT EXISTS: the prior two production runs under this pipeline
# (job 4969059, job 4970580) established that old-seen-only KD, 6 epochs, and
# no protect_weight/pretrained-anchor term leave RankExt's best arm at 53.13%
# all_seen, far below the historical Exp1/R7 FactorOrth+full-KD ceiling of
# 70.72% (independently reproduced bit-for-bit on the modern codebase by job
# 4961962, via supervisor_exp1_cifar100_5x20_fixed_rankext.py). The forensic
# gap analysis (R7/final_recovery_and_improvement_design_audit.md) identified
# full-100-way KD, the `protect_weight` old-class semantic-subspace
# protection term, KD-conditional new-block-warmup suppression, and 9 (not 6)
# epochs as the highest-probability remaining contributors. This script tests
# that combination directly, for the first time under the current, corrected
# (non-buggy classifier-restoration) pipeline.
#
# IMPORTANT DISCOVERY THAT SHAPED THIS SCRIPT'S CONSTRUCTION: the RankExt
# trainer class this file inherits (`DeltaOrthRankExtensionTrainer`, copied
# unchanged from final_8arm_family_specific_5x20.py) ALREADY CONTAINS the
# complete historical `protect_weight`/`pretrained_anchor_weight`/
# `compute_old_semantic_subspace`/full-KD (`kd_class_scope="full"`) mechanism,
# carried over verbatim from R7/vit_lora_cifar100_full5step_n5.py by an
# earlier port and simply never activated for job 4970580's 8 arms (its new
# method names were deliberately chosen NOT to match
# `RANKEXT_PROJECTED_PROTECT_METHODS`/`RANKEXT_NEW_BLOCK_WARMUP_DISABLED_
# METHODS` -- see that script's own implementation audit, Section 2). This
# script therefore does NOT reimplement RankExt's historical mechanisms from
# scratch: it CONFIGURES the two new RankExt KD arms below (`kd_class_scope=
# "full"`, membership in the two sets above) to activate code that is a
# byte-for-byte match of R7/vit_lora_cifar100_full5step_n5.py's own
# `DeltaOrthRankExtensionTrainer.compute_loss()`/`compute_old_semantic_
# subspace()` (both directly compared during preparation of this script;
# see the design plan's Section 6/13/19). This is the intended reading of
# "use R7/vit_lora_cifar100_full5step_n5.py as the behavioral base": the
# MECHANISM is that file's, ported (previously) verbatim; only the FIXED
# classifier restoration and the current pipeline's superior logging/
# checkpoint-safety infrastructure differ from that file, both deliberately.
#
# DELIBERATE CHANGES vs. final_8arm_family_specific_5x20.py:
#   1. Both families now train for 9 epochs/CL step (was 6), matching the
#      historical Exp1/R7 protocol exactly -- see PROTOCOL LOCK below.
#   2. SimpleAvg's KD arms use weight=1.0 with 1-epoch warmup (the current
#      pipeline's ORIGINAL, job-4969059 default, known best ~76.66%), not
#      job 4970580's weight=0.5/no-warmup variant.
#   3. A NEW SimpleAvg arm is added: old-seen KD at temperature T=4 (all
#      other KD hyperparameters identical to the T=2 arm), for a clean T2-
#      vs-T4 ablation -- never tested before under any prior run.
#   4. Two RankExt arms are switched from old-seen KD to the HISTORICAL
#      full-100-way KD (`kd_class_scope="full"`), weight=1.0, NO KD-weight
#      warmup (matching Exp1's own flat-weight convention -- old-seen KD's
#      warmup mechanism is simply not applicable to a mechanism Exp1 never
#      had).
#   5. Those same two RankExt KD arms now activate `protect_weight=30.0`
#      (the OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION
#      mechanism) and have new-block output warmup DISABLED -- both exactly
#      matching Exp1's own treatment of its two historical KD-bearing RankExt
#      methods (`rank_extension_kd_only_T2`, `rank_extension_orth_factor_
#      lam_50_kd_T2`), reproduced here under NEW method names per this task's
#      exact-name specification.
#   6. Reporting-units bug fix: the prior script's summary table stored
#      `all_seen`/`first_step`/`later_steps_mean`/`old_new_gap` as fractions
#      (0-1) while `restricted_mean` was a percentage (0-100) in the SAME
#      row -- discovered during the job-4970580 forensic analysis. This
#      script stores every human-facing accuracy metric as a 0-100
#      percentage, consistently, with a runtime sanity assertion before the
#      summary table is ever written to disk.
#   7. Exactly 9 methods are active (5 SimpleAvg + 4 RankExt, Section 6 of
#      the task brief) -- enforced by a hard `EXPECTED_METHODS` set assertion
#      at module load, not just by METHODS_TO_RUN flag inspection.
#   8. Historical/prior-run reference values are stored in a table explicitly
#      tagged `status="REFERENCE_ONLY_NOT_RETRAINED"`, never silently mixed
#      into the newly-trained rows.
#   9. New, unique run/checkpoint root: `cifar100_5x20_final_9method_
#      performance_recovery_seed{SEED}` -- never shares a checkpoint
#      directory with job 4969059/4970580's runs. The config fingerprint
#      (`compute_method_config_fingerprint`) is extended to also cover
#      `protect_weight`, `rankext_new_block_warmup_enabled`, `merge_method`,
#      `classifier_restoration_mode`, and `lora_alpha`.
#
# EVERYTHING NOT LISTED ABOVE is preserved BYTE-IDENTICAL in mechanism to
# final_8arm_family_specific_5x20.py: dataset, class split/ordering, seed=42,
# optimizer, LR, weight decay, scheduler, LR warmup, gradient clipping, mixed
# precision, validation split, best-epoch selection (pure validation-CE
# argmin), classifier training/stitching/calibration, the FIXED (non-buggy)
# classifier-row restoration, open/restricted evaluation, FactorOrth's own
# formulation/lambda=50/warmup, DenseOrth's own formulation/lambda=20,
# per-method seed reset, RankExt's persistent-growing-model architecture, and
# arithmetic-only merging for SimpleAvg (no DO-Merge, no replay, anywhere).
#
# FINAL 9 METHODS (see EXPECTED_METHODS below for the enforced assertion):
#   SA-1: simple_avg                                            -- plain
#   SA-2: simple_avg_kd_oldseen_T2_warmup                       -- old-seen KD, T=2, weight=1.0, 1-epoch warmup
#   SA-3: simple_avg_kd_oldseen_T4_warmup                       -- SA-2 with T=4 (only difference)
#   SA-4: simple_avg_dense_orth_lam20                           -- DenseOrth, lambda=20
#   SA-5: simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup      -- SA-2 + SA-4 combined
#   RE-1: rank_extension                                        -- plain
#   RE-2: rank_extension_fullkd_T2_protect30                    -- HISTORICAL full-100-way KD, T=2, weight=1.0, no warmup, protect_weight=30, new-block warmup disabled
#   RE-3: rank_extension_factor_orth_lam50                      -- HISTORICAL FactorOrth, lambda=50
#   RE-4: rank_extension_factor_orth_lam50_fullkd_T2_protect30  -- RE-2 + RE-3 combined (the main final RankExt candidate)
#
# MERGE OPERATORS: SimpleAvg arithmetic-only. RankExt uses
# merge_method="persistent". TOTAL: 5 + 4 = 9 evaluated configurations.
#
# THIS FILE MUST NOT BE EXECUTED AS PART OF PREPARING IT. Per the task brief:
# no smoke test, no training, no evaluation, no calibration run, no SLURM
# submission. Only static validation (py_compile, AST/source inspection) was
# performed while building it -- see R7/final_9method_5x20_static_readiness.md.
# ==============================================================================
#

# In[ ]:


import os
import gc
import json
import random
import math
import inspect
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, concatenate_datasets
from torchvision import transforms

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    TrainingArguments,
    Trainer,
    set_seed,
)

from transformers.modeling_outputs import ImageClassifierOutput

from peft import LoraConfig, get_peft_model

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

try:
    # Used only for the smooth (PCHIP) within-step interpolation in the live
    # convergence plots (Task 2 / Task 3). Falls back to plain polylines if
    # scipy isn't available on the cluster node so plotting never blocks training.
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# =============================================================================
# COLAB ADAPTATION BLOCK (2026-09-20) -- everything in this block is NEW vs.
# experiments_prepared/rankext_headlr_small_probe.py (the cluster probe this
# file is copied from, byte-for-byte, apart from this block + the
# COLAB_BATCH_SIZE line further below). Nothing about the RankExt algorithm,
# KD/protect30/FactorOrth mechanics, row-norm calibration, seed, class order,
# task/epoch reduction, or LR-sweep intent is touched here -- see
# R7/rankext_headlr_small_probe_colab_readiness.md for the full diff audit.
# =============================================================================

# --- Colab-safe HuggingFace dataset cache location (spec Section 7) --------
# The dataset-loading call below (now "uoft-cs/cifar100", see its own
# COLAB FIX comment) reads these env vars at call time; setting them here,
# before that call, redirects the download/cache from the default
# `~/.cache/huggingface` to a Colab-conventional location under /content.
# Harmless no-op on the cluster (never set there) or if the user pre-sets
# these vars.
os.environ.setdefault("HF_HOME", "/content/data/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/content/data/hf_datasets_cache")

# --- GPU-REQUIRED HARD FAIL (spec Section 5) --------------------------------
# This probe must never silently fall back to CPU -- a CPU run would produce
# misleading (extremely slow, and potentially numerically-inconsistent-enough
# to confound the LR comparison via different effective batch timing) results
# without any obvious signal. Fail loudly instead, before any dataset
# download or model construction happens.
print("=" * 80)
print("COLAB GPU PREFLIGHT CHECK")
print("=" * 80)
print(f"torch version: {torch.__version__}")
print(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "FATAL: torch.cuda.is_available() is False -- no GPU detected. "
        "This probe REFUSES to run on CPU (Colab Runtime -> Change runtime "
        "type -> GPU, then re-run). Silently training on CPU is not "
        "supported -- see spec Section 5 / R7/rankext_headlr_small_probe_"
        "colab_readiness.md."
    )
_gpu_name = torch.cuda.get_device_name(0)
_gpu_total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU name: {_gpu_name}")
print(f"GPU total VRAM: {_gpu_total_vram_gb:.2f} GiB")
print("=" * 80)

# --- COLAB_BATCH_SIZE env-var override (spec Section 6) --------------------
# Wired to BATCH_LORA (the only batch-size constant RankExt's training path
# actually consumes -- BATCH_FT is a full-fine-tune constant, unused by any
# RankExt/FactorOrth/protect30/KD code path in this file). No other
# hyperparameter (rank, LoRA config, KD, protect30, FactorOrth, task count,
# epoch count, LR sweep) is touched by this override -- see the BATCH_LORA
# assignment itself, a few lines below, for the actual application point and
# the documented conservative-default reasoning.
COLAB_BATCH_SIZE = os.environ.get("COLAB_BATCH_SIZE", None)


# In[ ]:


# STRICT-REVIEW (B4, seed readiness): SEED is the SINGLE constant controlling
# every source of randomness in this script -- set_seed()/random.seed()/
# np.random.seed()/torch.manual_seed() below all derive from it, as does every
# per-class/per-step dataset shuffle elsewhere (each offsets SEED by a class
# or step index, e.g. `seed=SEED + int(cls)`, so they stay reproducible and
# distinct from each other without introducing a second free-floating seed
# constant anywhere). A pending multi-seed sweep (supervisor decision) needs
# only to change THIS line -- nothing else in the script hardcodes a seed.
# SEED is also stamped onto every per-method config table/JSON dump (CFG's
# "seed" column, propagated into method_hyperparameter_summary.csv,
# hyperparameter_consistency_check.csv, training_loss_history_by_epoch.csv,
# supervisor_selected_accuracy_comparison.csv, final_metrics_all_methods.csv)
# and into configs/run_config.json, so which seed produced a given saved
# table is always recoverable from that table alone.
#
# CANONICAL 3-SEED REPLICATION (2026-09-06): the "pending multi-seed sweep"
# noted above is now this change -- SEED is externally settable via the
# REPLICATION_SEED environment variable, defaulting to 42 (the canonical
# R6-15 seed) when unset. Seed 123 was run as an independent-seed replicate
# (R6-17, RESTORED 2026-09-08 after that replication completed and was
# analyzed -- see thesis_agent/reports/r6_seed42_seed123_replication_analysis.md;
# the default below briefly read "123" during that run's preparation and has
# now been reverted to the canonical default). Nothing else about SEED's
# role changed (no new RNG call, no new determinism setting; every
# per-class/per-step shuffle still derives from this one constant). `os` is
# already imported above.
SEED = int(os.environ.get("REPLICATION_SEED", "42"))
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# RANKEXT HEAD-LR SMALL PROBE (2026-09-20) -- isolated diagnostic copy of the
# canonical experiments_prepared/final_9method_5x20_performance_recovery.py
# (the file that produced job4971615). Does NOT import from or modify that
# file or experiments_prepared/final_8method_5x20_refined.py.
#
# Purpose: a SMALL, CONTROLLED sweep of RankExt's head/classifier LR
# multiplier, restored to the job4971615 (pre-task-scale-calibration)
# RankExt pipeline -- NOT a benchmark, NOT a redesign. Only 3 things differ
# from the canonical script below: (1) which methods actually run (SimpleAvg
# entirely off; only the two RankExt arms under test), (2) NUM_STEPS/
# RANKEXT_EPOCHS reduced for a cheap smoke test, (3) rank_extension's own
# head_lr_multiplier is read from an env var so one job can sweep several
# values. Every other constant (rank schedule, LoRA targets, KD mechanism,
# protect_weight, FactorOrth lambda, classifier restoration, optimizer,
# scheduler, base LR, row-norm calibration) is byte-identical to the
# canonical file -- see R7/rankext_headlr_small_probe_readiness.md for the
# full diff-based audit proving this.
#
# This canonical file contains NO task-scale-calibration code whatsoever
# (PerTaskScaleCalibration/RankExtCalibratedModel/
# apply_rankext_task_scale_calibration were added only in
# final_8method_5x20_refined.py, never here) -- so copying from here is
# safe-by-construction for "new calibration must be inactive". The explicit
# runtime guard below is a belt-and-suspenders check, not a real risk.
RANKEXT_TASK_SCALE_CALIBRATION_ENABLED = False
for _forbidden_calib_name in (
    "PerTaskScaleCalibration", "RankExtCalibratedModel",
    "apply_rankext_task_scale_calibration", "fit_rankext_task_scale_calibration",
    "collect_rankext_logits_labels",
):
    assert _forbidden_calib_name not in dir(), (
        f"PROBE INTEGRITY FAILURE: new task-scale calibration symbol "
        f"{_forbidden_calib_name!r} must not exist in this probe script"
    )
assert RANKEXT_TASK_SCALE_CALIBRATION_ENABLED is False, (
    "PROBE INTEGRITY FAILURE: new task-scale calibration must stay disabled"
)

# PROBE SMOKE-SCALE REDUCTION (documented in full in the readiness report):
# 5 steps -> 3 steps (Task1, Task1->2 transition, Task2->3 transition), same
# seed/class order/CLASSES_PER_STEP=20 (i.e. the SAME first 60-of-100 classes
# in the same order the full run would use -- just stopping after step 3
# instead of continuing to 5). 9 epochs/step -> 2 epochs/step (still enough
# epochs to see within-task convergence trend and a transition, at a fraction
# of the cost). NOT thesis-scale, NOT used to draw final accuracy claims.
PROBE_NUM_STEPS = 3
PROBE_RANKEXT_EPOCHS = 2

# Head-LR multiplier under test, read from an env var so a single sbatch job
# can loop over the small candidate set {0.5, 1.0, 2.0} (optionally 5.0; 10.0
# only ever as an explicitly-labeled historical-stress reference, never a
# preferred candidate) by invoking this script once per value -- mirrors the
# existing REPLICATION_SEED env-var convention just above. Historical
# baseline (job4971615) is 1.0.
RANKEXT_HEADLR_PROBE_MULTIPLIER = float(os.environ.get("RANKEXT_HEADLR_PROBE_MULTIPLIER", "1.0"))

# JOINT PACK ADDITION -- GPU SAME-JOB PREFLIGHT: FAST_RUN is now genuinely
# functional (was a dead flag, naming-only, in Exp1/the earlier packs),
# gated behind an environment variable that defaults OFF (real config,
# unchanged) so this same script serves BOTH phases of the single-job
# preflight-then-real-run sbatch design: `FAST_RUN_DEBUG=1 python this.py`
# (Phase 1, tiny/fast, isolated output) then a normal `python this.py`
# (Phase 2, real config, real output) -- see the sbatch launcher. NUM_STEPS/
# CLASSES_PER_STEP/rank/schedule/epochs-count-as-hyperparameter and every
# other scientific constant are NEVER touched by FAST_RUN -- only per-class
# image counts, epoch COUNT (not epoch logic), and the output root directory
# are reduced/redirected when FAST_RUN is True. See each such constant's own
# "if FAST_RUN" branch below for the exact, minimal set of reductions.
FAST_RUN = os.environ.get("FAST_RUN_DEBUG", "0") == "1"
DEBUG_MODE = FAST_RUN

# PROTOCOL-DEPTH VALIDATION (R6 roadmap Stage 3, 2026-08-24): CIFAR-100
# 5x20 -> 20x5, controlled protocol-only change to test the age-dependent-
# drift hypothesis directly (does deepening cumulative block count amplify
# forgetting at matched classes-seen/cumulative-rank checkpoints?). Result
# (job 20260824_020825): STRONGLY SUPPORTED -- see the 20x5 analysis report.
#
# PROTOCOL-DEPTH VALIDATION, INVERSE-DEPTH LEG (R6 roadmap Stage 4,
# 2026-08-24): 20x5 -> 4x25. Same controlled protocol-only manipulation, now
# testing the INVERSE direction: if repeated incremental update/freeze
# events are a major forgetting driver, making the protocol SHALLOWER than
# the settled 5x20 baseline should improve retention. This is the second,
# opposite-direction leg of the same single-variable (incremental depth)
# manipulation: 20 steps -> 5 steps -> 4 steps, final classes/rank/backbone/
# class order/learning mechanisms all held fixed (see NUM_STEPS/
# CLASSES_PER_STEP and RANKEXT_RANK_SCHEDULE below for the companion
# changes that keep total classes=100 and final cumulative rank=80).
#
# PRE-REGISTERED FALSIFIABLE PREDICTION (written before this run; do not
# revise after seeing results): if repeated incremental update/freeze
# events are a major contributor to forgetting, then relative to 5x20,
# 4x25 should show (a) less forgetting / higher BWT, (b) earlier groups
# (G1-G3) retaining higher final open accuracy than their 5x20 counterparts,
# (c) smaller restricted-vs-open gaps, (d) weaker age-dependent degradation
# (flatter age slope / weaker age-alignment correlation), (e) final all_seen
# improved or at minimum competitive with 5x20. Fresh/current-step learning
# may become slightly harder since each step now covers 25 classes instead
# of 20 -- that is an expected, not disqualifying, side effect. A result
# similar to 5x20 indicates saturation around 4-5 steps; a result WORSE than
# 5x20 indicates a non-monotonic depth/task-size trade-off where an
# individual task becoming too large begins to dominate. A failure of 4x25
# to improve over 5x20 would weaken the simple monotonic "fewer incremental
# events = better retention" interpretation, not confirm the age-drift
# hypothesis' absence outright (see the inverse-depth report's alternative-
# explanation section for how the two are told apart).
#
# Explicit "4x25" marker in the run name -- the .py filename's own "n5"
# segment is unrelated legacy notebook-version naming, NOT a classes-per-
# step marker, and is deliberately left untouched (renaming it would touch
# unrelated infrastructure this task did not ask for).
#
# FINAL KD-WEIGHT EXPERIMENT (R6 roadmap, CLOSED 2026-08-25): single-lever
# test of KD_WEIGHT 1.0 -> 0.75 on the 4x25 flagship only, job
# ..._4x25_kdw075_flagship_with_orth_rankext_EPOCH3_MAIN_20260824_195751.
# Result: all_seen 73.48 vs the KD_WEIGHT=1.0 baseline's 74.07 (-0.59pp),
# forgetting_metric 0.2015 vs 0.1712 (worse), BWT -0.0855 vs -0.0529 (worse),
# with a -13.20pp G1 (oldest group) retention collapse only partially offset
# by +8.24pp G2 / +2.76pp G4 gains -- a net regression, not an improvement,
# and the restricted-vs-open gap (the flagship's dominant bottleneck) was
# essentially unchanged in aggregate (~93.3-93.4% restricted either way), so
# KD weight is confirmed NOT to be the active lever on it. KD_WEIGHT is
# reverted to 1.0 below (the settled value) and this branch is CLOSED -- no
# further KD-weight sweep. See the KDw0.75 run directory itself (retained,
# not deleted) for the full analysis.
#
# FINAL THESIS COMPARISON (R6 roadmap, closing run): all 8 principal
# supervisor-selected methods (SimpleAvg x4, RankExt x4 -- see
# SUPERVISOR_SELECTED_METHOD_SPECS above) reactivated together via
# METHODS_TO_RUN below, at the settled KD_WEIGHT=1.0 / LAMBDA_ORTH=50.0 /
# RANKEXT_RANK_SCHEDULE=[20,40,60,80] 4x25 configuration, for the final
# canonical thesis comparison table/plots. No scientific setting is changed
# for this run beyond reactivating the 4 simple_avg methods (already-existing
# code path, previously deactivated only to reduce runtime -- see
# METHODS_TO_RUN's own comments).
# WIDE RANKEXT CAPACITY-SENSITIVITY EXPERIMENT (2026-09-02, CLOSED): produced
# R6-16 (job 4933319), analyzed in
# thesis_agent/reports/r6_wide_rank_capacity_analysis.md. That control run is
# complete and untouched; USE_RANKEXT_RANK_SCHEDULE_WIDE is reverted to False
# below (see that flag's own comment) so this string no longer needs to name
# a wide-schedule run.
#
# RUN NAMING: RUN_NAME_BASE encodes SEED so this run writes to its own
# distinct output directory and can never overwrite R6-15's own directory
# (`clip_vit_lora_cifar100_4x25_final_8methods_thesis_comparison_EPOCH3_MAIN_
# 20260825_203830`, a fixed historical path, unaffected by any change here),
# R6-16's, or a future seed's.
#
# EXPERIMENT 1 (supervisor-requested, prepared 2026-09-10, finalized
# 2026-09-11, NOT LAUNCHED -- see
# thesis_agent/reports/supervisor_cifar_followup_audit.md): "rerun the 5x20
# scenario with fixed rank extension, same rank as before the fix."
#
# EXACT DESCRIPTION (per the 2026-09-11 audit addendum -- use this phrasing,
# not "reproduces job 4904629"): a 5x20 RE-EVALUATION UNDER THE CURRENT FAIR/
# CANONICAL PIPELINE, using the corrected RankExt implementation and the
# historical RankExt cumulative schedule [16,32,48,64,80]. This is
# deliberately NOT a byte-for-byte reproduction of job 4904629
# (R6/results_4904629_light) -- that job's own SimpleAvg config (4 target
# modules, "global" calibration) is an obsolete, since-fixed asymmetry (see
# the PROVENANCE NOTE below) and is intentionally NOT restored here.
#
# This file is a dedicated copy of the restored canonical
# vit_lora_cifar100_full5step_n5.py, with ONLY the protocol/rank/epoch
# constants below changed: protocol=5x20, RankExt schedule
# [16,32,48,64,80] (recovered from thesis_agent/sources/R6_catalog.md's 5x20
# job-ID entries and independently corroborated by old/vit_lora_cifar100_
# full5step_n5.py's embedded protocol-depth-validation comments -- NOT
# guessed), LORA_R=80 (unchanged), epochs=9 (matching every historical 5x20
# job). Every SimpleAvg/RankExt/KD/FactorOrth/calibration/classifier/
# open-restricted-evaluation/BWT/forgetting/best-epoch-selection ALGORITHM is
# reused UNCHANGED from current n5.py -- current n5.py is the sole authority
# for all method definitions; nothing here reimplements or copies any
# pre-fix RankExt code path.
#
# WHY A FULL COPY, NOT A THIN IMPORT WRAPPER: vit_lora_cifar100_full5step_n5.py
# executes top-to-bottom as a converted notebook (not designed as an
# importable module), and its one prior import-without-training mechanism
# (N5StopAfterSetup / N5_SKIP_TRAINING_DRIVER) was deliberately removed
# during the CIFAR-100 canonical restoration specifically because it was
# ImageNet-smoke-test-only scaffolding with no other user -- see
# thesis_agent/reports/supervisor_cifar_followup_audit.md. Re-adding an
# import-time short-circuit purely to avoid this copy would reintroduce that
# exact removed mechanism for a single one-off experiment. A dedicated copy
# with only the constants below overridden is the lower-risk choice: current
# n5.py itself is never modified, and every non-overridden line here is
# byte-identical to it (diff-able at any time to prove no algorithm drifted).
# JOINT PACK ADDITION (GPU preflight): an unmistakable, unambiguous suffix
# when FAST_RUN_DEBUG=1 -- combined with ROOT_RESULTS_DIR's own separate
# "results_debug_preflight" root below (not just a differently-named
# subdirectory under the same "results" root), this makes it structurally
# impossible for a debug run's output/checkpoints to be discovered, loaded,
# or mistaken for real scientific output by a real (FAST_RUN=False) run --
# two independent isolation mechanisms, not one.
# BUGFIX (found via local sanity check): an earlier version of this line
# appended a debug-only suffix here too (belt-and-suspenders on top of
# ROOT_RESULTS_DIR's own separate root tree below). Combined with this
# already-long base name, the RankExt arms' longer per-step diagnostic
# filenames (e.g. "rank_extension_dense_orth_lam20_kd_oldseen_T2_warmup_
# step_5_trainable_parameters.csv") pushed the full path past Windows'
# 260-char MAX_PATH, crashing os.makedirs/open with a FileNotFoundError that
# looks like a missing-directory bug but is purely a path-length issue.
# Removed -- ROOT_RESULTS_DIR="results_debug_preflight" (a completely
# separate top-level tree, never a subdirectory of "results") is already a
# complete, unambiguous, structural isolation guarantee on its own; a second
# marker in the run name added path length without adding real safety.
# RENAMED for the final 8-arm family-specific script -- deliberately isolated
# from job 4969059's RUN_NAME_BASE (and from every other prior run's) so this
# script can never write into, or be confused with, any prior run's results/
# checkpoints directory. Kept short for the same Windows MAX_PATH reason
# documented in the comment above (RankExt per-step diagnostic filenames are
# long).
# PROBE: isolated from the canonical script's own RUN_NAME_BASE (never
# collides with job4971615's or job4972616's results/checkpoints), AND
# includes the swept head-LR multiplier so each LR value invoked in the same
# sbatch job (see the sweep loop in rankext_headlr_small_probe.sbatch) writes
# to its own distinct output directory rather than overwriting a shared one.
_PROBE_LR_TAG = "lr" + str(RANKEXT_HEADLR_PROBE_MULTIPLIER).replace(".", "p")
RUN_NAME_BASE = f"rankext_headlr_probe_{PROBE_NUM_STEPS}x20_{PROBE_RANKEXT_EPOCHS}ep_{_PROBE_LR_TAG}_seed{SEED}"
RUN_NAME = f"{RUN_NAME_BASE}_{'FAST_RUN_DEBUG' if FAST_RUN else 'EPOCH3_MAIN'}"

# PROVENANCE NOTE (2026-09-11, cross-checked against old/vit_lora_cifar100_
# full5step_n5.py and R6/results_4904629_light -- see
# thesis_agent/reports/supervisor_cifar_followup_audit.md "Old notebook
# cross-check" addendum): the pre-fix 5x20 reference job (4904629, and its
# duplicate results_fix2_20260721_light) ran simple_avg with 4 target modules
# (q_proj/k_proj/v_proj/out_proj) and "global" calibration -- NOT the 2-module
# (q_proj/v_proj)/"confidence_weighted_regime_grouped" config TARGET_MODULES_
# BY_FAMILY and CALIBRATION_MODE_BY_FAMILY below actually have (inherited
# unchanged from the restored canonical file). This is a REAL, confirmed
# difference from the pre-fix reference, deliberately NOT reproduced here:
# both the 4->2 module change and the calibration-consistency change are
# part of the CURRENT CORRECTED METHODOLOGY (a cross-family fairness fix and
# the fix for the R6-14-style calibration asymmetry, respectively), not part
# of "the historical rank/protocol to preserve" -- reproducing job 4904629's
# SimpleAvg config exactly would reintroduce the R6-14 calibration asymmetry
# this project has already fixed. Only protocol (5x20), RankExt's own rank
# schedule, and epoch count are matched to the pre-fix reference; SimpleAvg's
# target-module count and both families' calibration algorithm intentionally
# stay at their CURRENT (fixed) values. See the provenance table in the audit
# report for the full reconciliation across all three sources (old notebook,
# job 4904629 artifacts, current n5.py).
MODEL_CHECKPOINT = "openai/clip-vit-base-patch16"

NUM_CLASSES = 100
assert NUM_CLASSES == 100, f"This script is CIFAR-100 ONLY; got NUM_CLASSES={NUM_CLASSES}"
# PROTOCOL-DEPTH VALIDATION: 5x20 -> 20x5 -> 4x25. Total classes (100) and
# class ORDER are unchanged (see class_splits below, derived generically
# from these two constants, not a hardcoded literal) -- classes are simply
# chunked contiguously off native CIFAR-100 label order 0..99, never
# shuffled, so this generalizes by construction: under 4x25, step1 = classes
# 0-24 (bit-identical to 5x20 step1's class SET plus 5 more classes from
# 5x20 step2), step2 = 25-49, step3 = 50-74, step4 = 75-99 (verified
# programmatically in the pre-flight synthetic test). Unlike 20x5, 4x25's
# per-step class blocks do NOT align to the historical 20-class step
# boundaries at every step (25 does not evenly divide 20) -- only the FINAL
# checkpoint (100 classes) is an exact class-count/rank match to a 5x20
# checkpoint; see protocol_depth_macro_checkpoint_comparison.csv's
# corresponding_5x20_step column (computed from an explicit classes-seen +
# cumulative-rank match, not a step-index coincidence) for how this is
# reported without overclaiming intermediate equivalence. No other
# scientific setting changes as a result of this edit alone -- see
# RANKEXT_RANK_SCHEDULE below for the companion change needed to keep final
# cumulative rank at 80 (not 320).
# EXPERIMENT 1 OVERRIDE: 4x25 canonical -> 5x20 (the pre-fix reference
# protocol -- see thesis_agent/reports/supervisor_cifar_followup_audit.md).
# PROBE: 5 -> 3 steps (see PROBE_NUM_STEPS comment above). Same seed/class
# order/CLASSES_PER_STEP -- this literally reuses the first 3 of the
# canonical run's 5 steps, unchanged, just doesn't continue past step 3.
NUM_STEPS = PROBE_NUM_STEPS
CLASSES_PER_STEP = 20

# PROTOCOL LOCK relaxed for this PROBE ONLY -- CLASSES_PER_STEP stays 20
# (untouched); NUM_STEPS is intentionally PROBE_NUM_STEPS (3), not the full
# 5, for this smoke test (see PROBE_NUM_STEPS's own comment above).
assert NUM_STEPS == PROBE_NUM_STEPS, f"This PROBE is {PROBE_NUM_STEPS}x20 ONLY; got NUM_STEPS={NUM_STEPS}"
assert CLASSES_PER_STEP == 20, f"This PROBE is {PROBE_NUM_STEPS}x20 ONLY; got CLASSES_PER_STEP={CLASSES_PER_STEP}"

# --- Epoch budget -----------------------------------------------------------
# R3 (the EPOCH3 run, analysis in analysis_R3/reports/convergence_analysis_R3.txt)
# showed that with EPOCHS=3, train CE was STILL IMPROVING by >=3% relative in the
# final epoch for the large majority of (method, step) combinations (mean final-
# epoch relative improvement ~30% across all 10 methods x 5 steps), including both
# top-2 methods by final accuracy (rank_extension_orth_factor_lam_50_kd_T2 and
# simple_avg_factor_orth_kd_T2, the latter still improving 29-35% per step at
# epoch 3). A geometric-decay extrapolation of the per-epoch relative-improvement
# curve puts the epoch count needed to fall under a 3% convergence threshold at a
# median of +3 epochs (75th pct +4), i.e. EPOCHS=6-7. We set EPOCHS=6 (doubling the
# previous budget, 30 vs 15 global epochs) as a practical compromise: it captures
# most of the remaining convergence gains (including for the slower-converging
# top-2 methods) without doubling compute again beyond what R3's evidence supports.
# See analysis_R3/reports/convergence_analysis_R3.txt for the full per-method,
# per-step numbers behind this recommendation.
#
# PRE-THESIS FIX 3 (6 -> 9): the EPOCH6 run's own convergence re-check
# (analysis_R4/reports/rigorous_assessment_new_vs_old.txt, Section 1) applied the
# identical >=3%-relative-improvement test at epoch 6 and found the split is NOT
# uniform: the 4 non-KD methods (SimpleAvg, RankExt, and their FactorOrth variants
# without KD) are still under-trained in every one of their 5 steps at epoch 6
# (10-23% relative train-CE improvement in the final epoch), while the 4 KD (T=2)
# methods are already marginally converged by epoch 6 (mostly <3% in steps 2-5).
# The same geometric-decay extrapolation used for the 3->6 jump above puts the
# median additional epochs needed beyond 6 at +2 (75th pct +4, worst-case single
# (method,step) outlier +9). We keep EPOCHS uniform across all methods at 9 (a flat
# budget keeps the 8-method comparison protocol clean -- no method gets a
# compute-budget advantage the others didn't), even though the KD methods
# individually converge earlier: empirically they plateau by ~epoch 6, so epochs
# 7-9 for KD methods are mostly "free" extra training that best-epoch selection
# (PRE-THESIS FIX 1 below) will now correctly avoid over-fitting into, while the
# non-KD methods use the additional epochs to keep closing their convergence gap.
# EPOCH REDUCTION EXPERIMENT (2026-08-25, explicit user directive): 9 -> 7,
# uniformly across every epoch-budget constant (no method-specific override --
# LORA_EPOCHS/RANKEXT_EPOCHS are the two actually consumed by the 8 active
# methods' training paths (train_independent_loras() / run_rank_extension_
# variant()) and by every reporting/plotting consumer that reads epoch counts;
# FT/JOINT/ORTH/SCRATCH stay in lockstep for consistency even though their
# training paths are currently disabled via METHODS_TO_RUN, same as every
# prior epoch-budget change in this file's history above). Existing
# best-epoch (val-CE) selection logic (PRE-THESIS FIX 1, USE_BEST_EPOCH_
# SELECTION) is UNCHANGED -- this is a training-duration change only, not a
# stopping-criterion change; no early stopping is introduced. Rationale:
# per-(method,step) plateau analysis of the prior EPOCHS=9 run's own
# training_loss_history_by_epoch.csv found the RankExt family's two KD-
# carrying variants already flat/noisy (no further-than-1%-relative val-CE
# movement) by epoch ~4-6, while its two non-KD variants and simple_avg's
# FactorOrth variant still show small (<1% relative) genuine improvement out
# to epoch 8-9 -- smaller than the ~1pt run-to-run reproducibility noise floor
# already measured between nominally-identical historical reruns. EPOCHS=7
# is chosen as the point that preserves that noise-floor-scale residual for
# only the tail 1-2 epochs, not as a value with zero truncation risk.
# EXPERIMENT 1 OVERRIDE: canonical EPOCHS=7 (tuned for the 4x25 protocol) ->
# 9, uniformly, matching every pre-fix AND fixed-iteration 5x20 job in
# thesis_agent/sources/R6_catalog.md (all used epochs=9) -- this is the
# "keep all other scientific settings as comparable as possible" requirement
# for a pre-fix/post-fix 5x20 comparison, not a new tuning decision. Every
# epoch-budget constant changed in lockstep, same convention as every prior
# epoch-budget change in the canonical file's history (FT/JOINT/ORTH/SCRATCH
# stay in lockstep even though disabled via METHODS_TO_RUN).
FULL_FT_EPOCHS = 9
FULL_LORA_EPOCHS = 9
FULL_JOINT_EPOCHS = 9
FULL_ORTH_EPOCHS = 9
FULL_RANKEXT_EPOCHS = 9

SCRATCH_EPOCHS = 9

FT_EPOCHS = 9
# FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT (task brief Section 5, explicit
# instruction): BOTH families use 9 epochs/CL step, matching the historical
# Exp1/R7 protocol exactly -- NOT the prior two production runs' 6-epoch
# budget (job 4969059, job 4970580). This is an UNCONDITIONAL, flat value --
# deliberately NOT gated on FAST_RUN (this script is never run in
# FAST_RUN_DEBUG mode at all, per the task brief's "no smoke test" rule; see
# the hard assertions immediately below, which would otherwise be
# meaningless if a debug path could silently override them).
LORA_EPOCHS = 9
JOINT_EPOCHS = 9
ORTH_EPOCHS = 9
# PROBE: 9 -> PROBE_RANKEXT_EPOCHS (2). SimpleAvg never runs in this probe
# (LORA_EPOCHS is vestigial here), so only RANKEXT_EPOCHS is reduced.
RANKEXT_EPOCHS = PROBE_RANKEXT_EPOCHS

# PROTOCOL LOCK relaxed for this PROBE ONLY (documented in
# R7/rankext_headlr_small_probe_readiness.md) -- the canonical script's
# original unconditional "must be 9" locks are replaced by explicit
# PROBE-scale checks so this reduction is loud/intentional, not silent.
assert RANKEXT_EPOCHS == PROBE_RANKEXT_EPOCHS, f"RANKEXT_EPOCHS must be {PROBE_RANKEXT_EPOCHS} for this probe, got {RANKEXT_EPOCHS}"


BATCH_FT = 8
ACCUM_FT = 2

# COLAB ADAPTATION: default conservative for a 16GB T4 (Colab's most common
# free-tier GPU). Reasoning: CLIP ViT-B/16 with LoRA adapters up to rank 80
# on q/v projections is a small trainable-parameter footprint (the backbone
# is frozen and run at inference-mode memory cost for its forward pass; only
# LoRA + classifier-head activations need gradient buffers) -- the cluster's
# BATCH_LORA=16 ran comfortably on A40 (48GB), which has far more headroom
# than a T4. 8 is chosen as a genuinely conservative default (half the
# cluster value) to leave margin for a shared/variable-overhead Colab
# instance, NOT because BATCH_LORA=16 is known to OOM on a T4 -- it likely
# would not, but this has not been empirically measured on Colab hardware,
# so the safer default is used. Override via the COLAB_BATCH_SIZE env var
# (e.g. `COLAB_BATCH_SIZE=16` to match the cluster exactly on an L4/A100).
# On CUDA OOM here, re-run with a SMALLER COLAB_BATCH_SIZE (e.g. 4) -- do NOT
# change rank/LoRA/KD/protect30/FactorOrth/task-count/epoch-count/LR to
# "rescue" an OOM; batch size is the only knob this adaptation authorizes.
BATCH_LORA = int(COLAB_BATCH_SIZE) if COLAB_BATCH_SIZE else 8
ACCUM_LORA = 1
print(f"COLAB ADAPTATION: BATCH_LORA={BATCH_LORA} "
      f"({'from COLAB_BATCH_SIZE env var' if COLAB_BATCH_SIZE else 'conservative default for 16GB T4'})")


LR_FT = 3e-5
# RESTORED (2026-08-25, explicit user correction): LR_LORA reverted back to
# 5e-5, matching the R6 reference file byte-for-byte. A same-day STRICT-
# FAIRNESS REDESIGN had briefly unified this to LR_RANKEXT's 1e-4 (see git
# history for that analysis if it needs to be revisited), but per explicit
# instruction this run restores SimpleAvg to its canonical reference-file
# structure exactly. LR_RANKEXT below is UNTOUCHED by this revert -- current
# RankExt setup stays exactly as-is.
LR_LORA = 5e-5
LR_JOINT = 5e-5

LR_ORTH = 5e-5
LR_RANKEXT = 1e-4

# Overfitting review of R3 (analysis_R3/reports/overfitting_analysis_R3.txt) found
# only 4/150 (2.7%) epoch-level overfitting-signature events (val CE rising while
# train CE falls), all with small severity (<0.02 CE) -- i.e. MILD, not the kind of
# overfitting that calls for a stronger weight-decay regularizer. WEIGHT_DECAY=0.05
# was already a reasonable value for this, so it is left unchanged.
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.10
SCHED = "cosine"

USE_FP16 = torch.cuda.is_available()


# RESTORED (2026-08-25, explicit user correction, overrides the STRICT-
# FAIRNESS REDESIGN note this comment block used to carry): SimpleAvg's rank
# is back to its original/canonical project structure -- fixed rank=80 at
# EVERY step, no rank growth, no capacity-matching redesign. The prior
# same-day edit had changed LORA_R 80 -> 20 to try to match RankExt's
# per-step NEW-rank budget rather than its final cumulative rank (see git
# history for the full capacity analysis: because simple_avg's per-step delta
# gets extract_lora_state()'d into a dense B@A matrix and then
# simple_average_deltas()/apply_deltas_to_base() SUMS four such deltas
# directly into the base weight, four independent rank-80 deltas can reach an
# achievable merged rank up to min(4*80,768)=320 -- genuinely NOT capacity-
# equal to rank_extension's persistent, provable rank<=80 cap, despite both
# being nominally labeled "rank 80"). That capacity mismatch is REAL and
# UNCHANGED by this revert -- it is not being disputed or hidden, just no
# longer treated as something to fix by altering rank itself: per the
# explicit instruction accompanying this revert, rank is SimpleAvg's defining
# architectural axis (fixed-rank independent-then-merge vs. RankExt's
# incremental cumulative-rank growth is literally the structural distinction
# the whole comparison exists to study), so fairness is enforced only on
# shared CONTROLLABLE settings -- target modules, head-LR multiplier, base
# LR, and the pair-4 KD+FactorOrth loss-coefficient scaling all stay unified
# per the still-standing fairness redesign (see TARGET_MODULES_BY_FAMILY /
# HEAD_LR_MULTIPLIER_BY_FAMILY / LR_LORA==LR_RANKEXT / COMBINED_LOSS_SCALE_
# ENABLED's own comments below, all untouched by this revert) -- rank itself
# is excluded from that fairness pass by design, not by oversight.
LORA_R = 80
LORA_ALPHA = 2 * LORA_R
# Bumped 0.05 -> 0.1 for the EPOCH6 run: R3's overfitting was mild (see
# WEIGHT_DECAY comment above), but EPOCHS was doubling (3 -> 6) which roughly
# doubles the number of gradient updates each LoRA adapter sees per CL step, so a
# slightly stronger dropout was meant as a cheap, low-risk hedge against the extra
# epochs turning today's mild overfitting into something worse.
#
# PRE-THESIS FIX 4 (0.1 -> back to 0.05): the EPOCH6 run's rigorous re-check
# (analysis_R4/reports/rigorous_assessment_new_vs_old.txt, Section 3) found the
# hedge did not pay off as intended -- the epoch-over-epoch "train down / val up"
# overfitting-signature rate quadrupled (5.0% -> 22.0% of transitions) despite the
# extra dropout, and the one method that got measurably WORSE than the EPOCH3
# baseline (rank_extension_orth_factor_lam_50_kd_T2, the #1-ranked method, 68.15%
# -> 67.98%) regressed via exactly this pattern: its own step-1 val CE was best at
# local epoch 4 and got worse by epoch 6. Two things changed between R3 and this
# run at once (epochs AND dropout), so dropout's specific contribution to that
# regression can't be fully isolated -- but with best-epoch (val-CE) checkpoint
# selection now GENUINELY wired in (PRE-THESIS FIX 1: an epoch that overfits no
# longer gets kept, it just won't be selected as the per-step checkpoint), a
# second, blunter regularizer on top is redundant and only adds a confound to the
# epoch-budget comparison (FIX 3, 6->9). Reverting to 0.05 isolates "more epochs +
# working best-epoch selection" as the change under test for this run.
LORA_DROPOUT = 0.05
# ACCURACY-PUSH CHANGE 1: expanded q_proj/v_proj -> q_proj/k_proj/v_proj/out_proj
# for more adaptation capacity per CL step. Mechanically safe to extend (verified
# by reading the code, not assumed): extract_lora_state(), the factor-orth
# component computation, compute_delta_orth_components(), and
# find_clip_target_linear_modules()/GrowingRankLoRALinear wrapping all iterate
# named_modules() keyed on TARGET_MODULES generically -- none of them are
# hardcoded to q/v, so factor-orth, delta-trace, and rank-extension all cover the
# expanded set automatically with no further code changes. factor_total_mean is a
# .mean() over layers (not a .sum()), so LAMBDA_ORTH=50 stays comparably
# calibrated whether it's averaging over 2 or 4 module-types per layer. All four
# CLIP attention projections are hidden_size->hidden_size Linear layers, so there
# is no shape mismatch. Main real risk is overfitting / compute cost: this
# roughly doubles trainable LoRA params per step (and roughly doubles
# rank_extension's per-step compute) against the same small per-step dataset,
# on top of R3/R4 already flagging mild overfitting signatures at the smaller
# 2-module setting -- worth watching in the results, not just assuming a win.
# Revert to ["q_proj", "v_proj"] to disable (and update the pinned assert below).
#
# REVERT (2026-07-16, analysis_rankext_firststep/report.txt): this expansion is
# now FAMILY-CONDITIONAL, same pattern as CALIBRATION_ENABLED_FAMILIES below.
# rank_extension reverts to its BASELINE-proven 2-module setup
# (q_proj/v_proj only) -- the diagnostic report traced rank_extension's
# factor-orth collapse (first_step 8.35%->0.10%) partly to the 4-module
# expansion doubling the number of simultaneous per-layer orthogonality
# constraints enforced at every CL-step boundary, on top of head_lr x10 (see
# HEAD_LR_MULTIPLIER_BY_FAMILY below). simple_avg keeps the 4-module setup
# (it was never implicated in that collapse and benefits from the extra
# capacity: SimpleAvg+FactorOrth was the best single method in the calibfix
# run at 75.5%). TARGET_MODULES itself remains the simple_avg / default value
# (also used by any legacy/disabled training path below that predates
# family-conditional target modules); use TARGET_MODULES_BY_FAMILY /
# family_target_modules() for anything that knows its family.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# RESTORED (2026-08-25, explicit user correction): simple_avg's target
# modules reverted back to the 4-module ACCURACY-PUSH CHANGE 1 set (q,k,v,out
# -- see that change's own comment above), matching the R6 reference file
# (.../R6/vit_lora_cifar100_full5step_n5.py) byte-for-byte. A same-day STRICT-
# FAIRNESS REDESIGN had briefly narrowed this to q_proj/v_proj (matching
# rank_extension) to remove a target-module confound between families -- see
# git history for that analysis if it needs to be revisited -- but per
# explicit instruction this run restores SimpleAvg to its canonical
# reference-file structure exactly, not a fairness-redesigned variant.
# FINAL CORRECTION (2026-08-25, explicit user directive): simple_avg's target
# modules set to q_proj/v_proj ONLY, for all 4 simple_avg methods -- narrower
# than both the R6 reference file (q,k,v,out) and the earlier same-day STRICT-
# FAIRNESS REDESIGN (which paired q,v with a rank=20 capacity redesign). This
# time ONLY target_modules changes: rank (LORA_R=80, fixed, no growth) and
# alpha (LORA_ALPHA=160) are explicitly UNCHANGED, per instruction -- see
# LORA_R's own "RESTORED" comment above, still in force. rank_extension's own
# target modules (q_proj/v_proj) are UNTOUCHED by this edit -- see
# TARGET_MODULES_BY_FAMILY["rank_extension"] below, unchanged, still exactly
# as the current (correct) RankExt setup has it; the two families now happen
# to share the same 2-module set again, but that is this edit's incidental
# result for simple_avg, not a change made to rank_extension.
TARGET_MODULES_BY_FAMILY = {
    "simple_avg": ["q_proj", "v_proj"],
    "rank_extension": ["q_proj", "v_proj"],
}


def family_target_modules(family):
    """Per-family LoRA target modules. Unlisted families fall back to the
    global TARGET_MODULES default (never matched by method-name substring --
    same principle as family_applies_calibration() below)."""
    return list(TARGET_MODULES_BY_FAMILY.get(str(family), TARGET_MODULES))

# ACCURACY-PUSH CHANGE 2 (flag): rehearsal-free, post-merge-only classifier
# row-norm calibration (WA-style weight alignment, Zhao et al. 2020). See
# calibrate_classifier_row_norms() below for the mechanism and rationale.
#
# POST-INCIDENT FIX (analysis_rankext_drop/report.txt): this used to be applied
# identically to all 8 methods right before final evaluation. That corrupted
# rank_extension's two KD variants (68.0%->26.2% and 59.3%->50.1% all_seen
# accuracy) while leaving its two non-KD variants fine and helping simple_avg
# across the board. Root cause: calibrate_classifier_row_norms() computes ONE
# global target row-norm from all 100 rows and rescales each CL step's 20-row
# block to match it. simple_avg's classifier is 5 independently-reinitialized
# heads stitched together (each trained with 80 permanent negatives) -- exactly
# the scale-imbalance case WA calibration was designed to fix, and it
# empirically helps there. rank_extension's classifier is a single persistent
# matrix, incrementally trained with per-row gradient masking
# (add_classifier_row_gradient_mask) and protected/frozen rows across later
# steps (restore_protected_classifier_rows) -- it is self-consistent by
# construction, and a KD-driven row-norm difference between its blocks (steps
# 2-5 train against a distillation loss step 1 never sees) lets the SHARED
# global target_norm miscalibrate even step 1's untouched, frozen rows. Now
# gated per family via CALIBRATION_ENABLED_FAMILIES rather than one global
# switch, and consulted per-method through each method config's
# "apply_calibration" field (see add_method() / ACTIVE_METHOD_MAP) -- never by
# matching on method-name substrings.
USE_CLASSIFIER_CALIBRATION = True

# ACCURACY-PUSH CHANGE 2b (FIX 1, analysis_recency_fix/report.txt): family-aware,
# REGIME-GROUPED calibration for rank_extension. The POST-INCIDENT FIX above
# disabled ALL calibration for rank_extension because the single GLOBAL target
# norm (mean over all 100 rows) mixed step 1's untouched, teacher-less rows
# with steps 2-N's KD-trained rows and corrupted the KD variants (68.0%->26.2%,
# 59.3%->50.1%). analysis_recency_fix/report.txt traces the low OPEN-argmax
# numbers for ALL FOUR rank_extension variants (not just the KD ones) to a
# separate, still-unaddressed mechanism -- recency bias in the open 100-way
# argmax: tables/per_step_accuracy_open_vs_restricted_by_method.csv (WIDERANK
# run) shows RESTRICTED (step-local 20-way) accuracy in the 80-97% range at
# EVERY step for every rank_extension variant, while OPEN accuracy for
# early/middle steps collapses toward 0-50% (e.g. rank_extension_orth_factor_
# lam_50_kd_T2: restricted 93-97% at every step, open 45-70% at steps 1-4). The
# frozen blocks are preserving old-class knowledge correctly -- their
# classifier rows are simply losing the 100-way argmax competition on SCALE,
# not on discriminative quality. RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED
# re-enables calibration for rank_extension using a GROUPED target instead of
# the single global one: for KD variants, step 1 (teacher-less) is calibrated
# to its OWN group mean (a singleton group -> no-op, so its untouched rows are
# never rescaled using KD-regime statistics -- avoiding the exact corruption
# mechanism above); steps 2..N (all KD-trained against a distillation signal
# step 1 never sees) are calibrated as ONE group to THEIR OWN shared mean,
# directly equalizing norm growth across steps 2..N without mixing in step 1.
# For non-KD variants there is only one training regime across all NUM_STEPS
# steps, so this reduces to the same global calibration the POST-INCIDENT FIX
# said was "fine" for those two variants (only the KD variants were corrupted
# before). See calibrate_classifier_row_norms(mode=...) for the implementation
# and CALIBRATION_MODE_BY_FAMILY for the per-family selector. Flag-gated so it
# can be reverted in one line; recorded per-method via the "calibration_mode"
# column in run_config.json / hyperparameters_by_method.json.
RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED = True

# ACCURACY-PUSH CHANGE 2c (FIX 2, analysis_recency_fix2/report.txt): confidence
# -weighted regime-grouped calibration for rank_extension. The FIX 1 run
# (R3/results_fix1_20260721_light) proved FIX 1's core assumption wrong: for
# the two NON-KD rank_extension variants, FIX 1's regime_grouped mode achieves
# essentially PERFECT mean row-norm equalization across all 5 steps (ratio ==
# 1.0000000 to 7 decimals post-calibration, tables/classifier_row_norm_
# diagnostics_by_method_step.csv) yet open-argmax accuracy for early/middle
# steps barely moved (e.g. rank_extension step 1: 0.8%->1.15%, step 3:
# 2.2%->2.95% -- noise-level, not the ~+30-75pp analysis_recency_fix/
# projected_improvement.csv projected). Inter-step mean row-norm imbalance was
# only ever a ~15-29% multiplicative spread pre-calibration for the non-KD
# variants (far too small to be the sole cause of 50-80pp open-accuracy gaps)
# and was <5% and NON-monotonic for the KD variants -- yet the gap itself is
# 20-80pp and has a distinctive BOWL shape (worst at steps 2-3, better at both
# step 1 and step 5), which does not track row-norm ratio at all. What the gap
# DOES track, for the KD variants specifically, is each step's own final-epoch
# validation CE loss (tables/training_loss_history_by_epoch.csv):
# rank_extension_orth_factor_lam_50_kd_T2's val_ce_loss is 0.23 at step 1,
# jumps to 0.93 at step 2 (worst), then recovers monotonically through step 5
# (0.76, 0.75, 0.62) -- the same bowl shape as the accuracy gap (-26, -48,
# -40, -23, -4). Step 2 is the hardest KD regime (first step with a real
# teacher, teacher itself only just finished training) and step 1 is KD-free
# (cleanest signal); FIX 1's plain group-MEAN target treats every step in a
# group identically regardless of how well-trained it actually was. FIX 2
# keeps FIX 1's exact grouping (KD: {step1} no-op + {steps2..N} one group;
# non-KD: all NUM_STEPS steps one group -- unchanged, since Task B found no
# evidence step 1's no-op status was hurting it) but sets each step's TARGET
# norm to the group mean times a bounded, monotonic BOOST derived from that
# step's own final-epoch val_ce_loss relative to its group's mean val_ce_loss
# (worse-than-average steps get boosted above the group mean, better-than-
# average steps get pulled slightly below it) -- ONLY for KD variants; the
# boost is forced to 1.0 (exactly FIX 1's plain mean-match, no-op difference)
# for non-KD variants, since Task B found no clean val_ce_loss/gap
# correlation there -- see calibrate_classifier_row_norms_confidence_weighted()
# for the exact formula and gating.
# This is still a pure post-hoc, per-step-UNIFORM rescale (restricted accuracy
# stays exactly invariant, same as FIX 1/the original WA calibration -- no
# retraining, no new parameters), just with a smarter, evidence-backed target
# instead of a flat mean. Expected effect is DELIBERATELY modest (single-digit
# pp on the worst KD steps, near-zero on the non-KD variants where val_ce_loss
# does not track the gap either -- see report.txt Task B) -- FIX 1 already
# proved the ceiling on pure scale-only correction is much lower than the
# original restricted-accuracy upper bound implied. Flag-gated so it can be
# reverted to plain FIX 1 behavior in one line; when False, CALIBRATION_MODE_
# BY_FAMILY falls back to "regime_grouped" (FIX 1) exactly as before.
RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED = True

# RANK_EXT FIRST_STEP FIX (task 2 decision doc, 2026-08-17): the opt-in
# "feature-anchor lever" (two extra methods, "rank_extension_featanchor" /
# "rank_extension_orth_factor_featanchor", plus a DEFAULT-OFF "fixed_base"
# mode, all toggled per-method via METHODS_TO_RUN) has been REMOVED -- see the
# git history for that code. This constant/mechanism replaces it: read from
# the completed R3 run's diagnostics before deciding what to build --
#
# 1. feature_alignment_diagnostics_by_method_step.csv: own_minus_recent_cos_gap
#    (cosine to the OLD step's own classifier row minus cosine to the best-
#    matching RECENT step's row, measured on the final model) is strongly
#    negative for both non-KD rank_extension baselines -- rank_extension:
#    -0.41/-0.38/-0.38/-0.32 across steps 1-4; rank_extension_orth_factor_lam_50:
#    -0.21/-0.17/-0.18/-0.13 -- meaning old-step images' features have drifted
#    CLOSER to a recent step's classifier direction than to their own by the
#    end of training. The two KD variants (rank_extension_kd_only_T2,
#    rank_extension_orth_factor_lam_50_kd_T2), which already have healthy
#    first_step accuracy (51.3%, 54.0% vs 0.0%/5.65% for the non-KD pair),
#    show near-zero gaps (-0.005 to +0.14) -- feature-drift magnitude tracks
#    first_step failure almost exactly.
# 2. classifier_bias_diagnostics_by_method_step.csv: bias_offset_vs_grand_mean
#    never exceeds ~0.011 in magnitude for ANY method/step -- two orders of
#    magnitude smaller than the cosine gaps above. A classifier-logit-offset
#    explanation cannot account for a 0.0% open-vs-38.9%-restricted accuracy
#    collapse (per_step_accuracy_open_vs_restricted_by_method.csv, rank_extension
#    step 1) at that scale.
#
# Conclusion: feature drift, not classifier bias, drives rank_ext first_step
# failure -- so the fix is option (a) (functional anchor to a non-drifting
# reference), not (b) (logit/margin calibration). Implemented as an
# UNCONDITIONAL, always-on mechanism (no more per-method opt-in) in
# run_rank_extension_variant()/DeltaOrthRankExtensionTrainer.compute_loss():
# every rank_extension-family method that does NOT use KD anchors its CLS
# hidden state (cosine distance) to the frozen PRETRAINED CLIP backbone (no
# LoRA contribution at all, not just "previous step") on every training step,
# including step 1. KD methods are left alone -- their existing logit-KD
# teacher already produces the same near-zero drift per (1) above, so adding a
# second anchor mechanism on top would only add risk with no diagnosed upside.
# Weight kept at 1.0 -- the only value with any measured precedent (the
# now-removed chained variant used this weight and it materially fixed
# first_step, 0.0% -> 51.9%, without hurting later_steps, 25.85% -> 51.45%).
#
# Expected effect: first_step should rise materially for "rank_extension" and
# "rank_extension_orth_factor_lam_50" (the two non-KD core methods), plausibly
# by a similar order of magnitude to the measured chained result above, since
# anchoring to the pretrained backbone is at least as constraining as
# anchoring to the immediate predecessor step (no cumulative multi-hop drift
# possible). Risk: this loss actively resists the shared backbone moving away
# from pretrained-feature space at EVERY step, which is also the mechanism
# that lets new-class (later_steps) accuracy improve step over step -- some
# tension with later_steps/new-class learning is possible even though the
# analogous chained mechanism measured a later_steps IMPROVEMENT, not a
# regression; if a real training run shows later_steps or new-class accuracy
# regressing materially, RANKEXT_PRETRAINED_ANCHOR_WEIGHT below is the single
# knob to lower.
RANKEXT_PRETRAINED_ANCHOR_WEIGHT = 1.0

# OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION (R6 follow-up,
# 2026-08-21): additive training-side mechanism, KD RankExt methods only (see
# RANKEXT_PROJECTED_PROTECT_METHODS below). Motivation (R6 NCM/staleness
# analysis): old-representation drift -- the shared backbone's embedding of an
# already-learned class keeps moving as later steps add new LoRA rank blocks
# into the SAME weight matrices -- is the primary diagnosed failure mechanism
# (frozen NCM prototypes go stale monotonically with age; cross-step/new-class
# overlap may also contribute but is not separately established by the current
# diagnostics). This loss penalizes the component of each current-step
# training image's student-vs-previous-step-teacher feature displacement that
# falls inside the subspace spanned by the teacher's already-learned
# classifier-row directions -- i.e. "don't move old-class-relevant directions
# more than necessary while learning new classes," leaving the orthogonal
# (plausibly new-class-relevant) directions completely free. Never applied to
# old images (current-step images only, teacher forward is no_grad) --
# compatible with strict no-replay CIL. Weight kept at 1.0 as the initial,
# untuned value pending a real training run; see
# RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT's own comment below for the sweep
# plan.
RANKEXT_PROJECTED_PROTECT_METHODS = {
    "rank_extension_kd_only_T2",
    # RENAMED (2026-08-25): "..._lam_50_kd_T2" -> "..._lam_15_kd_T2" (pair-4's
    # effective lambda was 15.0 under the strict-fairness rescaling, not 50.0)
    # -- see METHODS_TO_RUN's own comment for the full rename. RENAMED BACK
    # (2026-08-25, FULL-STRENGTH COMBINED EXPERIMENT): COMBINED_LOSS_SCALE_
    # ENABLED=False restores the unscaled lambda=50.0, so this identifier is
    # "..._lam_50_kd_T2" again. protect30's BEHAVIOR is unchanged throughout
    # both renames: this method is still in this set, still gets
    # protect_weight=30.0, only the spelling of its identifier changed.
    "rank_extension_orth_factor_lam_50_kd_T2",
    # ORTH-LAMBDA INTERACTION ABLATION (2026-08-22, job 4914807 follow-up)
    # REMOVED 2026-08-23: rank_extension_orth_factor_lam_25_kd_T2 tested
    # whether lambda_factor_orth=50 was over-constraining/overlapping with
    # projected feature protection (lambda_protect=30). Result (job 4915286):
    # all_seen 70.58 vs the lam_50 flagship's 70.62 -- a -0.04pp difference,
    # inside noise, with drift/cosine diagnostics equal-to-slightly-worse than
    # lam_50. Hypothesis NOT supported; variant removed from the active set.
    #
    # FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT (task brief Section 19): the
    # two NEW full-KD RankExt arms are added here under their own, non-legacy
    # names, to activate the SAME unchanged protect_weight=30.0 mechanism
    # above (never a new implementation).
    "rank_extension_fullkd_T2_protect30",
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
}
RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT = 30.0

# STRUCTURAL NULL-SPACE RANKEXT (R6 roadmap Stage 2) -- CLOSED 2026-08-24.
# A v_proj-only, per-layer-pulled-back-basis hard structural constraint
# (rank_extension_nullspace / _kd_only_T2 / _orth_factor_lam_50_kd_T2) was
# designed, implemented, and evaluated in job 4917775. Result: the local
# leakage guarantee was enforced essentially exactly (residual-space leakage
# ~1e-14), but the effect did not propagate into a material downstream
# result -- flagship all_seen 70.73 vs the settled 70.62 baseline (+0.11pp,
# noise), with step1 -0.90pp and step5 -2.75pp regressions offsetting gains
# at steps 2-4, and the non-KD variant regressing materially without KD to
# stabilize it. Evidence pointed to unconstrained q_proj and/or downstream
# nonlinear propagation (LayerNorm/MLP/cross-layer mixing) as the more
# likely dominant channels, since near-perfect LOCAL isolation on v_proj
# produced only a weak, partial GLOBAL effect. The mechanism and its 3
# methods have been fully removed from this file (this is not a disabled/
# dormant flag -- see git history at commit 8153008 for the removed
# implementation if it needs to be revisited). RANKEXT_PROJECTED_PROTECT_
# METHODS above and the settled soft feature-protection mechanism it gates
# are UNRELATED and remain fully active -- do not confuse the two.

# Master switch above still gates calibration overall (False disables it for
# every method, same as before). When True, CALIBRATION_ENABLED_FAMILIES
# decides which families actually get it. simple_avg: keep True (empirically
# helps -- see report). rank_extension: now True, gated behind
# RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED above -- its persistent, row-masked
# classifier is still not the independently-reinitialized-heads scenario plain
# WA/global calibration targets (that mode remains OFF for it), but the
# regime-grouped mode below is specifically designed around its training
# structure. Add new families here explicitly; unlisted families default to no
# calibration (see family_applies_calibration() below).
CALIBRATION_ENABLED_FAMILIES = {
    "simple_avg": True,
    "rank_extension": bool(RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED),
}

# Per-family calibration ALGORITHM (only consulted when
# family_applies_calibration(family) is True). "global": original single
# target-norm-over-all-rows behavior (Zhao et al. 2020 WA, unchanged for
# simple_avg). "regime_grouped": FIX 1, rank_extension only.
# "confidence_weighted_regime_grouped": FIX 2 (analysis_recency_fix2/
# report.txt), originally rank_extension only -- same grouping as
# "regime_grouped" but each step's target norm is boosted/damped by its own
# training-quality signal (see RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED
# above and calibrate_classifier_row_norms_confidence_weighted()); for
# rank_extension only selected when RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED
# is ALSO True (FIX 2 is layered on top of FIX 1's grouping, not a replacement
# for it). CALIBRATION EXPERIMENT (2026-08-25): also now selected for
# simple_avg -- see the dated note directly above CALIBRATION_MODE_BY_FAMILY
# below for the rationale; simple_avg has no FIX-1-style prerequisite flag
# since it never used "regime_grouped" as an intermediate step. "off" is never
# actually selected while CALIBRATION_ENABLED_FAMILIES also gates the family,
# but is included so this dict alone documents intent if that invariant is
# ever changed.
# CALIBRATION EXPERIMENT (2026-08-25, explicit user directive): simple_avg
# switched from "global" to "confidence_weighted_regime_grouped" -- the SAME
# algorithm already used by rank_extension (calibrate_classifier_row_norms_
# confidence_weighted(), see that function's docstring). Rationale: prior R6
# diagnostics (per_step_accuracy_open_vs_restricted_by_method.csv,
# classifier_row_norm_diagnostics_by_method_step.csv) showed restricted
# (closed-set) accuracy stays flat (92-95%) across all 4 simple_avg variants
# while open-set accuracy collapses for the KD/FactorOrth variants, and
# post-calibration row-norm ratios under "global" mode are already exactly
# 1.0 -- i.e. the existing flat-mean calibration already perfectly equalizes
# MAGNITUDE, so the residual open-set failure must be a SHAPE/structure
# problem the flat rescale cannot reach. confidence_weighted_regime_grouped
# targets exactly that axis (a per-step, validation-CE-derived boost on top
# of the group mean) and is already implemented and validated for
# rank_extension; nothing about calibrate_classifier_row_norms_confidence_
# weighted() is rank_extension-specific (it only reads model.classifier.
# weight, NUM_STEPS, classes_for_step(), and the module-global epoch_loss_rows
# accumulator filtered by method_name -- all family-agnostic and already
# populated for simple_avg methods via the same shared EpochValidationCallback
# used by both families). See run_simple_avg_variant()'s calibration dispatch
# below -- previously that function only ever called calibrate_classifier_
# row_norms() (the "global"/"regime_grouped" function), never the confidence-
# weighted one; that dispatch gap is fixed alongside this config flip so the
# mode value actually takes effect instead of silently falling through to
# flat single-group behavior. rank_extension's own calibration is UNCHANGED.
CALIBRATION_MODE_BY_FAMILY = {
    "simple_avg": "confidence_weighted_regime_grouped",
    "rank_extension": (
        "confidence_weighted_regime_grouped"
        if (RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED and RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED)
        else "regime_grouped" if RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED
        else "off"
    ),
}


def family_applies_calibration(family):
    """True iff USE_CLASSIFIER_CALIBRATION is on AND this family opted in via
    CALIBRATION_ENABLED_FAMILIES. Unlisted families default to False (safer
    than silently calibrating a family nobody has reasoned about)."""
    return bool(USE_CLASSIFIER_CALIBRATION) and bool(CALIBRATION_ENABLED_FAMILIES.get(str(family), False))


def family_calibration_mode(family):
    """Per-family calibration algorithm selector -- see
    CALIBRATION_MODE_BY_FAMILY above. Only meaningful when
    family_applies_calibration(family) is True; unlisted families default to
    "global" (matches the pre-existing, single-mode behavior)."""
    return str(CALIBRATION_MODE_BY_FAMILY.get(str(family), "global"))

# ACCURACY-PUSH CHANGE 3 (flag): classifier-head LR = LR_LORA/LR_RANKEXT times
# this multiplier, via HeadLRTrainerMixin.create_optimizer() below. Set to 1.0 to
# fully disable (falls back to the untouched stock Trainer.create_optimizer()).
#
# REVERT (2026-07-16, analysis_rankext_firststep/report.txt): now FAMILY-
# CONDITIONAL, same pattern as CALIBRATION_ENABLED_FAMILIES /
# TARGET_MODULES_BY_FAMILY. rank_extension reverts to x1.0 (BASELINE, no
# multiplier) -- the report flagged head_lr x10 as a plausible AMPLIFIER
# (not sole cause) of the transient step-boundary CE spike that factor-orth
# enlarges for rank_extension, since a 10x classifier LR turns a noisy
# transient loss spike into a much larger one-shot weight change. It was
# REJECTED as a *sufficient* cause on its own (applied uniformly in the
# calibfix run to all 4 rank_extension variants, only 2 of which collapsed),
# but reverting it removes one more untested variable while we test the
# lambda-warmup fix below, and BASELINE (x1.0) is the config that actually
# produced rank_extension's 68.0 historical best. simple_avg keeps x10 (never
# implicated; SimpleAvg+FactorOrth's 75.5% was achieved WITH it).
HEAD_LR_MULTIPLIER = 10.0

# RESTORED (2026-08-25, explicit user correction): simple_avg's head-LR
# multiplier reverted back to x10, matching the R6 reference file byte-for-
# byte. A same-day STRICT-FAIRNESS REDESIGN had briefly unified this to x1.0
# for both families (see git history for that analysis if it needs to be
# revisited), but per explicit instruction this run restores SimpleAvg to its
# canonical reference-file structure exactly. rank_extension's own head-LR
# multiplier (x1.0, its own proven-safe BASELINE -- see the REVERT comment
# above for why x10 is unsafe specifically for rank_extension+factor-orth) is
# UNTOUCHED by this revert.
HEAD_LR_MULTIPLIER_BY_FAMILY = {
    "simple_avg": float(HEAD_LR_MULTIPLIER),
    # PROBE: historical baseline is 1.0 (job4971615); swept via
    # RANKEXT_HEADLR_PROBE_MULTIPLIER env var (defaults to 1.0 = historical
    # baseline when unset, so this file is a no-op behavioral match to the
    # canonical script if run without the env var).
    "rank_extension": float(RANKEXT_HEADLR_PROBE_MULTIPLIER),
}


def family_head_lr_multiplier(family):
    """Per-family classifier-head LR multiplier. Unlisted families fall back
    to the global HEAD_LR_MULTIPLIER default."""
    return float(HEAD_LR_MULTIPLIER_BY_FAMILY.get(str(family), HEAD_LR_MULTIPLIER))


REPLAY_PER_CLASS = 20
RANKEXT_REPLAY_PER_CLASS = REPLAY_PER_CLASS


LAMBDA_ORTH = 50.0
LAMBDA_ORTH_DELTA_TRACE = LAMBDA_ORTH
LAMBDA_ORTH_FACTOR = LAMBDA_ORTH

# SINGLEJOB PACK ADDITION: DenseOrth (Arms 3/4) uses its OWN fixed lambda,
# independent of LAMBDA_ORTH=50.0 (which stays reserved for the untouched
# historical FactorOrth/delta_trace code paths -- FactorOrth is not retrained
# in this script, but LAMBDA_ORTH is left unchanged for provenance/diff
# purposes). Value fixed per the pre-registered experiment design (lambda=20),
# not tuned. DENSE_ORTH_WARMUP_EPOCHS matches R8's own audited convention
# (supervisor_regularizer_repair_both_families_r8.py: ORTH_WARMUP_EPOCHS=1.0,
# unconditional 1-epoch linear ramp for every DenseOrth variant, no asymmetry)
# -- unlike Exp1's existing COMBINED_ORTH_WARMUP_* mechanism (factor_orth
# warmup, gated to only apply when KD is ALSO active), DenseOrth's warmup here
# is unconditional: it applies identically whether or not KD is also active
# (Arm 3 and Arm 4 both get it), matching R8's "no asymmetry" design exactly.
DENSE_ORTH_LAMBDA = 20.0
DENSE_ORTH_WARMUP_EPOCHS = 1.0

# SINGLEJOB PACK ADDITION: old-seen KD (Arms 1/2/4) KD-weight warmup epochs.
# Arm 1 uses 0.0 (no warmup -- flat kd_weight from the first batch, matching
# R7's historical KD exactly except for the old-seen class mask). Arms 2/4
# use 1.0, matching R8's KD_WARMUP_EPOCHS=1.0 (1-epoch linear ramp 0->1).
KD_OLDSEEN_WARMUP_EPOCHS_ARM1 = 0.0
KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4 = 1.0
# FINAL KD-WEIGHT EXPERIMENT (R6 roadmap, CLOSED 2026-08-25): the one-off
# single-point KD_WEIGHT=0.75 treatment (job ..._4x25_kdw075_flagship_..._
# 20260824_195751) is done and did not improve on the settled value -- see
# the RUN_NAME_BASE comment above for the result summary. KD_WEIGHT is
# reverted to its settled 1.0 here for the final 8-method thesis comparison
# run; this is once again the single global constant every KD-active
# method's config metadata (rank_extension_kd_only_T2, simple_avg_kd_T2,
# rank_extension_orth_factor_lam_50_kd_T2 [renamed back 2026-08-25, see
# COMBINED_LOSS_SCALE_ENABLED's own comment below], simple_avg_factor_orth_kd_T2)
# reads its nominal kd_weight from -- no per-method override exists or is
# introduced here. T (2.0), LAMBDA_ORTH (50.0), and every other
# hyperparameter below are untouched.
KD_WEIGHT = 1.0
KD_TEMPERATURES = [2.0]
KD_TEMPERATURE = KD_TEMPERATURES[-1]

# ACCURACY-PUSH CANDIDATE (flag, default ON): SimpleAvg+FactorOrth+KD applies
# BOTH penalties at their full single-mechanism strength (lambda_orth=50,
# kd_weight=1.0 -- identical to simple_avg_factor_orth and simple_avg_kd_T2
# individually) and in the calibfix run this scored 63.71% all_seen -- BELOW
# both components alone (SimpleAvg+FactorOrth 75.54%, SimpleAvg+KD 71.51%).
#
# Evidence for a full-strength MAGNITUDE conflict, checked directly against
# training_loss_history_by_epoch.csv from the calibfix run before choosing
# this fix over a timing-based one:
#   - factor_orth_loss_weighted for simple_avg_factor_orth_kd_T2 is a violent
#     transient, 100-3200x train_ce_loss, concentrated ENTIRELY in
#     local_epoch==1 of each step (e.g. step 5: weighted orth=3218 vs
#     train_ce=2.54, kd_weighted=1.19 -- orth outweighs CE+KD combined by
#     roughly 850x at that single epoch), then collapses 3-4 orders of
#     magnitude by local_epoch==2 and is negligible for the rest of the step.
#   - kd_loss_weighted over the SAME step is NOT spiking or front-loaded the
#     same way -- it declines smoothly and monotonically across all 9 epochs
#     (step 5: 1.19 -> 0.67, roughly halving, comparable in scale to
#     train_ce_loss throughout). KD does not exhibit the kind of transient
#     that would point to a TIMING mismatch (e.g. "KD dominates late while
#     orth dominates early") -- both terms are largest at the SAME moment
#     (local_epoch==1), not different moments. This is why the chosen fix
#     below is magnitude-scaling, not a KD annealing schedule (see the
#     REJECTED alternative noted next to
#     COMBINED_ORTH_WARMUP_ENABLED further down).
# Halving each term's contribution ONLY when both are simultaneously active
# is a standard multi-objective balancing move (the combined method must be
# tuned as a combination, not a naive sum of two full-strength single-purpose
# settings), not a removal of either mechanism. It changes ONLY this one
# method's own hyperparameters -- simple_avg_factor_orth and simple_avg_kd_T2
# keep their original, independently-proven full-strength values unchanged
# (enforced in build_active_method_configs() via the lambda_orth_scale /
# kd_weight_scale args to add_method(), applied only to the
# simple_avg_factor_orth_kd_T2 call site -- and visible in the per-method
# config tables, since this method's lambda_orth/kd_weight columns will now
# read differently from its two single-penalty siblings).
# NOTE: "combined >= max(components)" is the hypothesis this scaling is meant
# to test, not a guaranteed outcome -- verify against the actual rerun.
# Set to False to restore the naive full-strength sum (the calibfix
# behavior that produced 63.71%).
#
# STRICT-FAIRNESS REDESIGN, pair-4 decision (2026-08-25, user directive): this
# scaling now ALSO applies to rank_extension_orth_factor_lam_15_kd_T2 (see
# that add_method() call site in build_active_method_configs() -- RENAMED
# 2026-08-25 from "..._lam_50_kd_T2", since its true effective lambda is
# 15.0, not 50.0; see METHODS_TO_RUN's own comment for the full rename) --
# the user's explicit choice, given a genuine conflict between strict
# coefficient parity and the training-stability evidence above, was to bring
# RankExt's side of the FactorOrth+KD pair DOWN to SimpleAvg's already-
# validated stable point (kd=0.5, lambda=15) rather than push SimpleAvg's
# side UP into its documented 63.71% collapse. This makes THIS RUN's
# rank_extension_orth_factor_lam_15_kd_T2 a deliberately different, rescaled
# configuration from the one that produced
# the historical all_seen=74.07 result (kd=1.0, lambda=50) -- that historical
# run/config is untouched and retained separately, not reproduced by this run.
# KD_WEIGHT and LAMBDA_ORTH themselves stay at their global settled values
# (1.0 / 50.0) and are UNCHANGED for every other method in both families
# (rank_extension_kd_only_T2, simple_avg_kd_T2 use KD_WEIGHT=1.0 unscaled;
# rank_extension_orth_factor_lam_50, simple_avg_factor_orth use
# LAMBDA_ORTH=50.0 unscaled) -- this flag affects ONLY the two combined
# FactorOrth+KD methods (pair 4), one per family, symmetrically.
# FULL-STRENGTH COMBINED EXPERIMENT (2026-08-25, explicit user directive):
# scaling DISABLED (True -> False). Both add_method() call sites that used to
# receive lambda_orth_scale=COMBINED_LAMBDA_ORTH_SCALE / kd_weight_scale=
# COMBINED_KD_WEIGHT_SCALE (simple_avg_factor_orth_kd_T2 and rank_extension's
# combined method) resolve their scale args via the existing `float(...) if
# COMBINED_LOSS_SCALE_ENABLED else 1.0` ternaries defined just below
# (_combined_lambda_scale / _combined_kd_scale in build_active_method_
# configs()) -- flipping this one flag is therefore sufficient to make BOTH
# combined methods' EFFECTIVE lambda_orth/kd_weight resolve to the full,
# unscaled LAMBDA_ORTH=50.0/KD_WEIGHT=1.0 globals, symmetrically, with no
# other code path touched. COMBINED_LAMBDA_ORTH_SCALE/COMBINED_KD_WEIGHT_SCALE
# themselves are left defined-but-inert below (not deleted) so the historical
# 0.5/15 fairness-rescaled configuration remains reconstructable by flipping
# this one flag back -- same "flag-gated, revert in one line" convention this
# file already uses throughout. This does NOT touch COMBINED_ORTH_WARMUP_
# ENABLED/EPOCHS or RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED/EPOCHS -- both existing,
# family-specific warmup mechanisms are unchanged; they will now ramp toward
# the larger, unscaled lambda_orth=50 (rather than 15) for the two combined
# methods, which is the correct, automatic consequence of this change, not a
# new warmup being added anywhere.
COMBINED_LOSS_SCALE_ENABLED = False
COMBINED_LAMBDA_ORTH_SCALE = 0.3
COMBINED_KD_WEIGHT_SCALE = 0.5

# STRICT-REVIEW SECOND ITERATION (B2, 2026-07-17): the halving above was
# directionally right (simple_avg_factor_orth_kd_T2 all_seen: 63.71% calibfix
# -> 69.13% NEW/revert) but still below simple_avg_kd_T2's own 71.51% floor.
# Re-checked training_loss_history_by_epoch.csv from the NEW/revert run
# (analysis_strict_review/report.txt Part B2) before choosing the next move:
#
#   TRANSIENT (local_epoch 1-2 of steps 2-5), i.e. the conflict point:
#     mean factor_orth_loss_weighted = 507.6, mean (ce+kd) = 2.25
#     -> orth still outweighs ce+kd by ~226x on average (44.9x-509.4x per
#     step, WORSE at later steps: 44.9x@S2, 241.8x@S3, 449.9x@S4, 509.4x@S5),
#     despite lambda_orth already being halved (50->25). Cutting lambda alone
#     cannot fix this at any reasonable value: even another 40% cut (25->15,
#     the scale change below) only takes the ratio from ~226x to ~135x on
#     average -- still completely dominant. Magnitude-scaling has hit
#     diminishing returns; the transient needs a TIMING fix, not a bigger
#     magnitude cut. This is why COMBINED_ORTH_WARMUP_ENABLED is turned ON
#     below instead of pushing lambda_orth_scale much further down.
#
#   STEADY STATE (local_epoch >= 3 of steps 2-5), i.e. everywhere else:
#     mean factor_orth_loss_weighted = 0.0235, mean (ce+kd) = 0.871
#     -> orth is already only ~2.7% of ce+kd at lambda_orth_scale=0.5 -- NOT
#     dominant, nothing here supports a further cut being necessary. The
#     modest extra trim to 0.3 (lambda 25->15) below is kept small and
#     explicitly a secondary safety margin for the tail of the warmup ramp,
#     not a response to steady-state dominance (there isn't any).
#     mean kd_loss_weighted / mean train_ce_loss = 1.873 for THIS method.
#     Checked against simple_avg_kd_T2 (kd_weight_scale=1.0, no orth) at the
#     SAME epoch selection: its own steady-state kd/ce ratio is 1.757 --
#     essentially the SAME ratio the combined method already has at half
#     kd_weight. KD is therefore already behaving proportionately, not being
#     starved -- there is no evidence in this run that kd_weight is the
#     under-tuned term, so COMBINED_KD_WEIGHT_SCALE is left at 0.5 rather than
#     raised toward 0.7 as originally floated. Raising it further with no
#     supporting signal, on top of an already-large KD/CE ratio, risks
#     over-anchoring to the teacher and suppressing later_steps (new-class)
#     fitting -- exactly the metric (66.30% vs simple_avg_factor_orth's
#     76.63%) where this method is currently furthest behind its own
#     single-mechanism siblings. "Exceeding both single-mechanism components"
#     remains a hypothesis under test, not a guaranteed outcome of this
#     change -- verify against the actual instrumented rerun.
COMBINED_ORTH_WARMUP_ENABLED = True
COMBINED_ORTH_WARMUP_EPOCHS = 1.0
# (REJECTED ALTERNATIVE, still holds: a KD-side annealing schedule. kd_loss_
# weighted has no spike/front-loading in either the calibfix or NEW run --
# still a smooth, comparable-to-CE curve across the whole step in both -- so
# there remains no timing-mismatch signal for a KD-specific schedule to
# correct; only the orth term gets a warmup.)

# JOINT PACK ADDITION (GPU preflight): reduced ONLY under FAST_RUN_DEBUG=1.
VALIDATION_PER_CLASS = 3 if FAST_RUN else 25
LOSS_NA_FILL = 0.0

METHOD_DISPLAY_NAME_MAP = {
    "simple_avg": "SimpleAvg",
    "simple_avg_kd_T1": "SimpleAvg + KD T1",
    "simple_avg_kd_T2": "SimpleAvg + KD T2",
    "simple_avg_delta_orth": "SimpleAvg + DeltaTrace",
    "simple_avg_delta_orth_kd_T1": "SimpleAvg + DeltaTrace + KD T1",
    "simple_avg_delta_orth_kd_T2": "SimpleAvg + DeltaTrace + KD T2",
    "simple_avg_factor_orth": "SimpleAvg + FactorOrth",
    "simple_avg_factor_orth_kd_T1": "SimpleAvg + FactorOrth + KD T1",
    "simple_avg_factor_orth_kd_T2": "SimpleAvg + FactorOrth + KD T2",
    "rank_extension": "RankExt",
    "rank_extension_kd_only_T1": "RankExt + KD T1",
    "rank_extension_kd_only_T2": "RankExt + KD T2",
    "rank_extension_orth_delta_trace_lam_50": "RankExt + DeltaTrace",
    "rank_extension_orth_delta_trace_lam_50_kd_T1": "RankExt + DeltaTrace + KD T1",
    "rank_extension_orth_delta_trace_lam_50_kd_T2": "RankExt + DeltaTrace + KD T2",
    "rank_extension_orth_factor_lam_50": "RankExt + FactorOrth",
    "rank_extension_orth_factor_lam_50_kd_T1": "RankExt + FactorOrth + KD T1",
    # RENAMED (2026-08-25): key "..._lam_50_kd_T2" -> "..._lam_15_kd_T2" (see
    # METHODS_TO_RUN's comment); display name text unchanged (still just
    # "RankExt + FactorOrth + KD T2" -- the lambda value was never part of the
    # human-readable display string, only the internal identifier). RENAMED
    # BACK (2026-08-25, FULL-STRENGTH COMBINED EXPERIMENT): key is
    # "..._lam_50_kd_T2" again, matching its restored effective lambda=50.0
    # (see COMBINED_LOSS_SCALE_ENABLED's own comment). Display text still
    # unchanged.
    "rank_extension_orth_factor_lam_50_kd_T2": "RankExt + FactorOrth + KD T2",
}

METHOD_ALIAS_NAME_MAP = {
    "simple_avg": "simple_avg",
    "simple_avg_kd_T2": "simple_avg_kd_T2",
    "simple_avg_factor_orth": "simple_avg_factor_orth_lam_50",
    "simple_avg_factor_orth_kd_T2": "simple_avg_factor_orth_lam_50_kd_T2",
    "rank_extension": "rank_extension",
    "rank_extension_kd_only_T2": "rank_extension_kd_only_T2",
    "rank_extension_orth_factor_lam_50": "rank_extension_orth_factor_lam_50",
    # RENAMED (2026-08-25): "..._lam_50_kd_T2" -> "..._lam_15_kd_T2" (both key
    # and value) -- see METHODS_TO_RUN's comment for the full rename. RENAMED
    # BACK (2026-08-25, FULL-STRENGTH COMBINED EXPERIMENT): both key and value
    # are "..._lam_50_kd_T2" again -- see COMBINED_LOSS_SCALE_ENABLED's own
    # comment.
    "rank_extension_orth_factor_lam_50_kd_T2": "rank_extension_orth_factor_lam_50_kd_T2",
}

SUPERVISOR_SELECTED_METHOD_SPECS = [
    {
        "internal_method_name": "simple_avg",
        "supervisor_requested_name": "simple_avg",
        "display_name": "SimpleAvg",
        "family": "simple_avg",
        "factor_lambda": 0.0,
        "kd_temperature": 0.0,
        "kd_weight": 0.0,
    },
    {
        "internal_method_name": "rank_extension",
        "supervisor_requested_name": "rank_extension",
        "display_name": "RankExt",
        "family": "rank_extension",
        "factor_lambda": 0.0,
        "kd_temperature": 0.0,
        "kd_weight": 0.0,
    },
    {
        "internal_method_name": "simple_avg_factor_orth",
        "supervisor_requested_name": "simple_avg_factor_orth_lam_50",
        "display_name": "SimpleAvg + FactorOrth",
        "family": "simple_avg",
        "factor_lambda": 50.0,
        "kd_temperature": 0.0,
        "kd_weight": 0.0,
    },
    {
        "internal_method_name": "rank_extension_orth_factor_lam_50",
        "supervisor_requested_name": "rank_extension_orth_factor_lam_50",
        "display_name": "RankExt + FactorOrth",
        "family": "rank_extension",
        "factor_lambda": 50.0,
        "kd_temperature": 0.0,
        "kd_weight": 0.0,
    },
    {
        "internal_method_name": "simple_avg_kd_T2",
        "supervisor_requested_name": "simple_avg_kd_T2",
        "display_name": "SimpleAvg + KD T2",
        "family": "simple_avg",
        "factor_lambda": 0.0,
        "kd_temperature": 2.0,
        "kd_weight": float(KD_WEIGHT),
    },
    {
        "internal_method_name": "rank_extension_kd_only_T2",
        "supervisor_requested_name": "rank_extension_kd_only_T2",
        "display_name": "RankExt + KD T2",
        "family": "rank_extension",
        "factor_lambda": 0.0,
        "kd_temperature": 2.0,
        "kd_weight": float(KD_WEIGHT),
    },
    {
        # STALE-METADATA FIX (2026-08-25, found during FULL-STRENGTH COMBINED
        # EXPERIMENT source audit): factor_lambda/kd_weight below were
        # HARDCODED literals (50.0 / float(KD_WEIGHT), i.e. always reading the
        # UNSCALED globals) ever since COMBINED_LAMBDA_ORTH_SCALE/COMBINED_
        # KD_WEIGHT_SCALE were introduced -- unlike the rank_extension combined
        # entry just below, which already computed its true scaled values.
        # This meant supervisor_method_mapping.csv reported this method as
        # kd=1.0/lambda=50.0 even while it was actually training at kd=0.5/
        # lambda=15.0 under the pair-4 rescaling (a real, pre-existing
        # metadata bug, not introduced by this edit). Both fields are now
        # computed expressions, matching the rank_extension entry's pattern --
        # correct regardless of COMBINED_LOSS_SCALE_ENABLED's value, today
        # (False -> 50.0/1.0, matching the actual FULL-STRENGTH training
        # config) and in the future if the scaling is ever re-enabled (True ->
        # 15.0/0.5 again, automatically, with no dict edit needed).
        "internal_method_name": "simple_avg_factor_orth_kd_T2",
        "supervisor_requested_name": "simple_avg_factor_orth_lam_50_kd_T2",
        "display_name": "SimpleAvg + FactorOrth + KD T2",
        "family": "simple_avg",
        "factor_lambda": float(LAMBDA_ORTH) * float(COMBINED_LAMBDA_ORTH_SCALE if COMBINED_LOSS_SCALE_ENABLED else 1.0),
        "kd_temperature": 2.0,
        "kd_weight": float(KD_WEIGHT) * float(COMBINED_KD_WEIGHT_SCALE if COMBINED_LOSS_SCALE_ENABLED else 1.0),
    },
    {
        # RENAMED (2026-08-25): "..._lam_50_kd_T2" -> "..._lam_15_kd_T2" (both
        # internal_method_name and supervisor_requested_name) -- see
        # METHODS_TO_RUN's comment for the full rename. factor_lambda also
        # corrected 50.0 -> a computed expression here: this field was
        # hardcoded to the UNSCALED LAMBDA_ORTH value even before the rename
        # (a pre-existing, separate inaccuracy in this spec list, not
        # introduced by the rename) -- this method's true effective
        # lambda_orth (LAMBDA_ORTH * COMBINED_LAMBDA_ORTH_SCALE = 50.0 * 0.3)
        # was 15.0 under the pair-4 rescaling; kd_weight below had the
        # identical SCALED-vs-UNSCALED issue (float(KD_WEIGHT) read the global
        # 1.0, not this method's actual scaled 0.5) and was corrected
        # alongside factor_lambda here, same root cause, for consistency
        # within this one dict entry.
        #
        # RENAMED BACK + RE-VERIFIED (2026-08-25, FULL-STRENGTH COMBINED
        # EXPERIMENT): identifiers are "..._lam_50_kd_T2" again, matching
        # COMBINED_LOSS_SCALE_ENABLED=False's restored effective lambda=50.0.
        # factor_lambda is now a computed expression (mirroring kd_weight's
        # existing pattern below) rather than a hardcoded 15.0 literal --
        # hardcoding it would have silently reintroduced the exact same
        # stale-metadata bug this comment describes, just with the wrong
        # constant baked in this time. The per-method ACTIVE_METHOD_CONFIGS/
        # hyperparameters_by_method.json rows remain the authoritative source
        # for both fields' true resolved values regardless.
        "internal_method_name": "rank_extension_orth_factor_lam_50_kd_T2",
        "supervisor_requested_name": "rank_extension_orth_factor_lam_50_kd_T2",
        "display_name": "RankExt + FactorOrth + KD T2",
        "family": "rank_extension",
        "factor_lambda": float(LAMBDA_ORTH) * float(COMBINED_LAMBDA_ORTH_SCALE if COMBINED_LOSS_SCALE_ENABLED else 1.0),
        "kd_temperature": 2.0,
        "kd_weight": float(KD_WEIGHT) * float(COMBINED_KD_WEIGHT_SCALE if COMBINED_LOSS_SCALE_ENABLED else 1.0),
    },
    # ORTH-LAMBDA INTERACTION ABLATION (2026-08-22, job 4914807 follow-up)
    # REMOVED 2026-08-23: rank_extension_orth_factor_lam_25_kd_T2 spec entry
    # deleted -- job 4915286 showed all_seen 70.58 vs the lam_50 flagship's
    # 70.62 (-0.04pp, noise). See RANKEXT_PROJECTED_PROTECT_METHODS above.
    #
    # STRUCTURAL NULL-SPACE RANKEXT (R6 roadmap Stage 2): 3 spec entries for
    # rank_extension_nullspace / _kd_only_T2 / _orth_factor_lam_50_kd_T2 were
    # added here 2026-08-23 and REMOVED 2026-08-24 after job 4917775 --
    # accuracy/representation benefit was noise-level, see git history at
    # commit 8153008 for the removed implementation.
]
SUPERVISOR_SELECTED_INTERNAL_METHODS = [
    spec["internal_method_name"] for spec in SUPERVISOR_SELECTED_METHOD_SPECS
]
SUPERVISOR_SELECTED_DISPLAY_NAMES = [
    spec["display_name"] for spec in SUPERVISOR_SELECTED_METHOD_SPECS
]


ORTH_LOSS_TYPE = "trace"
ORTH_SCALE_MODE = "squared_trace"
ORTH_TARGET_RATIO = 0.0
ORTH_LAMBDA_MIN = 1e-6
ORTH_LAMBDA_MAX = 1e3
ORTH_EPS = 1e-12
ORTH_LOSS_LOG_EVERY = 1
USE_IPC_CONSTRAINT = False
LAMBDA_IPC = 0.0
IPC_TOP_P = 0.10
IPC_IMPORTANCE_NUM_BATCHES = 8

LAMBDA_ORTH_TRACE_LIST = [LAMBDA_ORTH_DELTA_TRACE]
LAMBDA_ORTH_NORM_LIST = [500.0]
ORTH_NORM_EPS = 1e-12

ORTH_CONFIG_SWEEP = []
ENABLE_RANKEXT_ORTH_CONFIG_SWEEP = False
ORTH_DIAGNOSTICS = True
RANKEXT_DIAGNOSTICS = True


# PROTOCOL-DEPTH VALIDATION (2026-08-24): 5x20's schedule was
# [16,32,48,64,80] -- +16 rank per +20-class step, i.e. 0.8 rank units per
# class. 20x5 used +4 rank per +5-class step (SAME 0.8 ratio); this is the
# INVERSE-DEPTH leg, 4x25, using +20 rank per +25-class step -- again the
# SAME 0.8 rank-units-per-class ratio, deliberately preserved across all
# three protocols so the experiment manipulates incremental DEPTH (block
# count / block age) only, not final adapter capacity: this schedule ends
# at total_rank=80, identical to 5x20's and 20x5's final rank (verified by
# the LORA_R==RANKEXT_RANK_SCHEDULE[-1] assert below, unchanged). At
# fine-step 4 (the FINAL step) the cumulative rank is exactly 80 and
# classes-seen is exactly 100 -- bit-identical to 5x20's step5 and 20x5's
# step20 -- giving the one exact classes-seen AND cumulative-rank match
# across all three protocols (see protocol_depth_macro_checkpoint_
# comparison.csv). Fine-steps 1-3 (25/50/75 classes, rank 20/40/60) do NOT
# match any 5x20 checkpoint exactly (20/40/60/80 classes do not divide
# evenly by 25) -- do not treat them as equivalent to 5x20 steps 1-3; see
# the corresponding_5x20_step column, which is left NaN for those rows on
# purpose. Do NOT extend this to a 320-final-rank schedule (that would
# confound protocol depth with a 4x capacity increase) and do NOT retune
# the 0.8 ratio -- this is a controlled protocol-only change.
# EXPERIMENT 1 OVERRIDE: canonical 4x25 schedule [20,40,60,80] -> the
# pre-fix 5x20 reference's OWN schedule [16,32,48,64,80] (recovered from
# thesis_agent/sources/R6_catalog.md's 5x20 run-config field
# "lora_rank_schedule (rankext) = [16,32,48,64,80]", not guessed). Final
# cumulative rank stays 80, matching LORA_R=80 below -- this is the "same
# rank as before the fix" the supervisor asked for.
RANKEXT_RANK_SCHEDULE = [16, 32, 48, 64, 80]
RANKEXT_ALPHA_PER_RANK = 2.0

# ACCURACY-PUSH CANDIDATE (flag; now ON -- see "CAPACITY TEST" note below):
# wider per-step rank budget. The default schedule gives each CL step only 16
# fresh trainable ranks per target module, vs simple_avg retraining its full
# LORA_R=80 ranks from scratch every step. analysis_rankext_firststep/table1
# shows rank_extension trailing simple_avg by a wide margin on
# later_steps/all_seen in BOTH runs, consistent with (in addition to, not
# instead of, the forgetting-time issue documented in that report's Target 1c)
# a plain capacity bottleneck. This was originally an INDEPENDENT lever from
# RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED below, kept off together with it so a
# rerun could attribute any change to a single cause, per the report's
# "attribute honestly" standard.
# Parameter cost: final total_rank 160 vs 80 -- roughly doubles trainable
# rank-extension LoRA parameters per step by the last CL step (2 target
# modules x wider rank, still less total than the calibfix run's 4 modules x
# narrow rank, but meaningfully more than the 2-module/narrow-rank BASELINE
# config that actually scored 68.0). Watch train_val_gap_by_method.csv
# (overfitting score) if this is enabled -- R3/R4 already flagged mild
# overfitting signatures at the smaller capacity setting.
#
# CAPACITY TEST (next run, code-only change, no training this session):
# USE_RANKEXT_RANK_SCHEDULE_WIDE flipped OFF -> ON for the NEXT run, per
# explicit supervisor-relevant request, to directly test whether
# rank_extension's underperformance vs simple_avg is (at least partly) a
# capacity bottleneck: rank_extension has always been capped at final rank 80
# (2 target modules) while simple_avg gets full rank 80 over 4 target
# modules. This run intentionally gives rank_extension MORE total LoRA
# parameters than simple_avg (schedule [32,64,96,128,160] vs simple_avg's
# fixed 80) -- that asymmetry is the point of the test, not an oversight; see
# the "more parameters than simple_avg" note in CHANGES.md.
# NOTE this is no longer isolated from RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED
# (already ON, adopted from the strict-review run) -- unlike the original
# "one lever per run" plan, this run combines the capacity change with the
# already-adopted warmup fix, so any accuracy change here should be
# attributed to "wide schedule, on top of warmup" rather than to capacity in
# isolation. LAMBDA_ORTH is explicitly NOT rescaled (see CHANGES.md) --
# rank_extension's factor-orth loss is already 20-300x smaller than
# simple_avg's at the default rank, and rescaling lambda here would confound
# the capacity test with an orthogonality-strength change.
#
# CAPACITY TEST RESULT (2026-07-21, analysis_recency_fix/report.txt Task A.3):
# REVERTED OFF. The WIDERANK run (R3/R5/results_widerank_20260721_light) vs
# the STRICT run (R3/results_strict_20260717_light, identical config except
# this one flag) isolates the capacity change cleanly -- both runs have
# RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED=True, so the delta below is attributable
# to rank-schedule width alone. all_seen_accuracy deltas (STRICT -> WIDERANK):
# rank_extension +2.07 (21.82->23.89), rank_extension_orth_factor_lam_50
# -1.16 (31.21->30.05), rank_extension_kd_only_T2 -1.74 (58.85->57.11),
# rank_extension_orth_factor_lam_50_kd_T2 -1.81 (65.34->63.53) -- net -0.66
# average across the 4 rank_extension variants, 3 of 4 methods WORSE. Doubling
# trainable rank capacity did not help; per report.txt Task A.1/A.2, the
# bottleneck this run's low OPEN-argmax numbers actually measure is classifier
# recency bias (see RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED above), not
# representational capacity -- confirming the report's prediction before this
# revert. Reverted to the default (narrow, [16,32,48,64,80]) schedule so the
# NEXT run isolates FIX 1 (the calibration change) as the only lever versus
# the STRICT baseline, per the "one lever per run, attribute honestly"
# standard this codebase has held itself to throughout.
# PROTOCOL-DEPTH VALIDATION (2026-08-24): USE_RANKEXT_RANK_SCHEDULE_WIDE is
# False and has been for every R6 job since the revert noted above -- this
# schedule is DEAD CODE for training purposes, never selected by
# active_rankext_rank_schedule(). It is resized to NUM_STEPS=4 entries here
# (was 20, for the 20x5 job) ONLY to satisfy this file's own length/
# monotonicity asserts just below (they run unconditionally at import time
# regardless of the flag); resized by the exact same "2x the active default
# schedule, elementwise" relationship the original 5-entry version had
# (compare [16,32,48,64,80] to [32,64,96,128,160] above -- exactly 2x per
# entry), applied to the new 4-entry default RANKEXT_RANK_SCHEDULE
# ([20,40,60,80] -> [40,80,120,160]). Not a new scientific setting -- still
# never used while the flag stays False.
# EXPERIMENT 1 OVERRIDE: resized to NUM_STEPS=5 entries (was 4, for 4x25) to
# satisfy this file's unconditional length/monotonicity asserts below -- the
# flag stays False, so this list is still dead code for training purposes.
# Same "2x the active default schedule, elementwise" relationship as always.
RANKEXT_RANK_SCHEDULE_WIDE = [32, 64, 96, 128, 160]
# WIDE RANKEXT CAPACITY-SENSITIVITY EXPERIMENT (2026-09-02, supervisor-requested
# control): flipped False -> True to activate the pre-existing wide schedule
# [40,80,120,160] for the 4x25 protocol. This is a capacity-INCREASED /
# capacity-sensitivity control (each incremental step now appends a rank-40 new
# block instead of rank-20; cumulative rank 40/80/120/160 instead of
# 20/40/60/80), run to test whether more RankExt capacity narrows the gap to
# SimpleAvg. It is NOT "fully capacity-matched" to SimpleAvg. The default
# RANKEXT_RANK_SCHEDULE=[20,40,60,80] literal above is UNCHANGED and remains
# available (flip this flag back to False to restore the canonical config
# byte-for-byte). Effective LoRA scaling is UNCHANGED at 2.0 for every block at
# every step: GrowingRankLoRALinear.scaling = (RANKEXT_ALPHA_PER_RANK *
# total_rank) / total_rank = RANKEXT_ALPHA_PER_RANK = 2.0, independent of
# total_rank, so widening the schedule does not rescale frozen or new blocks.
# All four active RankExt variants pick this up automatically via
# active_rankext_rank_schedule(); SimpleAvg (rank=80, alpha=160) is untouched.
#
# REVERTED FOR CANONICAL 3-SEED REPLICATION (2026-09-06): the wide-schedule
# control experiment (R6-16, job 4933319) is complete and analyzed; this flag
# is restored to False so active_rankext_rank_schedule() resolves back to the
# canonical RANKEXT_RANK_SCHEDULE = [20, 40, 60, 80] (traced below, not just
# this constant) for the seed-123/seed-2026 canonical replication runs.
USE_RANKEXT_RANK_SCHEDULE_WIDE = False
# PROBE: this checks the WIDE schedule definition's own length (5, matching
# the canonical 5x20 protocol it was designed for), not this probe's reduced
# NUM_STEPS=3 -- WIDE is unused here (False) either way; only the schedule
# actually indexed by the (reduced) driver loop is RANKEXT_RANK_SCHEDULE,
# which is untouched and long enough (5 >= 3) regardless.
assert len(RANKEXT_RANK_SCHEDULE_WIDE) == 5
assert all(RANKEXT_RANK_SCHEDULE_WIDE[i] > RANKEXT_RANK_SCHEDULE_WIDE[i - 1] for i in range(1, NUM_STEPS))


def active_rankext_rank_schedule():
    """Resolves to RANKEXT_RANK_SCHEDULE_WIDE when USE_RANKEXT_RANK_SCHEDULE_WIDE
    is on, else the default RANKEXT_RANK_SCHEDULE. Single source of truth for
    every consumer (rank-triplet computation, reporting columns, hyperparameter
    dumps) so the flag can't drift out of sync between training and reporting."""
    return list(RANKEXT_RANK_SCHEDULE_WIDE) if USE_RANKEXT_RANK_SCHEDULE_WIDE else list(RANKEXT_RANK_SCHEDULE)


# PROTOCOL LOCK -- hard assertion (task brief Section 1).
assert active_rankext_rank_schedule() == [16, 32, 48, 64, 80], (
    f"RankExt rank schedule must be [16,32,48,64,80], got {active_rankext_rank_schedule()}"
)


def active_rankext_lora_alpha():
    """Effective LoRA alpha implied by RANKEXT_ALPHA_PER_RANK at the active
    schedule's FINAL rank -- i.e. the rank_extension-family analogue of the
    global LORA_ALPHA constant (which only ever describes simple_avg's fixed
    rank=80 config; see CHANGES.md "family-conditional alpha reporting fix").
    GrowingRankLoRALinear.scaling = RANKEXT_ALPHA_PER_RANK is constant at
    EVERY step regardless of total_rank (rankext_alpha = ALPHA_PER_RANK *
    total_rank, scaling = rankext_alpha / total_rank = ALPHA_PER_RANK), so the
    2:1 alpha/rank ratio this returns is preserved at every growth step, not
    just the final one -- this is a reporting convenience, not a separate
    scaling path."""
    return float(RANKEXT_ALPHA_PER_RANK) * float(active_rankext_rank_schedule()[-1])


# ACCURACY-PUSH CANDIDATE (flag, default OFF): ramp lambda_orth from 0 up to
# its full configured value linearly over the first
# RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS epochs of EACH CL step's local training
# (self.state.epoch resets to ~0 at the start of every step's own Trainer, so
# this is a per-step ramp, not a single ramp over the whole run), instead of
# applying it at full strength from local_epoch 0.
#
# Evidence (analysis_rankext_firststep/table4_factororth_trajectory_stats.csv
# and table4b): for rank_extension_orth_factor_lam_50, train_ce_loss at
# local_epoch==1 of every step transition (step>1) is consistently 1.5-3.4x
# higher than plain rank_extension's at the same point, and this ratio GROWS
# with step index (1.47x at step2 -> 3.37x at step5 in the calibfix run) --
# i.e. the orth penalty is punishing the fresh, barely-trained new rank block
# before it has had a chance to fit the task, and this gets worse as more
# frozen blocks accumulate. table4b separately shows END-of-step convergence
# (val_ce) is never worse for the orth variant than for plain rank_extension
# at any step in either run -- so the penalty isn't damaging final per-step
# fit, only this specific early-epoch transient. And the raw orth violation
# itself decays 2-4 orders of magnitude within 1-2 epochs on its own (table4:
# e.g. NEW step 5 weighted orth 33.0 at epoch1 -> 0.08 at epoch2), so a short
# ramp should be able to skip past the worst of the spike -- while the
# constraint is already both small and evidently harmless by the time it
# would re-engage at full strength.
# STRICT-REVIEW (B3, 2026-07-17): re-checked directly against
# training_loss_history_by_epoch.csv from THIS analysis's NEW/revert run
# (analysis_strict_review/report.txt Part B3) before flipping this on:
#   train_ce_loss at local_epoch==1, orth variant vs plain rank_extension,
#   same step: step2 1.64x, step3 1.89x, step4 2.31x, step5 2.04x -- the same
#   growing-with-step-index pattern documented above, confirmed to persist in
#   the reverted 2-module/head-lr-x1/no-calibration baseline (not an artifact
#   of the since-rolled-back 4-module/calibfix settings).
#   factor_orth_loss_weighted collapses within ONE epoch at every step in
#   this run too: step2 9.94->0.026, step3 18.81->0.046, step4 29.12->0.068,
#   step5 36.15->0.079 (99.7-99.8% drop by local_epoch==2) -- confirms a
#   1-epoch warmup window is long enough to skip past essentially all of the
#   spike while leaving the (already small, per table4_factororth_trajectory_
#   stats.csv) steady-state regularization untouched.
# Turned ON as the ONE lever for this run, specifically to chase
# RankExt+FactorOrth (non-KD) and RankExt+FactorOrth+KD back toward/above the
# family's historical 68.0 ceiling. USE_RANKEXT_RANK_SCHEDULE_WIDE stays False
# (see its own comment above) -- deliberately NOT enabled alongside this, so
# any change in the next run's rank_extension numbers can be attributed to
# warmup alone, per the "attribute honestly, one lever per run" standard this
# analysis is holding itself to. Still an untested hypothesis, not a
# confirmed fix -- BASELINE (this flag OFF) remains the config that actually
# produced rank_extension's proven 68.0 result; verify against the rerun.
RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED = True
RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS = 1.0

# ACCURACY-PUSH CANDIDATE (analysis_rankext_plain/candidate_evaluation.txt,
# 2026-07-23): new-rank-block OUTPUT warmup for rank_extension. Ramps the
# newly-added LoRA block's CONTRIBUTION TO THE FORWARD PASS (not its weights,
# not the frozen old block, not the classifier head) from 0 up to full
# strength linearly over the first RANKEXT_NEW_BLOCK_WARMUP_EPOCHS epochs of
# EACH CL step's local training, reusing the exact same ramp shape/formula as
# RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED just above (orth_lambda_warmup_multiplier(),
# called as-is with a new, independent flag/epochs pair).
#
# Rationale (analysis_rankext_plain/diagnosis.txt): plain rank_extension's
# restricted step-1 accuracy falls from a well-converged post-step-1 state
# (val_ce_loss=0.2266, identical to RankExt+KD's step-1 model bit-for-bit,
# since KD only starts biting at step 2) down to 57.7% by final eval, while
# RankExt+KD -- same step-1 model, ONLY steps 2-5 differ -- holds 94.35% on
# the exact same eval. Frozen old LoRA blocks (0.0 max abs diff, every layer,
# every step, all 4 variants -- classifier_row_norm_diagnostics_by_method_
# step.csv / *_frozen_rank_blocks.csv) and frozen classifier rows (doubly-
# redundant gradient-mask + hard-restore mechanism, independently verified
# from source) are both confirmed NOT drifting, ruling out weight corruption
# as the cause. The remaining explanation: the new block's earliest, least-
# informed gradient updates (B_new is zero-initialized every step, so the
# very first few batches set its initial direction from scratch) perturb the
# SHARED q/v-proj forward computation in a way that is disruptive to how the
# frozen old block's fixed contribution combines with the (also frozen) old
# classifier rows at eval time -- with nothing anchoring the new block toward
# cooperating with what is already there, unlike the KD/FactorOrth variants,
# which each supply exactly that anchor via a different mechanism. This
# warmup does not add any new anchoring signal itself -- it only slows the
# RATE at which the new block's raw, uninformed early updates can perturb the
# shared forward pass, on the theory (well precedented by RANKEXT_ORTH_
# LAMBDA_WARMUP_ENABLED's own evidence just above: local_epoch==1 train_ce_
# loss 1.5-3.4x higher than plain rank_extension's, decaying 2-4 orders of
# magnitude within 1-2 epochs) that the first ~1 epoch of any newly-added
# rank_extension mechanism's training is a uniquely volatile, poorly-
# conditioned transient. Applied identically to ALL FOUR rank_extension
# variants (plain, +FactorOrth, +KD, +FactorOrth+KD) -- see
# family_uses_new_block_warmup() and its one call site in
# run_rank_extension_variant() -- never to simple_avg (GrowingRankLoRALinear
# is the only class that ever reads this multiplier; simple_avg's PEFT LoRA
# layers never do, so simple_avg is unaffected by construction, not just by a
# gate). Flag defaults ON for the next run; existing (no-warmup, full
# strength from batch 1) behavior is fully preserved when OFF.
RANKEXT_NEW_BLOCK_WARMUP_ENABLED = True
# Same value as RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS on purpose: this is the one
# other place in the codebase that already tunes "how long should a within-
# step warmup for a freshly-added rank_extension mechanism last," and that
# duration is independently justified/validated there (see the comment on
# RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS above). Reusing it avoids stacking an
# untested new duration on top of an already-untested new mechanism.
RANKEXT_NEW_BLOCK_WARMUP_EPOCHS = 1.0

# Module-level state the new-block warmup multiplier lives in. A plain dict
# (not a bare global float) so it can be imported/read/written from any scope
# without a `global` statement at every call site. ALWAYS 1.0 outside the
# narrow window of a rank_extension step's own trainer.train() call -- see
# train_with_trainer()'s unconditional reset immediately after trainer.train()
# returns, which is the single choke point every eval call in this script
# passes through afterward.
_rankext_new_block_warmup_state = {"multiplier": 1.0}


def set_rankext_new_block_warmup_multiplier(value):
    _rankext_new_block_warmup_state["multiplier"] = float(value)


def get_rankext_new_block_warmup_multiplier():
    return float(_rankext_new_block_warmup_state["multiplier"])


def family_uses_new_block_warmup(family):
    """rank_extension only -- see RANKEXT_NEW_BLOCK_WARMUP_ENABLED comment
    above for why simple_avg is excluded (GrowingRankLoRALinear, the only
    module class that reads this multiplier, is rank_extension-exclusive)."""
    return bool(RANKEXT_NEW_BLOCK_WARMUP_ENABLED) and str(family) == "rank_extension"


RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS = {
    "rank_extension_kd_only_T2",
    # RENAMED (2026-08-25): "..._lam_50_kd_T2" -> "..._lam_15_kd_T2" -- see
    # METHODS_TO_RUN's own comment for the full rename. Behavior unchanged:
    # this method's new-block warmup stays disabled, only the identifier's
    # spelling changed (renaming it would otherwise have silently RE-ENABLED
    # new-block warmup for this method by dropping it out of this set).
    # RENAMED BACK (2026-08-25, FULL-STRENGTH COMBINED EXPERIMENT): key is
    # "..._lam_50_kd_T2" again -- same caution applies, same unchanged
    # behavior (new-block warmup stays disabled for this method).
    "rank_extension_orth_factor_lam_50_kd_T2",
    # FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT (task brief Section 20): the
    # two NEW full-KD RankExt arms disable new-block output warmup here,
    # matching Exp1's own treatment of its two historical KD-bearing RankExt
    # methods exactly.
    "rank_extension_fullkd_T2_protect30",
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
}


def method_rankext_new_block_warmup_epochs(method_name, family):
    """Resolve rank_extension new-block warmup per method.

    Keep the existing 1-epoch warmup for non-KD rank_extension methods, but
    disable it only for the two active T2 KD rank_extension methods.
    """
    if not family_uses_new_block_warmup(family):
        return 0.0
    if str(method_name) in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS:
        return 0.0
    return float(RANKEXT_NEW_BLOCK_WARMUP_EPOCHS)


# One row per (method, step, local_epoch) actually applied during training --
# see RankExtNewBlockWarmupCallback below and its CSV write near the other
# diagnostic tables (best_epoch_selection_rows / growing_overfitting_rows).
rankext_new_block_warmup_diagnostic_rows = []


def orth_lambda_warmup_multiplier(epoch_val, warmup_epochs, enabled):
    """Linear 0->1 ramp over the first `warmup_epochs` epochs of local
    (per-CL-step) training; always 1.0 (full strength, no-op) when `enabled`
    is False, `warmup_epochs` <= 0, or `epoch_val` is unavailable/NaN. Shared
    by both the rank_extension (Objective 1) and simple_avg-combined
    (Objective 2) warmup mechanisms below -- same formula, independently
    gated per call site."""
    if not enabled:
        return 1.0
    warmup_epochs = float(warmup_epochs)
    if warmup_epochs <= 0.0 or epoch_val is None:
        return 1.0
    epoch_val = float(epoch_val)
    if np.isnan(epoch_val):
        return 1.0
    return float(min(1.0, max(0.0, epoch_val / warmup_epochs)))


# FIX 2: restrict the active method set to EXACTLY the 8 supervisor-selected
# methods (see SUPERVISOR_SELECTED_METHOD_SPECS / SUPERVISOR_SELECTED_INTERNAL_METHODS
# below, which already listed these same 8). The delta-trace variants
# ("simple_avg_delta_orth" and "rank_extension_orth_delta_trace_lam_50", both
# trained in the previous run) are disabled here -- NOT deleted -- by flipping
# their base_method flags to False; build_active_method_configs() below simply
# skips add_method() calls whose base_method flag is False (see
# `if not METHODS_TO_RUN.get(base_method, False): return`), so their full
# training/orth-loss implementation (IndependentLoraOrthTrainer /
# DeltaOrthRankExtensionTrainer with orth_mode="delta_trace", etc.) is untouched
# and can be re-enabled later just by flipping these two flags back to True.
#
# A4 AUDIT (post Aug-7 crash), VERIFIED against the two actual saved run
# directories (results/..._20260723_173803 = July-23, results/
# ..._20260807_004316 = Aug-7):
#
# 1. All 10 currently-True flags below -- the 8 original supervisor methods
#    AND the 2 featanchor variants added 2026-08-05 -- are wired to train
#    FRESH, from scratch, every run; this script has no resume/cache/"skip if
#    results already exist" mechanism anywhere (no os.path.exists guard
#    around any train_with_trainer()/run_*_variant() call site;
#    BASE_OUTPUT_DIR is stamped with datetime.now() so every run writes into
#    a brand-new directory). METHODS_TO_RUN already had the correct 10-method
#    state on Aug-7, same as now -- this was never a config bug.
#
# 2. The REAL mechanism, confirmed by inspecting both saved runs' tables/:
#    this file's early per-method accumulator lists -- all_results,
#    method_summary_rows, train_diagnostic_rows (reset unconditionally at
#    module scope a few hundred lines above, NOT here) -- are then read by
#    orth_kd_train_rows/orth_kd_summary_rows via an explicit
#    `X if "X" in globals() else []` reuse guard (see just above
#    run_rank_extension_variant()'s call loop), and metrics_tables() reads
#    summary_table the same reuse-guarded way. This is a deliberate
#    notebook-safe convenience (this file began life as a Jupyter/Colab
#    notebook -- see the "# In[ ]:" cell markers throughout) so re-running a
#    handful of cells doesn't discard earlier cells' results. But it means:
#    if the Aug-7 session reused the SAME kernel as July-23 (no restart) and
#    only edited+re-ran the cells for the 2 NEW featanchor methods plus the
#    final report cells -- never re-running the top-of-file cell that resets
#    all_results/method_summary_rows/train_diagnostic_rows to [] -- every
#    accumulator silently kept its stale July-23 rows for the 8 old methods
#    and just appended the 2 new methods' rows on top. Confirmed directly:
#    tables/supervisor_selected_accuracy_comparison.csv (the EARLIER of two
#    writes to that filename in this file, the one before the post-processing
#    crash) has all 8 old methods' first_step/later_steps/all_seen numbers
#    matching the July-23 file to every trailing float digit (e.g.
#    20.080000000000002, 69.6125) -- not just "close", which independent GPU
#    retraining would essentially never produce -- while the 2 new featanchor
#    rows are distinct fresh numbers. This is carried-over notebook STATE,
#    not a training or config bug; the fix is operational, not code: restart
#    the kernel/runtime (a true fresh Python process) before this rerun, so
#    the module-level `all_results = []` / `method_summary_rows = []` /
#    `train_diagnostic_rows = []` resets actually run and every method trains
#    into genuinely empty accumulators.
#
# 3. Independently real and ALSO confirmed from the same two directories:
#    A1's color_map KeyError crashed post-processing at the
#    "15_supervisor_selected_train_val_ce.png" cell. Aug-7's plots/ has only
#    08-14 (nothing from 15 onward); its tables/ is missing
#    final_metrics_all_methods.csv, validation_diagnostics_by_method.csv, and
#    every other CSV metrics_tables()/valdiag()/HP write (all post-crash in
#    the old ordering); reports/ and logs/ are empty. July-23's directory has
#    all of the above plus 34 plots and 5 populated reports. So even with a
#    fresh kernel, the Aug-7 job would ALSO have lost its own final numbers
#    to this crash -- A1 (color_map fix) + A2 (these CSVs now write before
#    any plot, each plot try/excepted) fix that independently of point 2.
#
# FINAL THESIS COMPARISON (2026-08-25, post-KD-weight closure): the 8
# principal supervisor-selected methods (SUPERVISOR_SELECTED_METHOD_SPECS
# above -- simple_avg, simple_avg_kd_T2, simple_avg_factor_orth,
# simple_avg_factor_orth_kd_T2, rank_extension, rank_extension_kd_only_T2,
# rank_extension_orth_factor_lam_50, rank_extension_orth_factor_lam_50_kd_T2 --
# briefly "..._lam_15_kd_T2" under the pair-4 fairness rescaling, RENAMED BACK
# 2026-08-25 for the FULL-STRENGTH COMBINED EXPERIMENT, see METHODS_TO_RUN's
# comment)
# are ALL reactivated together here for the final canonical 4x25 comparison,
# at the settled KD_WEIGHT=1.0. The 4 simple_avg methods were previously kept
# deactivated only to reduce runtime during the RankExt-only protocol-depth /
# KD-weight experiments (BASELINE RESTORED 2026-08-24 and FINAL KD-WEIGHT
# EXPERIMENT notes below, both now historical) -- reactivating them here
# introduces no new mechanism, only restores execution of the existing,
# previously-exercised simple_avg code path (family-conditional target
# modules/head-LR/calibration -- see TARGET_MODULES_BY_FAMILY /
# HEAD_LR_MULTIPLIER_BY_FAMILY / CALIBRATION_MODE_BY_FAMILY -- are all
# preserved unchanged). Non-KD FactorOrth (both families) and the 2 KD
# families are reactivated together too, matching the full canonical 8. The
# 2 delta-trace variants (simple_avg_delta_orth, rank_extension_orth_
# delta_trace_lam_50, plus their _kd siblings) stay OUT -- excluded from the
# 8-method set per FIX 2 below, not part of this comparison. Historical
# baseline notes retained: the structural Null-Space RankExt v2 experiment
# (rank_extension_nullspace / _kd_only_T2 / _orth_factor_lam_50_kd_T2) is
# CLOSED -- job 4917775 showed accuracy/representation benefit at noise level
# (flagship all_seen +0.11pp, with a -0.90pp step1 and -2.75pp step5
# regression) despite the structural projection itself being enforced
# essentially exactly. Its 3 methods and all supporting machinery have been
# fully removed from this file (see git history at commit 8153008 for the
# removed implementation) and are not part of this comparison.
# Execution-selection only -- every implementation/config-construction path
# below is untouched and can be reactivated by flipping these booleans back,
# same as every prior disable in this dict.
METHODS_TO_RUN = {
    # JOINT PACK: SA-0 and RE-0 stay/become True (canonical same-run
    # baselines). simple_avg_kd/factor_orth/factor_orth_kd and
    # rank_extension_kd_only/orth_factor_lam_50/orth_factor_lam_50_kd are
    # False -- deliberately NOT retrained (historical 9-epoch R7 values are
    # reused as clearly-labeled secondary references only -- see the header
    # comment and the readiness report). RankExt IS in scope for this joint
    # job (unlike the earlier SimpleAvg-only pack) -- rank_extension=True and
    # 3 new rank_extension_*_oldseen*/dense_orth* flags below.
    # PROBE: SimpleAvg entirely OFF -- this probe is RankExt-only (spec:
    # "Do NOT run SimpleAvg").
    "simple_avg": False,
    "simple_avg_kd": False,
    "simple_avg_delta_orth": False,  # disabled for FIX 2 -- was True; delta-trace excluded from the 8-method set
    "simple_avg_delta_orth_kd": False,
    "simple_avg_factor_orth": False,
    "simple_avg_factor_orth_kd": False,
    # FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT: job 4970580's weight=0.5,
    # no-warmup SimpleAvg KD arms are REPLACED by this script's weight=1.0,
    # 1-epoch-warmup arms (the current pipeline's ORIGINAL job-4969059
    # default, known best ~76.66%) -- flipped False so those weight=0.5
    # arms are never retrained here.
    "simple_avg_kd_oldseen": False,
    # SA-2 (task brief Section 9): old-seen KD, T=2, weight=1.0, 1-epoch
    # warmup. Re-enabled from its job-4970580-era False.
    # PROBE: all SimpleAvg arms OFF (see "simple_avg" flag above).
    "simple_avg_kd_oldseen_warmup": False,
    "simple_avg_kd_oldseen_T4_warmup": False,
    "simple_avg_dense_orth": False,
    "simple_avg_dense_orth_kd_oldseen_warmup": False,
    # FINAL 9-METHOD SCRIPT: job 4970580's weight=0.5 variants are not part
    # of this script's 9-method set -- disabled.
    "simple_avg_kd_oldseen_weight0p5": False,
    "simple_avg_dense_orth_kd_oldseen_weight0p5": False,
    # PROBE: plain rank_extension (no FactorOrth/KD) is OFF -- out of scope
    # for this probe (spec section 4 tests only the FactorOrth-based arms).
    "rank_extension": False,
    "rank_extension_kd_only": False,
    "rank_extension_orth_delta_trace_lam_50": False,  # disabled for FIX 2 -- was True; delta-trace excluded from the 8-method set
    "rank_extension_orth_delta_trace_lam_50_kd": False,
    # FINAL 8-ARM SCRIPT: the historical rank_extension_orth_factor_lam_50
    # base_method key (already present, unused, in the current pipeline) is
    # kept False here -- FactorOrth is instead activated via a NEW, distinct
    # base_method key below (rank_extension_factor_orth_lam50_new) so this
    # run's FactorOrth arm never collides with the legacy identifier's own
    # special-casing in RANKEXT_PROJECTED_PROTECT_METHODS /
    # RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS (see the implementation audit
    # report's FactorOrth-porting section for why that separation matters).
    "rank_extension_orth_factor_lam_50": False,
    "rank_extension_kd_oldseen_warmup": False,
    "rank_extension_dense_orth": False,
    "rank_extension_dense_orth_kd_oldseen_warmup": False,
    # FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT: job 4970580's old-seen-KD
    # weight=1.5 RankExt arms are NOT part of this script's 9-method set --
    # disabled. RE-3 (FactorOrth alone, task brief Section 16) is unchanged
    # from job 4970580 and stays True.
    "rank_extension_kd_oldseen_weight1p5_warmup": False,
    # PROBE: secondary arm (spec section 4, "if computationally cheap
    # enough") -- FactorOrth50 alone, no KD.
    "rank_extension_factor_orth_lam50_new": True,
    "rank_extension_factor_orth_kd_oldseen_weight1p5_warmup": False,
    # PROBE: rank_extension_fullkd_T2_protect30 (KD+protect30, no FactorOrth)
    # is OFF -- out of scope for this probe.
    "rank_extension_fullkd_T2_protect30": False,
    # PROBE PRIMARY ARM (spec section 4): FactorOrth50 + full-KD T2 +
    # protect30 combined -- the historically strongest RankExt arm and the
    # one this probe is most about. Mechanism (KD scope/temperature/weight,
    # protect_weight, FactorOrth lambda, new-block-warmup-disabled) is
    # completely UNCHANGED from the canonical script -- only method
    # *selection*, NUM_STEPS, RANKEXT_EPOCHS, and rank_extension's head-LR
    # multiplier differ in this probe.
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30": True,
    # RENAMED (2026-08-25, user directive): "..._lam_50_kd" -> "..._lam_15_kd".
    # This is pair 4's RankExt side under the STRICT-FAIRNESS REDESIGN's pair-4
    # rescaling (COMBINED_LOSS_SCALE_ENABLED, see that flag's own comment) --
    # its EFFECTIVE lambda_orth is 15.0 (LAMBDA_ORTH=50.0 * COMBINED_LAMBDA_
    # ORTH_SCALE=0.3), not 50.0, so keeping "lam_50" in its identifier would be
    # factually wrong. Renaming (not just relabeling) so the internal method
    # name itself matches its true trained lambda, matching every other
    # consumer of this identifier -- see the corresponding renames at
    # RANKEXT_PROJECTED_PROTECT_METHODS, RANKEXT_NEW_BLOCK_WARMUP_DISABLED_
    # METHODS, METHOD_DISPLAY_NAME_MAP, METHOD_ALIAS_NAME_MAP, SUPERVISOR_
    # SELECTED_METHOD_SPECS, EXPECTED_ENABLED_METHOD_FAMILIES, this dict's own
    # key just below, build_active_method_configs()'s add_method() call, and
    # the VARIANT dict / summary-table filter lists further down -- ALL
    # updated together so no lookup silently breaks. Behavior was UNCHANGED
    # by that rename: same True/False state, same lambda_orth_scale=0.3/
    # kd_weight_scale=0.5 values, just consistently spelled "lam_15"
    # everywhere.
    #
    # RENAMED BACK (2026-08-25, FULL-STRENGTH COMBINED EXPERIMENT): key is
    # "..._lam_50_kd" again, matching COMBINED_LOSS_SCALE_ENABLED=False's
    # restored lambda_orth_scale=1.0/kd_weight_scale=1.0 (see that flag's own
    # comment above LAMBDA_ORTH). Same True/False state (still True) as
    # before both renames -- only the identifier's spelling changed each time.
    # SINGLEJOB PACK: flipped False -- see the top-of-dict comment (RankExt is
    # out of scope for this SimpleAvg-only job; this flag was True in Exp1).
    "rank_extension_orth_factor_lam_50_kd": False,
    # RANK_EXT FIRST_STEP FIX (task 2 decision doc, 2026-08-17): the
    # feature-anchor lever's 4 opt-in method flags (rank_extension_featanchor,
    # rank_extension_orth_factor_featanchor, and their DEFAULT-OFF
    # "_base"/fixed_base siblings) were removed from here entirely -- the fix
    # they were exploring is now unconditional for the 2 non-KD rank_extension
    # methods above, not a separate opt-in method. See
    # RANKEXT_PRETRAINED_ANCHOR_WEIGHT for the mechanism and diagnostic
    # evidence.
    "do_merging_simple": False,
    "joint_upper_bound": False,
    "full_finetune": False,
    "seq_ft_no_replay": False,
    "simple_avg_no_replay": False,
    "simple_avg_replay": False,
    "simple_avg_orth": False,
    "do_merging_simple_orth": False,
    "orthogonal_loss": False,
    "rank_extension_replay": False,
    "rank_extension_orth": False,
    "rank_extension_replay_orth": False,
    "rank_extension_orth_trace": False,
    "rank_extension_orth_norm": False,
    "rank_extension_kd_only_old": False,
    "rank_extension_orth_trace_abs_lam_1": False,
    "rank_extension_orth_norm_lam_500": False,
    "rank_extension_orth_factor_lam_0p1": False,
    "rank_extension_orth_factor_lam_0p5": False,
    "rank_extension_orth_factor_lam_1": False,
    "rank_extension_orth_factor_lam_10": False,
    "rank_extension_orth_factor_lam_50_oldactive_true": False,
    "rank_extension_orth_factor_lam_50_oldactive_false": False,
    "rank_extension_orth_factor_lam_50_kd_oldactive_true": False,
    "rank_extension_orth_factor_lam_50_kd_oldactive_false": False,
    "rank_extension_zero_old_merge": False,
    "rank_extension_zero_old_merge_orth_delta_trace_lam_50": False,
    "rank_extension_zero_old_merge_orth_delta_trace_lam_50_kd": False,
    "rank_extension_zero_old_merge_orth_factor_lam_50": False,
    "rank_extension_zero_old_merge_orth_factor_lam_50_kd": False,
    "rank_extension_zero_old_merge_orth_trace": False,
    "rank_extension_zero_old_merge_orth_norm": False,
    "rank_extension_orth_delta_trace_lam_1": False,
    "rank_extension_zero_old_merge_orth_delta_trace_lam_1": False,
    "rank_extension_orth_delta_trace_lam_1_kd": False,
    "rank_extension_zero_old_merge_orth_delta_trace_lam_1_kd": False,
}


def kd_temperature_tag(temp):
    temp = float(temp)
    if temp.is_integer():
        return f"T{int(temp)}"
    return "T" + str(temp).replace(".", "p")


def build_active_method_configs():
    configs = []

    def add_method(method_name, family, base_method, uses_kd=False, kd_temperature=0.0, uses_delta_trace=False, uses_factor_orth=False, lambda_orth_scale=1.0, kd_weight_scale=1.0, uses_dense_orth=False, kd_class_scope="full", kd_warmup_epochs=0.0):
        if not METHODS_TO_RUN.get(base_method, False):
            return
        rankext_new_block_warmup_epochs = method_rankext_new_block_warmup_epochs(method_name, family)
        configs.append({
            "method": str(method_name),
            "family": str(family),
            "base_method": str(base_method),
            "uses_kd": bool(uses_kd),
            "kd_temperature": float(kd_temperature) if uses_kd else 0.0,
            "kd_weight": float(KD_WEIGHT if uses_kd else 0.0) * float(kd_weight_scale),
            # SINGLEJOB PACK ADDITION: "full" (Exp1's historical KD, unused by
            # any method actually enabled in this script) vs "old_seen" (Arms
            # 1/2/4, see masked_kd_loss()). kd_warmup_epochs=0.0 reproduces
            # Exp1's flat (no-warmup) KD weight exactly (Arm 1); >0 activates
            # the linear ramp (Arms 2/4).
            "kd_class_scope": str(kd_class_scope) if uses_kd else "full",
            "kd_warmup_epochs": float(kd_warmup_epochs) if uses_kd else 0.0,
            "uses_delta_trace": bool(uses_delta_trace),
            "uses_factor_orth": bool(uses_factor_orth),
            # SINGLEJOB PACK ADDITION: DenseOrth (Arms 3/4) uses the FIXED
            # DENSE_ORTH_LAMBDA=20.0 constant, never LAMBDA_ORTH=50.0 (which
            # stays reserved for the untouched, unused-here FactorOrth/
            # delta_trace code paths) -- see DENSE_ORTH_LAMBDA's own comment.
            "uses_dense_orth": bool(uses_dense_orth),
            "lambda_orth": (
                float(LAMBDA_ORTH if (uses_delta_trace or uses_factor_orth) else (DENSE_ORTH_LAMBDA if uses_dense_orth else 0.0))
                * float(lambda_orth_scale)
            ),
            # ACCURACY-PUSH CANDIDATE bookkeeping: 1.0 for every method except
            # simple_avg_factor_orth_kd_T2 when COMBINED_LOSS_SCALE_ENABLED is
            # on -- recorded explicitly so the per-method config tables make
            # the scaling visible rather than silently folding it into
            # lambda_orth/kd_weight with no trace of *why* those differ from
            # the single-penalty siblings.
            "lambda_orth_scale": float(lambda_orth_scale),
            "kd_weight_scale": float(kd_weight_scale),
            "uses_replay": False,
            "uses_zero_old": False,
            # STRICT-FAIRNESS REDESIGN: "rank" used to be unconditionally
            # int(LORA_R) for every method -- silently correct for
            # rank_extension only because of the old (now-removed) LORA_R ==
            # RANKEXT_RANK_SCHEDULE[-1] invariant. Now family-conditional so
            # this column reports each family's own true value: simple_avg's
            # own per-step rank (LORA_R) vs rank_extension's final cumulative
            # rank (its rank schedule's last entry).
            "rank": int(LORA_R) if family == "simple_avg" else int(active_rankext_rank_schedule()[-1]),
            "rank_schedule": ("fixed:" + str(LORA_R)) if family == "simple_avg" else "->".join(str(v) for v in active_rankext_rank_schedule()),
            "target_modules": ", ".join(family_target_modules(family)),
            "head_lr_multiplier": family_head_lr_multiplier(family),
            "apply_calibration": family_applies_calibration(family),
            "calibration_mode": family_calibration_mode(family) if family_applies_calibration(family) else "off",
            # analysis_rankext_plain/ (2026-07-23): rank_extension-only,
            # method-specific after R6: non-KD rank_extension keeps the
            # existing warmup, KD rank_extension disables it.
            "rankext_new_block_warmup_enabled": rankext_new_block_warmup_epochs > 0.0,
            "rankext_new_block_warmup_epochs": rankext_new_block_warmup_epochs,
        })

    add_method("simple_avg", "simple_avg", "simple_avg")

    # ==========================================================================
    # SINGLEJOB PACK ADDITION -- Arms 1-4 (see this script's top-of-file
    # header for the full design rationale; full mechanism detail in
    # thesis_agent/reports/simpleavg_singlejob_improvement_pack_preparation_audit.md).
    # Each is gated by its OWN base_method flag in METHODS_TO_RUN, exactly
    # like every other method in this file -- disabling one here (e.g. to
    # rerun a subset) requires no other change.
    # ==========================================================================
    # SA-2 (task brief Section 9): old-seen KD, T=2, weight=1.0 (kd_weight_scale
    # =1.0, the current pipeline's ORIGINAL job-4969059 default), 1-epoch
    # warmup (KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4) -- known current best
    # ~76.66% under the 6-epoch protocol. masked_kd_loss()/
    # kd_class_scope="old_seen" mechanism itself is completely unchanged.
    add_method(
        "simple_avg_kd_oldseen_T2_warmup", "simple_avg", "simple_avg_kd_oldseen_warmup",
        uses_kd=True, kd_temperature=2.0, kd_class_scope="old_seen",
        kd_warmup_epochs=KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4, kd_weight_scale=1.0,
    )
    # SA-3 (task brief Section 10): NEW -- clean T2-vs-T4 ablation. Every
    # field identical to SA-2 above EXCEPT kd_temperature=4.0. T*T scaling is
    # applied generically inside masked_kd_loss()/compute_loss() via
    # `self.kd_temperature ** 2` -- never hardcoded per-temperature here.
    add_method(
        "simple_avg_kd_oldseen_T4_warmup", "simple_avg", "simple_avg_kd_oldseen_T4_warmup",
        uses_kd=True, kd_temperature=4.0, kd_class_scope="old_seen",
        kd_warmup_epochs=KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4, kd_weight_scale=1.0,
    )
    # SA-4 (task brief Section 11): DenseOrth, lambda=20, UNCHANGED --
    # same method_name/base_method/config as job 4969059/4970580's arm (its
    # own 1-epoch dense_orth_warmup_epochs ramp, see DENSE_ORTH_WARMUP_EPOCHS
    # and IndependentLoraOrthTrainer, is preserved automatically since
    # nothing about this call or that trainer class changed).
    add_method(
        "simple_avg_dense_orth_lam20", "simple_avg", "simple_avg_dense_orth",
        uses_dense_orth=True,
    )
    # SA-5 (task brief Section 12): DenseOrth lambda=20 + old-seen KD, T=2,
    # weight=1.0, 1-epoch warmup -- SA-2 + SA-4 combined.
    add_method(
        "simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup", "simple_avg", "simple_avg_dense_orth_kd_oldseen_warmup",
        uses_kd=True, kd_temperature=2.0, kd_class_scope="old_seen",
        kd_warmup_epochs=KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4, kd_weight_scale=1.0, uses_dense_orth=True,
    )
    # ==========================================================================
    # END SINGLEJOB PACK ADDITION. Everything below (historical KD/FactorOrth/
    # delta_trace SimpleAvg variants and all RankExt variants) is Exp1's own,
    # UNCHANGED code -- gated off via METHODS_TO_RUN for this run (historical
    # SimpleAvg controls are deliberately not retrained; RankExt is out of
    # scope for this SimpleAvg-only job), never deleted.
    # ==========================================================================

    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(f"simple_avg_kd_{kd_tag}", "simple_avg", "simple_avg_kd", uses_kd=True, kd_temperature=kd_temp)
    add_method("simple_avg_delta_orth", "simple_avg", "simple_avg_delta_orth", uses_delta_trace=True)
    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(f"simple_avg_delta_orth_kd_{kd_tag}", "simple_avg", "simple_avg_delta_orth_kd", uses_kd=True, kd_temperature=kd_temp, uses_delta_trace=True)
    add_method("simple_avg_factor_orth", "simple_avg", "simple_avg_factor_orth", uses_factor_orth=True)
    # Objective 2: scaling applies ONLY to this combined call site (both KD
    # and factor-orth active at once) -- simple_avg_factor_orth above and
    # simple_avg_kd_T2 below are untouched and keep full-strength values.
    _combined_lambda_scale = float(COMBINED_LAMBDA_ORTH_SCALE) if COMBINED_LOSS_SCALE_ENABLED else 1.0
    _combined_kd_scale = float(COMBINED_KD_WEIGHT_SCALE) if COMBINED_LOSS_SCALE_ENABLED else 1.0
    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(
            f"simple_avg_factor_orth_kd_{kd_tag}", "simple_avg", "simple_avg_factor_orth_kd",
            uses_kd=True, kd_temperature=kd_temp, uses_factor_orth=True,
            lambda_orth_scale=_combined_lambda_scale, kd_weight_scale=_combined_kd_scale,
        )

    add_method("rank_extension", "rank_extension", "rank_extension")

    # ==========================================================================
    # JOINT PACK ADDITION -- RE-1..RE-3 (see this script's top-of-file header
    # for the full design rationale). Each gated by its own base_method flag
    # in METHODS_TO_RUN, exactly like every other method in this file.
    # ==========================================================================
    # RE-2 (task brief Section 15): HISTORICAL full-100-way KD (kd_class_scope
    # defaults to "full" -- NOT passed as "old_seen"), T=2, weight=1.0
    # (kd_weight_scale=1.0), kd_warmup_epochs=0.0 -- NO KD-weight warmup,
    # matching Exp1's own flat-weight convention exactly (Exp1's KD had no
    # warmup mechanism at all). protect_weight=30.0 and new-block-warmup-
    # disabled are activated below by adding this method's name to
    # RANKEXT_PROJECTED_PROTECT_METHODS / RANKEXT_NEW_BLOCK_WARMUP_DISABLED_
    # METHODS -- the trainer code itself (DeltaOrthRankExtensionTrainer,
    # compute_old_semantic_subspace()) is UNCHANGED from
    # final_8arm_family_specific_5x20.py, which already carries this
    # mechanism over verbatim from R7/vit_lora_cifar100_full5step_n5.py (see
    # this script's header). Teacher construction (frozen previous-step
    # RankExt model) is unchanged from every prior run.
    add_method(
        "rank_extension_fullkd_T2_protect30", "rank_extension", "rank_extension_fullkd_T2_protect30",
        uses_kd=True, kd_temperature=2.0, kd_class_scope="full",
        kd_warmup_epochs=0.0, kd_weight_scale=1.0,
    )
    # RE-3 (task brief Section 16): HISTORICAL FactorOrth via
    # uses_factor_orth=True -- resolves to orth_mode="factor_orth" and
    # lambda_orth=LAMBDA_ORTH(50.0)*lambda_orth_scale(1.0, default)=50.0 --
    # i.e. EXACTLY Exp1/R7's historical RankExt FactorOrth arm. Kept under
    # its job-4970580 method/base_method names (unchanged, already correct)
    # so this run's FactorOrth-alone arm never matches
    # RANKEXT_PROJECTED_PROTECT_METHODS / RANKEXT_NEW_BLOCK_WARMUP_DISABLED_
    # METHODS -- protect_weight resolves to 0.0 and new-block-warmup stays
    # enabled, matching Exp1's own treatment of its non-KD FactorOrth arm
    # (task brief Section 16: "No protect loss unless historical code
    # directly shows it was active for FO-only" -- it does not). NO KD here.
    add_method(
        "rank_extension_factor_orth_lam50", "rank_extension", "rank_extension_factor_orth_lam50_new",
        uses_factor_orth=True,
    )
    # RE-4 (task brief Section 17): FactorOrth lambda=50 + HISTORICAL
    # full-100-way KD, T=2, weight=1.0, NO KD-weight warmup, protect_weight=
    # 30.0, new-block warmup DISABLED -- RE-2 + RE-3 combined. This is the
    # main final RankExt candidate: as close a reconstruction of Exp1's
    # 70.72%-producing recipe as the corrected (non-buggy) classifier
    # restoration allows. Deliberately NOT passing lambda_orth_scale/
    # kd_weight_scale tied to COMBINED_LOSS_SCALE_ENABLED (that historical
    # rescaling mechanism was introduced to avoid a full-strength
    # kd=1.0/lambda=50 training collapse seen for the analogous SimpleAvg
    # combined arm; job 4961962 already proved full-strength, unscaled
    # lambda=50/kd_weight=1.0 is SAFE and is exactly what reproduced 70.72%
    # bit-for-bit on the modern codebase -- see the design plan's Section 8).
    add_method(
        "rank_extension_factor_orth_lam50_fullkd_T2_protect30", "rank_extension",
        "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
        uses_kd=True, kd_temperature=2.0, kd_class_scope="full",
        kd_warmup_epochs=0.0, kd_weight_scale=1.0,
        uses_factor_orth=True,
    )
    # ==========================================================================
    # END JOINT PACK ADDITION (RankExt side). Everything below (historical
    # rank_extension_kd_only_T2/orth_factor_lam_50/orth_factor_lam_50_kd_T2)
    # is Exp1's own, UNCHANGED code -- gated off via METHODS_TO_RUN (not
    # retrained in this run, never deleted).
    # ==========================================================================

    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(f"rank_extension_kd_only_{kd_tag}", "rank_extension", "rank_extension_kd_only", uses_kd=True, kd_temperature=kd_temp)
    add_method("rank_extension_orth_delta_trace_lam_50", "rank_extension", "rank_extension_orth_delta_trace_lam_50", uses_delta_trace=True)
    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(f"rank_extension_orth_delta_trace_lam_50_kd_{kd_tag}", "rank_extension", "rank_extension_orth_delta_trace_lam_50_kd", uses_kd=True, kd_temperature=kd_temp, uses_delta_trace=True)
    add_method("rank_extension_orth_factor_lam_50", "rank_extension", "rank_extension_orth_factor_lam_50", uses_factor_orth=True)
    # STRICT-FAIRNESS REDESIGN, pair-4 decision (2026-08-25, user directive):
    # rather than either (a) removing simple_avg_factor_orth_kd_T2's combined-
    # loss scaling (risking reproducing its documented ~63.71% training
    # collapse at full-strength kd=1.0/lambda=50 -- see COMBINED_LOSS_SCALE_
    # ENABLED's own comment above for the evidence) or (b) leaving pair 4
    # coefficient-mismatched, the RankExt side of pair 4 is scaled DOWN to
    # match SimpleAvg's already-stabilized combined coefficients instead --
    # reusing the SAME COMBINED_KD_WEIGHT_SCALE (0.5) / COMBINED_LAMBDA_ORTH_
    # SCALE (0.3) constants applied to simple_avg_factor_orth_kd_T2 just
    # above, so both sides of pair 4 resolve to the IDENTICAL effective
    # kd_weight=0.5, lambda_orth=15 (KD_TEMPERATURES/T=2 untouched, KD_WEIGHT
    # and LAMBDA_ORTH globals untouched -- this affects ONLY this one method's
    # own resolved coefficients, exactly as the pre-existing mechanism already
    # does for simple_avg_factor_orth_kd_T2). This makes this method's config
    # in THIS run a DIFFERENT, deliberately-rescaled variant of the flagship --
    # NOT the same trained configuration that produced the historical
    # all_seen=74.07 result (kd=1.0/lambda=50, job 4918131, dir
    # ..._4x25_full_comparison_with_orth_rankext_..._160602, untouched and
    # retained separately) -- see this run's own fairness-audit report for the
    # explicit historical-vs-fairness-run distinction.
    #
    # RENAMED (2026-08-25, user directive): "..._lam_50_kd_T2" ->
    # "..._lam_15_kd_T2" -- the internal identifier was made to match its true
    # effective lambda (15.0, not 50.0) at the time, rather than relying on
    # the resolved config-table columns alone to disambiguate from the name
    # (which is what the PRIOR version of this comment had settled for; the
    # user correctly flagged that as still misleading).
    #
    # RENAMED BACK (2026-08-25, FULL-STRENGTH COMBINED EXPERIMENT, user
    # directive): "..._lam_15_kd_T2" -> "..._lam_50_kd_T2" -- _combined_
    # lambda_scale/_combined_kd_scale (defined just above, gated on
    # COMBINED_LOSS_SCALE_ENABLED=False) now resolve to 1.0/1.0, so this
    # method's true effective lambda_orth/kd_weight are 50.0/1.0 again; the
    # identifier is renamed back to match, same principle as the original
    # rename -- the name always tracks the true resolved value, not the other
    # way around. See METHODS_TO_RUN's own comment above (base_method key,
    # now "rank_extension_orth_factor_lam_50_kd" again) for the full list of
    # every other consumer renamed alongside this call site, both times.
    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(
            f"rank_extension_orth_factor_lam_50_kd_{kd_tag}", "rank_extension", "rank_extension_orth_factor_lam_50_kd",
            uses_kd=True, kd_temperature=kd_temp, uses_factor_orth=True,
            lambda_orth_scale=_combined_lambda_scale, kd_weight_scale=_combined_kd_scale,
        )

    return configs


ACTIVE_METHOD_CONFIGS = build_active_method_configs()
ACTIVE_METHOD_NAMES = [cfg["method"] for cfg in ACTIVE_METHOD_CONFIGS]
ACTIVE_METHOD_MAP = {cfg["method"]: cfg for cfg in ACTIVE_METHOD_CONFIGS}
_SUPERVISOR_SELECTED_METHOD_SPECS_BY_METHOD = {
    spec["internal_method_name"]: spec for spec in SUPERVISOR_SELECTED_METHOD_SPECS
}
ACTIVE_SUPERVISOR_SELECTED_METHOD_SPECS = [
    _SUPERVISOR_SELECTED_METHOD_SPECS_BY_METHOD[method_name]
    for method_name in ACTIVE_METHOD_NAMES
    if method_name in _SUPERVISOR_SELECTED_METHOD_SPECS_BY_METHOD
]
ACTIVE_SUPERVISOR_SELECTED_INTERNAL_METHODS = [
    spec["internal_method_name"] for spec in ACTIVE_SUPERVISOR_SELECTED_METHOD_SPECS
]
ACTIVE_SUPERVISOR_SELECTED_DISPLAY_NAMES = [
    spec["display_name"] for spec in ACTIVE_SUPERVISOR_SELECTED_METHOD_SPECS
]
ENABLED_METHOD_FAMILIES = [name for name, enabled in METHODS_TO_RUN.items() if enabled]
# FIX 2: "simple_avg_delta_orth" and "rank_extension_orth_delta_trace_lam_50"
# removed from the expected set to match the two flags flipped to False above --
# otherwise `assert set(ENABLED_METHOD_FAMILIES) == EXPECTED_ENABLED_METHOD_FAMILIES`
# below would fail as soon as those two were disabled.
# FINAL THESIS COMPARISON (2026-08-25): all 8 principal base_method flags,
# matching METHODS_TO_RUN above -- the settled 3-method RankEXT-only set
# (BASELINE RESTORED 2026-08-24) and the single-flagship KD-weight-treatment
# set (FINAL KD-WEIGHT EXPERIMENT, now closed) are both historical; this is
# the full canonical 8-method comparison.
# SINGLEJOB PACK FIX (infrastructure/validation-only -- readiness audit,
# Section "readiness risk"): this set is Exp1's own self-consistency check of
# "which base_method families are actually enabled." Exp1's original 8-name
# set is factually wrong for THIS script (its own METHODS_TO_RUN now enables
# a different 5-name set) and, left as-is, causes `assert
# set(ENABLED_METHOD_FAMILIES) == EXPECTED_ENABLED_METHOD_FAMILIES` (below) to
# fail immediately at module load, before any training starts. Updated to
# describe THIS experiment's actual, intentional active set -- preserving the
# check's original purpose (catch accidental METHODS_TO_RUN drift) rather
# than weakening or removing it. No training/loss/merge/calibration/metric
# code is touched by this change.
# JOINT PACK FIX (infrastructure/validation-only): extended from the
# SimpleAvg-only pack's 5-name set to this joint script's actual 9-name set
# (5 SimpleAvg + 4 RankExt base_methods) -- same rationale as that pack's own
# fix (see its own preparation audit report): preserves the check's original
# purpose (catch accidental METHODS_TO_RUN drift) rather than weakening it.
# FINAL 8-ARM SCRIPT: updated to this script's actual 8 enabled base_method
# keys (4 SimpleAvg + 4 RankExt) -- see METHODS_TO_RUN above for the full
# True/False dict this is checked against.
# PROBE: reduced to exactly the 2 active base_method keys (see
# METHODS_TO_RUN above) -- catches accidental drift the same way the
# canonical script's own version of this set did.
EXPECTED_ENABLED_METHOD_FAMILIES = {
    "rank_extension_factor_orth_lam50_new",
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
}

# FINAL THESIS COMPARISON: back to the settled KD_WEIGHT=1.0 (the KDw=0.75
# single-point treatment above is closed -- see KD_WEIGHT definition above).
assert KD_WEIGHT == 1.0
assert KD_TEMPERATURES == [2.0]
assert LAMBDA_ORTH == 50.0
assert LORA_R == 80
assert LORA_ALPHA == 160
# RESTORED (2026-08-25, explicit user correction): back to the original
# final-rank-matched invariant -- see LORA_R's own "RESTORED" comment above
# for why rank itself is deliberately excluded from the fairness pass (it is
# SimpleAvg's defining architectural axis, not a controllable setting). This
# also restores the ORIGINAL alpha check (RankExt's cumulative-final alpha
# vs LORA_ALPHA) in place of the now-removed alpha-per-rank-density variant
# the same-day rank redesign had introduced -- both forms are mathematically
# equivalent whenever LORA_R == RANKEXT_RANK_SCHEDULE[-1], which is true
# again after this revert, so nothing is lost, restoring the original form
# exactly.
assert LORA_R == RANKEXT_RANK_SCHEDULE[-1]
assert float(RANKEXT_ALPHA_PER_RANK * RANKEXT_RANK_SCHEDULE[-1]) == float(LORA_ALPHA)
# CAPACITY TEST verification: whichever schedule is ACTIVE (default or WIDE),
# the alpha/rank ratio GrowingRankLoRALinear actually uses at every growth
# step is exactly RANKEXT_ALPHA_PER_RANK (see that class's scaling formula --
# it cancels total_rank out identically at rank 16, 32, ..., 160), so this is
# a single reporting-consistency check, not a per-step re-derivation:
# active_rankext_lora_alpha() must equal ALPHA_PER_RANK * the active
# schedule's own final rank, for both the default and the wide schedule.
assert float(active_rankext_lora_alpha()) == float(RANKEXT_ALPHA_PER_RANK) * float(active_rankext_rank_schedule()[-1])

# =============================================================================
# EXPERIMENT 1 HARD ASSERTIONS + STARTUP DIAGNOSTICS (prepared 2026-09-10,
# finalized 2026-09-11, see
# thesis_agent/reports/supervisor_cifar_followup_audit.md). Fails fast,
# before epoch 1, if this file's own required overrides above did not take
# effect (e.g. an accidental merge/edit reverted one back to canonical, or a
# stray env var interfered). Every check below is against LIVE state (the
# actual resolved constants/objects at this point in module execution), not
# against this file's own source text.
# =============================================================================

# --- dataset / no ImageNet contamination -------------------------------
# (`dataset`/`LABEL_COL` do not exist yet at this point in module execution
# -- the dataset-load cell runs later, further down this file; those two
# checks are asserted there instead, right after `LABEL_COL` is set, still
# well before any training starts -- see "DATASET IDENTITY CHECK" below.)
assert NUM_CLASSES == 100, f"Experiment 1 requires 100 total classes, got {NUM_CLASSES}"
for _leftover_imagenet_name in (
    "DATASET_REGISTRY", "DATASET_NAME", "EXPERIMENT_LABEL", "IMAGENET_ROOT",
    "IMAGENET100_SYNSETS", "IMAGENET100_CLASS_ORDER",
    "IMAGENET100_SYNSETS_CMC_ALTERNATIVE", "IMAGENET100_WNID_TO_LABEL",
    "N5StopAfterSetup", "N5_SKIP_TRAINING_DRIVER",
):
    assert _leftover_imagenet_name not in globals(), (
        f"No ImageNet configuration must be active: found leftover name {_leftover_imagenet_name!r}"
    )
assert "imagenet" not in RUN_NAME_BASE.lower(), f"RUN_NAME_BASE must not mention imagenet: {RUN_NAME_BASE!r}"

# --- protocol / seed ---------------------------------------------------------
assert SEED == 42, f"Experiment 1 requires seed=42, got {SEED}"
# PROBE: 5x20 protocol reduced to a {PROBE_NUM_STEPS}x20 SMOKE (first
# PROBE_NUM_STEPS of the same 5 steps, same seed/class order/CLASSES_PER_STEP
# -- NOT the full 100-class protocol; NUM_CLASSES itself is untouched at 100,
# this probe just stops after step {PROBE_NUM_STEPS} instead of continuing to
# step 5).
assert NUM_STEPS == PROBE_NUM_STEPS and CLASSES_PER_STEP == 20, (
    f"PROBE requires the {PROBE_NUM_STEPS}x20 smoke protocol; got NUM_STEPS={NUM_STEPS}, "
    f"CLASSES_PER_STEP={CLASSES_PER_STEP}"
)
assert NUM_STEPS * CLASSES_PER_STEP <= NUM_CLASSES == 100
assert "4x25" not in RUN_NAME_BASE, f"RUN_NAME_BASE must not mention 4x25: {RUN_NAME_BASE!r}"
assert "seed123" not in RUN_NAME_BASE and "widerank40" not in RUN_NAME_BASE and "capacity_sensitivity" not in RUN_NAME_BASE, (
    f"RUN_NAME_BASE must not contain stale replication/capacity-test wording: {RUN_NAME_BASE!r}"
)

# --- RankExt: schedule + "current fixed implementation" evidence -----------
assert not USE_RANKEXT_RANK_SCHEDULE_WIDE, "Wide RankExt schedule (4x25 capacity ablation) must be OFF for Experiment 1"
assert active_rankext_rank_schedule() == [16, 32, 48, 64, 80], (
    f"Experiment 1 requires the historical 5x20 RankExt cumulative schedule "
    f"[16,32,48,64,80] (same rank as before the fix), got {active_rankext_rank_schedule()}"
)
assert float(RANKEXT_ALPHA_PER_RANK) == 2.0, (
    f"RankExt effective scaling must be 2.0 (constant at every step by construction), "
    f"got RANKEXT_ALPHA_PER_RANK={RANKEXT_ALPHA_PER_RANK}"
)
# These four flags are the concrete, checkable markers of "the CURRENT fixed
# RankExt implementation" (family-aware + confidence-weighted regime-grouped
# calibration, new-block warmup, orth-lambda warmup) -- all landed AFTER the
# pre-fix 5x20 reference (job 4904629) and are what "fixed" means throughout
# thesis_agent/reports/supervisor_cifar_followup_audit.md. If any is False,
# this is silently running the OLD, pre-fix RankExt behavior.
assert RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED is True, "Current fixed RankExt requires family-aware calibration ON"
assert RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED is True, "Current fixed RankExt requires confidence-weighted calibration ON"
assert RANKEXT_NEW_BLOCK_WARMUP_ENABLED is True, "Current fixed RankExt requires new-block warmup ON"
assert RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED is True, "Current fixed RankExt requires orth-lambda warmup ON"

# --- SimpleAvg (current canonical config, NOT job 4904629's obsolete one) --
assert LORA_R == 80 and LORA_ALPHA == 160, (
    f"Experiment 1's SimpleAvg config must stay at the current canonical "
    f"rank=80/alpha=160, got LORA_R={LORA_R}, LORA_ALPHA={LORA_ALPHA}"
)
assert float(LORA_ALPHA) / float(LORA_R) == 2.0, "SimpleAvg effective scaling must be 2.0"
assert TARGET_MODULES_BY_FAMILY["simple_avg"] == ["q_proj", "v_proj"], (
    "SimpleAvg must use the CURRENT 2-module (q_proj/v_proj) config, NOT job 4904629's "
    f"obsolete 4-module one, got {TARGET_MODULES_BY_FAMILY['simple_avg']}"
)
assert TARGET_MODULES_BY_FAMILY["rank_extension"] == ["q_proj", "v_proj"]

# --- calibration: current, consistent across BOTH families -----------------
assert CALIBRATION_MODE_BY_FAMILY["simple_avg"] == "confidence_weighted_regime_grouped", (
    "SimpleAvg must use confidence_weighted_regime_grouped, NOT job 4904629's obsolete "
    f"'global' mode, got {CALIBRATION_MODE_BY_FAMILY['simple_avg']!r}"
)
assert CALIBRATION_MODE_BY_FAMILY["rank_extension"] == "confidence_weighted_regime_grouped", (
    f"got {CALIBRATION_MODE_BY_FAMILY['rank_extension']!r}"
)
assert CALIBRATION_ENABLED_FAMILIES["simple_avg"] is True and CALIBRATION_ENABLED_FAMILIES["rank_extension"] is True

# --- KD: family-specific weights, T, T-squared, old-seen scope -------------
# FINAL 8-ARM SCRIPT: the module-global KD_WEIGHT/KD_TEMPERATURES constants
# stay at their historical canonical values (1.0 / [2.0]) UNCHANGED --
# family-specific weights (SA=0.5, RE=1.5) are applied per-arm via
# add_method()'s kd_weight_scale parameter (kd_weight = KD_WEIGHT *
# kd_weight_scale, see add_method()'s own formula), never by touching this
# global. This is intentional: it keeps "the nominal KD mechanism" (T=2,
# old-seen scope, T^2 scaling, batchmean KL) identical to job 4969059, and
# makes the per-family weight difference fully visible/auditable in each
# arm's own resolved config (ACTIVE_METHOD_MAP[...]["kd_weight"]) rather than
# hidden in a changed global constant.
assert KD_WEIGHT == 1.0 and KD_TEMPERATURES == [2.0], (
    f"KD config must stay canonical (weight=1.0, T=2.0), got weight={KD_WEIGHT}, T={KD_TEMPERATURES}"
)
# --- KD scope assertions (task brief Section 22): SimpleAvg KD arms must be
# old-seen-only; RankExt KD arms must be full-100-way. Prevents accidental
# cross-routing between the two families' KD mechanisms.
# PROBE: no SimpleAvg arms active -- empty tuple (loop below is then a no-op,
# not a broken lookup). Only the one active KD-bearing RankExt arm remains
# (rank_extension_fullkd_T2_protect30 is disabled in this probe).
_SA_KD_ARMS = ()
assert "simple_avg_kd_oldseen_T2_warmup" not in ACTIVE_METHOD_MAP
assert "simple_avg_kd_oldseen_T4_warmup" not in ACTIVE_METHOD_MAP
assert "simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup" not in ACTIVE_METHOD_MAP
_RE_KD_ARMS = (
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
)
assert "rank_extension_fullkd_T2_protect30" not in ACTIVE_METHOD_MAP
for _sa_kd_arm in _SA_KD_ARMS:
    assert ACTIVE_METHOD_MAP[_sa_kd_arm]["uses_kd"] is True, f"{_sa_kd_arm} must have uses_kd=True"
    assert ACTIVE_METHOD_MAP[_sa_kd_arm]["kd_class_scope"] == "old_seen", (
        f"{_sa_kd_arm} must use KD_SCOPE=old_seen, got {ACTIVE_METHOD_MAP[_sa_kd_arm]['kd_class_scope']}"
    )
    assert abs(float(ACTIVE_METHOD_MAP[_sa_kd_arm]["kd_weight"]) - 1.0) < 1e-9, (
        f"{_sa_kd_arm} must resolve to kd_weight=1.0, got {ACTIVE_METHOD_MAP[_sa_kd_arm]['kd_weight']}"
    )
    assert float(ACTIVE_METHOD_MAP[_sa_kd_arm]["kd_warmup_epochs"]) == 1.0, (
        f"{_sa_kd_arm} must have a 1-epoch KD-weight warmup"
    )
for _re_kd_arm in _RE_KD_ARMS:
    assert ACTIVE_METHOD_MAP[_re_kd_arm]["uses_kd"] is True, f"{_re_kd_arm} must have uses_kd=True"
    # NOTE: this codebase's own scope string for full-100-way KD is "full"
    # (not "full_100" -- see masked_kd_loss()/DeltaOrthRankExtensionTrainer.
    # compute_loss()'s kd_class_scope branch). This assertion checks the
    # actual, unambiguous scope value used by the running code, which IS the
    # "full 100-way KD" the task brief's KD_SCOPE=="full_100" refers to.
    assert ACTIVE_METHOD_MAP[_re_kd_arm]["kd_class_scope"] == "full", (
        f"{_re_kd_arm} must use full-100-way KD (kd_class_scope='full'), "
        f"got {ACTIVE_METHOD_MAP[_re_kd_arm]['kd_class_scope']} -- accidental old-seen routing"
    )
    assert abs(float(ACTIVE_METHOD_MAP[_re_kd_arm]["kd_weight"]) - 1.0) < 1e-9, (
        f"{_re_kd_arm} must resolve to kd_weight=1.0, got {ACTIVE_METHOD_MAP[_re_kd_arm]['kd_weight']}"
    )
    assert float(ACTIVE_METHOD_MAP[_re_kd_arm]["kd_warmup_epochs"]) == 0.0, (
        f"{_re_kd_arm} must have NO KD-weight warmup, matching Exp1's flat-weight convention"
    )
    assert _re_kd_arm in RANKEXT_PROJECTED_PROTECT_METHODS, (
        f"{_re_kd_arm} must activate protect_weight=30.0 (RANKEXT_PROJECTED_PROTECT_METHODS)"
    )
    assert _re_kd_arm in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS, (
        f"{_re_kd_arm} must have new-block output warmup disabled"
    )

# --- temperature assertions (task brief Section 23) -------------------------
# PROBE: only the one active KD-bearing RankExt arm is checked (SimpleAvg and
# rank_extension_fullkd_T2_protect30 are inactive in this probe -- see the
# "not in ACTIVE_METHOD_MAP" assertions just above).
assert float(ACTIVE_METHOD_MAP["rank_extension_factor_orth_lam50_fullkd_T2_protect30"]["kd_temperature"]) == 2.0
# T-squared scaling is a structural property of IndependentLoraOrthTrainer/
# DeltaOrthRankExtensionTrainer's compute_loss (`kd_loss = F.kl_div(...) *
# (self.kd_temperature ** 2)`, unconditional and generic -- never a
# hardcoded per-temperature multiplier) -- confirmed by direct source
# inspection during preparation of this script (T=4 -> factor 16, applied
# through this SAME generic formula, not a special-cased constant).

# --- lambda assertions (task brief Section 24): DenseOrth20 for SimpleAvg,
# FactorOrth50 for RankExt -- never mixed. ---------------------------------
assert LAMBDA_ORTH == 50.0, f"FactorOrth lambda must stay canonical (50), got {LAMBDA_ORTH}"
assert DENSE_ORTH_LAMBDA == 20.0, f"DenseOrth lambda must stay canonical (20), got {DENSE_ORTH_LAMBDA}"
# PROBE: no SimpleAvg arms active.
_SA_DENSE_ORTH_ARMS = ()
_RE_FACTOR_ORTH_ARMS = ("rank_extension_factor_orth_lam50", "rank_extension_factor_orth_lam50_fullkd_T2_protect30")
for _dense_orth_arm in _SA_DENSE_ORTH_ARMS:
    assert ACTIVE_METHOD_MAP[_dense_orth_arm]["uses_dense_orth"] is True, f"{_dense_orth_arm} must have uses_dense_orth=True"
    assert float(ACTIVE_METHOD_MAP[_dense_orth_arm]["lambda_orth"]) == float(DENSE_ORTH_LAMBDA), (
        f"{_dense_orth_arm} must use DENSE_ORTH_LAMBDA={DENSE_ORTH_LAMBDA}"
    )
for _factor_orth_arm in _RE_FACTOR_ORTH_ARMS:
    assert ACTIVE_METHOD_MAP[_factor_orth_arm]["uses_factor_orth"] is True, f"{_factor_orth_arm} must have uses_factor_orth=True"
    assert ACTIVE_METHOD_MAP[_factor_orth_arm]["uses_dense_orth"] is False, f"{_factor_orth_arm} must NOT use DenseOrth"
    assert float(ACTIVE_METHOD_MAP[_factor_orth_arm]["lambda_orth"]) == float(LAMBDA_ORTH), (
        f"{_factor_orth_arm} must resolve lambda_orth=LAMBDA_ORTH=50.0 (no rescaling), "
        f"got {ACTIVE_METHOD_MAP[_factor_orth_arm]['lambda_orth']}"
    )
# FactorOrth-alone (RE-3, non-KD) must NOT collide with the protect_weight /
# new-block-warmup-disabled sets -- only the two KD-bearing RankExt arms do.
assert "rank_extension_factor_orth_lam50" not in RANKEXT_PROJECTED_PROTECT_METHODS, (
    "rank_extension_factor_orth_lam50 (non-KD) must not activate protect_weight"
)
assert "rank_extension_factor_orth_lam50" not in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS, (
    "rank_extension_factor_orth_lam50 (non-KD) must keep new-block warmup enabled"
)
assert not any(ACTIVE_METHOD_MAP[m]["uses_dense_orth"] for m in ACTIVE_METHOD_NAMES if ACTIVE_METHOD_MAP[m]["family"] == "rank_extension"), (
    "No RankExt arm may use DenseOrth in this script -- RankExt's orthogonality is FactorOrth-only"
)
assert not any(ACTIVE_METHOD_MAP[m]["uses_factor_orth"] for m in ACTIVE_METHOD_NAMES if ACTIVE_METHOD_MAP[m]["family"] == "simple_avg"), (
    "No SimpleAvg arm may use FactorOrth in this script -- SimpleAvg's orthogonality is DenseOrth-only"
)

# --- method set: exact 2-name match, not just the family-flag set ----------
# PROBE: reduced from the canonical 9-arm set to exactly the 2 RankExt arms
# under test (spec section 4: combined arm primary, FactorOrth50-alone
# secondary; no SimpleAvg, no plain RankExt, no KD-only-no-FactorOrth arm).
EXPECTED_METHODS = {
    "rank_extension_factor_orth_lam50",
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
}
_EXPECTED_SINGLEJOB_METHOD_NAMES = EXPECTED_METHODS  # alias: preserved name used by the reporting section below
active_methods = list(ACTIVE_METHOD_NAMES)
assert set(active_methods) == EXPECTED_METHODS, (
    "This PROBE must run exactly these 2 RankExt arms -- mismatch: "
    f"{set(active_methods)} != {EXPECTED_METHODS}"
)
assert len(active_methods) == 2, f"Must be exactly 2 active methods in this probe, got {len(active_methods)}"
assert sum(1 for m in ACTIVE_METHOD_NAMES if ACTIVE_METHOD_MAP[m]["family"] == "simple_avg") == 0, (
    "PROBE must have zero SimpleAvg training arms"
)
assert sum(1 for m in ACTIVE_METHOD_NAMES if ACTIVE_METHOD_MAP[m]["family"] == "rank_extension") == 2, (
    "PROBE must have exactly 2 RankExt training arms"
)
# --- no-replay assertion (task brief Section 7) -----------------------------
assert all(not ACTIVE_METHOD_MAP[m].get("uses_replay", False) for m in ACTIVE_METHOD_NAMES), (
    "No active method may use replay in this script"
)
# (no-DO-Merge is asserted separately, right next to MERGE_METHODS_TO_EVALUATE's
# own definition further below in this file, since that constant does not
# exist yet at this point in module load.)
# PROBE: RANKEXT_EPOCHS is intentionally reduced (see PROBE_RANKEXT_EPOCHS);
# LORA_EPOCHS is vestigial since SimpleAvg never runs in this probe.
assert RANKEXT_EPOCHS == PROBE_RANKEXT_EPOCHS, (
    f"RANKEXT_EPOCHS must be {PROBE_RANKEXT_EPOCHS} in this probe, got {RANKEXT_EPOCHS}"
)
# JOINT PACK: no active-Exp2 schedule anywhere -- this is NOT a capacity
# ablation (user's task Section 8/27).
assert not USE_RANKEXT_RANK_SCHEDULE_WIDE, "Exp2's wide RankExt schedule must be OFF -- this is not a capacity ablation"
assert active_rankext_rank_schedule() == [16, 32, 48, 64, 80], (
    f"RankExt must use the canonical [16,32,48,64,80] schedule, not Exp2's [32,64,96,128,160], "
    f"got {active_rankext_rank_schedule()}"
)
assert len(ACTIVE_METHOD_NAMES) == 2, f"Expected 2 total training arms in this probe, got {len(ACTIVE_METHOD_NAMES)}"
assert set(ENABLED_METHOD_FAMILIES) == EXPECTED_ENABLED_METHOD_FAMILIES

print("=" * 80)
print("EXPERIMENT 1 STARTUP DIAGNOSTICS -- 5x20 re-evaluation under the current fair/")
print("canonical pipeline, corrected RankExt implementation, historical RankExt schedule")
print("=" * 80)
print(f"  dataset               = cifar100 (label_col diagnostic deferred -- 'dataset'/'LABEL_COL'")
print(f"                          do not exist yet at this point in module execution; see")
print(f"                          'EXPERIMENT 1 dataset identity check' printed later, right")
print(f"                          after LABEL_COL is set)")
print(f"  protocol              = {NUM_STEPS}x{CLASSES_PER_STEP} ({NUM_CLASSES} total classes)")
print(f"  seed                  = {SEED}")
print(f"  method_set            = {sorted(ACTIVE_METHOD_NAMES)} ({len(ACTIVE_METHOD_NAMES)} methods)")
print(f"  simple_avg_rank       = {LORA_R}")
print(f"  simple_avg_alpha      = {LORA_ALPHA}")
print(f"  simple_avg_scaling    = {float(LORA_ALPHA) / float(LORA_R)}")
print(f"  simple_avg_targets    = {TARGET_MODULES_BY_FAMILY['simple_avg']}")
print(f"  rankext_schedule      = {active_rankext_rank_schedule()}")
print(f"  rankext_scaling       = {RANKEXT_ALPHA_PER_RANK}")
print(f"  rankext_fixed_flags   = family_aware={RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED}, "
      f"confidence_weighted={RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED}, "
      f"new_block_warmup={RANKEXT_NEW_BLOCK_WARMUP_ENABLED}, "
      f"orth_lambda_warmup={RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED}")
print(f"  kd_weight / kd_T      = {KD_WEIGHT} / {KD_TEMPERATURES}")
print(f"  factororth_lambda     = {LAMBDA_ORTH}")
print(f"  epochs (lora/rankext) = {LORA_EPOCHS} / {RANKEXT_EPOCHS}")
print(f"  calibration_mode      = {CALIBRATION_MODE_BY_FAMILY}")
print(f"  run_name_base         = {RUN_NAME_BASE!r}")
print("EXPERIMENT 1 hard assertions PASSED.")
print("=" * 80)

# JOINT PACK ADDITION -- explicit readiness verification block (user's task
# Part 0): printed and asserted BEFORE any training starts, so a
# misconfiguration is visible in the log header, not discovered mid-run.
_re_arms = [m for m in ACTIVE_METHOD_NAMES if ACTIVE_METHOD_MAP[m]["family"] == "rank_extension"]
_sa_arms = [m for m in ACTIVE_METHOD_NAMES if ACTIVE_METHOD_MAP[m]["family"] == "simple_avg"]
# SimpleAvg is arithmetic-merge-ONLY here (no DO-Merge, no replay) -- 1
# merge per SA arm, not 2.
_expected_final_configs = len(_sa_arms) * 1 + len(_re_arms) * 1
print("=" * 80)
print("FINAL 9-METHOD 5x20 PERFORMANCE-RECOVERY EXPERIMENT -- READINESS VERIFICATION")
print("=" * 80)
print(f"SCRIPT UNDER TEST: experiments_prepared/final_9method_5x20_performance_recovery.py")
print(f"FAST_RUN_DEBUG: {FAST_RUN} (env FAST_RUN_DEBUG={os.environ.get('FAST_RUN_DEBUG', '<unset>')!r})")
# NOTE: ROOT_RESULTS_DIR is not yet defined at this point in module execution
# (it's set further down, right before BASE_OUTPUT_DIR/TABLES_DIR/etc.) --
# printed there instead, not here, to avoid a NameError.
print(f"NUM_EPOCHS: {LORA_EPOCHS} (LoRA) / {RANKEXT_EPOCHS} (RankExt)")
print(f"SIMPLEAVG TRAINING ARMS: {len(_sa_arms)} -> {sorted(_sa_arms)}")
print(f"RANKEXT TRAINING ARMS: {len(_re_arms)} -> {sorted(_re_arms)}")
print(f"TOTAL TRAINING ARMS: {len(ACTIVE_METHOD_NAMES)}")
_expected_merge_methods = ["arithmetic"]  # cross-checked against the real MERGE_METHODS_TO_EVALUATE constant at its own definition site, later in this file -- DO-Merge removed
print(f"SIMPLEAVG MERGES: {_expected_merge_methods}")
print(f"RANKEXT MERGE: persistent")
print(f"EXPECTED FINAL CONFIGURATIONS: {_expected_final_configs}")
print(f"SIMPLEAVG RANK: {LORA_R}")
print(f"SIMPLEAVG ALPHA: {LORA_ALPHA}")
print(f"SIMPLEAVG SCALING: {float(LORA_ALPHA) / float(LORA_R)}")
print(f"RANKEXT SCHEDULE: {active_rankext_rank_schedule()}")
print(f"RANKEXT INCREMENT: {[active_rankext_rank_schedule()[i] - (active_rankext_rank_schedule()[i-1] if i > 0 else 0) for i in range(len(active_rankext_rank_schedule()))]}")
print(f"RANKEXT FINAL RANK: {active_rankext_rank_schedule()[-1]}")
print(f"RANKEXT SCALING: {RANKEXT_ALPHA_PER_RANK}")
print("=" * 80)
assert RANKEXT_EPOCHS == PROBE_RANKEXT_EPOCHS
assert len(_sa_arms) == 0 and len(_re_arms) == 2 and len(ACTIVE_METHOD_NAMES) == 2
assert _expected_final_configs == 2, f"Expected 2 final configurations in this probe, computed {_expected_final_configs}"
assert LORA_R == 80 and LORA_ALPHA == 160 and float(LORA_ALPHA) / float(LORA_R) == 2.0
assert active_rankext_rank_schedule() == [16, 32, 48, 64, 80]
assert active_rankext_rank_schedule()[-1] == 80
assert float(RANKEXT_ALPHA_PER_RANK) == 2.0
print("READINESS VERIFICATION PASSED -- all 0-section checks hold.")
print("=" * 80)
# ACCURACY-PUSH CHANGE 1: pinned set updated from ["q_proj", "v_proj"] to include
# k_proj/out_proj. This is now the simple_avg-family value specifically (see
# TARGET_MODULES_BY_FAMILY REVERT note above) -- keep this assert in sync with
# TARGET_MODULES above -- it exists to catch silent drift between the two,
# not to gate the value itself.
assert TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "out_proj"]
# REVERT (2026-07-16): rank_extension's target modules are pinned back to the
# BASELINE-proven 2-module set. Keep in sync with TARGET_MODULES_BY_FAMILY.
assert TARGET_MODULES_BY_FAMILY["rank_extension"] == ["q_proj", "v_proj"]
# FINAL CORRECTION (2026-08-25): simple_avg is pinned to q_proj/v_proj only
# (see TARGET_MODULES_BY_FAMILY's own comment above) -- no longer equal to the
# global 4-module TARGET_MODULES default, by explicit instruction.
assert TARGET_MODULES_BY_FAMILY["simple_avg"] == ["q_proj", "v_proj"]
assert set(ENABLED_METHOD_FAMILIES) == EXPECTED_ENABLED_METHOD_FAMILIES
assert not any(cfg["uses_replay"] for cfg in ACTIVE_METHOD_CONFIGS)
assert not any(cfg["uses_zero_old"] for cfg in ACTIVE_METHOD_CONFIGS)
# STRICT-FAIRNESS REDESIGN: family-conditional, matching add_method()'s own
# "rank" field logic above -- simple_avg's configs report their own per-step
# rank (LORA_R); rank_extension's report their final cumulative rank.
assert all(
    (cfg["rank"] == LORA_R) if cfg["family"] == "simple_avg"
    else (cfg["rank"] == active_rankext_rank_schedule()[-1])
    for cfg in ACTIVE_METHOD_CONFIGS
)
# Per-family target_modules check (was a single global comparison before the
# REVERT above made this family-conditional).
assert all(
    cfg["target_modules"] == ", ".join(family_target_modules(cfg["family"]))
    for cfg in ACTIVE_METHOD_CONFIGS
)
assert all(
    cfg["head_lr_multiplier"] == family_head_lr_multiplier(cfg["family"])
    for cfg in ACTIVE_METHOD_CONFIGS
)

# FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT: this check originally verified
# every KD base_method swept the SAME global KD_TEMPERATURES=[2.0] list (a
# leftover multi-temperature-sweep convention from Exp1). SA-3
# (simple_avg_kd_oldseen_T4_warmup) is a deliberate, explicit T=4 one-off
# (task brief Section 10's clean T2-vs-T4 ablation), never part of that
# sweep convention -- excluded here by name, and separately asserted to be
# exactly T=4.0 in the temperature-assertions block above.
_T4_ABLATION_BASE_METHODS = {"simple_avg_kd_oldseen_T4_warmup"}
kd_method_temperature_map = {}
for cfg in ACTIVE_METHOD_CONFIGS:
    if cfg["uses_kd"] and cfg["base_method"] not in _T4_ABLATION_BASE_METHODS:
        kd_method_temperature_map.setdefault(cfg["base_method"], []).append(float(cfg["kd_temperature"]))
for base_method, temps in kd_method_temperature_map.items():
    assert sorted(temps) == sorted(KD_TEMPERATURES), f"KD sweep mismatch for {base_method}: {temps}"
# PROBE: SA-3 T4 arm is inactive in this probe (SimpleAvg entirely off).
assert "simple_avg_kd_oldseen_T4_warmup" not in ACTIVE_METHOD_MAP


# JOINT PACK ADDITION (GPU preflight): an entirely SEPARATE root directory
# tree when FAST_RUN_DEBUG=1 -- not a subdirectory of "results", so no glob/
# listing of real output could ever accidentally pick up debug output, and
# CHECKPOINTS_DIR (derived from ROOT_RESULTS_DIR below) is automatically
# isolated too, with zero additional code.
ROOT_RESULTS_DIR = "results_debug_preflight" if FAST_RUN else "results"
print(f"ROOT_RESULTS_DIR: {ROOT_RESULTS_DIR!r} {'(ISOLATED DEBUG-ONLY -- never real scientific output)' if FAST_RUN else '(real output)'}")

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_OUTPUT_DIR = os.path.join(
    ROOT_RESULTS_DIR,
    f"{RUN_NAME}_{RUN_TAG}"
)

TABLES_DIR = os.path.join(BASE_OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")
LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "logs")
CONFIGS_DIR = os.path.join(BASE_OUTPUT_DIR, "configs")
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models")

# SINGLEJOB PACK ADDITION -- CRITICAL RESUME-CORRECTNESS DETAIL: RUN_TAG (see
# its own definition above) is `datetime.now().strftime(...)`, a FRESH
# timestamp every time this script launches. BASE_OUTPUT_DIR embeds RUN_TAG,
# so if this job crashes and SLURM restarts/resubmits it, a plain
# `os.path.join(BASE_OUTPUT_DIR, "checkpoints")` would point at a brand-new,
# empty directory every restart and NEVER find a previous attempt's saved
# specialists -- defeating the entire point of persistence. CHECKPOINTS_DIR
# therefore depends ONLY on RUN_NAME_BASE (deterministic given SEED, stable
# across restarts of the same job), never on RUN_TAG, so every restart of
# this job finds the same checkpoint directory regardless of when it restarts.
CHECKPOINTS_DIR = os.path.join(ROOT_RESULTS_DIR, f"{RUN_NAME_BASE}_checkpoints")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

all_results = []
method_summary_rows = []
train_diagnostic_rows = []
epoch_loss_rows = []
# PRE-THESIS FIX 1: one row per (method, step) logging which epoch's weights were
# actually kept after best-epoch selection (see USE_BEST_EPOCH_SELECTION comment).
best_epoch_selection_rows = []
# PRE-THESIS FIX 2: per-CL-step (1..NUM_STEPS) accuracy of each method's FINAL
# model, plus the raw ingredients for backward_transfer/forward_transfer (see
# evaluate_per_step_accuracy() and the run_*_variant functions below).
per_step_accuracy_rows = []
# EVAL-PIPELINE AUDIT ADD (analysis_pipeline_audit/report.txt): closed-set
# companion to per_step_accuracy_rows above. Same model, same eval_ds, same
# forward pass (computed alongside the open-set accuracy in
# evaluate_per_step_accuracy(), not a separate re-evaluation) -- only the
# argmax candidate set differs (restricted to the eval subset's own classes,
# instead of open over all NUM_CLASSES). See restricted_argmax_accuracy() for
# why this was added: it isolates a step's own representation quality from
# cross-class classifier-row-norm competition (which the report shows is the
# dominant driver of "step_5 always looks best", especially for
# apply_calibration=False methods). One row per (method, step_id), long
# format, mirrors per_step_accuracy_rows's schema exactly.
per_step_accuracy_restricted_rows = []
# FIX 1 diagnostic (analysis_recency_fix/report.txt): one row per (method,
# step_id, phase) logging the FINAL merged/final classifier's per-step-block
# row-norm statistics -- mean row norm for that step's 20-class block, and its
# ratio to step 1's mean row norm (a direct, numerical answer to "are later
# steps' rows several times larger than earlier steps'?"). Logged
# unconditionally (both phase="pre_calibration", always, and
# phase="post_calibration", only when apply_calibration is True for that
# method) from inside calibrate_classifier_row_norms() / its call sites below,
# so this table has evidence for EVERY method regardless of whether
# calibration is applied to it -- R5 (WIDERANK run) predates this
# instrumentation and has no equivalent data, which is why Task A.2 of that
# analysis could only report shapes, not norms.
classifier_row_norm_diagnostic_rows = []
# FIX 2 diagnostic (analysis_recency_fix2/report.txt): one row per (method,
# step_id) logging the inputs and outputs of
# calibrate_classifier_row_norms_confidence_weighted()'s boost-factor
# computation -- the val_ce_loss used, the group it was compared against, the
# resulting boost factor, and the pre/post target norms. Only populated for
# methods actually calibrated with mode="confidence_weighted_regime_grouped";
# lets the next run's report show the boost factors that were actually
# applied, not just the resulting row norms (which classifier_row_norm_
# diagnostic_rows above already covers).
classifier_confidence_calibration_diagnostic_rows = []
# PRE-THESIS FIX 2: {method_name: {step_idx: {task_step: accuracy_fraction}}} --
# only populated for rank_extension family (the only family with a genuinely
# evolving model to checkpoint mid-training); used to draw a true forgetting
# curve (accuracy on task i as training progresses through later steps).
rank_extension_stepwise_accuracy_by_method = {}

# =============================================================================
# Task 2: live convergence plotting + tables, generated DURING the run.
#
# Every method's per-step training funnels through train_with_trainer() (below),
# which appends that step's val-CE rows into the module-global `epoch_loss_rows`
# list, and through a custom Trainer subclass whose per-batch rows get appended
# into the module-global `train_diagnostic_rows` list right after each step
# finishes (see the `orth_train_records.extend(...)` call sites). That means, right
# after any (method, step) finishes, both lists already contain everything logged
# for that method so far -- enough to redraw a progress plot without waiting for
# the whole multi-hour run to complete.
#
# refresh_live_convergence(method_name) is called at each of those call sites. It:
#   1. rebuilds a per-(step, epoch) train/val CE frame for `method_name` from the
#      two accumulator lists above (whatever has been logged so far),
#   2. overwrites plots/live_convergence_<method>.png (smooth PCHIP curve within
#      each CL step, markers on real data points, line breaks at step boundaries),
#   3. rewrites tables/all_methods_convergence_table.csv (all methods trained so
#      far) and a *provisional* tables/top2_convergence_table.csv (ranked by
#      lowest mean val CE seen so far, since true final accuracy isn't known until
#      the whole run ends).
# Both tables are overwritten with the authoritative, final versions (built from
# training_loss_history_df, and the TRUE top-2-by-accuracy) once the full run
# finishes and ranking_table is computed -- see the "FINAL (authoritative)
# convergence tables" cell near the end of the notebook.
# =============================================================================
LIVE_CONVERGENCE_ENABLED = True


def _epoch_bucket_live(epoch_value):
    """Round a (possibly fractional) HF Trainer `state.epoch` up to the epoch
    index it belongs to, e.g. 0.97 -> 1, 1.995 -> 2. Duplicated (rather than
    reused) from the later `epoch_bucket()` helper because this is called during
    training, before that helper is defined further down the script."""
    if pd.isna(epoch_value):
        return np.nan
    return int(max(1, math.ceil(float(epoch_value) - 1e-12)))


def _build_method_epoch_frame(method_name):
    """Progressive (not-yet-complete) per-(step, epoch) train/val CE frame for
    `method_name`, built from whatever is in train_diagnostic_rows/epoch_loss_rows
    so far. See module-level comment above for why this is safe to call mid-run."""
    train_rows = [r for r in train_diagnostic_rows if r.get("method") == method_name]
    if len(train_rows) == 0:
        return pd.DataFrame(columns=["step_id", "epoch", "train_ce_loss", "val_ce_loss"])

    tdf = pd.DataFrame(train_rows)
    tdf["epoch_id"] = tdf["epoch"].apply(_epoch_bucket_live)
    tdf = tdf.dropna(subset=["epoch_id"])
    train_epoch = (
        tdf.groupby(["step", "epoch_id"], as_index=False)["ce_loss"]
        .mean()
        .rename(columns={"step": "step_id", "epoch_id": "epoch", "ce_loss": "train_ce_loss"})
    )
    train_epoch["step_id"] = train_epoch["step_id"].astype(int)
    train_epoch["epoch"] = train_epoch["epoch"].astype(int)

    val_rows = [r for r in epoch_loss_rows if r.get("method_name") == method_name]
    if len(val_rows) > 0:
        vdf = pd.DataFrame(val_rows)[["step_id", "epoch", "val_ce_loss"]].copy()
        vdf["step_id"] = vdf["step_id"].astype(int)
        vdf["epoch"] = vdf["epoch"].astype(int)
        merged = train_epoch.merge(vdf, on=["step_id", "epoch"], how="left")
    else:
        merged = train_epoch.copy()
        merged["val_ce_loss"] = np.nan

    return merged.sort_values(["step_id", "epoch"]).reset_index(drop=True)


def _plot_step_broken_series(ax, df, y_col, color, label, lw=1.8, ms=4.5, marker="o",
                              linestyle="-", x_col=None):
    """Plot `y_col` against a "global epoch so far" index (either `x_col` if given,
    or a freshly computed running 1..N index), smoothing WITHIN each CL step with a
    PCHIP spline (markers on the real data points) and breaking the line at step
    boundaries -- a new CL step introduces new classes / a reset classifier row, so
    one continuous line across steps would visually imply a continuity that isn't
    there. Falls back to plain polylines if scipy is unavailable (see _HAVE_SCIPY)."""
    df = df.reset_index(drop=True)
    if x_col is None:
        df["_global_epoch"] = np.arange(1, len(df) + 1)
        x_col = "_global_epoch"
    first = True
    for _, g in df.groupby("step_id", sort=True):
        xs = g[x_col].to_numpy(dtype=float)
        ys = g[y_col].to_numpy(dtype=float)
        valid = ~np.isnan(ys)
        xs, ys = xs[valid], ys[valid]
        if len(xs) == 0:
            continue
        if len(xs) >= 3 and _HAVE_SCIPY:
            xs_dense = np.linspace(xs.min(), xs.max(), 25)
            ax.plot(xs_dense, PchipInterpolator(xs, ys)(xs_dense), color=color, lw=lw,
                     linestyle=linestyle, label=(label if first else None))
        else:
            ax.plot(xs, ys, color=color, lw=lw, linestyle=linestyle, label=(label if first else None))
        ax.plot(xs, ys, marker=marker, ms=ms, lw=0, color=color)
        first = False


def _write_live_convergence_tables():
    all_rows = []
    methods_seen = sorted({r.get("method") for r in train_diagnostic_rows if r.get("method")})
    for m in methods_seen:
        df = _build_method_epoch_frame(m)
        if len(df) == 0:
            continue
        df = df.copy()
        df["method_name"] = m
        df["display_name"] = METHOD_DISPLAY_NAME_MAP.get(m, m)
        all_rows.append(df)
    if not all_rows:
        return
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df = all_df[["method_name", "display_name", "step_id", "epoch", "train_ce_loss", "val_ce_loss"]]
    all_df.to_csv(os.path.join(TABLES_DIR, "all_methods_convergence_table.csv"), index=False)

    # Provisional ranking: lowest mean val CE seen so far. Overwritten at the end
    # of the run with the true top-2-by-final-accuracy methods (see the
    # "FINAL (authoritative) convergence tables" cell near the end of the script).
    mean_val = all_df.groupby("method_name")["val_ce_loss"].mean().dropna().sort_values()
    provisional_top2 = mean_val.index[:2].tolist()
    top2_df = all_df[all_df["method_name"].isin(provisional_top2)]
    top2_df.to_csv(os.path.join(TABLES_DIR, "top2_convergence_table.csv"), index=False)


def refresh_live_convergence(method_name):
    """Call after each CL step finishes for `method_name` (Task 2). Best-effort:
    a plotting failure must never crash the actual training run."""
    if not LIVE_CONVERGENCE_ENABLED:
        return
    try:
        df = _build_method_epoch_frame(method_name)
        if len(df) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        df["_global_epoch"] = np.arange(1, len(df) + 1)
        _plot_step_broken_series(ax, df, "train_ce_loss", "#1f77b4", "train CE", x_col="_global_epoch")
        _plot_step_broken_series(ax, df, "val_ce_loss", "#d62728", "val CE", x_col="_global_epoch")

        # TASK 3(a) (decision doc, 2026-08-17): mark each CL step's SELECTED
        # (best-val-CE) epoch directly on the curve. best_epoch_selection_rows
        # already has one row per (method, step) with the true selected epoch
        # (USE_BEST_EPOCH_SELECTION's own bookkeeping, not re-derived here) by
        # the time this step's refresh_live_convergence() call happens.
        # Plotted as a gold star at (this row's global-epoch position, that
        # epoch's val CE) so it is visually obvious that the "rise after the
        # minimum" the raw curve shows is exactly what best-epoch selection
        # already discards when merging into the final model -- see the
        # module comment near USE_BEST_EPOCH_SELECTION for why the reload
        # always uses this epoch regardless of how many epochs the curve
        # keeps plotting after it.
        best_rows_this_method = [r for r in best_epoch_selection_rows if r.get("method_name") == method_name]
        first_marker = True
        for r in best_rows_this_method:
            match = df[(df["step_id"] == int(r["step_id"])) & (df["epoch"] == int(r["selected_epoch"]))]
            if len(match) == 0:
                continue
            mrow = match.iloc[0]
            ax.plot(
                mrow["_global_epoch"], mrow["val_ce_loss"],
                marker="*", ms=14, mec="black", mew=0.6, color="gold", zorder=5, linestyle="None",
                label=("selected (best) epoch" if first_marker else None),
            )
            first_marker = False

        step_sizes = df.groupby("step_id").size()
        boundary = 0
        for step_id in sorted(df["step_id"].unique())[:-1]:
            boundary += int(step_sizes.loc[step_id])
            ax.axvline(boundary + 0.5, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.set_xlabel("global epoch so far (dotted = CL step boundary)")
        ax.set_ylabel("CE loss")
        disp = METHOD_DISPLAY_NAME_MAP.get(method_name, method_name)
        last_step = int(df["step_id"].max())
        # EVAL-PIPELINE AUDIT ADD: val CE here is each CL step's OWN local
        # validation split, evaluated with the model AS IT STOOD during that
        # step's own training -- it is a different measurement from the
        # retrospective per-step accuracy in per_task_accuracy_heatmap.png /
        # per_step_accuracy_by_method.csv (which evaluates the method's FINAL
        # model on the same class group, after every later step has run). See
        # analysis_pipeline_audit/report.txt for why conflating the two reads
        # as an inconsistency that isn't one.
        ax.set_title(
            f"{disp} -- live convergence (through step {last_step}/{NUM_STEPS})\n"
            "val CE = this step's OWN local val split, model as of THIS step (in-context, not retrospective)",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"live_convergence_{method_name}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        _write_live_convergence_tables()
    except Exception as exc:
        print(f"[live convergence] WARNING: failed to refresh live artifacts for {method_name}: {exc}")


print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
print("FP16:", USE_FP16)
print("Checkpoint:", MODEL_CHECKPOINT)
print("BASE_OUTPUT_DIR:", BASE_OUTPUT_DIR)
print("TABLES_DIR:", TABLES_DIR)
print("PLOTS_DIR:", PLOTS_DIR)
print("REPORTS_DIR:", REPORTS_DIR)
print("LOGS_DIR:", LOGS_DIR)
print("CONFIGS_DIR:", CONFIGS_DIR)
print("MODELS_DIR:", MODELS_DIR)

print("\nRun mode:")
print({
    "FAST_RUN": FAST_RUN,
    "DEBUG_MODE": DEBUG_MODE,
    "RUN_NAME": RUN_NAME,
    "SCRATCH_EPOCHS": SCRATCH_EPOCHS,
})

print("\nEpochs:")
print({
    "FT_EPOCHS": FT_EPOCHS,
    "LORA_EPOCHS": LORA_EPOCHS,
    "JOINT_EPOCHS": JOINT_EPOCHS,
    "ORTH_EPOCHS": ORTH_EPOCHS,
    "RANKEXT_EPOCHS": RANKEXT_EPOCHS,
})

print("\nLoRA:")
print({
    "LORA_R": LORA_R,
    "LORA_ALPHA": LORA_ALPHA,
    "LORA_DROPOUT": LORA_DROPOUT,
    # FINAL CORRECTION (2026-08-25): TARGET_MODULES is only the unused global
    # fallback again (simple_avg is pinned to q_proj/v_proj explicitly, no
    # longer equal to this 4-module default) -- see TARGET_MODULES_BY_FAMILY
    # on the next line for what simple_avg/rank_extension actually use.
    "TARGET_MODULES (unused fallback default)": TARGET_MODULES,
    "TARGET_MODULES_BY_FAMILY": TARGET_MODULES_BY_FAMILY,
    "HEAD_LR_MULTIPLIER_BY_FAMILY": HEAD_LR_MULTIPLIER_BY_FAMILY,
    "RANKEXT_RANK_SCHEDULE_active": active_rankext_rank_schedule(),
})

print("\nOrth/KD config:")
print({
    "LAMBDA_ORTH": LAMBDA_ORTH,
    "LAMBDA_ORTH_DELTA_TRACE": LAMBDA_ORTH_DELTA_TRACE,
    "LAMBDA_ORTH_FACTOR": LAMBDA_ORTH_FACTOR,
    "KD_WEIGHT": KD_WEIGHT,
    "KD_TEMPERATURES": KD_TEMPERATURES,
    "ORTH_NORM_EPS": ORTH_NORM_EPS,
    "ORTH_LOSS_LOG_EVERY": ORTH_LOSS_LOG_EVERY,
})

print("\nValidation split:")
print({
    "VALIDATION_PER_CLASS": VALIDATION_PER_CLASS,
    "LOSS_NA_FILL": LOSS_NA_FILL,
})

print("\nRank extension:")
print({
    "RANKEXT_RANK_SCHEDULE (default)": RANKEXT_RANK_SCHEDULE,
    "RANKEXT_RANK_SCHEDULE_WIDE": RANKEXT_RANK_SCHEDULE_WIDE,
    "USE_RANKEXT_RANK_SCHEDULE_WIDE": USE_RANKEXT_RANK_SCHEDULE_WIDE,
    "active_rankext_rank_schedule()": active_rankext_rank_schedule(),
    "RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED": RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED,
    "RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS": RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS,
    "RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED": RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED,
    "RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED": RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED,
    "RANKEXT_ALPHA_PER_RANK": RANKEXT_ALPHA_PER_RANK,
    "LR_RANKEXT": LR_RANKEXT,
    "REPLAY_PER_CLASS": REPLAY_PER_CLASS,
    "RANKEXT_REPLAY_PER_CLASS": RANKEXT_REPLAY_PER_CLASS,
    "RANKEXT_DIAGNOSTICS": RANKEXT_DIAGNOSTICS,
})

print("\nMethods:")
print(json.dumps(METHODS_TO_RUN, indent=2))

enabled_methods = list(ACTIVE_METHOD_NAMES)
disabled_methods = [m for m, enabled in METHODS_TO_RUN.items() if not enabled]
print("Enabled method families for this run:", ENABLED_METHOD_FAMILIES)
print("Expanded active methods for this run:", enabled_methods)
print("Disabled methods/flags for this run:", disabled_methods)


# In[ ]:


# COLAB FIX: the bare, un-namespaced "cifar100" dataset id fails to load
# under current huggingface_hub/datasets versions on Colab -- the library's
# newer HF-URI parser rejects the legacy single-segment repo id when
# resolving the dataset's revision-pinned .huggingface.yaml, raising
# `HfUriError: Repository id must be 'namespace/name', got 'cifar100'`
# (this does NOT happen on the cluster, which pins older library versions;
# the canonical, unmodified cluster probe still uses the bare "cifar100" id
# and must NOT be changed). "uoft-cs/cifar100" is the actively-maintained,
# namespaced Hub mirror of the same dataset -- verified identical schema
# (img/fine_label/coarse_label columns, train=50000/test=10000 rows), so no
# downstream code changes are needed. Fall back to the bare id only if the
# namespaced mirror is ever unavailable, so this still works in an
# environment where the old id resolves fine.
try:
    dataset = load_dataset("uoft-cs/cifar100")
    print("Loaded dataset via namespaced Colab-safe mirror: uoft-cs/cifar100")
except Exception as e:
    print(f"uoft-cs/cifar100 load failed ({e!r}); falling back to bare 'cifar100' id")
    dataset = load_dataset("cifar100")

LABEL_COL = "fine_label" if "fine_label" in dataset["train"].column_names else "label"
IMAGE_COL = "img" if "img" in dataset["train"].column_names else "image"

# DATASET IDENTITY CHECK (deferred from the EXPERIMENT 1 HARD ASSERTIONS
# block above -- `dataset`/`LABEL_COL` do not exist until this point). Still
# runs well before any training starts.
assert LABEL_COL == "fine_label", f"Expected CIFAR-100's fine_label column, got {LABEL_COL!r}"
assert "fine_label" in dataset["train"].column_names and "coarse_label" in dataset["train"].column_names, (
    "Loaded dataset does not look like CIFAR-100 (missing fine_label/coarse_label columns) -- "
    f"got columns: {dataset['train'].column_names}"
)
assert dataset["train"].num_rows == 50000 and dataset["test"].num_rows == 10000, (
    "Loaded dataset does not match CIFAR-100's known split sizes (train=50000, test=10000) -- "
    f"got train={dataset['train'].num_rows}, test={dataset['test'].num_rows}"
)
print(f"EXPERIMENT 1 dataset identity check PASSED: CIFAR-100, "
      f"train={dataset['train'].num_rows}, test={dataset['test'].num_rows}, label_col={LABEL_COL!r}")

# PROTOCOL-DEPTH VALIDATION (2026-08-24): was a hardcoded 5-entry literal
# ([range(0,20), range(20,40), ...]) -- the one genuinely protocol-specific
# assumption found in this file's step/class logic (everything else already
# read NUM_STEPS/CLASSES_PER_STEP/classes_for_step() symbolically). Now
# derived generically from NUM_STEPS/CLASSES_PER_STEP so it is correct for
# BOTH 5x20 (unchanged: reduces to the exact same 5 ranges above) and 20x5
# (20 ranges of 5). Classes are NOT shuffled/permuted here or anywhere else
# in this construction -- native CIFAR-100 label order 0..99 is simply
# chunked contiguously -- so this generalization, by construction, preserves
# class order/content exactly: fine-steps 1-4 under 20x5 are classes 0-19
# split into four 5-class blocks, i.e. bit-identical to 5x20's step 1 class
# SET (just partitioned into more, smaller steps), and likewise for every
# other 20-class group (verified programmatically in the pre-flight
# synthetic test below).
class_splits = [
    list(range(i * CLASSES_PER_STEP, (i + 1) * CLASSES_PER_STEP))
    for i in range(NUM_STEPS)
]

# PROTOCOL LOCK relaxed for this PROBE ONLY: only the first PROBE_NUM_STEPS
# of the canonical 5 groups are required to exist -- same class order, same
# class IDs per group as the full run (0-19, 20-39, 40-59, ...), this probe
# just stops requesting groups past PROBE_NUM_STEPS.
_CANONICAL_5X20_CLASS_SPLITS = [
    list(range(0, 20)),
    list(range(20, 40)),
    list(range(40, 60)),
    list(range(60, 80)),
    list(range(80, 100)),
]
assert class_splits == _CANONICAL_5X20_CLASS_SPLITS[:PROBE_NUM_STEPS], (
    f"class_splits must match the first {PROBE_NUM_STEPS} groups of the canonical "
    f"5x20 CIFAR-100 partition (same class order/IDs as the full run), got {class_splits}"
)

first_step_classes = class_splits[0]
later_step_classes = [c for split in class_splits[1:] for c in split]
all_classes = [c for split in class_splits for c in split]

def classes_for_step(step_idx):
    return class_splits[step_idx]

def old_seen_class_ids(step_idx):
    """SINGLEJOB PACK ADDITION (mechanism ported from R8's old_seen_class_ids(),
    supervisor_regularizer_repair_both_families_r8.py -- verbatim mechanism,
    adapted only to call THIS script's own single-argument classes_for_step()
    instead of R8's two-argument version): C_old = union of
    classes_for_step(i) for i in range(step_idx) -- the exact previously-seen
    class set under this script's actual protocol, not an independent
    contiguous-range formula. step_idx is 0-based (step_idx=0 is CL step 1),
    returns [] there (range(0) is empty) -- KD is inactive at step 1 for the
    old-seen arms for exactly the same reason it already is for the
    historical full-100-way KD arm (build_simple_avg_teacher_model requires
    len(step_states)>0, which is also False at step 1), so this never needs a
    special-cased guard here."""
    ids = []
    for old_step in range(step_idx):
        ids.extend(classes_for_step(old_step))
    return ids

def filter_by_classes(ds, class_ids):
    class_ids = set(class_ids)
    return ds.filter(lambda x: int(x[LABEL_COL]) in class_ids)

print("Dataset columns:", dataset["train"].column_names)
print("Label column:", LABEL_COL)
print("Image column:", IMAGE_COL)
for i, cls in enumerate(class_splits, start=1):
    print(f"Step {i}: {cls[0]}-{cls[-1]}")


# In[ ]:


image_processor = CLIPImageProcessor.from_pretrained(MODEL_CHECKPOINT)

if hasattr(image_processor, "crop_size") and image_processor.crop_size is not None:
    H = int(image_processor.crop_size.get("height", 224))
    W = int(image_processor.crop_size.get("width", 224))
else:
    H = W = 224

train_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomCrop((H, W), padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    ),
])

def to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, dict):
        if "array" in x:
            x = x["array"]
        elif "bytes" in x:
            import io
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")

    if isinstance(x, list):
        x = np.array(x, dtype=np.uint8)

    if isinstance(x, np.ndarray):
        arr = np.squeeze(x).astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return Image.fromarray(arr).convert("RGB")

    return x

def preprocess_train(ex):
    ex["pixel_values"] = [train_transform(to_pil(img)) for img in ex[IMAGE_COL]]
    ex["labels"] = [int(y) for y in ex[LABEL_COL]]
    return ex

def preprocess_val(ex):
    ex["pixel_values"] = [val_transform(to_pil(img)) for img in ex[IMAGE_COL]]
    ex["labels"] = [int(y) for y in ex[LABEL_COL]]
    return ex

def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([int(e["labels"]) for e in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        # CRASH FIX (rank_extension feature-anchor lever): this compute_metrics
        # is shared by every trainer class in the script (simple_avg,
        # rank_extension, and the plain Trainer used by
        # evaluate_seen_step_accuracies()), so it can't assume `logits` is
        # always a single tensor. RankExtensionTrainer.preprocess_logits_for_
        # metrics() already narrows predictions to one classification-logits
        # tensor before Trainer hands them off, so this is only a defensive
        # backstop -- pick the classification-logits element explicitly by
        # shape (2-D, last dim == NUM_CLASSES) rather than assuming
        # predictions[0].
        candidates = [
            t for t in logits
            if hasattr(t, "shape") and len(t.shape) == 2 and t.shape[-1] == NUM_CLASSES
        ]
        if len(candidates) != 1:
            raise ValueError(
                "compute_metrics: expected exactly one 2-D (*, "
                f"{NUM_CLASSES}) classification-logits array among {len(logits)} "
                f"prediction elements, found {len(candidates)}. Shapes: "
                f"{[getattr(t, 'shape', type(t)) for t in logits]}"
            )
        logits = candidates[0]
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float((preds == labels).mean())
    }

print("Image size:", H, W)
print("CLIP mean:", image_processor.image_mean)
print("CLIP std:", image_processor.image_std)


# In[ ]:


class CLIPVisionForCIFAR100(nn.Module):
    """
    CLIP-ViT vision encoder + trainable CIFAR-100 classifier.

    This uses:
    openai/clip-vit-base-patch16

    The text encoder is not used.
    Only the CLIP vision backbone is used.
    """

    def __init__(self, checkpoint, num_labels):
        super().__init__()

        self.vision_model = CLIPVisionModel.from_pretrained(checkpoint, use_safetensors=True)

        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.config = self.vision_model.config
        self.config.num_labels = num_labels
        self.config.id2label = {i: str(i) for i in range(num_labels)}
        self.config.label2id = {str(i): i for i in range(num_labels)}

    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def fresh_pretrained_model():
    """
    Fresh CLIP-ViT vision model with a CIFAR-100 classifier.
    """
    return CLIPVisionForCIFAR100(
        checkpoint=MODEL_CHECKPOINT,
        num_labels=NUM_CLASSES,
    )

def disable_incompatible_torchao_for_peft():
    """
    Colab may have an old torchao installed. Recent PEFT checks torchao during
    LoRA injection and raises before falling back to normal nn.Linear LoRA.
    This guard disables only PEFT's torchao LoRA dispatcher when that version
    check fails; it does not change the LoRA method.
    """
    try:
        import peft.import_utils as peft_import_utils

        try:
            peft_import_utils.is_torchao_available()
            return
        except ImportError as e:
            if "incompatible version of torchao" not in str(e):
                raise

        peft_import_utils.is_torchao_available = lambda: False

        try:
            import peft.tuners.lora.torchao as peft_lora_torchao
            peft_lora_torchao.is_torchao_available = lambda: False
        except Exception:
            pass

        print(
            "[PEFT compatibility] Disabled torchao LoRA dispatcher "
            "because installed torchao is incompatible with PEFT."
        )
    except ImportError:
        return

def add_lora(model, target_modules=None):
    """
    Add LoRA to the CLIP-ViT attention projection modules listed in
    `target_modules` (defaults to the global TARGET_MODULES for callers that
    predate family-conditional target modules -- see TARGET_MODULES_BY_FAMILY
    / family_target_modules() above; the simple_avg-family training loop
    passes family_target_modules("simple_avg") explicitly).
    """
    disable_incompatible_torchao_for_peft()
    resolved_target_modules = list(TARGET_MODULES) if target_modules is None else list(target_modules)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=resolved_target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config)
    return model

print("LoRA target modules (default/simple_avg):", TARGET_MODULES)
print("LoRA target modules by family (TARGET_MODULES_BY_FAMILY):", TARGET_MODULES_BY_FAMILY)
print("Classifier calibration master switch (USE_CLASSIFIER_CALIBRATION):", USE_CLASSIFIER_CALIBRATION)
print("Classifier calibration by family (CALIBRATION_ENABLED_FAMILIES):", CALIBRATION_ENABLED_FAMILIES)
print("Classifier calibration mode by family (CALIBRATION_MODE_BY_FAMILY):", CALIBRATION_MODE_BY_FAMILY)
print("Rank-extension family-aware calibration (RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED):", RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED)
print("Rank-extension confidence-weighted calibration (RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED):", RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED)
print("Head LR multiplier (default/simple_avg, HEAD_LR_MULTIPLIER):", HEAD_LR_MULTIPLIER)
print("Head LR multiplier by family (HEAD_LR_MULTIPLIER_BY_FAMILY):", HEAD_LR_MULTIPLIER_BY_FAMILY)
print("Rank-extension rank schedule in effect:", active_rankext_rank_schedule(),
      "(wide schedule enabled)" if USE_RANKEXT_RANK_SCHEDULE_WIDE else "(default schedule)")
print("Rank-extension orth-lambda warmup enabled:", RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED,
      "| warmup_epochs:", RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS)
print("Combined SimpleAvg+FactorOrth+KD loss scaling enabled:", COMBINED_LOSS_SCALE_ENABLED,
      "| lambda_orth_scale:", COMBINED_LAMBDA_ORTH_SCALE, "| kd_weight_scale:", COMBINED_KD_WEIGHT_SCALE)
print("Combined SimpleAvg+FactorOrth+KD orth warmup enabled:", COMBINED_ORTH_WARMUP_ENABLED,
      "| warmup_epochs:", COMBINED_ORTH_WARMUP_EPOCHS)


# In[ ]:


def build_classwise_train_val_splits(train_ds, val_per_class):
    train_parts = []
    val_parts = []
    rows = []

    for cls in all_classes:
        cls_ds = filter_by_classes(train_ds, [cls]).shuffle(seed=SEED + int(cls))
        if len(cls_ds) <= 1:
            raise ValueError(f"Need at least 2 examples for class {cls}, got {len(cls_ds)}")

        n_val = int(min(val_per_class, len(cls_ds) - 1))
        if n_val <= 0:
            raise ValueError(f"Validation split for class {cls} is empty. val_per_class={val_per_class}")

        val_parts.append(cls_ds.select(range(n_val)))
        train_slice = cls_ds.select(range(n_val, len(cls_ds)))
        if FAST_RUN:
            # JOINT PACK ADDITION (GPU preflight): cap TRAIN images/class only
            # under FAST_RUN_DEBUG=1 -- execution-coverage only, real run
            # (FAST_RUN=False) is completely unaffected by this branch.
            train_slice = train_slice.select(range(min(len(train_slice), 15)))
        train_parts.append(train_slice)
        rows.append({
            "class_id": int(cls),
            "train_count": int(len(train_slice)),
            "val_count": int(n_val),
        })

    return concatenate_datasets(train_parts), concatenate_datasets(val_parts), pd.DataFrame(rows)


train_source, val_source, train_val_split_df = build_classwise_train_val_splits(
    dataset["train"],
    val_per_class=VALIDATION_PER_CLASS,
)
validation_split_path = os.path.join(TABLES_DIR, "validation_split_summary.csv")
train_val_split_df.to_csv(validation_split_path, index=False)
print("Saved validation split summary:", validation_split_path)


def build_replay_dataset(old_classes, replay_per_class):
    if len(old_classes) == 0 or replay_per_class <= 0:
        return None

    parts = []

    for cls in old_classes:
        cls_ds = filter_by_classes(train_source, [cls])
        n = min(replay_per_class, len(cls_ds))
        cls_ds = cls_ds.shuffle(seed=SEED).select(range(n))
        parts.append(cls_ds)

    replay_ds = concatenate_datasets(parts)
    return replay_ds


def make_train_dataset(step_idx, replay_per_class=0):
    current_classes = classes_for_step(step_idx)
    current_ds = filter_by_classes(train_source, current_classes)

    old_classes = []
    for old_step in range(step_idx):
        old_classes.extend(classes_for_step(old_step))

    replay_ds = build_replay_dataset(
        old_classes=old_classes,
        replay_per_class=replay_per_class,
    )

    if replay_ds is None:
        final_ds = current_ds
    else:
        final_ds = concatenate_datasets([current_ds, replay_ds])

    final_ds = final_ds.shuffle(seed=SEED + step_idx)
    final_ds = final_ds.with_transform(preprocess_train)

    print(
        f"Step {step_idx + 1} | "
        f"current={len(current_ds)} | "
        f"replay={0 if replay_ds is None else len(replay_ds)} | "
        f"total={len(final_ds)}"
    )

    return final_ds


def make_val_dataset(class_ids):
    ds = filter_by_classes(val_source, class_ids)
    if FAST_RUN:
        # JOINT PACK ADDITION (GPU preflight): cap the per-step validation
        # set (used for best-epoch selection during training) ONLY under
        # FAST_RUN_DEBUG=1 -- real run unaffected.
        ds = ds.shuffle(seed=SEED).select(range(min(len(ds), 40)))
    ds = ds.with_transform(preprocess_val)
    return ds


def make_eval_dataset(class_ids):
    ds = filter_by_classes(dataset["test"], class_ids)
    if FAST_RUN:
        # JOINT PACK ADDITION (GPU preflight): cap the open-evaluation
        # (first_step/later_steps/all_seen) TEST set ONLY under
        # FAST_RUN_DEBUG=1 -- this is what made the earlier local CPU debug
        # run slow (only train images were capped there); capping it here
        # too is what makes the GPU preflight finish in minutes, not hours.
        # Real run (FAST_RUN=False) uses the full test set, unaffected.
        ds = ds.shuffle(seed=SEED).select(range(min(len(ds), 40)))
    ds = ds.with_transform(preprocess_val)
    return ds


def make_joint_train_dataset():
    ds = train_source.shuffle(seed=SEED)
    ds = ds.with_transform(preprocess_train)
    return ds


def make_joint_eval_dataset():
    ds = val_source
    ds = ds.with_transform(preprocess_val)
    return ds


eval_first = make_eval_dataset(first_step_classes)
eval_later = make_eval_dataset(later_step_classes)
eval_all_seen = make_eval_dataset(all_classes)

print("train_source:", len(train_source))
print("val_source:", len(val_source))
print("first_step eval:", len(eval_first))
print("later_steps eval:", len(eval_later))
print("all_seen eval:", len(eval_all_seen))


# In[ ]:


from transformers import TrainerCallback


# Task 3 mitigation: best-epoch (val-CE) checkpoint selection. R3's overfitting
# was mild overall (see WEIGHT_DECAY comment), but it was not zero -- a handful of
# (method, step) runs did have their last epoch be a small step backwards on val
# CE. Since EPOCHS is doubling (3 -> 6), that "last epoch is worst epoch" case is
# more likely to occur somewhere in the sweep.
#
# PRE-THESIS FIX 1: the EPOCH6 run's rigorous re-check
# (analysis_R4/reports/rigorous_assessment_new_vs_old.txt, Section 3) audited this
# end to end and found it was NOT actually taking effect, for two independent
# reasons in the original (pre-fix) implementation:
#
#   (a) WRONG METRIC. `metric_for_best_model="eval_loss"` was pointed at HF
#       Trainer's built-in `eval_loss`, which for IndependentLoraOrthTrainer /
#       DeltaOrthRankExtensionTrainer is `ce_loss + lambda_orth*orth_loss +
#       kd_weight*kd_loss` (compute_loss() returns the TOTAL weighted loss, and
#       Trainer's default prediction_step reuses compute_loss() for eval). For
#       lambda_orth=50 methods this total is dominated by the orth penalty (not
#       CE), and for KD methods it is contaminated by the KD term -- so "best
#       eval_loss" silently meant "best total regularized loss", not "best
#       validation CE" as documented and intended, for 6 of the 8 methods.
#
#   (b) UNVERIFIABLE RELOAD PATH. Even where the metric was correct (the 2 plain
#       methods, lambda_orth=0/kd_weight=0), correctness depended on HF Trainer's
#       internal load_best_model_at_end machinery correctly round-tripping a
#       PeftModel with modules_to_save=["classifier"] through
#       save_strategy="epoch" checkpoints and back -- a PEFT/Trainer interaction
#       that varies across transformers versions and was never independently
#       verified for this model wrapping.
#
# Fix: stop depending on HF Trainer's built-in best-model machinery entirely.
# EpochValidationCallback (below) already computes the one metric we actually
# want -- pure validation CE via compute_dataset_ce_loss(), which calls
# model(**batch) directly and is NOT affected by compute_loss() overrides -- once
# per epoch. It now ALSO keeps an in-memory CPU snapshot of just the trainable
# (LoRA + classifier) parameters whenever that snapshot's val CE improves on the
# best seen so far, and train_with_trainer() explicitly copies that snapshot back
# into `model` after trainer.train() finishes (see the snapshot/reload code in
# EpochValidationCallback and train_with_trainer below). This is simple enough to
# verify by reading the code, is independent of any Trainer/PEFT checkpoint I/O
# version quirk, and is keyed on the exact metric ("lowest validation CE within
# this CL step") the mitigation was always supposed to use. Every (method, step)'s
# selected epoch is logged to tables/best_epoch_selected_by_method_step.csv.
USE_BEST_EPOCH_SELECTION = True

# analysis_simple_avg_overfit/report.txt: simple_avg's train/val CE curves show
# GROWING (not bounded) within-step overfitting at steps 3-4 -- val CE falls to
# a minimum then rises again while train CE keeps falling. Investigation found
# best-epoch selection (above) already reloads that per-step minimum-val-CE
# checkpoint (selected_epoch=2 for both step 3 and step 4 of plain simple_avg,
# well before the rise), so this pattern is NOT currently reaching final
# accuracy -- reducing LORA_R or raising LORA_DROPOUT would therefore be
# solving an already-solved problem while risking the top method
# (simple_avg_factor_orth, whose own steps 3-4 select LATE epochs 7/9 because
# it is still genuinely improving there, not overfitting -- cutting its
# capacity/regularization is exactly the regression pattern the 2026-07-15
# LORA_DROPOUT 0.1->0.05 revert already documented). Decision: do not touch
# rank or dropout; instead add a standing, purely-diagnostic audit trail (one
# row per method/step, derived entirely from data best-epoch selection already
# collects) so every future run can see at a glance whether growing
# overfitting occurred and whether best-epoch selection actually protected
# that step's final accuracy, instead of requiring a manual investigation like
# this one. Purely additive: computed AFTER training from already-logged
# rows, touches no model weights, no training loop, no other saved column.
GROWING_OVERFITTING_DIAGNOSTICS_ENABLED = True
# Absolute val-CE rise from the selected (best) epoch to the configured final
# epoch above which a (method, step) is flagged as "growing overfitting"
# rather than noise. 0.05 chosen from analysis_simple_avg_overfit/report.txt's
# own numbers: simple_avg step 3/4 rise 0.12-0.15 (clearly real), while flat
# methods/steps (e.g. simple_avg_factor_orth step 3/4) rise <=0.004.
GROWING_OVERFITTING_VAL_CE_RISE_THRESHOLD = 0.05

# B1 (task 3 decision doc, 2026-08-17): adaptive per-step early stopping on
# val-CE -- NOW ENABLED, scoped to plain "simple_avg" only (see
# adaptive_early_stop_applies_to_method() just below). USE_BEST_EPOCH_
# SELECTION above already reloads the true val-CE-minimum checkpoint
# regardless of how many epochs actually ran this step, so enabling this can
# only shorten wall-clock/epoch-curve length; it can NEVER change which
# epoch's weights get merged into the final model or any reported accuracy,
# because the reload always targets the best snapshot seen so far, which is
# already captured by the time patience runs out (stopping only ever happens
# AFTER best_epoch + patience epochs). MUST be adaptive (patience-since-
# improvement, not a fixed epoch cap) so it would never fire on
# simple_avg_factor_orth even if it weren't also explicitly excluded below --
# its own best epoch trends LATE (e.g. 2->9 across steps -- see
# tables/best_epoch_generalization_gap_by_method_step.csv, B1's other
# diagnostic above): a method still improving at epoch 9 never accumulates
# PATIENCE consecutive non-improving epochs. Plain simple_avg's val CE
# plateaus/rises after epoch ~2-3 every CL step instead (selected_epoch
# 2, 5, 4, 2, 6 out of 9 configured), which is exactly the "curve keeps
# rising after the true minimum" pattern this flag makes visually go away in
# plots/live_convergence_simple_avg.png (see refresh_live_convergence()).
ADAPTIVE_PER_STEP_EARLY_STOP_ENABLED = True
# Epochs since the last val-CE improvement before stopping.
ADAPTIVE_PER_STEP_EARLY_STOP_PATIENCE = 3
# Never stop before this epoch, regardless of patience -- guards against a
# noisy early-epoch "improvement" causing a premature patience countdown.
ADAPTIVE_PER_STEP_EARLY_STOP_MIN_EPOCH = 3


def adaptive_early_stop_applies_to_method(method_name):
    """B1 (task 3 decision doc, 2026-08-17): restricts
    ADAPTIVE_PER_STEP_EARLY_STOP_ENABLED to plain "simple_avg" ONLY -- not
    simple_avg_kd_T2 / simple_avg_factor_orth / simple_avg_factor_orth_kd_T2,
    and not any rank_extension method. This is a second, EXPLICIT safety net
    on top of the patience-based design (which the flag's own comment already
    argues should never fire on simple_avg_factor_orth given this run's
    numbers) so simple_avg_factor_orth cannot be stopped early even if a
    future run's val-CE curve behaves differently than this one's -- task 3
    is explicit that it "must NOT stop simple_avg_factor_orth early". Every
    other simple_avg sibling and all of rank_extension are left off too, so
    their convergence plots/curves stay exactly as they were -- task 3 only
    asked to fix simple_avg's. Accuracy is unaffected regardless of which
    methods this returns True for (see ADAPTIVE_PER_STEP_EARLY_STOP_ENABLED's
    own comment) -- this function only controls which methods' PLOTS get
    shorter."""
    return str(method_name) == "simple_avg"


def get_training_args(
    output_dir,
    epochs,
    lr,
    batch_size,
    accum_steps,
    train_dataset_len=None,
    eval_strategy="epoch",
):
    """
    Trainer settings.

    We use warmup_steps instead of warmup_ratio because warmup_ratio is deprecated
    in newer Transformers versions.

    PRE-THESIS FIX 1: no longer accepts/wires load_best_model_at_end /
    metric_for_best_model / save_total_limit -- best-epoch selection is now done
    explicitly in train_with_trainer()/EpochValidationCallback using an in-memory
    trainable-parameter snapshot keyed on true validation CE (see the
    USE_BEST_EPOCH_SELECTION comment above). We therefore never need Trainer to
    write epoch checkpoints to disk; save_strategy is always "no".
    """

    if train_dataset_len is not None:
        steps_per_epoch = math.ceil(train_dataset_len / batch_size / accum_steps)
        total_steps = int(steps_per_epoch * epochs)
        warmup_steps = int(WARMUP_RATIO * total_steps)
    else:
        warmup_steps = 0

    kwargs = dict(
        output_dir=output_dir,
        remove_unused_columns=False,
        save_strategy="no",
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type=SCHED,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=accum_steps,
        fp16=USE_FP16,
        # JOINT PACK ADDITION (GPU preflight, cross-platform correctness):
        # 0 workers ONLY under FAST_RUN_DEBUG=1. This script has no
        # `if __name__=="__main__":` guard (converted-notebook, top-to-bottom
        # execution), so DataLoader(num_workers>0) crashes on Windows'
        # spawn-based multiprocessing with "attempt to start a new process
        # before the current process has finished its bootstrapping phase"
        # -- a real, foreseeable failure for anyone running the preflight
        # locally on Windows before submitting the real cluster job. Real
        # runs (FAST_RUN=False) are UNCHANGED (still 4 workers) -- this has
        # zero effect on Phase 2 or on the Linux GPU cluster (fork-based
        # multiprocessing has no such restriction, so num_workers=4 would
        # work there even under FAST_RUN, but 0 is fine for a tiny debug
        # dataset and keeps this fix minimal/single-purpose).
        dataloader_num_workers=(0 if FAST_RUN else 4),
        logging_steps=50,
        report_to="none",
        max_grad_norm=1.0,
    )

    sig = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy

    return TrainingArguments(**kwargs)


def compute_dataset_ce_loss(model, eval_ds, batch_size=32):
    if eval_ds is None or len(eval_ds) == 0:
        return np.nan

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_samples = 0
    loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            outputs = model(**batch)
            batch_loss = outputs.loss
            batch_size_now = int(batch["labels"].shape[0])
            total_loss += float(batch_loss.detach().cpu().item()) * batch_size_now
            total_samples += batch_size_now

    if was_training:
        model.train()

    if total_samples == 0:
        return np.nan
    return float(total_loss / total_samples)


class EpochValidationCallback(TrainerCallback):
    def __init__(self, method_name, display_name, step_idx, eval_dataset, eval_batch_size=32,
                 track_best_epoch=False):
        self.method_name = str(method_name)
        self.display_name = str(display_name)
        self.step_idx = int(step_idx)
        self.eval_dataset = eval_dataset
        self.eval_batch_size = int(eval_batch_size)
        self.epoch_rows = []
        self.trainer = None
        # PRE-THESIS FIX 1: explicit best-epoch (val-CE) tracking. See the
        # USE_BEST_EPOCH_SELECTION comment above get_training_args for why this
        # replaced HF Trainer's built-in load_best_model_at_end mechanism.
        self.track_best_epoch = bool(track_best_epoch)
        self.best_val_ce = float("inf")
        self.best_epoch = None
        self.best_state_dict = None

    def bind_trainer(self, trainer):
        self.trainer = trainer

    def _snapshot_trainable_params(self, model):
        # Only LoRA + classifier (modules_to_save) params require_grad; the frozen
        # CLIP-ViT backbone does not, so this snapshot is small (a few MB at most)
        # regardless of how many epochs are checked.
        return {
            name: param.detach().to("cpu").clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.eval_dataset is None:
            return control

        epoch_raw = float(state.epoch) if state.epoch is not None else np.nan
        if np.isnan(epoch_raw):
            return control
        epoch_int = int(max(1, round(epoch_raw)))
        val_ce_loss = compute_dataset_ce_loss(
            model=model,
            eval_ds=self.eval_dataset,
            batch_size=int(args.per_device_eval_batch_size),
        )

        if self.track_best_epoch and not np.isnan(val_ce_loss) and val_ce_loss < self.best_val_ce:
            self.best_val_ce = float(val_ce_loss)
            self.best_epoch = epoch_int
            self.best_state_dict = self._snapshot_trainable_params(model)

        learning_rate = np.nan
        if self.trainer is not None and getattr(self.trainer, "optimizer", None) is not None:
            if len(self.trainer.optimizer.param_groups) > 0:
                learning_rate = float(self.trainer.optimizer.param_groups[0].get("lr", np.nan))

        row = {
            "method_name": self.method_name,
            "display_name": self.display_name,
            "step_id": int(self.step_idx + 1),
            "epoch": epoch_int,
            "val_ce_loss": float(val_ce_loss) if not np.isnan(val_ce_loss) else np.nan,
            "val_total_loss": np.nan,
            "learning_rate": learning_rate,
        }
        self.epoch_rows.append(row)
        print(
            f"[val ce] method={self.method_name} | step={row['step_id']} | "
            f"epoch={row['epoch']} | val_ce={row['val_ce_loss']:.6f} | lr={row['learning_rate']:.6g}"
        )

        # B1 (task 3 decision doc, 2026-08-17): patience-since-best-val-CE
        # stop, gated the same way best-epoch tracking is (self.track_best_epoch)
        # so it can only ever fire once self.best_epoch is already set -- the
        # reload in train_with_trainer() always uses self.best_state_dict, so
        # stopping here never changes which epoch's weights end up in the
        # final model. adaptive_early_stop_applies_to_method() restricts this
        # to plain "simple_avg" only -- see that function's own comment for
        # why simple_avg_factor_orth is explicitly excluded rather than
        # relying only on its late-trending best epoch to never trip patience.
        if (
            ADAPTIVE_PER_STEP_EARLY_STOP_ENABLED
            and adaptive_early_stop_applies_to_method(self.method_name)
            and self.track_best_epoch
            and self.best_epoch is not None
            and epoch_int >= ADAPTIVE_PER_STEP_EARLY_STOP_MIN_EPOCH
            and (epoch_int - self.best_epoch) >= ADAPTIVE_PER_STEP_EARLY_STOP_PATIENCE
        ):
            print(
                f"[adaptive early stop] method={self.method_name} | step={row['step_id']} | "
                f"stopping at epoch {epoch_int} (best_epoch={self.best_epoch}, "
                f"patience={ADAPTIVE_PER_STEP_EARLY_STOP_PATIENCE}) -- best-val-CE snapshot "
                f"already captured and unaffected by stopping here."
            )
            control.should_training_stop = True

        model.train()
        return control


def train_with_trainer(
    model,
    train_ds,
    eval_ds,
    output_dir,
    epochs,
    lr,
    batch_size,
    accum_steps,
    trainer_cls=Trainer,
    display_name=None,
    epoch_loss_records=None,
    best_epoch_selection_records=None,
    rankext_new_block_warmup_epochs=None,
    rankext_new_block_warmup_diagnostic_records=None,
    **trainer_kwargs,
):
    args = get_training_args(
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accum_steps=accum_steps,
        train_dataset_len=len(train_ds),
        eval_strategy="epoch",
    )

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        **trainer_kwargs,
    )

    # PRE-THESIS FIX 1: best-epoch selection is only meaningful with an eval_ds to
    # measure val CE against (same gate the old load_best_model_at_end flag used).
    track_best_epoch = bool(USE_BEST_EPOCH_SELECTION and eval_ds is not None)

    epoch_callback = None
    if eval_ds is not None:
        epoch_callback = EpochValidationCallback(
            method_name=trainer_kwargs.get("method_name", getattr(model, "_method_name", "unknown")),
            display_name=(
                trainer_kwargs.get("display_name")
                or display_name
                or trainer_kwargs.get("method_name", getattr(model, "_method_name", "unknown"))
            ),
            step_idx=int(trainer_kwargs.get("step_idx", -1)),
            eval_dataset=eval_ds,
            eval_batch_size=int(args.per_device_eval_batch_size),
            track_best_epoch=track_best_epoch,
        )
        trainer.add_callback(epoch_callback)
        epoch_callback.bind_trainer(trainer)

    # RANKEXT_NEW_BLOCK_WARMUP_ENABLED (analysis_rankext_plain/): only ever
    # non-None for rank_extension call sites (run_rank_extension_variant()) --
    # simple_avg's call to this same function never passes it, so
    # `set_rankext_new_block_warmup_multiplier` is never even invoked for
    # simple_avg, let alone the callback attached. Reset to the neutral 1.0
    # unconditionally BEFORE train() too (not just after), so a stale value
    # left over from a previous rank_extension step can never leak into a
    # step that has the flag off / into any non-rank_extension training call.
    set_rankext_new_block_warmup_multiplier(1.0)
    if rankext_new_block_warmup_epochs is not None:
        warmup_callback = RankExtNewBlockWarmupCallback(
            method_name=trainer_kwargs.get("method_name", getattr(model, "_method_name", "unknown")),
            step_idx=int(trainer_kwargs.get("step_idx", -1)),
            warmup_epochs=float(rankext_new_block_warmup_epochs),
            diagnostic_rows=(
                rankext_new_block_warmup_diagnostic_records
                if rankext_new_block_warmup_diagnostic_records is not None
                else rankext_new_block_warmup_diagnostic_rows
            ),
        )
        trainer.add_callback(warmup_callback)

    trainer.train()

    # Unconditional reset back to the neutral 1.0 -- every eval call in this
    # script (forward_transfer probes, seen-step accuracy, final evaluate_
    # model/evaluate_per_step_accuracy) happens strictly after some
    # train_with_trainer() call returns, so this is the single choke point
    # that guarantees eval never sees a warmed-down new-block contribution.
    # Redundant with RankExtNewBlockWarmupCallback.on_train_end() above by
    # design (same belt-and-suspenders pattern as classifier-row restore).
    set_rankext_new_block_warmup_multiplier(1.0)

    # PRE-THESIS FIX 1: explicitly reload the best-val-CE epoch's trainable-param
    # snapshot into `model` (same object trainer.train() just updated in place),
    # instead of trusting HF Trainer's built-in load_best_model_at_end (see the
    # USE_BEST_EPOCH_SELECTION comment above get_training_args for why that was
    # unreliable). `model` is mutated in place so every caller downstream of
    # train_with_trainer (extract_lora_state, extract_rank_extension_state,
    # merge/eval code, etc.) sees the reload without any other code changes.
    final_epoch_int = int(epochs)
    if (
        track_best_epoch
        and epoch_callback is not None
        and epoch_callback.best_state_dict is not None
    ):
        model_state = dict(model.named_parameters())
        with torch.no_grad():
            for name, snapshot_tensor in epoch_callback.best_state_dict.items():
                if name in model_state:
                    model_state[name].copy_(
                        snapshot_tensor.to(
                            device=model_state[name].device,
                            dtype=model_state[name].dtype,
                        )
                    )
        selected_epoch = int(epoch_callback.best_epoch)
        selected_val_ce = float(epoch_callback.best_val_ce)
    else:
        selected_epoch = final_epoch_int
        selected_val_ce = (
            float(epoch_callback.epoch_rows[-1]["val_ce_loss"])
            if epoch_callback is not None and len(epoch_callback.epoch_rows) > 0
            else np.nan
        )

    if epoch_callback is not None and best_epoch_selection_records is not None:
        final_epoch_val_ce = (
            float(epoch_callback.epoch_rows[-1]["val_ce_loss"])
            if len(epoch_callback.epoch_rows) > 0
            else np.nan
        )
        best_epoch_selection_records.append({
            "method_name": epoch_callback.method_name,
            "display_name": epoch_callback.display_name,
            "step_id": epoch_callback.step_idx + 1,
            "epochs_configured": final_epoch_int,
            "best_epoch_selection_enabled": bool(track_best_epoch),
            "selected_epoch": selected_epoch,
            "selected_val_ce": selected_val_ce,
            "final_epoch_val_ce": final_epoch_val_ce,
            "selected_epoch_lt_final": bool(selected_epoch < final_epoch_int),
        })
        print(
            f"[best-epoch selection] method={epoch_callback.method_name} | "
            f"step={epoch_callback.step_idx + 1} | selected_epoch={selected_epoch}/{final_epoch_int} | "
            f"selected_val_ce={selected_val_ce:.6f} | final_epoch_val_ce={final_epoch_val_ce:.6f}"
        )

    eval_out = trainer.evaluate() if eval_ds is not None else {}

    if epoch_callback is not None and epoch_loss_records is not None and len(epoch_callback.epoch_rows) > 0:
        epoch_loss_records.extend(epoch_callback.epoch_rows)

    return trainer, eval_out


def evaluate_model(model, method_name):
    args = get_training_args(
        output_dir=os.path.join(MODELS_DIR, f"eval_{method_name}"),
        epochs=1,
        lr=LR_LORA,
        batch_size=BATCH_LORA,
        accum_steps=ACCUM_LORA,
        train_dataset_len=None,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    eval_first_out = trainer.evaluate(eval_dataset=eval_first)
    eval_later_out = trainer.evaluate(eval_dataset=eval_later)
    eval_all_out = trainer.evaluate(eval_dataset=eval_all_seen)

    rows = [
        {
            "method": method_name,
            "eval_set": "first_step",
            "accuracy": float(eval_first_out["eval_accuracy"]),
            "loss": float(eval_first_out["eval_loss"]),
        },
        {
            "method": method_name,
            "eval_set": "later_steps",
            "accuracy": float(eval_later_out["eval_accuracy"]),
            "loss": float(eval_later_out["eval_loss"]),
        },
        {
            "method": method_name,
            "eval_set": "all_seen",
            "accuracy": float(eval_all_out["eval_accuracy"]),
            "loss": float(eval_all_out["eval_loss"]),
        },
    ]

    all_results.extend(rows)

    print(pd.DataFrame(rows))
    return rows


# PRE-THESIS FIX 2: true per-CL-step (1..NUM_STEPS) accuracy, plus
# backward_transfer / forward_transfer. These were previously hardcoded to NaN in
# supervisor_selected_accuracy_comparison.csv / final_metrics_all_methods.csv --
# only the 3 aggregated eval groups (first_step/later_steps/all_seen) were ever
# populated, for either family. See the run_simple_avg_variant() /
# run_rank_extension_variant() call sites below for how these are wired in.
FORWARD_TRANSFER_RANDOM_BASELINE = 1.0 / float(CLASSES_PER_STEP)


def evaluate_single_step_accuracy(model, step_idx):
    """Evaluate `model`'s accuracy on exactly one CL step's own 20-class group."""
    args = get_training_args(
        output_dir=os.path.join(MODELS_DIR, "tmp_single_step_eval"),
        epochs=1,
        lr=LR_LORA,
        batch_size=BATCH_LORA,
        accum_steps=ACCUM_LORA,
        train_dataset_len=None,
        eval_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    eval_ds = make_eval_dataset(classes_for_step(step_idx))
    out = trainer.evaluate(eval_dataset=eval_ds)
    return float(out["eval_accuracy"])


def restricted_argmax_accuracy(logits, labels, allowed_class_ids):
    """
    EVAL-PIPELINE AUDIT ADD (analysis_pipeline_audit/report.txt "root cause"
    section). Same accuracy definition as compute_metrics() -- argmax over the
    classifier's logits, compared to the ground-truth label -- except the
    argmax candidate set is masked down to EXACTLY `allowed_class_ids` first,
    instead of being left open over all NUM_CLASSES.

    Why this exists: compute_metrics()'s open-100-way argmax is the standard,
    intended class-incremental-learning evaluation protocol and is NOT being
    replaced here. But it is only a fair cross-step comparison if every
    class's classifier row is on a comparable scale, and this codebase's own
    calibrate_classifier_row_norms() docstring already documents that this
    does not hold in general (WA-style row-norm imbalance), with calibration
    only enabled for the simple_avg family (CALIBRATION_ENABLED_FAMILIES).
    For rank_extension (calibration off), the most-recently-trained class
    group's rows compete on an artificially favorable scale under the open
    argmax, which is the verified mechanism behind "step_5 always looks best"
    (see the report). This function computes the SAME model's accuracy on the
    SAME eval subset with that specific confound removed, so the two numbers
    can be compared side by side per (method, step) without re-training or
    re-evaluating -- it is computed from the identical logits produced by the
    single forward pass in evaluate_per_step_accuracy(), not a second model
    call.

    `logits`: ndarray [N, NUM_CLASSES]. `labels`: ndarray [N] of global class
    ids. `allowed_class_ids`: iterable of the class ids that are actually
    valid answers for this eval subset (everything else is masked to -inf
    before argmax, so it can never win).
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    mask = np.full(logits.shape[1], -np.inf, dtype=np.float64)
    allowed = sorted({int(c) for c in allowed_class_ids})
    mask[allowed] = 0.0
    masked_logits = logits + mask[None, :]
    preds = np.argmax(masked_logits, axis=1)
    return float((preds == labels).mean())


def evaluate_per_step_accuracy(model, method_name):
    """
    Evaluate `model` (intended to be each method's FINAL merged/final model,
    called once after that method's whole training is complete) on each of the
    NUM_STEPS individual CL-step class groups separately, and log the result
    into the module-global per_step_accuracy_rows accumulator (long format:
    method, step_id, accuracy -- this is what the 8-methods x 5-steps accuracy
    heatmap and the CSV `per_step_accuracy` column are built from).

    Self-contained (builds its own eval-only Trainer) rather than reusing
    evaluate_seen_step_accuracies(), which is defined later in this script but
    needs to be callable here since the simple_avg family's training loop runs
    (and calls this function) before that later definition is reached.

    EVAL-PIPELINE AUDIT ADD: also computes and logs, into the module-global
    per_step_accuracy_restricted_rows accumulator, a closed-set companion
    accuracy per step (see restricted_argmax_accuracy() docstring) -- from the
    SAME trainer.predict() call used for the existing open-set number, so this
    adds no extra forward passes and cannot change the pre-existing open-set
    per_step_map/per_step_accuracy_rows values at all (same model state, same
    eval_ds, same uniform evaluation function across every step; the only new
    thing is a second, masked accuracy computed from the same logits).

    Returns a {step_idx: accuracy_fraction} map (0..1, NOT percent) for the
    caller to also use in backward_transfer/forward_transfer computations --
    this is the OPEN-set map, unchanged from before this diagnostic was added.
    """
    args = get_training_args(
        output_dir=os.path.join(MODELS_DIR, "tmp_per_step_eval"),
        epochs=1,
        lr=LR_LORA,
        batch_size=BATCH_LORA,
        accum_steps=ACCUM_LORA,
        train_dataset_len=None,
        eval_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    per_step_map = {}
    restricted_map = {}
    for step_idx in range(NUM_STEPS):
        current_classes = classes_for_step(step_idx)
        eval_ds = make_eval_dataset(current_classes)
        pred_out = trainer.predict(eval_ds)
        logits = np.asarray(pred_out.predictions)
        labels = np.asarray(pred_out.label_ids)
        # Same computation compute_metrics() would have done -- kept
        # inline (not called through compute_metrics) so both the open and
        # restricted numbers come from exactly one forward pass's logits.
        open_acc = float((np.argmax(logits, axis=1) == labels).mean())
        per_step_map[step_idx] = open_acc
        restricted_map[step_idx] = restricted_argmax_accuracy(logits, labels, current_classes)

    for step_idx in range(NUM_STEPS):
        acc = per_step_map.get(step_idx, np.nan)
        per_step_accuracy_rows.append({
            "method": method_name,
            "step_id": int(step_idx + 1),
            "accuracy": float(acc) * 100.0 if not np.isnan(acc) else np.nan,
        })
        racc = restricted_map.get(step_idx, np.nan)
        per_step_accuracy_restricted_rows.append({
            "method": method_name,
            "step_id": int(step_idx + 1),
            "accuracy_restricted": float(racc) * 100.0 if not np.isnan(racc) else np.nan,
            "accuracy_open": float(acc) * 100.0 if not np.isnan(acc) else np.nan,
            "recency_bias_gap": (float(acc) - float(racc)) * 100.0 if not (np.isnan(acc) or np.isnan(racc)) else np.nan,
        })
    return per_step_map


def compute_backward_transfer(diagonal_map, final_map):
    """
    Standard GEM-style backward transfer (Lopez-Paz & Ranzato 2017): mean over
    tasks i=1..T-1 of (final accuracy on task i - accuracy on task i measured
    right when it was learned). Task T (the last-learned step) is excluded --
    there is no "later" checkpoint to compare it against. Positive = later
    training helped earlier tasks; negative = forgetting.

    `diagonal_map`/`final_map` are {step_idx: accuracy_fraction} maps (0..1).
    Returns NaN (never a fabricated 0.0) if fewer than 1 comparable step pair is
    available.
    """
    if not diagonal_map or not final_map:
        return np.nan
    common_steps = sorted(set(diagonal_map.keys()) & set(final_map.keys()))
    last_step = max(final_map.keys())
    deltas = [
        final_map[s] - diagonal_map[s]
        for s in common_steps
        if s != last_step and not np.isnan(diagonal_map[s]) and not np.isnan(final_map[s])
    ]
    if len(deltas) == 0:
        return np.nan
    return float(np.mean(deltas))


def compute_forward_transfer(probe_map):
    """
    Standard-style forward transfer: mean over tasks i=2..T of (zero-shot
    accuracy on task i's class group, using the model as it stood right BEFORE
    training on task i, minus the random-chance baseline for a
    CLASSES_PER_STEP-way classification subset). Only meaningful for a model
    that genuinely carries state forward between steps (rank_extension); the
    caller should pass an empty/None probe_map (-> NaN, not a fabricated
    number) for merge-based families like simple_avg, where every step starts
    from the same fresh pretrained backbone and there is no well-defined
    "model before training step i" that differs across i.

    `probe_map` is a {step_idx: accuracy_fraction} map (0..1).
    """
    if not probe_map:
        return np.nan
    vals = [v - FORWARD_TRANSFER_RANDOM_BASELINE for v in probe_map.values() if not np.isnan(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


# In[ ]:


def normalize_module_name(name):
    prefixes = [
        "base_model.model.",
        "model.",
    ]

    out = name

    for p in prefixes:
        if out.startswith(p):
            out = out[len(p):]

    return out

def extract_lora_state(model):
    """
    Extract:
    - LoRA delta_W per target module
    - classifier weights

    PEFT convention:
    A shape = [r, in_features]
    B shape = [out_features, r]
    delta_W = B @ A * scaling
    """
    state = {
        "deltas": {},
        "lora_A": {},
        "lora_B": {},
        "scaling": {},
        "classifier_weight": None,
        "classifier_bias": None,
    }

    for name, module in model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )

        if not has_lora:
            continue

        adapter_name = "default"
        A = module.lora_A[adapter_name].weight.detach().cpu().float().clone()
        B = module.lora_B[adapter_name].weight.detach().cpu().float().clone()

        scaling = (
            module.scaling[adapter_name]
            if isinstance(module.scaling, dict)
            else module.scaling
        )

        scaling = float(scaling)
        delta = scaling * (B @ A)

        plain_name = normalize_module_name(name)
        state["deltas"][plain_name] = delta.clone()
        state["lora_A"][plain_name] = A
        state["lora_B"][plain_name] = B
        state["scaling"][plain_name] = scaling

    for name, tensor in model.state_dict().items():
        if "classifier.modules_to_save.default.weight" in name:
            state["classifier_weight"] = tensor.detach().cpu().clone()

        if "classifier.modules_to_save.default.bias" in name:
            state["classifier_bias"] = tensor.detach().cpu().clone()

    return state

def get_submodule_by_name(model, module_name):
    module_name = normalize_module_name(module_name)

    current = model

    for part in module_name.split("."):
        if part == "":
            continue
        current = getattr(current, part)

    return current

def simple_average_deltas(step_states):
    keys = sorted(step_states[0]["deltas"].keys())
    merged = {}

    for key in keys:
        vals = []

        for state in step_states:
            if key in state["deltas"]:
                vals.append(state["deltas"][key].float())

        merged[key] = torch.stack(vals, dim=0).mean(dim=0)

    return merged

# STRICT-REVIEW ADD (B1, 2026-07-17): merge-mechanism diagnostics for the
# simple_avg family, to PROVE (not just hypothesize) whether factor-orth's
# old-class ("first_step") damage is caused by later steps' orthogonality-
# constrained deltas diluting/cancelling step 1's own contribution once
# simple_average_deltas() averages them together -- see
# analysis_revert_run/report.txt Comment 1c, which flagged this as
# "plausible but unprovable from currently saved data" for lack of exactly
# this logging.
#
# Purely diagnostic: reads already-extracted CPU tensors out of step_states
# (the output of extract_lora_state(), already computed for every simple_avg
# variant regardless of this change) and merged_delta (already computed by
# simple_average_deltas() regardless of this change). Every operation here is
# a norm/dot-product on small in-memory matrices (numpy-speed, no CUDA sync,
# no forward/backward pass, no new model instantiation) -- negligible cost,
# and nothing here can affect model weights, gradients, or the training loop:
# it runs strictly AFTER train_independent_loras() has already returned.
_MERGE_MECHANISM_CSV_INITIALIZED = set()


def log_merge_mechanism(method_name, step_states, merged_delta, csv_path):
    """Per (method, target_module): ||dW_t|| and cos(dW_1, dW_t) for every task
    t=1..N BEFORE the merge (from step_states, one row per task), plus the
    merged delta's own norm and its cosine similarity to step 1's delta AFTER
    the merge (one 'MERGED' summary row per module). merged_norm_over_mean_
    individual_norm < 1 with cos_dW1_merged << 1 (or negative) is the direct,
    numeric signature of destructive dilution/cancellation of step 1's
    direction; merged_norm_over_mean_individual_norm close to 1 with
    cos_dW1_merged close to 1 would instead support "orth is redundant but
    harmless" for this family. Appends to csv_path; the file is reset once per
    process (first call) so repeated method calls within one run accumulate
    correctly without carrying over rows from a stale previous run.
    """
    if len(step_states) == 0:
        return None

    keys = sorted(step_states[0]["deltas"].keys())
    n_tasks = len(step_states)
    rows = []

    for key in keys:
        task_deltas = [step_states[t]["deltas"].get(key) for t in range(n_tasks)]
        task_deltas = [d.float() if d is not None else None for d in task_deltas]
        dW1 = task_deltas[0]
        norm1 = float(torch.linalg.norm(dW1)) if dW1 is not None else float("nan")
        individual_norms = [float(torch.linalg.norm(d)) for d in task_deltas if d is not None]
        mean_individual_norm = float(np.mean(individual_norms)) if individual_norms else float("nan")

        for t, dWt in enumerate(task_deltas):
            if dWt is None:
                continue
            norm_t = float(torch.linalg.norm(dWt))
            if dW1 is not None and norm1 > 0 and norm_t > 0:
                cos_1t = 1.0 if t == 0 else float(
                    torch.dot(dW1.reshape(-1), dWt.reshape(-1)) / (norm1 * norm_t)
                )
            else:
                cos_1t = float("nan")
            rows.append({
                "method": method_name, "target_module": key, "task_step": t + 1,
                "phase": "pre_merge", "dW_norm": norm_t,
                "dW_norm_over_dW1_norm": (norm_t / norm1) if norm1 > 0 else float("nan"),
                "cos_dW1_dWt": cos_1t, "n_tasks_in_merge": n_tasks,
            })

        merged = merged_delta.get(key)
        if merged is not None:
            merged_norm = float(torch.linalg.norm(merged))
            cos_1_merged = (
                float(torch.dot(dW1.reshape(-1), merged.reshape(-1)) / (norm1 * merged_norm))
                if (dW1 is not None and norm1 > 0 and merged_norm > 0) else float("nan")
            )
            rows.append({
                "method": method_name, "target_module": key, "task_step": "MERGED",
                "phase": "post_merge", "dW_norm": merged_norm,
                "dW_norm_over_dW1_norm": (merged_norm / norm1) if norm1 > 0 else float("nan"),
                "cos_dW1_dWt": cos_1_merged, "n_tasks_in_merge": n_tasks,
                "merged_norm_over_mean_individual_norm": (
                    merged_norm / mean_individual_norm if mean_individual_norm > 0 else float("nan")
                ),
            })

    df = pd.DataFrame(rows)
    write_header = csv_path not in _MERGE_MECHANISM_CSV_INITIALIZED
    if write_header and os.path.exists(csv_path):
        os.remove(csv_path)
    _MERGE_MECHANISM_CSV_INITIALIZED.add(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    return df

def column_normalize(mat, eps=1e-12):
    return mat / torch.linalg.norm(mat, dim=0, keepdim=True).clamp_min(eps)

def column_decouple_delta(delta, eps=1e-12):
    magnitudes = torch.linalg.norm(delta, dim=0, keepdim=True).clamp_min(eps)
    directions = delta / magnitudes
    return magnitudes, directions

def mean_pairwise_cosine(flat_vectors, eps=1e-12):
    if len(flat_vectors) < 2:
        return None

    sims = []

    for i in range(len(flat_vectors)):
        vi = flat_vectors[i]
        vi = vi / torch.linalg.norm(vi).clamp_min(eps)

        for j in range(i + 1, len(flat_vectors)):
            vj = flat_vectors[j]
            vj = vj / torch.linalg.norm(vj).clamp_min(eps)
            sims.append(torch.dot(vi, vj).item())

    if len(sims) == 0:
        return None

    return float(sum(sims) / len(sims))

def orthogonalize_task_directions(task_deltas, eps=1e-12):
    mags = []
    dirs = []
    flat_dirs = []

    for delta in task_deltas:
        mag, direction = column_decouple_delta(delta, eps=eps)
        mags.append(mag)
        dirs.append(direction)
        flat_dirs.append(direction.reshape(-1))

    ortho_flat = []

    for v in flat_dirs:
        u = v.clone()

        for q in ortho_flat:
            u = u - torch.dot(u, q) * q

        n = torch.linalg.norm(u)

        if n < eps:
            u = v / torch.linalg.norm(v).clamp_min(eps)
        else:
            u = u / n

        ortho_flat.append(u)

    ortho_dirs = [
        column_normalize(u.reshape_as(dirs[i]), eps=eps)
        for i, u in enumerate(ortho_flat)
    ]

    return mags, ortho_dirs

def do_merge_deltas(step_states, eps=1e-12, use_orthogonalize=True, verbose=True):
    """
    DO-Merging-inspired: layer-wise orthogonalized, column-wise decoupled LoRA delta merging.
    """
    all_keys = sorted(set().union(*[set(s["deltas"].keys()) for s in step_states]))
    merged = {}

    layer_delta_counts = []
    cos_before_values = []
    cos_after_values = []
    col_mag_means = []
    col_mag_stds = []

    for key in all_keys:
        task_deltas = []

        for state in step_states:
            if key in state["deltas"]:
                task_deltas.append(state["deltas"][key].detach().cpu().float())

        if len(task_deltas) == 0:
            continue

        layer_delta_counts.append(len(task_deltas))

        mags_before = []
        dirs_before = []

        for delta in task_deltas:
            mag, direction = column_decouple_delta(delta, eps=eps)
            mags_before.append(mag)
            dirs_before.append(direction)

        flat_before = [d.reshape(-1) for d in dirs_before]
        cos_before = mean_pairwise_cosine(flat_before, eps=eps)

        if cos_before is not None:
            cos_before_values.append(cos_before)

        if len(task_deltas) == 1:
            merged[key] = task_deltas[0].clone()
            continue

        if use_orthogonalize:
            mags, dirs = orthogonalize_task_directions(task_deltas, eps=eps)
        else:
            mags = mags_before
            dirs = dirs_before

        flat_after = [d.reshape(-1) for d in dirs]
        cos_after = mean_pairwise_cosine(flat_after, eps=eps)

        if cos_after is not None:
            cos_after_values.append(cos_after)

        mag_stack = torch.stack(mags, dim=0)
        col_mag_means.append(float(mag_stack.mean().item()))
        col_mag_stds.append(float(mag_stack.std(unbiased=False).item()))

        merged_mag = mag_stack.mean(dim=0)
        merged_dir = torch.stack(dirs, dim=0).mean(dim=0)
        merged_dir = column_normalize(merged_dir, eps=eps)

        merged_delta = merged_dir * merged_mag

        if merged_delta.shape != task_deltas[0].shape:
            raise ValueError(
                f"Shape mismatch for {key}: merged={tuple(merged_delta.shape)} vs ref={tuple(task_deltas[0].shape)}"
            )

        merged[key] = merged_delta

    if verbose:
        print(f"[DO-Merging] merged {len(merged)} layers")

        if len(layer_delta_counts) > 0:
            mean_tasks = sum(layer_delta_counts) / len(layer_delta_counts)
            print(f"[DO-Merging] avg task deltas per layer: {mean_tasks:.2f}")

        if len(cos_before_values) > 0:
            print(
                f"[DO-Merging] avg pairwise cosine before orthogonalization: {sum(cos_before_values) / len(cos_before_values):.6f}"
            )
        else:
            print("[DO-Merging] avg pairwise cosine before orthogonalization: n/a")

        if len(cos_after_values) > 0:
            print(
                f"[DO-Merging] avg pairwise cosine after orthogonalization: {sum(cos_after_values) / len(cos_after_values):.6f}"
            )
        else:
            print("[DO-Merging] avg pairwise cosine after orthogonalization: n/a")

        if len(col_mag_means) > 0:
            print(
                f"[DO-Merging] column magnitude mean/std across layers: {sum(col_mag_means) / len(col_mag_means):.6f} / {sum(col_mag_stds) / len(col_mag_stds):.6f}"
            )
        else:
            print("[DO-Merging] column magnitude mean/std across layers: n/a")

    return merged

def apply_deltas_to_base(merged_deltas, step_states):
    """
    Apply merged LoRA deltas to a fresh CLIP-ViT model and stitch classifier rows.
    """
    model = fresh_pretrained_model()

    with torch.no_grad():
        for key, delta in merged_deltas.items():
            try:
                module = get_submodule_by_name(model, key)
            except Exception as e:
                print("Could not find module:", key, "|", e)
                continue

            if not hasattr(module, "weight"):
                print("Module has no weight:", key)
                continue

            module.weight.add_(
                delta.to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            )

        for step_idx, state in enumerate(step_states):
            classes = classes_for_step(step_idx)

            if state["classifier_weight"] is None:
                print("Missing classifier for step", step_idx + 1)
                continue

            w = state["classifier_weight"].to(model.classifier.weight.device)
            b = state["classifier_bias"].to(model.classifier.bias.device)

            for c in classes:
                model.classifier.weight[c].copy_(w[c])
                model.classifier.bias[c].copy_(b[c])

    return model

def log_classifier_row_norm_diagnostics(model, method_name, phase, eps=1e-8):
    """
    FIX 1 diagnostic (analysis_recency_fix/report.txt): appends one row per CL
    step to the module-global classifier_row_norm_diagnostic_rows
    accumulator, recording that step's classifier weight-row-block mean norm
    and its ratio to step 1's mean norm -- direct numerical evidence for "are
    later steps' rows several times larger than earlier steps'?" (R5/WIDERANK
    predates this instrumentation and only has parameter shapes on disk, not
    norms; see report.txt Task A.2). Read-only: never mutates the model. Call
    with phase="pre_calibration" before any calibration is applied
    (unconditionally, for every method) and phase="post_calibration" after
    calibrate_classifier_row_norms() runs (only for apply_calibration=True
    methods).
    """
    with torch.no_grad():
        row_norms = model.classifier.weight.norm(dim=1).detach().cpu()
    step_means = [
        float(row_norms[list(classes_for_step(step_idx))].mean().item())
        for step_idx in range(NUM_STEPS)
    ]
    step1_mean = max(step_means[0], eps)
    for step_idx, m in enumerate(step_means):
        classifier_row_norm_diagnostic_rows.append({
            "method": method_name,
            "step_id": int(step_idx + 1),
            "phase": phase,
            "mean_row_norm": m,
            "row_norm_ratio_vs_step1": m / step1_mean,
        })


def calibrate_classifier_row_norms(model, eps=1e-8, mode="global", uses_kd=False, method_name=None):
    """
    ACCURACY-PUSH CHANGE 2: rehearsal-free, post-merge-only classifier
    calibration (WA-style weight alignment, Zhao et al. 2020, "Maintaining
    Discrimination and Fairness in Class Incremental Learning").

    mode="global" (default, unchanged behavior): rescales each CL step's
    20-class weight-row block so its mean row norm matches ONE target norm
    computed as the mean over ALL 100 rows. Pure post-hoc correction on the
    already-merged/stitched classifier -- no retraining, no rehearsal data.

    mode="regime_grouped" (FIX 1, analysis_recency_fix/report.txt,
    rank_extension only -- see RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED above
    for the full rationale): instead of ONE global target mixing every step,
    partitions the NUM_STEPS step-blocks into homogeneous training-regime
    GROUPS and computes an independent target norm per group. When uses_kd is
    True: group A = {step 1} (always teacher-less -- there is no prior-step
    checkpoint yet at step 1), group B = {steps 2..NUM_STEPS} (all trained
    against a KD teacher). Group A is a singleton, so its "target" equals its
    own current mean -- a deliberate no-op that leaves step 1's rows
    completely untouched by later steps' KD-regime statistics (this is
    exactly the cross-regime mixing that corrupted the KD variants under the
    old global mode; see the POST-INCIDENT FIX note this function used to
    carry). Group B is calibrated to ITS OWN shared mean, directly equalizing
    norm growth across steps 2..NUM_STEPS. When uses_kd is False, there is
    only one training regime across every step, so this mode reduces to the
    same result as mode="global".

    Rationale specific to this codebase: for the simple_avg family,
    apply_deltas_to_base() stitches together 5 classifier row-blocks that each
    came from a SEPARATE freshly-initialized nn.Linear (fresh_pretrained_model()
    is called fresh per step in train_independent_loras()), trained under
    different conditions (KD vs not, orth vs not) and, in every case, trained
    with a 100-way softmax where 80 of the 100 classes are permanently absent
    that step (pure negatives, never positive) -- a well-known recipe for
    severe cross-group row-norm imbalance; mode="global" is used for this
    family. For rank_extension the classifier is shared/incremental rather
    than independently re-initialized, and mode="regime_grouped" is used
    instead (see above).

    Only the weight rows are rescaled, not the bias, matching the original WA
    formulation (bias reflects class prior/frequency, not representation
    scale, so rescaling it is not part of the correction).

    Logs pre- and post-calibration row-norm diagnostics via
    log_classifier_row_norm_diagnostics() when method_name is given -- a
    read-only side effect on the module-global
    classifier_row_norm_diagnostic_rows accumulator, not on the model.
    """
    if method_name is not None:
        log_classifier_row_norm_diagnostics(model, method_name, phase="pre_calibration", eps=eps)

    with torch.no_grad():
        W = model.classifier.weight
        row_norms = W.norm(dim=1)

        if mode == "regime_grouped" and uses_kd:
            groups = [[0], list(range(1, NUM_STEPS))]
        else:
            groups = [list(range(NUM_STEPS))]

        for group in groups:
            group_idx = torch.tensor(
                [c for step_idx in group for c in classes_for_step(step_idx)],
                device=W.device,
                dtype=torch.long,
            )
            group_target_norm = float(row_norms[group_idx].mean().item())

            for step_idx in group:
                idx = torch.tensor(
                    list(classes_for_step(step_idx)),
                    device=W.device,
                    dtype=torch.long,
                )
                step_norm = float(row_norms[idx].mean().clamp_min(eps).item())
                scale = group_target_norm / step_norm
                W[idx] *= scale

    if method_name is not None:
        log_classifier_row_norm_diagnostics(model, method_name, phase="post_calibration", eps=eps)

    return model


def calibrate_classifier_row_norms_confidence_weighted(
    model, epoch_loss_rows, method_name, eps=1e-8, uses_kd=False,
    gamma=0.65, boost_min=0.85, boost_max=1.3,
):
    """
    FIX 2 (analysis_recency_fix2/report.txt): confidence-weighted regime-
    grouped classifier calibration for rank_extension. NEW function, not a
    patch to calibrate_classifier_row_norms() -- mode="regime_grouped" (FIX 1)
    remains fully runnable/comparable via that function unchanged; this is a
    separate, genuinely different targeting rule layered on the SAME grouping.

    Uses the identical step grouping as calibrate_classifier_row_norms(
    mode="regime_grouped"): when uses_kd is True, group A = {step 1}
    (singleton, teacher-less, always a no-op -- report.txt Task B found no
    evidence this was hurting step 1 relative to the calibrated steps, so FIX
    2 does not change it); group B = {steps 2..NUM_STEPS}. When uses_kd is
    False, one group covers every step (same as FIX 1 reduces to for the
    non-KD variants).

    Within each non-singleton group, IF uses_kd is True, instead of rescaling
    every step to the SAME flat group-mean row norm (FIX 1's plain
    mean-matching), each step's TARGET norm is the group mean times a
    bounded, monotonic boost factor derived from that step's own final-epoch
    validation CE loss relative to its group's mean validation CE loss (the
    signal report.txt Task B found DOES track the gap's bowl shape for the KD
    variants -- val_ce jumps sharply at step 2, the worst-affected step, and
    recovers toward step 5):

        relative_difficulty = step_val_ce / group_mean_val_ce   (>1: this step
            fit its own 20-way validation set WORSE than its groupmates did;
            <1: better)
        boost = clamp(relative_difficulty ** gamma, boost_min, boost_max)
        target_norm = group_target_norm(plain mean, as in FIX 1) * boost

    gamma < 1 dampens the boost so a step with e.g. 2x the group's mean val_ce
    does not get a full 2x norm target (which risks overshooting into
    dominating the open-argmax competition for OTHER steps' images, not just
    recovering its own); boost_min/boost_max additionally hard-clip the
    per-step multiplier to a narrow, safe band around 1.0. A step with no
    logged val_ce_loss (missing data) gets boost=1.0, i.e. falls back exactly
    to FIX 1's plain group-mean target for that step -- never a worse-tested
    failure mode than FIX 1 already validated as safe.

    IF uses_kd is False, boost is ALWAYS 1.0 for every step -- report.txt Task
    B found NO clean val_ce_loss/gap correlation for the non-KD variants (e.g.
    rank_extension's step 3 has the LOWEST val_ce of all 5 steps yet the
    WORST open-vs-restricted gap), so boosting on that signal there would be
    acting on noise, not evidence. With boost forced to 1.0 this function is
    numerically IDENTICAL to calibrate_classifier_row_norms(mode=
    "regime_grouped") for every non-KD rank_extension variant -- by
    construction, not by tuning, so FIX 2 is provably no worse than FIX 1 for
    those two variants.

    Only the weight rows are rescaled, not the bias (same convention as
    calibrate_classifier_row_norms()). Logs pre/post row-norm diagnostics via
    log_classifier_row_norm_diagnostics() (identical table/columns as FIX 1,
    so the next run's Task A row-norm review works unchanged) plus a second,
    FIX-2-specific diagnostic row per (method, step_id) into the module-global
    classifier_confidence_calibration_diagnostic_rows accumulator recording
    the val_ce_loss, relative_difficulty, and boost actually used.
    """
    log_classifier_row_norm_diagnostics(model, method_name, phase="pre_calibration", eps=eps)

    method_epoch_rows = [r for r in epoch_loss_rows if r.get("method_name") == method_name]
    final_val_ce_by_step = {}
    best_epoch_by_step = {}
    for r in method_epoch_rows:
        val_ce = r.get("val_ce_loss", float("nan"))
        if val_ce is None or (isinstance(val_ce, float) and np.isnan(val_ce)):
            continue
        step_id = int(r["step_id"])
        epoch = int(r["epoch"])
        if step_id not in best_epoch_by_step or epoch >= best_epoch_by_step[step_id]:
            best_epoch_by_step[step_id] = epoch
            final_val_ce_by_step[step_id] = float(val_ce)

    with torch.no_grad():
        W = model.classifier.weight
        row_norms = W.norm(dim=1)

        if uses_kd:
            groups = [[0], list(range(1, NUM_STEPS))]
        else:
            groups = [list(range(NUM_STEPS))]

        for group in groups:
            group_idx = torch.tensor(
                [c for step_idx in group for c in classes_for_step(step_idx)],
                device=W.device,
                dtype=torch.long,
            )
            group_target_norm = float(row_norms[group_idx].mean().item())

            group_val_ces = [final_val_ce_by_step[s + 1] for s in group if (s + 1) in final_val_ce_by_step]
            group_mean_val_ce = float(np.mean(group_val_ces)) if len(group_val_ces) > 0 else None

            for step_idx in group:
                idx = torch.tensor(
                    list(classes_for_step(step_idx)),
                    device=W.device,
                    dtype=torch.long,
                )
                step_norm = float(row_norms[idx].mean().clamp_min(eps).item())

                step_val_ce = final_val_ce_by_step.get(step_idx + 1)
                # report.txt Task B: the val_ce_loss -> gap correlation was
                # only established for the KD variants (val_ce jumps sharply
                # at step 2 and recovers toward step 5, matching the gap's
                # bowl shape almost exactly). For non-KD variants Task B found
                # NO clean correlation (e.g. rank_extension's step 3 has the
                # LOWEST val_ce of all 5 steps yet the WORST open-vs-restricted
                # gap) -- boosting on an uncorrelated signal there risks doing
                # active harm, not just failing to help. So uses_kd gates the
                # boost entirely: non-KD groups always get boost=1.0, which
                # makes this function numerically IDENTICAL to
                # calibrate_classifier_row_norms(mode="regime_grouped") for
                # every non-KD rank_extension variant -- provably no worse
                # than FIX 1 there, by construction, not just by tuning.
                if uses_kd and len(group) > 1 and step_val_ce is not None and group_mean_val_ce is not None and group_mean_val_ce > eps:
                    relative_difficulty = step_val_ce / group_mean_val_ce
                    boost = float(np.clip(relative_difficulty ** gamma, boost_min, boost_max))
                else:
                    relative_difficulty = 1.0
                    boost = 1.0

                step_target_norm = group_target_norm * boost
                scale = step_target_norm / step_norm
                W[idx] *= scale

                classifier_confidence_calibration_diagnostic_rows.append({
                    "method": method_name,
                    "step_id": int(step_idx + 1),
                    "val_ce_loss": step_val_ce,
                    "group_mean_val_ce_loss": group_mean_val_ce,
                    "relative_difficulty": relative_difficulty,
                    "boost_factor": boost,
                    "group_mean_row_norm": group_target_norm,
                    "target_row_norm": step_target_norm,
                })

    log_classifier_row_norm_diagnostics(model, method_name, phase="post_calibration", eps=eps)

    return model


# RANKEXT DRIFT DIAGNOSTIC (decision doc, 2026-08-05): bias-offset + feature-
# alignment checks for the final calibrated rank_extension model. Answers,
# with a real number instead of architecture inference: is the open-argmax
# collapse a classifier-BIAS-offset problem (untested by any prior fix --
# calibrate_classifier_row_norms()/_confidence_weighted() above only ever
# rescale model.classifier.WEIGHT rows, never .bias) or a shared-backbone
# FEATURE-DRIFT problem (old-step images' features pulled toward the newest
# step's classifier directions)? Runs for all 4 EXISTING rank_extension
# methods (not just the 2 new feature-anchor ones) so the non-KD vs. KD
# contrast this run needs is a direct read of the two output CSVs, not a
# separate analysis. Purely read-only: model.eval(), no_grad, no weights
# touched, does not affect calibration, merge, or evaluate_model()'s own
# forward passes below.
rankext_bias_diagnostic_rows = []
rankext_feature_alignment_diagnostic_rows = []
RANKEXT_DRIFT_DIAGNOSTICS_ENABLED = True


def log_rankext_drift_diagnostics(model, method_name, eps=1e-8):
    """Bias-offset + feature-alignment diagnostic on the final (post-
    calibration) rank_extension model. Appends to the two module-global lists
    above; does not mutate model weights or return anything."""
    with torch.no_grad():
        W, b = model.classifier.weight, model.classifier.bias
        grand_bias_mean = float(b.mean().item())
        for step_idx in range(NUM_STEPS):
            idx = torch.tensor(list(classes_for_step(step_idx)), device=b.device, dtype=torch.long)
            step_bias_mean = float(b[idx].mean().item())
            rankext_bias_diagnostic_rows.append({
                "method": method_name,
                "step_id": step_idx + 1,
                "step_bias_mean": step_bias_mean,
                "grand_bias_mean": grand_bias_mean,
                "bias_offset_vs_grand_mean": step_bias_mean - grand_bias_mean,
            })

        W_norm = W / W.norm(dim=1, keepdim=True).clamp_min(eps)
        recent_idx = torch.tensor(list(classes_for_step(NUM_STEPS - 1)), device=W.device, dtype=torch.long)
        recent_rows_norm = W_norm[recent_idx]

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        for step_idx in range(NUM_STEPS - 1):  # old steps only -- last step has no "more recent" step to compare against
            val_ds = make_val_dataset(classes_for_step(step_idx))
            if val_ds is None or len(val_ds) == 0:
                continue
            loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
            own_sum, recent_sum, n = 0.0, 0.0, 0
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                feats = model.vision_model(pixel_values=pixel_values, return_dict=True).pooler_output
                feats_norm = feats / feats.norm(dim=1, keepdim=True).clamp_min(eps)
                own_cos = (feats_norm * W_norm[labels]).sum(dim=1)
                recent_cos = (feats_norm @ recent_rows_norm.T).max(dim=1).values
                own_sum += float(own_cos.sum().item())
                recent_sum += float(recent_cos.sum().item())
                n += int(labels.shape[0])
            if n > 0:
                rankext_feature_alignment_diagnostic_rows.append({
                    "method": method_name,
                    "old_step_id": step_idx + 1,
                    # PROTOCOL-DEPTH VALIDATION (2026-08-24): explicit age
                    # column (steps since this class group's own step
                    # finished, i.e. how many later block-addition events it
                    # has been exposed to) -- this loop already generalized
                    # correctly to any NUM_STEPS (range(NUM_STEPS-1) ->
                    # old_step_id 1..19 under 20x5); this field just makes
                    # the resulting age axis explicit instead of requiring
                    # the reader to compute NUM_STEPS - old_step_id by hand.
                    "age_in_steps": NUM_STEPS - (step_idx + 1),
                    "mean_cos_own_class_row": own_sum / n,
                    "mean_cos_best_recent_step_row": recent_sum / n,
                    "own_minus_recent_cos_gap": (own_sum - recent_sum) / n,
                    "n_images": n,
                })
        if was_training:
            model.train()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# In[ ]:


if METHODS_TO_RUN.get("full_finetune", False):
    full_ft_model = fresh_pretrained_model()

    for step_idx in range(NUM_STEPS):
        train_ds = make_train_dataset(step_idx, replay_per_class=0)
        eval_ds = make_eval_dataset(classes_for_step(step_idx))

        out_dir = os.path.join(
            MODELS_DIR,
            f"full_finetune_step_{step_idx + 1}",
        )

        print(
            f"\n===== full_finetune | "
            f"step {step_idx + 1}/{NUM_STEPS} ====="
        )

        train_with_trainer(
            model=full_ft_model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            output_dir=out_dir,
            epochs=FT_EPOCHS,
            lr=LR_FT,
            batch_size=BATCH_FT,
            accum_steps=ACCUM_FT,
        )

    full_ft_eval_rows = evaluate_model(full_ft_model, "full_finetune")
    full_ft_eval_map = {row["eval_set"]: float(row["accuracy"]) for row in full_ft_eval_rows}
    method_summary_rows.append({
        "method": "full_finetune",
        "orth_mode": "none",
        "lambda_orth": 0.0,
        "zero_old_merge": False,
        "use_kd": False,
        "kd_weight": 0.0,
        "kd_temperature": 0.0,
        "replay_per_class": 0,
        "old_active_in_forward": np.nan,
        "first_step": full_ft_eval_map.get("first_step", np.nan),
        "later_steps": full_ft_eval_map.get("later_steps", np.nan),
        "all_seen": full_ft_eval_map.get("all_seen", np.nan),
        "old_new_gap": full_ft_eval_map.get("first_step", np.nan) - full_ft_eval_map.get("later_steps", np.nan),
        "avg_forgetting": np.nan,
    })

    del full_ft_model
    cleanup()

else:
    print("Skipping full_finetune")


# In[ ]:


def average_delta_reference_state(step_states):
    if len(step_states) == 0:
        return None

    keys = sorted(step_states[0]["deltas"].keys())
    ref = {}
    for key in keys:
        vals = [state["deltas"][key].float() for state in step_states if key in state["deltas"]]
        if len(vals) == 0:
            continue
        ref[key] = torch.stack(vals, dim=0).mean(dim=0)
    return ref

def list_delta_reference_state(step_states):
    """SINGLEJOB PACK ADDITION. Like average_delta_reference_state() just
    above, but returns the LIST of each previous step's own dense delta per
    module instead of their average -- the exact reference shape R8's
    DenseOrth formulation (dense_orth_penalty in
    supervisor_regularizer_repair_both_families_r8.py) requires:
    mean_i(cos(current_delta, dW_i)^2) over EACH individual previous delta,
    not one averaged reference. Deltas here are the same already-extracted,
    detached CPU tensors extract_lora_state() produced (scaling*(B@A)) --
    nothing is recomputed or retrained to build this reference."""
    if len(step_states) == 0:
        return None
    keys = sorted(step_states[0]["deltas"].keys())
    ref = {}
    for key in keys:
        vals = [state["deltas"][key].float() for state in step_states if key in state["deltas"]]
        if len(vals) == 0:
            continue
        ref[key] = vals
    return ref

def compute_independent_lora_orth_components(model, reference_weights, eps=1e-8):
    raw_trace_terms = []
    cosine_terms = []
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )
        if not has_lora:
            continue

        plain_name = normalize_module_name(name)
        if plain_name not in reference_weights:
            continue

        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight
        scaling = module.scaling["default"] if isinstance(module.scaling, dict) else module.scaling
        delta = (B @ A) * float(scaling)
        previous_delta = reference_weights[plain_name].to(device=delta.device, dtype=delta.dtype)

        raw_trace = torch.sum(previous_delta * delta)
        delta_norm = torch.linalg.norm(delta).clamp_min(eps)
        ref_norm = torch.linalg.norm(previous_delta).clamp_min(eps)
        cosine = raw_trace / (ref_norm * delta_norm)

        raw_trace_terms.append(raw_trace)
        cosine_terms.append(cosine)

    if len(cosine_terms) == 0:
        zero = torch.tensor(0.0, device=device)
        return {
            "num_layers": 0,
            "orth_loss_raw": zero,
            "orth_loss_abs": zero,
            "orth_loss_squared": zero,
            "mean_cosine_alignment": zero,
            "raw_trace_mean_unnormalized": zero,
        }

    raw_tensor = torch.stack(raw_trace_terms)
    cosine_tensor = torch.stack(cosine_terms)
    return {
        "num_layers": int(cosine_tensor.numel()),
        "orth_loss_raw": cosine_tensor.mean(),
        "orth_loss_abs": cosine_tensor.abs().mean(),
        "orth_loss_squared": cosine_tensor.pow(2).mean(),
        "mean_cosine_alignment": cosine_tensor.mean(),
        "raw_trace_mean_unnormalized": raw_tensor.mean(),
    }

def average_factor_reference_state(step_states):
    if len(step_states) == 0:
        return None

    keys = sorted(step_states[0]["lora_A"].keys())
    ref = {"lora_A": {}, "lora_B": {}}
    for key in keys:
        ref["lora_A"][key] = torch.stack([state["lora_A"][key].float() for state in step_states], dim=0).mean(dim=0)
        ref["lora_B"][key] = torch.stack([state["lora_B"][key].float() for state in step_states], dim=0).mean(dim=0)
    return ref

def compute_independent_lora_dense_orth_components(model, dense_orth_reference_state, eps=1e-12):
    """SINGLEJOB PACK ADDITION -- DenseOrth (Arms 3/4), mechanism ported
    verbatim from supervisor_regularizer_repair_both_families_r8.py's
    dense_cosine_sq()/dense_orth_penalty() (audited in
    thesis_agent/reports/r8_denseorth_formulation_and_gradient_audit.md),
    adapted only to the model-iteration pattern already used by
    compute_independent_lora_orth_components() just above (same LoRA-module
    detection, same normalize_module_name() keying) so it plugs into THIS
    trainer's compute_loss the same way factor_orth/delta_trace already do.

    For each target module: current_delta = scaling*(B@A), the LIVE tensor
    from the model currently being trained (carries gradients -- A/B are not
    detached here). For every INDIVIDUAL previous step's own dense delta
    dW_i (already-detached CPU tensors from dense_orth_reference_state,
    i.e. list_delta_reference_state()'s per-module list -- never an averaged
    reference, and never previous steps' raw A/B factors):

        cos_i = <current_delta, dW_i>_F / (||current_delta||_F ||dW_i||_F + eps)
        module_penalty = mean_i(cos_i^2)

    Aggregated across modules by taking the mean over modules (identical
    convention to factor_orth's `factor_total_mean`/delta_trace's
    `orth_loss_abs`, both of which also .mean() a per-layer cosine_tensor).
    Dense reconstruction (B@A) rather than raw A/B factor coordinates makes
    this penalty basis-invariant: any (A,B) -> (M A, B M^-1) reparameterization
    that leaves the dense product B@A unchanged leaves this penalty unchanged
    too -- unlike compute_independent_lora_factor_orth_components() below,
    which compares A/B coordinates directly.
    """
    device = next(model.parameters()).device
    zero = torch.tensor(0.0, device=device)

    if not dense_orth_reference_state:
        return {"num_layers": 0, "dense_orth_mean": zero}

    per_module_means = []

    for name, module in model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )
        if not has_lora:
            continue

        plain_name = normalize_module_name(name)
        if plain_name not in dense_orth_reference_state:
            continue

        previous_deltas = dense_orth_reference_state[plain_name]
        if not previous_deltas:
            continue

        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight
        scaling = module.scaling["default"] if isinstance(module.scaling, dict) else module.scaling
        current_delta = (B @ A) * float(scaling)
        current_norm = torch.linalg.norm(current_delta)

        cos_sq_terms = []
        for prev_delta in previous_deltas:
            prev_delta = prev_delta.to(device=current_delta.device, dtype=current_delta.dtype).detach()
            inner = torch.sum(current_delta * prev_delta)
            prev_norm = torch.linalg.norm(prev_delta)
            cos = inner / (current_norm * prev_norm + eps)
            cos_sq_terms.append(cos * cos)

        per_module_means.append(torch.stack(cos_sq_terms).mean())

    if len(per_module_means) == 0:
        return {"num_layers": 0, "dense_orth_mean": zero}

    return {
        "num_layers": int(len(per_module_means)),
        "dense_orth_mean": torch.stack(per_module_means).mean(),
    }

def build_simple_avg_teacher_model(step_states):
    if len(step_states) == 0:
        return None

    teacher_delta = simple_average_deltas(step_states)
    teacher_model = apply_deltas_to_base(teacher_delta, step_states)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    assert not any(p.requires_grad for p in teacher_model.parameters())
    return teacher_model

def masked_kd_loss(student_logits, teacher_logits, old_class_ids, temperature):
    """SINGLEJOB PACK ADDITION (Arms 1/2/4) -- mechanism ported verbatim from
    supervisor_regularizer_repair_both_families_r8.py's masked_kd_loss(),
    already audited in thesis_agent/reports/
    exp1_exp2_r8_supervisor_decision_synthesis.md. `teacher_model` here is
    STILL build_simple_avg_teacher_model(step_states) -- the same frozen,
    running dense-merged prior model Exp1's historical full-100-way KD
    already uses, queried on the SAME current-step-only images (no replay,
    unchanged) -- ONLY the loss computation changes: teacher/student logits
    are sliced to `old_class_ids` (the union of already-introduced classes,
    old_seen_class_ids() above) BEFORE any softmax is taken, so softmax is a
    true renormalization over the old-seen subset (sums to 1 over those
    classes only), not a post-hoc mask of a full 100-way distribution.
    Current/future classes are excluded from the KD term entirely (never
    sliced in). KL direction and T^2 scaling are unchanged from the
    historical full-100-way form: KL(teacher||student)*T^2, T=2, computed via
    the same F.kl_div(student_log_probs, teacher_probs, "batchmean") pattern
    used at the historical full-100-way KD call site below."""
    idx = torch.as_tensor(list(old_class_ids), device=student_logits.device, dtype=torch.long)
    teacher_old = teacher_logits.index_select(dim=-1, index=idx)
    student_old = student_logits.index_select(dim=-1, index=idx)
    teacher_probs = F.softmax(teacher_old / float(temperature), dim=-1)
    student_log_probs = F.log_softmax(student_old / float(temperature), dim=-1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kd * (float(temperature) ** 2)

def compute_independent_lora_factor_orth_components(model, factor_reference_state, eps=1e-12):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    factor_a_terms = []
    factor_b_terms = []
    factor_total_terms = []
    a_overlap_terms = []
    b_overlap_terms = []

    if factor_reference_state is None:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return {
            "num_layers": 0,
            "factor_A_mean": zero,
            "factor_B_mean": zero,
            "factor_total_mean": zero,
            "mean_A_overlap": zero,
            "mean_B_overlap": zero,
        }

    for name, module in model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )
        if not has_lora:
            continue

        plain_name = normalize_module_name(name)
        if plain_name not in factor_reference_state["lora_A"]:
            continue

        A_old = factor_reference_state["lora_A"][plain_name].to(device=device, dtype=dtype)
        B_old = factor_reference_state["lora_B"][plain_name].to(device=device, dtype=dtype)
        A_new = module.lora_A["default"].weight
        B_new = module.lora_B["default"].weight

        A_old_hat = A_old / A_old.norm(dim=1, keepdim=True).clamp_min(eps)
        A_new_hat = A_new / A_new.norm(dim=1, keepdim=True).clamp_min(eps)
        A_overlap = A_old_hat @ A_new_hat.T
        factor_a = torch.sum(A_overlap.pow(2))

        B_old_hat = B_old / B_old.norm(dim=0, keepdim=True).clamp_min(eps)
        B_new_hat = B_new / B_new.norm(dim=0, keepdim=True).clamp_min(eps)
        B_overlap = B_old_hat.T @ B_new_hat
        factor_b = torch.sum(B_overlap.pow(2))

        factor_a_terms.append(factor_a)
        factor_b_terms.append(factor_b)
        factor_total_terms.append(factor_a + factor_b)
        a_overlap_terms.append(A_overlap.abs().mean())
        b_overlap_terms.append(B_overlap.abs().mean())

    if len(factor_total_terms) == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return {
            "num_layers": 0,
            "factor_A_mean": zero,
            "factor_B_mean": zero,
            "factor_total_mean": zero,
            "mean_A_overlap": zero,
            "mean_B_overlap": zero,
        }

    return {
        "num_layers": int(len(factor_total_terms)),
        "factor_A_mean": torch.stack(factor_a_terms).mean(),
        "factor_B_mean": torch.stack(factor_b_terms).mean(),
        "factor_total_mean": torch.stack(factor_total_terms).mean(),
        "mean_A_overlap": torch.stack(a_overlap_terms).mean(),
        "mean_B_overlap": torch.stack(b_overlap_terms).mean(),
    }

def build_head_lr_param_groups(model, decay_parameter_names, base_lr, head_lr_multiplier, weight_decay):
    """
    ACCURACY-PUSH CHANGE 3: split trainable params into classifier-head vs
    other (LoRA/rank-extension) groups, giving the head base_lr *
    head_lr_multiplier while everything else keeps base_lr. Mirrors stock
    Trainer.create_optimizer()'s decay/no-decay split (bias and norm params get
    weight_decay=0.0) so this only changes the LR split, nothing else about how
    AdamW is configured.

    "classifier" as a substring safely identifies the head in both families:
    simple_avg wraps it via PEFT modules_to_save (name contains
    "classifier.modules_to_save.default.weight/bias"; the frozen
    "classifier.original_module.*" copy has requires_grad=False and is filtered
    out below), while rank_extension's classifier is a plain nn.Linear
    ("classifier.weight"/"classifier.bias"). No LoRA/rank-extension parameter
    name (lora_A/lora_B/.A_new/.B_new) ever contains "classifier".
    """
    head_decay, head_no_decay, other_decay, other_no_decay = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = "classifier" in name
        is_decay = name in decay_parameter_names
        if is_head and is_decay:
            head_decay.append(param)
        elif is_head and not is_decay:
            head_no_decay.append(param)
        elif is_decay:
            other_decay.append(param)
        else:
            other_no_decay.append(param)

    groups = []
    if other_decay:
        groups.append({"params": other_decay, "weight_decay": weight_decay, "lr": base_lr})
    if other_no_decay:
        groups.append({"params": other_no_decay, "weight_decay": 0.0, "lr": base_lr})
    if head_decay:
        groups.append({"params": head_decay, "weight_decay": weight_decay, "lr": base_lr * head_lr_multiplier})
    if head_no_decay:
        groups.append({"params": head_no_decay, "weight_decay": 0.0, "lr": base_lr * head_lr_multiplier})

    return groups


class HeadLRTrainerMixin:
    """
    ACCURACY-PUSH CHANGE 3: overrides create_optimizer() to give the classifier
    head a per-family multiplier times the base LR (REVERT 2026-07-16: now
    resolved per-instance via family_head_lr_multiplier(), not the bare
    HEAD_LR_MULTIPLIER global -- see HEAD_LR_MULTIPLIER_BY_FAMILY above).
    When the resolved multiplier == 1.0 this is a no-op that defers to the
    untouched stock Trainer.create_optimizer() (single global LR, identical to
    before this change), so the flag fully disables the change for that
    family, not just neutralizes it.

    Every class that mixes this in (IndependentLoraOrthTrainer,
    RankExtensionTrainer) already sets self.method_name in __init__ before
    training starts, and every active method name is a key in
    ACTIVE_METHOD_MAP with a "family" field -- so the family lookup below is
    always resolvable at the point create_optimizer() actually runs (during
    Trainer.train(), never before __init__ returns).

    This project never uses SageMaker model-parallel training, so (unlike stock
    Trainer.create_optimizer()) this always reads self.model directly rather
    than branching on is_sagemaker_mp_enabled().
    """

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        method_name = getattr(self, "method_name", None)
        family = ACTIVE_METHOD_MAP.get(str(method_name), {}).get("family")
        head_lr_multiplier = family_head_lr_multiplier(family) if family is not None else float(HEAD_LR_MULTIPLIER)

        if float(head_lr_multiplier) == 1.0:
            return super().create_optimizer()

        opt_model = self.model
        decay_parameter_names = set(self.get_decay_parameter_names(opt_model))
        grouped_params = build_head_lr_param_groups(
            model=opt_model,
            decay_parameter_names=decay_parameter_names,
            base_lr=float(self.args.learning_rate),
            head_lr_multiplier=float(head_lr_multiplier),
            weight_decay=float(self.args.weight_decay),
        )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(grouped_params, **optimizer_kwargs)
        return self.optimizer


class IndependentLoraOrthTrainer(HeadLRTrainerMixin, Trainer):
    def __init__(
        self,
        *args,
        reference_weights=None,
        factor_reference_state=None,
        dense_orth_reference_state=None,
        lambda_orth=0.0,
        orth_mode="none",
        teacher_model=None,
        kd_weight=0.0,
        kd_temperature=2.0,
        kd_class_scope="full",
        kd_old_class_ids=None,
        kd_warmup_epochs=0.0,
        dense_orth_warmup_epochs=DENSE_ORTH_WARMUP_EPOCHS,
        method_name="unknown",
        step_idx=-1,
        orth_eps=1e-12,
        log_every_steps=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_weights = reference_weights or {}
        self.factor_reference_state = factor_reference_state
        # SINGLEJOB PACK ADDITION (Arms 3/4): dict[module_name] -> LIST of
        # each previous step's own dense delta (list_delta_reference_state()),
        # never an averaged reference -- see compute_independent_lora_dense_
        # orth_components()'s docstring for why this must be a list.
        self.dense_orth_reference_state = dense_orth_reference_state
        self.lambda_orth = float(lambda_orth)
        self.orth_mode = "none" if orth_mode is None else str(orth_mode)
        self.teacher_model = teacher_model
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
        # SINGLEJOB PACK ADDITION (Arms 1/2/4): "full" reproduces Exp1's
        # historical full-100-way KD exactly (unused in this script's active
        # arms -- Arm 0 has kd_weight=0, so this branch is never actually
        # exercised here, but is kept so the mechanism is provably identical
        # to Exp1 rather than deleted). "old_seen" activates masked_kd_loss().
        self.kd_class_scope = str(kd_class_scope)
        self.kd_old_class_ids = kd_old_class_ids
        # SINGLEJOB PACK ADDITION: KD-weight linear warmup (Arm 1: 0.0 ->
        # no-op, multiplier always 1.0, i.e. flat kd_weight from batch 1,
        # matching Exp1's historical KD exactly; Arms 2/4: 1.0, matching R8's
        # KD_WARMUP_EPOCHS). Orthogonal to (independent of) the pre-existing
        # COMBINED_ORTH_WARMUP_* mechanism below, which is untouched and still
        # governs ONLY factor_orth's own lambda warmup for the (unused-here)
        # historical combined method.
        self.kd_warmup_epochs = float(kd_warmup_epochs)
        # SINGLEJOB PACK ADDITION (Arms 3/4): DenseOrth's own lambda warmup,
        # UNCONDITIONAL whenever orth_mode=="dense_orth" (R8's "no asymmetry"
        # convention -- applies whether or not KD is also active), separate
        # from the pre-existing COMBINED_ORTH_WARMUP_* mechanism (factor_orth
        # -only, KD-gated) below.
        self.dense_orth_warmup_epochs = float(dense_orth_warmup_epochs)
        self.method_name = str(method_name)
        self.step_idx = int(step_idx)
        self.orth_eps = float(orth_eps)
        self.log_every_steps = max(1, int(log_every_steps))
        self._rows = []
        self._teacher_ready = False

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        orth_loss_used = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        orth_loss_raw = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        factor_comps = {
            "num_layers": 0,
            "factor_A_mean": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype),
            "factor_B_mean": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype),
            "factor_total_mean": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype),
            "mean_A_overlap": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype),
            "mean_B_overlap": torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype),
        }

        if self.orth_mode == "delta_trace":
            orth_comps = compute_independent_lora_orth_components(
                model=model,
                reference_weights=self.reference_weights,
                eps=self.orth_eps,
            )
            orth_loss_used = orth_comps["orth_loss_abs"]
            orth_loss_raw = orth_comps["raw_trace_mean_unnormalized"]
        elif self.orth_mode == "factor_orth":
            factor_comps = compute_independent_lora_factor_orth_components(
                model=model,
                factor_reference_state=self.factor_reference_state,
                eps=self.orth_eps,
            )
            orth_loss_used = factor_comps["factor_total_mean"]
            orth_loss_raw = orth_loss_used
        elif self.orth_mode == "dense_orth":
            # SINGLEJOB PACK ADDITION (Arms 3/4).
            dense_orth_comps = compute_independent_lora_dense_orth_components(
                model=model,
                dense_orth_reference_state=self.dense_orth_reference_state,
                eps=self.orth_eps,
            )
            orth_loss_used = dense_orth_comps["dense_orth_mean"]
            orth_loss_raw = orth_loss_used

        kd_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        teacher_active = self.teacher_model is not None and self.kd_weight > 0.0

        # Objective 2 (STRICT-REVIEW B2: now ON by default -- see
        # COMBINED_ORTH_WARMUP_ENABLED): ramps lambda_orth up over the first
        # COMBINED_ORTH_WARMUP_EPOCHS epochs of each step, but ONLY when KD is
        # ALSO active this step (teacher_active) and orth_mode is
        # "factor_orth" -- so this only ever touches
        # simple_avg_factor_orth_kd_T2 (the combined method), never plain
        # simple_avg_factor_orth (kd_weight=0 there, teacher_active is always
        # False). Independent of, and stackable with, COMBINED_LOSS_SCALE_ENABLED.
        epoch_val = float(self.state.epoch) if self.state.epoch is not None else np.nan
        combined_warmup_multiplier = orth_lambda_warmup_multiplier(
            epoch_val,
            COMBINED_ORTH_WARMUP_EPOCHS,
            bool(COMBINED_ORTH_WARMUP_ENABLED and teacher_active and self.orth_mode == "factor_orth"),
        )
        # SINGLEJOB PACK ADDITION (Arms 3/4): DenseOrth's OWN, unconditional
        # 1-epoch lambda warmup (R8 convention -- see dense_orth_warmup_epochs'
        # docstring in __init__), computed independently of
        # combined_warmup_multiplier above (which stays scoped to
        # factor_orth+KD only, per its own pre-existing comment, and is a
        # no-op here since self.orth_mode is never "factor_orth" for these
        # arms). The two mechanisms are mutually exclusive by construction
        # (orth_mode is a single string), so exactly one of them ever applies
        # a non-1.0 multiplier for any given arm.
        dense_orth_warmup_multiplier = orth_lambda_warmup_multiplier(
            epoch_val,
            self.dense_orth_warmup_epochs,
            bool(self.orth_mode == "dense_orth"),
        )
        if self.orth_mode == "dense_orth":
            effective_lambda_orth = float(self.lambda_orth) * dense_orth_warmup_multiplier
        else:
            effective_lambda_orth = float(self.lambda_orth) * combined_warmup_multiplier
        weighted_orth = effective_lambda_orth * orth_loss_used

        # SINGLEJOB PACK ADDITION (Arms 1/2/4): KD-weight linear warmup,
        # independent of the orth-lambda warmups above. multiplier is always
        # 1.0 (no-op) when self.kd_warmup_epochs<=0 (Arm 1 and every
        # unmodified Exp1 method, including Arm 0 where kd_weight=0 anyway).
        kd_warmup_multiplier = orth_lambda_warmup_multiplier(
            epoch_val,
            self.kd_warmup_epochs,
            bool(self.kd_warmup_epochs > 0.0),
        )

        if teacher_active:
            if not self._teacher_ready:
                self.teacher_model.to(device=ce_loss.device)
                self.teacher_model.eval()
                self._teacher_ready = True
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits.detach()
            # SINGLEJOB PACK ADDITION (Arms 1/2/4): old-seen-only KD. Teacher
            # construction, image source (current-step-only, no replay), KL
            # direction, and T^2 scaling are ALL unchanged from the historical
            # full-100-way branch below -- only which logits enter the
            # softmax changes (see masked_kd_loss()'s docstring).
            if self.kd_class_scope == "old_seen":
                kd_loss = masked_kd_loss(
                    outputs.logits, teacher_logits, self.kd_old_class_ids, self.kd_temperature,
                )
            else:
                student_log_probs = F.log_softmax(outputs.logits / self.kd_temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / self.kd_temperature, dim=-1)
                kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.kd_temperature ** 2)

        weighted_kd = float(self.kd_weight) * float(kd_warmup_multiplier) * kd_loss
        loss = ce_loss + weighted_orth + weighted_kd

        ce_v = float(ce_loss.detach().cpu().item())
        orth_raw_v = float(orth_loss_raw.detach().cpu().item())
        orth_used_v = float(orth_loss_used.detach().cpu().item())
        weighted_orth_v = float(weighted_orth.detach().cpu().item())
        kd_loss_v = float(kd_loss.detach().cpu().item())
        weighted_kd_v = float(weighted_kd.detach().cpu().item())
        total_loss_v = float(loss.detach().cpu().item())
        row = {
            "method": self.method_name,
            "step": int(self.step_idx + 1),
            "epoch": epoch_val,
            "ce_loss": ce_v,
            "raw_inner": orth_raw_v,
            "abs_inner": abs(orth_raw_v),
            "norm_sq_orth": np.nan,
            "orth_loss_raw": orth_raw_v,
            "orth_loss": orth_used_v,
            "orth_loss_used": orth_used_v,
            "lambda_orth": float(self.lambda_orth),
            # Objective 2 transparency: nominal (configured, already reflects
            # COMBINED_LAMBDA_ORTH_SCALE for the combined method) vs the
            # warmup-scaled value actually applied this batch. Equal only when
            # COMBINED_ORTH_WARMUP_ENABLED is off, or once local_epoch exceeds
            # COMBINED_ORTH_WARMUP_EPOCHS within a step (now on by default).
            "lambda_orth_warmup_multiplier": float(
                dense_orth_warmup_multiplier if self.orth_mode == "dense_orth" else combined_warmup_multiplier
            ),
            "lambda_orth_times_loss": weighted_orth_v,
            "orth_ratio_abs_weighted_over_ce": abs(weighted_orth_v) / (ce_v + float(self.orth_eps)),
            "weighted_orth_over_CE": abs(weighted_orth_v) / (ce_v + float(self.orth_eps)),
            "orth_mode": self.orth_mode,
            "num_layers_used": int(
                factor_comps["num_layers"] if self.orth_mode == "factor_orth"
                else (dense_orth_comps["num_layers"] if self.orth_mode == "dense_orth" else len(self.reference_weights))
            ),
            "old_norm_mean": np.nan,
            "new_norm_mean": np.nan,
            "factor_A_penalty_mean": float(factor_comps["factor_A_mean"].detach().cpu().item()),
            "factor_B_penalty_mean": float(factor_comps["factor_B_mean"].detach().cpu().item()),
            "factor_total_penalty_mean": float(factor_comps["factor_total_mean"].detach().cpu().item()),
            "weighted_factor_orth_mean": float((effective_lambda_orth * factor_comps["factor_total_mean"]).detach().cpu().item()),
            "weighted_factor_orth_over_CE": abs(float((effective_lambda_orth * factor_comps["factor_total_mean"]).detach().cpu().item())) / (ce_v + float(self.orth_eps)),
            "mean_A_overlap": float(factor_comps["mean_A_overlap"].detach().cpu().item()),
            "mean_B_overlap": float(factor_comps["mean_B_overlap"].detach().cpu().item()),
            # SINGLEJOB PACK ADDITION diagnostic fields (Arms 1-4). Present for
            # every row in THIS script (all logged rows come from
            # IndependentLoraOrthTrainer -- RankExt's trainer is never
            # exercised, its flags are all False), so the CSV schema stays
            # uniform across the whole run.
            "dense_orth_mean": float(dense_orth_comps["dense_orth_mean"].detach().cpu().item()) if self.orth_mode == "dense_orth" else np.nan,
            "dense_orth_warmup_multiplier": float(dense_orth_warmup_multiplier),
            "kd_class_scope": str(self.kd_class_scope),
            "kd_warmup_epochs": float(self.kd_warmup_epochs),
            "kd_warmup_multiplier": float(kd_warmup_multiplier),
            "kd_loss": kd_loss_v,
            "weighted_kd_loss": weighted_kd_v,
            "kd_over_CE": weighted_kd_v / (ce_v + float(self.orth_eps)),
            "kd_weight": float(self.kd_weight),
            "kd_temperature": float(self.kd_temperature),
            "teacher_active": bool(teacher_active),
            "total_loss": total_loss_v,
            "effective_lambda": float(effective_lambda_orth),
        }
        self._rows.append(row)

        if len(self._rows) % self.log_every_steps == 0:
            print(
                f"[simple orth/kd train] method={self.method_name} | step={row['step']} | epoch={row['epoch']:.4f} | "
                f"ce={row['ce_loss']:.6f} | orth={row['orth_loss_used']:.6f} | "
                f"kd={row['kd_loss']:.6f} | total={row['total_loss']:.6f} | "
                f"lambda={row['lambda_orth']:.6g} (warmup x{row['lambda_orth_warmup_multiplier']:.3g}) | "
                f"kd_weight={row['kd_weight']:.6g}"
            )

        return (loss, outputs) if return_outputs else loss

    def consume_logged_losses(self):
        if len(self._rows) == 0:
            return None
        out = pd.DataFrame(self._rows).copy()
        self._rows = []
        return out

def train_independent_loras(
    method_name,
    method_prefix,
    replay_per_class=0,
    use_orth=False,
    orth_mode=None,
    use_kd=False,
    kd_weight=0.0,
    kd_temperature=2.0,
    orth_train_records=None,
    lambda_orth=None,
    kd_class_scope="full",
    kd_warmup_epochs=0.0,
    dense_orth_warmup_epochs=DENSE_ORTH_WARMUP_EPOCHS,
):
    step_states = []
    # PRE-THESIS FIX 2: {step_idx: accuracy_fraction} -- the accuracy of THIS
    # step's own independently-trained specialist LoRA, evaluated on its own
    # class group right after training and before it gets merged/averaged with
    # the other steps' specialists. This is the simple_avg-family analog of
    # "accuracy on task i measured right after learning it" (the diagonal a_i,i
    # backward_transfer needs) -- the closest honestly-available equivalent,
    # since simple_avg's steps are trained independently rather than
    # incrementally, so there is no single evolving "model at step i".
    specialist_diagonal_accuracy = {}
    active_orth_mode = None if orth_mode is None else str(orth_mode)
    # `lambda_orth=None` (the default) preserves the pre-Objective-2 behavior
    # of deriving the value from the global LAMBDA_ORTH; callers that need a
    # per-method override (e.g. run_simple_avg_variant() passing
    # method_cfg["lambda_orth"], which reflects COMBINED_LAMBDA_ORTH_SCALE for
    # simple_avg_factor_orth_kd_T2) pass it explicitly so the value actually
    # used in training matches what the config tables report -- same
    # single-source-of-truth principle as apply_calibration.
    resolved_lambda_orth = float(LAMBDA_ORTH if use_orth else 0.0) if lambda_orth is None else float(lambda_orth)
    simple_avg_target_modules = family_target_modules("simple_avg")

    for step_idx in range(NUM_STEPS):
        model = fresh_pretrained_model()
        model = add_lora(model, target_modules=simple_avg_target_modules)
        model.print_trainable_parameters()

        teacher_model = None
        factor_reference_state = None
        reference_weights = None
        dense_orth_reference_state = None
        kd_old_class_ids_for_step = None
        if use_kd and len(step_states) > 0:
            teacher_model = build_simple_avg_teacher_model(step_states)
            assert not any(p.requires_grad for p in teacher_model.parameters())
            # SINGLEJOB PACK ADDITION (Arms 1/2/4): teacher construction is
            # UNCHANGED (still build_simple_avg_teacher_model, still the
            # running dense-merged prior model) -- only the class scope used
            # inside compute_loss's loss computation differs, resolved here so
            # it's computed once per step rather than once per batch.
            if kd_class_scope == "old_seen":
                kd_old_class_ids_for_step = old_seen_class_ids(step_idx)
        if use_orth and active_orth_mode == "factor_orth" and len(step_states) > 0:
            factor_reference_state = average_factor_reference_state(step_states)
        if use_orth and active_orth_mode == "delta_trace" and len(step_states) > 0:
            reference_weights = average_delta_reference_state(step_states)
        if use_orth and active_orth_mode == "dense_orth" and len(step_states) > 0:
            # SINGLEJOB PACK ADDITION (Arms 3/4): list of INDIVIDUAL previous
            # steps' dense deltas (never an average) -- see
            # list_delta_reference_state()'s docstring.
            dense_orth_reference_state = list_delta_reference_state(step_states)

        train_ds = make_train_dataset(
            step_idx=step_idx,
            replay_per_class=replay_per_class,
        )
        eval_ds = make_val_dataset(classes_for_step(step_idx))
        out_dir = os.path.join(MODELS_DIR, f"{method_prefix}_step_{step_idx + 1}")

        print(
            f"\n===== {method_name} | step {step_idx + 1}/{NUM_STEPS} | "
            f"replay_per_class={replay_per_class} | orth={use_orth} | orth_mode={active_orth_mode} | "
            f"lambda_orth={resolved_lambda_orth:.6g} | use_kd={use_kd} | "
            f"teacher_active={teacher_model is not None} ====="
        )

        trainer_cls = IndependentLoraOrthTrainer
        trainer_kwargs = {
            "reference_weights": reference_weights,
            "factor_reference_state": factor_reference_state,
            "dense_orth_reference_state": dense_orth_reference_state,
            "lambda_orth": resolved_lambda_orth,
            "orth_mode": "none" if not use_orth else active_orth_mode,
            "teacher_model": teacher_model,
            "kd_weight": float(kd_weight) if teacher_model is not None else 0.0,
            "kd_temperature": float(kd_temperature),
            "kd_class_scope": str(kd_class_scope),
            "kd_old_class_ids": kd_old_class_ids_for_step,
            "kd_warmup_epochs": float(kd_warmup_epochs),
            "dense_orth_warmup_epochs": float(dense_orth_warmup_epochs),
            "method_name": method_name,
            "step_idx": int(step_idx),
            "orth_eps": float(ORTH_EPS),
            "log_every_steps": int(ORTH_LOSS_LOG_EVERY),
        }

        trainer, _ = train_with_trainer(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            output_dir=out_dir,
            epochs=LORA_EPOCHS,
            lr=LR_LORA,
            batch_size=BATCH_LORA,
            accum_steps=ACCUM_LORA,
            trainer_cls=trainer_cls,
            display_name=METHOD_DISPLAY_NAME_MAP.get(method_name, method_name),
            epoch_loss_records=epoch_loss_rows,
            best_epoch_selection_records=best_epoch_selection_rows,
            **trainer_kwargs,
        )

        if isinstance(trainer, IndependentLoraOrthTrainer):
            loss_rows_df = trainer.consume_logged_losses()
            if loss_rows_df is not None and orth_train_records is not None and len(loss_rows_df) > 0:
                orth_train_records.extend(loss_rows_df.to_dict("records"))

        # Task 2: refresh this method's live convergence plot/tables now that
        # step_idx + 1 has finished (train + val CE rows for it are now available
        # in the module-global accumulator lists).
        refresh_live_convergence(method_name)

        # PRE-THESIS FIX 2: evaluate this step's specialist on its own class
        # group BEFORE extracting/discarding it -- `model` here already holds
        # the best-epoch-selected weights (FIX 1), so this measures the
        # specialist at its own best checkpoint, consistent with what actually
        # gets merged into step_states below.
        specialist_diagonal_accuracy[int(step_idx)] = evaluate_single_step_accuracy(model, step_idx)

        state = extract_lora_state(model)
        step_states.append(state)

        if teacher_model is not None:
            del teacher_model
        del model
        cleanup()

    return step_states, specialist_diagonal_accuracy


# In[ ]:


step_states_no_replay = None
step_states_no_replay_orth = None
step_states_simple_kd = None
step_states_simple_factor_orth = None
step_states_simple_factor_orth_kd = None
simple_avg_step_states = {}

def append_simple_method_summary(method_name, eval_rows, backward_transfer=np.nan, forward_transfer=np.nan):
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    eval_map = {row["eval_set"]: float(row["accuracy"]) for row in eval_rows}
    method_summary_rows.append({
        "method": method_name,
        "orth_mode": (
            "delta_trace"
            if method_cfg["uses_delta_trace"]
            else ("factor_orth" if method_cfg["uses_factor_orth"] else "none")
        ),
        "lambda_orth": float(method_cfg["lambda_orth"]),
        "zero_old_merge": False,
        "use_kd": bool(method_cfg["uses_kd"]),
        "kd_weight": float(method_cfg["kd_weight"]),
        "kd_temperature": float(method_cfg["kd_temperature"]),
        "replay_per_class": 0,
        "old_active_in_forward": np.nan,
        "first_step": eval_map.get("first_step", np.nan),
        "later_steps": eval_map.get("later_steps", np.nan),
        "all_seen": eval_map.get("all_seen", np.nan),
        "old_new_gap": eval_map.get("first_step", np.nan) - eval_map.get("later_steps", np.nan),
        # PRE-THESIS FIX 2: avg_forgetting stays NaN for simple_avg -- that
        # column's formula (compute_average_forgetting) needs a full stepwise
        # "model at step i evaluated on task j<=i" matrix that only the
        # incrementally-evolving rank_extension family has. backward_transfer
        # IS honestly computable for simple_avg (see run_simple_avg_variant);
        # forward_transfer is not (no well-defined "model before step i" for a
        # merge-based family) and is left NaN, never fabricated.
        # backward_transfer/forward_transfer use the SAME fraction (0..1, not
        # percent) convention as avg_forgetting elsewhere in this table/CSV.
        "avg_forgetting": np.nan,
        "backward_transfer": float(backward_transfer) if not np.isnan(backward_transfer) else np.nan,
        "forward_transfer": float(forward_transfer) if not np.isnan(forward_transfer) else np.nan,
    })


# SINGLEJOB PACK ADDITION: a SEPARATE result table, not a reuse of
# method_summary_rows/append_simple_method_summary() above. That table is
# keyed ONLY by method_name (one row per training method, one implicit merge)
# -- reusing it for two merges of the same training arm would silently
# conflate or overwrite rows (confirmed during the preparation audit: it has
# no merge-aware key at all). This table has two EXPLICIT columns,
# training_method and merge_method, per the required result-table structure
# -- never encoded into one ambiguous name string.
training_merge_summary_rows = []


def append_training_merge_summary(
    training_method, merge_method, eval_rows, backward_transfer=np.nan, specialist_fingerprint=None,
    family=None, restricted_mean=np.nan, forgetting_metric=np.nan,
):
    method_cfg = ACTIVE_METHOD_MAP[training_method]
    eval_map = {row["eval_set"]: float(row["accuracy"]) for row in eval_rows}
    training_merge_summary_rows.append({
        # JOINT PACK ADDITION: family made explicit (not inferred from
        # method_cfg here, so a caller can never accidentally mislabel a row)
        # -- defaults to method_cfg's own family when not passed (SimpleAvg's
        # call site relies on this default).
        "family": str(family) if family is not None else str(method_cfg["family"]),
        "training_method": training_method,
        "merge_method": merge_method,
        "seed": int(SEED),
        "epochs": int(RANKEXT_EPOCHS) if str(method_cfg["family"]) == "rank_extension" else int(LORA_EPOCHS),
        "rank_config": (
            "->".join(str(v) for v in get_rank_extension_rank_schedule())
            if str(method_cfg["family"]) == "rank_extension" else f"fixed:{int(method_cfg['rank'])}"
        ),
        "restricted_mean": float(restricted_mean) if not (isinstance(restricted_mean, float) and np.isnan(restricted_mean)) else np.nan,
        "forgetting_metric": float(forgetting_metric) if not (isinstance(forgetting_metric, float) and np.isnan(forgetting_metric)) else np.nan,
        # READINESS-AUDIT ADDITION: makes the same-specialist provenance
        # explicitly visible in the primary result table -- every row for a
        # given training_method (across all its merge_method rows) must show
        # the IDENTICAL fingerprint, proving the merges were computed from the
        # exact same persisted specialist state (see
        # compute_step_states_fingerprint()'s docstring; asserted, not just
        # displayed -- evaluate_arm_with_merge() raises before reaching this
        # call if the fingerprint doesn't match).
        "specialist_fingerprint": specialist_fingerprint,
        "uses_kd": bool(method_cfg["uses_kd"]),
        "kd_class_scope": method_cfg.get("kd_class_scope", "full"),
        "kd_warmup_epochs": float(method_cfg.get("kd_warmup_epochs", 0.0)),
        "kd_weight": float(method_cfg["kd_weight"]),
        "kd_temperature": float(method_cfg["kd_temperature"]),
        "uses_dense_orth": bool(method_cfg.get("uses_dense_orth", False)),
        "lambda_orth": float(method_cfg["lambda_orth"]),
        "rank": int(method_cfg["rank"]),
        "first_step": eval_map.get("first_step", np.nan),
        "later_steps": eval_map.get("later_steps", np.nan),
        "all_seen": eval_map.get("all_seen", np.nan),
        "old_new_gap": eval_map.get("first_step", np.nan) - eval_map.get("later_steps", np.nan),
        # Same convention as append_simple_method_summary(): forward_transfer
        # is not well-defined for this merge-based family and stays NaN,
        # never fabricated; backward_transfer uses the same 0..1 fraction
        # convention.
        "backward_transfer": float(backward_transfer) if not np.isnan(backward_transfer) else np.nan,
        "forward_transfer": np.nan,
    })


def run_simple_avg_variant(method_name):
    """PRESERVED, UNCHANGED from Exp1 (not called by this script's own
    execution loop below, which uses the persistence-aware
    ensure_arm_trained()/evaluate_arm_with_merge() pair instead) -- kept
    verbatim so the ORIGINAL, single-merge, non-resumable pipeline remains
    available/diff-able against Exp1 if ever needed, and so nothing that
    still calls this function by name elsewhere breaks."""
    # REPRODUCIBILITY-ONLY FIX (2026-08-25, not a scientific change): SEED is
    # set once at module load (see SEED's own definition comment) and never
    # reset per method, so model/adapter init, dropout, and every other
    # global-RNG-dependent draw silently depend on how many methods already
    # trained earlier in THIS run -- i.e. results become a function of
    # execution order/method-count, not just hyperparameters (e.g. the
    # flagship trained 3rd-of-3 in the historical 74.07 run but 8th-of-8 in
    # the 8-method final comparison). Resetting to the SAME experiment seed
    # here, once per top-level method call and before any model/adapter/
    # optimizer/DataLoader construction happens inside train_independent_
    # loras() below, makes every method start from an identical, order-
    # independent RNG state -- this is the single top-level entry point for
    # every simple_avg-family method (see simple_avg_execution_order's call
    # loop below, exactly one call per active method). Uses the project's
    # existing set_seed() (transformers.set_seed -- covers random, numpy,
    # torch CPU, and torch.cuda.manual_seed_all for every device) rather than
    # duplicating manual seeding code. Per-class/per-step dataset shuffles
    # (seed=SEED+offset elsewhere in this file) are untouched by this call.
    set_seed(SEED)
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    step_states, specialist_diagonal_accuracy = train_independent_loras(
        method_name=method_name,
        method_prefix=f"{method_name}_source",
        replay_per_class=0,
        use_orth=bool(method_cfg["uses_delta_trace"] or method_cfg["uses_factor_orth"] or method_cfg.get("uses_dense_orth", False)),
        orth_mode=(
            "delta_trace"
            if method_cfg["uses_delta_trace"]
            else ("factor_orth" if method_cfg["uses_factor_orth"]
            else ("dense_orth" if method_cfg.get("uses_dense_orth", False) else None))
        ),
        use_kd=bool(method_cfg["uses_kd"]),
        kd_weight=float(method_cfg["kd_weight"]),
        kd_temperature=float(method_cfg["kd_temperature"]),
        kd_class_scope=method_cfg.get("kd_class_scope", "full"),
        kd_warmup_epochs=float(method_cfg.get("kd_warmup_epochs", 0.0)),
        orth_train_records=train_diagnostic_rows,
        # Objective 2: threads method_cfg["lambda_orth"] through so
        # COMBINED_LAMBDA_ORTH_SCALE actually reaches training for
        # simple_avg_factor_orth_kd_T2, not just the reported config tables --
        # every other method's lambda_orth_scale is 1.0 so this is a no-op for
        # them (identical to the previous LAMBDA_ORTH-derived behavior).
        lambda_orth=float(method_cfg["lambda_orth"]),
    )
    simple_avg_step_states[method_name] = step_states

    merged_delta = simple_average_deltas(step_states)
    # STRICT-REVIEW ADD (B1): diagnostic-only, see log_merge_mechanism()
    # docstring. Runs for all 4 simple_avg variants (this function is their
    # shared code path); does not affect merged_delta or anything downstream.
    log_merge_mechanism(
        method_name=method_name,
        step_states=step_states,
        merged_delta=merged_delta,
        csv_path=os.path.join(TABLES_DIR, "merge_mechanism_by_method_step.csv"),
    )
    merged_model = apply_deltas_to_base(
        merged_deltas=merged_delta,
        step_states=step_states,
    )

    if method_cfg["apply_calibration"]:
        calibration_mode = method_cfg.get("calibration_mode", "global")
        # CALIBRATION EXPERIMENT (2026-08-25): mirrors run_rank_extension_
        # variant()'s dispatch exactly (see that function's identical if/else
        # just above its own calibrate_classifier_row_norms(...) call). Before
        # this run, this branch did not exist here -- CALIBRATION_MODE_BY_
        # FAMILY["simple_avg"] was always "global", so calibrate_classifier_
        # row_norms_confidence_weighted() was only ever reachable from the
        # rank_extension path. Now that simple_avg's mode can also be
        # "confidence_weighted_regime_grouped", this function needs the same
        # two-way dispatch or the mode value would silently fall through
        # calibrate_classifier_row_norms()'s own mode check (which only
        # special-cases the literal string "regime_grouped") into flat
        # single-group ("global"-equivalent) behavior -- no error, just a
        # config value that quietly does nothing. epoch_loss_rows is the same
        # module-global accumulator run_rank_extension_variant() passes in
        # (populated by the shared EpochValidationCallback for both families).
        if calibration_mode == "confidence_weighted_regime_grouped":
            merged_model = calibrate_classifier_row_norms_confidence_weighted(
                merged_model,
                epoch_loss_rows=epoch_loss_rows,
                method_name=method_name,
                uses_kd=bool(method_cfg["uses_kd"]),
            )
        else:
            merged_model = calibrate_classifier_row_norms(
                merged_model,
                mode=calibration_mode,
                uses_kd=bool(method_cfg["uses_kd"]),
                method_name=method_name,
            )
    else:
        # FIX 1 diagnostic: still record pre-calibration row-norm stats for
        # non-calibrated methods so classifier_row_norm_diagnostic_rows has
        # full coverage across all 8 methods, not just the calibrated ones.
        log_classifier_row_norm_diagnostics(merged_model, method_name, phase="pre_calibration")

    eval_rows = evaluate_model(merged_model, method_name)

    # PRE-THESIS FIX 2: per-CL-step accuracy of the FINAL (merged) model, plus
    # backward_transfer against the specialist diagonal computed during
    # training. forward_transfer is not well-defined for this merge-based
    # family (see append_simple_method_summary docstring note) so it stays NaN.
    final_per_step_accuracy = evaluate_per_step_accuracy(merged_model, method_name)
    backward_transfer = compute_backward_transfer(specialist_diagonal_accuracy, final_per_step_accuracy)
    print(
        f"[simple_avg summary] method={method_name} | "
        f"backward_transfer={backward_transfer} | forward_transfer=NaN (not defined for this family)"
    )

    append_simple_method_summary(method_name, eval_rows, backward_transfer=backward_transfer, forward_transfer=np.nan)

    del merged_model
    cleanup()


# ==============================================================================
# SINGLEJOB PACK ADDITION -- persistence-aware, two-merge-operator execution.
# This REPLACES the plain `for method_name in simple_avg_execution_order:
# run_simple_avg_variant(method_name)` loop with a resume-aware training phase
# (ensure_arm_trained) followed by an explicit, two-dimensional evaluation
# phase (evaluate_arm_with_merge) run once per (training arm, merge operator)
# pair. run_simple_avg_variant() above is NOT called by this loop -- it is
# preserved only for diff/reference purposes.
# ==============================================================================

def _arm_checkpoint_paths(method_name):
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{method_name}__step_states.pt")
    marker_path = os.path.join(CHECKPOINTS_DIR, f"{method_name}__DONE.marker")
    return ckpt_path, marker_path


def compute_method_config_fingerprint(method_name):
    """FINAL-AUDIT FIX (persistence safety): a checkpoint on disk is keyed
    only by method_name under a SEED-derived CHECKPOINTS_DIR (stable across
    restarts of this job by design -- see CHECKPOINTS_DIR's own comment).
    That means a checkpoint saved by an OLDER edit of this script (different
    epochs/rank/schedule/lambda/KD settings for the same method_name) would
    otherwise be silently reloaded and treated as valid for the CURRENT
    config on any later resubmission -- the specialist/state fingerprints
    above only prove a reload is byte-identical to what was SAVED, never that
    what was saved matches what THIS run's config actually wants. This
    fingerprint covers every field that affects training for `method_name`
    (epochs, rank/schedule, orth mode+lambda, KD scope/weight/temperature/
    warmup, target modules, seed, protocol shape) and is stored alongside
    each checkpoint; load_arm_checkpoint_if_present/load_rankext_checkpoint_
    if_present reject (and force a retrain of) any checkpoint whose stored
    fingerprint does not match the CURRENT config's fingerprint for that
    method, rather than silently reusing stale/incompatible state."""
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    family = str(method_cfg["family"])
    fingerprint_fields = {
        "family": family,
        "rank": int(method_cfg["rank"]),
        "epochs": int(RANKEXT_EPOCHS if family == "rank_extension" else LORA_EPOCHS),
        "uses_kd": bool(method_cfg["uses_kd"]),
        "kd_class_scope": str(method_cfg.get("kd_class_scope", "full")),
        "kd_weight": float(method_cfg["kd_weight"]),
        "kd_temperature": float(method_cfg["kd_temperature"]),
        "kd_warmup_epochs": float(method_cfg.get("kd_warmup_epochs", 0.0)),
        "uses_dense_orth": bool(method_cfg.get("uses_dense_orth", False)),
        "uses_delta_trace": bool(method_cfg["uses_delta_trace"]),
        "uses_factor_orth": bool(method_cfg["uses_factor_orth"]),
        "lambda_orth": float(method_cfg["lambda_orth"]),
        "target_modules": sorted(family_target_modules(family)),
        "seed": int(SEED),
        "num_steps": int(NUM_STEPS),
        "classes_per_step": int(CLASSES_PER_STEP),
        "rankext_schedule": (list(get_rank_extension_rank_schedule()) if family == "rank_extension" else None),
        # FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT (task brief Section 30):
        # extended fingerprint fields not covered above.
        "lora_alpha": float(LORA_ALPHA),
        "protect_weight": (
            float(RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT) if method_name in RANKEXT_PROJECTED_PROTECT_METHODS else 0.0
        ),
        "rankext_new_block_warmup_enabled": (
            bool(method_name not in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS) if family == "rank_extension" else None
        ),
        "merge_method": ("arithmetic" if family == "simple_avg" else "persistent"),
        "classifier_restoration_mode": "fixed_inplace_assignment",  # tensor[idx] = value, never the historical .copy_() no-op
    }
    import hashlib
    blob = json.dumps(fingerprint_fields, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_arm_checkpoint_atomic(method_name, step_states, specialist_diagonal_accuracy):
    """Atomic, crash-safe persistence for one completed training arm. Writes
    to a temp file first, then os.replace()s it into place (atomic on both
    POSIX and Windows within the same filesystem) BEFORE writing the
    completion marker file -- so a crash mid-write leaves either nothing (a
    stray .tmp file, ignored on resume) or a fully-written checkpoint with no
    marker yet (also correctly treated as "not done" by
    load_arm_checkpoint_if_present, which requires BOTH files to exist),
    never a half-written checkpoint mistaken for a complete one. Saves BOTH
    the dense delta tensors AND the raw LoRA A/B factors (extract_lora_state()
    already returns both in each step_states[i] dict -- nothing extra to
    compute), plus the classifier weight/bias per step and the specialist
    diagonal accuracy dict train_independent_loras() returns -- everything
    needed to re-evaluate this arm under ANY merge operator later without
    ever retraining it again."""
    ckpt_path, marker_path = _arm_checkpoint_paths(method_name)
    tmp_path = ckpt_path + ".tmp"
    # READINESS-AUDIT ADDITION: fingerprint computed and stored AT SAVE TIME,
    # so a later reload can prove (not just assume) the reloaded step_states
    # are byte-identical to what this training run actually produced.
    specialist_fingerprint = compute_step_states_fingerprint(step_states)
    config_fingerprint = compute_method_config_fingerprint(method_name)
    payload = {
        "method_name": method_name,
        "step_states": step_states,
        "specialist_diagonal_accuracy": specialist_diagonal_accuracy,
        "specialist_fingerprint": specialist_fingerprint,
        "config_fingerprint": config_fingerprint,
        "saved_at_utc": datetime.utcnow().isoformat(),
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    with open(marker_path, "w") as f:
        f.write("done\n")
    print(f"[checkpoint] saved {method_name} -> {ckpt_path} | specialist_fingerprint={specialist_fingerprint}")


def load_arm_checkpoint_if_present(method_name):
    ckpt_path, marker_path = _arm_checkpoint_paths(method_name)
    if os.path.exists(marker_path) and os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location="cpu")
        # READINESS-AUDIT ADDITION: verify the reload is byte-identical to
        # what was saved -- a genuine reload-integrity check (not a
        # same-object triviality, since this is a fresh deserialization),
        # required for the resume test (Section 5 of the readiness audit).
        # Older checkpoints saved before this fix has no stored fingerprint
        # (specialist_fingerprint is None) -- skip verification rather than
        # falsely fail on a legitimately older, still-valid checkpoint.
        stored_fp = payload.get("specialist_fingerprint")
        if stored_fp is not None:
            reloaded_fp = compute_step_states_fingerprint(payload["step_states"])
            assert reloaded_fp == stored_fp, (
                f"CHECKPOINT RELOAD INTEGRITY CHECK FAILED for {method_name}: "
                f"reloaded fingerprint {reloaded_fp} != saved fingerprint {stored_fp}"
            )
        # FINAL-AUDIT FIX (persistence safety): reject a checkpoint whose
        # config_fingerprint does not match THIS run's current config for
        # method_name (e.g. the script was edited -- epochs/rank/lambda/KD
        # settings changed -- between the run that saved it and this one).
        # A stale/incompatible checkpoint is treated as absent (forces a
        # fresh retrain of just this arm), never silently reused. Checkpoints
        # saved before this fix have no stored config_fingerprint (None) --
        # also treated as stale/unverifiable rather than trusted blindly.
        stored_cfg_fp = payload.get("config_fingerprint")
        current_cfg_fp = compute_method_config_fingerprint(method_name)
        if stored_cfg_fp != current_cfg_fp:
            print(
                f"[resume] REJECTING stale checkpoint for {method_name}: "
                f"stored config_fingerprint={stored_cfg_fp} != current config_fingerprint={current_cfg_fp} "
                "-- this method's config changed since the checkpoint was saved; retraining from scratch."
            )
            return None
        print(
            f"[resume] found completed checkpoint for {method_name} "
            f"(saved_at_utc={payload.get('saved_at_utc')}, specialist_fingerprint={stored_fp}, "
            f"config_fingerprint={stored_cfg_fp}) -- skipping training."
        )
        return payload
    return None


def ensure_arm_trained(method_name):
    """Phase 1 of 2 for each arm (training). Resume-aware: if a completed
    checkpoint already exists on disk (from a previous, possibly crashed, run
    of THIS same job -- CHECKPOINTS_DIR is stable across restarts, see its
    own definition above), load it and skip training entirely; otherwise
    train exactly as the ORIGINAL run_simple_avg_variant() always did (same
    train_independent_loras() call, same config-driven dispatch), then save
    the result atomically. Populates simple_avg_step_states[method_name]
    either way."""
    cached = load_arm_checkpoint_if_present(method_name)
    if cached is not None:
        step_states = cached["step_states"]
        specialist_diagonal_accuracy = cached["specialist_diagonal_accuracy"]
        simple_avg_step_states[method_name] = step_states
        return step_states, specialist_diagonal_accuracy

    set_seed(SEED)  # same per-method reset as run_simple_avg_variant() -- see its own comment above
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    step_states, specialist_diagonal_accuracy = train_independent_loras(
        method_name=method_name,
        method_prefix=f"{method_name}_source",
        replay_per_class=0,
        use_orth=bool(method_cfg["uses_delta_trace"] or method_cfg["uses_factor_orth"] or method_cfg.get("uses_dense_orth", False)),
        orth_mode=(
            "delta_trace" if method_cfg["uses_delta_trace"]
            else ("factor_orth" if method_cfg["uses_factor_orth"]
            else ("dense_orth" if method_cfg.get("uses_dense_orth", False) else None))
        ),
        use_kd=bool(method_cfg["uses_kd"]),
        kd_weight=float(method_cfg["kd_weight"]),
        kd_temperature=float(method_cfg["kd_temperature"]),
        kd_class_scope=method_cfg.get("kd_class_scope", "full"),
        kd_warmup_epochs=float(method_cfg.get("kd_warmup_epochs", 0.0)),
        orth_train_records=train_diagnostic_rows,
        lambda_orth=float(method_cfg["lambda_orth"]),
    )
    simple_avg_step_states[method_name] = step_states
    save_arm_checkpoint_atomic(method_name, step_states, specialist_diagonal_accuracy)
    return step_states, specialist_diagonal_accuracy


def compute_step_states_fingerprint(step_states):
    """READINESS-AUDIT ADDITION -- pure integrity check, not a scientific
    computation. A stable (non-cryptographic-strength, but collision-safe for
    this purpose) content fingerprint of a step_states list's dense deltas +
    classifier weight/bias, used ONLY to prove that arithmetic and do_merge
    reference the EXACT SAME persisted specialist state for a given arm (the
    explicit "same-specialist ID/delta-hash" integrity assertion), and that a
    reloaded checkpoint (Section 7 of the preparation audit) is byte-identical
    to what was saved. Never read by any loss/merge/calibration/metric code."""
    import hashlib
    h = hashlib.sha256()
    for state in step_states:
        for key in sorted(state["deltas"].keys()):
            h.update(key.encode("utf-8"))
            h.update(state["deltas"][key].detach().cpu().contiguous().numpy().tobytes())
        h.update(state["classifier_weight"].detach().cpu().contiguous().numpy().tobytes())
        h.update(state["classifier_bias"].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def evaluate_arm_with_merge(training_method, merge_method, step_states, specialist_diagonal_accuracy, specialist_fingerprint):
    """Phase 2 of 2 for each arm (merge + evaluate), called once per
    (training_method, merge_method) pair from the SAME saved step_states --
    no specialist retraining occurs here. Identical evaluation pipeline for
    BOTH merge operators (apply_deltas_to_base -> calibration dispatch ->
    evaluate_model -> evaluate_per_step_accuracy -> compute_backward_transfer
    -- the exact same sequence run_simple_avg_variant() always used), so any
    accuracy difference between merge_method="arithmetic" and
    merge_method="do_merge" for the SAME training_method is strictly a
    merge-operator effect, never a calibration/classifier/evaluation-code
    difference. This is the audited fix for the gap found in Exp1's own
    dormant `do_merging_simple` block (which skips calibration entirely --
    see the preparation audit report) -- that block is left untouched and
    unused; this function is the correct replacement, not a patch to it.

    READINESS-AUDIT ADDITION: `specialist_fingerprint` is asserted against a
    fresh fingerprint of the `step_states` this call actually received --
    trivially true today (both merges are called from the same in-memory
    object within one loop iteration below) but a real, standing integrity
    guard against any future refactor that might accidentally pass a
    different/stale state to one of the two merges.

    READINESS-AUDIT ADDITION: for merge_method=="arithmetic" ONLY, the
    already-computed eval_rows are ALSO filed under the PLAIN (untagged)
    training_method name into all_results/method_summary_rows (Exp1's own
    legacy, single-merge-per-method result tables) -- zero extra model
    forward passes (the numbers are copied, not recomputed), so Exp1's large
    inherited downstream reporting/plotting tail (which expects exactly one
    row per ACTIVE_METHOD_NAMES entry, keyed by plain name) has real,
    non-empty data for this experiment's primary/arithmetic merge, instead of
    either crashing on absent tagged names or silently degrading to NaN. The
    do_merge results are NEVER filed under the plain name -- they remain
    visible ONLY in this script's own dedicated training_merge_summary_rows
    table, so the legacy tables cannot silently conflate the two merges."""
    method_cfg = ACTIVE_METHOD_MAP[training_method]
    tagged_name = f"{training_method}__merge_{merge_method}"

    _fresh_fingerprint = compute_step_states_fingerprint(step_states)
    assert _fresh_fingerprint == specialist_fingerprint, (
        f"SAME-SPECIALIST INTEGRITY CHECK FAILED for {training_method}/{merge_method}: "
        f"step_states fingerprint {_fresh_fingerprint} != expected {specialist_fingerprint}. "
        "Arithmetic and do_merge must reference the exact same persisted specialist state."
    )

    if merge_method == "arithmetic":
        merged_delta = simple_average_deltas(step_states)
    elif merge_method == "do_merge":
        merged_delta = do_merge_deltas(step_states)  # Exp1's OWN function (L3903-region), UNCHANGED
    else:
        raise ValueError(f"Unknown merge_method: {merge_method!r}")

    log_merge_mechanism(
        method_name=tagged_name,
        step_states=step_states,
        merged_delta=merged_delta,
        csv_path=os.path.join(TABLES_DIR, "merge_mechanism_by_training_and_merge_method.csv"),
    )

    merged_model = apply_deltas_to_base(merged_deltas=merged_delta, step_states=step_states)

    if method_cfg["apply_calibration"]:
        calibration_mode = method_cfg.get("calibration_mode", "global")
        if calibration_mode == "confidence_weighted_regime_grouped":
            merged_model = calibrate_classifier_row_norms_confidence_weighted(
                merged_model,
                epoch_loss_rows=epoch_loss_rows,
                method_name=tagged_name,
                uses_kd=bool(method_cfg["uses_kd"]),
            )
        else:
            merged_model = calibrate_classifier_row_norms(
                merged_model,
                mode=calibration_mode,
                uses_kd=bool(method_cfg["uses_kd"]),
                method_name=tagged_name,
            )
    else:
        log_classifier_row_norm_diagnostics(merged_model, tagged_name, phase="pre_calibration")

    eval_rows = evaluate_model(merged_model, tagged_name)
    final_per_step_accuracy = evaluate_per_step_accuracy(merged_model, tagged_name)
    backward_transfer = compute_backward_transfer(specialist_diagonal_accuracy, final_per_step_accuracy)
    print(
        f"[singlejob eval] training={training_method} | merge={merge_method} | "
        f"specialist_fingerprint={specialist_fingerprint} | backward_transfer={backward_transfer}"
    )

    # JOINT PACK FIX (schema-completeness, infrastructure-only): populate
    # restricted_mean for SimpleAvg rows too, from the SAME per_step_accuracy_
    # restricted_rows accumulator evaluate_per_step_accuracy() just populated
    # under tagged_name -- mirrors RankExt's finalize_rank_extension_arm()
    # (which already computes this). No metric computation changes; this is
    # purely reading an already-computed value into the result table.
    # BUGFIX (found via local FAST_RUN_DEBUG sanity run): the actual column
    # name per_step_accuracy_restricted_rows uses is "accuracy_restricted",
    # not "accuracy" -- confirmed by reading evaluate_per_step_accuracy()'s
    # own row-construction code (L3997-4003). The original "accuracy" key
    # never existed and would KeyError on the very first arm's finalization
    # in BOTH Phase 1 and Phase 2 (this is not a debug-only bug).
    _sa_restricted_rows_this_arm = [r for r in per_step_accuracy_restricted_rows if r.get("method") == tagged_name]
    _sa_restricted_mean = (
        float(np.mean([r["accuracy_restricted"] for r in _sa_restricted_rows_this_arm]))
        if len(_sa_restricted_rows_this_arm) > 0 else np.nan
    )

    append_training_merge_summary(
        training_method, merge_method, eval_rows,
        backward_transfer=backward_transfer, specialist_fingerprint=specialist_fingerprint,
        restricted_mean=_sa_restricted_mean,
    )

    if merge_method == "arithmetic":
        # READINESS-AUDIT ADDITION -- legacy-compat population, see docstring.
        eval_rows_plain = [{**row, "method": training_method} for row in eval_rows]
        all_results.extend(eval_rows_plain)
        append_simple_method_summary(
            training_method, eval_rows_plain, backward_transfer=backward_transfer, forward_transfer=np.nan,
        )

    del merged_model
    cleanup()


# FINAL 8-ARM SCRIPT: DO-Merge is REMOVED from this script entirely, per the
# task brief ("Do NOT use DO-Merge anywhere in this experiment"). Only
# arithmetic merging (simple_average_deltas(), unchanged) is evaluated for
# SimpleAvg.
MERGE_METHODS_TO_EVALUATE = ["arithmetic"]
assert MERGE_METHODS_TO_EVALUATE == _expected_merge_methods, (
    "MERGE_METHODS_TO_EVALUATE drifted from the readiness-verification block's expectation"
)

simple_avg_execution_order = [cfg["method"] for cfg in ACTIVE_METHOD_CONFIGS if cfg["family"] == "simple_avg"]
for method_name in simple_avg_execution_order:
    base_method = ACTIVE_METHOD_MAP[method_name]["base_method"]
    if not METHODS_TO_RUN.get(base_method, False):
        print(f"Skipping {method_name} because {base_method} is disabled")
        continue

    _step_states, _specialist_diag_acc = ensure_arm_trained(method_name)
    _specialist_fingerprint = compute_step_states_fingerprint(_step_states)
    for _merge_method in MERGE_METHODS_TO_EVALUATE:
        evaluate_arm_with_merge(method_name, _merge_method, _step_states, _specialist_diag_acc, _specialist_fingerprint)

# ==============================================================================
# SINGLEJOB PACK ADDITION -- write the primary training x merge result table
# to disk IMMEDIATELY after the execution loop above, before any of Exp1's
# downstream reporting/plotting code (below this point, inherited unaudited
# from the 8-method RankExt+SimpleAvg comparison) runs. This guarantees the
# scientifically important numbers are safely persisted even if some later,
# RankExt-oriented plot/table generator (not re-verified for this reduced,
# RankExt-free 5-arm/2-merge configuration) errors out -- see the preparation
# audit report's "static validation" section for exactly what was and was not
# re-verified.
# ==============================================================================
training_merge_summary_df = pd.DataFrame(training_merge_summary_rows)
training_merge_summary_csv_path = os.path.join(TABLES_DIR, "simpleavg_singlejob_training_x_merge_results.csv")
training_merge_summary_df.to_csv(training_merge_summary_csv_path, index=False)
print(f"[singlejob] wrote {len(training_merge_summary_df)} training x merge rows -> {training_merge_summary_csv_path}")

# Historical R7 (Exp1) canonical seed-42 reference values, per the explicit
# instruction NOT to retrain these -- hardcoded from the already-published,
# already-verified R7 result artifacts (see thesis_agent/reports/
# exp1_exp2_r8_supervisor_decision_synthesis.md Section 4 / R7's own
# final_metrics_all_methods.csv), written to a SEPARATE CSV so hardcoded
# historical numbers are never conflated in the same table/provenance as this
# run's own freshly-computed results.
historical_r7_reference_rows = [
    {"training_method": "simple_avg", "merge_method": "arithmetic", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED vs this run, see report)", "first_step": 77.75, "later_steps": 74.5125, "all_seen": 75.16, "restricted_mean": 92.28, "backward_transfer": -0.212},
    {"training_method": "simple_avg_kd_T2", "merge_method": "arithmetic", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED vs this run, see report)", "first_step": 92.35, "later_steps": 55.6375, "all_seen": 62.98, "restricted_mean": 91.91, "backward_transfer": -0.252},
    {"training_method": "simple_avg_factor_orth", "merge_method": "arithmetic", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED vs this run, see report)", "first_step": 76.45, "later_steps": 73.75, "all_seen": 74.29, "restricted_mean": 93.24, "backward_transfer": -0.227},
    {"training_method": "simple_avg_factor_orth_kd_T2", "merge_method": "arithmetic", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED vs this run, see report)", "first_step": 69.10, "later_steps": 69.475, "all_seen": 69.40, "restricted_mean": 92.85, "backward_transfer": -0.219},
    # RankExt historical R7/Exp1 reference (9 epochs, OLD non-restoring
    # classifier-row bug, full-100-way KD -- triple-confounded vs this run's
    # 4 RankExt arms; see the implementation audit report's cross-run
    # comparison section for the required caveats).
    {"training_method": "rank_extension", "merge_method": "persistent", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED, old restoration bug + full-KD, see report)", "first_step": np.nan, "later_steps": np.nan, "all_seen": 35.31, "restricted_mean": np.nan, "backward_transfer": np.nan},
    {"training_method": "rank_extension_kd_full", "merge_method": "persistent", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED, old restoration bug + full-KD, see report)", "first_step": np.nan, "later_steps": np.nan, "all_seen": 64.44, "restricted_mean": np.nan, "backward_transfer": np.nan},
    {"training_method": "rank_extension_orth_factor_lam_50", "merge_method": "persistent", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED, old restoration bug, see report)", "first_step": np.nan, "later_steps": np.nan, "all_seen": 42.21, "restricted_mean": np.nan, "backward_transfer": np.nan},
    {"training_method": "rank_extension_orth_factor_lam_50_kd_full", "merge_method": "persistent", "source": "R7/Exp1 seed42, 9ep (historical, not retrained -- CONFOUNDED, old restoration bug + full-KD, see report)", "first_step": np.nan, "later_steps": np.nan, "all_seen": 70.72, "restricted_mean": np.nan, "backward_transfer": np.nan},
    # Current-pipeline job 4969059 reference (6 epochs, corrected
    # restoration, old-seen KD, DenseOrth -- the DIRECT baseline for this
    # run's family-specific redesign; only the orth formulation/KD weight
    # changed relative to these).
    {"training_method": "simple_avg", "merge_method": "arithmetic", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 81.50, "later_steps": 74.60, "all_seen": 75.98, "restricted_mean": 92.14, "backward_transfer": -0.2022},
    {"training_method": "simple_avg_kd_oldseen_T2_weight1_no_warmup", "merge_method": "arithmetic", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 62.55, "later_steps": 78.175, "all_seen": 75.05, "restricted_mean": 93.45, "backward_transfer": -0.2151},
    {"training_method": "simple_avg_kd_oldseen_T2_weight1_warmup", "merge_method": "arithmetic", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 78.40, "later_steps": 76.225, "all_seen": 76.66, "restricted_mean": 93.19, "backward_transfer": -0.1946},
    {"training_method": "simple_avg_dense_orth_lam20", "merge_method": "arithmetic", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 81.00, "later_steps": 73.6375, "all_seen": 75.11, "restricted_mean": 91.94, "backward_transfer": -0.2169},
    {"training_method": "simple_avg_dense_orth_lam20_kd_oldseen_T2_weight1_warmup", "merge_method": "arithmetic", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 78.30, "later_steps": 75.6625, "all_seen": 76.19, "restricted_mean": 92.91, "backward_transfer": -0.1929},
    {"training_method": "rank_extension", "merge_method": "persistent", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 8.4, "later_steps": 36.2875, "all_seen": 30.71, "restricted_mean": 89.95, "backward_transfer": -0.8069},
    {"training_method": "rank_extension_kd_oldseen_T2_weight1_warmup", "merge_method": "persistent", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 0.25, "later_steps": 58.9875, "all_seen": 47.24, "restricted_mean": np.nan, "backward_transfer": -0.558},
    {"training_method": "rank_extension_dense_orth_lam20", "merge_method": "persistent", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 9.45, "later_steps": 36.1125, "all_seen": 30.78, "restricted_mean": np.nan, "backward_transfer": -0.805625},
    {"training_method": "rank_extension_dense_orth_lam20_kd_oldseen_T2_weight1_warmup", "merge_method": "persistent", "source": "job4969059 seed42, 6ep, current pipeline (not retrained)", "first_step": 0.25, "later_steps": 58.375, "all_seen": 46.75, "restricted_mean": np.nan, "backward_transfer": -0.563375},
]
historical_r7_reference_csv_path = os.path.join(TABLES_DIR, "historical_r7_reference_values_not_retrained.csv")
pd.DataFrame(historical_r7_reference_rows).to_csv(historical_r7_reference_csv_path, index=False)
print(f"[singlejob] wrote historical R7 reference rows -> {historical_r7_reference_csv_path}")

step_states_no_replay = simple_avg_step_states.get("simple_avg")
step_states_no_replay_orth = simple_avg_step_states.get("simple_avg_factor_orth")
step_states_simple_kd = simple_avg_step_states.get("simple_avg_kd_T2")
step_states_simple_factor_orth = simple_avg_step_states.get("simple_avg_factor_orth")
step_states_simple_factor_orth_kd = simple_avg_step_states.get("simple_avg_factor_orth_kd_T2")

if len(train_diagnostic_rows) > 0:
    print(f"[simple_avg] accumulated training-loss rows: {len(train_diagnostic_rows)}")


# In[ ]:


step_states_replay = None

if METHODS_TO_RUN["simple_avg_replay"]:
    step_states_replay = train_independent_loras(
        method_name="simple_avg_replay",
        method_prefix="simple_avg_replay_source",
        replay_per_class=REPLAY_PER_CLASS,
        orth_train_records=train_diagnostic_rows,
    )

    replay_delta = simple_average_deltas(step_states_replay)
    replay_model = apply_deltas_to_base(
        merged_deltas=replay_delta,
        step_states=step_states_replay,
    )

    replay_eval_rows = evaluate_model(replay_model, "simple_avg_replay")
    append_simple_method_summary("simple_avg_replay", replay_eval_rows, use_kd=False, orth_mode="none", lambda_orth=0.0, replay_per_class=REPLAY_PER_CLASS)

    del replay_model
    cleanup()

else:
    print("Skipping simple_avg_replay")


# In[ ]:


if METHODS_TO_RUN["do_merging_simple"]:
    assert step_states_no_replay is not None, "step_states_no_replay is required for do_merging_simple"

    do_delta = do_merge_deltas(step_states_no_replay)
    do_layer_count = len(do_delta)
    expected_do_layers = len(step_states_no_replay[0]["deltas"]) if len(step_states_no_replay) > 0 else 0
    print(f"[DO-Merging] merged layer count: {do_layer_count}")
    if abs(do_layer_count - expected_do_layers) > 2:
        print(
            f"[WARNING] do_merging_simple merged {do_layer_count} layers; expected around {expected_do_layers}."
        )
    do_model = apply_deltas_to_base(
        merged_deltas=do_delta,
        step_states=step_states_no_replay,
    )

    do_eval_rows = evaluate_model(do_model, "do_merging_simple")
    do_eval_map = {row["eval_set"]: float(row["accuracy"]) for row in do_eval_rows}
    method_summary_rows.append({
        "method": "do_merging_simple",
        "orth_mode": "none",
        "lambda_orth": 0.0,
        "zero_old_merge": False,
        "use_kd": False,
        "kd_weight": 0.0,
        "kd_temperature": 0.0,
        "replay_per_class": 0,
        "old_active_in_forward": np.nan,
        "first_step": do_eval_map.get("first_step", np.nan),
        "later_steps": do_eval_map.get("later_steps", np.nan),
        "all_seen": do_eval_map.get("all_seen", np.nan),
        "old_new_gap": do_eval_map.get("first_step", np.nan) - do_eval_map.get("later_steps", np.nan),
        "avg_forgetting": np.nan,
    })

    del do_model
    cleanup()

else:
    print("Skipping do_merging_simple")

if METHODS_TO_RUN.get("do_merging_simple_orth", False):
    assert step_states_no_replay_orth is not None, "step_states_no_replay_orth is required for do_merging_simple_orth"

    do_orth_delta = do_merge_deltas(step_states_no_replay_orth)
    do_orth_model = apply_deltas_to_base(
        merged_deltas=do_orth_delta,
        step_states=step_states_no_replay_orth,
    )

    evaluate_model(do_orth_model, "do_merging_simple_orth")

    del do_orth_model
    cleanup()

else:
    print("Skipping do_merging_simple_orth")


# In[ ]:


def extract_reference_weights_for_orth(peft_model):
    """
    Extract M_(t-1) for every current LoRA target module.
    These are the base q_proj/v_proj weights before training the current LoRA.
    """
    reference_weights = {}

    for name, module in peft_model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )

        if not has_lora:
            continue

        plain_name = normalize_module_name(name)

        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            reference_weights[plain_name] = module.base_layer.weight.detach().cpu().float().clone()
        elif hasattr(module, "weight"):
            reference_weights[plain_name] = module.weight.detach().cpu().float().clone()

    return reference_weights

def compute_orth_penalty(model, reference_weights, eps=1e-8):

    penalties = []
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )

        if not has_lora:
            continue

        plain_name = normalize_module_name(name)

        if plain_name not in reference_weights:
            continue

        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight

        scaling = (
            module.scaling["default"]
            if isinstance(module.scaling, dict)
            else module.scaling
        )

        delta = (B @ A) * float(scaling)
        old_weight = reference_weights[plain_name].to(
            device=delta.device,
            dtype=delta.dtype,
        )

        trace_value = torch.sum(old_weight * delta)
        normalized_trace = trace_value / (
            torch.linalg.norm(old_weight).clamp_min(eps)
            * torch.linalg.norm(delta).clamp_min(eps)
        )

        penalties.append(normalized_trace.pow(2))

    if not penalties:
        return torch.tensor(0.0, device=device)

    return torch.stack(penalties).mean()

def compute_orth_diagnostics(model, reference_weights, eps=1e-8):
    
    rows = []

    for name, module in model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )

        if not has_lora:
            continue

        plain_name = normalize_module_name(name)

        if plain_name not in reference_weights:
            continue

        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight

        scaling = (
            module.scaling["default"]
            if isinstance(module.scaling, dict)
            else module.scaling
        )

        delta = (B @ A) * float(scaling)
        old_weight = reference_weights[plain_name].to(
            device=delta.device,
            dtype=delta.dtype,
        )

        trace_value = torch.sum(old_weight * delta)
        normalized_trace = trace_value / (
            torch.linalg.norm(old_weight).clamp_min(eps)
            * torch.linalg.norm(delta).clamp_min(eps)
        )

        rows.append({
            "layer": plain_name,
            "raw_trace": float(trace_value.detach().cpu()),
            "normalized_trace": float(normalized_trace.detach().cpu()),
            "squared_penalty": float(normalized_trace.pow(2).detach().cpu()),
            "delta_norm": float(torch.linalg.norm(delta).detach().cpu()),
            "reference_norm": float(torch.linalg.norm(old_weight).detach().cpu()),
        })

    if len(rows) == 0:
        print("[orth diagnostics] no matched LoRA/reference layers")
        return pd.DataFrame()

    diag_df = pd.DataFrame(rows)
    summary = diag_df[
        [
            "raw_trace",
            "normalized_trace",
            "squared_penalty",
            "delta_norm",
            "reference_norm",
        ]
    ].mean()

    print("[orth diagnostics] mean over matched q_proj/v_proj layers")
    print(summary.round(6))

    return diag_df

class OrthogonalLossTrainer(Trainer):


    def __init__(self, *args, reference_weights=None, lambda_orth=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_weights = reference_weights or {}
        self.lambda_orth = float(lambda_orth)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        orth_loss = compute_orth_penalty(
            model=model,
            reference_weights=self.reference_weights,
        )

        loss = ce_loss + self.lambda_orth * orth_loss

        return (loss, outputs) if return_outputs else loss

if METHODS_TO_RUN["orthogonal_loss"]:
    orth_model = fresh_pretrained_model()

    for step_idx in range(NUM_STEPS):
        print(f"\n===== orthogonal_loss | step {step_idx + 1}/{NUM_STEPS} =====")

        orth_peft_model = add_lora(orth_model)
        orth_peft_model.print_trainable_parameters()

        reference_weights = extract_reference_weights_for_orth(orth_peft_model)

        train_ds = make_train_dataset(step_idx, replay_per_class=0)
        eval_ds = make_eval_dataset(classes_for_step(step_idx))

        train_with_trainer(
            model=orth_peft_model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            output_dir=os.path.join(MODELS_DIR, f"orthogonal_loss_step_{step_idx + 1}"),
            epochs=ORTH_EPOCHS,
            lr=LR_ORTH,
            batch_size=BATCH_LORA,
            accum_steps=ACCUM_LORA,
            trainer_cls=OrthogonalLossTrainer,
            reference_weights=reference_weights,
            lambda_orth=LAMBDA_ORTH,
        )

        if ORTH_DIAGNOSTICS:
            compute_orth_diagnostics(
                model=orth_peft_model,
                reference_weights=reference_weights,
            )

        orth_model = orth_peft_model.merge_and_unload()

        del orth_peft_model
        cleanup()

    evaluate_model(orth_model, "orthogonal_loss")

    del orth_model
    cleanup()

else:
    print("Skipping orthogonal_loss")


# In[ ]:


from transformers import TrainerCallback


class GrowingRankLoRALinear(nn.Module):
    """
    One growing LoRA pair per layer.
    Frozen slice stores previous ranks; new slice is trainable.
    """

    def __init__(
        self,
        base_layer,
        total_rank,
        frozen_A=None,
        frozen_B=None,
        dropout=0.0,
        old_active_in_forward=True,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.total_rank = int(total_rank)
        self.old_active_in_forward = bool(old_active_in_forward)

        if frozen_A is None or frozen_B is None:
            self.frozen_rank = 0
        else:
            if frozen_A.shape[0] != frozen_B.shape[1]:
                raise ValueError(
                    f"A/B frozen rank mismatch: A={tuple(frozen_A.shape)}, B={tuple(frozen_B.shape)}"
                )
            self.frozen_rank = int(frozen_A.shape[0])

        self.new_rank = self.total_rank - self.frozen_rank
        if self.new_rank < 0:
            raise ValueError(
                f"new_rank < 0 | total_rank={self.total_rank} frozen_rank={self.frozen_rank}"
            )

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.rankext_alpha = RANKEXT_ALPHA_PER_RANK * self.total_rank
        self.scaling = self.rankext_alpha / self.total_rank
        self.dropout = nn.Dropout(dropout)

        if self.frozen_rank > 0:
            self.A_frozen = nn.Parameter(frozen_A.detach().clone().float(), requires_grad=False)
            self.B_frozen = nn.Parameter(frozen_B.detach().clone().float(), requires_grad=False)
        else:
            self.A_frozen = None
            self.B_frozen = None

        if self.new_rank > 0:
            self.A_new = nn.Parameter(torch.zeros(self.new_rank, self.in_features))
            self.B_new = nn.Parameter(torch.zeros(self.out_features, self.new_rank))
            nn.init.kaiming_uniform_(self.A_new, a=np.sqrt(5))
            nn.init.zeros_(self.B_new)
        else:
            self.A_new = None
            self.B_new = None

    def full_A_B(self):
        A_parts = []
        B_parts = []
        if self.frozen_rank > 0:
            A_parts.append(self.A_frozen.to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype))
            B_parts.append(self.B_frozen.to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype))
        if self.new_rank > 0:
            A_parts.append(self.A_new)
            B_parts.append(self.B_new)
        if len(A_parts) == 0:
            raise ValueError("No LoRA blocks available in full_A_B.")
        A = torch.cat(A_parts, dim=0)
        B = torch.cat(B_parts, dim=1)
        return A, B

    def current_new_delta(self):
        if self.new_rank <= 0:
            return None
        return (self.B_new @ self.A_new) * float(self.scaling)

    def cumulative_old_delta(self):
        if self.frozen_rank <= 0:
            return None
        return (self.B_frozen @ self.A_frozen) * float(self.scaling)

    def forward(self, x):
        base_out = self.base_layer(x)
        x_dropped = self.dropout(x)
        out = base_out

        if self.old_active_in_forward and self.frozen_rank > 0:
            hidden_old = torch.matmul(x_dropped, self.A_frozen.T)
            lora_old = torch.matmul(hidden_old, self.B_frozen.T)
            out = out + self.scaling * lora_old

        if self.new_rank > 0:
            hidden_new = torch.matmul(x_dropped, self.A_new.T)
            lora_new = torch.matmul(hidden_new, self.B_new.T)
            # RANKEXT_NEW_BLOCK_WARMUP_ENABLED (analysis_rankext_plain/): scales
            # only the NEW block's contribution, never the base/frozen-old
            # term above. Always exactly 1.0 (a true no-op, not just close to
            # it) outside a rank_extension training step -- see
            # get_rankext_new_block_warmup_multiplier()'s module-level state
            # and train_with_trainer()'s unconditional reset after
            # trainer.train() returns.
            new_block_multiplier = get_rankext_new_block_warmup_multiplier()
            out = out + self.scaling * lora_new * new_block_multiplier

        return out


def get_parent_module_and_child_name(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def find_clip_target_linear_modules(model, target_modules=None):
    resolved_target_modules = list(TARGET_MODULES) if target_modules is None else list(target_modules)
    target_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(target_name) for target_name in resolved_target_modules):
            target_names.append(name)
    return target_names


def get_rank_extension_rank_schedule():
    schedule = [int(v) for v in active_rankext_rank_schedule()]
    if len(schedule) != NUM_STEPS:
        raise ValueError(f"active rank schedule must have NUM_STEPS={NUM_STEPS} entries, got {schedule}")
    for i in range(1, len(schedule)):
        if schedule[i] <= schedule[i - 1]:
            raise ValueError(f"RANKEXT_RANK_SCHEDULE must be strictly increasing, got {schedule}")
    return schedule


def get_rank_extension_rank_triplet(step_idx):
    schedule = get_rank_extension_rank_schedule()
    total_rank = int(schedule[step_idx])
    frozen_rank = int(schedule[step_idx - 1]) if step_idx > 0 else 0
    new_rank = int(total_rank - frozen_rank)
    if new_rank <= 0:
        raise ValueError(
            f"Rank schedule must leave a positive new block at each step. step_idx={step_idx}, schedule={schedule}"
        )
    return total_rank, frozen_rank, new_rank


def build_rank_extension_model(previous_rank_state=None, step_idx=0, old_active_in_forward=True):
    model = fresh_pretrained_model()

    for _, p in model.vision_model.named_parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    total_rank, expected_frozen_rank, expected_new_rank = get_rank_extension_rank_triplet(step_idx)
    rankext_target_modules = family_target_modules("rank_extension")
    target_names = find_clip_target_linear_modules(model, target_modules=rankext_target_modules)
    model._rank_extension_target_names = list(target_names)
    model._rank_extension_old_active_in_forward = bool(old_active_in_forward)

    print(f"[rank_extension] Step {step_idx + 1}")
    print(f"  total_rank: {total_rank}")
    print(f"  target linear modules: {len(target_names)}")
    print(f"  target module names: {rankext_target_modules}")
    print(f"  rank schedule: {get_rank_extension_rank_schedule()}")
    print(f"  expected_frozen_rank: {expected_frozen_rank}")
    print(f"  expected_new_rank: {expected_new_rank}")
    print(f"  old_active_in_forward: {bool(old_active_in_forward)}")

    for module_name in target_names:
        parent, child_name = get_parent_module_and_child_name(model, module_name)
        base_layer = getattr(parent, child_name)

        frozen_A = None
        frozen_B = None
        if previous_rank_state is not None and module_name in previous_rank_state["lora"]:
            frozen_A = previous_rank_state["lora"][module_name]["A"]
            frozen_B = previous_rank_state["lora"][module_name]["B"]

        setattr(
            parent,
            child_name,
            GrowingRankLoRALinear(
                base_layer=base_layer,
                total_rank=total_rank,
                frozen_A=frozen_A,
                frozen_B=frozen_B,
                dropout=LORA_DROPOUT,
                old_active_in_forward=old_active_in_forward,
            ),
        )

    if previous_rank_state is not None and previous_rank_state["classifier_weight"] is not None:
        with torch.no_grad():
            model.classifier.weight.copy_(
                previous_rank_state["classifier_weight"].to(
                    device=model.classifier.weight.device,
                    dtype=model.classifier.weight.dtype,
                )
            )
            model.classifier.bias.copy_(
                previous_rank_state["classifier_bias"].to(
                    device=model.classifier.bias.device,
                    dtype=model.classifier.bias.dtype,
                )
            )

    return model


def extract_rank_extension_state(model):
    state = {"lora": {}, "classifier_weight": None, "classifier_bias": None}
    for name, module in model.named_modules():
        if isinstance(module, GrowingRankLoRALinear):
            A, B = module.full_A_B()
            state["lora"][name] = {
                "A": A.detach().cpu().clone(),
                "B": B.detach().cpu().clone(),
                "scaling": float(module.scaling),
                "total_rank": int(module.total_rank),
                "frozen_rank": int(module.frozen_rank),
                "new_rank": int(module.new_rank),
                "rankext_alpha": float(module.rankext_alpha),
            }
    state["classifier_weight"] = model.classifier.weight.detach().cpu().clone()
    state["classifier_bias"] = model.classifier.bias.detach().cpu().clone()
    return state


def rank_extension_trainable_classifier_classes(step_idx, replay_per_class):
    classes = list(classes_for_step(step_idx))
    if replay_per_class > 0:
        for old_step in range(step_idx):
            classes.extend(classes_for_step(old_step))
    return sorted(set(int(c) for c in classes))


def add_classifier_row_gradient_mask(model, trainable_classes):
    trainable_classes = set(int(c) for c in trainable_classes)
    mask_w = torch.zeros_like(model.classifier.weight)
    mask_b = torch.zeros_like(model.classifier.bias)
    for c in trainable_classes:
        mask_w[c, :] = 1.0
        mask_b[c] = 1.0
    hook_w = model.classifier.weight.register_hook(
        lambda grad: grad * mask_w.to(device=grad.device, dtype=grad.dtype)
    )
    hook_b = model.classifier.bias.register_hook(
        lambda grad: grad * mask_b.to(device=grad.device, dtype=grad.dtype)
    )
    return [hook_w, hook_b]


def snapshot_protected_classifier_rows(model, trainable_classes):
    trainable_classes = set(int(c) for c in trainable_classes)
    protected_rows = [c for c in range(NUM_CLASSES) if c not in trainable_classes]
    return {
        "rows": protected_rows,
        "weight": model.classifier.weight.detach().cpu().clone(),
        "bias": model.classifier.bias.detach().cpu().clone(),
    }


def restore_protected_classifier_rows(model, snapshot):
    """FINAL-AUDIT FIX (root cause: PyTorch advanced-indexing semantics):
    `tensor[long_tensor_index].copy_(value)` does NOT mutate `tensor` in
    place -- indexing with a LongTensor is "advanced indexing" and
    `__getitem__` returns a NEW (copied) tensor; `.copy_()` then writes into
    that throwaway copy, which is immediately discarded. The previous version
    of this function used exactly that pattern for both `weight` and `bias`,
    making the "hard restoration" a complete, silent no-op every single time
    it was called (both the per-CL-step call in train_rank_extension_arm()
    and the per-BATCH-STEP call from ClassifierRowRestoreCallback.on_step_end
    -- so protected classifier rows for old classes were never actually
    restored, only gradient-masked, for the entire lifetime of this script,
    including in the unmodified reference implementation this function was
    ported from verbatim). Fixed by using `tensor[index] = value`
    (`__setitem__`), which PyTorch implements as an in-place `index_put_` and
    genuinely mutates the original parameter -- confirmed against a minimal
    repro (`x[idx].copy_(v)` leaves `x` unchanged; `x[idx] = v` does not)
    before applying this fix. Verified empirically via
    classifier_protected_row_max_diff() (unaffected by this bug -- it reads
    values directly, not through this indexing pattern), which is exactly the
    diagnostic that should now report ~0.0 post-restore instead of a
    nonzero/growing drift.
    """
    rows = snapshot["rows"]
    if len(rows) == 0:
        return
    with torch.no_grad():
        row_idx = torch.tensor(rows, device=model.classifier.weight.device, dtype=torch.long)
        model.classifier.weight[row_idx] = snapshot["weight"][rows].to(
            device=model.classifier.weight.device,
            dtype=model.classifier.weight.dtype,
        )
        model.classifier.bias[row_idx] = snapshot["bias"][rows].to(
            device=model.classifier.bias.device,
            dtype=model.classifier.bias.dtype,
        )


def classifier_protected_row_max_diff(model, snapshot):
    rows = snapshot["rows"]
    if len(rows) == 0:
        return 0.0
    with torch.no_grad():
        weight_diff = (
            model.classifier.weight.detach().cpu()[rows] - snapshot["weight"][rows]
        ).abs().max().item()
        bias_diff = (
            model.classifier.bias.detach().cpu()[rows] - snapshot["bias"][rows]
        ).abs().max().item()
    return max(weight_diff, bias_diff)


# FINAL 8-ARM SCRIPT ADDITION (Part 11, classifier restoration diagnostics).
# Pure diagnostics -- reuses the exact same snapshot/model-state read as
# classifier_protected_row_max_diff() above (never called from
# restore_protected_classifier_rows(), so it cannot affect what gets
# restored); this just also returns weight and bias separately instead of
# collapsing them into one max() value, so both can be logged as their own
# columns per the task brief.
def classifier_protected_row_diffs(model, snapshot):
    rows = snapshot["rows"]
    if len(rows) == 0:
        return 0.0, 0.0
    with torch.no_grad():
        weight_diff = (
            model.classifier.weight.detach().cpu()[rows] - snapshot["weight"][rows]
        ).abs().max().item()
        bias_diff = (
            model.classifier.bias.detach().cpu()[rows] - snapshot["bias"][rows]
        ).abs().max().item()
    return float(weight_diff), float(bias_diff)


# FINAL 8-ARM SCRIPT ADDITION (Part 10, CL retention/plasticity diagnostics).
# Restricted-accuracy analogue of evaluate_seen_step_accuracies() (defined
# later in this file, both are used together from train_rank_extension_arm()
# below) -- same eval-only Trainer construction, same per-previously-seen-step
# loop, but reads restricted_argmax_accuracy() (already used elsewhere in this
# file for the FINAL-model-only per-step table) at every intermediate CL step
# instead of only at the very end. Pure evaluation; no training-time behavior
# is touched by this function.
def evaluate_seen_step_accuracies_restricted(model, upto_step_idx):
    args = get_training_args(
        output_dir=os.path.join(MODELS_DIR, "tmp_seen_eval_restricted"),
        epochs=1,
        lr=LR_LORA,
        batch_size=BATCH_LORA,
        accum_steps=ACCUM_LORA,
        train_dataset_len=None,
        eval_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    step_acc = {}
    for task_step in range(upto_step_idx + 1):
        task_classes = classes_for_step(task_step)
        eval_ds = make_eval_dataset(task_classes)
        pred_out = trainer.predict(eval_ds)
        logits = np.asarray(pred_out.predictions)
        labels = np.asarray(pred_out.label_ids)
        step_acc[int(task_step)] = float(restricted_argmax_accuracy(logits, labels, task_classes))
    return step_acc


class ClassifierRowRestoreCallback(TrainerCallback):
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            restore_protected_classifier_rows(model, self.snapshot)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            restore_protected_classifier_rows(model, self.snapshot)
        return control


class RankExtNewBlockWarmupCallback(TrainerCallback):
    """RANKEXT_NEW_BLOCK_WARMUP_ENABLED (analysis_rankext_plain/): drives the
    module-level new-block-output multiplier GrowingRankLoRALinear.forward()
    reads. on_step_begin updates the multiplier every step (fine-grained,
    smooth within-epoch ramp -- fires before that step's forward/backward, so
    the multiplier used in a given batch's forward pass is always the value
    computed from this step's own state.epoch, never a stale one). One
    diagnostic row is appended per (method, step, local_epoch) instead of per
    batch, to keep the saved table small while still recording whether the
    ramp actually happened. on_train_end resets to 1.0 as a second,
    redundant safety net on top of train_with_trainer()'s own unconditional
    reset (same belt-and-suspenders pattern as ClassifierRowRestoreCallback
    above)."""

    def __init__(self, method_name, step_idx, warmup_epochs, diagnostic_rows):
        self.method_name = str(method_name)
        self.step_idx = int(step_idx)
        self.warmup_epochs = float(warmup_epochs)
        self.diagnostic_rows = diagnostic_rows
        self._logged_epochs = set()

    def _current_multiplier(self, state):
        epoch_val = float(state.epoch) if state.epoch is not None else float("nan")
        return orth_lambda_warmup_multiplier(epoch_val, self.warmup_epochs, True), epoch_val

    def on_step_begin(self, args, state, control, **kwargs):
        multiplier, _ = self._current_multiplier(state)
        set_rankext_new_block_warmup_multiplier(multiplier)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        multiplier, epoch_val = self._current_multiplier(state)
        epoch_int = int(max(0, round(epoch_val))) if not np.isnan(epoch_val) else 0
        if epoch_int not in self._logged_epochs:
            self._logged_epochs.add(epoch_int)
            self.diagnostic_rows.append({
                "method_name": self.method_name,
                "step_id": self.step_idx + 1,
                "local_epoch": epoch_int,
                "new_block_warmup_multiplier": multiplier,
                "warmup_epochs_configured": self.warmup_epochs,
            })
        return control

    def on_train_end(self, args, state, control, **kwargs):
        set_rankext_new_block_warmup_multiplier(1.0)
        return control


class RankExtensionTrainer(HeadLRTrainerMixin, Trainer):
    def __init__(self, *args, classifier_snapshot=None, **kwargs):
        # CRASH FIX (rank_extension feature-anchor lever): Trainer.
        # prediction_step() (i.e. every eval/evaluate() call) invokes this
        # same DeltaOrthRankExtensionTrainer.compute_loss() used for
        # training, and compute_loss()'s `outputs` can carry more than one
        # non-loss tensor (classification logits + penultimate CLS hidden
        # state, only when the feature-anchor lever is active -- see that
        # method). HF packs every non-loss ModelOutput entry into
        # `predictions`, so without this hook compute_metrics() would
        # receive a (logits, hidden_states) tuple instead of a plain logits
        # tensor. Wiring it in here -- rather than relying solely on
        # compute_metrics() being defensive -- also drops the (much larger)
        # hidden-state tensors right after each eval batch instead of
        # accumulating them across the whole eval set. Only set if the
        # caller hasn't already supplied one (setdefault, not overwrite).
        kwargs.setdefault("preprocess_logits_for_metrics", self._select_classification_logits)
        super().__init__(*args, **kwargs)
        self.classifier_snapshot = classifier_snapshot
        if classifier_snapshot is not None:
            self.add_callback(ClassifierRowRestoreCallback(classifier_snapshot))

    @staticmethod
    def _select_classification_logits(logits, labels):
        """
        `preprocess_logits_for_metrics` hook (called once per eval batch,
        before predictions are accumulated). `logits` is whatever Trainer.
        prediction_step() extracted from compute_loss()'s `outputs` -- a
        single tensor normally, or a tuple/list when `outputs` had more than
        one non-loss field. Picked explicitly by shape (2-D, last dim ==
        NUM_CLASSES) rather than assuming position 0, since the ModelOutput
        field order (logits, hidden_states, attentions) is a transformers
        implementation detail, not a contract this code should rely on.
        """
        if not isinstance(logits, (tuple, list)):
            return logits
        candidates = [
            t for t in logits
            if torch.is_tensor(t) and t.dim() == 2 and t.shape[-1] == NUM_CLASSES
        ]
        if len(candidates) != 1:
            raise ValueError(
                "RankExtensionTrainer.preprocess_logits_for_metrics: expected exactly one "
                f"2-D (*, {NUM_CLASSES}) classification-logits tensor among {len(logits)} "
                f"prediction elements, found {len(candidates)}. Shapes: "
                f"{[tuple(t.shape) if torch.is_tensor(t) else type(t) for t in logits]}"
            )
        return candidates[0]


def assert_rank_extension_structure(model, step_idx):
    expected_total_rank, expected_frozen_rank, expected_new_rank = get_rank_extension_rank_triplet(step_idx)
    expected_target_names = getattr(model, "_rank_extension_target_names", None)
    if expected_target_names is None:
        raise AssertionError("Missing _rank_extension_target_names on rank-extension model.")

    module_map = dict(model.named_modules())
    wrapped_names = [name for name, module in module_map.items() if isinstance(module, GrowingRankLoRALinear)]
    if sorted(wrapped_names) != sorted(expected_target_names):
        missing = sorted(set(expected_target_names) - set(wrapped_names))
        extra = sorted(set(wrapped_names) - set(expected_target_names))
        raise AssertionError(f"Rank-extension wrapper mismatch. missing={missing}, extra={extra}")

    for name in expected_target_names:
        module = module_map[name]
        if module.total_rank != expected_total_rank:
            raise AssertionError(f"{name} total_rank={module.total_rank}, expected={expected_total_rank}")
        if module.frozen_rank != expected_frozen_rank:
            raise AssertionError(f"{name} frozen_rank={module.frozen_rank}, expected={expected_frozen_rank}")
        if module.new_rank != expected_new_rank:
            raise AssertionError(f"{name} new_rank={module.new_rank}, expected={expected_new_rank}")
        if module.A_frozen is not None and module.A_frozen.requires_grad:
            raise AssertionError(f"{name}.A_frozen unexpectedly requires grad.")
        if module.B_frozen is not None and module.B_frozen.requires_grad:
            raise AssertionError(f"{name}.B_frozen unexpectedly requires grad.")
        if module.A_new is None or not module.A_new.requires_grad:
            raise AssertionError(f"{name}.A_new missing/frozen.")
        if module.B_new is None or not module.B_new.requires_grad:
            raise AssertionError(f"{name}.B_new missing/frozen.")

    trainable_lora_names = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (".A_" in name or ".B_" in name)
    ]
    bad_lora_names = [
        name for name in trainable_lora_names
        if not (name.endswith(".A_new") or name.endswith(".B_new"))
    ]
    if bad_lora_names:
        raise AssertionError(f"Only A_new/B_new may be trainable LoRA params, got {bad_lora_names}")

    print(
        f"[rank_extension assertions] step={step_idx + 1} | "
        f"total_rank={expected_total_rank} | frozen_rank={expected_frozen_rank} | new_rank={expected_new_rank}"
    )


def snapshot_frozen_rank_blocks(model):
    snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, GrowingRankLoRALinear) and module.frozen_rank > 0:
            snapshot[name] = {
                "A": module.A_frozen.detach().cpu().clone(),
                "B": module.B_frozen.detach().cpu().clone(),
            }
    return snapshot


def check_frozen_rank_blocks_unchanged(model, snapshot, label, csv_path=None):
    if len(snapshot) == 0:
        print(f"[rank_extension diagnostics] {label}: no frozen rank blocks to compare")
        return pd.DataFrame()

    rows = []
    module_map = dict(model.named_modules())
    for name, before in snapshot.items():
        module = module_map[name]
        rows.append({
            "layer": name,
            "A_max_abs_diff": float((module.A_frozen.detach().cpu() - before["A"]).abs().max().item()),
            "B_max_abs_diff": float((module.B_frozen.detach().cpu() - before["B"]).abs().max().item()),
        })

    diag_df = pd.DataFrame(rows)
    max_a = float(diag_df["A_max_abs_diff"].max())
    max_b = float(diag_df["B_max_abs_diff"].max())
    print(f"[rank_extension diagnostics] {label}: max frozen A diff={max_a:.10f}, max frozen B diff={max_b:.10f}")

    if csv_path is not None:
        diag_df.to_csv(csv_path, index=False)
        print("[rank_extension diagnostics] saved frozen-block diagnostics:", csv_path)

    return diag_df


def save_rank_extension_structure_csv(model, method_name, step_idx, csv_path):
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, GrowingRankLoRALinear):
            continue
        rows.append({
            "method": method_name,
            "step": int(step_idx + 1),
            "layer": name,
            "total_rank": int(module.total_rank),
            "frozen_rank": int(module.frozen_rank),
            "new_rank": int(module.new_rank),
            "old_active_in_forward": bool(module.old_active_in_forward),
            "rankext_alpha": float(module.rankext_alpha),
            "scaling": float(module.scaling),
            "has_A_frozen": bool(module.A_frozen is not None),
            "has_B_frozen": bool(module.B_frozen is not None),
            "has_A_new": bool(module.A_new is not None),
            "has_B_new": bool(module.B_new is not None),
        })

    structure_df = pd.DataFrame(rows)
    structure_df.to_csv(csv_path, index=False)
    print("[rank_extension diagnostics] saved rank-structure diagnostics:", csv_path)
    return structure_df


def save_trainable_parameters_csv(model, method_name, step_idx, csv_path):
    rows = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".A_new") or name.endswith(".B_new"):
            param_group = "lora_new"
        elif name.startswith("classifier."):
            param_group = "classifier"
        else:
            param_group = "other"
        rows.append({
            "method": method_name,
            "step": int(step_idx + 1),
            "parameter": name,
            "group": param_group,
            "shape": list(param.shape),
            "numel": int(param.numel()),
        })

    trainable_df = pd.DataFrame(rows)
    trainable_df.to_csv(csv_path, index=False)
    print("[rank_extension diagnostics] saved trainable-parameter diagnostics:", csv_path)
    return trainable_df


def compute_delta_orth_components(model, eps=1e-12):
    trace_terms = []
    abs_trace_terms = []
    norm_terms = []
    old_norm_terms = []
    new_norm_terms = []
    factor_a_terms = []
    factor_b_terms = []
    factor_total_terms = []
    a_overlap_terms = []
    b_overlap_terms = []
    rows = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for name, module in model.named_modules():
        if not isinstance(module, GrowingRankLoRALinear):
            continue
        if module.new_rank <= 0:
            continue

        old_delta = module.cumulative_old_delta()
        new_delta = module.current_new_delta()
        if new_delta is None:
            continue
        if old_delta is None:
            old_delta = torch.zeros_like(new_delta)

        trace_overlap = torch.sum(old_delta * new_delta)
        orth_penalty = torch.abs(trace_overlap)
        old_sq = old_delta.norm(p="fro").pow(2)
        new_sq = new_delta.norm(p="fro").pow(2)
        denom = old_sq * new_sq + float(eps)
        norm_sq = trace_overlap.pow(2) / denom

        trace_terms.append(trace_overlap)
        abs_trace_terms.append(orth_penalty)
        norm_terms.append(norm_sq)
        old_norm_terms.append(old_delta.norm(p="fro"))
        new_norm_terms.append(new_delta.norm(p="fro"))

        if module.frozen_rank > 0 and module.A_frozen is not None and module.B_frozen is not None:
            A_old = module.A_frozen.to(device=new_delta.device, dtype=new_delta.dtype)
            A_new = module.A_new
            B_old = module.B_frozen.to(device=new_delta.device, dtype=new_delta.dtype)
            B_new = module.B_new

            A_old_hat = A_old / A_old.norm(dim=1, keepdim=True).clamp_min(eps)
            A_new_hat = A_new / A_new.norm(dim=1, keepdim=True).clamp_min(eps)
            A_overlap = A_old_hat @ A_new_hat.T
            factor_a = torch.sum(A_overlap.pow(2))

            B_old_hat = B_old / B_old.norm(dim=0, keepdim=True).clamp_min(eps)
            B_new_hat = B_new / B_new.norm(dim=0, keepdim=True).clamp_min(eps)
            B_overlap = B_old_hat.T @ B_new_hat
            factor_b = torch.sum(B_overlap.pow(2))

            factor_total = factor_a + factor_b
            factor_a_terms.append(factor_a)
            factor_b_terms.append(factor_b)
            factor_total_terms.append(factor_total)
            a_overlap_terms.append(A_overlap.abs().mean())
            b_overlap_terms.append(B_overlap.abs().mean())
        rows.append({
            "layer": name,
            "inner_trace": float(trace_overlap.detach().cpu().item()),
            "abs_inner_trace": float(orth_penalty.detach().cpu().item()),
            "old_norm": float(old_norm_terms[-1].detach().cpu().item()),
            "new_norm": float(new_norm_terms[-1].detach().cpu().item()),
            "norm_sq": float(norm_sq.detach().cpu().item()),
        })

    if len(trace_terms) == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return {
            "num_layers": 0,
            "trace_mean": zero,
            "trace_abs_mean": zero,
            "norm_sq_mean": zero,
            "old_norm_mean": zero,
            "new_norm_mean": zero,
            "factor_A_mean": zero,
            "factor_B_mean": zero,
            "factor_total_mean": zero,
            "mean_A_overlap": zero,
            "mean_B_overlap": zero,
            "diag_df": pd.DataFrame(),
        }

    return {
        "num_layers": int(len(trace_terms)),
        "trace_mean": torch.stack(trace_terms).mean(),
        "trace_abs_mean": torch.stack(abs_trace_terms).mean(),
        "norm_sq_mean": torch.stack(norm_terms).mean(),
        "old_norm_mean": torch.stack(old_norm_terms).mean(),
        "new_norm_mean": torch.stack(new_norm_terms).mean(),
        "factor_A_mean": (torch.stack(factor_a_terms).mean() if len(factor_a_terms) > 0 else torch.tensor(0.0, device=device, dtype=dtype)),
        "factor_B_mean": (torch.stack(factor_b_terms).mean() if len(factor_b_terms) > 0 else torch.tensor(0.0, device=device, dtype=dtype)),
        "factor_total_mean": (torch.stack(factor_total_terms).mean() if len(factor_total_terms) > 0 else torch.tensor(0.0, device=device, dtype=dtype)),
        "mean_A_overlap": (torch.stack(a_overlap_terms).mean() if len(a_overlap_terms) > 0 else torch.tensor(0.0, device=device, dtype=dtype)),
        "mean_B_overlap": (torch.stack(b_overlap_terms).mean() if len(b_overlap_terms) > 0 else torch.tensor(0.0, device=device, dtype=dtype)),
        "diag_df": pd.DataFrame(rows),
    }


def compute_rankext_dense_orth_components(model, step_idx, eps=1e-12):
    """JOINT PACK ADDITION (RankExt Arms RE-2/RE-3) -- DenseOrth for RankExt,
    same audited formula as SimpleAvg's compute_independent_lora_dense_orth_
    components(): for the CURRENT step's own new block delta (LIVE,
    gradient-carrying, module.current_new_delta() == scaling*(B_new@A_new)),
    compare against EACH PREVIOUS STEP's OWN INDIVIDUAL block delta (never
    the cumulative combined old delta compute_delta_orth_components() above
    uses, and never previous steps' raw A/B factors):

        c_i = <current_new_delta, block_i_delta>_F /
              (||current_new_delta||_F ||block_i_delta||_F + eps)
        module_penalty = mean_i(c_i^2)

    Unlike SimpleAvg (independent specialists -- needs a persisted per-step
    delta list), RankExt's own frozen A_frozen/B_frozen tensor at step_idx
    ALREADY contains every previous step's block concatenated in step order
    (GrowingRankLoRALinear.full_A_B()/build_rank_extension_model() build it
    that way), and the rank schedule (get_rank_extension_rank_schedule(),
    fixed and known) gives the exact row/column boundaries of each original
    step's own contribution within that concatenation. So each previous
    step's OWN block delta is recovered by slicing A_frozen/B_frozen at
    those boundaries -- no extra state needs to be saved or threaded through
    the training loop. Detached (no grad) for every previous block; only the
    current new block carries gradients. Inactive at step 1 (frozen_rank==0,
    no previous blocks to compare against) -- same "no penalty yet" convention
    KD already uses (teacher_model is None at step 1). Aggregated across
    modules by the mean, same convention as compute_delta_orth_components()/
    compute_independent_lora_factor_orth_components() above.
    """
    device = next(model.parameters()).device
    zero = torch.tensor(0.0, device=device)
    schedule = get_rank_extension_rank_schedule()
    # Boundaries of each ORIGINAL step's own block within any concatenated
    # (frozen+new) A/B tensor: step 0 occupies rows/cols [0, schedule[0]),
    # step 1 occupies [schedule[0], schedule[1]), etc.
    boundaries = [(0 if j == 0 else schedule[j - 1], schedule[j]) for j in range(step_idx)]

    if step_idx <= 0 or len(boundaries) == 0:
        return {"num_layers": 0, "dense_orth_mean": zero}

    per_module_means = []
    for name, module in model.named_modules():
        if not isinstance(module, GrowingRankLoRALinear):
            continue
        current_delta = module.current_new_delta()
        if current_delta is None:
            continue
        if module.A_frozen is None or module.B_frozen is None:
            continue

        A_frozen = module.A_frozen.to(device=current_delta.device, dtype=current_delta.dtype)
        B_frozen = module.B_frozen.to(device=current_delta.device, dtype=current_delta.dtype)
        current_norm = torch.linalg.norm(current_delta)

        cos_sq_terms = []
        for (start, end) in boundaries:
            A_block = A_frozen[start:end, :].detach()
            B_block = B_frozen[:, start:end].detach()
            block_delta = (B_block @ A_block) * float(module.scaling)
            inner = torch.sum(current_delta * block_delta)
            block_norm = torch.linalg.norm(block_delta)
            cos = inner / (current_norm * block_norm + eps)
            cos_sq_terms.append(cos * cos)

        if len(cos_sq_terms) > 0:
            per_module_means.append(torch.stack(cos_sq_terms).mean())

    if len(per_module_means) == 0:
        return {"num_layers": 0, "dense_orth_mean": zero}

    return {
        "num_layers": int(len(per_module_means)),
        "dense_orth_mean": torch.stack(per_module_means).mean(),
    }


def cumulative_orth_formula_label(step_idx):
    t = int(step_idx + 1)
    if t <= 1:
        return "orth disabled at step 1 (no previous delta)"
    old_terms = " + ".join([f"L{i}" for i in range(1, t)])
    return f"orth({old_terms}, L{t})"


def collect_rank_block_grad_norms(model, train_ds):
    if len(train_ds) == 0:
        return {
            "frozen_A_grad_norm_mean": 0.0,
            "frozen_B_grad_norm_mean": 0.0,
            "new_A_grad_norm_mean": 0.0,
            "new_B_grad_norm_mean": 0.0,
        }

    device = next(model.parameters()).device
    model.train()
    model.zero_grad(set_to_none=True)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    out = model(**batch)
    out.loss.backward()

    frozen_a = []
    frozen_b = []
    new_a = []
    new_b = []

    for _, module in model.named_modules():
        if not isinstance(module, GrowingRankLoRALinear):
            continue
        if module.A_frozen is not None:
            g = module.A_frozen.grad
            frozen_a.append(0.0 if g is None else float(g.norm().detach().cpu().item()))
        if module.B_frozen is not None:
            g = module.B_frozen.grad
            frozen_b.append(0.0 if g is None else float(g.norm().detach().cpu().item()))
        if module.A_new is not None:
            g = module.A_new.grad
            new_a.append(0.0 if g is None else float(g.norm().detach().cpu().item()))
        if module.B_new is not None:
            g = module.B_new.grad
            new_b.append(0.0 if g is None else float(g.norm().detach().cpu().item()))

    model.zero_grad(set_to_none=True)

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else 0.0

    return {
        "frozen_A_grad_norm_mean": _mean(frozen_a),
        "frozen_B_grad_norm_mean": _mean(frozen_b),
        "new_A_grad_norm_mean": _mean(new_a),
        "new_B_grad_norm_mean": _mean(new_b),
    }


# OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION (R6 follow-up,
# 2026-08-21): see RANKEXT_PROJECTED_PROTECT_METHODS above for the mechanism
# writeup. The two helpers below are used only by DeltaOrthRankExtensionTrainer
# below, gated to KD RankExt methods via the trainer's protect_weight kwarg.

def resolve_post_layernorm(vision_wrapper_model):
    """Locate the CLIPVisionTransformer's post_layernorm module starting from
    a CLIPVisionForCIFAR100-wrapped model's OWN `.vision_model` attribute
    (i.e. pass `some_model.vision_model`, not `some_model`).
    `CLIPVisionForCIFAR100.vision_model` is a `CLIPVisionModel`, and
    `CLIPVisionModel` itself has an internal attribute ALSO named
    `vision_model` (the actual `CLIPVisionTransformer`) -- real HF nesting,
    not a typo -- so `post_layernorm` can sit at either
    `vision_wrapper_model.post_layernorm` (if a bare CLIPVisionTransformer/
    CLIPVisionModel-with-flattened-attrs was passed) or one level deeper at
    `vision_wrapper_model.vision_model.post_layernorm` (the actual case for
    this file's models). Resolved defensively (checked both ways) rather than
    hardcoding the double-`.vision_model` path blindly, since it is easy to
    get this one wrong. Applying the returned module to a CLS token pulled
    from an ALREADY-COMPUTED hidden_states[-1] (same forward pass, same
    dropout draw) reproduces `.pooler_output` exactly (verified empirically:
    bit-identical to the CLIPVisionModel-returned pooler_output on the same
    input) -- this lets the projected-feature-consolidation loss below read
    the STUDENT's pooler_output without a second, independently-stochastic
    forward pass through the backbone."""
    if hasattr(vision_wrapper_model, "post_layernorm"):
        return vision_wrapper_model.post_layernorm
    if hasattr(vision_wrapper_model, "vision_model") and hasattr(vision_wrapper_model.vision_model, "post_layernorm"):
        return vision_wrapper_model.vision_model.post_layernorm
    raise AttributeError(
        "resolve_post_layernorm: could not locate post_layernorm on the given "
        "vision backbone at either nesting depth -- CLIP model structure may "
        "have changed; refusing to guess."
    )


def compute_old_semantic_subspace(teacher_model, old_class_ids, eps=1e-8):
    """Build an orthonormal basis P_old [hidden_size, k] of the CLASS-
    DISCRIMINATIVE span of the frozen previous-step teacher's classifier rows
    for `old_class_ids` (Part 1 of the R6 projected-feature-consolidation
    design):
      1. W_old = teacher.classifier.weight[old_class_ids]  (frozen, detached)
      2. L2-normalize each row individually -> W_hat.
      3. Remove the rows' common mean direction -> W_tilde (protects
         class-DISCRIMINATIVE directions, not whatever shared/common
         component the rows happen to share).
      4. SVD of W_tilde; P_old's columns are the right singular vectors
         (row-space basis) whose singular values clear the SAME default
         numerical-rank tolerance torch.linalg.matrix_rank uses
         (S.max() * max(rows, cols) * finfo(dtype).eps) -- k is therefore
         the ACTUAL supported rank of this specific (mean-removed) row
         matrix, not assumed to equal len(old_class_ids) (mean-removal alone
         guarantees rank <= len(old_class_ids) - 1).

    Returns None if old_class_ids is empty (step 1: no old subspace exists
    yet -- caller must treat this as "protection inactive", never as an
    error) or if the resulting rank is 0. Otherwise returns a CPU float32
    tensor, fully detached (no grad history whatsoever: computed entirely
    inside torch.no_grad() from an already-detached, requires_grad=False
    teacher weight). Computed ONCE by the caller (Trainer.__init__, before
    any batch is seen) -- this function does no caching itself, callers must
    not invoke it per-batch.
    """
    old_class_ids = sorted(int(c) for c in old_class_ids)
    if len(old_class_ids) == 0:
        return None
    with torch.no_grad():
        idx = torch.tensor(old_class_ids, dtype=torch.long)
        W = teacher_model.classifier.weight.detach().to("cpu", dtype=torch.float32)[idx]  # [C_old, H]
        W_hat = W / W.norm(dim=1, keepdim=True).clamp_min(eps)                             # L2-normalize rows
        W_tilde = W_hat - W_hat.mean(dim=0, keepdim=True)                                   # remove common mean
        _, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)                           # Vh: [r, H]
        if S.numel() == 0:
            return None
        tol = float(S.max().item()) * max(W_tilde.shape) * torch.finfo(W_tilde.dtype).eps
        rank_k = int((S > tol).sum().item())
        if rank_k == 0:
            return None
        P_old = Vh[:rank_k].t().contiguous().detach()  # [H, k], orthonormal columns
    return P_old


class DeltaOrthRankExtensionTrainer(RankExtensionTrainer):
    def __init__(
        self,
        *args,
        lambda_orth=0.0,
        orth_mode="abs_trace",
        orth_eps=1e-12,
        log_every_steps=1,
        method_name="unknown",
        step_idx=-1,
        teacher_model=None,
        kd_weight=0.0,
        kd_temperature=2.0,
        kd_class_scope="full",
        kd_old_class_ids=None,
        kd_warmup_epochs=0.0,
        dense_orth_warmup_epochs=DENSE_ORTH_WARMUP_EPOCHS,
        pretrained_anchor_weight=0.0,
        protect_weight=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_orth = float(lambda_orth)
        self.orth_mode = "none" if orth_mode is None else str(orth_mode)
        self.orth_eps = float(orth_eps)
        self.log_every_steps = max(1, int(log_every_steps))
        self.method_name = str(method_name)
        self.step_idx = int(step_idx)
        self.teacher_model = teacher_model
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
        # JOINT PACK ADDITION (RankExt Arms RE-1/RE-3) -- old-seen-only KD,
        # same mechanism/naming as SimpleAvg's IndependentLoraOrthTrainer.
        # `kd_old_class_ids` here is DISTINCT from self.old_class_ids below
        # (the pre-existing, protect_weight-specific attribute) -- different
        # mechanism, different purpose, deliberately not reused/renamed.
        self.kd_class_scope = str(kd_class_scope)
        self.kd_old_class_ids = kd_old_class_ids
        self.kd_warmup_epochs = float(kd_warmup_epochs)
        # JOINT PACK ADDITION (RankExt Arms RE-2/RE-3) -- DenseOrth's own
        # unconditional lambda warmup, independent of RANKEXT_ORTH_LAMBDA_
        # WARMUP_* below (which stays scoped to the pre-existing orth_mode
        # values only).
        self.dense_orth_warmup_epochs = float(dense_orth_warmup_epochs)
        # RANK_EXT FIRST_STEP FIX (task 2 decision doc, 2026-08-17 -- see
        # RANKEXT_PRETRAINED_ANCHOR_WEIGHT above for the diagnostic evidence
        # and mechanism writeup): shares the same teacher_model slot as KD
        # (both need a frozen reference model run on the current batch), but
        # is a genuinely separate loss term (CLS-hidden-state cosine distance,
        # not logit KL-divergence). Unconditionally mutually exclusive with
        # kd_weight>0 at the one call site that drives every rank_extension
        # method (run_rank_extension_variant() never sets both nonzero for
        # the same method) but the compute_loss() gate below does not itself
        # assume that -- both could in principle be active together.
        self.pretrained_anchor_weight = float(pretrained_anchor_weight)
        self._rows = []
        self._teacher_ready = False

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        # OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION (R6
        # follow-up, 2026-08-21): see RANKEXT_PROJECTED_PROTECT_METHODS above
        # for the mechanism writeup. protect_weight is 0.0 for every method
        # except the two KD RankExt methods (set by run_rank_extension_variant()
        # below) -- self.P_old stays None whenever protect_weight is 0.0, no
        # matter what teacher_model/step_idx are, so this block is a true
        # no-op (zero extra compute, zero extra state) for every other
        # method, INCLUDING the non-KD rank_extension / rank_extension_orth_
        # factor_lam_50 methods (whose teacher_model here is the pretrained-
        # anchor model, not a real classifier checkpoint -- deliberately never
        # touched). Computed ONCE per step, here in __init__, from the frozen
        # teacher's classifier rows for classes seen strictly before this
        # step (classes_for_step(0..step_idx-1)) -- empty at step_idx==0
        # (step 1), so P_old is None and protection is exactly zero there, no
        # separate step-1 special case needed anywhere else. self.P_old is
        # moved to the training device lazily in compute_loss() (mirrors
        # self._teacher_ready below -- the teacher itself is not yet on
        # device at this point in __init__ either), never recomputed there.
        self.protect_weight = float(protect_weight)
        self.old_class_ids = (
            sorted(int(c) for s in range(self.step_idx) for c in classes_for_step(s))
            if (self.protect_weight > 0.0 and self.teacher_model is not None and self.step_idx > 0)
            else []
        )
        self.P_old = (
            compute_old_semantic_subspace(self.teacher_model, self.old_class_ids)
            if (self.protect_weight > 0.0 and self.teacher_model is not None and len(self.old_class_ids) > 0)
            else None
        )
        self.protect_k = int(self.P_old.shape[1]) if self.P_old is not None else 0
        self._protect_ready = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # RANK_EXT FIRST_STEP FIX: only request hidden_states (mild extra
        # memory/compute in the encoder) when this trainer instance actually
        # needs them -- False (unchanged forward call) for every KD method
        # and for simple_avg (different trainer entirely), so only the
        # non-KD rank_extension methods that get the pretrained-backbone
        # anchor pay this cost. CRASH FIX: also gated on model.training --
        # Trainer.prediction_step() (every eval/evaluate() call) invokes this
        # SAME compute_loss() under torch.no_grad() with model.eval() already
        # applied, so hidden_states is never requested on the eval forward
        # either (features are a training-loss-only concept here; see
        # RankExtensionTrainer.preprocess_logits_for_metrics() /
        # compute_metrics() for the defensive backstop that still applies if
        # that ever changes). OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE
        # CONSOLIDATION (R6 follow-up, 2026-08-21): also requested whenever
        # protect_weight > 0.0 (KD RankExt methods, steps 2-5 only -- see
        # __init__ above) -- reuses this SAME hidden_states[-1] CLS token
        # (post_layernorm'd below) for the student side of the protection
        # loss instead of a second, independently-stochastic forward pass.
        # self.protect_weight is 0.0 for every other method/step, so this
        # `or` is a true no-op there -- need_hidden's value is bit-for-bit
        # unchanged from before for non-KD methods and for KD methods at
        # step 1.
        need_hidden = (self.pretrained_anchor_weight > 0.0 or self.protect_weight > 0.0) and model.training
        outputs = model(**inputs, output_hidden_states=need_hidden)
        ce_loss = outputs.loss

        zero = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        comps = {
            "num_layers": 0,
            "trace_mean": zero,
            "trace_abs_mean": zero,
            "norm_sq_mean": zero,
            "old_norm_mean": zero,
            "new_norm_mean": zero,
            "factor_A_mean": zero,
            "factor_B_mean": zero,
            "factor_total_mean": zero,
            "mean_A_overlap": zero,
            "mean_B_overlap": zero,
        }
        raw_inner = zero
        abs_inner = zero
        norm_sq = zero
        orth_loss_used = zero
        # JOINT PACK ADDITION (RankExt Arms RE-2/RE-3).
        dense_orth_comps = {"num_layers": 0, "dense_orth_mean": zero}

        if self.orth_mode != "none" and self.orth_mode != "dense_orth":
            comps = compute_delta_orth_components(model=model, eps=self.orth_eps)
            raw_inner = comps["trace_mean"]
            abs_inner = comps["trace_abs_mean"]
            norm_sq = comps["norm_sq_mean"]
        elif self.orth_mode == "dense_orth":
            dense_orth_comps = compute_rankext_dense_orth_components(
                model=model, step_idx=self.step_idx, eps=self.orth_eps,
            )

        if self.orth_mode == "none":
            orth_loss_used = zero
        elif self.orth_mode in ["trace", "trace_abs", "delta_trace", "abs_trace"]:
            orth_loss_used = abs_inner
        elif self.orth_mode == "norm":
            orth_loss_used = norm_sq
        elif self.orth_mode == "factor_orth":
            orth_loss_used = comps["factor_total_mean"]
        elif self.orth_mode == "dense_orth":
            # JOINT PACK ADDITION (RankExt Arms RE-2/RE-3).
            orth_loss_used = dense_orth_comps["dense_orth_mean"]
        else:
            raise ValueError(f"Unknown orth_mode={self.orth_mode}")

        # Objective 1b: RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED ramps lambda_orth up
        # over the first RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS epochs of THIS CL
        # step's local training (self.state.epoch resets to ~0 at the start of
        # every step's own Trainer -- see build_rank_extension_model()/
        # run_rank_extension_variant(), a fresh model+Trainer per step_idx).
        # No-op (multiplier stays 1.0) when the flag is off -- see the
        # justification comment next to the flag definition above.
        epoch_val_for_warmup = float(self.state.epoch) if self.state.epoch is not None else np.nan
        orth_warmup_multiplier = orth_lambda_warmup_multiplier(
            epoch_val_for_warmup, RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS, RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED,
        )
        effective_lambda_orth = float(self.lambda_orth) * orth_warmup_multiplier

        weighted = effective_lambda_orth * orth_loss_used
        kd_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        teacher_active = self.teacher_model is not None and self.kd_weight > 0.0

        # RANK_EXT FIRST_STEP FIX: separate gate, separate loss term. Compares
        # the CLS token of the LAST hidden state (pre-classifier, pre-final-
        # layernorm-pooling) between the current model and a frozen snapshot of
        # the PRETRAINED CLIP backbone (no LoRA contribution), both run on the
        # SAME current-step batch already being trained on (rehearsal-free --
        # no old images). Cosine distance, not logit KL-divergence -- no
        # softmax, no classifier, no temperature; a genuinely different
        # mechanism from the KD block above, not a re-skin of it.
        pretrained_anchor_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        # CRASH FIX: mirrors need_hidden's model.training gate above. Without
        # this, pretrained_anchor_active could be True on the eval forward
        # while outputs.hidden_states is None (need_hidden is now False
        # whenever model.training is False), and the
        # outputs.hidden_states[-1] access below would raise. Eval-time loss
        # for the anchored (non-KD) rank_extension methods is therefore
        # CE(+orth) only, same as every other rank_extension variant.
        pretrained_anchor_active = self.teacher_model is not None and self.pretrained_anchor_weight > 0.0 and model.training

        # OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION (R6
        # follow-up, 2026-08-21): protect_active is defined here (ahead of
        # its own diagnostic-block below) purely so the teacher_active branch
        # immediately below can decide, in ONE place, whether this same
        # forward call also needs hidden_states -- see the merged-forward
        # comment there. protect_loss/diagnostics themselves are still
        # computed in the protect_active block further down; nothing here
        # changes what that block computes, only what teacher call it reuses.
        protect_active = self.protect_weight > 0.0 and self.P_old is not None and model.training

        if teacher_active or protect_active:
            if not self._teacher_ready:
                self.teacher_model.to(device=ce_loss.device)
                self.teacher_model.eval()
                self._teacher_ready = True
            # SHARED TEACHER FORWARD (R6 follow-up, 2026-08-21): protect_active
            # is only ever True together with teacher_active for the two KD
            # RankExt methods this mechanism targets (protect_weight and
            # kd_weight are both gated on the identical `use_kd and
            # teacher_model is not None` condition in run_rank_extension_
            # variant() below), so ONE forward through self.teacher_model
            # serves both existing KD (logits) and the projected-protection
            # loss (pooled CLS features) -- output_hidden_states is requested
            # on this SAME call, rather than issuing a second, independent
            # teacher forward. output_hidden_states does not change the
            # computed logits at all (same forward graph, same numbers, only
            # what gets additionally collected/returned) -- teacher_logits/
            # kd_loss below are therefore numerically IDENTICAL to before this
            # refactor. Written defensively with `teacher_active or
            # protect_active` (rather than assuming they always co-occur) so
            # this stays correct even if a future config sets kd_weight=0
            # while protect_weight>0 for some method.
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs, output_hidden_states=protect_active)
                if protect_active:
                    teacher_post_ln = resolve_post_layernorm(self.teacher_model.vision_model)
                    teacher_pooled_for_protect = teacher_post_ln(teacher_outputs.hidden_states[-1][:, 0, :])
                if teacher_active:
                    teacher_logits = teacher_outputs.logits.detach()

            if teacher_active:
                # JOINT PACK ADDITION (RankExt Arms RE-1/RE-3) -- old-seen-only
                # KD, identical mechanism to SimpleAvg's masked_kd_loss() (same
                # function, reused unchanged). Teacher is STILL the frozen
                # previous-step RankExt model built above (build_rank_extension_
                # model(previous_rank_state, step_idx-1, ...)) -- unchanged from
                # historical full-100-way KD; only which logits enter the
                # softmax differs.
                if self.kd_class_scope == "old_seen":
                    kd_loss = masked_kd_loss(
                        outputs.logits, teacher_logits, self.kd_old_class_ids, self.kd_temperature,
                    )
                else:
                    student_log_probs = F.log_softmax(outputs.logits / self.kd_temperature, dim=-1)
                    teacher_probs = F.softmax(teacher_logits / self.kd_temperature, dim=-1)
                    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.kd_temperature ** 2)

        # JOINT PACK ADDITION (RankExt Arms RE-1/RE-3) -- KD-weight linear
        # warmup, same mechanism/convention as SimpleAvg's IndependentLoraOrth
        # Trainer. Always 1.0 (no-op) when kd_warmup_epochs<=0.
        kd_warmup_multiplier = orth_lambda_warmup_multiplier(
            epoch_val_for_warmup, self.kd_warmup_epochs, bool(self.kd_warmup_epochs > 0.0),
        )

        if pretrained_anchor_active:
            if not self._teacher_ready:
                self.teacher_model.to(device=ce_loss.device)
                self.teacher_model.eval()
                self._teacher_ready = True
            with torch.no_grad():
                teacher_hidden = self.teacher_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :]
            student_hidden = outputs.hidden_states[-1][:, 0, :]
            pretrained_anchor_loss = (1.0 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)).mean()

        # OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION (R6
        # follow-up, 2026-08-21): see RANKEXT_PROJECTED_PROTECT_METHODS above
        # for the mechanism writeup and __init__ above for P_old's
        # construction (frozen, computed once per step from the teacher's
        # OLD-class classifier rows only -- never the live student, never an
        # old image). protect_active itself was already computed above (ahead
        # of the shared teacher-forward block) -- not redefined here, just
        # reused. protect_active is False whenever P_old is None (step 1, or
        # protect_weight==0.0 for every non-KD-RankExt / non-selected
        # method), so this block never runs there -- non-selected methods are
        # bit-for-bit unaffected. eval-time gated off by model.training, same
        # reasoning as pretrained_anchor_active above.
        protect_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        protect_mean_cos = float("nan")
        protect_mean_full_drift_sq = float("nan")
        protect_mean_projected_drift_sq = float("nan")
        protect_mean_projected_drift_fraction = float("nan")

        if protect_active:
            if not self._protect_ready:
                self.P_old = self.P_old.to(device=ce_loss.device, dtype=torch.float32)
                self._protect_ready = True
            # Teacher: teacher_pooled_for_protect was already computed above,
            # inside the SAME no_grad forward that also produced teacher_logits
            # for the existing KD loss -- no separate teacher forward here.
            teacher_pooled = teacher_pooled_for_protect
            # Student: reuse the CLS token already computed by the ONE
            # `outputs = model(...)` call above (need_hidden guarantees
            # output_hidden_states=True whenever protect_active can be True)
            # and apply the model's own post_layernorm -- exactly reproduces
            # pooler_output (verified empirically bit-identical) WITHOUT a
            # second, independently-stochastic (dropout) forward pass through
            # the student backbone.
            post_ln = resolve_post_layernorm(model.vision_model)
            student_pooled = post_ln(outputs.hidden_states[-1][:, 0, :])

            z_s = F.normalize(student_pooled.to(torch.float32), p=2, dim=1, eps=1e-8)
            z_t = F.normalize(teacher_pooled.to(torch.float32), p=2, dim=1, eps=1e-8)
            delta_z = z_s - z_t                                    # [B, H], grad flows via z_s only
            proj = delta_z @ self.P_old                            # [B, k]
            # L_protect = (1/(B*k)) * ||Delta Z @ P_old||_F^2 == mean over all
            # B*k projected entries -- equivalent formulation, avoids an
            # explicit (and error-prone) manual B*k division.
            protect_loss = proj.pow(2).mean().to(dtype=ce_loss.dtype)

            with torch.no_grad():
                full_drift_sq = delta_z.pow(2).sum(dim=1)          # [B]
                proj_drift_sq = proj.pow(2).sum(dim=1)              # [B]
                protect_mean_cos = float((z_s * z_t).sum(dim=1).mean().item())
                protect_mean_full_drift_sq = float(full_drift_sq.mean().item())
                protect_mean_projected_drift_sq = float(proj_drift_sq.mean().item())
                protect_mean_projected_drift_fraction = float(
                    (proj_drift_sq / full_drift_sq.clamp_min(1e-8)).mean().item()
                )

        weighted_kd = float(self.kd_weight) * float(kd_warmup_multiplier) * kd_loss
        weighted_pretrained_anchor = float(self.pretrained_anchor_weight) * pretrained_anchor_loss
        weighted_protect = float(self.protect_weight) * protect_loss
        loss = ce_loss + weighted + weighted_kd + weighted_pretrained_anchor + weighted_protect

        ce_v = float(ce_loss.detach().cpu().item())
        raw_inner_v = float(raw_inner.detach().cpu().item())
        abs_inner_v = float(abs_inner.detach().cpu().item())
        norm_sq_v = float(norm_sq.detach().cpu().item())
        orth_used_v = float(orth_loss_used.detach().cpu().item())
        weighted_v = float(weighted.detach().cpu().item())
        kd_loss_v = float(kd_loss.detach().cpu().item())
        weighted_kd_v = float(weighted_kd.detach().cpu().item())
        protect_loss_v = float(protect_loss.detach().cpu().item())
        weighted_protect_v = float(weighted_protect.detach().cpu().item())
        total_loss_v = float(loss.detach().cpu().item())
        ratio_v = abs(weighted_v) / (ce_v + float(self.orth_eps))
        factor_a_v = float(comps["factor_A_mean"].detach().cpu().item())
        factor_b_v = float(comps["factor_B_mean"].detach().cpu().item())
        factor_total_v = float(comps["factor_total_mean"].detach().cpu().item())
        mean_a_overlap_v = float(comps["mean_A_overlap"].detach().cpu().item())
        mean_b_overlap_v = float(comps["mean_B_overlap"].detach().cpu().item())
        weighted_factor_v = float((effective_lambda_orth * comps["factor_total_mean"]).detach().cpu().item())
        weighted_factor_ratio_v = abs(weighted_factor_v) / (ce_v + float(self.orth_eps))
        row = {
            "method": self.method_name,
            "step": int(self.step_idx + 1),
            "epoch": epoch_val_for_warmup,
            "ce_loss": ce_v,
            "raw_inner": raw_inner_v,
            "abs_inner": abs_inner_v,
            "norm_sq_orth": norm_sq_v,
            "orth_loss_raw": raw_inner_v,
            "orth_loss": orth_used_v,
            "orth_loss_used": orth_used_v,
            "lambda_orth": float(self.lambda_orth),
            # Objective 1b transparency: nominal (configured) lambda_orth vs
            # the warmup-scaled value actually applied this batch. Equal only
            # when RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED is off, or once local_
            # epoch exceeds RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS (now on by default).
            "lambda_orth_warmup_multiplier": float(orth_warmup_multiplier),
            "lambda_orth_times_loss": weighted_v,
            "orth_ratio_abs_weighted_over_ce": ratio_v,
            "weighted_orth_over_CE": ratio_v,
            "orth_mode": self.orth_mode,
            "num_layers_used": int(comps["num_layers"]),
            "old_norm_mean": float(comps["old_norm_mean"].detach().cpu().item()),
            "new_norm_mean": float(comps["new_norm_mean"].detach().cpu().item()),
            "factor_A_penalty_mean": factor_a_v,
            "factor_B_penalty_mean": factor_b_v,
            "factor_total_penalty_mean": factor_total_v,
            "weighted_factor_orth_mean": weighted_factor_v,
            "weighted_factor_orth_over_CE": weighted_factor_ratio_v,
            "mean_A_overlap": mean_a_overlap_v,
            "mean_B_overlap": mean_b_overlap_v,
            "kd_loss": kd_loss_v,
            "weighted_kd_loss": weighted_kd_v,
            "kd_over_CE": weighted_kd_v / (ce_v + float(self.orth_eps)),
            "kd_weight": float(self.kd_weight),
            "kd_temperature": float(self.kd_temperature),
            "teacher_active": bool(teacher_active),
            # RANK_EXT FIRST_STEP FIX: separate, reportable columns -- stay
            # 0.0 / False for KD methods and simple_avg, since
            # pretrained_anchor_weight is 0.0 there.
            "pretrained_anchor_loss": float(pretrained_anchor_loss.detach().cpu().item()),
            "weighted_pretrained_anchor_loss": float(weighted_pretrained_anchor.detach().cpu().item()),
            "pretrained_anchor_over_CE": float(weighted_pretrained_anchor.detach().cpu().item()) / (ce_v + float(self.orth_eps)),
            "pretrained_anchor_weight": float(self.pretrained_anchor_weight),
            "pretrained_anchor_active": bool(pretrained_anchor_active),
            # OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION (R6
            # follow-up, 2026-08-21): stays 0.0/False/0/NaN for every method
            # except the two KD RankExt methods at steps 2-5 (protect_weight
            # is 0.0 everywhere else, and P_old is None at step 1 even for
            # those two methods -- see __init__ above).
            "protect_loss": protect_loss_v,
            "weighted_protect_loss": weighted_protect_v,
            "protect_over_CE": weighted_protect_v / (ce_v + float(self.orth_eps)),
            "protect_weight": float(self.protect_weight),
            "protect_active": bool(protect_active),
            "protect_k": int(self.protect_k),
            "protect_n_old_classes": int(len(self.old_class_ids)),
            "protect_mean_student_teacher_cos": protect_mean_cos,
            "protect_mean_full_drift_sq": protect_mean_full_drift_sq,
            "protect_mean_projected_drift_sq": protect_mean_projected_drift_sq,
            "protect_mean_projected_drift_fraction": protect_mean_projected_drift_fraction,
            "total_loss": total_loss_v,
            "effective_lambda": float(effective_lambda_orth),
            # FINAL 8-ARM SCRIPT ADDITION (Part 9, granular-logging-gap fix):
            # pure additive logging fields -- every value here is already
            # computed above from existing local variables; nothing here
            # changes `loss` (already returned unchanged, computed before
            # this dict). Mirrors the field names SimpleAvg's
            # IndependentLoraOrthTrainer already logs, so both families'
            # per-batch logs share a consistent schema.
            "kd_class_scope": str(self.kd_class_scope),
            "kd_warmup_epochs": float(self.kd_warmup_epochs),
            "kd_warmup_multiplier": float(kd_warmup_multiplier),
            "effective_kd_coefficient": float(self.kd_weight) * float(kd_warmup_multiplier),
            "dense_orth_mean": (
                float(dense_orth_comps["dense_orth_mean"].detach().cpu().item())
                if self.orth_mode == "dense_orth" else np.nan
            ),
            # RankExt's DenseOrth path (orth_mode=="dense_orth") uses the SAME
            # RANKEXT_ORTH_LAMBDA_WARMUP_* multiplier as every other orth_mode
            # here (orth_warmup_multiplier, already applied into
            # effective_lambda above) -- unlike SimpleAvg's
            # IndependentLoraOrthTrainer, this class's own
            # `dense_orth_warmup_epochs` constructor parameter is NOT
            # separately consulted anywhere in this compute_loss(). Logged
            # here as an honest alias of orth_warmup_multiplier (not a second,
            # independently-computed value) so this is verifiable rather than
            # assumed; NaN when this mode isn't active, matching the
            # NaN-for-inactive-mechanisms convention used throughout. (Not
            # exercised by any of this script's 8 active arms -- RankExt is
            # FactorOrth-only here -- but kept for schema parity / future use.)
            "dense_orth_warmup_multiplier": (
                float(orth_warmup_multiplier) if self.orth_mode == "dense_orth" else np.nan
            ),
            "effective_orth_coefficient": float(effective_lambda_orth),
            "orth_warmup_multiplier": float(orth_warmup_multiplier),
        }
        self._rows.append(row)

        if len(self._rows) % self.log_every_steps == 0:
            print(
                f"[orth train] method={self.method_name} | step={row['step']} | epoch={row['epoch']:.4f} | "
                f"ce={row['ce_loss']:.6f} | orth={row['orth_loss_used']:.6f} | "
                f"kd={row['kd_loss']:.6f} | anchor={row['pretrained_anchor_loss']:.6f} | "
                f"protect={row['protect_loss']:.6f} (k={row['protect_k']}, "
                f"n_old={row['protect_n_old_classes']}, r={row['protect_mean_projected_drift_fraction']:.4f}) | "
                f"total={row['total_loss']:.6f} | "
                f"lambda={row['lambda_orth']:.6g} (warmup x{row['lambda_orth_warmup_multiplier']:.3g}) | "
                f"kd_weight={row['kd_weight']:.6g} | anchor_weight={row['pretrained_anchor_weight']:.6g} | "
                f"protect_weight={row['protect_weight']:.6g} | "
                f"ratio={row['orth_ratio_abs_weighted_over_ce']:.6f}"
            )

        return (loss, outputs) if return_outputs else loss

    def consume_logged_losses(self):
        if len(self._rows) == 0:
            return None
        out = pd.DataFrame(self._rows).copy()
        self._rows = []
        return out


def evaluate_seen_step_accuracies(model, upto_step_idx):
    args = get_training_args(
        output_dir=os.path.join(MODELS_DIR, "tmp_seen_eval"),
        epochs=1,
        lr=LR_LORA,
        batch_size=BATCH_LORA,
        accum_steps=ACCUM_LORA,
        train_dataset_len=None,
        eval_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    step_acc = {}
    for task_step in range(upto_step_idx + 1):
        eval_ds = make_eval_dataset(classes_for_step(task_step))
        out = trainer.evaluate(eval_dataset=eval_ds)
        step_acc[int(task_step)] = float(out["eval_accuracy"])
    return step_acc


def compute_average_forgetting(stepwise_task_accuracies):
    if len(stepwise_task_accuracies) == 0:
        return np.nan
    all_steps = sorted(stepwise_task_accuracies.keys())
    final_step = all_steps[-1]
    forgetting_values = []
    for task_step in all_steps:
        if task_step == final_step:
            continue
        vals = []
        for s in all_steps:
            if s < task_step:
                continue
            if task_step in stepwise_task_accuracies[s]:
                vals.append(stepwise_task_accuracies[s][task_step])
        if len(vals) == 0:
            continue
        best_acc = max(vals)
        final_acc = stepwise_task_accuracies[final_step].get(task_step, np.nan)
        if np.isnan(final_acc):
            continue
        forgetting_values.append(float(best_acc - final_acc))
    if len(forgetting_values) == 0:
        return np.nan
    return float(np.mean(forgetting_values))


def safe_lambda_tag(val):
    v = float(val)
    if abs(v - 1e-4) < 1e-12:
        return "lam_1em4"
    if abs(v - 1e-3) < 1e-12:
        return "lam_1em3"
    if abs(v - 0.01) < 1e-12:
        return "lam_001"
    if abs(v - 0.05) < 1e-12:
        return "lam_005"
    if abs(v - 0.1) < 1e-12:
        return "lam_01"
    s = f"{v:g}".replace("-", "m").replace(".", "p")
    return f"lam_{s}"


def print_rank_extension_step_diagnostics(
    method_name,
    step_idx,
    model,
    replay_per_class,
    old_active_in_forward,
    frozen_diff_df,
    grad_stats,
    eps=1e-12,
):
    modules = [m for _, m in model.named_modules() if isinstance(m, GrowingRankLoRALinear)]
    if len(modules) == 0:
        return
    ref = modules[0]
    total_rank = int(ref.total_rank)
    frozen_rank = int(ref.frozen_rank)
    new_rank = int(ref.new_rank)
    active_new_slice = (
        f"{frozen_rank + 1}-{frozen_rank + new_rank}" if new_rank > 0 else "none"
    )
    frozen_slice = f"1-{frozen_rank}" if frozen_rank > 0 else "none"

    comps = compute_delta_orth_components(model=model, eps=eps)
    raw_trace = float(comps["trace_mean"].detach().cpu().item())
    norm_sq = float(comps["norm_sq_mean"].detach().cpu().item())
    old_norm = float(comps["old_norm_mean"].detach().cpu().item())
    new_norm = float(comps["new_norm_mean"].detach().cpu().item())

    max_a_diff = 0.0
    max_b_diff = 0.0
    if len(frozen_diff_df) > 0:
        max_a_diff = float(frozen_diff_df["A_max_abs_diff"].max())
        max_b_diff = float(frozen_diff_df["B_max_abs_diff"].max())

    print(f"[rank_extension diagnostics] method={method_name} step={step_idx + 1}")
    print(f"  total_lora_rank={total_rank}")
    print(f"  active_trainable_rank_slice={active_new_slice}")
    print(f"  frozen_copied_rank_slice={frozen_slice}")
    print(f"  replay_disabled={(int(replay_per_class) == 0)}")
    print(f"  old_slices_active_in_forward={bool(old_active_in_forward)}")
    print(f"  cumulative_old_delta_norm={old_norm:.8f}")
    print(f"  current_new_delta_norm={new_norm:.8f}")
    print(f"  raw_trace_inner_sum(old*new)={raw_trace:.8f}")
    print(f"  normalized_squared_orth={norm_sq:.8f}")
    print(f"  frozen_A_grad_norm_mean={grad_stats['frozen_A_grad_norm_mean']:.8f}")
    print(f"  frozen_B_grad_norm_mean={grad_stats['frozen_B_grad_norm_mean']:.8f}")
    print(f"  new_A_grad_norm_mean={grad_stats['new_A_grad_norm_mean']:.8f}")
    print(f"  new_B_grad_norm_mean={grad_stats['new_B_grad_norm_mean']:.8f}")
    print(f"  frozen_old_A_max_abs_diff={max_a_diff:.10f}")
    print(f"  frozen_old_B_max_abs_diff={max_b_diff:.10f}")
    print(f"  cumulative_orth_check={cumulative_orth_formula_label(step_idx)}")


def run_rank_extension_variant(
    method_name,
    replay_per_class=0,
    use_orth=False,
    orth_mode=None,
    lambda_orth=0.0,
    zero_old_merge=False,
    use_kd=False,
    kd_weight=0.0,
    kd_temperature=2.0,
    orth_eval_records=None,
    orth_train_records=None,
    orth_summary_records=None,
):
    """PRESERVED, UNCHANGED from Exp1 (not called by this script's own
    execution loop below, which uses the persistence-aware
    train_rank_extension_arm()/finalize_rank_extension_arm() pair instead) --
    kept verbatim so the ORIGINAL, non-resumable pipeline remains
    available/diff-able against Exp1. JOINT PACK: this function does NOT
    accept kd_class_scope/kd_warmup_epochs -- old-seen KD/KD-warmup are only
    reachable through train_rank_extension_arm() below."""
    # REPRODUCIBILITY-ONLY FIX (2026-08-25, not a scientific change): same
    # method-order RNG independence fix as run_simple_avg_variant() above --
    # see that function's comment for the full rationale. This is the single
    # top-level entry point for every rank_extension-family method (see
    # rank_extension_execution_order's call loop below, exactly one call per
    # active method), called here before this function builds any
    # model/adapter/optimizer/DataLoader (including the shared non-drifting
    # teacher built once before the step loop, a few lines down).
    set_seed(SEED)
    previous_rank_state = None
    stepwise_task_accuracies = {}
    # PRE-THESIS FIX 2: {step_idx: accuracy_fraction} zero-shot forward-transfer
    # probes -- accuracy on step_idx's OWN class group, measured with the model
    # as it stood right before step_idx's training began (i.e. after step_idx-1,
    # carrying forward everything learned so far but with an untrained new rank
    # slice for step_idx). Only rank_extension has a genuinely evolving model to
    # probe this way; see compute_forward_transfer()'s docstring.
    forward_transfer_probe = {}
    active_orth_mode = "none" if not use_orth else (None if orth_mode is None else str(orth_mode))
    active_lambda_orth = float(lambda_orth)
    active_kd_weight = float(kd_weight)
    active_kd_temperature = float(kd_temperature)
    # R6 follow-up: keep the existing new-block warmup for non-KD rank_extension
    # methods, but disable it for KD variants so KD remains the stabilizer.
    # None (not 0.0) when disabled, so train_with_trainer() skips attaching the
    # callback entirely rather than attaching a no-op one.
    active_new_block_warmup_epochs_value = method_rankext_new_block_warmup_epochs(
        method_name,
        "rank_extension",
    )
    active_new_block_warmup_epochs = (
        float(active_new_block_warmup_epochs_value)
        if active_new_block_warmup_epochs_value > 0.0
        else None
    )

    # RANK_EXT FIRST_STEP FIX (task 2 decision doc, 2026-08-17 -- see
    # RANKEXT_PRETRAINED_ANCHOR_WEIGHT above for the diagnostic evidence):
    # build ONE non-drifting teacher, ONCE, before the step loop, and reuse
    # the SAME instance for every step (including step 1, which has no
    # previous_rank_state to chain from). This is the FROZEN PRETRAINED CLIP
    # backbone with no LoRA contribution: calling
    # build_rank_extension_model(previous_rank_state=None, ...) gives a model
    # whose new LoRA block has B_new zero-initialized (see
    # GrowingRankLoRALinear.__init__ -- nn.init.zeros_(self.B_new)) and no
    # frozen block at all, so lora_new/lora_old are both exactly 0 in
    # forward() and the model's output is mathematically identical to the raw
    # pretrained backbone -- reusing this construction path (rather than
    # hand-stripping LoRA) keeps the teacher's forward pass on the exact same
    # code path as every student, with zero risk of an accidental behavioral
    # difference. Frozen (eval + requires_grad=False) and never trained, so it
    # cannot drift across steps by construction. Unconditional for every
    # non-KD rank_extension method -- no more per-method opt-in.
    pretrained_anchor_model = None
    if not use_kd:
        pretrained_anchor_model = build_rank_extension_model(
            previous_rank_state=None,
            step_idx=0,
            old_active_in_forward=True,
        )
        pretrained_anchor_model.eval()
        for p in pretrained_anchor_model.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in pretrained_anchor_model.parameters())

    for step_idx in range(NUM_STEPS):
        current_classes = classes_for_step(step_idx)
        trainable_classifier_classes = rank_extension_trainable_classifier_classes(
            step_idx=step_idx,
            replay_per_class=replay_per_class,
        )

        old_active_in_forward = not (zero_old_merge and step_idx > 0)
        teacher_model = None
        if pretrained_anchor_model is not None:
            # RANK_EXT FIRST_STEP FIX: non-drifting reference, active for
            # EVERY step including step 1 -- this is the whole point of the
            # fix (see the pretrained_anchor_model construction above).
            # Unlike the KD branch below, this never depends on
            # previous_rank_state, so step 1's pretrained_anchor_weight
            # (gated on `teacher_model is not None` at the trainer_kwargs
            # assembly below) is nonzero here.
            teacher_model = pretrained_anchor_model
        elif use_kd and previous_rank_state is not None:
            # KD teacher: "the model as it stood at the end of the previous
            # step" -- unreachable for non-KD methods (pretrained_anchor_model
            # is non-None for those, so the branch above always wins). None at
            # step 1 (previous_rank_state is None yet), same as before.
            teacher_old_active_in_forward = not (zero_old_merge and (step_idx - 1) > 0)
            teacher_model = build_rank_extension_model(
                previous_rank_state=previous_rank_state,
                step_idx=step_idx - 1,
                old_active_in_forward=teacher_old_active_in_forward,
            )
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            assert not any(p.requires_grad for p in teacher_model.parameters())

        model = build_rank_extension_model(
            previous_rank_state=previous_rank_state,
            step_idx=step_idx,
            old_active_in_forward=old_active_in_forward,
        )
        assert_rank_extension_structure(model=model, step_idx=step_idx)
        save_rank_extension_structure_csv(
            model=model,
            method_name=method_name,
            step_idx=step_idx,
            csv_path=os.path.join(
                TABLES_DIR,
                f"{method_name}_step_{step_idx + 1}_rank_structure.csv",
            ),
        )
        save_trainable_parameters_csv(
            model=model,
            method_name=method_name,
            step_idx=step_idx,
            csv_path=os.path.join(
                TABLES_DIR,
                f"{method_name}_step_{step_idx + 1}_trainable_parameters.csv",
            ),
        )

        # PRE-THESIS FIX 2: forward-transfer probe -- `model` at this point
        # carries forward all previously learned steps but has NOT yet trained
        # on step_idx's data (that happens below), so evaluating it now on
        # step_idx's own class group is a genuine zero-shot transfer measurement.
        # Step 0 has no prior model to transfer from, so it is skipped (matches
        # the standard FWT definition, which averages over tasks 2..T).
        if step_idx > 0:
            forward_transfer_probe[int(step_idx)] = evaluate_single_step_accuracy(model, step_idx)

        hooks = add_classifier_row_gradient_mask(
            model=model,
            trainable_classes=trainable_classifier_classes,
        )
        classifier_snapshot = snapshot_protected_classifier_rows(
            model=model,
            trainable_classes=trainable_classifier_classes,
        )
        frozen_snapshot = snapshot_frozen_rank_blocks(model)

        train_ds = make_train_dataset(step_idx=step_idx, replay_per_class=replay_per_class)
        eval_ds = make_val_dataset(current_classes)
        out_dir = os.path.join(MODELS_DIR, f"{method_name}_step_{step_idx + 1}")

        trainer_cls = DeltaOrthRankExtensionTrainer
        trainer_kwargs = {
            "classifier_snapshot": classifier_snapshot,
            "lambda_orth": active_lambda_orth,
            "orth_mode": active_orth_mode,
            "orth_eps": float(ORTH_NORM_EPS),
            "log_every_steps": int(ORTH_LOSS_LOG_EVERY),
            "method_name": method_name,
            "step_idx": int(step_idx),
            "teacher_model": teacher_model,
            "kd_weight": active_kd_weight if (use_kd and teacher_model is not None) else 0.0,
            "kd_temperature": active_kd_temperature,
            "pretrained_anchor_weight": RANKEXT_PRETRAINED_ANCHOR_WEIGHT if (not use_kd and teacher_model is not None) else 0.0,
            # OLD-CLASS SEMANTIC-SUBSPACE PROJECTED FEATURE CONSOLIDATION (R6
            # follow-up, 2026-08-21): nonzero ONLY for the two methods in
            # RANKEXT_PROJECTED_PROTECT_METHODS, and only when this is a real
            # KD step with a real previous-step teacher checkpoint (use_kd
            # and teacher_model is not None -- same condition kd_weight above
            # already uses; step_idx==0 has teacher_model is None already, so
            # no separate step-1 check is needed here either). 0.0 for every
            # other method, INCLUDING the two non-KD rank_extension methods
            # (whose teacher_model is the pretrained-anchor model, never a
            # real classifier checkpoint) -- existing KD activation logic
            # above (kd_weight, teacher_model construction) is untouched.
            "protect_weight": (
                RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT
                if (method_name in RANKEXT_PROJECTED_PROTECT_METHODS and use_kd and teacher_model is not None)
                else 0.0
            ),
        }

        total_rank, frozen_rank, new_rank = get_rank_extension_rank_triplet(step_idx)
        print(
            f"\n===== {method_name} | step {step_idx + 1}/{NUM_STEPS} | "
            f"rank={total_rank} | "
            f"frozen_rank={frozen_rank} | "
            f"new_rank={new_rank} | "
            f"replay_per_class={replay_per_class} | "
            f"orth={use_orth} | "
            f"orth_mode={active_orth_mode} | "
            f"lambda_orth={active_lambda_orth:.6g} | "
            f"use_kd={use_kd} | "
            f"pretrained_anchor_active={(not use_kd) and teacher_model is not None} | "
            f"teacher_active={teacher_model is not None} | "
            f"old_active_in_forward={old_active_in_forward} ====="
        )

        trainer, _ = train_with_trainer(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            output_dir=out_dir,
            epochs=RANKEXT_EPOCHS,
            lr=LR_RANKEXT,
            batch_size=BATCH_LORA,
            accum_steps=ACCUM_LORA,
            trainer_cls=trainer_cls,
            display_name=METHOD_DISPLAY_NAME_MAP.get(method_name, method_name),
            epoch_loss_records=epoch_loss_rows,
            best_epoch_selection_records=best_epoch_selection_rows,
            rankext_new_block_warmup_epochs=active_new_block_warmup_epochs,
            rankext_new_block_warmup_diagnostic_records=rankext_new_block_warmup_diagnostic_rows,
            **trainer_kwargs,
        )

        if isinstance(trainer, DeltaOrthRankExtensionTrainer):
            loss_rows_df = trainer.consume_logged_losses()
            if loss_rows_df is not None and orth_train_records is not None and len(loss_rows_df) > 0:
                orth_train_records.extend(loss_rows_df.to_dict("records"))

        # Task 2: refresh this method's live convergence plot/tables now that
        # step_idx + 1 has finished.
        refresh_live_convergence(method_name)

        frozen_diff_df = check_frozen_rank_blocks_unchanged(
            model=model,
            snapshot=frozen_snapshot,
            label=f"{method_name} step {step_idx + 1}",
            csv_path=os.path.join(
                TABLES_DIR,
                f"{method_name}_step_{step_idx + 1}_frozen_rank_blocks.csv",
            ),
        )
        restore_protected_classifier_rows(model, classifier_snapshot)
        protected_diff = classifier_protected_row_max_diff(model, classifier_snapshot)
        print(f"[rank_extension diagnostics] protected classifier row max diff after restore: {protected_diff:.10f}")

        grad_stats = collect_rank_block_grad_norms(model=model, train_ds=train_ds)
        print_rank_extension_step_diagnostics(
            method_name=method_name,
            step_idx=step_idx,
            model=model,
            replay_per_class=replay_per_class,
            old_active_in_forward=old_active_in_forward,
            frozen_diff_df=frozen_diff_df,
            grad_stats=grad_stats,
            eps=float(ORTH_NORM_EPS),
        )

        seen_acc = evaluate_seen_step_accuracies(model=model, upto_step_idx=step_idx)
        stepwise_task_accuracies[int(step_idx)] = seen_acc

        for h in hooks:
            h.remove()

        previous_rank_state = extract_rank_extension_state(model)
        if teacher_model is not None:
            del teacher_model
        del model
        cleanup()

    final_old_active_in_forward = not (zero_old_merge and (NUM_STEPS - 1) > 0)
    final_rank_model = build_rank_extension_model(
        previous_rank_state=previous_rank_state,
        step_idx=NUM_STEPS - 1,
        old_active_in_forward=final_old_active_in_forward,
    )

    if ACTIVE_METHOD_MAP[method_name]["apply_calibration"]:
        calibration_mode = ACTIVE_METHOD_MAP[method_name].get("calibration_mode", "global")
        if calibration_mode == "confidence_weighted_regime_grouped":
            # FIX 2 (analysis_recency_fix2/report.txt): originally rank_extension
            # only, since CALIBRATION_MODE_BY_FAMILY["simple_avg"] was always
            # "global". CALIBRATION EXPERIMENT (2026-08-25): simple_avg can now
            # also resolve to this mode -- run_simple_avg_variant() carries the
            # identical if/else dispatch (added alongside that config change),
            # so this is no longer the only call site that can hit this branch.
            final_rank_model = calibrate_classifier_row_norms_confidence_weighted(
                final_rank_model,
                epoch_loss_rows=epoch_loss_rows,
                method_name=method_name,
                uses_kd=bool(ACTIVE_METHOD_MAP[method_name]["uses_kd"]),
            )
        else:
            final_rank_model = calibrate_classifier_row_norms(
                final_rank_model,
                mode=calibration_mode,
                uses_kd=bool(ACTIVE_METHOD_MAP[method_name]["uses_kd"]),
                method_name=method_name,
            )
    else:
        # FIX 1 diagnostic: still record pre-calibration row-norm stats for
        # non-calibrated methods so classifier_row_norm_diagnostic_rows has
        # full coverage across all 8 methods, not just the calibrated ones.
        log_classifier_row_norm_diagnostics(final_rank_model, method_name, phase="pre_calibration")

    # RANKEXT DRIFT DIAGNOSTIC: read-only, on the exact final (post-
    # calibration) model this run evaluates -- runs for all rank_extension
    # methods, KD and non-KD alike, so the two output CSVs give a direct
    # non-KD-vs-KD contrast.
    if RANKEXT_DRIFT_DIAGNOSTICS_ENABLED:
        log_rankext_drift_diagnostics(final_rank_model, method_name)

    eval_rows = evaluate_model(final_rank_model, method_name)

    # PRE-THESIS FIX 2: keep the full stepwise accuracy matrix for the
    # forgetting-curve plot (see the "supervisor automation cell" section).
    rank_extension_stepwise_accuracy_by_method[method_name] = dict(stepwise_task_accuracies)

    avg_forgetting = compute_average_forgetting(stepwise_task_accuracies)

    # PRE-THESIS FIX 2: per-CL-step accuracy of the FINAL model (feeds the
    # 8-methods x 5-steps heatmap), plus backward_transfer computed against the
    # diagonal a_i,i already collected in stepwise_task_accuracies during
    # training, plus forward_transfer from the zero-shot probes collected above.
    final_per_step_accuracy = evaluate_per_step_accuracy(final_rank_model, method_name)
    diagonal_accuracy = {
        step_idx: stepwise_task_accuracies[step_idx].get(step_idx, np.nan)
        for step_idx in stepwise_task_accuracies
    }
    backward_transfer = compute_backward_transfer(diagonal_accuracy, final_per_step_accuracy)
    forward_transfer = compute_forward_transfer(forward_transfer_probe)

    print(
        f"[rank_extension summary] method={method_name} | avg_forgetting={avg_forgetting} | "
        f"backward_transfer={backward_transfer} | forward_transfer={forward_transfer}"
    )

    if (use_orth or use_kd) and orth_eval_records is not None:
        for row in eval_rows:
            orth_eval_records.append({
                "method": row["method"],
                "eval_set": row["eval_set"],
                "accuracy": row["accuracy"],
                "loss": row["loss"],
                "orth_mode": active_orth_mode,
                "lambda_orth": float(active_lambda_orth),
                "zero_old_merge": bool(zero_old_merge),
                "use_kd": bool(use_kd),
                "kd_weight": float(active_kd_weight),
                "kd_temperature": float(active_kd_temperature),
                "replay_per_class": int(replay_per_class),
                "old_active_in_forward": bool(final_old_active_in_forward),
                "avg_forgetting": float(avg_forgetting) if not np.isnan(avg_forgetting) else np.nan,
            })

    if orth_summary_records is not None:
        eval_map = {row["eval_set"]: float(row["accuracy"]) for row in eval_rows}
        orth_summary_records.append({
            "method": method_name,
            "orth_mode": "none" if not use_orth else active_orth_mode,
            "lambda_orth": float(active_lambda_orth),
            "zero_old_merge": bool(zero_old_merge),
            "use_kd": bool(use_kd),
            "kd_weight": float(active_kd_weight),
            "kd_temperature": float(active_kd_temperature),
            "replay_per_class": int(replay_per_class),
            "old_active_in_forward": bool(final_old_active_in_forward),
            "first_step": eval_map.get("first_step", np.nan),
            "later_steps": eval_map.get("later_steps", np.nan),
            "all_seen": eval_map.get("all_seen", np.nan),
            "old_new_gap": eval_map.get("first_step", np.nan) - eval_map.get("later_steps", np.nan),
            "avg_forgetting": float(avg_forgetting) if not np.isnan(avg_forgetting) else np.nan,
            # backward_transfer/forward_transfer use the SAME fraction (0..1,
            # not percent) convention as avg_forgetting in this table.
            "backward_transfer": float(backward_transfer) if not np.isnan(backward_transfer) else np.nan,
            "forward_transfer": float(forward_transfer) if not np.isnan(forward_transfer) else np.nan,
        })

    del final_rank_model
    # RANK_EXT FIRST_STEP FIX: pretrained_anchor_model (if built) lived for
    # the whole function, unlike the KD teacher_model (rebuilt/discarded every
    # step) -- free it explicitly here rather than relying on it falling out
    # of scope.
    if pretrained_anchor_model is not None:
        del pretrained_anchor_model
    cleanup()


# ==============================================================================
# JOINT PACK ADDITION -- persistence-aware RankExt training. Splits the
# original run_rank_extension_variant() above into a training phase
# (train_rank_extension_arm, the per-step loop -- checkpointed after it
# finishes) and a finalize phase (finalize_rank_extension_arm, the
# build-final-model/calibrate/evaluate/append tail -- rerun on every
# invocation, cheap, no training). RankExt has only ONE "merge" (its own
# final persistent-model state -- merge_method="persistent"), so unlike
# SimpleAvg there is no second evaluate-with-a-different-merge call.
# ==============================================================================

def compute_rankext_state_fingerprint(previous_rank_state):
    """READINESS-AUDIT-CONVENTION integrity fingerprint (see
    compute_step_states_fingerprint() for the SimpleAvg analog) -- a stable
    content hash of the final RankExt state's per-module LoRA A/B (frozen +
    new, i.e. full_A_B()'s concatenation) and classifier weight/bias. Used
    only to prove a reloaded checkpoint is byte-identical to what was saved;
    never read by any loss/training/eval code."""
    import hashlib
    h = hashlib.sha256()
    for key in sorted(previous_rank_state["lora"].keys()):
        entry = previous_rank_state["lora"][key]
        h.update(key.encode("utf-8"))
        h.update(entry["A"].detach().cpu().contiguous().numpy().tobytes())
        h.update(entry["B"].detach().cpu().contiguous().numpy().tobytes())
    h.update(previous_rank_state["classifier_weight"].detach().cpu().contiguous().numpy().tobytes())
    h.update(previous_rank_state["classifier_bias"].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _rankext_checkpoint_paths(method_name):
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{method_name}__rankext_state.pt")
    marker_path = os.path.join(CHECKPOINTS_DIR, f"{method_name}__DONE.marker")
    return ckpt_path, marker_path


def save_rankext_checkpoint_atomic(method_name, previous_rank_state, stepwise_task_accuracies, forward_transfer_probe):
    """Atomic, crash-safe persistence for one completed RankExt training arm
    -- same tmp-file/os.replace()/marker convention as
    save_arm_checkpoint_atomic() (SimpleAvg). Saves the FULL final
    previous_rank_state (every module's concatenated frozen+new LoRA A/B,
    i.e. all 5 accumulated blocks -- NOT just the last block -- plus the
    final classifier weight/bias), the diagonal-accuracy matrix needed for
    BWT, and the forward-transfer probes -- everything finalize_rank_
    extension_arm() needs to reconstruct the final model and every metric
    without ever retraining."""
    ckpt_path, marker_path = _rankext_checkpoint_paths(method_name)
    tmp_path = ckpt_path + ".tmp"
    fingerprint = compute_rankext_state_fingerprint(previous_rank_state)
    config_fingerprint = compute_method_config_fingerprint(method_name)
    payload = {
        "method_name": method_name,
        "previous_rank_state": previous_rank_state,
        "stepwise_task_accuracies": stepwise_task_accuracies,
        "forward_transfer_probe": forward_transfer_probe,
        "rankext_state_fingerprint": fingerprint,
        "config_fingerprint": config_fingerprint,
        "saved_at_utc": datetime.utcnow().isoformat(),
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    with open(marker_path, "w") as f:
        f.write("done\n")
    print(f"[checkpoint] saved {method_name} -> {ckpt_path} | rankext_state_fingerprint={fingerprint}")


def load_rankext_checkpoint_if_present(method_name):
    ckpt_path, marker_path = _rankext_checkpoint_paths(method_name)
    if os.path.exists(marker_path) and os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location="cpu")
        stored_fp = payload.get("rankext_state_fingerprint")
        if stored_fp is not None:
            reloaded_fp = compute_rankext_state_fingerprint(payload["previous_rank_state"])
            assert reloaded_fp == stored_fp, (
                f"CHECKPOINT RELOAD INTEGRITY CHECK FAILED for {method_name}: "
                f"reloaded fingerprint {reloaded_fp} != saved fingerprint {stored_fp}"
            )
        # FINAL-AUDIT FIX (persistence safety) -- see compute_method_config_
        # fingerprint()'s docstring / the SimpleAvg analog in
        # load_arm_checkpoint_if_present() for the full rationale.
        stored_cfg_fp = payload.get("config_fingerprint")
        current_cfg_fp = compute_method_config_fingerprint(method_name)
        if stored_cfg_fp != current_cfg_fp:
            print(
                f"[resume] REJECTING stale checkpoint for {method_name}: "
                f"stored config_fingerprint={stored_cfg_fp} != current config_fingerprint={current_cfg_fp} "
                "-- this method's config changed since the checkpoint was saved; retraining from scratch."
            )
            return None
        print(
            f"[resume] found completed checkpoint for {method_name} "
            f"(saved_at_utc={payload.get('saved_at_utc')}, rankext_state_fingerprint={stored_fp}, "
            f"config_fingerprint={stored_cfg_fp}) -- skipping training."
        )
        return payload
    return None


def train_rank_extension_arm(
    method_name,
    replay_per_class=0,
    use_orth=False,
    orth_mode=None,
    lambda_orth=0.0,
    zero_old_merge=False,
    use_kd=False,
    kd_weight=0.0,
    kd_temperature=2.0,
    kd_class_scope="full",
    kd_warmup_epochs=0.0,
    orth_train_records=None,
    classifier_restoration_records=None,
    cl_trajectory_records=None,
):
    """Phase 1 of 2 (training) for one RankExt arm -- the per-step loop,
    extracted verbatim from run_rank_extension_variant() above (same
    teacher/classifier-protection/new-block-warmup/frozen-block-verification
    machinery, byte-identical for every param this function shares with the
    original), plus JOINT PACK threading of kd_class_scope/kd_warmup_epochs
    (Arms RE-1/RE-3) into DeltaOrthRankExtensionTrainer's new constructor
    params. Returns (previous_rank_state, stepwise_task_accuracies,
    forward_transfer_probe) -- everything finalize_rank_extension_arm()
    needs; does not build/evaluate the final model itself."""
    set_seed(SEED)
    previous_rank_state = None
    stepwise_task_accuracies = {}
    forward_transfer_probe = {}
    active_orth_mode = "none" if not use_orth else (None if orth_mode is None else str(orth_mode))
    active_lambda_orth = float(lambda_orth)
    active_kd_weight = float(kd_weight)
    active_kd_temperature = float(kd_temperature)
    active_new_block_warmup_epochs_value = method_rankext_new_block_warmup_epochs(method_name, "rank_extension")
    active_new_block_warmup_epochs = (
        float(active_new_block_warmup_epochs_value) if active_new_block_warmup_epochs_value > 0.0 else None
    )

    pretrained_anchor_model = None
    if not use_kd:
        pretrained_anchor_model = build_rank_extension_model(
            previous_rank_state=None, step_idx=0, old_active_in_forward=True,
        )
        pretrained_anchor_model.eval()
        for p in pretrained_anchor_model.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in pretrained_anchor_model.parameters())

    for step_idx in range(NUM_STEPS):
        current_classes = classes_for_step(step_idx)
        trainable_classifier_classes = rank_extension_trainable_classifier_classes(
            step_idx=step_idx, replay_per_class=replay_per_class,
        )

        old_active_in_forward = not (zero_old_merge and step_idx > 0)
        teacher_model = None
        if pretrained_anchor_model is not None:
            teacher_model = pretrained_anchor_model
        elif use_kd and previous_rank_state is not None:
            teacher_old_active_in_forward = not (zero_old_merge and (step_idx - 1) > 0)
            teacher_model = build_rank_extension_model(
                previous_rank_state=previous_rank_state,
                step_idx=step_idx - 1,
                old_active_in_forward=teacher_old_active_in_forward,
            )
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            assert not any(p.requires_grad for p in teacher_model.parameters())

        # JOINT PACK ADDITION (RankExt Arms RE-1/RE-3): teacher construction
        # is UNCHANGED (still "the model as it stood at the end of the
        # previous step") -- only the KD loss's class scope differs.
        kd_old_class_ids_for_step = None
        if use_kd and kd_class_scope == "old_seen":
            kd_old_class_ids_for_step = old_seen_class_ids(step_idx)

        model = build_rank_extension_model(
            previous_rank_state=previous_rank_state, step_idx=step_idx, old_active_in_forward=old_active_in_forward,
        )
        assert_rank_extension_structure(model=model, step_idx=step_idx)
        save_rank_extension_structure_csv(
            model=model, method_name=method_name, step_idx=step_idx,
            csv_path=os.path.join(TABLES_DIR, f"{method_name}_step_{step_idx + 1}_rank_structure.csv"),
        )
        save_trainable_parameters_csv(
            model=model, method_name=method_name, step_idx=step_idx,
            csv_path=os.path.join(TABLES_DIR, f"{method_name}_step_{step_idx + 1}_trainable_parameters.csv"),
        )

        if step_idx > 0:
            forward_transfer_probe[int(step_idx)] = evaluate_single_step_accuracy(model, step_idx)

        hooks = add_classifier_row_gradient_mask(model=model, trainable_classes=trainable_classifier_classes)
        classifier_snapshot = snapshot_protected_classifier_rows(model=model, trainable_classes=trainable_classifier_classes)
        frozen_snapshot = snapshot_frozen_rank_blocks(model)

        train_ds = make_train_dataset(step_idx=step_idx, replay_per_class=replay_per_class)
        eval_ds = make_val_dataset(current_classes)
        out_dir = os.path.join(MODELS_DIR, f"{method_name}_step_{step_idx + 1}")

        trainer_cls = DeltaOrthRankExtensionTrainer
        trainer_kwargs = {
            "classifier_snapshot": classifier_snapshot,
            "lambda_orth": active_lambda_orth,
            "orth_mode": active_orth_mode,
            "orth_eps": float(ORTH_NORM_EPS),
            "log_every_steps": int(ORTH_LOSS_LOG_EVERY),
            "method_name": method_name,
            "step_idx": int(step_idx),
            "teacher_model": teacher_model,
            "kd_weight": active_kd_weight if (use_kd and teacher_model is not None) else 0.0,
            "kd_temperature": active_kd_temperature,
            "kd_class_scope": str(kd_class_scope),
            "kd_old_class_ids": kd_old_class_ids_for_step,
            "kd_warmup_epochs": float(kd_warmup_epochs),
            "pretrained_anchor_weight": RANKEXT_PRETRAINED_ANCHOR_WEIGHT if (not use_kd and teacher_model is not None) else 0.0,
            "protect_weight": (
                RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT
                if (method_name in RANKEXT_PROJECTED_PROTECT_METHODS and use_kd and teacher_model is not None)
                else 0.0
            ),
        }

        total_rank, frozen_rank, new_rank = get_rank_extension_rank_triplet(step_idx)
        print(
            f"\n===== {method_name} | step {step_idx + 1}/{NUM_STEPS} | rank={total_rank} | "
            f"frozen_rank={frozen_rank} | new_rank={new_rank} | replay_per_class={replay_per_class} | "
            f"orth={use_orth} | orth_mode={active_orth_mode} | lambda_orth={active_lambda_orth:.6g} | "
            f"use_kd={use_kd} | kd_class_scope={kd_class_scope} | kd_warmup_epochs={kd_warmup_epochs} | "
            f"pretrained_anchor_active={(not use_kd) and teacher_model is not None} | "
            f"teacher_active={teacher_model is not None} | old_active_in_forward={old_active_in_forward} ====="
        )

        trainer, _ = train_with_trainer(
            model=model, train_ds=train_ds, eval_ds=eval_ds, output_dir=out_dir,
            epochs=RANKEXT_EPOCHS, lr=LR_RANKEXT, batch_size=BATCH_LORA, accum_steps=ACCUM_LORA,
            trainer_cls=trainer_cls, display_name=METHOD_DISPLAY_NAME_MAP.get(method_name, method_name),
            epoch_loss_records=epoch_loss_rows, best_epoch_selection_records=best_epoch_selection_rows,
            rankext_new_block_warmup_epochs=active_new_block_warmup_epochs,
            rankext_new_block_warmup_diagnostic_records=rankext_new_block_warmup_diagnostic_rows,
            **trainer_kwargs,
        )

        if isinstance(trainer, DeltaOrthRankExtensionTrainer):
            loss_rows_df = trainer.consume_logged_losses()
            if loss_rows_df is not None and orth_train_records is not None and len(loss_rows_df) > 0:
                orth_train_records.extend(loss_rows_df.to_dict("records"))

        refresh_live_convergence(method_name)

        frozen_diff_df = check_frozen_rank_blocks_unchanged(
            model=model, snapshot=frozen_snapshot, label=f"{method_name} step {step_idx + 1}",
            csv_path=os.path.join(TABLES_DIR, f"{method_name}_step_{step_idx + 1}_frozen_rank_blocks.csv"),
        )
        restore_protected_classifier_rows(model, classifier_snapshot)
        protected_weight_diff, protected_bias_diff = classifier_protected_row_diffs(model, classifier_snapshot)
        print(
            f"[rank_extension diagnostics] protected classifier row max diff after restore: "
            f"weight={protected_weight_diff:.10f} bias={protected_bias_diff:.10f}"
        )
        # FINAL 8-ARM SCRIPT ADDITION (Part 11): persist what was previously
        # only printed. `classifier_snapshot["rows"]` (the protected, i.e.
        # NOT currently trainable, class rows at this step) already excludes
        # step_idx==0's classes since no rows are protected at step 1 --
        # n_protected_rows==0 there, matching every other diagnostic in this
        # file's step-1 convention.
        if classifier_restoration_records is not None:
            classifier_restoration_records.append({
                "method": method_name,
                "step": int(step_idx + 1),
                "n_protected_rows": int(len(classifier_snapshot["rows"])),
                "protected_weight_max_abs_diff": float(protected_weight_diff),
                "protected_bias_max_abs_diff": float(protected_bias_diff),
            })

        grad_stats = collect_rank_block_grad_norms(model=model, train_ds=train_ds)
        print_rank_extension_step_diagnostics(
            method_name=method_name, step_idx=step_idx, model=model, replay_per_class=replay_per_class,
            old_active_in_forward=old_active_in_forward, frozen_diff_df=frozen_diff_df,
            grad_stats=grad_stats, eps=float(ORTH_NORM_EPS),
        )

        seen_acc = evaluate_seen_step_accuracies(model=model, upto_step_idx=step_idx)
        stepwise_task_accuracies[int(step_idx)] = seen_acc

        # FINAL 8-ARM SCRIPT ADDITION (Part 10): restricted-accuracy
        # counterpart of seen_acc above, evaluated at the SAME point (right
        # after this step's model is finalized, before moving to the next
        # step) -- gives the full open+restricted retention/plasticity
        # trajectory the task brief asks for. is_current_task/is_old_task are
        # derived downstream (in the final reporting section) from
        # after_training_step vs. evaluated_task, not stored redundantly here.
        if cl_trajectory_records is not None:
            seen_acc_restricted = evaluate_seen_step_accuracies_restricted(model=model, upto_step_idx=step_idx)
            for _eval_task_idx, _open_acc in seen_acc.items():
                cl_trajectory_records.append({
                    "method": method_name,
                    "after_training_step": int(step_idx + 1),
                    "evaluated_task_step": int(_eval_task_idx + 1),
                    "open_accuracy": float(_open_acc) * 100.0,
                    "restricted_accuracy": float(seen_acc_restricted.get(_eval_task_idx, np.nan)) * 100.0,
                    "is_current_task": bool(_eval_task_idx == step_idx),
                    "is_old_task": bool(_eval_task_idx < step_idx),
                })

        for h in hooks:
            h.remove()

        previous_rank_state = extract_rank_extension_state(model)
        if teacher_model is not None:
            del teacher_model
        del model
        cleanup()

    if pretrained_anchor_model is not None:
        del pretrained_anchor_model
    cleanup()

    return previous_rank_state, stepwise_task_accuracies, forward_transfer_probe


def finalize_rank_extension_arm(
    method_name, previous_rank_state, stepwise_task_accuracies, forward_transfer_probe,
    zero_old_merge=False, use_orth=False, orth_mode=None, lambda_orth=0.0,
    use_kd=False, kd_weight=0.0, kd_temperature=2.0,
    orth_eval_records=None, orth_summary_records=None, specialist_fingerprint=None,
):
    """Phase 2 of 2 (finalize + evaluate) for one RankExt arm -- extracted
    verbatim from run_rank_extension_variant() above, called from a saved
    (possibly reloaded-from-disk) previous_rank_state; no training occurs
    here. RankExt's only merge_method is "persistent" -- the final model IS
    the fully-trained persistent state, there is no separate post-hoc merge
    operator (Section 7 of the readiness audit: DO-Merging/arithmetic
    averaging are SimpleAvg-only, never applied here). Also appends to THIS
    script's own joint training_merge_summary_rows table (family=
    "rank_extension", merge_method="persistent"), in addition to the
    pre-existing orth_eval_records/orth_summary_records (method_summary_rows)
    tables Exp1's own downstream reporting expects -- same "populate both"
    convention used for SimpleAvg's arithmetic merge."""
    active_orth_mode = "none" if not use_orth else (None if orth_mode is None else str(orth_mode))
    active_lambda_orth = float(lambda_orth)
    active_kd_weight = float(kd_weight)
    active_kd_temperature = float(kd_temperature)

    final_old_active_in_forward = not (zero_old_merge and (NUM_STEPS - 1) > 0)
    final_rank_model = build_rank_extension_model(
        previous_rank_state=previous_rank_state, step_idx=NUM_STEPS - 1, old_active_in_forward=final_old_active_in_forward,
    )

    if ACTIVE_METHOD_MAP[method_name]["apply_calibration"]:
        calibration_mode = ACTIVE_METHOD_MAP[method_name].get("calibration_mode", "global")
        if calibration_mode == "confidence_weighted_regime_grouped":
            final_rank_model = calibrate_classifier_row_norms_confidence_weighted(
                final_rank_model, epoch_loss_rows=epoch_loss_rows, method_name=method_name,
                uses_kd=bool(ACTIVE_METHOD_MAP[method_name]["uses_kd"]),
            )
        else:
            final_rank_model = calibrate_classifier_row_norms(
                final_rank_model, mode=calibration_mode, uses_kd=bool(ACTIVE_METHOD_MAP[method_name]["uses_kd"]),
                method_name=method_name,
            )
    else:
        log_classifier_row_norm_diagnostics(final_rank_model, method_name, phase="pre_calibration")

    if RANKEXT_DRIFT_DIAGNOSTICS_ENABLED:
        log_rankext_drift_diagnostics(final_rank_model, method_name)

    eval_rows = evaluate_model(final_rank_model, method_name)
    rank_extension_stepwise_accuracy_by_method[method_name] = dict(stepwise_task_accuracies)
    avg_forgetting = compute_average_forgetting(stepwise_task_accuracies)

    final_per_step_accuracy = evaluate_per_step_accuracy(final_rank_model, method_name)
    diagonal_accuracy = {
        step_idx: stepwise_task_accuracies[step_idx].get(step_idx, np.nan) for step_idx in stepwise_task_accuracies
    }
    backward_transfer = compute_backward_transfer(diagonal_accuracy, final_per_step_accuracy)
    forward_transfer = compute_forward_transfer(forward_transfer_probe)

    print(
        f"[rank_extension summary] method={method_name} | avg_forgetting={avg_forgetting} | "
        f"backward_transfer={backward_transfer} | forward_transfer={forward_transfer} | "
        f"specialist_fingerprint={specialist_fingerprint}"
    )

    if (use_orth or use_kd) and orth_eval_records is not None:
        for row in eval_rows:
            orth_eval_records.append({
                "method": row["method"], "eval_set": row["eval_set"], "accuracy": row["accuracy"], "loss": row["loss"],
                "orth_mode": active_orth_mode, "lambda_orth": float(active_lambda_orth),
                "zero_old_merge": bool(zero_old_merge), "use_kd": bool(use_kd),
                "kd_weight": float(active_kd_weight), "kd_temperature": float(active_kd_temperature),
                "replay_per_class": 0, "old_active_in_forward": bool(final_old_active_in_forward),
                "avg_forgetting": float(avg_forgetting) if not np.isnan(avg_forgetting) else np.nan,
            })

    eval_map = {row["eval_set"]: float(row["accuracy"]) for row in eval_rows}
    if orth_summary_records is not None:
        orth_summary_records.append({
            "method": method_name, "orth_mode": "none" if not use_orth else active_orth_mode,
            "lambda_orth": float(active_lambda_orth), "zero_old_merge": bool(zero_old_merge),
            "use_kd": bool(use_kd), "kd_weight": float(active_kd_weight), "kd_temperature": float(active_kd_temperature),
            "replay_per_class": 0, "old_active_in_forward": bool(final_old_active_in_forward),
            "first_step": eval_map.get("first_step", np.nan), "later_steps": eval_map.get("later_steps", np.nan),
            "all_seen": eval_map.get("all_seen", np.nan),
            "old_new_gap": eval_map.get("first_step", np.nan) - eval_map.get("later_steps", np.nan),
            "avg_forgetting": float(avg_forgetting) if not np.isnan(avg_forgetting) else np.nan,
            "backward_transfer": float(backward_transfer) if not np.isnan(backward_transfer) else np.nan,
            "forward_transfer": float(forward_transfer) if not np.isnan(forward_transfer) else np.nan,
        })

    # JOINT PACK ADDITION: restricted mean, from the module-global accumulator
    # evaluate_per_step_accuracy() just populated for this method_name.
    # BUGFIX (found via local FAST_RUN_DEBUG sanity run): the correct column
    # name is "accuracy_restricted" (see evaluate_per_step_accuracy()'s row
    # construction, L3997-4003) -- "accuracy" never existed in this
    # accumulator and would KeyError here on every RankExt arm's
    # finalization, in both Phase 1 and Phase 2.
    restricted_rows_this_method = [r for r in per_step_accuracy_restricted_rows if r.get("method") == method_name]
    restricted_mean = (
        float(np.mean([r["accuracy_restricted"] for r in restricted_rows_this_method]))
        if len(restricted_rows_this_method) > 0 else np.nan
    )

    append_training_merge_summary(
        method_name, "persistent", eval_rows, backward_transfer=backward_transfer,
        specialist_fingerprint=specialist_fingerprint, family="rank_extension",
        restricted_mean=restricted_mean, forgetting_metric=avg_forgetting,
    )

    del final_rank_model
    cleanup()


orth_kd_eval_rows = []
orth_kd_train_rows = train_diagnostic_rows if "train_diagnostic_rows" in globals() else []
orth_kd_summary_rows = method_summary_rows if "method_summary_rows" in globals() else []
# FINAL 8-ARM SCRIPT ADDITION (Parts 10/11) -- fresh accumulators, populated
# only on a genuine (non-cached) training pass through train_rank_extension_arm()
# below, exactly like orth_kd_train_rows above (both are skipped identically
# on checkpoint-resume, which is not expected on this script's first,
# from-scratch production run).
classifier_restoration_rows = []
cl_trajectory_rows = []

rank_extension_execution_order = [cfg["method"] for cfg in ACTIVE_METHOD_CONFIGS if cfg["family"] == "rank_extension"]
for method_name in rank_extension_execution_order:
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    base_method = method_cfg["base_method"]
    if not METHODS_TO_RUN.get(base_method, False):
        print(f"Skipping {method_name} because {base_method} is disabled")
        continue

    _re_use_orth = bool(method_cfg["uses_delta_trace"] or method_cfg["uses_factor_orth"] or method_cfg.get("uses_dense_orth", False))
    _re_orth_mode = (
        "delta_trace" if method_cfg["uses_delta_trace"]
        else ("factor_orth" if method_cfg["uses_factor_orth"]
        else ("dense_orth" if method_cfg.get("uses_dense_orth", False) else None))
    )
    _re_kd_class_scope = method_cfg.get("kd_class_scope", "full")
    _re_kd_warmup_epochs = float(method_cfg.get("kd_warmup_epochs", 0.0))

    _rankext_cached = load_rankext_checkpoint_if_present(method_name)
    if _rankext_cached is not None:
        _rankext_prev_state = _rankext_cached["previous_rank_state"]
        _rankext_stepwise_acc = _rankext_cached["stepwise_task_accuracies"]
        _rankext_fwd_probe = _rankext_cached["forward_transfer_probe"]
        _rankext_fingerprint = _rankext_cached["rankext_state_fingerprint"]
    else:
        _rankext_prev_state, _rankext_stepwise_acc, _rankext_fwd_probe = train_rank_extension_arm(
            method_name=method_name,
            replay_per_class=0,
            use_orth=_re_use_orth,
            orth_mode=_re_orth_mode,
            lambda_orth=float(method_cfg["lambda_orth"]),
            zero_old_merge=False,
            use_kd=bool(method_cfg["uses_kd"]),
            kd_weight=float(method_cfg["kd_weight"]),
            kd_temperature=float(method_cfg["kd_temperature"]),
            kd_class_scope=_re_kd_class_scope,
            kd_warmup_epochs=_re_kd_warmup_epochs,
            orth_train_records=orth_kd_train_rows,
            classifier_restoration_records=classifier_restoration_rows,
            cl_trajectory_records=cl_trajectory_rows,
        )
        save_rankext_checkpoint_atomic(method_name, _rankext_prev_state, _rankext_stepwise_acc, _rankext_fwd_probe)
        _rankext_fingerprint = compute_rankext_state_fingerprint(_rankext_prev_state)

    finalize_rank_extension_arm(
        method_name=method_name,
        previous_rank_state=_rankext_prev_state,
        stepwise_task_accuracies=_rankext_stepwise_acc,
        forward_transfer_probe=_rankext_fwd_probe,
        zero_old_merge=False,
        use_orth=_re_use_orth,
        orth_mode=_re_orth_mode,
        lambda_orth=float(method_cfg["lambda_orth"]),
        use_kd=bool(method_cfg["uses_kd"]),
        kd_weight=float(method_cfg["kd_weight"]),
        kd_temperature=float(method_cfg["kd_temperature"]),
        orth_eval_records=orth_kd_eval_rows,
        orth_summary_records=orth_kd_summary_rows,
        specialist_fingerprint=_rankext_fingerprint,
    )

if len(orth_kd_train_rows) > 0:
    print(f"[rank_extension] accumulated training-loss rows: {len(orth_kd_train_rows)}")


# ==============================================================================
# FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT -- REPORTING SECTION. Everything
# below reads ONLY from in-memory accumulators already populated, UNCHANGED,
# by the training/evaluation loops above (training_merge_summary_rows,
# method_summary_rows, per_step_accuracy_rows, per_step_accuracy_restricted_rows,
# orth_kd_train_rows/train_diagnostic_rows, epoch_loss_rows,
# best_epoch_selection_rows, classifier_restoration_rows, cl_trajectory_rows,
# rankext_new_block_warmup_diagnostic_rows) -- no new training or evaluation
# happens here, only writing/aggregating what was already computed.
# ==============================================================================

print("=" * 80)
print("FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT -- REPORTING SECTION START")
print("=" * 80)

EXPECTED_9_METHODS = sorted(EXPECTED_METHODS)
print(f"Expected 9 methods: {EXPECTED_9_METHODS}")

# ------------------------------------------------------------------------
# 1) Granular per-batch loss log for ALL 9 arms.
# ------------------------------------------------------------------------
final9_batch_log_df = pd.DataFrame(orth_kd_train_rows)
final9_batch_log_path = os.path.join(LOGS_DIR, "final_9method_training_loss_history_by_batch.csv")
final9_batch_log_df.to_csv(final9_batch_log_path, index=False)
print(f"[reporting] wrote {len(final9_batch_log_df)} per-batch loss rows -> {final9_batch_log_path}")
_methods_in_batch_log = set(final9_batch_log_df["method"].unique()) if "method" in final9_batch_log_df.columns else set()
print(f"[reporting] per-batch log covers methods: {sorted(_methods_in_batch_log)}")

# Aggregate to per-(method, step, epoch) means -- a lighter-weight table for
# the same "by CL step / by epoch" DenseOrth/KD/FactorOrth breakdown the
# lambda/KD design audit found was unavailable in job 4969059.
if len(final9_batch_log_df) > 0 and {"method", "step", "epoch"}.issubset(final9_batch_log_df.columns):
    _agg_cols = [
        c for c in [
            "ce_loss", "total_loss", "kd_loss", "weighted_kd_loss", "kd_weight",
            "kd_warmup_multiplier", "effective_kd_coefficient", "kd_over_CE",
            "orth_loss_used", "lambda_orth_times_loss", "lambda_orth",
            "effective_lambda", "effective_orth_coefficient", "orth_warmup_multiplier",
            "weighted_orth_over_CE", "dense_orth_mean", "factor_total_penalty_mean",
            "weighted_factor_orth_mean", "weighted_factor_orth_over_CE",
            "protect_loss", "weighted_protect", "protect_weight",
        ] if c in final9_batch_log_df.columns
    ]
    final9_epoch_agg_df = (
        final9_batch_log_df.groupby(["method", "step", "epoch"], as_index=False)[_agg_cols].mean()
    )
    final9_epoch_agg_path = os.path.join(TABLES_DIR, "final_9method_loss_by_method_step_epoch.csv")
    final9_epoch_agg_df.to_csv(final9_epoch_agg_path, index=False)
    print(f"[reporting] wrote {len(final9_epoch_agg_df)} (method, step, epoch) aggregated loss rows -> {final9_epoch_agg_path}")
else:
    final9_epoch_agg_df = pd.DataFrame()
    print("[reporting][WARNING] per-batch log is empty or missing method/step/epoch columns -- no epoch aggregation written")

# ------------------------------------------------------------------------
# 2) Full per-epoch train/val CE history for all 9 arms (raw accumulator,
#    already populated per-arm/per-step/per-epoch, unconditionally, by
#    train_with_trainer() for every call site in both families).
# ------------------------------------------------------------------------
final9_epoch_loss_df = pd.DataFrame(epoch_loss_rows)
final9_epoch_loss_path = os.path.join(TABLES_DIR, "final_9method_training_loss_history_by_epoch.csv")
final9_epoch_loss_df.to_csv(final9_epoch_loss_path, index=False)
print(f"[reporting] wrote {len(final9_epoch_loss_df)} raw per-epoch rows -> {final9_epoch_loss_path}")

final9_best_epoch_df = pd.DataFrame(best_epoch_selection_rows)
final9_best_epoch_path = os.path.join(TABLES_DIR, "final_9method_best_epoch_selection_by_method_step.csv")
final9_best_epoch_df.to_csv(final9_best_epoch_path, index=False)
print(f"[reporting] wrote {len(final9_best_epoch_df)} best-epoch-selection rows -> {final9_best_epoch_path}")

final9_new_block_warmup_df = pd.DataFrame(rankext_new_block_warmup_diagnostic_rows)
final9_new_block_warmup_path = os.path.join(TABLES_DIR, "final_9method_rankext_new_block_warmup_diagnostics.csv")
final9_new_block_warmup_df.to_csv(final9_new_block_warmup_path, index=False)
print(f"[reporting] wrote {len(final9_new_block_warmup_df)} new-block-warmup diagnostic rows -> {final9_new_block_warmup_path}")

# ------------------------------------------------------------------------
# 3) Per-step accuracy (open + restricted), all 9 arms.
# ------------------------------------------------------------------------
final9_per_step_open_df = pd.DataFrame(per_step_accuracy_rows)
final9_per_step_open_path = os.path.join(TABLES_DIR, "final_9method_per_step_accuracy_open.csv")
final9_per_step_open_df.to_csv(final9_per_step_open_path, index=False)

final9_per_step_restricted_df = pd.DataFrame(per_step_accuracy_restricted_rows)
final9_per_step_restricted_path = os.path.join(TABLES_DIR, "final_9method_per_step_accuracy_open_vs_restricted.csv")
final9_per_step_restricted_df.to_csv(final9_per_step_restricted_path, index=False)
print(
    f"[reporting] wrote {len(final9_per_step_open_df)} open per-step rows and "
    f"{len(final9_per_step_restricted_df)} open-vs-restricted per-step rows"
)

# ------------------------------------------------------------------------
# 4) Classifier protected-row restoration diagnostics.
# ------------------------------------------------------------------------
final9_classifier_restoration_df = pd.DataFrame(classifier_restoration_rows)
final9_classifier_restoration_path = os.path.join(TABLES_DIR, "final_9method_classifier_restoration_diagnostics.csv")
final9_classifier_restoration_df.to_csv(final9_classifier_restoration_path, index=False)
print(f"[reporting] wrote {len(final9_classifier_restoration_df)} classifier-restoration rows -> {final9_classifier_restoration_path}")
if len(final9_classifier_restoration_df) > 0:
    _max_w = float(final9_classifier_restoration_df["protected_weight_max_abs_diff"].max())
    _max_b = float(final9_classifier_restoration_df["protected_bias_max_abs_diff"].max())
    print(f"[reporting] classifier restoration: max weight diff across all RankExt arms/steps = {_max_w:.10f}")
    print(f"[reporting] classifier restoration: max bias diff across all RankExt arms/steps = {_max_b:.10f}")
    assert _max_w < 1e-6 and _max_b < 1e-6, (
        "STOP CONDITION: classifier protected-row restoration failed for at least one "
        f"RankExt arm/step -- max weight diff={_max_w}, max bias diff={_max_b} (expected ~0.0)."
    )

# ------------------------------------------------------------------------
# 5) CL retention/plasticity trajectory -- RankExt only; SimpleAvg has no
#    persistent inter-step model to re-evaluate (5 independently trained
#    specialists, merged exactly once at the very end), so a step-by-step
#    trajectory is structurally undefined for it -- NOT fabricated here
#    (task brief Section 27).
# ------------------------------------------------------------------------
final9_cl_trajectory_df = pd.DataFrame(cl_trajectory_rows)
final9_cl_trajectory_path = os.path.join(TABLES_DIR, "final_9method_rankext_cl_retention_plasticity_trajectory.csv")
final9_cl_trajectory_df.to_csv(final9_cl_trajectory_path, index=False)
print(f"[reporting] wrote {len(final9_cl_trajectory_df)} RankExt CL trajectory rows -> {final9_cl_trajectory_path}")
print(
    "[reporting] NOTE: no equivalent trajectory table is produced for SimpleAvg -- its 5 arms "
    "each train 5 independent per-step specialists with no intermediate merged model between "
    "steps, so accuracy on previously-seen steps after training step k is not a well-defined "
    "quantity for k<5 in that architecture (only the final, fully-merged model at k=5 is ever "
    "evaluated). SimpleAvg's per-step ACQUISITION signal is instead visible in "
    "final_9method_training_loss_history_by_epoch.csv (each specialist's own train/val CE at "
    "training time), and its RETENTION signal in restricted_mean/backward_transfer in the main "
    "summary table below -- these are the closest honest analogues, not a fabricated substitute "
    "for the RankExt-style trajectory."
)

# ------------------------------------------------------------------------
# 6) FINAL SUMMARY TABLE -- all 9 arms.
#
# REPORTING-UNITS FIX (task brief Section 28): the prior script
# (final_8arm_family_specific_5x20.py) stored `all_seen`/`first_step`/
# `later_steps_mean`/`old_new_gap` as FRACTIONS (0-1, straight from
# compute_metrics()'s 0-1 accuracy) while `restricted_mean` was already a
# PERCENTAGE (0-100, from evaluate_per_step_accuracy()'s `accuracy_restricted`
# column, which is itself `float(racc) * 100.0`) -- an internal scale
# inconsistency discovered during the job-4970580 forensic analysis
# (R7/final_8arm_job4970580_forensic_analysis.md, Section 1). This produced
# a downstream bug in that script's own auto-generated cross-run comparison
# table (comparing a 0-1 fraction directly against historical 0-100
# percentages, yielding nonsensical ~-75pp deltas). Fixed here by
# multiplying `all_seen`/`first_step`/`later_steps_mean`/`old_new_gap` by
# 100.0 at construction time, so every human-facing accuracy metric in this
# table is consistently 0-100 -- verified by the sanity assertion below,
# which runs BEFORE the table is written to disk.
# ------------------------------------------------------------------------
final9_training_merge_df = pd.DataFrame(training_merge_summary_rows)
assert set(final9_training_merge_df["training_method"].unique()) == set(EXPECTED_9_METHODS), (
    "STOP CONDITION: the result table is missing at least one of the 9 expected arms -- "
    f"got {sorted(final9_training_merge_df['training_method'].unique())}, expected {EXPECTED_9_METHODS}"
)

# validation_CE: mean of best-epoch-selected validation CE across the 5 CL
# steps for each method (from best_epoch_selection_rows, confirmed schema:
# method_name/step_id/selected_val_ce). validation_accuracy is NOT produced
# anywhere in this pipeline (only validation CE is tracked during training,
# for best-epoch selection) -- left as NaN rather than fabricated.
if len(final9_best_epoch_df) > 0 and "method_name" in final9_best_epoch_df.columns:
    _val_ce_by_method = final9_best_epoch_df.groupby("method_name")["selected_val_ce"].mean()
else:
    _val_ce_by_method = pd.Series(dtype=float)

_PERCENTAGE_COLUMNS = ["all_seen", "restricted_mean", "first_step", "later_steps_mean", "old_new_gap"]

_summary_rows = []
for _method in EXPECTED_9_METHODS:
    _cfg = ACTIVE_METHOD_MAP[_method]
    _row = final9_training_merge_df[final9_training_merge_df["training_method"] == _method].iloc[0]
    _orth_type = (
        "factor_orth" if _cfg["uses_factor_orth"]
        else ("dense_orth" if _cfg["uses_dense_orth"] else "none")
    )
    _all_seen_pct = float(_row["all_seen"]) * 100.0
    _first_step_pct = float(_row["first_step"]) * 100.0
    _later_steps_pct = float(_row["later_steps"]) * 100.0
    _summary_rows.append({
        "family": _cfg["family"],
        "method": _method,
        "kd_scope": _cfg.get("kd_class_scope", "full") if _cfg["uses_kd"] else "n/a",
        "kd_T": float(_cfg["kd_temperature"]) if _cfg["uses_kd"] else np.nan,
        "kd_weight": float(_cfg["kd_weight"]) if _cfg["uses_kd"] else 0.0,
        "kd_warmup_epochs": float(_cfg.get("kd_warmup_epochs", 0.0)) if _cfg["uses_kd"] else np.nan,
        "protect_weight": float(RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT) if _method in RANKEXT_PROJECTED_PROTECT_METHODS else 0.0,
        "new_block_warmup_enabled": bool(_method not in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS) if _cfg["family"] == "rank_extension" else np.nan,
        "orth_type": _orth_type,
        "lambda_orth": float(_cfg["lambda_orth"]),
        "all_seen": _all_seen_pct,
        "restricted_mean": float(_row["restricted_mean"]) if not pd.isna(_row["restricted_mean"]) else np.nan,
        "first_step": _first_step_pct,
        "later_steps_mean": _later_steps_pct,
        "old_new_gap": _first_step_pct - _later_steps_pct,
        "BWT": float(_row["backward_transfer"]) if not pd.isna(_row["backward_transfer"]) else np.nan,
        "forgetting": float(_row["forgetting_metric"]) if not pd.isna(_row["forgetting_metric"]) else np.nan,
        "validation_CE_mean_across_steps": float(_val_ce_by_method.get(_method, np.nan)),
        "validation_accuracy": np.nan,  # not tracked by this pipeline
    })
final9_summary_df = pd.DataFrame(_summary_rows)

# Reporting-units sanity check (task brief Section 28) -- runs BEFORE saving.
for _col in _PERCENTAGE_COLUMNS:
    _vals = final9_summary_df[_col].dropna()
    _bad = _vals[(_vals < -100.0) | (_vals > 100.0)]
    assert len(_bad) == 0, (
        f"STOP CONDITION: reporting-units sanity check failed for column '{_col}' -- "
        f"found value(s) outside [-100, 100]: {_bad.tolist()}. Every human-facing accuracy "
        "metric in this table must be a 0-100 percentage, never a 0-1 fraction."
    )
    # all_seen/restricted_mean/first_step/later_steps_mean are non-negative
    # percentages; old_new_gap is a signed difference of two such percentages.
    if _col != "old_new_gap":
        _neg = _vals[_vals < 0.0]
        assert len(_neg) == 0, f"STOP CONDITION: column '{_col}' must be non-negative, found {_neg.tolist()}"

final9_summary_path = os.path.join(TABLES_DIR, "final_9method_summary_table.csv")
final9_summary_df.to_csv(final9_summary_path, index=False)
print(f"[reporting] wrote final 9-method summary table (units: all values 0-100 percentage) -> {final9_summary_path}")
print(final9_summary_df.to_string(index=False))

# ------------------------------------------------------------------------
# 7) AUTOMATIC WITHIN-RUN COMPARISONS -- strongest evidence in this report
#    (same job, same epochs, same seed).
# ------------------------------------------------------------------------
def _all_seen(method_name):
    return float(final9_summary_df.loc[final9_summary_df["method"] == method_name, "all_seen"].iloc[0])

_sa_plain = "simple_avg"
_sa_kd_t2 = "simple_avg_kd_oldseen_T2_warmup"
_sa_kd_t4 = "simple_avg_kd_oldseen_T4_warmup"
_sa_orth = "simple_avg_dense_orth_lam20"
_sa_combined = "simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup"
_re_plain = "rank_extension"
_re_kd = "rank_extension_fullkd_T2_protect30"
_re_orth = "rank_extension_factor_orth_lam50"
_re_combined = "rank_extension_factor_orth_lam50_fullkd_T2_protect30"

_within_run_deltas = [
    {"comparison": "SA KD-T2 - SA", "delta_all_seen_pp": _all_seen(_sa_kd_t2) - _all_seen(_sa_plain)},
    {"comparison": "SA KD-T4 - SA", "delta_all_seen_pp": _all_seen(_sa_kd_t4) - _all_seen(_sa_plain)},
    {"comparison": "SA KD-T4 - SA KD-T2 (temperature ablation)", "delta_all_seen_pp": _all_seen(_sa_kd_t4) - _all_seen(_sa_kd_t2)},
    {"comparison": "SA DenseOrth20 - SA", "delta_all_seen_pp": _all_seen(_sa_orth) - _all_seen(_sa_plain)},
    {"comparison": "SA DenseOrth20+KD-T2 - SA", "delta_all_seen_pp": _all_seen(_sa_combined) - _all_seen(_sa_plain)},
    {"comparison": "SA DenseOrth20+KD-T2 - SA KD-T2", "delta_all_seen_pp": _all_seen(_sa_combined) - _all_seen(_sa_kd_t2)},
    {"comparison": "RE full-KD - RE", "delta_all_seen_pp": _all_seen(_re_kd) - _all_seen(_re_plain)},
    {"comparison": "RE FactorOrth50 - RE", "delta_all_seen_pp": _all_seen(_re_orth) - _all_seen(_re_plain)},
    {"comparison": "RE FactorOrth50+full-KD - RE", "delta_all_seen_pp": _all_seen(_re_combined) - _all_seen(_re_plain)},
    {"comparison": "RE FactorOrth50+full-KD - RE full-KD", "delta_all_seen_pp": _all_seen(_re_combined) - _all_seen(_re_kd)},
]
final9_within_run_df = pd.DataFrame(_within_run_deltas)
final9_within_run_path = os.path.join(TABLES_DIR, "final_9method_automatic_within_run_comparisons.csv")
final9_within_run_df.to_csv(final9_within_run_path, index=False)
print("\n[reporting] AUTOMATIC WITHIN-RUN COMPARISONS (all-seen accuracy, percentage points):")
print(final9_within_run_df.to_string(index=False))

# ------------------------------------------------------------------------
# 8) REFERENCE ROWS -- historical/prior-run values, NEVER retrained here.
#    (task brief Section 29): every row is tagged status=
#    "REFERENCE_ONLY_NOT_RETRAINED" with an explicit source, and is never
#    silently merged into the newly-trained final9_summary_df above. All
#    values below are read verbatim from the task brief / prior reports,
#    never recomputed from raw logs of another run.
# ------------------------------------------------------------------------
_reference_rows = [
    # --- Historical RankExt Exp1 / job 4961962 (9ep, full-100-way KD,
    #     FactorOrth lambda=50, protect_weight=30, broken classifier
    #     restoration, verified bit-for-bit reproduced on the modern
    #     codebase by job 4961962) ---
    {"family": "rank_extension", "method": "rank_extension (historical Exp1/job4961962)", "all_seen": 35.31, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
    {"family": "rank_extension", "method": "rank_extension_fullkd_T2 (historical Exp1/job4961962, full KD, no protect breakdown isolated)", "all_seen": 64.44, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
    {"family": "rank_extension", "method": "rank_extension_factor_orth_lam50 (historical Exp1/job4961962)", "all_seen": 42.21, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
    {"family": "rank_extension", "method": "rank_extension_factor_orth_lam50_fullkd_T2_protect30 (historical Exp1/job4961962, THE 70.72% TARGET)", "all_seen": 70.72, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
    # --- Current RankExt job 4970580 (6ep, old-seen KD weight1.5, corrected
    #     classifier restoration, no protect_weight) ---
    {"family": "rank_extension", "method": "rank_extension (job4970580, 6ep)", "all_seen": 30.56, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 3"},
    {"family": "rank_extension", "method": "rank_extension_kd_oldseen_T2_weight1p5_warmup (job4970580, 6ep)", "all_seen": 48.53, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 3"},
    {"family": "rank_extension", "method": "rank_extension_factor_orth_lam50 (job4970580, 6ep)", "all_seen": 40.37, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 3"},
    {"family": "rank_extension", "method": "rank_extension_factor_orth_lam50_kd_oldseen_T2_weight1p5_warmup (job4970580, 6ep)", "all_seen": 53.13, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 3"},
    # --- Current SimpleAvg (job4969059/job4970580, 6ep) ---
    {"family": "simple_avg", "method": "simple_avg (current, 6ep)", "all_seen": 75.98, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 13"},
    {"family": "simple_avg", "method": "simple_avg_kd_oldseen_T2_weight1_no_warmup (job4969059, 6ep)", "all_seen": 75.05, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 13"},
    {"family": "simple_avg", "method": "simple_avg_kd_oldseen_T2_weight1_warmup (job4969059, 6ep)", "all_seen": 76.66, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 13"},
    {"family": "simple_avg", "method": "simple_avg_kd_oldseen_T2_weight0p5_no_warmup (job4970580, 6ep)", "all_seen": 75.77, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 13"},
    {"family": "simple_avg", "method": "simple_avg_dense_orth_lam20 (current, 6ep)", "all_seen": 75.11, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 13"},
    {"family": "simple_avg", "method": "simple_avg_dense_orth_lam20_kd_oldseen_T2_weight1_warmup (job4969059, 6ep)", "all_seen": 76.19, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_8arm_job4970580_forensic_analysis.md Section 13"},
    # --- Historical SimpleAvg Exp1/R7 (9ep, full-100-way KD, FactorOrth) ---
    {"family": "simple_avg", "method": "simple_avg (historical Exp1)", "all_seen": 75.16, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
    {"family": "simple_avg", "method": "simple_avg_fullkd_T2 (historical Exp1)", "all_seen": 62.98, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
    {"family": "simple_avg", "method": "simple_avg_factor_orth (historical Exp1)", "all_seen": 74.29, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
    {"family": "simple_avg", "method": "simple_avg_factor_orth_fullkd_T2 (historical Exp1)", "all_seen": 69.40, "status": "REFERENCE_ONLY_NOT_RETRAINED", "source": "R7/final_recovery_and_improvement_design_audit.md Section 2/3"},
]
final9_reference_df = pd.DataFrame(_reference_rows)
final9_reference_path = os.path.join(TABLES_DIR, "final_9method_reference_values_not_retrained.csv")
final9_reference_df.to_csv(final9_reference_path, index=False)
print(f"\n[reporting] wrote {len(final9_reference_df)} REFERENCE_ONLY_NOT_RETRAINED rows -> {final9_reference_path}")

# ------------------------------------------------------------------------
# 9) CROSS-RUN COMPARISON vs. job 4970580 (current pipeline) and historical
#    R7/Exp1 -- CLEARLY LABELED AS CONFOUNDED, never treated as causal.
#    Within-run comparisons (Section 7 above) are the strongest evidence;
#    everything here differs by MORE than one factor (epochs AND KD scope
#    AND protect_weight AND new-block warmup, simultaneously, for the two
#    RankExt KD arms) and must not be read causally.
# ------------------------------------------------------------------------
_cross_run_rows = [
    {
        "method": _sa_plain, "this_run_all_seen": _all_seen(_sa_plain),
        "job4970580_all_seen": 75.98, "r7_exp1_all_seen": 75.16,
        "confound_note": "epochs differ (9ep this run vs 6ep job4970580); both are otherwise the same plain mechanism",
    },
    {
        "method": f"{_sa_kd_t2} (nearest: job4969059 SA KD weight1 warmup)",
        "this_run_all_seen": _all_seen(_sa_kd_t2),
        "job4970580_all_seen": 76.66, "r7_exp1_all_seen": 62.98,
        "confound_note": "epochs differ (9 vs 6) vs the job4969059 reference; R7 comparison additionally confounded by full-vs-old-seen KD scope",
    },
    {
        "method": f"{_sa_kd_t4} (NEW arm, no direct prior-run analogue)",
        "this_run_all_seen": _all_seen(_sa_kd_t4),
        "job4970580_all_seen": np.nan, "r7_exp1_all_seen": np.nan,
        "confound_note": "never tested in any prior run -- only a within-run T2-vs-T4 comparison exists (Section 7 above)",
    },
    {
        "method": _sa_orth, "this_run_all_seen": _all_seen(_sa_orth),
        "job4970580_all_seen": 75.11, "r7_exp1_all_seen": 74.29,
        "confound_note": "epochs differ (9 vs 6) vs job4970580; R7 used FactorOrth, a different formulation, on top of the epoch difference",
    },
    {
        "method": f"{_sa_combined} (nearest: job4969059 combined)",
        "this_run_all_seen": _all_seen(_sa_combined),
        "job4970580_all_seen": 76.19, "r7_exp1_all_seen": 69.40,
        "confound_note": "epochs differ (9 vs 6); R7 comparison additionally confounded by FactorOrth-vs-DenseOrth and full-vs-old-seen KD",
    },
    {
        "method": _re_plain, "this_run_all_seen": _all_seen(_re_plain),
        "job4970580_all_seen": 30.56, "r7_exp1_all_seen": 35.31,
        "confound_note": "epochs differ (9 vs 6) vs job4970580; R7 comparison additionally confounded by the old (non-restoring) classifier-row bug",
    },
    {
        "method": f"{_re_kd} (nearest: job4970580 RE old-seen KD weight1.5 / R7/Exp1 full-KD)",
        "this_run_all_seen": _all_seen(_re_kd),
        "job4970580_all_seen": 48.53, "r7_exp1_all_seen": 64.44,
        "confound_note": "KD scope differs by design (full vs job4970580's old-seen), protect_weight=30 newly active, epochs differ (9 vs 6); R7/Exp1 is the only row using the SAME mechanism (full-KD+protect30), differing only by the corrected (vs. buggy) classifier restoration -- the single most informative row in this table",
    },
    {
        "method": f"{_re_orth} (nearest: job4970580 RE FactorOrth50 / R7/Exp1 FactorOrth50)",
        "this_run_all_seen": _all_seen(_re_orth),
        "job4970580_all_seen": 40.37, "r7_exp1_all_seen": 42.21,
        "confound_note": "FactorOrth mechanism/lambda unchanged in all three; only epochs differ (9ep this run and R7/Exp1 vs 6ep job4970580) -- the cleanest comparison in this table",
    },
    {
        "method": f"{_re_combined} (nearest: job4970580 combined / R7/Exp1 FactorOrth+full-KD -- THE MAIN CANDIDATE FOR RECOVERING/EXCEEDING 70.72%)",
        "this_run_all_seen": _all_seen(_re_combined),
        "job4970580_all_seen": 53.13, "r7_exp1_all_seen": 70.72,
        "confound_note": "vs job4970580: KD scope (full vs old-seen) and protect_weight (30 vs 0) both differ by design, plus epochs (9 vs 6); vs R7/Exp1: differs ONLY by the corrected (non-buggy) classifier restoration -- see R7/final_recovery_and_improvement_design_audit.md Section 6 for why that fix is argued unlikely to be the source of any residual gap",
    },
]
final9_cross_run_df = pd.DataFrame(_cross_run_rows)
final9_cross_run_df["delta_vs_job4970580_pp"] = final9_cross_run_df["this_run_all_seen"] - final9_cross_run_df["job4970580_all_seen"]
final9_cross_run_df["delta_vs_r7_exp1_pp"] = final9_cross_run_df["this_run_all_seen"] - final9_cross_run_df["r7_exp1_all_seen"]
final9_cross_run_path = os.path.join(TABLES_DIR, "final_9method_cross_run_comparison_CONFOUNDED.csv")
final9_cross_run_df.to_csv(final9_cross_run_path, index=False)
print("\n[reporting] CROSS-RUN COMPARISON (CONFOUNDED -- see confound_note per row, NOT causal):")
print(final9_cross_run_df.to_string(index=False))

# ------------------------------------------------------------------------
# 10) Output checklist.
# ------------------------------------------------------------------------
_overall_pass = (
    set(final9_training_merge_df["training_method"].unique()) == set(EXPECTED_9_METHODS)
    and len(_methods_in_batch_log) >= 7  # at least the 7 non-plain arms must have granular logging
    and len(final9_classifier_restoration_df) > 0
    and len(final9_cl_trajectory_df) > 0
)
_checklist_lines = [
    "OVERALL PASS" if _overall_pass else "OVERALL FAIL -- see details below",
    f"9 expected methods present in results table: {set(final9_training_merge_df['training_method'].unique()) == set(EXPECTED_9_METHODS)}",
    f"Methods with granular per-batch logging: {sorted(_methods_in_batch_log)} ({len(_methods_in_batch_log)}/9)",
    f"Classifier restoration diagnostics rows: {len(final9_classifier_restoration_df)}",
    f"CL retention/plasticity trajectory rows (RankExt): {len(final9_cl_trajectory_df)}",
    f"Summary table path: {final9_summary_path}",
]
final9_checklist_path = os.path.join(REPORTS_DIR, "final_9method_output_checklist.txt")
with open(final9_checklist_path, "w", encoding="utf-8") as _f:
    _f.write("\n".join(_checklist_lines) + "\n")
print(f"\n[reporting] wrote output checklist -> {final9_checklist_path}")
for _line in _checklist_lines:
    print(f"  {_line}")

print("=" * 80)
print("FINAL 9-METHOD PERFORMANCE-RECOVERY SCRIPT -- REPORTING SECTION COMPLETE")
print("=" * 80)
