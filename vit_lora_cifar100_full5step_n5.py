#!/usr/bin/env python
# coding: utf-8

# 
# This notebook implements a focused continual-learning comparison setup for selected `simple_avg` and `rank_extension` variants only.
# 
# Main setup:
# 
# - CLIP-ViT vision encoder: `openai/clip-vit-base-patch16`
# - Split CIFAR-100 continual learning
# - 5 steps, 20 classes per step
# - Replay, zero-old ablations, joint training, full finetuning, DO-merging, and other extra ablations are disabled in this focused run.
# - LoRA target modules are `q_proj`, `k_proj`, `v_proj`, `out_proj` (all four
#   CLIP attention projections; previously `q_proj`/`v_proj` only).
# 
# Method notes:
# 
# - `simple_avg` trains one independent LoRA per step with LoRA rank `80` and alpha `160`, then merges step deltas by simple averaging.
# - KD uses fixed `KD_WEIGHT = 1.0` and is swept only over `KD_TEMPERATURES = [2.0]`.
# - `simple_avg_delta_orth` / `rank_extension_orth_delta_trace_lam_50` use delta-trace orthogonality as an add-on regularizer.
#   (Implemented but currently DISABLED via METHODS_TO_RUN -- see below.)
# - `simple_avg_factor_orth` / `rank_extension_orth_factor_lam_50` keep the normal LoRA forward update `Delta_W = B A` and add factor-level orthogonality only as a regularizer on LoRA factor spaces.
# - `rank_extension` is a true growing-rank LoRA with rank schedule `16 -> 32 -> 48 -> 64 -> 80`: previous `A/B` slices are copied forward, frozen, and only the newest `A/B` slice is trainable.
# - Rank-extension alpha scales with total rank using `rankext_alpha = RANKEXT_ALPHA_PER_RANK * total_rank`.
# - `delta_trace` and `factor_orth` use the same fixed `LAMBDA_ORTH = 50.0` for direct comparability.
# - `simple_avg` and `rank_extension` are compared at the same final rank `80` with `LORA_ALPHA = 160`.
# - Detailed loss components are logged for CE, KD, delta-trace, factor-orth, and total loss.
# - Only the 8 methods below are active; KD-T1 sweeps and the delta-trace(+KD) combos
#   are disabled via `METHODS_TO_RUN` (implementation kept, not deleted -- flip
#   "simple_avg_delta_orth" / "rank_extension_orth_delta_trace_lam_50" back to True
#   there to re-enable delta-trace).
#
# Active comparison set (8 methods):
#
# - `simple_avg`
# - `simple_avg_kd_T2`
# - `simple_avg_factor_orth`
# - `simple_avg_factor_orth_kd_T2`
# - `rank_extension`
# - `rank_extension_kd_only_T2`
# - `rank_extension_orth_factor_lam_50`
# - `rank_extension_orth_factor_lam_50_kd_T2`
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
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEBUG_MODE = False
FAST_RUN = False

RUN_NAME_BASE = "clip_vit_lora_cifar100_full_comparison_with_orth_rankext"
RUN_NAME = f"{RUN_NAME_BASE}_{'FAST_RUN_DEBUG' if FAST_RUN else 'EPOCH3_MAIN'}"

MODEL_CHECKPOINT = "openai/clip-vit-base-patch16"

NUM_CLASSES = 100
NUM_STEPS = 5
CLASSES_PER_STEP = 20




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
FULL_FT_EPOCHS = 9
FULL_LORA_EPOCHS = 9
FULL_JOINT_EPOCHS = 9
FULL_ORTH_EPOCHS = 9
FULL_RANKEXT_EPOCHS = 9

SCRATCH_EPOCHS = 9

FT_EPOCHS = 9
LORA_EPOCHS = 9
JOINT_EPOCHS = 9
ORTH_EPOCHS = 9
RANKEXT_EPOCHS = 9


BATCH_FT = 8
ACCUM_FT = 2

BATCH_LORA = 16
ACCUM_LORA = 1


LR_FT = 3e-5
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

TARGET_MODULES_BY_FAMILY = {
    "simple_avg": list(TARGET_MODULES),
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

# Master switch above still gates calibration overall (False disables it for
# every method, same as before). When True, CALIBRATION_ENABLED_FAMILIES
# decides which families actually get it. simple_avg: keep True (empirically
# helps -- see report). rank_extension: False (its persistent, row-masked
# classifier is not the independently-reinitialized-heads scenario WA
# calibration targets, and calibrating it corrupted the KD variants). Add new
# families here explicitly; unlisted families default to no calibration (see
# family_applies_calibration() below).
CALIBRATION_ENABLED_FAMILIES = {
    "simple_avg": True,
    "rank_extension": False,
}


def family_applies_calibration(family):
    """True iff USE_CLASSIFIER_CALIBRATION is on AND this family opted in via
    CALIBRATION_ENABLED_FAMILIES. Unlisted families default to False (safer
    than silently calibrating a family nobody has reasoned about)."""
    return bool(USE_CLASSIFIER_CALIBRATION) and bool(CALIBRATION_ENABLED_FAMILIES.get(str(family), False))

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

HEAD_LR_MULTIPLIER_BY_FAMILY = {
    "simple_avg": float(HEAD_LR_MULTIPLIER),
    "rank_extension": 1.0,
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
COMBINED_LOSS_SCALE_ENABLED = True
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

VALIDATION_PER_CLASS = 25
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
        "internal_method_name": "simple_avg_factor_orth_kd_T2",
        "supervisor_requested_name": "simple_avg_factor_orth_lam_50_kd_T2",
        "display_name": "SimpleAvg + FactorOrth + KD T2",
        "family": "simple_avg",
        "factor_lambda": 50.0,
        "kd_temperature": 2.0,
        "kd_weight": float(KD_WEIGHT),
    },
    {
        "internal_method_name": "rank_extension_orth_factor_lam_50_kd_T2",
        "supervisor_requested_name": "rank_extension_orth_factor_lam_50_kd_T2",
        "display_name": "RankExt + FactorOrth + KD T2",
        "family": "rank_extension",
        "factor_lambda": 50.0,
        "kd_temperature": 2.0,
        "kd_weight": float(KD_WEIGHT),
    },
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


RANKEXT_RANK_SCHEDULE = [16, 32, 48, 64, 80]
RANKEXT_ALPHA_PER_RANK = 2.0

# ACCURACY-PUSH CANDIDATE (flag, default OFF): wider per-step rank budget.
# The default schedule gives each CL step only 16 fresh trainable ranks per
# target module, vs simple_avg retraining its full LORA_R=80 ranks from
# scratch every step. analysis_rankext_firststep/table1 shows rank_extension
# trailing simple_avg by a wide margin on later_steps/all_seen in BOTH runs,
# consistent with (in addition to, not instead of, the forgetting-time issue
# documented in that report's Target 1c) a plain capacity bottleneck. This is
# an INDEPENDENT lever from RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED below -- keep
# them off together first and enable one at a time so a rerun can attribute
# any change to a single cause, per the report's "attribute honestly" standard.
# Parameter cost: final total_rank 160 vs 80 -- roughly doubles trainable
# rank-extension LoRA parameters per step by the last CL step (2 target
# modules x wider rank, still less total than the calibfix run's 4 modules x
# narrow rank, but meaningfully more than the 2-module/narrow-rank BASELINE
# config that actually scored 68.0). Watch train_val_gap_by_method.csv
# (overfitting score) if this is enabled -- R3/R4 already flagged mild
# overfitting signatures at the smaller capacity setting.
RANKEXT_RANK_SCHEDULE_WIDE = [32, 64, 96, 128, 160]
USE_RANKEXT_RANK_SCHEDULE_WIDE = False
assert len(RANKEXT_RANK_SCHEDULE_WIDE) == NUM_STEPS
assert all(RANKEXT_RANK_SCHEDULE_WIDE[i] > RANKEXT_RANK_SCHEDULE_WIDE[i - 1] for i in range(1, NUM_STEPS))


def active_rankext_rank_schedule():
    """Resolves to RANKEXT_RANK_SCHEDULE_WIDE when USE_RANKEXT_RANK_SCHEDULE_WIDE
    is on, else the default RANKEXT_RANK_SCHEDULE. Single source of truth for
    every consumer (rank-triplet computation, reporting columns, hyperparameter
    dumps) so the flag can't drift out of sync between training and reporting."""
    return list(RANKEXT_RANK_SCHEDULE_WIDE) if USE_RANKEXT_RANK_SCHEDULE_WIDE else list(RANKEXT_RANK_SCHEDULE)


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
METHODS_TO_RUN = {
    "simple_avg": True,
    "simple_avg_kd": True,
    "simple_avg_delta_orth": False,  # disabled for FIX 2 -- was True; delta-trace excluded from the 8-method set
    "simple_avg_delta_orth_kd": False,
    "simple_avg_factor_orth": True,
    "simple_avg_factor_orth_kd": True,
    "rank_extension": True,
    "rank_extension_kd_only": True,
    "rank_extension_orth_delta_trace_lam_50": False,  # disabled for FIX 2 -- was True; delta-trace excluded from the 8-method set
    "rank_extension_orth_delta_trace_lam_50_kd": False,
    "rank_extension_orth_factor_lam_50": True,
    "rank_extension_orth_factor_lam_50_kd": True,
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

    def add_method(method_name, family, base_method, uses_kd=False, kd_temperature=0.0, uses_delta_trace=False, uses_factor_orth=False, lambda_orth_scale=1.0, kd_weight_scale=1.0):
        if not METHODS_TO_RUN.get(base_method, False):
            return
        configs.append({
            "method": str(method_name),
            "family": str(family),
            "base_method": str(base_method),
            "uses_kd": bool(uses_kd),
            "kd_temperature": float(kd_temperature) if uses_kd else 0.0,
            "kd_weight": float(KD_WEIGHT if uses_kd else 0.0) * float(kd_weight_scale),
            "uses_delta_trace": bool(uses_delta_trace),
            "uses_factor_orth": bool(uses_factor_orth),
            "lambda_orth": float(LAMBDA_ORTH if (uses_delta_trace or uses_factor_orth) else 0.0) * float(lambda_orth_scale),
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
            "rank": int(LORA_R),
            "rank_schedule": ("fixed:" + str(LORA_R)) if family == "simple_avg" else "->".join(str(v) for v in active_rankext_rank_schedule()),
            "target_modules": ", ".join(family_target_modules(family)),
            "head_lr_multiplier": family_head_lr_multiplier(family),
            "apply_calibration": family_applies_calibration(family),
        })

    add_method("simple_avg", "simple_avg", "simple_avg")
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
    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(f"rank_extension_kd_only_{kd_tag}", "rank_extension", "rank_extension_kd_only", uses_kd=True, kd_temperature=kd_temp)
    add_method("rank_extension_orth_delta_trace_lam_50", "rank_extension", "rank_extension_orth_delta_trace_lam_50", uses_delta_trace=True)
    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(f"rank_extension_orth_delta_trace_lam_50_kd_{kd_tag}", "rank_extension", "rank_extension_orth_delta_trace_lam_50_kd", uses_kd=True, kd_temperature=kd_temp, uses_delta_trace=True)
    add_method("rank_extension_orth_factor_lam_50", "rank_extension", "rank_extension_orth_factor_lam_50", uses_factor_orth=True)
    for kd_temp in KD_TEMPERATURES:
        kd_tag = kd_temperature_tag(kd_temp)
        add_method(f"rank_extension_orth_factor_lam_50_kd_{kd_tag}", "rank_extension", "rank_extension_orth_factor_lam_50_kd", uses_kd=True, kd_temperature=kd_temp, uses_factor_orth=True)

    return configs


ACTIVE_METHOD_CONFIGS = build_active_method_configs()
ACTIVE_METHOD_NAMES = [cfg["method"] for cfg in ACTIVE_METHOD_CONFIGS]
ACTIVE_METHOD_MAP = {cfg["method"]: cfg for cfg in ACTIVE_METHOD_CONFIGS}
ENABLED_METHOD_FAMILIES = [name for name, enabled in METHODS_TO_RUN.items() if enabled]
# FIX 2: "simple_avg_delta_orth" and "rank_extension_orth_delta_trace_lam_50"
# removed from the expected set to match the two flags flipped to False above --
# otherwise `assert set(ENABLED_METHOD_FAMILIES) == EXPECTED_ENABLED_METHOD_FAMILIES`
# below would fail as soon as those two were disabled.
EXPECTED_ENABLED_METHOD_FAMILIES = {
    "simple_avg",
    "simple_avg_kd",
    "simple_avg_factor_orth",
    "simple_avg_factor_orth_kd",
    "rank_extension",
    "rank_extension_kd_only",
    "rank_extension_orth_factor_lam_50",
    "rank_extension_orth_factor_lam_50_kd",
}

assert KD_WEIGHT == 1.0
assert KD_TEMPERATURES == [2.0]
assert LAMBDA_ORTH == 50.0
assert LORA_R == 80
assert LORA_ALPHA == 160
assert LORA_R == RANKEXT_RANK_SCHEDULE[-1]
assert float(RANKEXT_ALPHA_PER_RANK * RANKEXT_RANK_SCHEDULE[-1]) == float(LORA_ALPHA)
# ACCURACY-PUSH CHANGE 1: pinned set updated from ["q_proj", "v_proj"] to include
# k_proj/out_proj. This is now the simple_avg-family value specifically (see
# TARGET_MODULES_BY_FAMILY REVERT note above) -- keep this assert in sync with
# TARGET_MODULES above -- it exists to catch silent drift between the two,
# not to gate the value itself.
assert TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "out_proj"]
# REVERT (2026-07-16): rank_extension's target modules are pinned back to the
# BASELINE-proven 2-module set. Keep in sync with TARGET_MODULES_BY_FAMILY.
assert TARGET_MODULES_BY_FAMILY["rank_extension"] == ["q_proj", "v_proj"]
assert TARGET_MODULES_BY_FAMILY["simple_avg"] == TARGET_MODULES
assert set(ENABLED_METHOD_FAMILIES) == EXPECTED_ENABLED_METHOD_FAMILIES
assert not any(cfg["uses_replay"] for cfg in ACTIVE_METHOD_CONFIGS)
assert not any(cfg["uses_zero_old"] for cfg in ACTIVE_METHOD_CONFIGS)
assert all(cfg["rank"] == LORA_R for cfg in ACTIVE_METHOD_CONFIGS)
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

kd_method_temperature_map = {}
for cfg in ACTIVE_METHOD_CONFIGS:
    if cfg["uses_kd"]:
        kd_method_temperature_map.setdefault(cfg["base_method"], []).append(float(cfg["kd_temperature"]))
for base_method, temps in kd_method_temperature_map.items():
    assert sorted(temps) == sorted(KD_TEMPERATURES), f"KD sweep mismatch for {base_method}: {temps}"


ROOT_RESULTS_DIR = "results"

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

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
        _plot_step_broken_series(ax, df, "train_ce_loss", "#1f77b4", "train CE")
        _plot_step_broken_series(ax, df, "val_ce_loss", "#d62728", "val CE")
        step_sizes = df.groupby("step_id").size()
        boundary = 0
        for step_id in sorted(df["step_id"].unique())[:-1]:
            boundary += int(step_sizes.loc[step_id])
            ax.axvline(boundary + 0.5, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.set_xlabel("global epoch so far (dotted = CL step boundary)")
        ax.set_ylabel("CE loss")
        disp = METHOD_DISPLAY_NAME_MAP.get(method_name, method_name)
        last_step = int(df["step_id"].max())
        ax.set_title(f"{disp} -- live convergence (through step {last_step}/{NUM_STEPS})")
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
    "TARGET_MODULES (default/simple_avg)": TARGET_MODULES,
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


dataset = load_dataset("cifar100")

LABEL_COL = "fine_label" if "fine_label" in dataset["train"].column_names else "label"
IMAGE_COL = "img" if "img" in dataset["train"].column_names else "image"

class_splits = [
    list(range(0, 20)),
    list(range(20, 40)),
    list(range(40, 60)),
    list(range(60, 80)),
    list(range(80, 100)),
]

first_step_classes = class_splits[0]
later_step_classes = [c for split in class_splits[1:] for c in split]
all_classes = [c for split in class_splits for c in split]

def classes_for_step(step_idx):
    return class_splits[step_idx]

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
        train_parts.append(cls_ds.select(range(n_val, len(cls_ds))))
        rows.append({
            "class_id": int(cls),
            "train_count": int(len(cls_ds) - n_val),
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
    ds = ds.with_transform(preprocess_val)
    return ds


def make_eval_dataset(class_ids):
    ds = filter_by_classes(dataset["test"], class_ids)
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
        dataloader_num_workers=4,
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

    trainer.train()

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

    Returns a {step_idx: accuracy_fraction} map (0..1, NOT percent) for the
    caller to also use in backward_transfer/forward_transfer computations.
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
    for step_idx in range(NUM_STEPS):
        eval_ds = make_eval_dataset(classes_for_step(step_idx))
        out = trainer.evaluate(eval_dataset=eval_ds)
        per_step_map[step_idx] = float(out["eval_accuracy"])

    for step_idx in range(NUM_STEPS):
        acc = per_step_map.get(step_idx, np.nan)
        per_step_accuracy_rows.append({
            "method": method_name,
            "step_id": int(step_idx + 1),
            "accuracy": float(acc) * 100.0 if not np.isnan(acc) else np.nan,
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

def calibrate_classifier_row_norms(model, eps=1e-8):
    """
    ACCURACY-PUSH CHANGE 2: rehearsal-free, post-merge-only classifier
    calibration (WA-style weight alignment, Zhao et al. 2020, "Maintaining
    Discrimination and Fairness in Class Incremental Learning").

    Rescales each CL step's 20-class weight-row block so its mean row norm
    matches the global (all 100 rows) mean row norm, then evaluates. This is a
    pure post-hoc correction on the already-merged/stitched classifier -- no
    retraining, no rehearsal data, applied identically to every method.

    Rationale specific to this codebase: for the simple_avg family,
    apply_deltas_to_base() stitches together 5 classifier row-blocks that each
    came from a SEPARATE freshly-initialized nn.Linear (fresh_pretrained_model()
    is called fresh per step in train_independent_loras()), trained under
    different conditions (KD vs not, orth vs not) and, in every case, trained
    with a 100-way softmax where 80 of the 100 classes are permanently absent
    that step (pure negatives, never positive) -- a well-known recipe for
    severe cross-group row-norm imbalance. For rank_extension the classifier is
    shared/incremental rather than independently re-initialized, but the same
    row-norm-driven scale bias can still arise across step-groups trained at
    different points in the schedule, so the identical correction is applied
    there too (uniform protocol across all 8 methods).

    Only the weight rows are rescaled, not the bias, matching the original WA
    formulation (bias reflects class prior/frequency, not representation
    scale, so rescaling it is not part of the correction).
    """
    with torch.no_grad():
        W = model.classifier.weight
        row_norms = W.norm(dim=1)
        target_norm = float(row_norms.mean().item())

        for step_idx in range(NUM_STEPS):
            idx = torch.tensor(
                list(classes_for_step(step_idx)),
                device=W.device,
                dtype=torch.long,
            )
            group_norm = float(row_norms[idx].mean().clamp_min(eps).item())
            scale = target_norm / group_norm
            W[idx] *= scale

    return model

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
        lambda_orth=0.0,
        orth_mode="none",
        teacher_model=None,
        kd_weight=0.0,
        kd_temperature=2.0,
        method_name="unknown",
        step_idx=-1,
        orth_eps=1e-12,
        log_every_steps=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_weights = reference_weights or {}
        self.factor_reference_state = factor_reference_state
        self.lambda_orth = float(lambda_orth)
        self.orth_mode = "none" if orth_mode is None else str(orth_mode)
        self.teacher_model = teacher_model
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
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
        effective_lambda_orth = float(self.lambda_orth) * combined_warmup_multiplier
        weighted_orth = effective_lambda_orth * orth_loss_used

        if teacher_active:
            if not self._teacher_ready:
                self.teacher_model.to(device=ce_loss.device)
                self.teacher_model.eval()
                self._teacher_ready = True
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits.detach()
            student_log_probs = F.log_softmax(outputs.logits / self.kd_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.kd_temperature, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.kd_temperature ** 2)

        weighted_kd = float(self.kd_weight) * kd_loss
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
            "lambda_orth_warmup_multiplier": float(combined_warmup_multiplier),
            "lambda_orth_times_loss": weighted_orth_v,
            "orth_ratio_abs_weighted_over_ce": abs(weighted_orth_v) / (ce_v + float(self.orth_eps)),
            "weighted_orth_over_CE": abs(weighted_orth_v) / (ce_v + float(self.orth_eps)),
            "orth_mode": self.orth_mode,
            "num_layers_used": int(factor_comps["num_layers"] if self.orth_mode == "factor_orth" else len(self.reference_weights)),
            "old_norm_mean": np.nan,
            "new_norm_mean": np.nan,
            "factor_A_penalty_mean": float(factor_comps["factor_A_mean"].detach().cpu().item()),
            "factor_B_penalty_mean": float(factor_comps["factor_B_mean"].detach().cpu().item()),
            "factor_total_penalty_mean": float(factor_comps["factor_total_mean"].detach().cpu().item()),
            "weighted_factor_orth_mean": float((effective_lambda_orth * factor_comps["factor_total_mean"]).detach().cpu().item()),
            "weighted_factor_orth_over_CE": abs(float((effective_lambda_orth * factor_comps["factor_total_mean"]).detach().cpu().item())) / (ce_v + float(self.orth_eps)),
            "mean_A_overlap": float(factor_comps["mean_A_overlap"].detach().cpu().item()),
            "mean_B_overlap": float(factor_comps["mean_B_overlap"].detach().cpu().item()),
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
        if use_kd and len(step_states) > 0:
            teacher_model = build_simple_avg_teacher_model(step_states)
            assert not any(p.requires_grad for p in teacher_model.parameters())
        if use_orth and active_orth_mode == "factor_orth" and len(step_states) > 0:
            factor_reference_state = average_factor_reference_state(step_states)
        if use_orth and active_orth_mode == "delta_trace" and len(step_states) > 0:
            reference_weights = average_delta_reference_state(step_states)

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
            "lambda_orth": resolved_lambda_orth,
            "orth_mode": "none" if not use_orth else active_orth_mode,
            "teacher_model": teacher_model,
            "kd_weight": float(kd_weight) if teacher_model is not None else 0.0,
            "kd_temperature": float(kd_temperature),
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


def run_simple_avg_variant(method_name):
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    step_states, specialist_diagonal_accuracy = train_independent_loras(
        method_name=method_name,
        method_prefix=f"{method_name}_source",
        replay_per_class=0,
        use_orth=bool(method_cfg["uses_delta_trace"] or method_cfg["uses_factor_orth"]),
        orth_mode=(
            "delta_trace"
            if method_cfg["uses_delta_trace"]
            else ("factor_orth" if method_cfg["uses_factor_orth"] else None)
        ),
        use_kd=bool(method_cfg["uses_kd"]),
        kd_weight=float(method_cfg["kd_weight"]),
        kd_temperature=float(method_cfg["kd_temperature"]),
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
        merged_model = calibrate_classifier_row_norms(merged_model)

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


simple_avg_execution_order = [cfg["method"] for cfg in ACTIVE_METHOD_CONFIGS if cfg["family"] == "simple_avg"]
for method_name in simple_avg_execution_order:
    base_method = ACTIVE_METHOD_MAP[method_name]["base_method"]
    if METHODS_TO_RUN.get(base_method, False):
        run_simple_avg_variant(method_name)
    else:
        print(f"Skipping {method_name} because {base_method} is disabled")

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
            out = out + self.scaling * lora_new

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
    rows = snapshot["rows"]
    if len(rows) == 0:
        return
    with torch.no_grad():
        row_idx = torch.tensor(rows, device=model.classifier.weight.device, dtype=torch.long)
        model.classifier.weight[row_idx].copy_(
            snapshot["weight"][rows].to(
                device=model.classifier.weight.device,
                dtype=model.classifier.weight.dtype,
            )
        )
        model.classifier.bias[row_idx].copy_(
            snapshot["bias"][rows].to(
                device=model.classifier.bias.device,
                dtype=model.classifier.bias.dtype,
            )
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


class RankExtensionTrainer(HeadLRTrainerMixin, Trainer):
    def __init__(self, *args, classifier_snapshot=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_snapshot = classifier_snapshot
        if classifier_snapshot is not None:
            self.add_callback(ClassifierRowRestoreCallback(classifier_snapshot))


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
        self._rows = []
        self._teacher_ready = False

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
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

        if self.orth_mode != "none":
            comps = compute_delta_orth_components(model=model, eps=self.orth_eps)
            raw_inner = comps["trace_mean"]
            abs_inner = comps["trace_abs_mean"]
            norm_sq = comps["norm_sq_mean"]

        if self.orth_mode == "none":
            orth_loss_used = zero
        elif self.orth_mode in ["trace", "trace_abs", "delta_trace", "abs_trace"]:
            orth_loss_used = abs_inner
        elif self.orth_mode == "norm":
            orth_loss_used = norm_sq
        elif self.orth_mode == "factor_orth":
            orth_loss_used = comps["factor_total_mean"]
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

        if teacher_active:
            if not self._teacher_ready:
                self.teacher_model.to(device=ce_loss.device)
                self.teacher_model.eval()
                self._teacher_ready = True
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits.detach()
            student_log_probs = F.log_softmax(outputs.logits / self.kd_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.kd_temperature, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.kd_temperature ** 2)

        weighted_kd = float(self.kd_weight) * kd_loss
        loss = ce_loss + weighted + weighted_kd

        ce_v = float(ce_loss.detach().cpu().item())
        raw_inner_v = float(raw_inner.detach().cpu().item())
        abs_inner_v = float(abs_inner.detach().cpu().item())
        norm_sq_v = float(norm_sq.detach().cpu().item())
        orth_used_v = float(orth_loss_used.detach().cpu().item())
        weighted_v = float(weighted.detach().cpu().item())
        kd_loss_v = float(kd_loss.detach().cpu().item())
        weighted_kd_v = float(weighted_kd.detach().cpu().item())
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
            "total_loss": total_loss_v,
            "effective_lambda": float(effective_lambda_orth),
        }
        self._rows.append(row)

        if len(self._rows) % self.log_every_steps == 0:
            print(
                f"[orth train] method={self.method_name} | step={row['step']} | epoch={row['epoch']:.4f} | "
                f"ce={row['ce_loss']:.6f} | orth={row['orth_loss_used']:.6f} | "
                f"kd={row['kd_loss']:.6f} | total={row['total_loss']:.6f} | "
                f"lambda={row['lambda_orth']:.6g} (warmup x{row['lambda_orth_warmup_multiplier']:.3g}) | "
                f"kd_weight={row['kd_weight']:.6g} | ratio={row['orth_ratio_abs_weighted_over_ce']:.6f}"
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

    for step_idx in range(NUM_STEPS):
        current_classes = classes_for_step(step_idx)
        trainable_classifier_classes = rank_extension_trainable_classifier_classes(
            step_idx=step_idx,
            replay_per_class=replay_per_class,
        )

        old_active_in_forward = not (zero_old_merge and step_idx > 0)
        teacher_model = None
        if use_kd and previous_rank_state is not None:
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
            "kd_weight": active_kd_weight if teacher_model is not None else 0.0,
            "kd_temperature": active_kd_temperature,
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
        final_rank_model = calibrate_classifier_row_norms(final_rank_model)

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
    cleanup()


orth_kd_eval_rows = []
orth_kd_train_rows = train_diagnostic_rows if "train_diagnostic_rows" in globals() else []
orth_kd_summary_rows = method_summary_rows if "method_summary_rows" in globals() else []

rank_extension_execution_order = [cfg["method"] for cfg in ACTIVE_METHOD_CONFIGS if cfg["family"] == "rank_extension"]
for method_name in rank_extension_execution_order:
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    base_method = method_cfg["base_method"]
    if not METHODS_TO_RUN.get(base_method, False):
        print(f"Skipping {method_name} because {base_method} is disabled")
        continue

    run_rank_extension_variant(
        method_name=method_name,
        replay_per_class=0,
        use_orth=bool(method_cfg["uses_delta_trace"] or method_cfg["uses_factor_orth"]),
        orth_mode=(
            "delta_trace"
            if method_cfg["uses_delta_trace"]
            else ("factor_orth" if method_cfg["uses_factor_orth"] else None)
        ),
        lambda_orth=float(method_cfg["lambda_orth"]),
        zero_old_merge=False,
        use_kd=bool(method_cfg["uses_kd"]),
        kd_weight=float(method_cfg["kd_weight"]),
        kd_temperature=float(method_cfg["kd_temperature"]),
        orth_eval_records=orth_kd_eval_rows,
        orth_train_records=orth_kd_train_rows,
        orth_summary_records=orth_kd_summary_rows,
    )

if len(orth_kd_train_rows) > 0:
    print(f"[rank_extension] accumulated training-loss rows: {len(orth_kd_train_rows)}")


# In[ ]:


if METHODS_TO_RUN["joint_upper_bound"]:
    joint_model = fresh_pretrained_model()

    train_joint = make_joint_train_dataset()
    test_joint = make_joint_eval_dataset()

    print("\n===== joint_upper_bound =====")

    train_with_trainer(
        model=joint_model,
        train_ds=train_joint,
        eval_ds=test_joint,
        output_dir=os.path.join(MODELS_DIR, "joint_upper_bound"),
        epochs=JOINT_EPOCHS,
        lr=LR_JOINT,
        batch_size=BATCH_FT,
        accum_steps=ACCUM_FT,
    )

    joint_eval_rows = evaluate_model(joint_model, "joint_upper_bound")
    joint_eval_map = {row["eval_set"]: float(row["accuracy"]) for row in joint_eval_rows}
    method_summary_rows.append({
        "method": "joint_upper_bound",
        "orth_mode": "none",
        "lambda_orth": 0.0,
        "zero_old_merge": False,
        "use_kd": False,
        "kd_weight": 0.0,
        "kd_temperature": 0.0,
        "replay_per_class": 0,
        "old_active_in_forward": np.nan,
        "first_step": joint_eval_map.get("first_step", np.nan),
        "later_steps": joint_eval_map.get("later_steps", np.nan),
        "all_seen": joint_eval_map.get("all_seen", np.nan),
        "old_new_gap": joint_eval_map.get("first_step", np.nan) - joint_eval_map.get("later_steps", np.nan),
        "avg_forgetting": np.nan,
    })

    del joint_model
    cleanup()

else:
    print("Skipping joint_upper_bound")


# In[ ]:


results_df = pd.DataFrame(all_results)

results_path = os.path.join(TABLES_DIR, "all_results_clip_vit_full_comparison.csv")
results_df.to_csv(results_path, index=False)

print("Saved:", results_path)
results_df


# In[ ]:


active_method_order = list(ACTIVE_METHOD_NAMES)


def gain_vs(summary_lookup, method_name, ref_name):
    if (not ref_name) or method_name not in summary_lookup.index or ref_name not in summary_lookup.index:
        return np.nan
    return float(summary_lookup.loc[method_name, "all_seen"] - summary_lookup.loc[ref_name, "all_seen"])


def kd_reference_method(method_name):
    cfg = ACTIVE_METHOD_MAP[method_name]
    if not cfg["uses_kd"]:
        return ""
    if cfg["family"] == "simple_avg":
        if cfg["uses_delta_trace"]:
            return "simple_avg_delta_orth"
        if cfg["uses_factor_orth"]:
            return "simple_avg_factor_orth"
        return "simple_avg"
    if cfg["uses_delta_trace"]:
        return "rank_extension_orth_delta_trace_lam_50"
    if cfg["uses_factor_orth"]:
        return "rank_extension_orth_factor_lam_50"
    return "rank_extension"


def orth_reference_method(method_name):
    cfg = ACTIVE_METHOD_MAP[method_name]
    if not (cfg["uses_delta_trace"] or cfg["uses_factor_orth"]):
        return ""
    if cfg["family"] == "simple_avg":
        if cfg["uses_kd"]:
            return f"simple_avg_kd_{kd_temperature_tag(cfg['kd_temperature'])}"
        return "simple_avg"
    if cfg["uses_kd"]:
        return f"rank_extension_kd_only_{kd_temperature_tag(cfg['kd_temperature'])}"
    return "rank_extension"


def epoch_bucket(epoch_value):
    if pd.isna(epoch_value):
        return np.nan
    return int(max(1, math.ceil(float(epoch_value) - 1e-12)))


def safe_ratio(numer, denom, eps=1e-12):
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.zeros_like(numer, dtype=float)
    mask = np.abs(denom) > float(eps)
    out[mask] = numer[mask] / denom[mask]
    return out


def rankext_new_rank_per_step_string():
    schedule = [int(v) for v in active_rankext_rank_schedule()]
    new_blocks = [schedule[0]] + [schedule[i] - schedule[i - 1] for i in range(1, len(schedule))]
    return "->".join(str(v) for v in new_blocks)


method_config_df = pd.DataFrame(ACTIVE_METHOD_CONFIGS).copy()
method_config_df["display_name"] = method_config_df["method"].map(METHOD_DISPLAY_NAME_MAP).fillna(method_config_df["method"])
method_config_df["supervisor_requested_name"] = method_config_df["method"].map(METHOD_ALIAS_NAME_MAP).fillna(method_config_df["method"])
method_config_df["orth_type"] = np.where(
    method_config_df["uses_delta_trace"],
    "delta_trace",
    np.where(method_config_df["uses_factor_orth"], "factor_orth", "none"),
)
method_config_df["lambda_delta_trace"] = np.where(method_config_df["uses_delta_trace"], method_config_df["lambda_orth"], 0.0)
method_config_df["lambda_factor_orth"] = np.where(method_config_df["uses_factor_orth"], method_config_df["lambda_orth"], 0.0)
method_config_df["kd_enabled"] = method_config_df["uses_kd"].astype(bool)
method_config_df["replay_per_class"] = 0
method_config_df["zero_old_merge"] = False
method_config_df["rankext_new_rank_per_step"] = np.where(
    method_config_df["family"] == "rank_extension",
    rankext_new_rank_per_step_string(),
    "",
)
method_config_df["lora_rank_or_current_rank"] = np.where(
    method_config_df["family"] == "simple_avg",
    method_config_df["rank"].astype(int).astype(str),
    method_config_df["rank_schedule"],
)
method_config_df["lora_alpha"] = float(LORA_ALPHA)
method_config_df["internal_method_name"] = method_config_df["method"]

supervisor_method_mapping_df = pd.DataFrame(SUPERVISOR_SELECTED_METHOD_SPECS)
supervisor_method_mapping_path = os.path.join(TABLES_DIR, "supervisor_method_mapping.csv")
supervisor_method_mapping_df.to_csv(supervisor_method_mapping_path, index=False)
print("Saved supervisor method mapping:", supervisor_method_mapping_path)

results_df = pd.DataFrame(all_results)
results_df = results_df[results_df["method"].isin(active_method_order)].copy() if len(results_df) > 0 else pd.DataFrame(columns=["method", "eval_set", "accuracy", "loss"])
results_path = os.path.join(TABLES_DIR, "all_results_selected_methods.csv")
results_df.to_csv(results_path, index=False)
print("Saved raw selected-method results:", results_path)

method_summary_df = pd.DataFrame(method_summary_rows)
if len(method_summary_df) > 0:
    method_summary_df = method_summary_df[method_summary_df["method"].isin(active_method_order)].copy()
    method_summary_df = method_summary_df.drop_duplicates(subset=["method"], keep="last")
else:
    method_summary_df = pd.DataFrame(columns=["method", "first_step", "later_steps", "all_seen", "old_new_gap", "avg_forgetting", "old_active_in_forward", "backward_transfer", "forward_transfer"])

train_diag_df = pd.DataFrame(train_diagnostic_rows)
epoch_val_df = pd.DataFrame(epoch_loss_rows)
loss_fill_note = "Non-applicable auxiliary losses are logged as NaN; unavailable validation total loss is logged as NaN."

loss_component_cols = [
    "train_ce_loss",
    "train_total_loss",
    "train_kd_loss_raw",
    "train_kd_loss_weighted",
    "train_delta_trace_loss_raw",
    "train_delta_trace_loss_weighted",
    "train_factor_orth_loss_raw",
    "train_factor_orth_loss_weighted",
    "train_delta_trace_weighted_over_ce",
    "train_factor_orth_weighted_over_ce",
    "train_kd_weighted_over_ce",
]

if len(train_diag_df) > 0:
    train_diag_df = train_diag_df[train_diag_df["method"].isin(active_method_order)].copy()
    train_diag_df = train_diag_df.merge(
        method_config_df[[
            "method",
            "display_name",
            "family",
            "base_method",
            "uses_kd",
            "kd_temperature",
            "kd_weight",
            "uses_delta_trace",
            "uses_factor_orth",
            "lambda_orth",
            "lambda_delta_trace",
            "lambda_factor_orth",
            "orth_type",
            "rank",
            "rank_schedule",
            "target_modules",
        ]],
        on="method",
        how="left",
        suffixes=("", "_cfg"),
    )
    train_diag_df["step_id"] = train_diag_df["step"].astype(int)
    train_diag_df["epoch_id"] = train_diag_df["epoch"].apply(epoch_bucket).astype(int)
    train_diag_df["train_ce_loss"] = train_diag_df["ce_loss"].fillna(np.nan)
    train_diag_df["train_total_loss"] = train_diag_df["total_loss"].fillna(np.nan)
    train_diag_df["train_kd_loss_raw"] = np.where(train_diag_df["uses_kd"], train_diag_df["kd_loss"], np.nan)
    train_diag_df["train_kd_loss_weighted"] = np.where(train_diag_df["uses_kd"], train_diag_df["weighted_kd_loss"], np.nan)
    train_diag_df["train_delta_trace_loss_raw"] = np.where(train_diag_df["uses_delta_trace"], train_diag_df["orth_loss"], np.nan)
    train_diag_df["train_delta_trace_loss_weighted"] = np.where(train_diag_df["uses_delta_trace"], train_diag_df["lambda_orth_times_loss"], np.nan)
    train_diag_df["train_factor_orth_loss_raw"] = np.where(train_diag_df["uses_factor_orth"], train_diag_df["factor_total_penalty_mean"], np.nan)
    train_diag_df["train_factor_orth_loss_weighted"] = np.where(train_diag_df["uses_factor_orth"], train_diag_df["weighted_factor_orth_mean"], np.nan)
    train_diag_df["train_kd_weighted_over_ce"] = safe_ratio(train_diag_df["train_kd_loss_weighted"], train_diag_df["train_ce_loss"])
    train_diag_df["train_delta_trace_weighted_over_ce"] = safe_ratio(train_diag_df["train_delta_trace_loss_weighted"], train_diag_df["train_ce_loss"])
    train_diag_df["train_factor_orth_weighted_over_ce"] = safe_ratio(train_diag_df["train_factor_orth_loss_weighted"], train_diag_df["train_ce_loss"])
    train_diag_df = train_diag_df.sort_values(["method", "step_id", "epoch", "epoch_id"]).reset_index(drop=True)

    train_epoch_df = (
        train_diag_df.groupby(
            [
                "method",
                "display_name",
                "family",
                "base_method",
                "step_id",
                "epoch_id",
                "uses_kd",
                "kd_temperature",
                "kd_weight",
                "uses_delta_trace",
                "uses_factor_orth",
                "lambda_orth",
                "lambda_delta_trace",
                "lambda_factor_orth",
                "orth_type",
                "rank",
                "rank_schedule",
                "target_modules",
            ],
            as_index=False,
        )[loss_component_cols]
        .mean()
        .rename(columns={"method": "method_name", "epoch_id": "epoch"})
    )
else:
    train_diag_df = pd.DataFrame(columns=["method", "step", "epoch"])
    train_epoch_df = pd.DataFrame(columns=[
        "method_name",
        "display_name",
        "family",
        "base_method",
        "step_id",
        "epoch",
        "uses_kd",
        "kd_temperature",
        "kd_weight",
        "uses_delta_trace",
        "uses_factor_orth",
        "lambda_orth",
        "lambda_delta_trace",
        "lambda_factor_orth",
        "orth_type",
        "rank",
        "rank_schedule",
        "target_modules",
    ] + loss_component_cols)

if len(epoch_val_df) > 0:
    epoch_val_df = epoch_val_df.copy()
    epoch_val_df["step_id"] = epoch_val_df["step_id"].astype(int)
    epoch_val_df["epoch"] = epoch_val_df["epoch"].astype(int)
else:
    epoch_val_df = pd.DataFrame(columns=[
        "method_name",
        "display_name",
        "step_id",
        "epoch",
        "val_ce_loss",
        "val_total_loss",
        "learning_rate",
    ])

training_loss_history_df = train_epoch_df.merge(
    epoch_val_df,
    on=["method_name", "display_name", "step_id", "epoch"],
    how="left",
)
if "learning_rate" not in training_loss_history_df.columns:
    training_loss_history_df["learning_rate"] = np.nan
if "val_ce_loss" not in training_loss_history_df.columns:
    training_loss_history_df["val_ce_loss"] = np.nan
if "val_total_loss" not in training_loss_history_df.columns:
    training_loss_history_df["val_total_loss"] = np.nan
training_loss_history_df = training_loss_history_df.sort_values(["method_name", "step_id", "epoch"]).reset_index(drop=True)

training_loss_history_path = os.path.join(TABLES_DIR, "training_loss_history_by_epoch.csv")
training_loss_history_df.to_csv(training_loss_history_path, index=False)
print("Saved training loss history:", training_loss_history_path)
print(loss_fill_note)

# PRE-THESIS FIX 1: which epoch's weights were actually kept per (method, step)
# after best-epoch (val-CE) selection -- see USE_BEST_EPOCH_SELECTION comment and
# train_with_trainer(). One row per (method, step) that went through
# train_with_trainer with an eval_ds (i.e. every step of every active method).
best_epoch_selection_df = pd.DataFrame(best_epoch_selection_rows)
if len(best_epoch_selection_df) > 0:
    best_epoch_selection_df = best_epoch_selection_df[
        best_epoch_selection_df["method_name"].isin(active_method_order)
    ].copy()
    best_epoch_selection_df = best_epoch_selection_df.sort_values(
        ["method_name", "step_id"]
    ).reset_index(drop=True)
else:
    best_epoch_selection_df = pd.DataFrame(columns=[
        "method_name", "display_name", "step_id", "epochs_configured",
        "best_epoch_selection_enabled", "selected_epoch", "selected_val_ce",
        "final_epoch_val_ce", "selected_epoch_lt_final",
    ])
best_epoch_selection_path = os.path.join(TABLES_DIR, "best_epoch_selected_by_method_step.csv")
best_epoch_selection_df.to_csv(best_epoch_selection_path, index=False)
print("Saved best-epoch selection log:", best_epoch_selection_path)
if len(best_epoch_selection_df) > 0:
    n_selected_lt_final = int(best_epoch_selection_df["selected_epoch_lt_final"].sum())
    print(
        f"[best-epoch selection] {n_selected_lt_final}/{len(best_epoch_selection_df)} "
        f"(method, step) pairs selected an epoch earlier than the configured final epoch."
    )

loss_summary_rows = []
for method_name in active_method_order:
    method_cfg = ACTIVE_METHOD_MAP[method_name]
    display_name = METHOD_DISPLAY_NAME_MAP.get(method_name, method_name)
    method_rows = training_loss_history_df[training_loss_history_df["method_name"] == method_name].copy()

    def _mean(col_name):
        if len(method_rows) == 0 or col_name not in method_rows.columns:
            return np.nan
        return float(method_rows[col_name].mean())

    def _final(col_name):
        if len(method_rows) == 0 or col_name not in method_rows.columns:
            return np.nan
        return float(method_rows[col_name].iloc[-1])

    loss_summary_rows.append({
        "method_name": method_name,
        "display_name": display_name,
        "family": method_cfg["family"],
        "mean_train_ce_loss": _mean("train_ce_loss"),
        "final_train_ce_loss": _final("train_ce_loss"),
        "mean_val_ce_loss": _mean("val_ce_loss"),
        "final_val_ce_loss": _final("val_ce_loss"),
        "mean_train_total_loss": _mean("train_total_loss"),
        "final_train_total_loss": _final("train_total_loss"),
        "mean_val_total_loss": _mean("val_total_loss"),
        "final_val_total_loss": _final("val_total_loss"),
        "mean_train_kd_loss_raw": _mean("train_kd_loss_raw"),
        "final_train_kd_loss_raw": _final("train_kd_loss_raw"),
        "mean_train_kd_loss_weighted": _mean("train_kd_loss_weighted"),
        "final_train_kd_loss_weighted": _final("train_kd_loss_weighted"),
        "mean_train_factor_orth_loss_raw": _mean("train_factor_orth_loss_raw"),
        "final_train_factor_orth_loss_raw": _final("train_factor_orth_loss_raw"),
        "mean_train_factor_orth_loss_weighted": _mean("train_factor_orth_loss_weighted"),
        "final_train_factor_orth_loss_weighted": _final("train_factor_orth_loss_weighted"),
        "mean_train_delta_trace_loss_raw": _mean("train_delta_trace_loss_raw"),
        "final_train_delta_trace_loss_raw": _final("train_delta_trace_loss_raw"),
        "mean_train_delta_trace_loss_weighted": _mean("train_delta_trace_loss_weighted"),
        "final_train_delta_trace_loss_weighted": _final("train_delta_trace_loss_weighted"),
        "mean_kd_weighted_over_ce": _mean("train_kd_weighted_over_ce"),
        "mean_factor_orth_weighted_over_ce": _mean("train_factor_orth_weighted_over_ce"),
        "mean_delta_trace_weighted_over_ce": _mean("train_delta_trace_weighted_over_ce"),
        "logged_epoch_rows": int(len(method_rows)),
    })

loss_summary_by_method_df = pd.DataFrame(loss_summary_rows)
loss_summary_by_method_path = os.path.join(TABLES_DIR, "loss_summary_by_method.csv")
loss_summary_by_method_df.to_csv(loss_summary_by_method_path, index=False)
print("Saved loss summary:", loss_summary_by_method_path)

loss_components_summary_df = method_config_df[[
    "method",
    "display_name",
    "family",
    "base_method",
    "uses_kd",
    "kd_temperature",
    "kd_weight",
    "uses_delta_trace",
    "uses_factor_orth",
    "lambda_orth",
    "lambda_delta_trace",
    "lambda_factor_orth",
    "orth_type",
    "rank",
    "rank_schedule",
    "target_modules",
]].merge(
    loss_summary_by_method_df,
    left_on=["method", "display_name", "family"],
    right_on=["method_name", "display_name", "family"],
    how="left",
)
loss_components_summary_df["ce_loss_mean"] = loss_components_summary_df["mean_train_ce_loss"]
loss_components_summary_df["ce_loss_final"] = loss_components_summary_df["final_train_ce_loss"]
loss_components_summary_df["val_ce_loss_mean"] = loss_components_summary_df["mean_val_ce_loss"]
loss_components_summary_df["val_ce_loss_final"] = loss_components_summary_df["final_val_ce_loss"]
loss_components_summary_df["total_loss_mean"] = loss_components_summary_df["mean_train_total_loss"]
loss_components_summary_df["total_loss_final"] = loss_components_summary_df["final_train_total_loss"]
loss_components_summary_df["kd_loss_raw_mean"] = loss_components_summary_df["mean_train_kd_loss_raw"]
loss_components_summary_df["kd_loss_raw_final"] = loss_components_summary_df["final_train_kd_loss_raw"]
loss_components_summary_df["kd_loss_weighted_mean"] = loss_components_summary_df["mean_train_kd_loss_weighted"]
loss_components_summary_df["kd_loss_weighted_final"] = loss_components_summary_df["final_train_kd_loss_weighted"]
loss_components_summary_df["factor_orth_loss_raw_mean"] = loss_components_summary_df["mean_train_factor_orth_loss_raw"]
loss_components_summary_df["factor_orth_loss_raw_final"] = loss_components_summary_df["final_train_factor_orth_loss_raw"]
loss_components_summary_df["factor_orth_loss_weighted_mean"] = loss_components_summary_df["mean_train_factor_orth_loss_weighted"]
loss_components_summary_df["factor_orth_loss_weighted_final"] = loss_components_summary_df["final_train_factor_orth_loss_weighted"]
loss_components_summary_df["delta_trace_loss_raw_mean"] = loss_components_summary_df["mean_train_delta_trace_loss_raw"]
loss_components_summary_df["delta_trace_loss_raw_final"] = loss_components_summary_df["final_train_delta_trace_loss_raw"]
loss_components_summary_df["delta_trace_loss_weighted_mean"] = loss_components_summary_df["mean_train_delta_trace_loss_weighted"]
loss_components_summary_df["delta_trace_loss_weighted_final"] = loss_components_summary_df["final_train_delta_trace_loss_weighted"]
loss_components_summary_df["kd_over_CE_mean"] = loss_components_summary_df["mean_kd_weighted_over_ce"]
loss_components_summary_df["delta_trace_over_CE_mean"] = loss_components_summary_df["mean_delta_trace_weighted_over_ce"]
loss_components_summary_df["factor_orth_over_CE_mean"] = loss_components_summary_df["mean_factor_orth_weighted_over_ce"]
loss_components_summary_path = os.path.join(TABLES_DIR, "loss_components_summary_by_method.csv")
loss_components_summary_df.to_csv(loss_components_summary_path, index=False)
print("Saved loss-component summary:", loss_components_summary_path)

final_table = results_df.pivot_table(index="method", columns="eval_set", values="accuracy", aggfunc="mean") if len(results_df) > 0 else pd.DataFrame(index=active_method_order)
for col in ["first_step", "later_steps", "all_seen"]:
    if col not in final_table.columns:
        final_table[col] = np.nan
final_table = final_table[["first_step", "later_steps", "all_seen"]]
final_table = final_table.reindex(active_method_order)
final_table_percent = (final_table * 100.0).reset_index().rename(columns={"index": "method"})
final_table_percent["old_new_gap"] = final_table_percent["first_step"] - final_table_percent["later_steps"]

summary_table = method_config_df.merge(final_table_percent, on="method", how="left")
summary_table = summary_table.merge(
    method_summary_df[[
        col for col in
        ["method", "avg_forgetting", "old_active_in_forward", "backward_transfer", "forward_transfer"]
        if col in method_summary_df.columns
    ]],
    on="method",
    how="left",
)
summary_table = summary_table.merge(
    loss_summary_by_method_df[[
        "method_name",
        "mean_train_ce_loss",
        "final_train_ce_loss",
        "mean_val_ce_loss",
        "final_val_ce_loss",
        "mean_train_total_loss",
        "final_train_total_loss",
        "mean_val_total_loss",
        "final_val_total_loss",
        "mean_kd_weighted_over_ce",
        "mean_factor_orth_weighted_over_ce",
        "mean_delta_trace_weighted_over_ce",
    ]],
    left_on="method",
    right_on="method_name",
    how="left",
)
if "method_name" in summary_table.columns:
    summary_table = summary_table.drop(columns=["method_name"])
summary_table["internal_method_name"] = summary_table["method"]
summary_table["factor_lambda"] = summary_table["lambda_factor_orth"]
summary_table["delta_trace_lambda"] = summary_table["lambda_delta_trace"]
summary_table["kd_enabled"] = summary_table["uses_kd"].astype(bool)
summary_table["replay_per_class"] = summary_table["replay_per_class"].fillna(0).astype(int)
summary_table["zero_old_merge"] = summary_table["zero_old_merge"].astype(bool)

simple_all_seen = float(summary_table.loc[summary_table["method"] == "simple_avg", "all_seen"].iloc[0]) if (summary_table["method"] == "simple_avg").any() else np.nan
summary_table["delta_vs_simple_avg"] = summary_table["all_seen"] - simple_all_seen
summary_table["rank_extension_minus_simple_avg"] = np.where(summary_table["family"] == "rank_extension", summary_table["all_seen"] - simple_all_seen, np.nan)

ranking_table = summary_table.sort_values(["all_seen", "first_step"], ascending=[False, False]).copy()
ranking_table["rank_all_seen"] = np.arange(1, len(ranking_table) + 1)
summary_table = summary_table.merge(ranking_table[["method", "rank_all_seen"]], on="method", how="left")
summary_lookup = summary_table.set_index("method")
summary_table["kd_gain"] = [gain_vs(summary_lookup, method_name, kd_reference_method(method_name)) for method_name in summary_table["method"]]
summary_table["orth_gain"] = [gain_vs(summary_lookup, method_name, orth_reference_method(method_name)) for method_name in summary_table["method"]]
summary_table["replay_gain"] = np.nan
joint_rows = summary_table[summary_table["base_method"] == "joint_upper_bound"]
joint_all_seen = float(joint_rows["all_seen"].iloc[0]) if len(joint_rows) > 0 else np.nan
summary_table["gap_to_joint_upper_bound"] = joint_all_seen - summary_table["all_seen"] if not np.isnan(joint_all_seen) else np.nan

final_accuracy_path = os.path.join(TABLES_DIR, "final_accuracy_selected_methods.csv")
ranking_table_path = os.path.join(TABLES_DIR, "ranking_by_all_seen_selected_methods.csv")
summary_table_path = os.path.join(TABLES_DIR, "summary_metrics_selected_methods.csv")
method_metadata_path = os.path.join(TABLES_DIR, "method_run_metadata_selected_methods.csv")

final_accuracy_df = final_table_percent[["method", "first_step", "later_steps", "all_seen", "old_new_gap"]].copy()
final_accuracy_df.to_csv(final_accuracy_path, index=False)
ranking_table[["method", "rank_all_seen", "first_step", "later_steps", "all_seen", "old_new_gap"]].to_csv(ranking_table_path, index=False)
summary_table.to_csv(summary_table_path, index=False)

# =============================================================================
# Task 2: FINAL (authoritative) convergence tables.
#
# During the run, refresh_live_convergence() kept overwriting
# tables/all_methods_convergence_table.csv and a *provisional*
# tables/top2_convergence_table.csv (ranked by lowest mean val CE so far, since
# true final accuracy wasn't known yet -- see the comment block above
# refresh_live_convergence()). Now that the whole run is done and ranking_table
# has the true rank_all_seen accuracy ranking, overwrite both with the
# authoritative version built from training_loss_history_df (the merged
# train+val CE table already assembled above), and re-rank the top-2 table by
# true final (all_seen) accuracy instead of the mid-run val-CE proxy.
# =============================================================================
if len(training_loss_history_df) > 0:
    _final_all_methods_conv = training_loss_history_df[
        ["method_name", "display_name", "step_id", "epoch", "train_ce_loss", "val_ce_loss"]
    ].sort_values(["method_name", "step_id", "epoch"]).reset_index(drop=True)
    _final_all_methods_conv.to_csv(os.path.join(TABLES_DIR, "all_methods_convergence_table.csv"), index=False)

    _true_top2_methods = ranking_table.sort_values("rank_all_seen")["method"].head(2).tolist()
    _final_top2_conv = _final_all_methods_conv[_final_all_methods_conv["method_name"].isin(_true_top2_methods)]
    _final_top2_conv.to_csv(os.path.join(TABLES_DIR, "top2_convergence_table.csv"), index=False)
    print("Finalized authoritative convergence tables. True top-2 by all_seen accuracy:", _true_top2_methods)

    # Also regenerate the two top-2 methods' live convergence plots one last time
    # so the PNGs reflect the complete, final run rather than whatever step they
    # last happened to be redrawn at.
    for _m in _true_top2_methods:
        refresh_live_convergence(_m)

method_run_metadata_cols = [
    "method",
    "display_name",
    "supervisor_requested_name",
    "family",
    "base_method",
    "uses_kd",
    "kd_temperature",
    "kd_weight",
    "uses_delta_trace",
    "uses_factor_orth",
    "lambda_orth",
    "lambda_delta_trace",
    "lambda_factor_orth",
    "uses_replay",
    "uses_zero_old",
    "rank",
    "rank_schedule",
    "target_modules",
    "first_step",
    "later_steps",
    "all_seen",
    "old_new_gap",
    "avg_forgetting",
]
method_run_metadata_df = summary_table[method_run_metadata_cols].copy()
method_run_metadata_df.to_csv(method_metadata_path, index=False)

supervisor_selected_accuracy_df = summary_table[summary_table["method"].isin(SUPERVISOR_SELECTED_INTERNAL_METHODS)].copy()
supervisor_selected_accuracy_df["method"] = pd.Categorical(
    supervisor_selected_accuracy_df["method"],
    categories=SUPERVISOR_SELECTED_INTERNAL_METHODS,
    ordered=True,
)
supervisor_selected_accuracy_df = supervisor_selected_accuracy_df.sort_values("method").reset_index(drop=True)
supervisor_selected_accuracy_df["forgetting_gap"] = supervisor_selected_accuracy_df["avg_forgetting"]
supervisor_selected_accuracy_export_df = supervisor_selected_accuracy_df[[
    "method",
    "display_name",
    "family",
    "lambda_factor_orth",
    "kd_temperature",
    "kd_weight",
    "first_step",
    "later_steps",
    "all_seen",
    "forgetting_gap",
]].rename(columns={
    "method": "internal_method_name",
    "lambda_factor_orth": "factor_lambda",
})
supervisor_selected_accuracy_export_df["seed"] = int(SEED)  # STRICT-REVIEW ADD (B4)
supervisor_selected_accuracy_path = os.path.join(TABLES_DIR, "supervisor_selected_accuracy_comparison.csv")
supervisor_selected_accuracy_export_df.to_csv(supervisor_selected_accuracy_path, index=False)
print("Saved supervisor-selected accuracy comparison:", supervisor_selected_accuracy_path)

method_hyperparameter_summary_df = summary_table[[
    "method",
    "display_name",
    "family",
    "orth_type",
    "lambda_delta_trace",
    "lambda_factor_orth",
    "kd_enabled",
    "kd_temperature",
    "kd_weight",
    "replay_per_class",
    "rankext_new_rank_per_step",
    "lora_rank_or_current_rank",
    "lora_alpha",
    "target_modules",
    "zero_old_merge",
    "old_active_in_forward",
    "supervisor_requested_name",
]].rename(columns={
    "method": "method_name",
})
method_hyperparameter_summary_df["seed"] = int(SEED)  # STRICT-REVIEW ADD (B4)
method_hyperparameter_summary_path = os.path.join(TABLES_DIR, "method_hyperparameter_summary.csv")
method_hyperparameter_summary_df.to_csv(method_hyperparameter_summary_path, index=False)
print("Saved method hyperparameter summary:", method_hyperparameter_summary_path)

missing_active_results = [m for m in active_method_order if m not in set(results_df["method"])]
print("Missing active methods in accuracy results:", missing_active_results)
if len(results_df) > 0:
    assert len(missing_active_results) == 0, f"Missing accuracy rows for active methods: {missing_active_results}"

missing_loss_logs = [m for m in active_method_order if m not in set(training_loss_history_df["method_name"])] if len(training_loss_history_df) > 0 else active_method_order
print("Missing active methods in epoch loss logs:", missing_loss_logs)
if len(training_loss_history_df) > 0:
    assert len(missing_loss_logs) == 0, f"Missing epoch loss rows for active methods: {missing_loss_logs}"

simple_factor_lambdas = sorted(summary_table.loc[summary_table["method"].isin(["simple_avg_factor_orth", "simple_avg_factor_orth_kd_T2"]), "lambda_factor_orth"].dropna().unique().tolist())
rankext_factor_lambdas = sorted(summary_table.loc[summary_table["method"].isin(["rank_extension_orth_factor_lam_50", "rank_extension_orth_factor_lam_50_kd_T2"]), "lambda_factor_orth"].dropna().unique().tolist())
delta_trace_lambdas = sorted(summary_table.loc[summary_table["uses_delta_trace"], "lambda_delta_trace"].dropna().unique().tolist())
factor_orth_lambdas = sorted(summary_table.loc[summary_table["uses_factor_orth"], "lambda_factor_orth"].dropna().unique().tolist())
kd_temperatures_used = sorted(summary_table.loc[summary_table["uses_kd"], "kd_temperature"].dropna().unique().tolist())
kd_weights_used = sorted(summary_table.loc[summary_table["uses_kd"], "kd_weight"].dropna().unique().tolist())
replay_settings_used = sorted(summary_table["replay_per_class"].dropna().unique().tolist())

simple_factor_ratio = float(loss_summary_by_method_df.loc[loss_summary_by_method_df["method_name"].isin(["simple_avg_factor_orth", "simple_avg_factor_orth_kd_T2"]), "mean_factor_orth_weighted_over_ce"].mean())
rankext_factor_ratio = float(loss_summary_by_method_df.loc[loss_summary_by_method_df["method_name"].isin(["rank_extension_orth_factor_lam_50", "rank_extension_orth_factor_lam_50_kd_T2"]), "mean_factor_orth_weighted_over_ce"].mean())
delta_trace_ratio = float(loss_summary_by_method_df.loc[loss_summary_by_method_df["method_name"].isin(["simple_avg_delta_orth", "simple_avg_delta_orth_kd_T2", "rank_extension_orth_delta_trace_lam_50", "rank_extension_orth_delta_trace_lam_50_kd_T2"]), "mean_delta_trace_weighted_over_ce"].mean())

print("\nSupervisor hyperparameter summary:")
print({
    "simple_avg_factor_orth_lambda": simple_factor_lambdas,
    "rank_extension_factor_orth_lambda": rankext_factor_lambdas,
    "delta_trace_lambda": delta_trace_lambdas,
    "kd_temperatures_used": kd_temperatures_used,
    "kd_weights_used": kd_weights_used,
    "replay_settings_used": replay_settings_used,
    "simple_vs_rankext_factor_lambda_match": simple_factor_lambdas == rankext_factor_lambdas,
    "delta_trace_vs_factor_orth_lambda_match": delta_trace_lambdas == factor_orth_lambdas,
    "mean_simple_factor_weighted_over_ce": simple_factor_ratio,
    "mean_rankext_factor_weighted_over_ce": rankext_factor_ratio,
    "mean_delta_trace_weighted_over_ce": delta_trace_ratio,
})
print("Validation CE uses held-out validation data from dataset['train']; final accuracy still uses dataset['test'] only.")

print("\nSaved final accuracy table:", final_accuracy_path)
print("Saved ranking table:", ranking_table_path)
print("Saved summary metrics table:", summary_table_path)
print("Saved method metadata:", method_metadata_path)

display(final_accuracy_df.round(2))
display(ranking_table[["method", "rank_all_seen", "all_seen"]].round(2))
display(summary_table.round(4))
display(loss_summary_by_method_df.round(6))
display(method_hyperparameter_summary_df)


# In[ ]:


summary_plot_df = summary_table.copy()
summary_plot_df["method"] = pd.Categorical(summary_plot_df["method"], categories=active_method_order, ordered=True)
summary_plot_df = summary_plot_df.sort_values("method").reset_index(drop=True)
summary_plot_df["plot_label"] = summary_plot_df["display_name"]

loss_plot_df = loss_components_summary_df.copy()
loss_plot_df["method"] = pd.Categorical(loss_plot_df["method"], categories=active_method_order, ordered=True)
loss_plot_df = loss_plot_df.sort_values("method").reset_index(drop=True)
loss_plot_df["plot_label"] = loss_plot_df["display_name"]


def save_current_plot(plot_name):
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved:", plot_path)


def add_step_guides(ax, epochs_per_step, total_steps, y_frac=0.96):
    for x in np.arange(epochs_per_step, total_steps * epochs_per_step, epochs_per_step):
        ax.axvline(float(x), color="#c7c7c7", linestyle=":", linewidth=0.9, zorder=0)
    y_top = ax.get_ylim()[1]
    for step_idx in range(total_steps):
        center = step_idx * float(epochs_per_step) + 0.5 * float(epochs_per_step)
        ax.text(center, y_top * y_frac, f"Step {step_idx + 1}", ha="center", va="top", fontsize=8, color="#555555")


def save_figure_object(fig, plot_name):
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved:", plot_path)


if len(loss_plot_df) > 0:
    plt.figure(figsize=(16, 10))
    plt.barh(loss_plot_df["plot_label"], loss_plot_df["ce_loss_mean"].fillna(0.0), color="#4c78a8")
    plt.xlabel("Mean train CE loss")
    plt.title("Train CE Loss by Method")
    save_current_plot("08_ce_loss_by_method.png")

    plt.figure(figsize=(16, 10))
    y = np.arange(len(loss_plot_df))
    plt.barh(y - 0.18, loss_plot_df["ce_loss_mean"].fillna(0.0), height=0.35, label="Train CE", color="#4c78a8")
    plt.barh(y + 0.18, loss_plot_df["val_ce_loss_mean"].fillna(0.0), height=0.35, label="Validation CE", color="#f58518")
    plt.yticks(y, loss_plot_df["plot_label"])
    plt.xlabel("Mean CE loss")
    plt.title("Mean Train vs Validation CE by Method")
    plt.legend()
    save_current_plot("08b_train_val_ce_loss_by_method.png")

kd_plot_df = loss_plot_df[loss_plot_df["uses_kd"]].copy()
if len(kd_plot_df) > 0:
    y = np.arange(len(kd_plot_df))
    plt.figure(figsize=(16, 10))
    plt.barh(y - 0.18, kd_plot_df["kd_loss_raw_mean"].fillna(0.0), height=0.35, label="KD raw", color="#4c78a8")
    plt.barh(y + 0.18, kd_plot_df["kd_loss_weighted_mean"].fillna(0.0), height=0.35, label="KD weighted", color="#f58518")
    plt.yticks(y, kd_plot_df["plot_label"])
    plt.xlabel("Mean KD loss")
    plt.title("KD Loss by KD Method")
    plt.legend()
    save_current_plot("09_kd_loss_by_method.png")

delta_plot_df = loss_plot_df[loss_plot_df["uses_delta_trace"]].copy()
if len(delta_plot_df) > 0:
    y = np.arange(len(delta_plot_df))
    plt.figure(figsize=(16, 10))
    plt.barh(y - 0.18, delta_plot_df["delta_trace_loss_raw_mean"].fillna(0.0), height=0.35, label="Delta-trace raw", color="#54a24b")
    plt.barh(y + 0.18, delta_plot_df["delta_trace_loss_weighted_mean"].fillna(0.0), height=0.35, label="Delta-trace weighted", color="#2f7d32")
    plt.yticks(y, delta_plot_df["plot_label"])
    plt.xlabel("Mean delta-trace loss")
    plt.title("Delta-Trace Loss by Method")
    plt.legend()
    save_current_plot("10_delta_trace_loss_by_method.png")

factor_plot_df = loss_plot_df[loss_plot_df["uses_factor_orth"]].copy()
if len(factor_plot_df) > 0:
    y = np.arange(len(factor_plot_df))
    plt.figure(figsize=(16, 10))
    plt.barh(y - 0.18, factor_plot_df["factor_orth_loss_raw_mean"].fillna(0.0), height=0.35, label="Factor-orth raw", color="#e45756")
    plt.barh(y + 0.18, factor_plot_df["factor_orth_loss_weighted_mean"].fillna(0.0), height=0.35, label="Factor-orth weighted", color="#b23a48")
    plt.yticks(y, factor_plot_df["plot_label"])
    plt.xlabel("Mean factor-orth loss")
    plt.title("Factor-Orth Loss by Method")
    plt.legend()
    save_current_plot("11_factor_orth_loss_by_method.png")

if len(loss_plot_df) > 0:
    plt.figure(figsize=(16, 10))
    plt.barh(loss_plot_df["plot_label"], loss_plot_df["total_loss_mean"].fillna(0.0), color="#b279a2")
    plt.xlabel("Mean train total loss")
    plt.title("Total Loss by Method")
    save_current_plot("12_total_loss_by_method.png")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    axes[0].bar(loss_plot_df["plot_label"], loss_plot_df["kd_over_CE_mean"].fillna(0.0), color="#4c78a8")
    axes[0].set_ylabel("KD / CE")
    axes[1].bar(loss_plot_df["plot_label"], loss_plot_df["delta_trace_over_CE_mean"].fillna(0.0), color="#54a24b")
    axes[1].set_ylabel("Delta / CE")
    axes[2].bar(loss_plot_df["plot_label"], loss_plot_df["factor_orth_over_CE_mean"].fillna(0.0), color="#e45756")
    axes[2].set_ylabel("Factor / CE")
    axes[2].tick_params(axis="x", rotation=30)
    fig.suptitle("Loss Ratio Diagnostics")
    save_figure_object(fig, "13_loss_ratio_diagnostics.png")

combined_df = loss_plot_df[(loss_plot_df["uses_kd"]) | (loss_plot_df["uses_delta_trace"]) | (loss_plot_df["uses_factor_orth"])].copy()
if len(combined_df) > 0:
    combined_df["orth_weighted_mean"] = np.where(
        combined_df["uses_delta_trace"],
        combined_df["delta_trace_loss_weighted_mean"].fillna(0.0),
        combined_df["factor_orth_loss_weighted_mean"].fillna(0.0),
    )
    plt.figure(figsize=(18, 10))
    plt.barh(combined_df["plot_label"], combined_df["ce_loss_mean"].fillna(0.0), label="Train CE", color="#4c78a8")
    plt.barh(combined_df["plot_label"], combined_df["kd_loss_weighted_mean"].fillna(0.0), left=combined_df["ce_loss_mean"].fillna(0.0), label="KD weighted", color="#f58518")
    plt.barh(
        combined_df["plot_label"],
        combined_df["orth_weighted_mean"].fillna(0.0),
        left=(combined_df["ce_loss_mean"].fillna(0.0) + combined_df["kd_loss_weighted_mean"].fillna(0.0)),
        label="Orth weighted",
        color="#54a24b",
    )
    plt.scatter(combined_df["total_loss_mean"].fillna(0.0), combined_df["plot_label"], color="black", label="Train total loss")
    plt.xlabel("Mean loss value")
    plt.title("Combined Loss Decomposition")
    plt.legend()
    save_current_plot("14_combined_loss_decomposition.png")

from matplotlib.lines import Line2D

selected_epoch_df = training_loss_history_df[training_loss_history_df["method_name"].isin(SUPERVISOR_SELECTED_INTERNAL_METHODS)].copy()
if len(selected_epoch_df) > 0:
    # FIX 1 (was: "KeyError: 'family'" here, which killed the previous cluster run
    # AFTER training had already finished, losing the final summary tables).
    #
    # Root cause: training_loss_history_df already carries its own "family" column
    # (it's one of the groupby keys used to build train_epoch_df earlier in this
    # script -- see the loss_component_cols groupby -- and it also survives the
    # empty-history fallback branch, which lists "family" in its column set too).
    # The old code then did
    #     selected_epoch_df.merge(method_config_df[["method", "family"]], ...)
    # on top of that -- since "family" exists on BOTH sides of that merge and is
    # not a join key, pandas silently renamed the result to "family_x"/"family_y"
    # instead of raising during the merge itself, so the *next* line
    # (selected_epoch_df["family"] == "simple_avg") is what actually raised
    # KeyError: 'family'.
    #
    # Fix: don't merge at all -- (re)derive "family" directly and unambiguously
    # from ACTIVE_METHOD_MAP (the single source of truth for each active method's
    # family), which cannot collide with any existing column on selected_epoch_df.
    # Methods not present in ACTIVE_METHOD_MAP (e.g. a stale/disabled method name
    # that somehow still shows up in the loss history) get "family"=NaN and are
    # dropped from this plot with a warning instead of crashing.
    selected_epoch_df["family"] = selected_epoch_df["method_name"].map(
        lambda m: ACTIVE_METHOD_MAP.get(m, {}).get("family", np.nan)
    )
    _missing_family = sorted(selected_epoch_df.loc[selected_epoch_df["family"].isna(), "method_name"].unique().tolist())
    if _missing_family:
        print(f"[15_supervisor_selected_train_val_ce] WARNING: no family mapping for methods {_missing_family}; dropping their rows from this plot.")
        selected_epoch_df = selected_epoch_df.dropna(subset=["family"])

    selected_epoch_df["epochs_per_step"] = np.where(
        selected_epoch_df["family"] == "simple_avg",
        float(LORA_EPOCHS),
        float(RANKEXT_EPOCHS),
    )
    selected_epoch_df["global_epoch"] = (selected_epoch_df["step_id"] - 1) * selected_epoch_df["epochs_per_step"] + selected_epoch_df["epoch"]

    color_map = {
        method_name: color
        for method_name, color in zip(
            SUPERVISOR_SELECTED_INTERNAL_METHODS,
            ["#1f77b4", "#d62728", "#ff7f0e", "#9467bd", "#2ca02c", "#8c564b", "#17becf", "#e377c2"],
        )
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, family in zip(axes, ["simple_avg", "rank_extension"]):
        # .get(m, {}) instead of ACTIVE_METHOD_MAP[m]: skip gracefully rather than
        # KeyError if a supervisor-selected method name is ever absent from the
        # active set (e.g. a disabled family), instead of assuming it is always active.
        family_methods = [m for m in SUPERVISOR_SELECTED_INTERNAL_METHODS if ACTIVE_METHOD_MAP.get(m, {}).get("family") == family]
        for method_name in family_methods:
            sub = selected_epoch_df[selected_epoch_df["method_name"] == method_name].sort_values(["step_id", "epoch"])
            if len(sub) == 0:
                continue
            color = color_map[method_name]
            # Task 3: smooth (PCHIP) curves within each CL step, broken at step
            # boundaries -- never one continuous line across steps (see
            # _plot_step_broken_series docstring for why).
            _plot_step_broken_series(ax, sub, "train_ce_loss", color, None,
                                      lw=1.8, linestyle="-", x_col="global_epoch")
            _plot_step_broken_series(ax, sub, "val_ce_loss", color, None,
                                      lw=1.6, linestyle="--", x_col="global_epoch")
        ax.set_ylabel("CE loss")
        ax.set_title("SimpleAvg Family" if family == "simple_avg" else "RankExt Family", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_xlim(1, NUM_STEPS * float(LORA_EPOCHS))
        add_step_guides(ax, epochs_per_step=float(LORA_EPOCHS), total_steps=NUM_STEPS)

        color_handles = [
            Line2D([0], [0], color=color_map[m], linewidth=2.0, label=METHOD_DISPLAY_NAME_MAP.get(m, m))
            for m in family_methods
        ]
        style_handles = [
            Line2D([0], [0], color="#333333", linewidth=2.0, linestyle="-", label="Train CE"),
            Line2D([0], [0], color="#333333", linewidth=2.0, linestyle="--", label="Validation CE"),
        ]
        ax.legend(handles=color_handles + style_handles, loc="upper right", frameon=False, fontsize=8)

    axes[-1].set_xlabel("Cumulative epoch")
    save_figure_object(fig, "15_supervisor_selected_train_val_ce.png")
else:
    print("Skipping 15_supervisor_selected_train_val_ce.png: no epoch-level loss rows available")

selected_acc_df = supervisor_selected_accuracy_export_df.copy()
if len(selected_acc_df) > 0:
    selected_acc_df["display_name"] = pd.Categorical(
        selected_acc_df["display_name"],
        categories=SUPERVISOR_SELECTED_DISPLAY_NAMES,
        ordered=True,
    )
    selected_acc_df = selected_acc_df.sort_values("display_name").reset_index(drop=True)
    x = np.arange(len(selected_acc_df))
    width = 0.24
    plt.figure(figsize=(16, 7))
    plt.bar(x - width, selected_acc_df["first_step"], width=width, label="first_step", color="#4c78a8")
    plt.bar(x, selected_acc_df["later_steps"], width=width, label="later_steps", color="#f58518")
    plt.bar(x + width, selected_acc_df["all_seen"], width=width, label="all_seen", color="#54a24b")
    plt.xticks(x, selected_acc_df["display_name"], rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Supervisor-Selected Accuracy Comparison")
    plt.legend()
    save_current_plot("17_supervisor_selected_accuracy_comparison.png")
else:
    print("Skipping 17_supervisor_selected_accuracy_comparison.png: no supervisor-selected accuracy rows available")


# In[ ]:


# Supervisor-ready automatic outputs, validation analysis, reports, and final checklist
from pathlib import Path
import json
from matplotlib.lines import Line2D

DPI = 220
REQ = list(SUPERVISOR_SELECTED_INTERNAL_METHODS)
SUPERVISOR_VARIANT_ORDER = ["Base", "KD (T=2)", "Factor-Orth", "KD + Factor-Orth"]
VARIANT = {"simple_avg":"Base","rank_extension":"Base","simple_avg_factor_orth":"Factor-Orth","rank_extension_orth_factor_lam_50":"Factor-Orth","simple_avg_kd_T2":"KD (T=2)","rank_extension_kd_only_T2":"KD (T=2)","simple_avg_factor_orth_kd_T2":"KD + Factor-Orth","rank_extension_orth_factor_lam_50_kd_T2":"KD + Factor-Orth"}
VCOL = {"Base":"#1f77b4","KD (T=2)":"#ff7f0e","Factor-Orth":"#d62728","KD + Factor-Orth":"#2ca02c"}
VSTYLE = {"Base":"-","KD (T=2)":"--","Factor-Orth":":","KD + Factor-Orth":"-."}
FAMS = ["simple_avg","rank_extension"]
FLAB = {"simple_avg":"Simple-Average Family","rank_extension":"Rank-Extension Family"}
for d in [TABLES_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR, CONFIGS_DIR, MODELS_DIR]: Path(d).mkdir(parents=True, exist_ok=True)
missing_outputs = []
assert not [m for m in REQ if m not in ACTIVE_METHOD_NAMES], f"Missing selected methods: {[m for m in REQ if m not in ACTIVE_METHOD_NAMES]}"

def txt(path, s): Path(path).write_text(str(s).rstrip()+"\n", encoding="utf-8")
def js(path, obj): Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
def ok(path): return Path(path).exists() and Path(path).stat().st_size > 0
def figsave(name): plt.tight_layout(); plt.savefig(Path(PLOTS_DIR)/name, dpi=DPI, bbox_inches="tight"); plt.close(); assert ok(Path(PLOTS_DIR)/name), name
def disp(m): return METHOD_DISPLAY_NAME_MAP.get(str(m), str(m))
def fam(m): return ACTIVE_METHOD_MAP.get(str(m), {}).get("family", "")
def variant(m): return VARIANT.get(str(m), "Other")

def cfg_df():
    c=pd.DataFrame(ACTIVE_METHOD_CONFIGS); c=c[c.method.isin(REQ)].copy(); c["method"]=pd.Categorical(c.method, REQ, ordered=True); c=c.sort_values("method")
    c["display_method_name"]=c.method.astype(str).map(METHOD_DISPLAY_NAME_MAP).fillna(c.method.astype(str)); c["lora_rank"]=np.where(c.family.eq("rank_extension"), active_rankext_rank_schedule()[-1], LORA_R)
    c["lora_alpha"]=LORA_ALPHA; c["lora_dropout"]=LORA_DROPOUT; c["batch_size"]=BATCH_LORA
    c["num_epochs"]=np.where(c.family.eq("rank_extension"), RANKEXT_EPOCHS, LORA_EPOCHS); c["learning_rate"]=np.where(c.family.eq("rank_extension"), LR_RANKEXT, LR_LORA)
    c["optimizer"]="AdamW"; c["scheduler"]=SCHED
    # POST-INCIDENT FIX: "apply_calibration", "target_modules", "head_lr_multiplier",
    # and "lambda_orth"/"kd_weight" (including their Objective-2 scale factors)
    # are already the per-method truth (set in add_method() via
    # family_applies_calibration() / family_target_modules() /
    # family_head_lr_multiplier() / lambda_orth_scale / kd_weight_scale) -- do
    # NOT broadcast the corresponding globals (TARGET_MODULES,
    # HEAD_LR_MULTIPLIER, LAMBDA_ORTH, USE_CLASSIFIER_CALIBRATION) over them
    # here, that would silently erase every per-family/per-method override for
    # every downstream CSV/JSON consumer -- this bug already bit
    # "target_modules" and "head_lr_multiplier" once (both were being
    # broadcast-overwritten here until this fix) after "apply_calibration" was
    # already correctly fixed. "use_classifier_calibration" is kept as an
    # alias of the same per-method value so existing readers of that column
    # name still see the real, per-method effective state rather than the
    # master switch.
    c["use_classifier_calibration"]=c["apply_calibration"]
    # STRICT-REVIEW ADD (B4): SEED is already the single source-of-truth
    # constant (defined once near the top of the script and threaded into
    # set_seed()/random.seed()/np.random.seed()/torch.manual_seed() and every
    # dataset shuffle -- see SEED's own definition comment). Stamping it onto
    # every per-method config row (and therefore into every table/JSON built
    # from CFG below) means a future multi-seed sweep is fully traceable from
    # any single saved artifact, not just configs/run_config.json, without
    # cross-referencing which run used which seed.
    c["seed"]=int(SEED)
    return c.reset_index(drop=True)
CFG=cfg_df()
js(Path(CONFIGS_DIR)/"run_config.json", {"run_name":RUN_NAME,"run_tag":RUN_TAG,"base_output_dir":BASE_OUTPUT_DIR,"model_checkpoint":MODEL_CHECKPOINT,"seed":SEED,"num_steps":NUM_STEPS,"classes_per_step":CLASSES_PER_STEP,"lora_rank":LORA_R,"lora_alpha":LORA_ALPHA,"lora_dropout":LORA_DROPOUT,"target_modules_default":TARGET_MODULES,"target_modules_by_family":{k:list(v) for k,v in TARGET_MODULES_BY_FAMILY.items()},"lambda_orth":LAMBDA_ORTH,"kd_temperatures":KD_TEMPERATURES,"kd_weight":KD_WEIGHT,"optimizer":"AdamW","scheduler":SCHED,"batch_size":BATCH_LORA,"use_classifier_calibration_master_switch":bool(USE_CLASSIFIER_CALIBRATION),"classifier_calibration_by_family":dict(CALIBRATION_ENABLED_FAMILIES),"head_lr_multiplier_default":float(HEAD_LR_MULTIPLIER),"head_lr_multiplier_by_family":{k:float(v) for k,v in HEAD_LR_MULTIPLIER_BY_FAMILY.items()},"rankext_rank_schedule_active":active_rankext_rank_schedule(),"rankext_rank_schedule_wide_enabled":bool(USE_RANKEXT_RANK_SCHEDULE_WIDE),"rankext_orth_lambda_warmup_enabled":bool(RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED),"rankext_orth_lambda_warmup_epochs":float(RANKEXT_ORTH_LAMBDA_WARMUP_EPOCHS),"combined_loss_scale_enabled":bool(COMBINED_LOSS_SCALE_ENABLED),"combined_lambda_orth_scale":float(COMBINED_LAMBDA_ORTH_SCALE),"combined_kd_weight_scale":float(COMBINED_KD_WEIGHT_SCALE),"combined_orth_warmup_enabled":bool(COMBINED_ORTH_WARMUP_ENABLED),"combined_orth_warmup_epochs":float(COMBINED_ORTH_WARMUP_EPOCHS)})
js(Path(CONFIGS_DIR)/"supervisor_selected_methods.json", SUPERVISOR_SELECTED_METHOD_SPECS)
js(Path(CONFIGS_DIR)/"hyperparameters_by_method.json", CFG.to_dict("records"))

def epoch_table():
    if "training_loss_history_df" not in globals() or len(training_loss_history_df)==0: return pd.DataFrame()
    e=training_loss_history_df[training_loss_history_df.method_name.isin(REQ)].copy(); e=e.merge(CFG[["method","display_method_name","lora_rank","lora_alpha","lora_dropout","target_modules","batch_size","num_epochs","optimizer","scheduler","seed"]], left_on="method_name", right_on="method", how="left")
    e["method"]=e.method_name; e["display_method_name"]=e.display_method_name.fillna(e.display_name); e["cl_step"]=e.step_id.astype(int); e["local_epoch"]=e.epoch.astype(int); e["global_epoch"]=(e.cl_step-1)*e.num_epochs.fillna(LORA_EPOCHS).astype(int)+e.local_epoch
    for a,b in {"train_kd_loss_raw":"kd_loss_raw","train_kd_loss_weighted":"kd_loss_weighted","train_factor_orth_loss_raw":"factor_orth_loss_raw","train_factor_orth_loss_weighted":"factor_orth_loss_weighted","train_delta_trace_loss_raw":"delta_trace_loss_raw","train_delta_trace_loss_weighted":"delta_trace_loss_weighted"}.items(): e[b]=e[a] if a in e else np.nan
    cols=["method","display_method_name","cl_step","local_epoch","global_epoch","train_ce_loss","val_ce_loss","train_total_loss","kd_loss_raw","kd_loss_weighted","factor_orth_loss_raw","factor_orth_loss_weighted","delta_trace_loss_raw","delta_trace_loss_weighted","learning_rate","lambda_orth","kd_temperature","lora_rank","lora_alpha","lora_dropout","target_modules","batch_size","num_epochs","optimizer","scheduler","seed"]
    for c in cols:
        if c not in e: e[c]=np.nan
    e=e[cols+[c for c in e.columns if c not in cols]].sort_values(["method","cl_step","local_epoch"]); e.to_csv(Path(TABLES_DIR)/"training_loss_history_by_epoch.csv", index=False); return e
E=epoch_table()
if len(E)==0:
    missing_outputs.append({"output":"tables/training_loss_history_by_epoch.csv","method":"all supervisor-selected methods","metric_or_column":"epoch loss rows","why":"training_loss_history_df was empty or unavailable after training.","required_or_optional":"required"})
else:
    for m in REQ:
        md=E[E.method==m]
        if len(md)==0:
            missing_outputs.append({"output":"tables/training_loss_history_by_epoch.csv","method":m,"metric_or_column":"method rows","why":"No epoch-level rows for this selected method reached the final automation cell.","required_or_optional":"required"})
        for col in ["train_ce_loss","val_ce_loss","train_total_loss"]:
            if len(md)>0 and (col not in md or md[col].isna().all()):
                missing_outputs.append({"output":"tables/training_loss_history_by_epoch.csv","method":m,"metric_or_column":col,"why":"Required epoch metric is missing or all NaN for this method.","required_or_optional":"required"})
        if ACTIVE_METHOD_MAP[m]["uses_kd"] and len(md)>0 and md["kd_loss_weighted"].isna().all():
            missing_outputs.append({"output":"tables/training_loss_history_by_epoch.csv","method":m,"metric_or_column":"kd_loss_weighted","why":"Method is configured as KD but weighted KD loss is all NaN.","required_or_optional":"required_for_kd_methods"})
        if ACTIVE_METHOD_MAP[m]["uses_factor_orth"] and len(md)>0 and md["factor_orth_loss_weighted"].isna().all():
            missing_outputs.append({"output":"tables/training_loss_history_by_epoch.csv","method":m,"metric_or_column":"factor_orth_loss_weighted","why":"Method is configured as factor-orth but weighted factor-orth loss is all NaN.","required_or_optional":"required_for_factor_orth_methods"})
if "train_diag_df" in globals() and len(train_diag_df)>0:
    B=train_diag_df[train_diag_df.method.isin(REQ)].copy() if "method" in train_diag_df else pd.DataFrame()
    if len(B)>0: B.to_csv(Path(LOGS_DIR)/"training_loss_history_by_batch.csv", index=False)

# PRE-THESIS FIX 2: long-format per-CL-step accuracy table (method, step_id,
# accuracy in %), the ingredient for both the `per_step_accuracy` column below
# and the 8-methods x 5-steps heatmap / forgetting-curve plots further down.
per_step_acc_df = pd.DataFrame(per_step_accuracy_rows)
if len(per_step_acc_df) > 0:
    per_step_acc_df = per_step_acc_df[per_step_acc_df["method"].isin(REQ)].copy()
    per_step_acc_df = per_step_acc_df.sort_values(["method", "step_id"]).reset_index(drop=True)
else:
    per_step_acc_df = pd.DataFrame(columns=["method", "step_id", "accuracy"])
per_step_acc_path = Path(TABLES_DIR) / "per_step_accuracy_by_method.csv"
per_step_acc_df.to_csv(per_step_acc_path, index=False)
print("Saved per-step accuracy:", per_step_acc_path)


def per_step_accuracy_json(method_name):
    sub = per_step_acc_df[per_step_acc_df.method == method_name].sort_values("step_id")
    if len(sub) == 0:
        return np.nan
    return json.dumps([None if pd.isna(v) else round(float(v), 4) for v in sub["accuracy"]])


def metrics_tables():
    s=summary_table.copy() if "summary_table" in globals() and len(summary_table)>0 else CFG.copy(); s=s[s.method.isin(REQ)].copy(); s["method"]=pd.Categorical(s.method, REQ, ordered=True); s=s.sort_values("method")
    for c in ["first_step","later_steps","all_seen","avg_forgetting","backward_transfer","forward_transfer"]:
        if c not in s: s[c]=np.nan
    out=pd.DataFrame({"method":s.method.astype(str),"display_method_name":s.get("display_name",s.method.astype(str)),"first_step_accuracy":s.first_step,"later_steps_accuracy":s.later_steps,"all_seen_accuracy":s.all_seen,"average_accuracy":s[["first_step","later_steps","all_seen"]].mean(axis=1),"final_accuracy":s.all_seen,"per_step_accuracy":s.method.astype(str).map(per_step_accuracy_json),"forgetting_metric":s.avg_forgetting,"backward_transfer":s.backward_transfer,"forward_transfer":s.forward_transfer})
    out["seed"]=int(SEED)  # STRICT-REVIEW ADD (B4)
    out.to_csv(Path(TABLES_DIR)/"supervisor_selected_accuracy_comparison.csv", index=False)
    allm=summary_table.copy() if "summary_table" in globals() and len(summary_table)>0 else s.copy()
    for c in ["first_step","later_steps","all_seen","avg_forgetting","backward_transfer","forward_transfer"]:
        if c not in allm: allm[c]=np.nan
    allout=pd.DataFrame({"method":allm.method.astype(str),"display_method_name":allm.get("display_name",allm.method.astype(str)),"first_step_accuracy":allm.first_step,"later_steps_accuracy":allm.later_steps,"all_seen_accuracy":allm.all_seen,"average_accuracy":allm[["first_step","later_steps","all_seen"]].mean(axis=1),"final_accuracy":allm.all_seen,"per_step_accuracy":allm.method.astype(str).map(per_step_accuracy_json),"forgetting_metric":allm.avg_forgetting,"backward_transfer":allm.backward_transfer,"forward_transfer":allm.forward_transfer})
    allout["seed"]=int(SEED)  # STRICT-REVIEW ADD (B4)
    allout.to_csv(Path(TABLES_DIR)/"final_metrics_all_methods.csv", index=False); return out
M=metrics_tables()
for m in REQ:
    mm=M[M.method==m]
    if len(mm)==0:
        missing_outputs.append({"output":"tables/supervisor_selected_accuracy_comparison.csv","method":m,"metric_or_column":"method row","why":"No final accuracy row exists for this selected method.","required_or_optional":"required"})
    elif mm[["first_step_accuracy","later_steps_accuracy","all_seen_accuracy"]].isna().to_numpy().all():
        missing_outputs.append({"output":"tables/supervisor_selected_accuracy_comparison.csv","method":m,"metric_or_column":"accuracy metrics","why":"Final accuracy metrics are all NaN for this selected method.","required_or_optional":"required"})

def heat(df, cols, name, title):
    d=df.copy(); d["display_method_name"]=pd.Categorical(d.display_method_name, SUPERVISOR_SELECTED_DISPLAY_NAMES, ordered=True); d=d.sort_values("display_method_name"); mat=d.set_index("display_method_name")[cols].apply(pd.to_numeric, errors="coerce")
    fig,ax=plt.subplots(figsize=(max(8,1.4*len(cols)+5), max(5,.55*len(mat)+2))); im=ax.imshow(mat.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=25, ha="right"); ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index); ax.set_title(title, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v=mat.iloc[i,j]; ax.text(j,i,"NA" if pd.isna(v) else f"{v:.1f}",ha="center",va="center",fontsize=9)
    fig.colorbar(im, ax=ax); figsave(name)
heat(M,["first_step_accuracy","later_steps_accuracy","all_seen_accuracy"],"supervisor_method_step_accuracy_heatmap.png","Available Accuracy Groups Heatmap")
heat(M,["first_step_accuracy","later_steps_accuracy","all_seen_accuracy","average_accuracy","forgetting_metric"],"supervisor_method_metric_heatmap.png","Method x Metric Heatmap")

# PRE-THESIS FIX 2: real 8-methods x 5-steps per-CL-step accuracy heatmap
# (previously a placeholder -- per-task/class-group accuracy was not retained).
if len(per_step_acc_df) > 0:
    _pt = per_step_acc_df.copy()
    _pt["display_method_name"] = _pt["method"].map(METHOD_DISPLAY_NAME_MAP).fillna(_pt["method"])
    _pt_mat = _pt.pivot(index="display_method_name", columns="step_id", values="accuracy")
    _pt_mat = _pt_mat.reindex(SUPERVISOR_SELECTED_DISPLAY_NAMES)
    _pt_mat.columns = [f"step_{c}" for c in _pt_mat.columns]
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(_pt_mat.columns) + 5), max(5, .55 * len(_pt_mat) + 2)))
    im = ax.imshow(_pt_mat.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(_pt_mat.columns))); ax.set_xticklabels(_pt_mat.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(_pt_mat))); ax.set_yticklabels(_pt_mat.index)
    ax.set_title("Per-CL-step accuracy (%) of each method's FINAL model", fontweight="bold")
    for i in range(_pt_mat.shape[0]):
        for j in range(_pt_mat.shape[1]):
            v = _pt_mat.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(v) else f"{v:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    figsave("per_task_accuracy_heatmap.png")
else:
    missing_outputs.append({"output":"plots/per_task_accuracy_heatmap.png","method":"all","metric_or_column":"per-task/class-group accuracy","why":"per_step_accuracy_rows was empty after training (see evaluate_per_step_accuracy call sites).","required_or_optional":"conditional"})
    plt.figure(figsize=(10,3)); plt.axis("off"); plt.text(.5,.5,"Per-task/class-group accuracy unavailable.\nSee reports/missing_outputs_or_metrics.txt.",ha="center",va="center"); figsave("per_task_accuracy_heatmap.png")

# PRE-THESIS FIX 2: forgetting curve per method. rank_extension methods get a
# TRUE forgetting curve (accuracy on task i re-measured after each later step,
# from the full stepwise matrix collected during training); simple_avg methods
# only ever have one checkpoint (the final merged model) so they get a single
# per-step-accuracy point per task instead -- plotted in a separate panel and
# clearly labeled, rather than faking an intermediate trajectory that family
# does not have.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
rankext_methods_present = [m for m in REQ if ACTIVE_METHOD_MAP.get(m, {}).get("family") == "rank_extension" and m in rank_extension_stepwise_accuracy_by_method]
for method_name in rankext_methods_present:
    matrix = rank_extension_stepwise_accuracy_by_method[method_name]
    for task_step in range(NUM_STEPS):
        xs, ys = [], []
        for later_step in range(task_step, NUM_STEPS):
            if later_step in matrix and task_step in matrix[later_step]:
                xs.append(later_step + 1)
                ys.append(matrix[later_step][task_step] * 100.0)
        if len(xs) >= 2:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6,
                    color=VCOL.get(variant(method_name), "#333333"),
                    linestyle=VSTYLE.get(variant(method_name), "-"),
                    label=f"{disp(method_name)} / task {task_step + 1}" if task_step == 0 else None,
                    alpha=0.85)
ax.set_xlabel("CL step at evaluation time")
ax.set_ylabel("Accuracy on task i (%)")
ax.set_title("RankExt family: true forgetting curves\n(one line per method, resampling task i after each later step)", fontsize=10)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
handles = [Line2D([0], [0], color=VCOL[v], linestyle=VSTYLE[v], lw=2, label=v) for v in SUPERVISOR_VARIANT_ORDER]
ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)

ax = axes[1]
simple_methods_present = [m for m in REQ if ACTIVE_METHOD_MAP.get(m, {}).get("family") == "simple_avg"]
_pt_simple = per_step_acc_df[per_step_acc_df["method"].isin(simple_methods_present)]
for method_name in simple_methods_present:
    sub = _pt_simple[_pt_simple["method"] == method_name].sort_values("step_id")
    if len(sub) == 0:
        continue
    ax.plot(sub["step_id"], sub["accuracy"], marker="o", ms=5, lw=1.6,
            color=VCOL.get(variant(method_name), "#333333"),
            linestyle=VSTYLE.get(variant(method_name), "-"),
            label=disp(method_name))
ax.set_xlabel("Task (CL step) index")
ax.set_ylabel("Accuracy on task i, FINAL model only (%)")
ax.set_title("SimpleAvg family: final-model accuracy per task\n(no intermediate checkpoints exist for this merge-based family)", fontsize=10)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
ax.legend(loc="lower left", fontsize=8, frameon=False)

fig.suptitle("Forgetting curves by method family", fontweight="bold")
figsave("forgetting_curve_by_method.png")

def lossgrid(metric,ylabel,name,title,methods=None,log=False,pos=False):
    d=E.copy();
    if methods: d=d[d.method.isin(methods)]
    if metric not in d: d[metric]=np.nan
    d[metric]=pd.to_numeric(d[metric],errors="coerce");
    if pos: d=d[d[metric]>0]
    if len(d.dropna(subset=[metric]))==0:
        plt.figure(figsize=(10,3)); plt.axis("off"); plt.text(.5,.5,f"No logged values for {ylabel}",ha="center",va="center"); figsave(name); return
    d["variant"]=d.method.map(variant); d["family"]=d.method.map(fam); fig,axs=plt.subplots(2,NUM_STEPS,figsize=(18,8),sharey=True); fig.suptitle(title,fontsize=22,fontweight="bold",y=.995)
    for r,f in enumerate(FAMS):
        axs[r,0].text(-.35,1.15,FLAB[f],transform=axs[r,0].transAxes,fontsize=16,fontweight="bold"); fd=d[d.family==f]
        for c,st in enumerate(range(1,NUM_STEPS+1)):
            ax=axs[r,c]; ax.set_title(f"Step {st}",color="#666"); ax.set_xlim(.9,max(LORA_EPOCHS,RANKEXT_EPOCHS)+.1); ax.set_xticks(range(1,max(LORA_EPOCHS,RANKEXT_EPOCHS)+1)); ax.grid(True,axis="y",color="#ddd");
            if c==0: ax.set_ylabel(ylabel)
            if r==1: ax.set_xlabel("Local epoch")
            if log: ax.set_yscale("log")
            for v in SUPERVISOR_VARIANT_ORDER:
                s=fd[(fd.cl_step==st)&(fd.variant==v)].sort_values("local_epoch"); y=pd.to_numeric(s[metric],errors="coerce"); good=np.isfinite(y)&((y>0) if pos else True)
                if len(s)>0 and good.any(): ax.plot(s.local_epoch[good], y[good], color=VCOL[v], linestyle=VSTYLE[v], lw=2.4)
    fig.legend([Line2D([0],[0],color=VCOL[v],linestyle=VSTYLE[v],lw=3) for v in SUPERVISOR_VARIANT_ORDER], SUPERVISOR_VARIANT_ORDER, loc="center left", bbox_to_anchor=(.915,.52), frameon=False); fig.tight_layout(rect=[.02,.02,.90,.95]); plt.savefig(Path(PLOTS_DIR)/name,dpi=DPI,bbox_inches="tight"); plt.close()
if len(E)>0:
    lossgrid("train_ce_loss","Train CE loss","train_ce_loss_by_method.png","Train CE Loss by Method and CL Step")
    lossgrid("val_ce_loss","Validation CE loss","validation_ce_loss_by_method.png","Validation CE Loss by Method and CL Step"); lossgrid("val_ce_loss","Validation CE loss","validation_ce_loss_clean.png","Validation CE Loss by CL Step")
    kd=[m for m in REQ if ACTIVE_METHOD_MAP[m]["uses_kd"]]; fo=[m for m in REQ if ACTIVE_METHOD_MAP[m]["uses_factor_orth"]]
    lossgrid("kd_loss_weighted","Weighted KD loss","kd_loss_by_method.png","KD Loss by Method",kd,pos=True); lossgrid("factor_orth_loss_weighted","Weighted factor-orth loss","factor_orth_loss_by_method.png","Factor-Orth Loss by Method",fo,pos=True)
    lossgrid("factor_orth_loss_weighted","Weighted factor-orth loss","factor_orth_weighted_loss_log.png","Factor-Orth Weighted Loss (log)",fo,log=True,pos=True); lossgrid("train_total_loss","Total train loss","total_loss_by_method.png","Total Loss by Method"); lossgrid("train_total_loss","Total train loss","total_loss_by_method_log.png","Total Loss by Method (log)",log=True,pos=True)
    # train-val combined
    T=E.copy(); T["variant"]=T.method.map(variant); T["family"]=T.method.map(fam); fig,axs=plt.subplots(2,NUM_STEPS,figsize=(18,8)); fig.suptitle("Train CE vs Validation CE by Method and CL Step",fontsize=22,fontweight="bold",y=.995)
    for r,f in enumerate(FAMS):
        axs[r,0].text(-.35,1.15,FLAB[f],transform=axs[r,0].transAxes,fontsize=16,fontweight="bold"); fd=T[T.family==f]
        for c,st in enumerate(range(1,NUM_STEPS+1)):
            ax=axs[r,c]; ax.set_title(f"Step {st}"); ax.grid(True,axis="y",color="#ddd");
            if c==0: ax.set_ylabel("CE loss")
            if r==1: ax.set_xlabel("Local epoch")
            for v in SUPERVISOR_VARIANT_ORDER:
                s=fd[(fd.cl_step==st)&(fd.variant==v)].sort_values("local_epoch")
                if len(s)>0: ax.plot(s.local_epoch,s.train_ce_loss,color=VCOL[v],linestyle=VSTYLE[v],lw=2.3); ax.plot(s.local_epoch,s.val_ce_loss,color=VCOL[v],linestyle=VSTYLE[v],lw=2.0,alpha=.45)
    fig.tight_layout(rect=[.02,.02,.90,.95]); plt.savefig(Path(PLOTS_DIR)/"train_val_ce_loss_by_method.png",dpi=DPI,bbox_inches="tight"); plt.close()
    # STRICT-REVIEW REDESIGN (2026-07-17, analysis_strict_review/report.txt A1):
    # the previous layout (rows=loss component, cols=family, all 4 variants
    # overlaid per cell) put a variant's CE and its Total on two DIFFERENT
    # panels with two DIFFERENT y-scales (CE linear, Total log) and no legend.
    # For the Base variant, Total IS CE (KD=orth=0 identically), so any visual
    # difference between those two panels was 100% a rendering artifact of
    # axis choice, never a real difference in the data -- proved numerically in
    # analysis_strict_review/report.txt Part A1a (same 18 numbers, two axes).
    # Fix: one panel PER VARIANT, all 4 quantities (CE, KD weighted, Factor-
    # Orth weighted, Total) plotted TOGETHER on the SAME (necessarily log,
    # since components span 1e-4 to 1e4) axis, so "Total = sum of the other
    # three" is checkable by eye in a single panel instead of inferred across
    # panels. Grid is now rows=variant (4), cols=family (2) -- same 8-panel
    # footprint as before.
    C=E.copy(); C["variant"]=C.method.map(variant); C["family"]=C.method.map(fam)
    method_by_family_variant={(fam(m),variant(m)):m for m in REQ}
    line_specs=[
        ("train_ce_loss","Train CE","#1f77b4","-",2.0),
        ("kd_loss_weighted","KD weighted","#ff7f0e","--",2.0),
        ("factor_orth_loss_weighted","Factor-Orth weighted","#d62728",":",2.0),
        ("train_total_loss","TOTAL (= sum of the above)","#000000","-",3.0),
    ]
    fig,axs=plt.subplots(len(SUPERVISOR_VARIANT_ORDER),2,figsize=(16,15),sharex=True)
    fig.suptitle("Combined Loss Decomposition -- per-variant panels\n"
                 "(all lines share ONE log-scale y-axis per panel; TOTAL is plotted, never a separate scale, so 'TOTAL = sum of the other lines' is directly checkable by eye)",
                 fontsize=16,fontweight="bold",y=.995)
    for rr,v in enumerate(SUPERVISOR_VARIANT_ORDER):
        for cc,f in enumerate(FAMS):
            ax=axs[rr,cc]
            m=method_by_family_variant.get((f,v))
            if rr==0: ax.set_title(FLAB[f],fontweight="bold")
            if cc==0: ax.set_ylabel(f"{v}\n(log scale)")
            ax.grid(True,axis="y",color="#ddd"); ax.set_yscale("log")
            for b in range(LORA_EPOCHS,NUM_STEPS*LORA_EPOCHS,LORA_EPOCHS): ax.axvline(b+.5,color="#bbb",linestyle=":",lw=1)
            fd=C[(C.family==f)&(C.method==m)] if m is not None else C.iloc[0:0]
            for st in range(1,NUM_STEPS+1):
                s=fd[fd.cl_step==st].sort_values("local_epoch")
                if len(s)==0: continue
                x=(st-1)*LORA_EPOCHS+s.local_epoch.astype(float)
                for met,lab,color,ls,lw in line_specs:
                    y=pd.to_numeric(s.get(met,np.nan),errors="coerce"); good=np.isfinite(y)&(y>0)
                    if good.any(): ax.plot(x[good],y[good],color=color,linestyle=ls,lw=lw)
            ax.set_xticks([(i*LORA_EPOCHS)+2 for i in range(NUM_STEPS)]); ax.set_xticklabels([f"S{i}" for i in range(1,NUM_STEPS+1)])
    legend_handles=[Line2D([0],[0],color=color,linestyle=ls,lw=max(lw,2.5)) for _,lab,color,ls,lw in line_specs]
    legend_labels=[lab for _,lab,_,_,_ in line_specs]
    fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(.915,.52), frameon=False, title="Line (all 4 on\nthe same axis)")
    fig.tight_layout(rect=[.02,.02,.90,.94]); plt.savefig(Path(PLOTS_DIR)/"combined_loss_decomposition.png",dpi=DPI,bbox_inches="tight"); plt.close()
else: missing_outputs.append({"output":"loss plots","method":"all","metric_or_column":"training_loss_history_by_epoch","why":"No epoch-level rows available","required_or_optional":"required"})

def valdiag():
    rows=[]; alook=M.set_index("method") if len(M)>0 else pd.DataFrame()
    for m in REQ:
        d=E[E.method==m].sort_values(["cl_step","local_epoch"]); v=d.dropna(subset=["val_ce_loss"])
        if len(v)==0: rows.append({"method":m,"display_method_name":disp(m),"overfitting_signal":"missing_validation"}); continue
        final=v.iloc[-1]; best=v.sort_values(["val_ce_loss","global_epoch"]).iloc[0]; inc=0
        for _,g in v.groupby("cl_step"):
            prev=None
            for _,r in g.sort_values("local_epoch").iterrows():
                if prev is not None and r.val_ce_loss>prev.val_ce_loss and r.train_ce_loss<prev.train_ce_loss: inc+=1
                prev=r
        fg=float(final.val_ce_loss-final.train_ce_loss); fmb=float(final.val_ce_loss-best.val_ce_loss); vr=float(v.val_ce_loss.max()-v.val_ce_loss.min()); flags=[]
        if inc: flags.append("val_up_train_down")
        if fg>1: flags.append("large_final_gap")
        if fmb>.25: flags.append("final_val_worse_than_best")
        if vr>1: flags.append("unstable_val_ce")
        sig="low" if not flags else ("strong" if len(flags)>1 else "moderate")
        rows.append({"method":m,"display_method_name":disp(m),"all_seen_accuracy":float(alook.loc[m,"all_seen_accuracy"]) if m in alook.index else np.nan,"final_validation_ce":float(final.val_ce_loss),"best_validation_ce":float(best.val_ce_loss),"global_epoch_of_best_validation_ce":int(best.global_epoch),"cl_step_of_best_validation_ce":int(best.cl_step),"local_epoch_of_best_validation_ce":int(best.local_epoch),"final_train_ce":float(final.train_ce_loss),"train_val_ce_gap_final_epoch":fg,"train_val_ce_gap_best_val_epoch":float(best.val_ce_loss-best.train_ce_loss),"validation_ce_std":float(v.val_ce_loss.std(ddof=0)),"validation_ce_range":vr,"validation_ce_trend":"decreasing" if final.val_ce_loss<v.iloc[0].val_ce_loss else "increasing","validation_ce_increases_while_train_ce_decreases":bool(inc),"num_val_up_train_down_events":inc,"final_validation_ce_minus_best":fmb,"overfitting_signal":sig,"overfitting_flags":";".join(flags) if flags else "none","overfitting_score":max(fg,0)+max(fmb,0)+.25*inc+.25*vr})
    D=pd.DataFrame(rows); D.to_csv(Path(TABLES_DIR)/"validation_diagnostics_by_method.csv",index=False); D.sort_values("best_validation_ce").to_csv(Path(TABLES_DIR)/"validation_ranking_by_best_val_ce.csv",index=False); D.sort_values("final_validation_ce").to_csv(Path(TABLES_DIR)/"validation_ranking_by_final_val_ce.csv",index=False); D.sort_values("train_val_ce_gap_final_epoch",ascending=False).to_csv(Path(TABLES_DIR)/"train_val_gap_by_method.csv",index=False); return D
D=valdiag()
for m in REQ:
    dd=D[D.method==m] if len(D)>0 and "method" in D else pd.DataFrame()
    if len(dd)==0 or "final_validation_ce" not in dd or dd["final_validation_ce"].isna().all():
        missing_outputs.append({"output":"tables/validation_diagnostics_by_method.csv","method":m,"metric_or_column":"final_validation_ce / val_ce_loss","why":"Validation CE was not available for this selected method.","required_or_optional":"required"})
if len(D)>0 and "final_validation_ce" in D:
    y=np.arange(len(D.sort_values("train_val_ce_gap_final_epoch"))); P=D.sort_values("train_val_ce_gap_final_epoch"); plt.figure(figsize=(12,6)); plt.barh(y,P.train_val_ce_gap_final_epoch); plt.yticks(y,P.display_method_name); plt.xlabel("Validation CE - Train CE"); plt.title("Train-Validation CE Gap by Method"); figsave("train_val_ce_gap_by_method.png")
    P=D.sort_values("best_validation_ce",ascending=False); y=np.arange(len(P)); plt.figure(figsize=(12,6)); plt.hlines(y,P.best_validation_ce,P.final_validation_ce,color="#999"); plt.scatter(P.best_validation_ce,y,label="best"); plt.scatter(P.final_validation_ce,y,label="final"); plt.yticks(y,P.display_method_name); plt.xlabel("Validation CE"); plt.title("Best vs Final Validation CE"); plt.legend(); figsave("best_vs_final_validation_ce.png")
    plt.figure(figsize=(10,7));
    for _,r in D.iterrows(): plt.scatter(r.final_validation_ce,r.all_seen_accuracy,s=90); plt.annotate(r.display_method_name,(r.final_validation_ce,r.all_seen_accuracy),xytext=(5,4),textcoords="offset points",fontsize=9)
    plt.xlabel("Final validation CE loss"); plt.ylabel("All-seen accuracy (%)"); plt.title("All-Seen Accuracy vs Final Validation CE"); plt.grid(True,color="#ddd"); figsave("accuracy_vs_validation_ce.png")
# Hyperparameter check
HP=CFG[["method","display_method_name","lora_rank","lora_alpha","lora_dropout","target_modules","num_epochs","learning_rate","batch_size","lambda_orth","kd_temperature","optimizer","scheduler","seed"]].copy(); HP.to_csv(Path(TABLES_DIR)/"hyperparameter_consistency_check.csv",index=False)
hp_note="Delta-trace and factor-orth variants use the same main hyperparameters when matched by family and KD temperature: LoRA rank/alpha/dropout, target modules, epochs, LR, batch size, optimizer, scheduler, KD temperature and KD weight. If simple_avg_delta_trace outperforms simple_avg_factor_orth, the difference is therefore more likely due to orthogonality formulation and loss scale than hyperparameter mismatch."
txt(Path(REPORTS_DIR)/"hyperparameter_consistency_notes.txt", "Hyperparameter consistency notes\n================================\n\n"+hp_note)
# Reports
if len(D)>0:
    ba=D.sort_values("all_seen_accuracy",ascending=False).iloc[0]; bv=D.sort_values("best_validation_ce").iloc[0]; bf=D.sort_values("final_validation_ce").iloc[0]; st=D.sort_values(["validation_ce_std","validation_ce_range"]).iloc[0]; of=D.sort_values("overfitting_score",ascending=False).iloc[0]
    val_report=f"""Validation-based result report\n==============================\n\nBest method by all-seen accuracy: {ba.display_method_name} ({ba.method}), {ba.all_seen_accuracy:.2f}%.\nBest method by best validation CE: {bv.display_method_name} ({bv.method}), {bv.best_validation_ce:.4f}.\nBest method by final validation CE: {bf.display_method_name} ({bf.method}), {bf.final_validation_ce:.4f}.\nMost stable method by validation CE: {st.display_method_name} ({st.method}), std={st.validation_ce_std:.4f}.\nStrongest overfitting signal: {of.display_method_name} ({of.method}), signal={of.overfitting_signal}, flags={of.overfitting_flags}.\n\nDo not judge methods only by final/test accuracy. High all-seen accuracy with high final validation CE indicates weaker validation behavior; low accuracy with low/stable validation CE indicates cleaner training dynamics but weaker final task performance. Validation CE is epoch-level only, so overfitting detection is useful but coarse.\n"""
else: val_report="Validation CE missing; validation-based ranking cannot be computed."
txt(Path(REPORTS_DIR)/"validation_based_result_report.txt", val_report)
miss="\n".join([f"- output: {x['output']}\n  method: {x['method']}\n  metric/column/file: {x['metric_or_column']}\n  why: {x['why']}\n  required_or_optional: {x['required_or_optional']}" for x in missing_outputs]) or "No required outputs were silently skipped."
txt(Path(REPORTS_DIR)/"missing_outputs_or_metrics.txt", "Missing outputs or metrics\n==========================\n\n"+miss)
files=[]
for root in [TABLES_DIR,PLOTS_DIR,REPORTS_DIR,LOGS_DIR,CONFIGS_DIR]: files += [str(x.relative_to(BASE_OUTPUT_DIR)) for x in sorted(Path(root).glob('*')) if x.is_file()]
summary=f"""Supervisor summary report\n=========================\n\nOfficial methods:\n{chr(10).join('- '+m for m in REQ)}\n\nMissing-method confirmation:\n- simple_avg_factor_orth included: {'simple_avg_factor_orth' in REQ}\n- simple_avg_factor_orth_kd_T2 included: {'simple_avg_factor_orth_kd_T2' in REQ}\n\nFinal accuracy ranking:\n{M.sort_values('all_seen_accuracy',ascending=False).to_string(index=False)}\n\nValidation ranking:\n{D.sort_values('final_validation_ce').to_string(index=False) if len(D)>0 else 'No validation rows.'}\n\nCE, KD, factor-orth, and total losses are logged/plotted with CL-step separation. Hyperparameter consistency answer: {hp_note}\n\nGenerated files:\n{chr(10).join('- '+f for f in files)}\n\nSupervisor requests are satisfied unless listed in reports/missing_outputs_or_metrics.txt.\n"""
txt(Path(REPORTS_DIR)/"supervisor_summary_report.txt", summary)
# Checklist
required=["tables/training_loss_history_by_epoch.csv","tables/supervisor_selected_accuracy_comparison.csv","tables/final_metrics_all_methods.csv","tables/validation_diagnostics_by_method.csv","tables/validation_ranking_by_best_val_ce.csv","tables/validation_ranking_by_final_val_ce.csv","tables/train_val_gap_by_method.csv","tables/hyperparameter_consistency_check.csv","tables/best_epoch_selected_by_method_step.csv","tables/per_step_accuracy_by_method.csv","plots/train_ce_loss_by_method.png","plots/validation_ce_loss_by_method.png","plots/train_val_ce_loss_by_method.png","plots/kd_loss_by_method.png","plots/factor_orth_loss_by_method.png","plots/total_loss_by_method.png","plots/combined_loss_decomposition.png","plots/supervisor_method_step_accuracy_heatmap.png","plots/supervisor_method_metric_heatmap.png","plots/per_task_accuracy_heatmap.png","plots/forgetting_curve_by_method.png","plots/accuracy_vs_validation_ce.png","plots/train_val_ce_gap_by_method.png","reports/validation_based_result_report.txt","reports/hyperparameter_consistency_notes.txt","reports/supervisor_summary_report.txt"]
lines=["Final supervisor-output checklist","=================================",""]; all_ok=True
for r in required:
    good=ok(Path(BASE_OUTPUT_DIR)/r); all_ok=all_ok and good; lines.append(("PASS " if good else "FAIL ")+r)
lines += ["", "OVERALL "+("PASS" if all_ok else "FAIL")]
check="\n".join(lines); print(check); txt(Path(REPORTS_DIR)/"output_checklist.txt", check); print("Supervisor-ready output directory:", BASE_OUTPUT_DIR)

