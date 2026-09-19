#!/bin/bash
# =============================================================================
# FINAL 8-METHOD CIFAR-100 5x20 REFINED EXPERIMENT -- REAL RUN LAUNCHER.
# CREATED ONLY -- NOT SUBMITTED.
#
# 8 training arms (4 SimpleAvg + 4 RankExt), BOTH families at 9 epochs/CL
# step (matching the historical Exp1/R7 protocol), CIFAR-100 5x20 only.
# Derived from experiments_prepared/final_9method_5x20_performance_recovery.py
# (job 4971615's script, UNCHANGED at its own path) by applying exactly three
# refinements, all documented at the top of the target script itself and in
# R7/final_8method_refined_readiness.md:
#   (a) removed the simple_avg_kd_oldseen_T4_warmup arm (8 methods, not 9)
#   (b) unified head-classifier LR multiplier to 1.0 for BOTH families
#       (see R7/head_lr_final_decision.md)
#   (c) added a validation-fitted, per-task-group logit-scale calibration
#       stage after RankExt's existing row-norm calibration (RankExt only,
#       no new method/column -- see the target script's own header)
#
#   SimpleAvg: plain, old-seen KD T=2 (weight=1.0, 1-epoch warmup),
#              DenseOrth lambda=20, DenseOrth lambda=20 + old-seen KD T=2.
#   RankExt:   plain, HISTORICAL full-100-way KD (T=2, weight=1.0, no warmup,
#              protect_weight=30, new-block warmup disabled), HISTORICAL
#              FactorOrth lambda=50, FactorOrth50 + full-KD combined (the
#              main final RankExt candidate) -- all 4 RankExt arms additionally
#              receive the new post-hoc logit-scale calibration as part of
#              their own evaluation.
# See R7/final_8method_refined_readiness.md for the full static-validation
# readiness audit this script and its target underwent before this launcher
# was written.
#
# Persistence/resume is built into the target script itself (CHECKPOINTS_DIR,
# isolated under this script's own RUN_NAME_BASE
# "cifar100_5x20_final_8method_refined_seed42" -- never overlaps with job
# 4969059's, job 4970580's, job 4971615's, or any other prior run's
# checkpoints/results) -- this launcher does not touch results/ or
# checkpoints in any way beyond running the script.
#
# No conda/module activation line is added here: the python interpreter is
# invoked via its absolute path (PYTHON below), so no shell env activation
# step is needed to put the right interpreter on PATH.
#
# 18-hour walltime, unchanged from the 9-method launcher: 8 methods instead
# of 9 is a modest per-run reduction, but both families still train at 9
# epochs/step and the two RankExt full-KD+protect30 arms still carry their
# full per-batch cost; the new post-hoc calibration stage adds only a small,
# fixed number of forward-only passes per RankExt arm at the very end of the
# run (no backbone/LoRA/classifier gradient steps), which is negligible next
# to training time -- the walltime is kept unchanged as a safety margin
# rather than tightened based on an unverified estimate.
# =============================================================================
#SBATCH --job-name=final_8method_5x20_refined
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=final_8method_5x20_refined_%j.out
#SBATCH --error=final_8method_5x20_refined_%j.err

set -euo pipefail

# Explicit, absolute repo path -- do NOT derive it from `dirname "$0"`. SLURM
# may execute a COPY of this script from a spool directory, in which case
# `dirname "$0"` would resolve there instead of the repo.
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_5x20_refined.py

# --- Minimal safe startup validation ---
if [ ! -x "$PYTHON" ]; then
    echo "FATAL: $PYTHON not found or not executable." >&2
    exit 1
fi
if [ ! -f "$SCRIPT" ]; then
    echo "FATAL: $SCRIPT not found under $(pwd)." >&2
    exit 1
fi

# Guard against an inherited debug/seed override from the submitting shell --
# the real run must see FAST_RUN==False and seed==42 (the script's own
# defaults), never a leftover smoke-test or replication-seed override. This
# script is never intended to run in FAST_RUN_DEBUG mode at all (no smoke
# test, per the task brief) -- unset defensively regardless.
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true

echo "================================================================================"
echo "FINAL 8-METHOD CIFAR-100 5x20 REFINED EXPERIMENT (REAL RUN)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $PYTHON"
echo "================================================================================"

export PYTHONUNBUFFERED=1

"$PYTHON" -u "$SCRIPT"

echo "================================================================================"
echo "FINAL 8-METHOD CIFAR-100 5x20 REFINED EXPERIMENT finished. End: $(date -Is)"
echo "================================================================================"
