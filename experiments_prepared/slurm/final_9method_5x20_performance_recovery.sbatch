#!/bin/bash
# =============================================================================
# FINAL 9-METHOD CIFAR-100 5x20 PERFORMANCE-RECOVERY EXPERIMENT -- REAL RUN
# LAUNCHER. CREATED ONLY -- NOT SUBMITTED.
#
# 9 training arms (5 SimpleAvg + 4 RankExt), BOTH families at 9 epochs/CL
# step (matching the historical Exp1/R7 protocol), CIFAR-100 5x20 only:
#   SimpleAvg: plain, old-seen KD T=2 (weight=1.0, 1-epoch warmup), old-seen
#              KD T=4 (same, T-only ablation), DenseOrth lambda=20, DenseOrth
#              lambda=20 + old-seen KD T=2.
#   RankExt:   plain, HISTORICAL full-100-way KD (T=2, weight=1.0, no warmup,
#              protect_weight=30, new-block warmup disabled), HISTORICAL
#              FactorOrth lambda=50, FactorOrth50 + full-KD combined (the
#              main final RankExt candidate).
# See R7/final_5x20_performance_recovery_plan.md for the full design
# rationale and R7/final_9method_5x20_static_readiness.md for the static
# validation this script and its target underwent before this launcher was
# written.
#
# Persistence/resume is built into the target script itself (CHECKPOINTS_DIR,
# isolated under this script's own RUN_NAME_BASE
# "cifar100_5x20_final_9method_performance_recovery_seed42" -- never overlaps
# with job 4969059's, job 4970580's, or any other prior run's
# checkpoints/results) -- this launcher does not touch results/ or
# checkpoints in any way beyond running the script.
#
# No conda/module activation line is added here: the python interpreter is
# invoked via its absolute path (PYTHON below), so no shell env activation
# step is needed to put the right interpreter on PATH.
#
# 18-hour walltime (vs. the 8-arm/6-epoch script's 12 hours): 9 methods
# instead of 8, both families at 9 epochs instead of 6 (a 1.5x per-arm
# training-time increase), and the two new RankExt KD arms carry heavier
# per-batch cost than old-seen KD (a full-100-way softmax/KL every KD batch,
# plus the protect_weight projected-feature term's extra hidden-state
# forward pass on those same batches).
# =============================================================================
#SBATCH --job-name=final_9method_5x20_perf_recovery
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=final_9method_5x20_performance_recovery_%j.out
#SBATCH --error=final_9method_5x20_performance_recovery_%j.err

set -euo pipefail

# Explicit, absolute repo path -- do NOT derive it from `dirname "$0"`. SLURM
# may execute a COPY of this script from a spool directory, in which case
# `dirname "$0"` would resolve there instead of the repo.
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_9method_5x20_performance_recovery.py

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
echo "FINAL 9-METHOD CIFAR-100 5x20 PERFORMANCE-RECOVERY EXPERIMENT (REAL RUN)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $PYTHON"
echo "================================================================================"

export PYTHONUNBUFFERED=1

"$PYTHON" -u "$SCRIPT"

echo "================================================================================"
echo "FINAL 9-METHOD CIFAR-100 5x20 PERFORMANCE-RECOVERY EXPERIMENT finished. End: $(date -Is)"
echo "================================================================================"
