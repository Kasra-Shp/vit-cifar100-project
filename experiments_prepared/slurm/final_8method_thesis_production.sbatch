#!/bin/bash
# =============================================================================
# FINAL 8-METHOD CIFAR-100 5x20 THESIS PRODUCTION EXPERIMENT -- REAL RUN
# LAUNCHER. CREATED ONLY -- NOT SUBMITTED.
#
# 8 training arms (4 SimpleAvg + 4 RankExt), BOTH families at 9 epochs/CL
# step, CIFAR-100 5x20 only. Consolidates the validated decisions from the
# SimpleAvg DenseOrth-lambda investigation and the RankExt head-LR decision
# probes into one clean benchmark:
#   SimpleAvg: plain, old-seen KD T=2 (weight as historically resolved by
#              the source script -- see R7/final_8method_thesis_production_
#              readiness.md for the exact verified value and any discrepancy
#              from the task brief's stated 0.5), DenseOrth lambda=1
#              (LOWERED from historical lambda=20 -- the primary scientific
#              change to this family), DenseOrth lambda=1 + old-seen KD T=2.
#              Head LR multiplier UNCHANGED at x10.
#   RankExt:   plain, HISTORICAL full-100-way KD (T=2, weight=1.0, no
#              warmup, protect_weight=30, new-block warmup disabled),
#              HISTORICAL FactorOrth lambda=50, FactorOrth50 + full-KD
#              combined. Head LR multiplier RAISED 1.0 -> 2.0 (the primary
#              scientific change to this family, applied uniformly to all
#              4 RankExt arms) -- evidence: R7/rankext_headlr_final_compare_
#              report.md. NEW task-scale calibration (job4972616) is ABSENT
#              (runtime-asserted); historical row-norm calibration is
#              retained unchanged.
# See R7/final_8method_thesis_production_readiness.md for the full config-
# diff audit against job4971615 and the static validation this script
# underwent before this launcher was written.
#
# Persistence/resume is built into the target script itself (CHECKPOINTS_DIR,
# isolated under this script's own RUN_NAME_BASE
# "cifar100_5x20_final_8method_thesis_production_seed42" -- never overlaps
# with job4971615's, job 4969059's, job 4970580's, or any other prior run's
# checkpoints/results) -- this launcher does not touch results/ or
# checkpoints in any way beyond running the script.
#
# No conda/module activation line is added here: the python interpreter is
# invoked via its absolute path (PYTHON below), so no shell env activation
# step is needed to put the right interpreter on PATH.
#
# 16-hour walltime (vs. job4971615's 18 hours for 9 arms): one fewer
# SimpleAvg arm (T4 removed), both families otherwise unchanged in
# per-arm training cost (DenseOrth's lambda coefficient and RankExt's head-
# LR multiplier do not change per-batch compute, only loss magnitude/
# optimizer step size).
# =============================================================================
#SBATCH --job-name=final_8method_thesis_production
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=final_8method_thesis_production_%j.out
#SBATCH --error=final_8method_thesis_production_%j.err

set -euo pipefail

# Explicit, absolute repo path -- do NOT derive it from `dirname "$0"`. SLURM
# may execute a COPY of this script from a spool directory, in which case
# `dirname "$0"` would resolve there instead of the repo.
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_thesis_production.py

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
# test -- this is the full production benchmark) -- unset defensively
# regardless.
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true

echo "================================================================================"
echo "FINAL 8-METHOD CIFAR-100 5x20 THESIS PRODUCTION EXPERIMENT (REAL RUN)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $PYTHON"
echo "================================================================================"

export PYTHONUNBUFFERED=1

"$PYTHON" -u "$SCRIPT"

echo "================================================================================"
echo "FINAL 8-METHOD CIFAR-100 5x20 THESIS PRODUCTION EXPERIMENT finished. End: $(date -Is)"
echo "================================================================================"
