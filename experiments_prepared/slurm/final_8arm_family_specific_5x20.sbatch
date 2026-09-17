#!/bin/bash
# =============================================================================
# FINAL 8-ARM FAMILY-SPECIFIC EXPERIMENT -- 5x20, seed 42. REAL RUN LAUNCHER.
#
# Restores the clean 4-arm factorial (Plain / KD / Orth / Orth+KD) for EACH
# family, using FAMILY-SPECIFIC hyperparameters on top of job 4969059's
# current corrected pipeline:
#   SimpleAvg: DenseOrth lambda=20 (unchanged), KD weight=0.5, no KD warmup.
#   RankExt:   HISTORICAL FactorOrth lambda=50 (restored), KD weight=1.5,
#              1-epoch KD warmup.
# One cluster job, 8 training arms (4 SimpleAvg + 4 RankExt), 8 evaluated
# configurations (SimpleAvg: 4 arms x {arithmetic} only -- DO-Merge removed;
# RankExt: 4 arms x {persistent}), 6 epochs for every arm. SimpleAvg
# rank=80/alpha=160/scaling=2; RankExt schedule [16,32,48,64,80]/scaling=2.
# See R7/final_8arm_family_specific_implementation_audit.md for the full
# design/porting/verification writeup and
# R7/lambda_and_kd_weight_design_audit.md for the hyperparameter rationale.
#
# Persistence/resume is built into the target script itself (CHECKPOINTS_DIR,
# isolated under this script's own RUN_NAME_BASE
# "cifar100_5x20_final_8arm_family_specific_seed42" -- never overlaps with
# job 4969059's or any other prior run's checkpoints/results) -- this
# launcher does not touch results/ or checkpoints in any way beyond running
# the script.
#
# No conda/module activation line is added here: the python interpreter is
# invoked via its absolute path (PYTHON below), so no shell env activation
# step is needed to put the right interpreter on PATH.
# =============================================================================
#SBATCH --job-name=final_8arm_family_specific
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=final_8arm_family_specific_5x20_%j.out
#SBATCH --error=final_8arm_family_specific_5x20_%j.err

set -euo pipefail

# Explicit, absolute repo path -- do NOT derive it from `dirname "$0"`. SLURM
# may execute a COPY of this script from a spool directory, in which case
# `dirname "$0"` would resolve there instead of the repo.
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8arm_family_specific_5x20.py

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
# defaults), never a leftover smoke-test or replication-seed override.
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true

echo "================================================================================"
echo "FINAL 8-ARM FAMILY-SPECIFIC EXPERIMENT -- 5x20, seed 42 (REAL RUN)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $PYTHON"
echo "================================================================================"

export PYTHONUNBUFFERED=1

"$PYTHON" -u "$SCRIPT"

echo "================================================================================"
echo "FINAL 8-ARM FAMILY-SPECIFIC EXPERIMENT finished. End: $(date -Is)"
echo "================================================================================"
