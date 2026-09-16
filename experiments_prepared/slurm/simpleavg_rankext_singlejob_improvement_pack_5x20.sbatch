#!/bin/bash
# =============================================================================
# SIMPLEAVG + RANKEXT JOINT SINGLE-JOB IMPROVEMENT PACK -- 5x20, seed 42.
# REAL RUN LAUNCHER. The GPU smoke/preflight test (cluster job 4968964) has
# already passed at the experiment level (all 9 training arms + all 14
# evaluated configurations completed, reporting completed, plot_failures=0,
# final supervisor-output checklist = OVERALL PASS). The only failure was in
# this launcher's own external structural gate ("Expected 14 rows, got 10"),
# which was checking the wrong file (simpleavg_singlejob_training_x_merge_
# results.csv only ever holds the 5 SimpleAvg arms x 2 merge strategies = 10
# rows; the 4 RankExt persistent results live in the general final metrics/
# reporting tables). That preflight-phase + gate logic has been removed
# entirely -- this launcher now starts the real experiment directly, once.
#
# One cluster job, 9 training arms (5 SimpleAvg + 4 RankExt), 14 evaluated
# configurations (SimpleAvg: 5 arms x {arithmetic, do_merge}; RankExt: 4 arms
# x {persistent}), 6 epochs for every arm. SimpleAvg rank=80/alpha=160/
# scaling=2; RankExt schedule [16,32,48,64,80]/scaling=2. Persistence/resume
# is built into the target script itself (CHECKPOINTS_DIR, isolated per
# RUN_NAME_BASE, stable across restarts, config-fingerprint-protected against
# stale/incompatible checkpoints) -- this launcher does not touch results/ or
# checkpoints in any way.
#
# No conda/module activation line is added here: the python interpreter is
# invoked via its absolute path (PYTHON below), so no shell env activation
# step is needed to put the right interpreter on PATH.
# =============================================================================
#SBATCH --job-name=sa_re_improve_5x20
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=sa_re_improve_5x20_%j.out
#SBATCH --error=sa_re_improve_5x20_%j.err

set -euo pipefail

# Explicit, absolute repo path -- do NOT derive it from `dirname "$0"`. SLURM
# may execute a COPY of this script from a spool directory, in which case
# `dirname "$0"` would resolve there instead of the repo.
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/simpleavg_rankext_singlejob_improvement_pack_5x20.py

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
echo "SIMPLEAVG + RANKEXT JOINT SINGLE-JOB IMPROVEMENT PACK -- 5x20, seed 42 (REAL RUN)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $PYTHON"
echo "================================================================================"

export PYTHONUNBUFFERED=1

"$PYTHON" -u "$SCRIPT"

echo "================================================================================"
echo "SIMPLEAVG + RANKEXT JOINT SINGLE-JOB IMPROVEMENT PACK finished. End: $(date -Is)"
echo "================================================================================"
