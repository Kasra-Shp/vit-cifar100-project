#!/bin/bash
# =============================================================================
# RANKEXT HEAD-LR SMALL PROBE -- diagnostic-only LAUNCHER. CREATED ONLY --
# NOT SUBMITTED (no cluster/SLURM access from the environment that prepared
# this file; see R7/rankext_headlr_small_probe_readiness.md).
#
# THIS IS NOT A BENCHMARK RUN. It is a small, controlled sweep of RankExt's
# head/classifier LR multiplier, restored to the job4971615 (pre-task-scale-
# calibration) RankExt pipeline. Runs experiments_prepared/
# rankext_headlr_small_probe.py once per LR-multiplier candidate in this same
# job allocation (a bash loop, not separate sbatch submissions), each
# invocation training ONLY the 2 RankExt arms under test
# (rank_extension_factor_orth_lam50_fullkd_T2_protect30 [primary],
# rank_extension_factor_orth_lam50 [secondary]) over a reduced 3x20 protocol
# (Task1, Task1->2 transition, Task2->3 transition) at 2 epochs/step instead
# of the full 5x20/9-epoch protocol. SimpleAvg and every other RankExt
# variant are OFF (see METHODS_TO_RUN in the probe script). Does NOT run,
# import, or modify experiments_prepared/final_9method_5x20_performance_
# recovery.py or experiments_prepared/final_8method_5x20_refined.py.
#
# Each LR value writes to its own isolated RUN_NAME_BASE (see the probe
# script's RUN_NAME_BASE construction) -- no cross-LR overwrite.
#
# LR candidate set: 0.5, 1.0 (historical job4971615 baseline), 2.0 -- the
# small, conservative sweep specified by the investigation (see
# R7/rankext_headlr_small_probe_readiness.md Section 5 for why no wider
# range is used; historical evidence flags x10 as a plausible instability
# amplifier for RankExt+FactorOrth, so it is never swept here as a preferred
# candidate).
# =============================================================================
#SBATCH --job-name=rankext_headlr_small_probe
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=rankext_headlr_small_probe_%j.out
#SBATCH --error=rankext_headlr_small_probe_%j.err

set -euo pipefail

# Explicit, absolute repo path -- do NOT derive it from `dirname "$0"`. SLURM
# may execute a COPY of this script from a spool directory, in which case
# `dirname "$0"` would resolve there instead of the repo.
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/rankext_headlr_small_probe.py

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
# every LR condition below must see the SAME seed=42, FAST_RUN=False (no
# smoke-mode shortcut beyond the probe's own built-in NUM_STEPS/RANKEXT_EPOCHS
# reduction), so conditions are compared under truly identical settings
# (spec Section 13: "Do NOT change other hyperparameters to rescue an LR").
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true

echo "================================================================================"
echo "RANKEXT HEAD-LR SMALL PROBE (DIAGNOSTIC ONLY -- NOT A THESIS BENCHMARK)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $PYTHON"
echo "================================================================================"

export PYTHONUNBUFFERED=1

# Small, conservative candidate set (spec Section 5). Add "5.0" here only if
# the 0.5/1.0/2.0 sweep completes comfortably within the walltime above and a
# 4th point is still wanted; "10.0" is never added as a preferred candidate,
# only ever as an explicitly-labeled historical-stress reference if a
# supervisor specifically asks for it.
LR_MULTIPLIERS=(0.5 1.0 2.0)

for LR in "${LR_MULTIPLIERS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "PROBE CONDITION: RANKEXT_HEADLR_PROBE_MULTIPLIER=$LR   Start: $(date -Is)"
    echo "--------------------------------------------------------------------------------"
    RANKEXT_HEADLR_PROBE_MULTIPLIER="$LR" "$PYTHON" -u "$SCRIPT"
    echo "PROBE CONDITION $LR finished: $(date -Is)"
done

echo "================================================================================"
echo "RANKEXT HEAD-LR SMALL PROBE finished. End: $(date -Is)"
echo "SMOKE / DIAGNOSTIC ONLY -- NOT THESIS BENCHMARK RESULTS."
echo "================================================================================"
