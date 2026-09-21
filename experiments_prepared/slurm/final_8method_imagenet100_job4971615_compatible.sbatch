#!/bin/bash
# =============================================================================
# JOB4971615-COMPATIBLE 8-METHOD IMAGENET-100 EXPERIMENT LAUNCHER
# (2026-09-21). CREATED ONLY -- NOT SUBMITTED.
#
# Dedicated launcher for experiments_prepared/final_8method_imagenet100_
# job4971615_compatible.py. Mirrors the structure, safety checks, and comment
# style of experiments_prepared/slurm/final_9method_imagenet100_5x20_
# performance_recovery.sbatch (the 9-method ImageNet launcher, UNCHANGED and
# left byte-for-byte as-is by this task) -- the only differences are: (1) it
# invokes the new 8-method script instead of the 9-method one; (2) SBATCH
# job-name/output/error filenames are specific to this run, so it can never
# overwrite the 9-method ImageNet job's, or any CIFAR-100 job's, own SLURM
# log files or results/checkpoints directories; (3) the startup banner and
# fatal-check messages reference this run's own script/report paths.
#
# The target script's own 8-method logic (4 SimpleAvg + 4 RankExt, both
# families at 9 epochs/CL step, seed 42, no replay, CLIP ViT-B/16,
# q_proj/v_proj LoRA targets) is the SAME 8-method set as the canonical
# CIFAR-100 production benchmark (job4971615) -- see the target script's own
# header and R7/imagenet100_job4971615_compatibility_readiness.md for the
# full structural compatibility audit.
#
# Persistence/resume is built into the target script itself (CHECKPOINTS_DIR,
# isolated under this script's own RUN_NAME_BASE
# "imagenet100_5x20_final_8method_job4971615_compatible_seed42" -- never
# overlaps with the 9-method ImageNet run's, the CIFAR-100 run's, or any
# other prior run's checkpoints/results) -- this launcher does not touch
# results/ or checkpoints in any way beyond running the script.
#
# No conda/module activation line is added here: the python interpreter is
# invoked via its absolute path (PYTHON below), so no shell env activation
# step is needed to put the right interpreter on PATH.
#
# WALLTIME: 30 hours, unchanged from the 9-method ImageNet launcher -- this
# script trains 8 arms instead of 9 (one fewer SimpleAvg KD arm), so real
# walltime should be strictly lower, but the allowance is kept conservative
# and unmeasured pending a real per-epoch throughput check (see the 9-method
# launcher's own WALLTIME comment for the full reasoning this is inherited
# from).
# =============================================================================
#SBATCH --job-name=final_8method_imagenet100_job4971615_compatible
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=final_8method_imagenet100_job4971615_compatible_%j.out
#SBATCH --error=final_8method_imagenet100_job4971615_compatible_%j.err

set -euo pipefail

# Explicit, absolute repo path -- do NOT derive it from `dirname "$0"`. SLURM
# may execute a COPY of this script from a spool directory, in which case
# `dirname "$0"` would resolve there instead of the repo.
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_imagenet100_job4971615_compatible.py

# --- Minimal safe startup validation ---
if [ ! -x "$PYTHON" ]; then
    echo "FATAL: $PYTHON not found or not executable." >&2
    exit 1
fi
if [ ! -f "$SCRIPT" ]; then
    echo "FATAL: $SCRIPT not found under $(pwd)." >&2
    exit 1
fi
if [ ! -f "experiments_prepared/splits/imagenet100_class_order_seed42.json" ]; then
    echo "FATAL: persistent class-order artifact missing -- see R7/imagenet100_job4971615_compatibility_readiness.md." >&2
    exit 1
fi

# Guard against an inherited debug/seed override from the submitting shell --
# the real run must see FAST_RUN==False and seed==42 (the script's own
# defaults), never a leftover smoke-test or replication-seed override. This
# script is never intended to run in FAST_RUN_DEBUG mode at all (no smoke
# test, matching both the CIFAR-100 and 9-method ImageNet launchers' own
# policy) -- unset defensively regardless.
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true

# IMAGENET-100-SPECIFIC: configurable HF `datasets` cache root. Defaulted
# here (matching the script's own default) so it is visible in the SLURM
# log even when not overridden; export IMAGENET100_CACHE_ROOT before
# submitting this job to point elsewhere (e.g. a scratch filesystem with
# more free space). Shared with the 9-method ImageNet run's cache root by
# design -- both scripts load the identical IMAGENET100_DATASET_ID, so
# reusing an already-populated HF `datasets` cache is correct, not a
# collision (HF `datasets` cache directories are dataset-identity-keyed, not
# run-keyed; nothing about this run's own RESULTS/CHECKPOINTS output is
# shared).
export IMAGENET100_CACHE_ROOT="${IMAGENET100_CACHE_ROOT:-/nfsd/lttm4/tesisti/shahrampour/datasets/huggingface}"
mkdir -p "$IMAGENET100_CACHE_ROOT"

echo "================================================================================"
echo "JOB4971615-COMPATIBLE 8-METHOD IMAGENET-100 EXPERIMENT (REAL RUN)"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Repo:   $(pwd)   Python: $PYTHON"
echo "HF dataset cache root: $IMAGENET100_CACHE_ROOT"
echo "NOTE: this job requires outbound network access to huggingface.co (and its"
echo "CDN) from THIS compute node to download/cache clane9/imagenet-100 on first"
echo "run. This has NOT been verified from an actual cluster compute node -- only"
echo "from a local desktop machine (see R7/imagenet100_job4971615_compatibility_"
echo "readiness.md and R7/imagenet100_5x20_static_readiness.md)."
echo "If this node has no internet access, the job will fail at the dataset-load"
echo "step; pre-stage the HF cache from a node that does, or contact the cluster"
echo "administrator about compute-node network policy, before relying on this job."
echo "================================================================================"

export PYTHONUNBUFFERED=1

"$PYTHON" -u "$SCRIPT"

echo "================================================================================"
echo "JOB4971615-COMPATIBLE 8-METHOD IMAGENET-100 EXPERIMENT finished. End: $(date -Is)"
echo "================================================================================"
