#!/bin/bash
# Full CIFAR-100 5x20 normalized Combined RankExt experiment.
# Created only; do NOT submit automatically.
# The training script derives its model/data/training machinery read-only from
# canonical job4971615 and trains exactly one new method.
#SBATCH --job-name=cifar4971615_norm_combined
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=cifar4971615_norm_combined_%j.out
#SBATCH --error=cifar4971615_norm_combined_%j.err

set -euo pipefail

cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
TRAIN_SCRIPT=experiments_prepared/cifar_job4971615_normalized_combined_full.py
CAL_SCRIPT=experiments_prepared/cifar_job4971615_normalized_combined_offline_calibration.py
RUN_DIR=R8/performance_improvement_research/cifar_job4971615_normalized_combined_full_seed42_9ep

if [ ! -x "$PYTHON" ]; then
    echo "FATAL: known working Python environment is missing: $PYTHON" >&2
    exit 1
fi
if [ ! -f "$TRAIN_SCRIPT" ] || [ ! -f "$CAL_SCRIPT" ]; then
    echo "FATAL: required experiment script is missing." >&2
    exit 1
fi

CUDA_OK="$($PYTHON -c 'import torch; print(int(torch.cuda.is_available()))')"
if [ "$CUDA_OK" != "1" ]; then
    echo "FATAL: CUDA is unavailable; refusing CPU fallback." >&2
    exit 1
fi

unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true
export PYTHONUNBUFFERED=1

echo "CIFAR job4971615 normalized Combined full run"
echo "Job=${SLURM_JOB_ID:-local} Host=$(hostname) Start=$(date -Is)"
echo "Training method: rank_extension_normalized_factor_orth_lam50_fullkd_T2_protect30"
echo "Run directory: $RUN_DIR"

"$PYTHON" -u "$TRAIN_SCRIPT"

# This script is CPU-only and consumes only the saved validation/test logits.
"$PYTHON" -u "$CAL_SCRIPT" --run-dir "$RUN_DIR"

echo "Finished: $(date -Is)"
