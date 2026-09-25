#!/bin/bash
# Cheap ImageNet-100 GPU preflight: cached indices, Task 1, one real batch,
# one no-grad CUDA forward, then exit. No training updates are performed.
#SBATCH --job-name=imagenet100_gpu_preflight
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=imagenet100_gpu_preflight_%j.out
#SBATCH --error=imagenet100_gpu_preflight_%j.err

set -euo pipefail

PROJECT_ROOT=/nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
export IMAGENET100_ROOT="${IMAGENET100_ROOT:-/nfsd/lttm4/datasets/ImageNet-1k_torch}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
"$PYTHON" -u experiments_prepared/imagenet100_gpu_preflight.py
