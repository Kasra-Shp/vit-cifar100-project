#!/bin/bash
# Fail-closed Method-8-only recovery.  Tasks 1-3 are loaded from the exact
# atomic boundary checkpoint; Task 4 restarts at epoch 0, then Task 5 runs.
# No methods 1-7 are enabled or trained by this launcher.
#SBATCH --job-name=imagenet100_method8_resume_ep9
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --output=imagenet100_method8_resume_%j.out
#SBATCH --error=imagenet100_method8_resume_%j.err

set -euo pipefail

PROJECT_ROOT=/nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_imagenet100_5x20_method8_resume.py
VERIFIER=tools/verify_imagenet100_dataset.py
export IMAGENET100_ROOT="${IMAGENET100_ROOT:-/nfsd/lttm4/datasets/ImageNet-1k_torch}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
[ -x "$PYTHON" ]
[ -f "$SCRIPT" ]
[ -f "$VERIFIER" ]
[ -d "$IMAGENET100_ROOT/train" ]
[ -d "$IMAGENET100_ROOT/val" ]

"$PYTHON" tools/validate_imagenet100_method8_resume.py
"$PYTHON" "$VERIFIER" --data-root "$IMAGENET100_ROOT" --source-mode shared_imagenet1k --max-readable-checks 24
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true
"$PYTHON" -u "$SCRIPT"
