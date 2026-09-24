#!/bin/bash
# ImageNet-100 canonical job4971615-equivalent 8-method benchmark, ep9.
# Prepared for later submission; this file is intentionally not submitted here.
# The compute node reads only IMAGENET100_ROOT and never downloads data.
#SBATCH --job-name=imagenet100_8method_canonical_ep9
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=imagenet100_8method_canonical_ep9_%j.out
#SBATCH --error=imagenet100_8method_canonical_ep9_%j.err

set -euo pipefail

PROJECT_ROOT=/nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_imagenet100_5x20_job4971615_canonical_ep9.py
VERIFIER=tools/verify_imagenet100_dataset.py
export IMAGENET100_ROOT="${IMAGENET100_ROOT:-/nfsd/lttm4/datasets/ImageNet-1k_torch}"
export PYTHONUNBUFFERED=1

fatal() { echo "FATAL: $*" >&2; exit 1; }
[ -d "$PROJECT_ROOT" ] || fatal "project root missing: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
[ -x "$PYTHON" ] || fatal "project environment missing: $PYTHON"
[ -f "$SCRIPT" ] || fatal "production script missing: $SCRIPT"
[ -f "$VERIFIER" ] || fatal "dataset verifier missing: $VERIFIER"
[ -n "${IMAGENET100_ROOT:-}" ] || fatal "IMAGENET100_ROOT is required"
[ -d "$IMAGENET100_ROOT" ] || fatal "IMAGENET100_ROOT is not a directory: $IMAGENET100_ROOT"
[ -d "$IMAGENET100_ROOT/train" ] || fatal "shared ImageNet-1K train directory missing: $IMAGENET100_ROOT/train"
[ -d "$IMAGENET100_ROOT/val" ] || fatal "shared ImageNet-1K val directory missing: $IMAGENET100_ROOT/val"

echo "======================================================================"
echo "ImageNet-100 canonical job4971615-equivalent benchmark"
echo "hostname: $(hostname)"
echo "date: $(date -Is)"
echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "dataset root: $IMAGENET100_ROOT"
echo "seed: 42"
echo "tasks: 5x20"
echo "epochs per task: 9"
echo "methods: SimpleAvg | SimpleAvg + KD | SimpleAvg + DenseOrth | SimpleAvg + DenseOrth + KD | RankExt | RankExt KD+Protect | RankExt FactorOrth | RankExt KD+FactorOrth"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "======================================================================"

# Full offline verification happens before importing the training script.
"$PYTHON" "$VERIFIER" --data-root "$IMAGENET100_ROOT" --source-mode shared_imagenet1k --max-readable-checks 24

echo "dataset source: shared ImageNet-1K filtered to the project 100 WNIDs"
echo "dataset copy/download/extraction: NONE"
echo "cluster network dataset access: NONE"
echo "calibration/test policy: seed42, 25 validation images/class for calibration, remainder frozen test"

# The experiment itself enforces IMAGENET100_ROOT, the shared WNID-filtered
# source, the exact task partition, local-only model files, and the fixed
# 8-method set.
unset FAST_RUN_DEBUG || true
unset REPLICATION_SEED || true
"$PYTHON" -u "$SCRIPT"

echo "ImageNet-100 canonical benchmark finished: $(date -Is)"
