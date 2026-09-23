#!/bin/bash
# Final improved 8-method CUB-200-2011 benchmark launcher.
# Prepared only; do not submit automatically.
# The production script preserves the refined SimpleAvg/plain/KD+Protect
# training paths and changes only the two RankExt FactorOrth arms to the
# validated task/rank-normalized lambda=50 implementation.
#SBATCH --job-name=final_8method_cub200_5x40_improved_rankext_ep9
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=final_8method_cub200_5x40_improved_rankext_ep9_%j.out
#SBATCH --error=final_8method_cub200_5x40_improved_rankext_ep9_%j.err

set -euo pipefail

cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project

PYTHON=/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python
SCRIPT=experiments_prepared/final_8method_cub200_5x40_improved_rankext_ep9.py
ARTIFACT=experiments_prepared/splits/cub200_class_order_seed42_5x40.json
RUN_NAME_BASE=cub200_5x40_final_8method_improved_rankext_seed42_ep9

SHARED_ROOT=/nfsd/lttm4/tesisti/shahrampour
export CUB200_CACHE_ROOT="${CUB200_CACHE_ROOT:-$SHARED_ROOT/datasets/huggingface}"
export HF_HOME="${HF_HOME:-$SHARED_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SHARED_ROOT/.cache/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SHARED_ROOT/.cache/huggingface/datasets}"

fatal() { echo "FATAL: $*" >&2; exit 1; }

[ -x "$PYTHON" ] || fatal "interpreter missing or not executable: $PYTHON"
[ -f "$SCRIPT" ] || fatal "production script missing: $SCRIPT"
[ -f "$ARTIFACT" ] || fatal "persistent split artifact missing: $ARTIFACT"

"$PYTHON" -c 'import datasets, transformers, torch; print("datasets:", datasets.__version__); print("transformers:", transformers.__version__); print("torch:", torch.__version__)'
"$PYTHON" -c 'import json, sys; p=sys.argv[1]; d=json.load(open(p, encoding="utf-8")); assert d["dataset_identifier"] == "Donghyun99/CUB-200-2011"; assert d["seed"] == 42; assert d["num_classes"] == 200; assert d["num_tasks"] == 5; assert d["classes_per_task"] == 40; assert len(d["tasks"]) == 5 and all(len(t["new_class_ids"]) == 40 for t in d["tasks"]); print("class-order artifact: PASS")' "$ARTIFACT"

unset FAST_RUN_DEBUG || true
unset CUB200_PREFLIGHT_ONLY || true
unset REPLICATION_SEED || true
export PYTHONUNBUFFERED=1

echo "CUB-200-2011 final improved 8-method run"
echo "Job ID: ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   Start: $(date -Is)"
echo "Python: $PYTHON"
echo "Script: $SCRIPT"
echo "Run base: $RUN_NAME_BASE"
echo "Protocol: 5 tasks x 40 classes, seed 42, epochs 9"
echo "SimpleAvg: unchanged; RankExt FactorOrth arms: normalized lambda=50"
echo "Rank schedule: [16,32,48,64,80]; RankExt head LR x2; KD T=2 weight=1; Protect=30"
echo "CUB cache: $CUB200_CACHE_ROOT"

"$PYTHON" -u "$SCRIPT"

echo "Final improved CUB run finished. End: $(date -Is)"
